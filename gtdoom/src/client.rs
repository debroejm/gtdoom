use async_trait::async_trait;
use educe::Educe;
use gtether::app::Application;
use gtether::console::command::{Command, CommandError, CommandRegistry, CommandTree, ParamCountCheck};
use gtether::console::gui::{BackgroundImage, ConsoleGui, ConsoleGuiBackground};
use gtether::console::log::ConsoleLog;
use gtether::console::Console;
use gtether::gui::input::{InputDelegateEvent, InputStateLayer, KeyCode, MouseButton, MouseScrollDelta};
use gtether::gui::window::winit::{CreateWindowInfo, WindowAttributes, WindowHandle, WinitDriver};
use gtether::net::driver::NoNetDriver;
use gtether::render::font::glyph::GlyphFontLoader;
use gtether::render::render_pass::EngineRenderPassBuilder;
use gtether::render::uniform::Uniform;
use gtether::resource::manager::ResourceManager;
use gtether::resource::source::fs::ResourceSourceFilesHandle;
use gtether::util::tick_loop::{TickLoopBuilder, TickLoopHandle};
use gtether::worker::WorkerPool;
use gtether::Engine;
use parking_lot::Mutex;
use parry3d::na::{Isometry3, Point3, Translation3};
use smol::future;
use std::sync::Arc;
use tracing::{debug, info};
use vulkano::device::Features;
use vulkano::format::Format;
use vulkano::image::SampleCount;
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp};

use crate::geometry::line::Line;
use crate::render::camera::{Camera, VP};
use crate::resource::lump::{AddSourceCommand, ListSourcesCommand, LumpFile, MoveSourceCommand, RemoveSourceCommand};
use crate::resource::map::blockmap::{Blockmap, BlockmapDimensions};
use crate::resource::map::model::RENDER_CHUNK_SIZE;
use crate::resource::map::model::{Chunk, MapModelData, MapRenderer};
use crate::resource::map::vertex::{FixedFloat, FixedVertex};
use crate::resource::map::Map;
use crate::resource::patch::PatchViewLoader;
use crate::WorkerPriority;

#[cfg(debug_assertions)]
fn contour_angle(contour: &[glm::TVec2<FixedFloat>]) -> FixedFloat {
    const TWO: FixedFloat = FixedFloat::lit("2.0");

    if contour.len() < 2 {
        return FixedFloat::ZERO
    }

    let mut iter = contour.iter();
    let first = iter.next().unwrap();
    let second = iter.next().unwrap();

    let mut total_angle = FixedFloat::ZERO;

    let mut prev = second;
    let mut prev_angle = cordic::atan2(second.y - first.y, second.x - first.x);
    for next in iter.chain([first, second]) {
        let next_angle = cordic::atan2(next.y - prev.y, next.x - prev.x);
        let mut diff = prev_angle - next_angle;
        if diff < -FixedFloat::PI {
            diff += TWO * FixedFloat::PI;
        } else if diff > FixedFloat::PI {
            diff -= TWO * FixedFloat::PI;
        }
        total_angle += diff;
        prev = next;
        prev_angle = next_angle;
    }

    total_angle
}

#[derive(Educe)]
#[educe(Debug)]
struct ChangeMapCommand {
    window: WindowHandle,
    #[educe(Debug(ignore))]
    resources: Arc<ResourceManager>,
    #[educe(Debug(ignore))]
    console_gui: Arc<ConsoleGui>,
    #[educe(Debug(ignore))]
    camera: Arc<Uniform<VP, Camera>>,
}

impl Command for ChangeMapCommand {
    fn handle(&self, parameters: &[String]) -> Result<(), CommandError> {
        ParamCountCheck::Equal(1).check(parameters.len() as u32)?;

        let map_name = parameters[0].as_str();
        let map = future::block_on(self.resources.get::<Map>(map_name))
            .map_err(|e| CommandError::CommandFailure(Box::new(e)))?;
        info!(?map_name, ?map, "Loaded map");

        let (chunks, map_model_data) = future::block_on(async {
            let map = map.read();

            let map_bounds = {
                let gl_nodes = map.gl_nodes().read();
                let gl_vertices = gl_nodes.gl_vertices.read();
                gl_vertices.calculate_bounds().map(|v| v.to_num::<i16>())
            };
            let block_dims = BlockmapDimensions::<RENDER_CHUNK_SIZE>::new(
                map_bounds.min.cast(),
                // Add +1 to size to force inclusive coverage of edges
                (map_bounds.max - map_bounds.min).map(|v| v as u32 + 1),
            );
            let builders = Blockmap::from_fn(
                block_dims,
                |_| Mutex::new(Chunk::builder(
                    self.resources.clone(),
                    self.window.renderer().clone(),
                ))
            );

            for linedef in map.linedefs().read().iter() {
                let line = Line::<FixedFloat>::new(
                    FixedVertex::from(linedef.start_vertex),
                    FixedVertex::from(linedef.end_vertex),
                );
                for (builder, metadata) in builders.query(line) {
                    builder.lock().add_linedef(&linedef, metadata.segment).await
                        .map_err(|e| CommandError::CommandFailure(Box::new(e)))?;
                }
            }

            {
                let gl_nodes = map.gl_nodes().read();
                let gl_vertices = gl_nodes.gl_vertices.read();
                for ssector in &gl_nodes.gl_ssectors {
                    let contour = ssector.contour(&gl_vertices)
                        .map(|v| *v.to_fixed().into_inner())
                        .collect::<Vec<_>>();
                    debug_assert!(
                        contour_angle(&contour) > FixedFloat::ZERO,
                        "CCW contour found: {contour:?}",
                    );
                    for (builder, metadata) in builders.query(contour) {
                        let contour = metadata.contour.into_iter()
                            .map(FixedVertex::from)
                            .collect::<Vec<_>>();
                        builder.lock().add_ssector(&ssector.sector, contour);
                    }
                }
            }

            let mut map_model_data = MapModelData::default();
            let mut chunks = Vec::with_capacity(builders.dimensions().total_block_count());
            for (_, builder) in builders.into_iter() {
                let chunk = builder.into_inner().build(&mut map_model_data).await
                    .map_err(|e| CommandError::CommandFailure(Box::new(e)))?;
                chunks.push(chunk);
            }
            let chunks = Blockmap::from_parts(block_dims, chunks);

            Ok::<_, CommandError>((Arc::new(chunks), map_model_data))
        })?;
        debug!(?map_name, "Parsed map model data");

        let renderer_camera = self.camera.clone();
        {
            let mut lock = renderer_camera.write();
            lock.view_mut().set_eye(glm::vec3(-500.0, -500.0, 1000.0));
            lock.view_mut().set_target(glm::vec3(0.0, 0.0, 0.0));
        }

        let render_pass = EngineRenderPassBuilder::new(self.window.renderer().clone())
            .attachment("depth", AttachmentDescription {
                format: Format::D16_UNORM,
                samples: SampleCount::Sample1,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::DontCare,
                ..Default::default()
            }, Some(1.0))
            .final_color_attachment("color", [0.0, 0.0, 0.0, 1.0]).unwrap()
            .begin_subpass()
            .depth_stencil_attachment("depth")
            .color_attachment("color")
            .handler(move |renderer, subpass| MapRenderer::new(
                renderer_camera,
                chunks,
                map_model_data,
                renderer,
                subpass,
            ).unwrap())
            .end_subpass()
            .begin_subpass()
            .color_attachment("color")
            .handler(self.console_gui.bootstrap_renderer())
            .end_subpass()
            .build().unwrap();
        self.window.renderer().set_render_pass(render_pass);

        Ok(())
    }
}

pub struct GtdoomClient {
    lump_files: ResourceSourceFilesHandle<LumpFile>,
    console: Arc<Console>,
    input_join_handle: Mutex<Option<TickLoopHandle>>,
}

impl GtdoomClient {
    pub fn new(
        workers: &WorkerPool<WorkerPriority>,
        lump_files: ResourceSourceFilesHandle<LumpFile>,
    ) -> Self {
        let console = Arc::new(Console::builder()
            .log(ConsoleLog::new(1000))
            .worker_config(WorkerPriority::Secondary, workers)
            .build());

        Self {
            lump_files,
            console,
            input_join_handle: Mutex::new(None),
        }
    }

    #[inline]
    pub fn console(&self) -> &Arc<Console> {
        &self.console
    }
}

#[async_trait(?Send)]
impl Application for GtdoomClient {
    type ApplicationDriver = WinitDriver;
    type NetworkingDriver = NoNetDriver;

    async fn init(&self, engine: &Arc<Engine<Self>>) {
        let window = engine.window_manager().create_window(CreateWindowInfo {
            attributes: WindowAttributes::default()
                .with_title("GTDoom"),
            device_features: Features {
                // These are used for texture array sizes determined at runtime
                runtime_descriptor_array: true,
                shader_sampled_image_array_non_uniform_indexing: true,
                ..Default::default()
            },
            ..Default::default()
        }).await;

        let (titlepic, console_font) = async {
            let titlepic = engine.resources().get_with_loader(
                "TITLEPIC",
                PatchViewLoader::new(window.renderer().clone()),
            );
            let console_font = engine.resources().get_with_loader(
                "console_font",
                GlyphFontLoader::new(window.renderer().clone()),
            );

            (
                titlepic.await.unwrap(),
                console_font.await.unwrap(),
            )
        }.await;

        let camera = Camera::new_uniform(
            window.renderer(),
            glm::vec3(-500.0, -500.0, 1000.0),
            glm::vec3(0.0, 0.0, 0.0),
        ).unwrap();
        let input_camera = camera.clone();

        let console_gui = ConsoleGui::builder(self.console.clone())
            .window(&window)
            .font(console_font)
            .background(ConsoleGuiBackground::Image(BackgroundImage::from_image(titlepic.clone())))
            .build().unwrap();

        let render_pass = EngineRenderPassBuilder::new(window.renderer().clone())
            .final_color_attachment("color", [0.0, 0.0, 0.0, 1.0]).unwrap()
            .begin_subpass()
                .color_attachment("color")
                .handler(console_gui.bootstrap_renderer())
            .end_subpass()
            .build().unwrap();
        window.renderer().set_render_pass(render_pass);

        let input = window.input_state().create_delegate();

        let mut input_join_handle = self.input_join_handle.lock();
        input_join_handle.take();
        *input_join_handle = Some(TickLoopBuilder::new()
            .name("camera-control")
            .tick_rate(60)
            .init(|| Ok::<(), ()>(()))
            .tick(move |_, delta| {
                let speed = 512.0;
                let rotate_speed = 0.005;
                let distance = delta.as_secs_f32() * speed;

                let mut view = input_camera.read().view().clone();

                let mut orient = view.orient();
                let mut right = view.right();

                // Only move on the X/Y plane
                orient.z = 0.0;
                right.z = 0.0;
                let orient = orient.normalize();
                let right = right.normalize();

                for event in input.events() {
                    match event {
                        InputDelegateEvent::MouseWheel(event) => {
                            let diff = match event.delta {
                                MouseScrollDelta::LineDelta(_, vert) => -100.0 * vert,
                                MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
                            };
                            let mut eye = *view.eye();
                            eye.z = (eye.z + diff).max(64.0);
                            view.set_eye(eye);
                        },
                        InputDelegateEvent::MouseMotion(motion) => {
                            if input.is_mouse_pressed(MouseButton::Middle).unwrap_or(false)
                                    && motion.x.abs() > 0.01 {
                                let offset: Point3<f32> = (view.eye() - view.target()).into();
                                let rotate_angle = (motion.x as f32) * rotate_speed;
                                let offset = Isometry3::rotation(glm::TVec3::z() * rotate_angle) * offset;
                                view.set_eye(*view.target() + glm::vec3(offset.x, offset.y, offset.z));
                            }
                        },
                        _ => {},
                    }
                }

                let mut translation = Translation3::default();
                if input.is_key_pressed(KeyCode::KeyW, None).unwrap_or(false) {
                    translation *= Translation3::from(orient * distance);
                }
                if input.is_key_pressed(KeyCode::KeyS, None).unwrap_or(false) {
                    translation *= Translation3::from(orient * -distance);
                }
                if input.is_key_pressed(KeyCode::KeyD, None).unwrap_or(false) {
                    translation *= Translation3::from(right * distance);
                }
                if input.is_key_pressed(KeyCode::KeyA, None).unwrap_or(false) {
                    translation *= Translation3::from(right * -distance);
                }
                view.translate(translation);

                {
                    let mut camera = input_camera.write();
                    *camera.view_mut() = view;
                    camera.tick(delta);
                }

                true
            })
            .spawn().unwrap()
        );

        let mut cmd_registry = self.console.registry();
        cmd_registry.register_command("changemap", Box::new(ChangeMapCommand {
            window,
            resources: engine.resources().clone(),
            console_gui,
            camera,
        })).unwrap();

        let mut sources_subcommands = CommandTree::default();
        sources_subcommands.register_command(
            "list",
            Box::new(ListSourcesCommand::new(self.lump_files.clone())),
        ).unwrap();
        sources_subcommands.register_command(
            "move",
            Box::new(MoveSourceCommand::new(self.lump_files.clone())),
        ).unwrap();
        sources_subcommands.register_command(
            "add",
            Box::new(AddSourceCommand::new(self.lump_files.clone())),
        ).unwrap();
        sources_subcommands.register_command(
            "remove",
            Box::new(RemoveSourceCommand::new(self.lump_files.clone())),
        ).unwrap();
        sources_subcommands.register_alias("delete", "remove").unwrap();
        cmd_registry
            .register_command("sources", Box::new(sources_subcommands)).unwrap()
            .register_alias("wads", "sources").unwrap();
    }

    async fn terminate(&self, _engine: &Arc<Engine<Self>>) {
        self.input_join_handle.lock().take();
    }
}