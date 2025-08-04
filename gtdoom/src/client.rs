use async_trait::async_trait;
use gtether::app::Application;
use gtether::console::gui::{BackgroundImage, ConsoleGui, ConsoleGuiBackground};
use gtether::console::log::ConsoleLog;
use gtether::console::Console;
use gtether::gui::input::{InputDelegateEvent, InputStateLayer, KeyCode, MouseButton, MouseScrollDelta};
use gtether::gui::window::winit::{CreateWindowInfo, WindowAttributes, WinitDriver};
use gtether::net::driver::NoNetDriver;
use gtether::render::font::glyph::GlyphFontLoader;
use gtether::render::render_pass::EngineRenderPassBuilder;
use gtether::resource::manager::LoadPriority;
use gtether::util::tick_loop::{TickLoopBuilder, TickLoopHandle};
use gtether::Engine;
use parking_lot::Mutex;
use std::sync::Arc;
use parry3d::na::{Isometry3, Point3, Translation3};
use tracing::info;
use vulkano::device::Features;
use vulkano::format::Format;
use vulkano::image::SampleCount;
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp};

use crate::render::camera::Camera;
use crate::resource::map::model::{Chunk, MapModelData, MapRenderer};
use crate::resource::map::Map;
use crate::resource::patch::PatchViewLoader;

pub struct GtdoomClient {
    console: Arc<Console>,
    input_join_handle: Mutex<Option<TickLoopHandle>>,
}

impl GtdoomClient {
    pub fn new() -> Self {
        let console = Arc::new(Console::builder()
            .log(ConsoleLog::new(1000))
            .build());

        Self {
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
            let titlepic = engine.resources().get_or_load(
                "TITLEPIC",
                PatchViewLoader::new(window.renderer().clone()),
                LoadPriority::Immediate,
            );
            let console_font = engine.resources().get_or_load(
                "console_font",
                GlyphFontLoader::new(window.renderer().clone()),
                LoadPriority::Immediate,
            );

            (
                titlepic.await.unwrap(),
                console_font.await.unwrap(),
            )
        }.await;

        let map_name = "E1M1";
        let map = engine.resources().get_or_load(
            map_name,
            Map::loader(window.renderer().clone()),
            LoadPriority::Immediate,
        ).await.unwrap();
        info!(?map_name, ?map, "Loaded map");

        let camera = Camera::new_uniform(
            window.renderer(),
            glm::vec3(-500.0, -500.0, 1000.0),
            glm::vec3(0.0, 0.0, 0.0),
        ).unwrap();
        let input_camera = camera.clone();

        let (chunk, map_model_data) = {
            let mut builder = Chunk::builder();
            for linedef in map.read().linedefs().read().iter() {
                builder.add_linedef(linedef);
            }
            let mut map_model_data = MapModelData::default();
            let chunk = Arc::new(builder.build(&mut map_model_data).unwrap());
            (chunk, map_model_data)
        };

        let console_gui = ConsoleGui::builder(self.console.clone())
            .window(&window)
            .font(console_font)
            .background(ConsoleGuiBackground::Image(BackgroundImage::from_image(titlepic.clone())))
            .build().unwrap();

        let render_pass = EngineRenderPassBuilder::new(window.renderer().clone())
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
                    camera,
                    chunk,
                    map_model_data,
                    renderer,
                    subpass,
                ).unwrap())
            .end_subpass()
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
    }

    async fn terminate(&self, _engine: &Arc<Engine<Self>>) {
        self.input_join_handle.lock().take();
    }
}