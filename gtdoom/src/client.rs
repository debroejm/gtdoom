use async_trait::async_trait;
use gtether::app::Application;
use gtether::console::gui::{BackgroundImage, ConsoleGui, ConsoleGuiBackground};
use gtether::console::log::ConsoleLog;
use gtether::console::Console;
use gtether::gui::window::winit::{CreateWindowInfo, WindowAttributes, WinitDriver};
use gtether::net::driver::NoNetDriver;
use gtether::render::font::glyph::GlyphFontLoader;
use gtether::render::render_pass::EngineRenderPassBuilder;
use gtether::resource::manager::LoadPriority;
use gtether::Engine;
use std::sync::Arc;
use tracing::info;

use crate::render_test::FlatRenderer;
use crate::resource::map::Map;
use crate::resource::patch::PatchViewLoader;

pub struct GtdoomClient {
    console: Arc<Console>,
}

impl GtdoomClient {
    pub fn new() -> Self {
        let console = Arc::new(Console::builder()
            .log(ConsoleLog::new(1000))
            .build());

        Self {
            console,
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

        let console_gui = ConsoleGui::builder(self.console.clone())
            .window(&window)
            .font(console_font)
            .background(ConsoleGuiBackground::Image(BackgroundImage::from_image(titlepic.clone())))
            .build().unwrap();

        let render_pass = EngineRenderPassBuilder::new(window.renderer().clone())
            .final_color_attachment("color", [0.0, 0.0, 0.0, 1.0]).unwrap()
            .begin_subpass()
                .color_attachment("color")
                .handler(move |renderer, subpass| FlatRenderer::new(
                    renderer,
                    subpass,
                    titlepic,
                ))
            .handler(console_gui.bootstrap_renderer())
            .end_subpass()
            .build().unwrap();
        window.renderer().set_render_pass(render_pass);

        let map_name = "E1M1";
        let map = engine.resources().get_or_load(
            map_name,
            Map::loader(),
            LoadPriority::Immediate,
        ).await.unwrap();
        info!(?map_name, ?map, "Loaded map");
    }
}