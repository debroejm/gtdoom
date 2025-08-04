extern crate nalgebra_glm as glm;

use gtether::console::log::ConsoleLogLayer;
use gtether::gui::window::winit::WinitDriver;
use gtether::net::driver::NoNetDriver;
use gtether::resource::manager::ResourceManager;
use gtether::resource::source::constant::ConstantResourceSource;
use gtether::EngineBuilder;
use smol::future;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use crate::client::GtdoomClient;
use crate::resource::lump::LumpFileSource;

mod client;
mod render;
mod resource;

fn main() {
    let app = GtdoomClient::new();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
        .with(ConsoleLogLayer::new(app.console().log()))
        .init();

    let resources = {
        let mut builder = ResourceManager::builder()
            .source(ConstantResourceSource::builder()
                .resource("console_font", include_bytes!("../assets/RobotoMono/RobotoMono-VariableFont_wght.ttf"))
                .build());
        let sources = std::env::args()
            .skip(1)
            .map(|arg| future::block_on(LumpFileSource::open(arg)).unwrap())
            .collect::<Vec<_>>();
        for source in sources {
            builder = builder.source(source);
        }
        builder.build()
    };

    EngineBuilder::new()
        .app(app)
        .app_driver(WinitDriver::builder()
            .application_name("gtDoom")
            .build())
        .networking_driver(NoNetDriver::new())
        .resources(resources)
        .build()
        .start();
}
