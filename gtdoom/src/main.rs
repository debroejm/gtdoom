extern crate nalgebra_glm as glm;

use gtether::console::log::ConsoleLogLayer;
use gtether::gui::window::winit::WinitDriver;
use gtether::net::driver::NoNetDriver;
use gtether::resource::manager::ResourceManager;
use gtether::resource::source::constant::ConstantResourceSource;
use gtether::resource::source::fs::ResourceSourceFiles;
use gtether::EngineBuilder;
use smol::future;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use crate::client::GtdoomClient;
use crate::resource::lump::LumpFile;

mod client;
mod render;
mod resource;

fn main() {
    let (resources, lump_files) = {
        let lump_files_source = ResourceSourceFiles::<LumpFile>::default();
        let handle = lump_files_source.handle();
        let manager = ResourceManager::builder()
            .source(ConstantResourceSource::builder()
                .resource("console_font", include_bytes!("../assets/RobotoMono/RobotoMono-VariableFont_wght.ttf"))
                .build())
            .source(lump_files_source)
            .build();
        (manager, handle)
    };
    
    let app = GtdoomClient::new(lump_files.clone());

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
        .with(ConsoleLogLayer::new(app.console().log()))
        .init();

    let sources = std::env::args()
        .skip(1)
        .map(|arg| future::block_on(LumpFile::open(arg)).unwrap())
        .collect::<Vec<_>>();
    {
        let mut lump_files = lump_files.write();
        for source in sources {
            lump_files.push(source);
        }
    }

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
