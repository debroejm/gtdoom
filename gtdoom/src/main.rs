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

mod resource;
mod client;
mod render_test;

fn main() {
    let app = GtdoomClient::new();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
        .with(ConsoleLogLayer::new(app.console().log()))
        .init();

    let (doom2, chex3, zandronum) = future::block_on(async move {
        let doom2 = LumpFileSource::open("DOOM2.WAD");
        let chex3 = LumpFileSource::open("chex3.wad");
        let zandronum = LumpFileSource::open("zandronum.pk3");

        (
            doom2.await.unwrap(),
            chex3.await.unwrap(),
            zandronum.await.unwrap(),
        )
    });
    let resources = ResourceManager::builder()
        .source(ConstantResourceSource::builder()
            .resource("console_font", include_bytes!("../assets/RobotoMono/RobotoMono-VariableFont_wght.ttf"))
            .build())
        .source(doom2)
        .source(chex3)
        .source(zandronum)
        .build();

    EngineBuilder::new()
        .app(app)
        .app_driver(WinitDriver::builder()
            .application_name("gtDoom")
            .build())
        .networking_driver(NoNetDriver::new())
        .resources(resources)
        .build()
        .start();

/*    let wad = future::block_on(WadData::open_wad("chex3.wad")).unwrap();
    println!("WAD type = {}", wad.w_type());*/
}
