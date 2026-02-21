extern crate nalgebra_glm as glm;

use std::num::NonZeroUsize;
use gtether::console::log::ConsoleLogLayer;
use gtether::gui::window::winit::WinitDriver;
use gtether::net::driver::NoNetDriver;
use gtether::resource::manager::{ResourceManager, ResourceManagerWorkerPriorityConfig};
use gtether::resource::source::constant::ConstantResourceSource;
use gtether::resource::source::fs::ResourceSourceFiles;
use gtether::EngineBuilder;
use gtether::worker::WorkerPool;
use smol::future;
use tracing::info;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use crate::client::GtdoomClient;
use crate::resource::lump::LumpFile;

mod client;
mod render;
mod resource;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkerPriority {
    Primary = 2,
    Secondary = 1,
    ResourceUpdates = 0,
}

fn main() {
    let worker_count = std::thread::available_parallelism()
        .map(|count| {
            NonZeroUsize::try_from((count.get() as f32 * 0.8).ceil() as usize)
                .unwrap_or(unsafe { NonZeroUsize::new_unchecked(1) })
                .max(unsafe { NonZeroUsize::new_unchecked(4) })
        })
        .unwrap_or(unsafe { NonZeroUsize::new_unchecked(4) });
    let workers = WorkerPool::<WorkerPriority>::builder()
        .worker_count(worker_count)
        .start();

    let (resources, lump_files) = {
        let lump_files_source = ResourceSourceFiles::<LumpFile>::default();
        let handle = lump_files_source.handle();
        let manager = ResourceManager::builder()
            .source(ConstantResourceSource::builder()
                .resource("console_font", include_bytes!("../assets/RobotoMono/RobotoMono-VariableFont_wght.ttf"))
                .build())
            .source(lump_files_source)
            .worker_config(ResourceManagerWorkerPriorityConfig {
                immediate: WorkerPriority::Primary,
                delayed: WorkerPriority::Secondary,
                update: WorkerPriority::ResourceUpdates,
            }, &workers)
            .build();
        (manager, handle)
    };
    
    let app = GtdoomClient::new(&workers, lump_files.clone());

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .finish()
        .with(ConsoleLogLayer::new(app.console().log()))
        .init();

    info!(?worker_count, "Workers configured");

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
