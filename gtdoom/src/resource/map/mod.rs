pub mod linedef;
pub mod sector;
pub mod sidedef;
pub mod vertex;

use async_trait::async_trait;
use educe::Educe;
use gtether::resource::{Resource, ResourceLoadContext, ResourceLoadError, ResourceLoader, ResourceReadData};
use std::fmt;
use std::fmt::Formatter;
use std::ops::Deref;
use std::sync::Arc;

use crate::resource::lump::MarkerLump;
use crate::resource::map::linedef::Linedefs;
use crate::resource::map::sector::Sectors;
use crate::resource::map::sidedef::Sidedefs;
use crate::resource::map::vertex::Vertices;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapType {
    Doom,
    Heretic,
    Hexen,
    Strife,
    ZDoom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapFormat {
    Doom,
    Hexen,
    UDMF,
}

fn fmt_resource_vec<T, D: Deref<Target=Vec<T>> + ?Sized + Send + Sync + 'static>(
    resource: &Arc<Resource<D>>,
    f: &mut Formatter<'_>,
) -> fmt::Result {
    write!(f, "[..; {}]", resource.read().deref().len())
}

#[derive(Educe)]
#[educe(Debug)]
pub struct Map {
    m_type: MapType,
    m_format: MapFormat,
    #[educe(Debug(method(fmt_resource_vec)))]
    sectors: Arc<Resource<Sectors>>,
    #[educe(Debug(method(fmt_resource_vec)))]
    vertices: Arc<Resource<Vertices>>,
    #[educe(Debug(method(fmt_resource_vec)))]
    sidedefs: Arc<Resource<Sidedefs>>,
    #[educe(Debug(method(fmt_resource_vec)))]
    linedefs: Arc<Resource<Linedefs>>,
}

impl Map {
    #[inline]
    pub fn loader() -> MapLoader {
        MapLoader {}
    }
}

pub struct MapLoader {}

#[async_trait]
impl ResourceLoader<Map> for MapLoader {
    async fn load(
        &self,
        _data: ResourceReadData,
        ctx: ResourceLoadContext,
    ) -> Result<Box<Map>, ResourceLoadError> {
        let endmap_fut = ctx.manager().get_or_load(
            ctx.id().join("ENDMAP"),
            MarkerLump::loader(),
            ctx.priority(),
        );
        let behavior_fut = ctx.manager().get_or_load(
            ctx.id().join("BEHAVIOR"),
            MarkerLump::loader(),
            ctx.priority(),
        );
        let (m_type, m_format) = match endmap_fut.await {
            // TODO: parse map_type from UDMF namespace
            Ok(_) => (MapType::ZDoom, MapFormat::UDMF),
            Err(_) => match behavior_fut.await {
                Ok(_) => (MapType::Hexen, MapFormat::Hexen),
                // TODO: How to determine Heretic/Strife map types?
                Err(_) => (MapType::Doom, MapFormat::Doom),
            }
        };

        match m_format {
            MapFormat::Doom | MapFormat::Hexen => {
                let sectors_fut = ctx.manager().get_or_load(
                    ctx.id().join("SECTORS"),
                    Sectors::lump_loader(m_type),
                    ctx.priority(),
                );
                let vertices_fut = ctx.manager().get_or_load(
                    ctx.id().join("VERTEXES"),
                    Vertices::lump_loader(),
                    ctx.priority(),
                );
                let sidedefs_fut = ctx.manager().get_or_load(
                    ctx.id().join("SIDEDEFS"),
                    Sidedefs::lump_loader(m_type),
                    ctx.priority(),
                );
                let linedefs_fut = ctx.manager().get_or_load(
                    ctx.id().join("LINEDEFS"),
                    Linedefs::lump_loader(m_type, m_format),
                    ctx.priority(),
                );

                Ok(Box::new(Map {
                    m_type,
                    m_format,
                    sectors: sectors_fut.await?,
                    vertices: vertices_fut.await?,
                    sidedefs: sidedefs_fut.await?,
                    linedefs: linedefs_fut.await?,
                }))
            },
            MapFormat::UDMF => Err(ResourceLoadError::from("UDMF map format not yet supported")),
        }
    }
}