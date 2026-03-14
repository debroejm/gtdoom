pub mod gl_node;
pub mod linedef;
pub mod model;
pub mod quadtree;
pub mod sector;
pub mod sidedef;
pub mod vertex;

use async_trait::async_trait;
use educe::Educe;
use gtether::render::Renderer;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{Resource, ResourceLoadError, ResourceLoader, ResourceReadData};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use crate::resource::indexed::IndexedVec;
use crate::resource::lump::MarkerLump;
use crate::resource::map::gl_node::GlNodes;
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

fn fmt_resource_buffer<T, D: Deref<Target=IndexedVec<T>> + ?Sized + Send + Sync + 'static>(
    resource: &Arc<Resource<D>>,
    f: &mut Formatter<'_>,
) -> fmt::Result {
    write!(f, "[.., {}]", resource.read().deref().len())
}

fn fmt_resource<T: ?Sized + Debug + Send + Sync + 'static>(
    resource: &Arc<Resource<T>>,
    f: &mut Formatter<'_>,
) -> fmt::Result {
    write!(f, "{:?}", &*resource.read())
}

#[derive(Educe)]
#[educe(Debug)]
pub struct Map {
    m_type: MapType,
    m_format: MapFormat,
    #[educe(Debug(method(fmt_resource_buffer)))]
    sectors: Arc<Resource<Sectors>>,
    #[educe(Debug(method(fmt_resource_vec)))]
    vertices: Arc<Resource<Vertices>>,
    #[educe(Debug(method(fmt_resource_buffer)))]
    sidedefs: Arc<Resource<Sidedefs>>,
    #[educe(Debug(method(fmt_resource_buffer)))]
    linedefs: Arc<Resource<Linedefs>>,
    #[educe(Debug(method(fmt_resource)))]
    gl_nodes: Arc<Resource<GlNodes>>,
}

impl Map {
    #[inline]
    pub fn loader(renderer: Arc<Renderer>) -> MapLoader {
        MapLoader {
            renderer,
        }
    }

    #[inline]
    pub fn linedefs(&self) -> &Arc<Resource<Linedefs>> {
        &self.linedefs
    }

    #[inline]
    pub fn gl_nodes(&self) -> &Arc<Resource<GlNodes>> {
        &self.gl_nodes
    }
}

pub struct MapLoader {
    renderer: Arc<Renderer>,
}

#[async_trait]
impl ResourceLoader for MapLoader {
    type Output = Map;

    async fn load(
        &self,
        _data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Map>, ResourceLoadError> {
        let endmap_fut = ctx.get::<MarkerLump>(
            PathBuf::from(ctx.id().as_str()).join("ENDMAP")
        );
        let behavior_fut = ctx.get::<MarkerLump>(
            PathBuf::from(ctx.id().as_str()).join("BEHAVIOR")
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
                let sectors_fut = ctx.get_with_loader(
                    PathBuf::from(ctx.id().as_str()).join("SECTORS"),
                    Sectors::lump_loader(self.renderer.clone(), m_type),
                );
                let vertices_fut = ctx.get_with_loader(
                    PathBuf::from(ctx.id().as_str()).join("VERTEXES"),
                    Vertices::lump_loader(),
                );
                let sidedefs_fut = ctx.get_with_loader(
                    PathBuf::from(ctx.id().as_str()).join("SIDEDEFS"),
                    Sidedefs::lump_loader(self.renderer.clone(), m_type),
                );
                let linedefs_fut = ctx.get_with_loader(
                    PathBuf::from(ctx.id().as_str()).join("LINEDEFS"),
                    Linedefs::lump_loader(self.renderer.clone(), m_type, m_format),
                );
                let gl_nodes_fut = ctx.get_with_loader(
                    PathBuf::from(ctx.id().as_str()).join("SSECTORS"),
                    GlNodes::lump_loader(self.renderer.clone(), m_type, m_format),
                ).or_loader_default();

                Ok(Box::new(Map {
                    m_type,
                    m_format,
                    sectors: sectors_fut.await?,
                    vertices: vertices_fut.await?,
                    sidedefs: sidedefs_fut.await?,
                    linedefs: linedefs_fut.await?,
                    gl_nodes: gl_nodes_fut.await?,
                }))
            },
            MapFormat::UDMF => Err(ResourceLoadError::from("UDMF map format not yet supported")),
        }
    }
}