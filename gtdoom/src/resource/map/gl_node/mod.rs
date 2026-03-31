use async_trait::async_trait;
use educe::Educe;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{Resource, ResourceLoadError, ResourceLoader, ResourceLoaderDefault, ResourceLock, ResourceReadData};
use smol::io::{AsyncRead, AsyncReadExt};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::warn;

use crate::geometry::Aabb;
use crate::resource::indexed::IndexedArc;
use crate::resource::map::linedef::{Linedef, Linedefs};
use crate::resource::map::sector::Sector;
use crate::resource::map::vertex::{FixedFloat, FixedVertex, IndexedFixedVertex, IndexedVertex, Vertices};
use crate::resource::map::{MapFormat, MapType};

mod builder;

pub use builder::GlNodesBuilder;

#[derive(thiserror::Error, Debug)]
pub enum GlNodesLoadError {
    #[error("not supported: {0}")]
    NotSupported(NodesMetadata),
    #[error("no vertex found for idx: {0}")]
    MissingVertex(u32),
    #[error("no linedef found for idx: {0}")]
    MissingLinedef(u32),
    #[error("no gl_seg found for idx: {0}")]
    MissingGlSeg(u32),
    #[error("invalid data found: {0}")]
    InvalidData(String),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error("invalid ID path: {0}; must be in the format <map>/*")]
    InvalidIdPath(PathBuf),
}

impl From<GlNodesLoadError> for ResourceLoadError {
    #[inline]
    fn from(value: GlNodesLoadError) -> Self {
        Self::from_error(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlIndexedVertex {
    Vertex(IndexedVertex),
    FixedVertex(IndexedFixedVertex),
}

impl GlIndexedVertex {
    #[inline]
    pub fn to_fixed(self) -> IndexedFixedVertex {
        match self {
            Self::Vertex(v) => IndexedFixedVertex::from(v),
            Self::FixedVertex(v) => v,
        }
    }
}

fn fmt_vec_len<T>(vec: &Vec<T>, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "[..; {}]", vec.len())
}

#[derive(Educe)]
#[educe(Debug)]
pub struct GlVertices {
    original_vertex_count: u32,
    #[educe(Debug(ignore))]
    original_vertices: Arc<Resource<Vertices>>,
    #[educe(Debug(method(fmt_vec_len)))]
    fixed_vertices: Vec<FixedVertex>,
}

impl GlVertices {
    #[inline]
    pub fn read(&self) -> GlVerticesReadLock<'_> {
        GlVerticesReadLock {
            gl_vertices: self,
            original_vertices: self.original_vertices.read(),
        }
    }
}

pub struct GlVerticesReadLock<'a> {
    gl_vertices: &'a GlVertices,
    original_vertices: ResourceLock<'a, Vertices>,
}

impl<'a> GlVerticesReadLock<'a> {
    pub fn contains_idx(&self, idx: u32) -> bool {
        if idx < self.gl_vertices.original_vertex_count {
            (idx as usize) < self.original_vertices.len()
        } else {
            ((idx - self.gl_vertices.original_vertex_count) as usize) < self.gl_vertices.fixed_vertices.len()
        }
    }

    pub fn get(&self, idx: u32) -> Option<GlIndexedVertex> {
        if idx < self.gl_vertices.original_vertex_count {
            let v = *self.original_vertices.get(idx as usize)?;
            Some(GlIndexedVertex::Vertex(IndexedVertex::new(v, idx as u16)))
        } else {
            let fixed_idx = idx - self.gl_vertices.original_vertex_count;
            let v = *self.gl_vertices.fixed_vertices.get(fixed_idx as usize)?;
            Some(GlIndexedVertex::FixedVertex(IndexedFixedVertex::new(v, idx)))
        }
    }

    pub fn get_index(&self, idx: u32) -> Option<GlVertexIndex> {
        if idx < self.gl_vertices.original_vertex_count {
            if (idx as usize) < self.original_vertices.len() {
                // TODO: check idx is below u16::MAX
                Some(GlVertexIndex::Original(idx as u16))
            } else {
                None
            }
        } else {
            let idx = idx - self.gl_vertices.original_vertex_count;
            if (idx as usize) < self.gl_vertices.fixed_vertices.len() {
                Some(GlVertexIndex::Fixed(idx))
            } else {
                None
            }
        }
    }

    pub fn get_indexed(&self, idx: GlVertexIndex) -> Option<GlIndexedVertex> {
        match idx {
            GlVertexIndex::Original(idx) => {
                if (idx as u32) < self.gl_vertices.original_vertex_count {
                    let v = *self.original_vertices.get(idx as usize)?;
                    Some(GlIndexedVertex::Vertex(IndexedVertex::new(v, idx)))
                } else {
                    None
                }
            },
            GlVertexIndex::Fixed(idx) => {
                let v = *self.gl_vertices.fixed_vertices.get(idx as usize)?;
                Some(GlIndexedVertex::FixedVertex(IndexedFixedVertex::new(v, idx)))
            }
        }
    }

    pub fn calculate_bounds(&self) -> Aabb<FixedFloat> {
        let original_bounds = self.original_vertices.calculate_bounds()
            .map(FixedFloat::from);

        if !self.gl_vertices.fixed_vertices.is_empty() {
            let bounds = self.gl_vertices.fixed_vertices.iter()
                .map(|v| &**v)
                .collect::<Aabb<_>>();
            bounds | original_bounds
        } else {
            original_bounds
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GlVertexIndex {
    Original(u16),
    Fixed(u32),
}

pub struct GlSeg {
    pub vertex_idx: GlVertexIndex,
    pub linedef: Option<IndexedArc<Linedef>>,
    pub front: bool,
}

impl GlSeg {
    const COMMON_BYTE_COUNT: usize = 8;
    pub const GLN_BYTE_COUNT: usize = 11;
    pub const GL2_BYTE_COUNT: usize = 13;
    pub const GL3_BYTE_COUNT: usize = Self::GL2_BYTE_COUNT;

    fn from_common_bytes(
        bytes: &[u8; Self::COMMON_BYTE_COUNT],
        vertices: &GlVerticesReadLock,
    ) -> Result<GlVertexIndex, GlNodesLoadError> {
        let vertex_idx = {
            let idx = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            vertices.get_index(idx)
                .ok_or(GlNodesLoadError::MissingVertex(idx))?
        };
        // TODO: is this actually needed anywhere?
        let _partner = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        Ok(vertex_idx)
    }

    pub fn from_gln_bytes(
        bytes: &[u8; Self::GLN_BYTE_COUNT],
        vertices: &GlVerticesReadLock,
        linedefs: &Linedefs,
    ) -> Result<Self, GlNodesLoadError> {
        let vertex_idx = Self::from_common_bytes(
            (&bytes[0..Self::COMMON_BYTE_COUNT]).try_into().unwrap(),
            vertices,
        )?;
        let linedef = {
            let idx = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
            if idx == 0xFFFFu16 {
                None
            } else {
                Some(linedefs.get(idx as usize)
                    .ok_or(GlNodesLoadError::MissingLinedef(idx as u32))?
                    .to_arc())
            }
        };
        let front = bytes[10] == 0;

        Ok(GlSeg {
            vertex_idx,
            linedef,
            front,
        })
    }

    pub fn from_gl2_bytes(
        bytes: &[u8; Self::GL2_BYTE_COUNT],
        vertices: &GlVerticesReadLock,
        linedefs: &Linedefs,
    ) -> Result<Self, GlNodesLoadError> {
        let vertex_idx = Self::from_common_bytes(
            (&bytes[0..Self::COMMON_BYTE_COUNT]).try_into().unwrap(),
            vertices,
        )?;
        let linedef = {
            let idx = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
            if idx == 0xFFFFFFFFu32 {
                None
            } else {
                Some(linedefs.get(idx as usize)
                    .ok_or(GlNodesLoadError::MissingLinedef(idx))?
                    .to_arc())
            }
        };
        let front = bytes[12] == 0;

        Ok(GlSeg {
            vertex_idx,
            linedef,
            front,
        })
    }

    #[inline]
    pub fn from_gl3_bytes(
        bytes: &[u8; Self::GL3_BYTE_COUNT],
        vertices: &GlVerticesReadLock,
        linedefs: &Linedefs,
    ) -> Result<Self, GlNodesLoadError> {
        Self::from_gl2_bytes(bytes, vertices, linedefs)
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct GlSSector {
    pub sector: IndexedArc<Sector>,
    #[educe(Debug(method(fmt_vec_len)))]
    pub segs: Vec<GlSeg>,
}

impl GlSSector {
    pub fn from_segs(
        gl_segs: Vec<GlSeg>,
    ) -> Result<Self, GlNodesLoadError> {
        let mut sector: Option<IndexedArc<Sector>> = None;
        for seg in &gl_segs {
            if let Some(linedef) = &seg.linedef {
                let sidedef = match seg.front {
                    true => &linedef.front_sidedef,
                    false => &linedef.back_sidedef,
                }.as_ref().ok_or(GlNodesLoadError::InvalidData(
                    "linedef on sub-sector should contain a valid sidedef".to_string()
                ))?;

                match &sector {
                    Some(sector) => {
                        if sidedef.sector.idx != sector.idx {
                            return Err(GlNodesLoadError::InvalidData(format!(
                                "multiple sector indices found in one subsector ({} != {})",
                                sidedef.sector.idx,
                                sector.idx,
                            )))
                        }
                    },
                    None => {
                        sector = Some(sidedef.sector.clone());
                    },
                }
            }
        }

        // TODO: Validate that sub-sector is convex

        let sector = match sector {
            Some(s) => s,
            None => {
                return Err(GlNodesLoadError::InvalidData("no parent sector found for sub-sector".to_owned()))
            },
        };

        Ok(Self {
            sector,
            segs: gl_segs,
        })
    }

    pub fn contour(&self, gl_vertices: &GlVerticesReadLock) -> impl Iterator<Item=GlIndexedVertex> {
        self.segs.iter()
            .map(|seg| {
                gl_vertices.get_indexed(seg.vertex_idx)
                    .expect("seg vertex idx should be valid")
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodesType {
    Vanilla,
    ZDoomExtended,
    ZDoomExtendedGl,
    ZDoomExtendedGl2,
    ZDoomExtendedGl3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodesMetadata {
    pub nodes_t: NodesType,
    pub compressed: bool,
}

impl Display for NodesMetadata {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match (self.nodes_t, self.compressed) {
            (NodesType::Vanilla, _) => write!(f, "<vanilla>"),
            (NodesType::ZDoomExtended, false) => write!(f, "XNOD"),
            (NodesType::ZDoomExtended, true) => write!(f, "ZNOD"),
            (NodesType::ZDoomExtendedGl, false) => write!(f, "XGLN"),
            (NodesType::ZDoomExtendedGl, true) => write!(f, "ZGLN"),
            (NodesType::ZDoomExtendedGl2, false) => write!(f, "XGL2"),
            (NodesType::ZDoomExtendedGl2, true) => write!(f, "ZGL2"),
            (NodesType::ZDoomExtendedGl3, false) => write!(f, "XGL3"),
            (NodesType::ZDoomExtendedGl3, true) => write!(f, "ZGL3"),
        }
    }
}

impl NodesMetadata {
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        let compressed = if bytes[0] == 0x58u8 {
            false
        } else if bytes[0] == 0x5Au8 {
            true
        } else {
            return Self {
                nodes_t: NodesType::Vanilla,
                compressed: false,
            }
        };

        let nodes_t = if bytes[1..4] == [0x4Eu8, 0x4Fu8, 0x44u8] {
            NodesType::ZDoomExtended
        } else if bytes[1..4] == [0x47u8, 0x4Cu8, 0x4Eu8] {
            NodesType::ZDoomExtendedGl
        } else if bytes[1..4] == [0x47u8, 0x4Cu8, 0x32u8] {
            NodesType::ZDoomExtendedGl2
        } else if bytes[1..4] == [0x47u8, 0x4Cu8, 0x33u8] {
            NodesType::ZDoomExtendedGl3
        } else {
            return Self {
                nodes_t: NodesType::Vanilla,
                compressed: false,
            }
        };

        Self {
            nodes_t,
            compressed,
        }
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct GlNodes {
    pub gl_vertices: GlVertices,
    #[educe(Debug(method(fmt_vec_len)))]
    pub gl_ssectors: Vec<GlSSector>,
}

impl GlNodes {
    #[inline]
    pub fn lump_loader(
        map_type: MapType,
        map_format: MapFormat,
    ) -> GlNodesLumpLoader {
        GlNodesLumpLoader {
            map_type,
            map_format,
        }
    }

    pub async fn read(
        mut reader: impl AsyncRead + Unpin,
        metadata: NodesMetadata,
        vertices: Arc<Resource<Vertices>>,
        linedefs: Arc<Resource<Linedefs>>,
    ) -> Result<Self, GlNodesLoadError> {
        let gl_vertices = {
            let mut header = [0; 8];
            reader.read_exact(&mut header).await?;

            let org_v_count = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let new_v_count = u32::from_le_bytes(header[4..8].try_into().unwrap());

            let mut verts = Vec::with_capacity(new_v_count as usize);
            let mut bytes = [0; FixedVertex::BYTE_COUNT];
            for _ in 0..new_v_count {
                reader.read_exact(&mut bytes).await?;
                verts.push(FixedVertex::from_bytes(&bytes));
            }

            GlVertices {
                original_vertex_count: org_v_count,
                original_vertices: vertices,
                fixed_vertices: verts,
            }
        };

        let subsector_counts = {
            let mut header = [0; 4];
            reader.read_exact(&mut header).await?;
            let count = u32::from_le_bytes(header);

            let mut subsector_counts = Vec::with_capacity(count as usize);
            let mut bytes = [0; 4];
            for _ in 0..count {
                reader.read_exact(&mut bytes).await?;
                subsector_counts.push(u32::from_le_bytes(bytes));
            }

            subsector_counts
        };

        let gl_segs = {
            let mut header = [0; 4];
            reader.read_exact(&mut header).await?;
            let count = u32::from_le_bytes(header);

            let linedefs_lock = linedefs.read();
            let gl_vertices_lock = gl_vertices.read();
            let mut gl_segs = Vec::with_capacity(count as usize);
            match metadata.nodes_t {
                NodesType::ZDoomExtendedGl => {
                    let mut bytes = [0; GlSeg::GLN_BYTE_COUNT];
                    for _ in 0..count {
                        reader.read_exact(&mut bytes).await?;
                        gl_segs.push(GlSeg::from_gln_bytes(
                            &bytes,
                            &gl_vertices_lock,
                            &linedefs_lock,
                        )?);
                    }
                },
                NodesType::ZDoomExtendedGl2 => {
                    let mut bytes = [0; GlSeg::GL2_BYTE_COUNT];
                    for _ in 0..count {
                        reader.read_exact(&mut bytes).await?;
                        gl_segs.push(GlSeg::from_gl2_bytes(
                            &bytes,
                            &gl_vertices_lock,
                            &linedefs_lock,
                        )?);
                    }
                },
                NodesType::ZDoomExtendedGl3 => {
                    let mut bytes = [0; GlSeg::GL3_BYTE_COUNT];
                    for _ in 0..count {
                        reader.read_exact(&mut bytes).await?;
                        gl_segs.push(GlSeg::from_gl3_bytes(
                            &bytes,
                            &gl_vertices_lock,
                            &linedefs_lock,
                        )?);
                    }
                },
                _ => return Err(GlNodesLoadError::NotSupported(metadata)),
            }

            gl_segs
        };

        let gl_ssectors = {
            let mut gl_ssectors = Vec::with_capacity(subsector_counts.len());
            let mut gl_segs_iter = gl_segs.into_iter();
            let mut base_idx = 0;
            for subsector_count in subsector_counts {
                let segs = gl_segs_iter.by_ref()
                    .take(subsector_count as usize)
                    .collect::<Vec<_>>();
                let segs_count = segs.len() as u32;
                if segs_count < subsector_count {
                    return Err(GlNodesLoadError::MissingGlSeg(base_idx + segs_count));
                }
                base_idx += subsector_count;
                gl_ssectors.push(GlSSector::from_segs(segs)?);
            }
            gl_ssectors
        };

        Ok(Self {
            gl_vertices,
            gl_ssectors,
        })
    }
}

pub struct GlNodesLumpLoader {
    map_type: MapType,
    map_format: MapFormat,
}

impl GlNodesLumpLoader {
    fn get_parent_id_path<P: AsRef<Path>>(
        &self,
        ctx: &ResourceLoadContext,
        path: P,
    ) -> Result<PathBuf, GlNodesLoadError> {
        let id = PathBuf::from(ctx.id().as_str());
        match id.parent() {
            Some(parent) => Ok(parent.join(path)),
            None => Err(GlNodesLoadError::InvalidIdPath(path.as_ref().to_path_buf()))
        }
    }
}

#[async_trait]
impl ResourceLoader for GlNodesLumpLoader {
    type Output = GlNodes;

    async fn load(
        &self,
        mut data: ResourceReadData,
        ctx: &ResourceLoadContext
    ) -> Result<Box<GlNodes>, ResourceLoadError> {
        let vertices_fut = ctx.get_with_loader(
            self.get_parent_id_path(ctx, "VERTEXES")?,
            Vertices::lump_loader(),
        );
        let linedefs_fut = ctx.get_with_loader(
            self.get_parent_id_path(ctx, "LINEDEFS")?,
            Linedefs::lump_loader(self.map_type, self.map_format),
        );

        let mut signature = [0; 4];
        data.read_exact(&mut signature).await?;
        let metadata = NodesMetadata::from_bytes(&signature);

        match (metadata.nodes_t, metadata.compressed) {
            (NodesType::ZDoomExtendedGl, false) |
            (NodesType::ZDoomExtendedGl2, false) |
            (NodesType::ZDoomExtendedGl3, false) => {
                Ok(Box::new(GlNodes::read(
                    data,
                    metadata,
                    vertices_fut.await?,
                    linedefs_fut.await?,
                ).await?))
            },
            _ => {
                warn!(?metadata, "Unsupported NodesType; treating lump as non-existent");
                // TODO: add support in upstream gTether to allow defaulting/alternate loading on
                //  error types other than just ResourceLoadError::NotFound()
                Err(ResourceLoadError::NotFound(ctx.id()))
            }
        }
    }
}

#[async_trait]
impl ResourceLoaderDefault for GlNodesLumpLoader {
    async fn load_default(
        &self,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Self::Output>, ResourceLoadError> {
        let vertices_fut = ctx.get_with_loader(
            self.get_parent_id_path(ctx, "VERTEXES")?,
            Vertices::lump_loader(),
        );
        let linedefs_fut = ctx.get_with_loader(
            self.get_parent_id_path(ctx, "LINEDEFS")?,
            Linedefs::lump_loader(self.map_type, self.map_format),
        );

        let vertices = vertices_fut.await?;
        let linedefs = linedefs_fut.await?;

        let mut builder = GlNodesBuilder::default();
        for linedef in linedefs.read().iter() {
            builder.add_linedef(linedef);
        }
        let gl_nodes = builder.build(vertices);

        Ok(Box::new(gl_nodes))
    }
}
