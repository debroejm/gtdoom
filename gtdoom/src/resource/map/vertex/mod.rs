use async_trait::async_trait;
use educe::Educe;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceLoadError, ResourceLoader, ResourceReadData};
use i_triangle::i_overlay::i_float::float::compatible::FloatPointCompatible;
use smol::io::{AsyncRead, AsyncReadExt};
use tracing::debug;

use crate::geometry::Aabb;
use crate::resource::indexed::Indexed;

mod _fixed;
pub use _fixed::*;

#[derive(Educe, Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
#[educe(Deref)]
#[repr(transparent)]
pub struct Vertex(glm::TVec2<i16>);

impl Vertex {
    pub const BYTE_COUNT: usize = 4;

    pub fn from_bytes(bytes: &[u8; Self::BYTE_COUNT]) -> Self {
        Self(glm::vec2(
            i16::from_le_bytes(bytes[0..2].try_into().unwrap()),
            i16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        ))
    }
    
    #[inline]
    pub fn to_3d(&self, z: f32) -> glm::TVec3<f32> {
        glm::vec3(self.x as f32, self.y as f32, z)
    }

    #[inline]
    pub fn to_float(&self) -> FloatVertex {
        FloatVertex(self.0.cast::<f32>())
    }
}

impl From<glm::TVec2<i16>> for Vertex {
    #[inline]
    fn from(value: glm::TVec2<i16>) -> Self {
        Self(value)
    }
}

impl From<Vertex> for glm::TVec2<i16> {
    #[inline]
    fn from(value: Vertex) -> Self {
        value.0
    }
}

pub type IndexedVertex = Indexed<Vertex, u16>;

#[derive(Educe, Debug, Clone, Copy, PartialEq)]
#[educe(Deref)]
#[repr(transparent)]
pub struct FloatVertex(glm::TVec2<f32>);

impl FloatVertex {
    #[inline]
    pub fn equal_eps(&self, other: &FloatVertex, epsilon: f32) -> bool {
        let b = glm::equal_eps(self, other, epsilon);
        b.x && b.y
    }
}

impl From<Vertex> for FloatVertex {
    #[inline]
    fn from(value: Vertex) -> Self {
        Self(value.0.cast::<f32>())
    }
}

impl FloatPointCompatible<f32> for FloatVertex {
    #[inline]
    fn from_xy(x: f32, y: f32) -> Self {
        Self(glm::vec2(x, y))
    }

    #[inline]
    fn x(&self) -> f32 {
        self.0.x
    }

    #[inline]
    fn y(&self) -> f32 {
        self.0.y
    }
}

#[derive(Educe, Debug, Clone, Copy, PartialEq)]
#[educe(Deref)]
pub struct IndexedFloatVertex {
    #[educe(Deref)]
    inner: FloatVertex,
    pub idx: u16,
}

impl IndexedFloatVertex {

}

#[derive(Educe)]
#[educe(Deref)]
pub struct Vertices(Vec<Vertex>);

impl Vertices {
    #[inline]
    pub fn lump_loader() -> VerticesLumpLoader {
        VerticesLumpLoader {}
    }

    pub async fn read(
        mut reader: impl AsyncRead + Unpin,
    ) -> Result<Self, std::io::Error> {
        let mut bytes = [0; Vertex::BYTE_COUNT];
        let mut vertices = Vec::new();
        loop {
            match reader.read_exact(&mut bytes).await {
                Ok(()) => vertices.push(Vertex::from_bytes(&bytes)),
                Err(error) => match error.kind() {
                    std::io::ErrorKind::UnexpectedEof => break,
                    _ => return Err(error),
                }
            }
        }
        debug!(count = ?vertices.len(), "Loaded Vertices");
        Ok(Self(vertices))
    }

    pub fn calculate_bounds(&self) -> Aabb<i16> {
        self.0.iter().map(|v| &**v).collect()
    }
}

pub struct VerticesLumpLoader {}

#[async_trait]
impl ResourceLoader for VerticesLumpLoader {
    type Output = Vertices;

    async fn load(
        &self,
        data: ResourceReadData,
        _ctx: &ResourceLoadContext,
    ) -> Result<Box<Vertices>, ResourceLoadError> {
        Ok(Box::new(Vertices::read(data).await?))
    }
}