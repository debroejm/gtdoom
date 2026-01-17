use async_trait::async_trait;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceLoadError, ResourceLoader, ResourceReadData};
use i_triangle::i_overlay::i_float::float::compatible::FloatPointCompatible;
use smol::io::{AsyncRead, AsyncReadExt};
use std::ops::Deref;
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Vertex(glm::TVec2<i16>);

impl Deref for Vertex {
    type Target = glm::TVec2<i16>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct FloatVertex(glm::TVec2<f32>);

impl FloatVertex {
    #[inline]
    pub fn equal_eps(&self, other: &FloatVertex, epsilon: f32) -> bool {
        let b = glm::equal_eps(self, other, epsilon);
        b.x && b.y
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

impl Deref for FloatVertex {
    type Target = glm::TVec2<f32>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Vertices(Vec<Vertex>);

impl Deref for Vertices {
    type Target = Vec<Vertex>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
}

pub struct VerticesLumpLoader {}

#[async_trait]
impl ResourceLoader<Vertices> for VerticesLumpLoader {
    async fn load(
        &self,
        data: ResourceReadData,
        _ctx: &ResourceLoadContext,
    ) -> Result<Box<Vertices>, ResourceLoadError> {
        Ok(Box::new(Vertices::read(data).await?))
    }
}