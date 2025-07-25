use async_trait::async_trait;
use gtether::resource::{Resource, ResourceLoadContext, ResourceLoadError, ResourceLoader, ResourceReadData};
use smol::io::{AsyncRead, AsyncReadExt};
use std::ops::Deref;
use std::sync::Arc;
use tracing::debug;

use crate::resource::map::sector::{Sector, Sectors};
use crate::resource::map::MapType;
use crate::resource::string_from_bytes_until_nul_checked;

#[derive(thiserror::Error, Debug)]
pub enum SidedefLoadError {
    #[error("no sector found for idx: {0}")]
    MissingSector(u16),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

impl From<SidedefLoadError> for ResourceLoadError {
    #[inline]
    fn from(value: SidedefLoadError) -> Self {
        ResourceLoadError::from_error(value)
    }
}

#[derive(Debug, Clone)]
pub struct SidedefProperties {
    pub alpha: f32,
    pub clip_mid_tex: bool,
    pub light: u8,
    pub light_absolute: bool,
    pub light_fog: bool,
    pub no_decals: bool,
    pub no_fake_contrast: bool,
    pub offset_bottom: glm::TVec2<f32>,
    pub offset_middle: glm::TVec2<f32>,
    pub offset_top: glm::TVec2<f32>,
    pub scale_bottom: glm::TVec2<f32>,
    pub scale_middle: glm::TVec2<f32>,
    pub scale_top: glm::TVec2<f32>,
    pub smooth_lighting: bool,
    pub wrap_mid_tex: bool,
}

impl Default for SidedefProperties {
    #[inline]
    fn default() -> Self {
        Self {
            alpha: 1.0,
            clip_mid_tex: false,
            light: 0,
            light_absolute: false,
            light_fog: false,
            no_decals: false,
            no_fake_contrast: false,
            offset_bottom: glm::vec2(0.0, 0.0),
            offset_middle: glm::vec2(0.0, 0.0),
            offset_top: glm::vec2(0.0, 0.0),
            scale_bottom: glm::vec2(1.0, 1.0),
            scale_middle: glm::vec2(1.0, 1.0),
            scale_top: glm::vec2(1.0, 1.0),
            smooth_lighting: false,
            wrap_mid_tex: false,
        }
    }
}

pub struct Sidedef {
    pub offset: glm::TVec2<i16>,
    // TODO: upper/lower/middle textures
    pub sector: Arc<Sector>,
    pub properties: SidedefProperties,
}

impl Sidedef {
    pub const BYTE_COUNT: usize = 30;

    pub fn from_bytes(bytes: &[u8; Self::BYTE_COUNT], sectors: &Sectors) -> Result<Self, SidedefLoadError> {
        let offset = glm::vec2(
            i16::from_le_bytes(bytes[0..2].try_into().unwrap()),
            i16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        );
        let upper_tex_name = string_from_bytes_until_nul_checked(&bytes[4..12]);
        let lower_tex_name = string_from_bytes_until_nul_checked(&bytes[12..20]);
        let middle_tex_name = string_from_bytes_until_nul_checked(&bytes[20..28]);
        let sector_idx = i16::from_le_bytes(bytes[28..30].try_into().unwrap()) as u16;
        let sector = sectors.get(sector_idx as usize)
            .ok_or(SidedefLoadError::MissingSector(sector_idx))?
            .clone();
        let properties = SidedefProperties::default();
        Ok(Self {
            offset,
            sector,
            properties,
        })
    }
}

pub struct Sidedefs {
    inner: Vec<Arc<Sidedef>>,
    sectors: Arc<Resource<Sectors>>,
}

impl Deref for Sidedefs {
    type Target = Vec<Arc<Sidedef>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Sidedefs {
    #[inline]
    pub fn lump_loader(map_type: MapType) -> SidedefsLumpLoader {
        SidedefsLumpLoader {
            map_type,
        }
    }

    pub async fn read(
        mut reader: impl AsyncRead + Unpin,
        sectors: Arc<Resource<Sectors>>,
    ) -> Result<Self, SidedefLoadError> {
        let mut sidedefs = Vec::new();

        {
            let sectors_lock = sectors.read();
            let mut bytes = [0; Sidedef::BYTE_COUNT];
            loop {
                match reader.read_exact(&mut bytes).await {
                    Ok(()) => sidedefs.push(Arc::new(Sidedef::from_bytes(&bytes, &sectors_lock)?)),
                    Err(error) => match error.kind() {
                        std::io::ErrorKind::UnexpectedEof => break,
                        _ => return Err(error.into()),
                    }
                }
            }
        }
        debug!(count = ?sidedefs.len(), "Loaded Sidedefs");

        Ok(Self {
            inner: sidedefs,
            sectors,
        })
    }
}

pub struct SidedefsLumpLoader {
    map_type: MapType,
}

#[async_trait]
impl ResourceLoader<Sidedefs> for SidedefsLumpLoader {
    async fn load(
        &self,
        data: ResourceReadData,
        ctx: ResourceLoadContext,
    ) -> Result<Box<Sidedefs>, ResourceLoadError> {
        let sectors = ctx.manager().get_or_load(
            ctx.id().parent().join("SECTORS"),
            Sectors::lump_loader(self.map_type),
            ctx.priority(),
        ).await?;
        Ok(Box::new(Sidedefs::read(data, sectors).await?))
    }
}