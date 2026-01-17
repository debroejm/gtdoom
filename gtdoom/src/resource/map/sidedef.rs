use async_trait::async_trait;
use gtether::render::Renderer;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{Resource, ResourceLoadError, ResourceLoader, ResourceReadData};
use smol::io::AsyncReadExt;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

use crate::resource::map::sector::{Sector, Sectors};
use crate::resource::map::MapType;
use crate::resource::string_from_bytes_until_nul_checked;
use crate::resource::texture::Texture;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SidedefLocation {
    Upper,
    Middle,
    Lower,
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
    pub upper_tex: Option<Arc<Resource<Texture>>>,
    pub lower_tex: Option<Arc<Resource<Texture>>>,
    pub middle_tex: Option<Arc<Resource<Texture>>>,
    pub sector: Arc<Sector>,
    pub properties: SidedefProperties,
}

impl Sidedef {
    pub const BYTE_COUNT: usize = 30;

    async fn load_texture(
        name: &str,
        renderer: &Arc<Renderer>,
        ctx: &ResourceLoadContext,
    ) -> Result<Option<Arc<Resource<Texture>>>, ResourceLoadError> {
        if name == "-" {
            return Ok(None)
        }

        // Prefer wall textures first
        let result = ctx.get_with_loader(
            PathBuf::from("Textures").join(name),
            Texture::loader(renderer.clone()),
        ).await;

        match result {
            Ok(flat) => Ok(Some(flat)),
            Err(ResourceLoadError::NotFound(_)) => {
                let tex = ctx.get_with_loader(
                    PathBuf::from("Flats").join(name),
                    Texture::flat_loader(renderer.clone()),
                ).await?;
                Ok(Some(tex))
            },
            Err(e) => Err(e),
        }
    }

    async fn load(
        bytes: &[u8; Self::BYTE_COUNT],
        sectors: &Sectors,
        renderer: &Arc<Renderer>,
        ctx: &ResourceLoadContext,
    ) -> Result<Self, ResourceLoadError> {
        let offset = glm::vec2(
            i16::from_le_bytes(bytes[0..2].try_into().unwrap()),
            i16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        );

        let upper_tex_name = string_from_bytes_until_nul_checked(&bytes[4..12]);
        let lower_tex_name = string_from_bytes_until_nul_checked(&bytes[12..20]);
        let middle_tex_name = string_from_bytes_until_nul_checked(&bytes[20..28]);

        let upper_tex = Self::load_texture(&upper_tex_name, renderer, ctx).await?;
        let lower_tex = Self::load_texture(&lower_tex_name, renderer, ctx).await?;
        let middle_tex = Self::load_texture(&middle_tex_name, renderer, ctx).await?;

        let sector_idx = i16::from_le_bytes(bytes[28..30].try_into().unwrap()) as u16;
        let sector = sectors.get(sector_idx as usize)
            .ok_or(SidedefLoadError::MissingSector(sector_idx))?
            .clone();
        let properties = SidedefProperties::default();
        Ok(Self {
            offset,
            upper_tex,
            lower_tex,
            middle_tex,
            sector,
            properties,
        })
    }

    #[inline]
    pub fn texture(&self, location: SidedefLocation) -> Option<&Arc<Resource<Texture>>> {
        match location {
            SidedefLocation::Upper => self.upper_tex.as_ref(),
            SidedefLocation::Middle => self.middle_tex.as_ref(),
            SidedefLocation::Lower => self.lower_tex.as_ref(),
        }
    }

    #[inline]
    pub fn offset(&self, location: SidedefLocation) -> glm::TVec2<f32> {
        let base = self.offset.cast::<f32>();
        match location {
            SidedefLocation::Upper => base + self.properties.offset_top,
            SidedefLocation::Middle => base + self.properties.offset_middle,
            SidedefLocation::Lower => base + self.properties.offset_bottom,
        }
    }

    #[inline]
    pub fn scale(&self, location: SidedefLocation) -> glm::TVec2<f32> {
        match location {
            SidedefLocation::Upper => self.properties.scale_top,
            SidedefLocation::Middle => self.properties.scale_middle,
            SidedefLocation::Lower => self.properties.scale_bottom,
        }
    }

    #[inline]
    pub fn light(&self) -> u8 {
        if self.properties.light_absolute {
            self.properties.light
        } else {
            (self.sector.light_level as u16 + self.properties.light as u16).min(255) as u8
        }
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
    pub fn lump_loader(renderer: Arc<Renderer>, map_type: MapType) -> SidedefsLumpLoader {
        SidedefsLumpLoader {
            renderer,
            map_type,
        }
    }
}

pub struct SidedefsLumpLoader {
    renderer: Arc<Renderer>,
    map_type: MapType,
}

#[async_trait]
impl ResourceLoader<Sidedefs> for SidedefsLumpLoader {
    async fn load(
        &self,
        mut data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Sidedefs>, ResourceLoadError> {
        let sectors = ctx.get_with_loader(
            PathBuf::from(ctx.id().as_str()).parent().unwrap().join("SECTORS"),
            Sectors::lump_loader(self.renderer.clone(), self.map_type),
        ).await?;

        let mut sidedefs = Vec::new();

        {
            let sectors_lock = sectors.read();
            let mut bytes = [0; Sidedef::BYTE_COUNT];
            loop {
                match data.read_exact(&mut bytes).await {
                    Ok(()) => sidedefs.push(Arc::new(Sidedef::load(
                        &bytes,
                        &sectors_lock,
                        &self.renderer,
                        ctx,
                    ).await?)),
                    Err(error) => match error.kind() {
                        std::io::ErrorKind::UnexpectedEof => break,
                        _ => return Err(error.into()),
                    }
                }
            }
        }
        debug!(count = ?sidedefs.len(), "Loaded Sidedefs");

        Ok(Box::new(Sidedefs {
            inner: sidedefs,
            sectors,
        }))
    }
}