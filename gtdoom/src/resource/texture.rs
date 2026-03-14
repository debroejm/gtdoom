use async_trait::async_trait;
use bitflags::bitflags;
use gtether::render::image::VKImageViewSource;
use gtether::render::Renderer;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceDefaultLoader, ResourceLoadError, ResourceLoader, ResourceReadData};
use image::{DynamicImage, GenericImage, GenericImageView};
use smol::io::AsyncReadExt;
use std::path::PathBuf;
use std::sync::Arc;
use educe::Educe;
use gtether::resource::id::ResourceId;
use vulkano::image::view::ImageView;

use crate::resource::flat::Flat;
use crate::resource::patch::{PNames, Patch};
use crate::resource::upload_rgba_image;

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct TextureFlags: u16 {
        const WorldPanning = 0x8000;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Doom,
    ZDoom,
}

impl TextureFormat {
    #[inline]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::Doom),
            2 => Some(Self::ZDoom),
            _ => None,
        }
    }

    #[inline]
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::Doom => 1,
            Self::ZDoom => 2,
        }
    }
}

#[derive(Educe, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[educe(Deref)]
pub enum TextureId {
    Flat(ResourceId),
    Texture(ResourceId),
}

impl From<TextureId> for ResourceId {
    #[inline]
    fn from(value: TextureId) -> Self {
        match value {
            TextureId::Flat(id) => id,
            TextureId::Texture(id) => id,
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    pub flags: TextureFlags,
    pub scale: glm::TVec2<u8>,
    pub size: glm::TVec2<u16>,
    pub image_view: Arc<ImageView>,
}

impl VKImageViewSource for Texture {
    #[inline]
    fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }
}

impl Texture {
    pub fn loader(renderer: Arc<Renderer>) -> TextureLoader {
        TextureLoader {
            renderer,
        }
    }

    pub fn flat_loader(renderer: Arc<Renderer>) -> FlatTextureLoader {
        FlatTextureLoader {
            renderer,
        }
    }
}

pub struct TextureLoader {
    renderer: Arc<Renderer>,
}

#[async_trait]
impl ResourceLoader for TextureLoader {
    type Output = Texture;

    async fn load(
        &self,
        mut data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Texture>, ResourceLoadError> {
        let mut header = [0; 1];
        data.read_exact(&mut header).await?;
        let format = TextureFormat::from_byte(header[0])
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown format type: {}", header[0]),
            ))?;
        match format {
            TextureFormat::Doom => {
                // Start loading pnames in the background
                let pnames_fut = ctx.get::<PNames>("PNAMES");

                let mut buf = [0; 10];
                data.read_exact(&mut buf).await?;
                let flags = TextureFlags::from_bits_truncate(
                    u16::from_le_bytes(buf[0..2].try_into().unwrap())
                );
                let scale = glm::vec2(buf[2], buf[3]);
                let size = glm::vec2(
                    u16::from_le_bytes(buf[4..6].try_into().unwrap()),
                    u16::from_le_bytes(buf[6..8].try_into().unwrap()),
                );

                let patch_count = i16::from_le_bytes(buf[8..10].try_into().unwrap()) as usize;
                let (strife_format, patch_count) = if patch_count == 0 {
                    // Assume Doom format, which has extra unused bytes
                    let mut buf = [0; 4];
                    data.read_exact(&mut buf).await?;
                    let patch_count = i16::from_le_bytes(buf[2..4].try_into().unwrap()) as usize;
                    (false, patch_count)
                } else {
                    (true, patch_count)
                };

                struct PatchInfo {
                    offset: glm::TVec2<i16>,
                    idx: usize,
                }

                let mut patch_infos = Vec::new();
                let mut patch_buf = [0; 10];
                for _ in 0..patch_count {
                    if strife_format {
                        data.read_exact(&mut patch_buf[0..6]).await?;
                    } else {
                        data.read_exact(&mut patch_buf).await?;
                    }

                    patch_infos.push(PatchInfo {
                        offset: glm::vec2(
                            i16::from_le_bytes(patch_buf[0..2].try_into().unwrap()),
                            i16::from_le_bytes(patch_buf[2..4].try_into().unwrap()),
                        ),
                        idx: i16::from_le_bytes(patch_buf[4..6].try_into().unwrap()) as usize,
                    });
                }

                let mut rgba_image = DynamicImage::new_rgba8(
                    size.x as u32,
                    size.y as u32,
                ).to_rgba8();

                let pnames_res = pnames_fut.await?;
                let pnames = pnames_res.read();

                for patch_info in patch_infos {
                    let patch_name = pnames.get(patch_info.idx)
                        .ok_or(ResourceLoadError::from(
                            format!("No patch found for idx: {}", patch_info.idx)
                        ))?;
                    // Try the patches namespace first
                    let result = ctx.get::<Patch>(
                        PathBuf::from("Patches").join(patch_name),
                    ).await;
                    let patch_res = match result {
                        Ok(patch) => patch,
                        Err(ResourceLoadError::NotFound(_)) => {
                            // If not found, try the global namespace
                            ctx.get::<Patch>(patch_name).await?
                        },
                        Err(e) => return Err(e),
                    };
                    let patch = patch_res.read();

                    let (sub_x, sub_width, copy_x) = if patch_info.offset.x < 0 {
                        let sub_x = ((patch_info.offset.x * -1) as u32).min(patch.image.width() - 1);
                        let sub_width = (patch.image.width() - 1 - sub_x).min(rgba_image.width());
                        (sub_x, sub_width, 0)
                    } else {
                        let copy_x = (patch_info.offset.x as u32).min(rgba_image.width() - 1);
                        let sub_width = patch.image.width().min(rgba_image.width() - 1 - copy_x);
                        (0, sub_width, copy_x)
                    };

                    let (sub_y, sub_height, copy_y) = if patch_info.offset.y < 0 {
                        let sub_y = ((patch_info.offset.y * -1) as u32).min(patch.image.height() - 1);
                        let sub_height = (patch.image.height() - 1 - sub_y).min(rgba_image.height());
                        (sub_y, sub_height, 0)
                    } else {
                        let copy_y = (patch_info.offset.y as u32).min(rgba_image.height() - 1);
                        let sub_height = patch.image.height().min(rgba_image.height() - 1 - copy_y);
                        (0, sub_height, copy_y)
                    };

                    rgba_image.copy_from(
                        &*patch.image.view(sub_x, sub_y, sub_width, sub_height),
                        copy_x, copy_y,
                    ).map_err(ResourceLoadError::from_error)?;
                }

                let image_view = upload_rgba_image(
                    &self.renderer,
                    rgba_image,
                ).await?;

                Ok(Box::new(Texture {
                    flags,
                    scale,
                    size,
                    image_view,
                }))
            },
            TextureFormat::ZDoom => {
                unimplemented!("ZDoom texture format not yet supported")
            }
        }
    }
}

pub struct FlatTextureLoader {
    renderer: Arc<Renderer>,
}

#[async_trait]
impl ResourceLoader for FlatTextureLoader {
    type Output = Texture;

    async fn load(
        &self,
        data: ResourceReadData,
        ctx: &ResourceLoadContext
    ) -> Result<Box<Texture>, ResourceLoadError> {
        let flat = Flat::default_loader().load(
            data,
            ctx,
        ).await?;

        let size = glm::vec2(flat.image.width() as u16, flat.image.height() as u16);

        let image_view = upload_rgba_image(
            &self.renderer,
            flat.image,
        ).await?;

        Ok(Box::new(Texture {
            flags: TextureFlags::empty(),
            scale: glm::vec2(8, 8),
            size,
            image_view,
        }))
    }
}