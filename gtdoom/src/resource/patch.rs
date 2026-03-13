use ahash::{HashMap, HashMapExt};
use async_trait::async_trait;
use gtether::render::image::VKImageViewSource;
use gtether::render::Renderer;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceDefaultLoader, ResourceLoadError, ResourceLoader, ResourceReadData};
use image::{DynamicImage, ImageReader, Rgba, RgbaImage};
use smol::io::{AsyncRead, AsyncReadExt};
use std::io;
use std::io::Cursor;
use std::sync::Arc;
use vulkano::image::view::ImageView;

use crate::resource::palette::{Palette, Palettes};
use crate::resource::{string_from_bytes_until_nul_checked, upload_rgba_image};

#[derive(Clone)]
pub struct Post {
    top_delta: u8,
    indices: Vec<u8>,
}

#[derive(Clone)]
pub struct PalettedPatch {
    size: glm::TVec2<usize>,
    // TODO: Use offsets
    #[allow(unused)]
    offset: glm::TVec2<i16>,
    columns: Vec<Vec<Post>>,
}

impl PalettedPatch {
    pub fn load(data: &[u8]) -> Result<Self, io::Error> {
        let (size, offset) = {
            let header_buf = data.get(0..8)
                .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?;
            let size = glm::vec2(
                u16::from_le_bytes(header_buf[0..2].try_into().unwrap()) as usize,
                u16::from_le_bytes(header_buf[2..4].try_into().unwrap()) as usize,
            );
            let offset = glm::vec2(
                i16::from_le_bytes(header_buf[4..6].try_into().unwrap()),
                i16::from_le_bytes(header_buf[6..8].try_into().unwrap()),
            );
            (size, offset)
        };

        let column_offsets = {
            let mut offsets = Vec::with_capacity(size.x);
            for col_idx in 0..size.x {
                let idx = 8 + (col_idx * 4);
                let offset = u32::from_le_bytes(
                    data.get(idx..(idx + 4))
                        .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?
                        .try_into().unwrap()
                ) as usize;
                offsets.push(offset);
            }
            offsets
        };

        let columns = {
            let mut columns: Vec<Vec<Post>> = Vec::with_capacity(size.x);
            let mut col_cache: HashMap<usize, Vec<Post>> = HashMap::new();

            for mut col_offset in column_offsets.into_iter() {
                if let Some(column) = col_cache.get(&col_offset) {
                    columns.push(column.clone());
                } else {
                    let mut column = Vec::new();
                    loop {
                        let top_delta = *data.get(col_offset)
                            .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?;
                        if top_delta == 0xFF {
                            // Reached end of this column
                            break
                        }

                        let length = *data.get(col_offset + 1)
                            .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?
                            as usize;

                        let indices = Vec::from(
                            data.get((col_offset + 3)..(col_offset + 3 + length))
                                .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?
                        );

                        column.push(Post {
                            top_delta,
                            indices,
                        });
                        col_offset += 4 + length;
                    }
                    col_cache.insert(col_offset, column.clone());
                    columns.push(column);
                }
            }

            columns
        };

        Ok(Self {
            size,
            offset,
            columns,
        })
    }

    pub fn to_patch(&self, palette: &Palette) -> Patch {
        let mut image = DynamicImage::new_rgba8(
            self.size.x as u32,
            self.size.y as u32,
        ).to_rgba8();

        for (px_x, column) in self.columns.iter().enumerate() {
            for post in column {
                for (px_y, px_idx) in post.indices.iter().enumerate() {
                    let px = palette[(*px_idx) as usize];
                    image.put_pixel(
                        px_x as u32,
                        post.top_delta as u32 + px_y as u32,
                        Rgba([px.x, px.y, px.z, 255]),
                    );
                }
            }
        }

        Patch {
            image,
            offset: self.offset,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Patch {
    pub image: RgbaImage,
    pub offset: glm::TVec2<i16>,
}

impl ResourceDefaultLoader for Patch {
    type Loader = PatchLoader;

    fn default_loader() -> Self::Loader {
        PatchLoader(())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PatchLoader(());

#[async_trait]
impl ResourceLoader for PatchLoader {
    type Output = Patch;

    async fn load(
        &self,
        mut data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Patch>, ResourceLoadError> {
        let mut raw = Vec::new();
        data.read_to_end(&mut raw).await?;

        let reader = ImageReader::new(Cursor::new(&raw))
            .with_guessed_format()?;
        if reader.format().is_some() {
            let image = reader.decode()
                .map_err(ResourceLoadError::from_error)?
                .into_rgba8();
            Ok(Box::new(Patch {
                image,
                offset: glm::vec2(0, 0),
            }))
        } else {
            let palettes_fut = ctx.get::<Palettes>("PLAYPAL");

            let patch = PalettedPatch::load(&raw)?;
            let palettes = palettes_fut.await?;
            let palettes_read = palettes.read();
            let palette = &palettes_read[0];

            Ok(Box::new(patch.to_patch(palette)))
        }
    }
}

#[derive(Debug, Clone)]
struct PatchView(Arc<ImageView>);

impl VKImageViewSource for PatchView {
    #[inline]
    fn image_view(&self) -> &Arc<ImageView> {
        &self.0
    }
}

pub struct PatchViewLoader {
    renderer: Arc<Renderer>,
}

impl PatchViewLoader {
    pub fn new(renderer: Arc<Renderer>) -> Self {
        Self {
            renderer,
        }
    }
}

#[async_trait]
impl ResourceLoader for PatchViewLoader {
    type Output = dyn VKImageViewSource;

    async fn load(
        &self,
        data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<dyn VKImageViewSource>, ResourceLoadError> {
        let patch = Patch::default_loader().load(
            data,
            ctx
        ).await?;

        let image_view = upload_rgba_image(
            &self.renderer,
            patch.image,
        ).await?;

        Ok(Box::new(PatchView(image_view)))
    }
}

pub struct PNames(Vec<String>);

impl PNames {
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&str> {
        self.0.get(idx).map(AsRef::as_ref)
    }

    pub async fn read(
        mut reader: impl AsyncRead + Unpin,
    ) -> Result<Self, io::Error> {
        let mut header = [0; 4];
        reader.read_exact(&mut header).await?;
        let count = i32::from_le_bytes(header);
        let mut pnames = Vec::with_capacity(count as usize);

        let mut buf = [0; 8];
        for _ in 0..count {
            reader.read_exact(&mut buf).await?;
            pnames.push(string_from_bytes_until_nul_checked(&buf));
        }

        Ok(Self(pnames))
    }
}

impl ResourceDefaultLoader for PNames {
    type Loader = PNamesLoader;

    fn default_loader() -> Self::Loader {
        PNamesLoader(())
    }
}

#[derive(Default, Clone, Copy)]
pub struct PNamesLoader(());

#[async_trait]
impl ResourceLoader for PNamesLoader {
    type Output = PNames;

    async fn load(
        &self,
        data: ResourceReadData,
        _ctx: &ResourceLoadContext,
    ) -> Result<Box<PNames>, ResourceLoadError> {
        Ok(Box::new(PNames::read(data).await?))
    }
}