use async_trait::async_trait;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceDefaultLoader, ResourceLoadError, ResourceLoader, ResourceReadData};
use image::{ImageBuffer, ImageReader, Rgba, RgbaImage};
use smol::io::AsyncReadExt;
use std::io::Cursor;

use crate::resource::palette::{Palette, Palettes};

#[derive(Clone)]
pub struct PalettedFlat([u8; PalettedFlat::BYTE_COUNT]);

impl PalettedFlat {
    pub const BYTE_COUNT: usize = 4096;

    #[inline]
    pub fn load(data: &[u8]) -> Result<Self, std::io::Error> {
        Ok(Self(
            data.get(0..Self::BYTE_COUNT)
                .ok_or(std::io::Error::from(std::io::ErrorKind::UnexpectedEof))?
                .try_into().unwrap()
        ))
    }

    pub fn to_flat(&self, palette: &Palette) -> Flat {
        let image = ImageBuffer::from_fn(
            64, 64,
            |x, y| {
                let idx = self.0[x as usize + (64 * y as usize)];
                let px = palette[idx as usize];
                Rgba([px.x, px.y, px.z, 255])
            }
        );

        Flat {
            image,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Flat {
    pub image: RgbaImage,
}

impl ResourceDefaultLoader for Flat {
    type Loader = FlatLoader;

    fn default_loader() -> Self::Loader {
        FlatLoader(())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FlatLoader(());

#[async_trait]
impl ResourceLoader<Flat> for FlatLoader {
    async fn load(
        &self,
        mut data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Flat>, ResourceLoadError> {
        let mut raw = Vec::new();
        data.read_to_end(&mut raw).await?;

        let reader = ImageReader::new(Cursor::new(&raw))
            .with_guessed_format()?;
        if reader.format().is_some() {
            let image = reader.decode()
                .map_err(ResourceLoadError::from_error)?
                .into_rgba8();
            Ok(Box::new(Flat {
                image,
            }))
        } else {
            let palettes_fut = ctx.get::<Palettes>("PLAYPAL");

            let flat = PalettedFlat::load(&raw)?;
            let palettes = palettes_fut.await?;
            let palettes_read = palettes.read();
            let palette = &palettes_read[0];

            Ok(Box::new(flat.to_flat(palette)))
        }
    }
}