use async_trait::async_trait;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceDefaultLoader, ResourceLoadError, ResourceLoader, ResourceReadData};
use smol::io::AsyncReadExt;
use std::ops::Deref;

pub struct Palette([glm::TVec3<u8>; 256]);

impl Deref for Palette {
    type Target = [glm::TVec3<u8>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Palette {
    fn from_raw(buffer: &[u8; 256 * 3]) -> Self {
        Self(std::array::from_fn(|idx| glm::vec3(
            buffer[idx * 3],
            buffer[(idx * 3) + 1],
            buffer[(idx * 3) + 2],
        )))
    }
}

pub struct Palettes(Vec<Palette>);

impl Deref for Palettes {
    type Target = Vec<Palette>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ResourceDefaultLoader for Palettes {
    type Loader = PaletteLoader;

    fn default_loader() -> Self::Loader {
        PaletteLoader(())
    }
}

#[derive(Clone, Copy)]
pub struct PaletteLoader(());

#[async_trait]
impl ResourceLoader for PaletteLoader {
    type Output = Palettes;

    async fn load(
        &self,
        mut data: ResourceReadData,
        _ctx: &ResourceLoadContext,
    ) -> Result<Box<Palettes>, ResourceLoadError> {
        let mut palette_buffer = [0; 256 * 3];
        let mut palettes = Vec::new();

        'main: loop {
            match data.read_exact(&mut palette_buffer).await {
                Ok(()) => palettes.push(Palette::from_raw(&palette_buffer)),
                Err(error) => {
                    match error.kind() {
                        std::io::ErrorKind::UnexpectedEof => break 'main,
                        _ => return Err(ResourceLoadError::from_error(error)),
                    }
                }
            }
        };

        Ok(Box::new(Palettes(palettes)))
    }
}