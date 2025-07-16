use ahash::{HashMap, HashMapExt};
use async_trait::async_trait;
use gtether::render::{EngineDevice, Renderer};
use gtether::resource::manager::{LoadPriority, ResourceManager};
use gtether::resource::path::ResourcePath;
use gtether::resource::{ResourceLoadError, ResourceLoader, ResourceReadData};
use image::{DynamicImage, ImageReader, Rgba, RgbaImage};
use smol::io::AsyncReadExt;
use std::io;
use std::io::Cursor;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, Validated};

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

impl Palettes {
    pub const LOADER: PaletteLoader = PaletteLoader(());
}

#[derive(Clone, Copy)]
pub struct PaletteLoader(());

#[async_trait]
impl ResourceLoader<Palettes> for PaletteLoader {
    async fn load(
        &self,
        _manager: &Arc<ResourceManager>,
        _id: ResourcePath,
        mut data: ResourceReadData,
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

    pub fn to_rgba8(&self, palette: &Palette) -> RgbaImage {
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

        image
    }
}

pub struct PatchView(Arc<ImageView>);

impl PatchView {
    #[inline]
    pub fn image_view(&self) -> &Arc<ImageView> {
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

    fn build_upload_buffer(
        device: &Arc<EngineDevice>,
        upload_buffer: Subbuffer<[u8]>,
        image: Arc<Image>,
    ) -> Result<Arc<PrimaryAutoCommandBuffer>, ResourceLoadError> {
        let mut uploads = AutoCommandBufferBuilder::primary(
            device.command_buffer_allocator().as_ref(),
            device.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)?;

        uploads.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            upload_buffer,
            image.clone(),
        )).map_err(ResourceLoadError::from_error)?;

        uploads
            .build()
            .map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)
    }
}

#[async_trait]
impl ResourceLoader<PatchView> for PatchViewLoader {
    async fn load(
        &self,
        // TODO: Move manager + things like load priority into a ctx - only access ref without Arc
        manager: &Arc<ResourceManager>,
        _id: ResourcePath,
        mut data: ResourceReadData,
    ) -> Result<Box<PatchView>, ResourceLoadError> {
        let mut raw = Vec::new();
        data.read_to_end(&mut raw).await
            .map_err(ResourceLoadError::from_error)?;

        let reader = ImageReader::new(Cursor::new(&raw))
            .with_guessed_format()
            // TODO: Add common error conversions such as IO errors to ResourceLoadError
            // TODO: Rename enum variant ResourceLoadError to avoid collisions
            .map_err(ResourceLoadError::from_error)?;
        let image_data = if reader.format().is_some() {
            reader.decode()
                .map_err(ResourceLoadError::from_error)?
                .into_rgba8()
        } else {
            let palettes_fut = manager.get_or_load(
                "PLAYPAL",
                Palettes::LOADER,
                LoadPriority::Immediate,
            );

            let patch = PalettedPatch::load(&raw)
                .map_err(ResourceLoadError::from_error)?;
            let palettes = palettes_fut.await?;
            let palettes_read = palettes.read();
            let palette = &palettes_read[0];

            patch.to_rgba8(palette)
        };
        let img_size = glm::vec2(image_data.width(), image_data.height());

        let image = Image::new(
            self.renderer.device().memory_allocator().clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [img_size.x, img_size.y, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        ).map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)?;

        let upload_buffer = Buffer::new_slice(
            self.renderer.device().memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (img_size.x * img_size.y * 4) as DeviceSize,
        ).map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)?;

        upload_buffer.write()
            .map_err(ResourceLoadError::from_error)?
            [..].copy_from_slice(&image_data.into_raw());

        let cmd_buffer = Self::build_upload_buffer(
            self.renderer.device(),
            upload_buffer,
            image.clone(),
        )?;

        cmd_buffer
            .execute(self.renderer.device().queue().clone())
            .map_err(ResourceLoadError::from_error)?
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)?
            .await
            .map_err(ResourceLoadError::from_error)?;

        let image_view = ImageView::new_default(image)
            .map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)?;

        Ok(Box::new(PatchView(image_view)))
    }
}