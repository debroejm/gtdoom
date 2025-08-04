use gtether::render::{EngineDevice, Renderer};
use gtether::resource::ResourceLoadError;
use image::RgbaImage;
use std::ffi::CStr;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, Validated, VulkanError};

pub mod flat;
pub mod lump;
pub mod map;
pub mod palette;
pub mod patch;
pub mod texture;

#[inline]
pub fn string_from_bytes_until_nul_checked(bytes: &[u8]) -> String {
    match CStr::from_bytes_until_nul(bytes) {
        Ok(c_str) => c_str.to_string_lossy().to_string(),
        Err(_) => String::from_utf8_lossy(bytes).to_string(),
    }
}

// Utility function to hide these mechanisms from async blocks, which get confused due to some of
// these types being !Send
fn build_image_upload_buffer(
    device: &Arc<EngineDevice>,
    upload_buffer: Subbuffer<[u8]>,
    image: Arc<Image>,
) -> Result<Arc<PrimaryAutoCommandBuffer>, VulkanError> {
    let mut uploads = AutoCommandBufferBuilder::primary(
        device.command_buffer_allocator().as_ref(),
        device.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).map_err(Validated::unwrap)?;

    uploads.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        upload_buffer,
        image.clone(),
    )).unwrap();

    uploads
        .build()
        .map_err(Validated::unwrap)
}

async fn upload_rgba_image(
    renderer: &Arc<Renderer>,
    rgba_image: RgbaImage,
) -> Result<Arc<ImageView>, ResourceLoadError> {
    let img_size = glm::vec2(rgba_image.width(), rgba_image.height());

    let image = Image::new(
        renderer.device().memory_allocator().clone(),
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
        renderer.device().memory_allocator().clone(),
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
        [..].copy_from_slice(&rgba_image.into_raw());

    let cmd_buffer = build_image_upload_buffer(
        renderer.device(),
        upload_buffer,
        image.clone(),
    ).map_err(ResourceLoadError::from_error)?;

    cmd_buffer
        .execute(renderer.device().queue().clone())
        .map_err(ResourceLoadError::from_error)?
        .then_signal_fence_and_flush()
        .map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)?
        .await
        .map_err(ResourceLoadError::from_error)?;

    ImageView::new_default(image)
        .map_err(Validated::unwrap).map_err(ResourceLoadError::from_error)
}