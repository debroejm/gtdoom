use std::sync::Arc;
use gtether::render::render_pass::EngineRenderHandler;
use gtether::render::{FlatVertex, Renderer, VulkanoError};
use gtether::render::descriptor_set::EngineDescriptorSet;
use gtether::render::image::ImageSampler;
use gtether::render::pipeline::{EngineGraphicsPipeline, VKGraphicsPipelineSource, ViewportType};
use gtether::resource::Resource;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::{Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::Validated;

use crate::resource::patch::PatchView;

mod vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;
            layout(location = 0) out vec2 tex_coords;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                tex_coords = vec2((position.x + 1.0) / 2.0, (position.y + 1.0) / 2.0);
            }
        ",
    }
}

mod frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec2 tex_coords;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform sampler2D s;

            void main() {
                f_color = texture(s, tex_coords, 1.0);
            }
        ",
    }
}

pub struct FlatRenderer {
    #[allow(unused)]
    patch: Arc<Resource<PatchView>>,
    graphics: Arc<EngineGraphicsPipeline>,
    buffer: Subbuffer<[FlatVertex]>,
    descriptor_set: EngineDescriptorSet,
}

impl FlatRenderer {
    pub fn new(
        renderer: &Arc<Renderer>,
        subpass: &Subpass,
        patch: Arc<Resource<PatchView>>,
    ) -> Self {
        let buffer = FlatVertex::buffer(
            renderer.device().memory_allocator().clone(),
            glm::vec2(-1.0, -1.0),
            glm::vec2(1.0, 1.0),
        );

        let background_vert = vert::load(renderer.device().vk_device().clone())
            .expect("Failed to create vertex shader module")
            .entry_point("main").unwrap();

        let background_frag = frag::load(renderer.device().vk_device().clone())
            .expect("Failed to create fragment shader module")
            .entry_point("main").unwrap();

        let vertex_input_state = Some(FlatVertex::per_vertex()
            .definition(&background_vert.info().input_interface)
            .unwrap());

        let stages = [
            PipelineShaderStageCreateInfo::new(background_vert),
            PipelineShaderStageCreateInfo::new(background_frag),
        ];

        let layout = PipelineLayout::new(
            renderer.device().vk_device().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(renderer.device().vk_device().clone())
                .unwrap(),
        ).unwrap();
        let descriptor_layout = layout.set_layouts().get(0).unwrap().clone();

        let create_info = GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state,
            subpass: Some(subpass.clone().into()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            input_assembly_state: Some(InputAssemblyState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        };

        let graphics = EngineGraphicsPipeline::new(
            renderer,
            create_info,
            ViewportType::TopLeft,
        );

        let sampler = Arc::new(ImageSampler::new(
            patch.read().image_view().clone(),
            Sampler::new(
                renderer.device().vk_device().clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    ..Default::default()
                }
            ).unwrap(),
        ));
        let cb_sampler = sampler.clone();
        patch.attach_update_callback_blocking(move |graphic| {
            let cb_sampler = cb_sampler.clone();
            async move {
                cb_sampler.set_image_view(graphic.read().image_view().clone());
            }
        });

        let descriptor_set = EngineDescriptorSet::builder(renderer.clone())
            .layout(descriptor_layout)
            .descriptor_source(0, sampler)
            .build();

        Self {
            patch,
            graphics,
            buffer,
            descriptor_set,
        }
    }
}

impl EngineRenderHandler for FlatRenderer {
    fn build_commands(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), Validated<VulkanoError>> {
        let graphics = self.graphics.vk_graphics();

        builder
            .bind_pipeline_graphics(graphics.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                0,
                self.descriptor_set.descriptor_set().map_err(VulkanoError::from_validated)?,
            )?
            .bind_vertex_buffers(0, self.buffer.clone())?
            .draw(
                self.buffer.len() as u32,
                1,
                0,
                0,
            )?;
        Ok(())
    }
}