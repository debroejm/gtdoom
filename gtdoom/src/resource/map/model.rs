use ahash::HashMap;
use fixed::traits::LossyFrom;
use gtether::render::descriptor_set::EngineDescriptorSet;
use gtether::render::image::{EngineResourceImages, EngineSampler};
use gtether::render::pipeline::{EngineGraphicsPipeline, VKGraphicsPipelineSource, ViewportType};
use gtether::render::render_pass::EngineRenderHandler;
use gtether::render::uniform::Uniform;
use gtether::render::{EngineDevice, Renderer, VulkanoError};
use gtether::resource::id::ResourceId;
use gtether::resource::Resource;
use i_triangle::float::triangulation::Triangulation;
use i_triangle::float::triangulator::Triangulator;
use i_triangle::i_overlay::core::solver::Solver;
use i_triangle::int::validation::Validation;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use tracing::{debug, warn};
use vulkano::buffer::{AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as VkVertex, VertexDefinition};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::Validated;

use crate::render::camera::{Camera, VP};
use crate::resource::map::gl_node::{GlSSector, GlVerticesReadLock};
use crate::resource::map::linedef::{Linedef, LinedefFlags};
use crate::resource::map::sidedef::{Sidedef, SidedefLocation};
use crate::resource::map::vertex::FixedVertex;
use crate::resource::texture::Texture;

const MAP_VERTEX_EPSILON: f32 = 0.1;
pub type MapVertexIndex = u32;
pub type MapTextureIndex = u32;

#[derive(BufferContents, VkVertex, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct MapVertex {
    #[format(R32G32B32_SFLOAT)]
    position: glm::TVec3<f32>,
    #[format(R32G32_SFLOAT)]
    uv: glm::TVec2<f32>,
    #[format(R32_SFLOAT)]
    light: f32,
    #[format(R16_UINT)]
    tag: u16,
}

impl MapVertex {
    pub fn equal_eps(&self, other: &MapVertex, epsilon: f32) -> bool {
        let pos = glm::equal_eps(&self.position, &other.position, epsilon);
        if !(pos.x && pos.y && pos.z) {
            return false
        }
        let uv = glm::equal_eps(&self.uv, &other.uv, 0.001);
        if !(uv.x && uv.y) {
            return false
        }
        if (self.light - other.light).abs() >= 0.001 {
            return false
        }
        self.tag == other.tag
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ChunkBuildError {
    #[error(transparent)]
    AllocateBufferError(#[from] AllocateBufferError),
}

impl From<Validated<AllocateBufferError>> for ChunkBuildError {
    #[inline]
    fn from(value: Validated<AllocateBufferError>) -> Self {
        Self::AllocateBufferError(value.unwrap())
    }
}

#[derive(Default, Debug, Clone)]
pub struct MapVertices(Vec<MapVertex>);

impl MapVertices {
    fn find_index(&self, search_vertex: &MapVertex) -> Option<MapVertexIndex> {
        for (idx, vertex) in self.0.iter().enumerate() {
            if vertex.equal_eps(search_vertex, MAP_VERTEX_EPSILON) {
                return Some(idx as MapVertexIndex);
            }
        }
        None
    }

    #[inline]
    pub fn get_or_insert(&mut self, vertex: MapVertex) -> MapVertexIndex {
        if let Some(idx) = self.find_index(&vertex) {
            idx
        } else {
            let idx = self.0.len();
            self.0.push(vertex);
            idx as MapVertexIndex
        }
    }

    #[inline]
    pub fn extend(&mut self, other: impl IntoIterator<Item=MapVertex>) -> Vec<MapVertexIndex> {
        other.into_iter()
            .map(|v| self.get_or_insert(v))
            .collect()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn into_vertex_buffer(
        self,
        device: &Arc<EngineDevice>,
    ) -> Result<Subbuffer<[MapVertex]>, AllocateBufferError> {
        Buffer::from_iter(
            device.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.0,
        ).map_err(Validated::unwrap)
    }
}

impl IntoIterator for MapVertices {
    type Item = MapVertex;
    type IntoIter = std::vec::IntoIter<MapVertex>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Default, Debug, Clone)]
pub struct MapVertexIndices(Vec<MapVertexIndex>);

impl MapVertexIndices {
    #[inline]
    pub fn extend(&mut self, other: impl IntoIterator<Item=MapVertexIndex>) {
        self.0.extend(other.into_iter())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn into_index_buffer(
        self,
        device: &Arc<EngineDevice>,
    ) -> Result<IndexBuffer, AllocateBufferError> {
        Ok(Buffer::from_iter(
            device.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.0,
        ).map_err(Validated::unwrap)?.into())
    }
}

impl IntoIterator for MapVertexIndices {
    type Item = MapVertexIndex;
    type IntoIter = std::vec::IntoIter<MapVertexIndex>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Default, Clone)]
pub struct MapTextures {
    textures: Vec<Arc<Resource<Texture>>>,
    index_map: HashMap<ResourceId, MapTextureIndex>,
}

impl Debug for MapTextures {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (idx, tex) in self.textures.iter().enumerate() {
            map.entry(&idx, tex.id());
        }
        map.finish()
    }
}

impl MapTextures {
    #[inline]
    pub fn get_or_insert(&mut self, texture: &Arc<Resource<Texture>>) -> MapTextureIndex {
        if let Some(idx) = self.index_map.get(texture.id()) {
            *idx
        } else {
            let idx = self.textures.len() as MapTextureIndex;
            self.index_map.insert(texture.id().clone(), idx);
            self.textures.push(texture.clone());
            idx
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    #[inline]
    pub fn into_descriptor_images(self) -> Arc<EngineResourceImages<Texture>> {
        EngineResourceImages::new(self.textures)
    }
}

#[derive(Default, Debug, Clone)]
pub struct MapModelData {
    pub vertices: MapVertices,
    pub vertex_indices: MapVertexIndices,
    pub textures: MapTextures,
}

struct IndexChunk {
    texture: Arc<Resource<Texture>>,
    indices: Vec<MapVertexIndex>,
}

pub struct ChunkBuilder {
    vertices: MapVertices,
    index_chunks: HashMap<ResourceId, IndexChunk>,
    triangulator: Triangulator<MapVertexIndex>,
    triangulation: Triangulation<FixedVertex, MapVertexIndex>,
}

impl Default for ChunkBuilder {
    fn default() -> Self {
        let mut triangulator = Triangulator::new(
            Self::TRIANGULATOR_BASE_CAPACITY,
            Validation::default(),
            Solver::default(),
        );
        triangulator.delaunay(true);
        triangulator.earcut(true);

        let triangulation = Triangulation::with_capacity(Self::TRIANGULATOR_BASE_CAPACITY);

        Self {
            vertices: MapVertices::default(),
            index_chunks: HashMap::default(),
            triangulator,
            triangulation,
        }
    }
}

impl ChunkBuilder {
    // TODO: increase this if necessary
    const TRIANGULATOR_BASE_CAPACITY: usize = 256;

    fn add_vertical_quad(
        &mut self,
        a: glm::TVec3<f32>,
        b: glm::TVec3<f32>,
        linedef: &Linedef,
        sidedef: &Sidedef,
        location: SidedefLocation,
    ) {
        let texture = sidedef.texture(location);
        if let Some(texture) = texture {
            let light = sidedef.light() as f32 / 255.0;
            let tag = sidedef.sector.tag;

            let uv_offset = sidedef.offset(location);
            let uv_scale = sidedef.scale(location);

            let tex_size = {
                let texture = texture.read();
                glm::vec2(texture.size.x as f32, texture.size.y as f32)
            };

            // Normalize the offset
            let uv_offset = glm::vec2(
                uv_offset.x / tex_size.x,
                uv_offset.y / tex_size.y,
            );

            // Normalize the size
            let uv_size = glm::vec2(
                glm::distance(&a.xy(), &b.xy()) / (tex_size.x * uv_scale.x),
                (b.z - a.z).abs() / (tex_size.y * uv_scale.y),
            );

            let pin_top = match location {
                SidedefLocation::Upper => linedef.flags.contains(LinedefFlags::DontPegTop),
                SidedefLocation::Middle | SidedefLocation::Lower =>
                    !linedef.flags.contains(LinedefFlags::DontPegBottom),
            };

            let (uv_a, uv_b) = if pin_top {
                // Pinned to top of quad
                (
                    glm::vec2(0.0 - uv_offset.x, uv_size.y + uv_offset.y),
                    glm::vec2(uv_size.x - uv_offset.x, 0.0 + uv_offset.y),
                )
            } else {
                // Pinned to bottom of quad
                (
                    glm::vec2(0.0 - uv_offset.x, 1.0 + uv_offset.y),
                    glm::vec2(uv_size.x - uv_offset.x, (1.0 - uv_size.y) + uv_offset.y),
                )
            };

            // c - b
            // | / |
            // a - d
            let i_a = self.vertices.get_or_insert(MapVertex {
                position: a,
                uv: uv_a,
                light,
                tag,
            });
            let i_b = self.vertices.get_or_insert(MapVertex {
                position: b,
                uv: uv_b,
                light,
                tag,
            });
            let i_c = self.vertices.get_or_insert(MapVertex {
                position: glm::vec3(a.x, a.y, b.z),
                uv: glm::vec2(uv_a.x, uv_b.y),
                light,
                tag,
            });
            let i_d = self.vertices.get_or_insert(MapVertex {
                position: glm::vec3(b.x, b.y, a.z),
                uv: glm::vec2(uv_b.x, uv_a.y),
                light,
                tag,
            });

            let texture = self.index_chunks.entry(texture.id().clone())
                .or_insert_with(|| IndexChunk {
                    texture: texture.clone(),
                    indices: Vec::new(),
                });
            texture.indices.extend_from_slice(&[i_a, i_b, i_c, i_a, i_d, i_b]);
        } else {
            warn!(?a, ?b, "Missing texture for quad");
        }
    }

    pub fn add_linedef(&mut self, linedef: &Linedef) {
        match (&linedef.front_sidedef, &linedef.back_sidedef) {
            (Some(front), Some(back)) => {
                // Lower
                if front.sector.floor_height < back.sector.floor_height {
                    self.add_vertical_quad(
                        linedef.start_vertex.to_3d(front.sector.floor_height as f32),
                        linedef.end_vertex.to_3d(back.sector.floor_height as f32),
                        linedef,
                        front,
                        SidedefLocation::Lower,
                    );
                } else if front.sector.floor_height > back.sector.floor_height {
                    self.add_vertical_quad(
                        linedef.end_vertex.to_3d(back.sector.floor_height as f32),
                        linedef.start_vertex.to_3d(front.sector.floor_height as f32),
                        linedef,
                        back,
                        SidedefLocation::Lower,
                    );
                }

                // Upper
                if front.sector.ceil_height > back.sector.ceil_height {
                    self.add_vertical_quad(
                        linedef.start_vertex.to_3d(back.sector.ceil_height as f32),
                        linedef.end_vertex.to_3d(front.sector.ceil_height as f32),
                        linedef,
                        front,
                        SidedefLocation::Upper,
                    );
                } else if front.sector.ceil_height < back.sector.ceil_height {
                    self.add_vertical_quad(
                        linedef.end_vertex.to_3d(front.sector.ceil_height as f32),
                        linedef.start_vertex.to_3d(back.sector.ceil_height as f32),
                        linedef,
                        back,
                        SidedefLocation::Upper,
                    );
                }

                // TODO: Add double-sided middle textures
            },
            (Some(front), None) => {
                // Middle
                self.add_vertical_quad(
                    linedef.start_vertex.to_3d(front.sector.floor_height as f32),
                    linedef.end_vertex.to_3d(front.sector.ceil_height as f32),
                    linedef,
                    front,
                    SidedefLocation::Middle,
                );
            },
            (None, Some(back)) => {
                // Middle
                self.add_vertical_quad(
                    linedef.end_vertex.to_3d(back.sector.floor_height as f32),
                    linedef.start_vertex.to_3d(back.sector.ceil_height as f32),
                    linedef,
                    back,
                    SidedefLocation::Middle,
                );
            },
            _ => {},
        }
    }

    pub fn add_ssector(&mut self, ssector: &GlSSector, gl_vertices: &GlVerticesReadLock) {
        let light = ssector.sector.light_level as f32 / 255.0;
        let tag = ssector.sector.tag;

        let points = ssector.contour(gl_vertices)
            .map(|indexed| indexed.to_fixed().into_inner())
            .collect::<Vec<_>>();
        self.triangulator.triangulate_into(&points, &mut self.triangulation);

        let (floor_indices, ceil_indices): (Vec<_>, Vec<_>) = self.triangulation.points.iter()
            .map(|v| {
                // Floor/Ceil flat texture UVs are based off of a global grid of 64x64 flats
                let uv = glm::vec2(f32::lossy_from(v.x) / 64.0, f32::lossy_from(v.y) / 64.0);
                let floor_idx = self.vertices.get_or_insert(MapVertex {
                    position: v.to_3d(ssector.sector.floor_height as f32),
                    uv, light, tag,
                });
                let ceil_idx = self.vertices.get_or_insert(MapVertex {
                    position: v.to_3d(ssector.sector.ceil_height as f32),
                    uv, light, tag,
                });
                (floor_idx, ceil_idx)
            })
            .unzip();

        {
            let texture = self.index_chunks.entry(ssector.sector.floor_tex.id().clone())
                .or_insert_with(|| IndexChunk {
                    texture: ssector.sector.floor_tex.clone(),
                    indices: Vec::new(),
                });
            for idx in self.triangulation.indices.iter() {
                texture.indices.push(floor_indices[*idx as usize]);
            }
        }

        {
            let texture = self.index_chunks.entry(ssector.sector.ceil_tex.id().clone())
                .or_insert_with(|| IndexChunk {
                    texture: ssector.sector.ceil_tex.clone(),
                    indices: Vec::new(),
                });
            // Ceiling textures need reversed indices because the triangles face the
            // opposite direction
            for idx in self.triangulation.indices.iter().rev() {
                texture.indices.push(ceil_indices[*idx as usize]);
            }
        }
    }

    pub fn build(
        self,
        model_data: &mut MapModelData,
    ) -> Result<Chunk, ChunkBuildError> {
        debug!(
            vertices = ?self.vertices.len(),
            "Building chunk"
        );

        let index_remap = model_data.vertices.extend(self.vertices);

        let mut index_segments = Vec::with_capacity(self.index_chunks.len());
        for chunk_tex in self.index_chunks.values() {
            let texture_id = model_data.textures.get_or_insert(&chunk_tex.texture);
            let first_index = model_data.vertex_indices.len() as MapVertexIndex;
            let index_count = chunk_tex.indices.len() as u32;
            model_data.vertex_indices.extend(chunk_tex.indices.iter()
                .map(|idx| index_remap[(*idx) as usize]));
            index_segments.push(ChunkIndexSegment {
                texture_id,
                first_index,
                index_count,
            });
        }

        Ok(Chunk {
            index_segments,
        })
    }
}

struct ChunkIndexSegment {
    texture_id: u32,
    first_index: MapVertexIndex,
    index_count: u32,
}

impl ChunkIndexSegment {
    fn to_pc(&self) -> shaders::frag::PCData {
        shaders::frag::PCData {
            tex_id: self.texture_id,
        }
    }
}

pub struct Chunk {
    index_segments: Vec<ChunkIndexSegment>,
}

impl Chunk {
    #[inline]
    pub fn builder() -> ChunkBuilder {
        ChunkBuilder::default()
    }

    pub fn draw(
        &self,
        graphics: &Arc<GraphicsPipeline>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        for segment in &self.index_segments {
            builder
                .push_constants(graphics.layout().clone(), 0, segment.to_pc()).unwrap()
                .draw_indexed(segment.index_count, 1, segment.first_index, 0, 0).unwrap();
        }
    }
}

mod shaders {
    pub mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 460

                layout(location = 0) in vec3 position;
                layout(location = 1) in vec2 uv;
                layout(location = 2) in float light;
                layout(location = 3) in uint tag;

                layout(location = 0) out vec2 tex_coords;
                layout(location = 1) out float out_light;
                layout(location = 2) out float testval;

                layout(set = 0, binding = 0) uniform VP_Data {
                    mat4 vp;
                } vp;

                void main() {
                    gl_Position = vp.vp * vec4(position, 1.0);
                    tex_coords = uv;
                    out_light = light;
                    testval = gl_Position.z;
                }
            ",
        }
    }

    pub mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 460
                #extension GL_EXT_nonuniform_qualifier : require

                layout(location = 0) in vec2 tex_coords;
                layout(location = 1) in float light;
                layout(location = 2) in float testval;
                layout(location = 0) out vec4 f_color;

                layout(push_constant) uniform PCData {
                    uint tex_id;
                } pc;

                layout(set = 1, binding = 0) uniform sampler s;
                layout(set = 1, binding = 1) uniform texture2D tex[];

                void main() {
                    vec4 color = texture(sampler2D(tex[nonuniformEXT(pc.tex_id)], s), tex_coords);
                    f_color = vec4(color.xyz * light, color.w);
                    //f_color = vec4(0.0, 0.0, testval, 1.0);
                }
            ",
        }
    }
}

pub struct MapRenderer {
    chunk: Arc<Chunk>,
    graphics: Arc<EngineGraphicsPipeline>,
    descriptor_set_0: EngineDescriptorSet,
    descriptor_set_1: EngineDescriptorSet,
    vertex_buffer: Subbuffer<[MapVertex]>,
    index_buffer: IndexBuffer,
}

impl MapRenderer {
    pub fn new(
        camera: Arc<Uniform<VP, Camera>>,
        chunk: Arc<Chunk>,
        model_data: MapModelData,
        renderer: &Arc<Renderer>,
        subpass: &Subpass,
    ) -> Result<Self, VulkanoError> {
        let vert = shaders::vert::load(renderer.device().vk_device().clone())
            .expect("Failed to create vertex shader module")
            .entry_point("main").unwrap();
        let frag = shaders::frag::load(renderer.device().vk_device().clone())
            .expect("Failed to create fragment shader module")
            .entry_point("main").unwrap();

        let vertex_input_state = Some(MapVertex::per_vertex()
            .definition(&vert.info().input_interface)
            .unwrap());

        let stages = [
            PipelineShaderStageCreateInfo::new(vert),
            PipelineShaderStageCreateInfo::new(frag),
        ];

        let pipeline_layout = {
            let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            // Set the ImageView count for the runtime descriptor array
            layout_create_info.set_layouts[1].bindings
                .get_mut(&1).unwrap()
                .descriptor_count = model_data.textures.len() as u32;
            PipelineLayout::new(
                renderer.device().vk_device().clone(),
                layout_create_info
                    .into_pipeline_layout_create_info(renderer.device().vk_device().clone()).unwrap()
            ).map_err(Validated::unwrap)?
        };
        let descriptor_layout_0 = pipeline_layout.set_layouts().get(0).unwrap().clone();
        let descriptor_layout_1 = pipeline_layout.set_layouts().get(1).unwrap().clone();

        let create_info = GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state,
            subpass: Some(subpass.clone().into()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
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
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        };

        let graphics = EngineGraphicsPipeline::new(
            renderer,
            create_info,
            ViewportType::BottomLeft,
        );

        let sampler = Arc::new(EngineSampler::new(Sampler::new(
            renderer.device().vk_device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Linear,
                address_mode: [
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::ClampToEdge,
                ],
                ..Default::default()
            }
        ).map_err(Validated::unwrap)?));

        let descriptor_set_0 = EngineDescriptorSet::builder(renderer.clone())
            .layout(descriptor_layout_0.clone())
            .descriptor_source(0, camera)
            .build();

        let descriptor_set_1 = EngineDescriptorSet::builder(renderer.clone())
            .layout(descriptor_layout_1.clone())
            .descriptor_source(0, sampler)
            .descriptor_source(1, model_data.textures.into_descriptor_images())
            .build();

        let vertex_buffer = model_data.vertices.into_vertex_buffer(renderer.device())?;
        let index_buffer = model_data.vertex_indices.into_index_buffer(renderer.device())?;

        Ok(Self {
            chunk,
            graphics,
            descriptor_set_0,
            descriptor_set_1,
            vertex_buffer,
            index_buffer,
        })
    }
}

impl EngineRenderHandler for MapRenderer {
    fn build_commands(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), VulkanoError> {
        let graphics = self.graphics.vk_graphics();

        builder
            .bind_pipeline_graphics(graphics.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                0,
                self.descriptor_set_0.descriptor_set().map_err(Validated::unwrap)?,
            ).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                1,
                self.descriptor_set_1.descriptor_set().map_err(Validated::unwrap)?,
            ).unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
            .bind_index_buffer(self.index_buffer.clone()).unwrap();
        self.chunk.draw(&graphics, builder);

        Ok(())
    }
}