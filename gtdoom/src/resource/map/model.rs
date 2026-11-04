use ahash::HashMap;
use fixed::traits::LossyFrom;
use gtether::render::descriptor_set::EngineDescriptorSet;
use gtether::render::image::{EngineResourceImages, EngineSampler};
use gtether::render::pipeline::{EngineGraphicsPipeline, VKGraphicsPipelineSource, ViewportType};
use gtether::render::render_pass::EngineRenderHandler;
use gtether::render::uniform::Uniform;
use gtether::render::{EngineDevice, Renderer, VulkanoError};
use gtether::resource::manager::ResourceManager;
use gtether::resource::{Resource, ResourceLoadError, ResourceLoadResult};
use i_triangle::float::triangulation::Triangulation;
use i_triangle::float::triangulator::Triangulator;
use i_triangle::i_overlay::core::solver::Solver;
use i_triangle::int::validation::Validation;
use parking_lot::RwLock;
use smol::future::FutureExt;
use std::fmt::{Debug, Formatter};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tracing::trace;
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
use vulkano::pipeline::{Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::Validated;

use crate::client::DebugConfig;
use crate::geometry::line::Line;
use crate::render::camera::{Camera, VP};
use crate::resource::map::blockmap::Blockmap;
use crate::resource::map::linedef::{Linedef, LinedefFlags};
use crate::resource::map::sector::Sector;
use crate::resource::map::sidedef::{Sidedef, SidedefLocation};
use crate::resource::map::vertex::{FixedFloat, FixedVertex};
use crate::resource::texture::{Texture, TextureId};

const MAP_VERTEX_EPSILON: f32 = 0.1;
pub const RENDER_CHUNK_SIZE: u32 = 256;
pub type MapVertexIndex = u32;
pub type MapTextureIndex = u16;

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
    #[format(R16_UINT)]
    tex_id: u16,
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
        self.tag == other.tag && self.tex_id == other.tex_id
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ChunkBuildError {
    #[error(transparent)]
    AllocateBufferError(#[from] AllocateBufferError),
    #[error(transparent)]
    ResourceLoadError(#[from] ResourceLoadError),
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

    pub fn adjust_texture_indices(&mut self, remap: &Vec<MapTextureIndex>) {
        for vertex in &mut self.0 {
            vertex.tex_id = remap[vertex.tex_id as usize];
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

enum TextureLoadMapStatus {
    Loading(Pin<Box<dyn Future<Output=ResourceLoadResult<Texture>>>>),
    Loaded(ResourceLoadResult<Texture>),
}

struct TextureLoadMapEntry {
    id: TextureId,
    status: TextureLoadMapStatus,
}

// TODO: split texture and texture metadata (e.g. size) loading when upstream gTether supports
//  keying cached resources by type and/or loader
struct TextureLoadMap {
    resources: Arc<ResourceManager>,
    renderer: Arc<Renderer>,
    entries: Vec<TextureLoadMapEntry>,
    index_map: HashMap<TextureId, MapTextureIndex>,
}

impl TextureLoadMap {
    fn new(
        resources: Arc<ResourceManager>,
        renderer: Arc<Renderer>,
    ) -> Self {
        Self {
            resources,
            renderer,
            entries: Vec::default(),
            index_map: HashMap::default(),
        }
    }

    fn register(&mut self, texture_id: TextureId) {
        if !self.index_map.contains_key(&texture_id) {
            let fut = match texture_id {
                TextureId::Texture(id) => {
                    // Prefer wall textures first
                    self.resources.get_with_loader(
                        PathBuf::from("Textures").join(id.as_str()),
                        Texture::loader(self.renderer.clone()),
                    ).or_get_with_loader(
                        PathBuf::from("Flats").join(id.as_str()),
                        Texture::flat_loader(self.renderer.clone()),
                    ).boxed()
                },
                TextureId::Flat(id) => {
                    // Prefer flats first
                    self.resources.get_with_loader(
                        PathBuf::from("Flats").join(id.as_str()),
                        Texture::flat_loader(self.renderer.clone()),
                    ).or_get_with_loader(
                        PathBuf::from("Textures").join(id.as_str()),
                        Texture::loader(self.renderer.clone()),
                    ).boxed()
                },
            };

            let idx = self.entries.len() as MapTextureIndex;
            self.entries.push(TextureLoadMapEntry {
                id: texture_id,
                status: TextureLoadMapStatus::Loading(fut),
            });
            self.index_map.insert(texture_id, idx);
        }
    }

    fn get_index(
        &mut self,
        texture_id: TextureId,
    ) -> MapTextureIndex {
        self.register(texture_id);
        self.index_map[&texture_id]
    }

    async fn get(
        &mut self,
        texture_id: TextureId,
    ) -> Result<(MapTextureIndex, Arc<Resource<Texture>>), ResourceLoadError> {
        let idx = self.get_index(texture_id);
        let entry = &mut self.entries[idx as usize];
        let result = match &mut entry.status {
            TextureLoadMapStatus::Loading(fut) => {
                let result = fut.await;
                entry.status = TextureLoadMapStatus::Loaded(result.clone());
                result
            },
            TextureLoadMapStatus::Loaded(result) => result.clone(),
        };
        result.map(|tex| (idx, tex))
    }
}

#[derive(Default, Clone)]
pub struct MapTextures {
    textures: Vec<Arc<Resource<Texture>>>,
    index_map: HashMap<TextureId, MapTextureIndex>,
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
    pub fn get_index(&self, texture_id: TextureId) -> Option<MapTextureIndex> {
        self.index_map.get(&texture_id).cloned()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.textures.len()
    }

    async fn extend_from_load_map(
        &mut self,
        load_map: TextureLoadMap,
    ) -> Result<Vec<MapTextureIndex>, ResourceLoadError> {
        let mut remap = vec![];
        for entry in load_map.entries {
            if let Some(idx) = self.index_map.get(&entry.id) {
                remap.push(*idx);
            } else {
                let idx = self.textures.len() as MapTextureIndex;
                let texture = match entry.status {
                    TextureLoadMapStatus::Loading(fut) => fut.await?,
                    TextureLoadMapStatus::Loaded(result) => result?,
                };
                self.textures.push(texture);
                self.index_map.insert(entry.id, idx);
                remap.push(idx);
            }
        }
        Ok(remap)
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

pub struct ChunkBuilder {
    vertices: MapVertices,
    textures: TextureLoadMap,
    indices: Vec<MapVertexIndex>,
    triangulator: Triangulator<MapVertexIndex>,
    triangulation: Triangulation<FixedVertex, MapVertexIndex>,
}

impl ChunkBuilder {
    // TODO: increase this if necessary
    const TRIANGULATOR_BASE_CAPACITY: usize = 256;

    fn new(
        resources: Arc<ResourceManager>,
        renderer: Arc<Renderer>,
    ) -> Self {
        let textures = TextureLoadMap::new(resources, renderer);
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
            textures,
            indices: vec![],
            triangulator,
            triangulation,
        }
    }

    async fn add_vertical_quad(
        &mut self,
        a: glm::TVec3<f32>,
        b: glm::TVec3<f32>,
        linedef: &Linedef,
        sidedef: &Sidedef,
        location: SidedefLocation,
    ) -> Result<(), ResourceLoadError> {
        let texture_id = sidedef.texture_id(location);
        if let Some(texture_id) = texture_id {
            let light = sidedef.light() as f32 / 255.0;
            let tag = sidedef.sector.tag;

            let uv_offset = sidedef.offset(location);
            let uv_scale = sidedef.scale(location);

            let (tex_id, tex_size) = {
                // TODO: load textures in the background
                let (tex_id, texture) = self.textures.get(texture_id).await?;
                let tex_lock = texture.read();
                (tex_id, glm::vec2(tex_lock.size.x as f32, tex_lock.size.y as f32))
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
                tex_id,
            });
            let i_b = self.vertices.get_or_insert(MapVertex {
                position: b,
                uv: uv_b,
                light,
                tag,
                tex_id,
            });
            let i_c = self.vertices.get_or_insert(MapVertex {
                position: glm::vec3(a.x, a.y, b.z),
                uv: glm::vec2(uv_a.x, uv_b.y),
                light,
                tag,
                tex_id,
            });
            let i_d = self.vertices.get_or_insert(MapVertex {
                position: glm::vec3(b.x, b.y, a.z),
                uv: glm::vec2(uv_b.x, uv_a.y),
                light,
                tag,
                tex_id,
            });

            self.indices.extend([i_a, i_b, i_c, i_a, i_d, i_b]);
            Ok(())
        } else {
            Err(ResourceLoadError::from(format!("Missing texture for quad (a={a:?}, b={b:?})")))
        }
    }

    pub async fn add_linedef(
        &mut self,
        linedef: &Linedef,
        segment: Line<FixedFloat>,
    ) -> Result<(), ResourceLoadError> {
        let start_vertex = FixedVertex::from(segment.a);
        let end_vertex = FixedVertex::from(segment.b);
        match (&linedef.front_sidedef, &linedef.back_sidedef) {
            (Some(front), Some(back)) => {
                // Lower
                if front.sector.floor_height < back.sector.floor_height {
                    self.add_vertical_quad(
                        start_vertex.to_3d(front.sector.floor_height as f32),
                        end_vertex.to_3d(back.sector.floor_height as f32),
                        linedef,
                        front,
                        SidedefLocation::Lower,
                    ).await?;
                } else if front.sector.floor_height > back.sector.floor_height {
                    self.add_vertical_quad(
                        end_vertex.to_3d(back.sector.floor_height as f32),
                        start_vertex.to_3d(front.sector.floor_height as f32),
                        linedef,
                        back,
                        SidedefLocation::Lower,
                    ).await?;
                }

                // Upper
                if front.sector.ceil_height > back.sector.ceil_height {
                    self.add_vertical_quad(
                        start_vertex.to_3d(back.sector.ceil_height as f32),
                        end_vertex.to_3d(front.sector.ceil_height as f32),
                        linedef,
                        front,
                        SidedefLocation::Upper,
                    ).await?;
                } else if front.sector.ceil_height < back.sector.ceil_height {
                    self.add_vertical_quad(
                        end_vertex.to_3d(front.sector.ceil_height as f32),
                        start_vertex.to_3d(back.sector.ceil_height as f32),
                        linedef,
                        back,
                        SidedefLocation::Upper,
                    ).await?;
                }

                // TODO: Add double-sided middle textures
            },
            (Some(front), None) => {
                // Middle
                self.add_vertical_quad(
                    start_vertex.to_3d(front.sector.floor_height as f32),
                    end_vertex.to_3d(front.sector.ceil_height as f32),
                    linedef,
                    front,
                    SidedefLocation::Middle,
                ).await?;
            },
            (None, Some(back)) => {
                // Middle
                self.add_vertical_quad(
                    end_vertex.to_3d(back.sector.floor_height as f32),
                    start_vertex.to_3d(back.sector.ceil_height as f32),
                    linedef,
                    back,
                    SidedefLocation::Middle,
                ).await?;
            },
            _ => {},
        }
        Ok(())
    }

    pub fn add_ssector(&mut self, sector: &Sector, contour: Vec<FixedVertex>) {
        let light = sector.light_level as f32 / 255.0;
        let tag = sector.tag;

        let floor_tex_id = self.textures.get_index(sector.floor_tex);
        let ceil_tex_id = self.textures.get_index(sector.ceil_tex);

        self.triangulator.triangulate_into(&contour, &mut self.triangulation);

        let (floor_indices, ceil_indices): (Vec<_>, Vec<_>) = self.triangulation.points.iter()
            .map(|v| {
                // Floor/Ceil flat texture UVs are based off of a global grid of 64x64 flats
                let uv = glm::vec2(f32::lossy_from(v.x) / 64.0, f32::lossy_from(v.y) / 64.0);
                let floor_idx = self.vertices.get_or_insert(MapVertex {
                    position: v.to_3d(sector.floor_height as f32),
                    uv, light, tag,
                    tex_id: floor_tex_id,
                });
                let ceil_idx = self.vertices.get_or_insert(MapVertex {
                    position: v.to_3d(sector.ceil_height as f32),
                    uv, light, tag,
                    tex_id: ceil_tex_id,
                });
                (floor_idx, ceil_idx)
            })
            .unzip();

        for idx in self.triangulation.indices.iter() {
            self.indices.push(floor_indices[*idx as usize]);
        }
        // Ceiling textures need reversed indices because the triangles face the
        // opposite direction
        for idx in self.triangulation.indices.iter().rev() {
            self.indices.push(ceil_indices[*idx as usize]);
        }
    }

    pub async fn build(
        self,
        model_data: &mut MapModelData,
    ) -> Result<Chunk, ChunkBuildError> {
        trace!(
            vertices = ?self.vertices.len(),
            "Building chunk"
        );

        let tex_index_remap = model_data.textures.extend_from_load_map(self.textures).await?;
        let mut vertices = self.vertices;
        vertices.adjust_texture_indices(&tex_index_remap);
        let vertex_index_remap = model_data.vertices.extend(vertices);

        let first_index = model_data.vertex_indices.len() as MapVertexIndex;
        let index_count = self.indices.len() as u32;
        model_data.vertex_indices.extend(
            self.indices.into_iter()
                .map(|idx| vertex_index_remap[idx as usize])
        );

        Ok(Chunk {
            first_index,
            index_count,
        })
    }
}

pub struct Chunk {
    first_index: MapVertexIndex,
    index_count: u32,
}

impl Chunk {
    #[inline]
    pub fn builder(
        resources: Arc<ResourceManager>,
        renderer: Arc<Renderer>,
    ) -> ChunkBuilder {
        ChunkBuilder::new(resources, renderer)
    }

    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        if self.index_count > 0 {
            builder
                .draw_indexed(
                    self.index_count,
                    1,
                    self.first_index,
                    0,
                    0,
                ).unwrap();
        }
    }
}

mod shaders {
    pub mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/shaders/map.vert",
        }
    }

    pub mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/map.frag",
        }
    }

    pub mod frag_wireframe {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/map.frag",
            define: [("WIREFRAME", "1")],
        }
    }
}

struct MapRendererGraphicsPipeline {
    graphics: Arc<EngineGraphicsPipeline>,
    descriptor_set_0: EngineDescriptorSet,
    descriptor_set_1: EngineDescriptorSet,
}

impl MapRendererGraphicsPipeline {
    fn new(
        wireframe: bool,
        camera: Arc<Uniform<VP, Camera>>,
        textures: Arc<EngineResourceImages<Texture>>,
        textures_count: u32,
        renderer: &Arc<Renderer>,
        subpass: &Subpass,
    ) -> Result<Self, VulkanoError> {
        let vert = shaders::vert::load(renderer.device().vk_device().clone())
            .expect("Failed to create vertex shader module")
            .entry_point("main").unwrap();
        let frag = match wireframe {
            false => shaders::frag::load(renderer.device().vk_device().clone())
                .expect("Failed to create fragment shader module")
                .entry_point("main").unwrap(),
            true => shaders::frag_wireframe::load(renderer.device().vk_device().clone())
                .expect("Failed to create fragment shader module")
                .entry_point("main").unwrap(),
        };

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
                .descriptor_count = textures_count;
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
            .descriptor_source(1, textures)
            .build();

        Ok(Self {
            graphics,
            descriptor_set_0,
            descriptor_set_1,
        })
    }
}

pub struct MapRenderer {
    debug_config: Arc<RwLock<DebugConfig>>,
    chunks: Arc<Blockmap<RENDER_CHUNK_SIZE, Chunk>>,
    vertex_buffer: Subbuffer<[MapVertex]>,
    index_buffer: IndexBuffer,
    graphics: MapRendererGraphicsPipeline,
    // TODO: Not super efficient to create both immediately;
    //  defer this so its only created when needed
    graphics_wireframe: MapRendererGraphicsPipeline,
}

impl MapRenderer {
    pub fn new(
        debug_config: Arc<RwLock<DebugConfig>>,
        camera: Arc<Uniform<VP, Camera>>,
        chunks: Arc<Blockmap<RENDER_CHUNK_SIZE, Chunk>>,
        model_data: MapModelData,
        renderer: &Arc<Renderer>,
        subpass: &Subpass,
    ) -> Result<Self, VulkanoError> {
        let vertex_buffer = model_data.vertices.into_vertex_buffer(renderer.device())?;
        let index_buffer = model_data.vertex_indices.into_index_buffer(renderer.device())?;

        let textures_count = model_data.textures.len() as u32;
        let textures = model_data.textures.into_descriptor_images();
        let graphics = MapRendererGraphicsPipeline::new(
            false,
            camera.clone(),
            textures.clone(),
            textures_count,
            renderer,
            subpass,
        )?;
        let graphics_wireframe = MapRendererGraphicsPipeline::new(
            true,
            camera,
            textures,
            textures_count,
            renderer,
            subpass,
        )?;

        Ok(Self {
            debug_config,
            chunks,
            vertex_buffer,
            index_buffer,
            graphics,
            graphics_wireframe,
        })
    }
}

impl EngineRenderHandler for MapRenderer {
    fn build_commands(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), VulkanoError> {
        let pipeline = match self.debug_config.read().wireframe {
            false => &self.graphics,
            true => &self.graphics_wireframe,
        };
        let graphics = pipeline.graphics.vk_graphics();

        builder
            .bind_pipeline_graphics(graphics.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                0,
                pipeline.descriptor_set_0.descriptor_set().map_err(Validated::unwrap)?,
            ).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                1,
                pipeline.descriptor_set_1.descriptor_set().map_err(Validated::unwrap)?,
            ).unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
            .bind_index_buffer(self.index_buffer.clone()).unwrap();
        for chunk in self.chunks.values() {
            chunk.draw(builder);
        }

        Ok(())
    }
}