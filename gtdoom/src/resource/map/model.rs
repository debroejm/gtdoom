use ahash::HashMap;
use gtether::render::descriptor_set::EngineDescriptorSet;
use gtether::render::image::{EngineResourceImages, EngineSampler};
use gtether::render::pipeline::{EngineGraphicsPipeline, VKGraphicsPipelineSource, ViewportType};
use gtether::render::render_pass::EngineRenderHandler;
use gtether::render::uniform::Uniform;
use gtether::render::{EngineDevice, Renderer, VulkanoError};
use gtether::resource::id::ResourceId;
use gtether::resource::Resource;
use i_triangle::float::triangulatable::Triangulatable;
use i_triangle::i_overlay::i_shape::base::data::{Contour, Shape as ContourShape};
use parry2d::na::{Isometry2, Point2};
use parry2d::query::PointQuery;
use parry2d::shape::{Compound, ConvexPolygon, SharedShape};
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
use crate::resource::map::linedef::{Linedef, LinedefFlags};
use crate::resource::map::sector::Sector;
use crate::resource::map::sidedef::{Sidedef, SidedefLocation};
use crate::resource::map::vertex::FloatVertex;
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
        let uv = glm::equal_eps(&self.uv, &other.uv, epsilon);
        if !(uv.x && uv.y) {
            return false
        }
        if !((self.light - other.light).abs() < epsilon) {
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

struct NestedCompound {
    contour: Contour<FloatVertex>,
    shape: Compound,
    sub_shapes: Vec<Box<NestedCompound>>,
}

impl NestedCompound {
    fn contour_angle(contour: &Contour<FloatVertex>) -> f32 {
        if contour.len() < 2 {
            return 0.0
        }

        let mut iter = contour.iter();
        let first = iter.next().unwrap();
        let second = iter.next().unwrap();

        let mut total_angle = 0.0;

        let mut prev = second;
        let mut prev_angle = (second.y - first.y).atan2(second.x - first.x);
        for next in iter.chain([first, second]) {
            let next_angle = (next.y - prev.y).atan2(next.x - prev.x);
            let mut diff = prev_angle - next_angle;
            if diff < -std::f32::consts::PI {
                diff += 2.0 * std::f32::consts::PI;
            } else if diff > std::f32::consts::PI {
                diff -= 2.0 * std::f32::consts::PI;
            }
            total_angle += diff;
            prev = next;
            prev_angle = next_angle;
        }

        total_angle
    }

    fn contour_to_complex_shape(contour: &mut Contour<FloatVertex>) -> Compound {
        if Self::contour_angle(contour) > 0.0 {
            // Is CW, needs to be CCW
            contour.reverse();
        }

        let shapes = contour.triangulate().into_delaunay().to_convex_polygons().into_iter()
            .map(|contour| {
                let convex = ConvexPolygon::from_convex_polyline(contour.into_iter()
                    .map(|v| Point2::new(v.x, v.y))
                    .collect()).unwrap();
                (Isometry2::identity(), SharedShape::new(convex))
            })
            .collect::<Vec<_>>();

        Compound::new(shapes)
    }

    fn new(mut contour: Contour<FloatVertex>) -> Self {
        let shape = Self::contour_to_complex_shape(&mut contour);
        Self {
            contour,
            shape,
            sub_shapes: Vec::new(),
        }
    }

    fn contains(&self, other: &NestedCompound) -> bool {
        // Sub-Sector shapes from Doom maps should NOT intersect, so if any point is inside, we can
        // assume the whole shape is inside
        let p = &other.shape.shapes()[0].1
            .as_convex_polygon().unwrap()
            .points()[0];
        self.shape.contains_local_point(p)
    }

    fn insert_sub_shape(&mut self, mut new_sub_shape: NestedCompound) {
        let sub_sub_shapes = self.sub_shapes
            .extract_if(.., |ss| new_sub_shape.contains(ss))
            .collect::<Vec<_>>();
        new_sub_shape.sub_shapes = sub_sub_shapes;

        for ss in &mut self.sub_shapes {
            if ss.contains(&new_sub_shape) {
                ss.insert_sub_shape(new_sub_shape);
                return
            }
        }

        self.sub_shapes.push(Box::new(new_sub_shape));
    }

    fn into_contour_shapes(self, output: &mut Vec<ContourShape<FloatVertex>>) {
        let mut contour_shape = ContourShape::with_capacity(1 + self.sub_shapes.len());
        contour_shape.push(self.contour);
        for mut hole in self.sub_shapes {
            hole.contour.reverse();
            contour_shape.push(hole.contour);
            for sub_shape in hole.sub_shapes {
                sub_shape.into_contour_shapes(output);
            }
        }
        output.push(contour_shape);
    }
}

struct SectorBuilder {
    sector: Arc<Sector>,
    open_contours: Vec<Contour<FloatVertex>>,
    closed_shapes: Vec<NestedCompound>,
}

impl SectorBuilder {
    fn new(sector: Arc<Sector>) -> Self {
        Self {
            sector,
            open_contours: Vec::new(),
            closed_shapes: Vec::new(),
        }
    }

    fn add_closed_contour(
        closed_shapes: &mut Vec<NestedCompound>,
        closed_contour: Contour<FloatVertex>,
    ) {
        let new_shape = NestedCompound::new(closed_contour);
        for closed_shape in closed_shapes.iter_mut() {
            if closed_shape.contains(&new_shape) {
                closed_shape.insert_sub_shape(new_shape);
                return
            }
        }
        closed_shapes.push(new_shape);
    }

    fn find_matching_indices(
        a: &Contour<FloatVertex>,
        b: &Contour<FloatVertex>,
    ) -> Option<(usize, usize)> {
        for i_b in 0..b.len() {
            for i_a in (0..a.len()).rev() {
                if a[i_a].equal_eps(&b[i_b], MAP_VERTEX_EPSILON) {
                    return Some((i_a, i_b))
                }
            }
        }
        None
    }

    fn add_line(&mut self, a: FloatVertex, b: FloatVertex) {
        // 1. Look for a prefix match first
        let mut new_segment = vec![b];
        for idx in 0..self.open_contours.len() {
            if let Some(first) = self.open_contours[idx].first() {
                if b.equal_eps(first, 0.01) {
                    let contour = self.open_contours.swap_remove(idx);
                    if a.equal_eps(contour.last().unwrap(), 0.01) {
                        // Closing this contour
                        Self::add_closed_contour(&mut self.closed_shapes, contour);
                        return
                    }
                    new_segment = contour;
                    break
                }
            }
        }

        // 2. Look for a suffix match to attach the new segment to
        for contour in &mut self.open_contours {
            if let Some(last) = contour.last() && last.equal_eps(&a, MAP_VERTEX_EPSILON) {
                // Find all closed contours created with the new segment
                while let Some((i_a, i_b)) = Self::find_matching_indices(contour, &new_segment) {
                    let mut closed_contour = contour.split_off(i_a);
                    let extra = new_segment.split_off(i_b);
                    closed_contour.extend(new_segment);
                    new_segment = extra;
                    Self::add_closed_contour(&mut self.closed_shapes, closed_contour);
                }

                // Add the remaining new segment to the contour
                contour.extend(new_segment);

                // Found a match, done processing
                return
            }
        }

        // 3. No suffix matches, add a new contour
        new_segment.insert(0, a);
        self.open_contours.push(new_segment);
    }

    fn add_linedef(&mut self, linedef: &Linedef) {
        let front_idx = linedef.front_sidedef.as_ref().map(|front| front.sector.idx);
        let back_idx = linedef.back_sidedef.as_ref().map(|back| back.sector.idx);
        match (front_idx, back_idx) {
            (Some(front_idx), Some(back_idx)) => {
                if front_idx == self.sector.idx {
                    if back_idx == self.sector.idx {
                        // Linedef is completely inside the sector, ignore for building geometry
                    } else {
                        self.add_line(linedef.start_vertex.to_float(), linedef.end_vertex.to_float());
                    }
                } else if back_idx == self.sector.idx {
                    // Flip the vertices for easy matching
                    self.add_line(linedef.end_vertex.to_float(), linedef.start_vertex.to_float());
                }
            },
            (Some(front_idx), None) => {
                if front_idx == self.sector.idx {
                    self.add_line(linedef.start_vertex.to_float(), linedef.end_vertex.to_float());
                }
            },
            (None, Some(back_idx)) => {
                if back_idx == self.sector.idx {
                    // Flip the vertices for easy matching
                    self.add_line(linedef.end_vertex.to_float(), linedef.start_vertex.to_float());
                }
            },
            _ => {},
        }
    }

    fn build(
        self,
        vertices: &mut MapVertices,
        index_chunks: &mut HashMap<ResourceId, IndexChunk>,
    ) {
        let light = self.sector.light_level as f32 / 255.0;
        let tag = self.sector.tag;

        if !self.open_contours.is_empty() {
            warn!(
                sector_id = ?self.sector.idx,
                count = ?self.open_contours.len(),
                contours = ?self.open_contours,
                "Found incomplete sector contours"
            );
        }

        let mut contour_shapes = Vec::new();
        for shape in self.closed_shapes {
            shape.into_contour_shapes(&mut contour_shapes);
        }

        for contour_shape in contour_shapes {
            let triangulation = contour_shape.triangulate()
                .into_delaunay()
                .to_triangulation::<usize>();

            let (floor_indices, ceil_indices): (Vec<_>, Vec<_>) = triangulation.points.into_iter()
                .map(|v| {
                    // Floor/Ceil flat texture UVs are based off of a global grid of 64x64 flats
                    let floor_idx = vertices.get_or_insert(MapVertex {
                        position: glm::vec3(v[0], v[1], self.sector.floor_height as f32),
                        uv: glm::vec2(v[0] / 64.0, v[1] / 64.0),
                        light, tag,
                    });
                    let ceil_idx = vertices.get_or_insert(MapVertex {
                        position: glm::vec3(v[0], v[1], self.sector.ceil_height as f32),
                        uv: glm::vec2(v[0] / 64.0, v[1] / 64.0),
                        light, tag,
                    });
                    (floor_idx, ceil_idx)
                })
                .unzip();

            {
                let texture = index_chunks.entry(self.sector.floor_tex.id().clone())
                    .or_insert_with(|| IndexChunk {
                        texture: self.sector.floor_tex.clone(),
                        indices: Vec::new(),
                    });
                for idx in triangulation.indices.iter() {
                    texture.indices.push(floor_indices[*idx]);
                }
            }

            {
                let texture = index_chunks.entry(self.sector.ceil_tex.id().clone())
                    .or_insert_with(|| IndexChunk {
                        texture: self.sector.ceil_tex.clone(),
                        indices: Vec::new(),
                    });
                // Ceiling textures need reversed indices because the triangles face the
                // opposite direction
                for idx in triangulation.indices.iter().rev() {
                    texture.indices.push(ceil_indices[*idx]);
                }
            }
        }
    }
}

struct IndexChunk {
    texture: Arc<Resource<Texture>>,
    indices: Vec<MapVertexIndex>,
}

#[derive(Default)]
pub struct ChunkBuilder {
    vertices: MapVertices,
    index_chunks: HashMap<ResourceId, IndexChunk>,
    sectors: HashMap<usize, SectorBuilder>,
}

impl ChunkBuilder {
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
                SidedefLocation::Middle => true,
                SidedefLocation::Lower => !linedef.flags.contains(LinedefFlags::DontPegBottom),
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

                self.sectors.entry(front.sector.idx)
                    .or_insert_with(|| SectorBuilder::new(front.sector.clone()))
                    .add_linedef(linedef);
                self.sectors.entry(back.sector.idx)
                    .or_insert_with(|| SectorBuilder::new(back.sector.clone()))
                    .add_linedef(linedef);
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

                self.sectors.entry(front.sector.idx)
                    .or_insert_with(|| SectorBuilder::new(front.sector.clone()))
                    .add_linedef(linedef);
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

                self.sectors.entry(back.sector.idx)
                    .or_insert_with(|| SectorBuilder::new(back.sector.clone()))
                    .add_linedef(linedef);
            },
            _ => {},
        }
    }

    pub fn build(
        mut self,
        model_data: &mut MapModelData,
    ) -> Result<Chunk, ChunkBuildError> {
        for (_, sector_builder) in self.sectors {
            sector_builder.build(&mut self.vertices, &mut self.index_chunks);
        }

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
    ) -> Result<(), Validated<VulkanoError>> {
        let graphics = self.graphics.vk_graphics();

        builder
            .bind_pipeline_graphics(graphics.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                0,
                self.descriptor_set_0.descriptor_set().map_err(VulkanoError::from_validated)?,
            ).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics.layout().clone(),
                1,
                self.descriptor_set_1.descriptor_set().map_err(VulkanoError::from_validated)?,
            ).unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
            .bind_index_buffer(self.index_buffer.clone()).unwrap();
        self.chunk.draw(&graphics, builder);

        Ok(())
    }
}