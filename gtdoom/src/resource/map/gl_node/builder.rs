use std::cmp::Ordering;
use ahash::HashMap;
use educe::Educe;
use gtether::resource::Resource;
use i_triangle::float::triangulatable::Triangulatable;
use i_triangle::i_overlay::i_shape::base::data::{Contour, Shape as ContourShape};
use std::sync::Arc;
use fixed::FixedI64;
use fixed::types::extra::U16;
use tracing::warn;

use crate::resource::indexed::{IndexedArc, IndexedRef};
use crate::resource::map::gl_node::{GlNodes, GlSSector, GlSeg, GlVertexIndex, GlVertices};
use crate::resource::map::linedef::Linedef;
use crate::resource::map::sector::Sector;
use crate::resource::map::vertex::{FixedFloat, FixedVertex, Vertex, Vertices};

const VERTEX_EPSILON: FixedFloat = FixedFloat::lit("0.01");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointInPolygonResult {
    Inside,
    Outside,
    Edge,
}

fn point_in_polygon(v: &FixedVertex, poly: &Contour<FixedVertex>) -> PointInPolygonResult {
    if poly.len() < 3 {
        return PointInPolygonResult::Outside
    }

    // Uses Dan Sunday's winding algorithm:
    //  https://web.archive.org/web/20130126163405/http://geomalgorithms.com/a03-_inclusion.html

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PointPosition {
        Left,
        On,
        Right,
    }

    fn position(v: &FixedVertex, p1: &FixedVertex, p2: &FixedVertex) -> PointPosition {
        // Need to convert to FixedI64 to avoid overflow during multiplication
        let diff = FixedI64::<U16>::from(p2.x - p1.x) * FixedI64::<U16>::from(v.y - p1.y)
            - FixedI64::<U16>::from(v.x - p1.x) * FixedI64::<U16>::from(p2.y - p1.y);
        if diff.abs() < VERTEX_EPSILON {
            PointPosition::On
        } else if diff.is_positive() {
            PointPosition::Left
        } else {
            PointPosition::Right
        }
    }

    let mut wn: isize = 0;

    let iter = poly.windows(2)
        .map(|slice| (&slice[0], &slice[1]))
        // Add the final segment that connects the last and first vertices
        .chain(std::iter::once((poly.last().unwrap(), poly.first().unwrap())));

    for (p1, p2) in iter {
        if p1.y <= v.y {
            if p2.y > v.y {
                match position(v, p1, p2) {
                    // We want to know if the point is truly _inside_, so points on the polygon's
                    //  edge don't count.
                    PointPosition::On => return PointInPolygonResult::Edge,
                    PointPosition::Left => {
                        // upward crossing with valid intersection
                        wn += 1;
                    },
                    _ => {}
                }
            }
        } else {
            if p2.y <= v.y {
                match position(v, p1, p2) {
                    // We want to know if the point is truly _inside_, so points on the polygon's
                    //  edge don't count.
                    PointPosition::On => return PointInPolygonResult::Edge,
                    PointPosition::Right => {
                        // downward crossing with valid intersection
                        wn -= 1;
                    },
                    _ => {}
                }
            }
        }
    }

    if wn != 0 {
        // point is inside when the winding number is non-zero
        PointInPolygonResult::Inside
    } else {
        PointInPolygonResult::Outside
    }
}

fn segment_in_segment(
    s1v1: &FixedVertex,
    s1v2: &FixedVertex,
    s2v1: &FixedVertex,
    s2v2: &FixedVertex,
) -> bool {
    let (s1v1, s1v2, s2v1, s2v2) = match (s1v1.x.cmp(&s1v2.x), s2v1.x.cmp(&s2v2.x)) {
        // fix ordering of points
        (Ordering::Less, Ordering::Less) => (s1v1, s1v2, s2v1, s2v2),
        (Ordering::Less, Ordering::Greater) => (s1v1, s1v2, s2v2, s2v1),
        (Ordering::Greater, Ordering::Less) => (s1v2, s1v1, s2v1, s2v2),
        (Ordering::Greater, Ordering::Greater) => (s1v2, s1v1, s2v2, s2v1),

        // both lines are vertical with infinite slopes, just check the y values
        (Ordering::Equal, Ordering::Equal) => {
            // fix ordering of points
            let (s1v1, s1v2) = if s1v1.y <= s1v2.y {
                (s1v1, s1v2)
            } else {
                (s1v2, s1v1)
            };
            let (s2v1, s2v2) = if s2v1.y <= s2v2.y {
                (s2v1, s2v2)
            } else {
                (s2v2, s2v1)
            };

            // adjust the first segment's points by the epsilon to account for near equal segments
            return (s1v1.y + VERTEX_EPSILON) >= s2v1.y && (s1v2.y - VERTEX_EPSILON) <= s2v2.y
        },

        // one line is vertical and the other isn't; not a segment in segment
        (Ordering::Equal, _) | (_, Ordering::Equal) => return false,
    };

    fn slope_intercept(v1: &FixedVertex, v2: &FixedVertex) -> (FixedFloat, FixedFloat) {
        let slope = (v2.y - v1.y) / (v2.x - v1.x);
        let intercept = v1.y - (slope * v1.x);
        (slope, intercept)
    }

    let (l1_slope, l1_intercept) = slope_intercept(s1v1, s1v2);
    let (l2_slope, l2_intercept) = slope_intercept(s2v1, s2v2);
    if (l2_slope - l1_slope).abs() < VERTEX_EPSILON
        && (l2_intercept - l1_intercept).abs() < VERTEX_EPSILON {
        // lines are the same, now check the first segment is in the bounds of the second
        // adjust the first segment's points by the epsilon to account for near equal segments
        (s1v1.x + VERTEX_EPSILON) >= s2v1.x && (s1v2.x - VERTEX_EPSILON) <= s2v2.x
    } else {
        false
    }
}

struct NestedCompound {
    contour: Contour<FixedVertex>,
    sub_shapes: Vec<NestedCompound>,
}

impl NestedCompound {
    fn new(contour: Contour<FixedVertex>) -> Self {
        Self {
            contour,
            sub_shapes: Vec::new(),
        }
    }

    fn contains(&self, other: &NestedCompound) -> bool {
        for p in &other.contour {
            match point_in_polygon(p, &self.contour) {
                PointInPolygonResult::Outside | PointInPolygonResult::Edge => {
                    // Doesn't contain, early exit
                    return false
                },
                _ => {},
            }
        }
        true
    }

    fn insert_sub_shape(&mut self, mut new_sub_shape: NestedCompound) {
        let sub_sub_shapes = self.sub_shapes
            .extract_if(.., |ss| new_sub_shape.contains(ss))
            .collect::<Vec<_>>();
        for sub_sub_shape in sub_sub_shapes {
            new_sub_shape.insert_sub_shape(sub_sub_shape);
        }

        for ss in &mut self.sub_shapes {
            if ss.contains(&new_sub_shape) {
                ss.insert_sub_shape(new_sub_shape);
                return
            }
        }

        self.sub_shapes.push(new_sub_shape);
    }

    fn into_contour_shapes(self, output: &mut Vec<ContourShape<FixedVertex>>) {
        let mut contour_shape = ContourShape::with_capacity(1 + self.sub_shapes.len());
        contour_shape.push(self.contour);
        for hole in self.sub_shapes {
            contour_shape.push(hole.contour);
            // Double-nested contours are their own top-level shape again
            for sub_shape in hole.sub_shapes {
                sub_shape.into_contour_shapes(output);
            }
        }
        output.push(contour_shape);
    }
}

struct GlSSectorBuilder {
    parent_sector: IndexedArc<Sector>,
    linedefs: Vec<(IndexedArc<Linedef>, bool)>,
    open_contours: Vec<Contour<FixedVertex>>,
    closed_shapes: Vec<NestedCompound>,
}

impl GlSSectorBuilder {
    fn new(parent_sector: IndexedArc<Sector>) -> Self {
        Self {
            parent_sector,
            linedefs: Vec::new(),
            open_contours: Vec::new(),
            closed_shapes: Vec::new(),
        }
    }

    fn add_closed_contour(
        closed_shapes: &mut Vec<NestedCompound>,
        closed_contour: Contour<FixedVertex>,
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
        a: &Contour<FixedVertex>,
        b: &Contour<FixedVertex>,
    ) -> Option<(usize, usize)> {
        for i_b in 0..b.len() {
            for i_a in (0..a.len()).rev() {
                if a[i_a].equal_eps(&b[i_b], VERTEX_EPSILON) {
                    return Some((i_a, i_b))
                }
            }
        }
        None
    }

    fn add_line(&mut self, a: FixedVertex, b: FixedVertex) {
        // 1. Look for a prefix match first
        let mut new_segment = vec![b];
        for idx in 0..self.open_contours.len() {
            if let Some(first) = self.open_contours[idx].first() && b.equal_eps(first, VERTEX_EPSILON) {
                let contour = self.open_contours.swap_remove(idx);
                if a.equal_eps(contour.last().unwrap(), VERTEX_EPSILON) {
                    // Closing this contour
                    Self::add_closed_contour(&mut self.closed_shapes, contour);
                    return
                }
                new_segment = contour;
                break
            }
        }

        // 2. Look for a suffix match to attach the new segment to
        for contour in &mut self.open_contours {
            if let Some(last) = contour.last() && last.equal_eps(&a, VERTEX_EPSILON) {
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

    fn add_linedef(&mut self, linedef: IndexedRef<Linedef>) {
        let front_idx = linedef.front_sidedef.as_ref().map(|front| front.sector.idx);
        let back_idx = linedef.back_sidedef.as_ref().map(|back| back.sector.idx);
        match (front_idx, back_idx) {
            (Some(front_idx), Some(back_idx)) => {
                if front_idx == self.parent_sector.idx {
                    if back_idx == self.parent_sector.idx {
                        // Linedef is completely inside the sector, ignore for building geometry
                    } else {
                        self.add_line(linedef.start_vertex.into(), linedef.end_vertex.into());
                        self.linedefs.push((linedef.to_arc(), true));
                    }
                } else if back_idx == self.parent_sector.idx {
                    // Flip the vertices for easy matching
                    self.add_line(linedef.end_vertex.into(), linedef.start_vertex.into());
                    self.linedefs.push((linedef.to_arc(), false));
                }
            },
            (Some(front_idx), None) => {
                if front_idx == self.parent_sector.idx {
                    self.add_line(linedef.start_vertex.into(), linedef.end_vertex.into());
                    self.linedefs.push((linedef.to_arc(), true));
                }
            },
            (None, Some(back_idx)) => {
                if back_idx == self.parent_sector.idx {
                    // Flip the vertices for easy matching
                    self.add_line(linedef.end_vertex.into(), linedef.start_vertex.into());
                    self.linedefs.push((linedef.to_arc(), false));
                }
            },
            _ => {},
        }
    }

    fn build(
        self,
        vertices: &Vertices,
        largest_vertex_idx: &mut u32,
        output_gl_vertices: &mut Vec<FixedVertex>,
        output_gl_ssectors: &mut Vec<GlSSector>,
    ) {
        fn vertex_to_idx(
            vertex: FixedVertex,
            vertices: &Vertices,
            output_gl_vertices: &mut Vec<FixedVertex>,
        ) -> GlVertexIndex {
            if let Ok(vertex) = Vertex::try_from(vertex) {
                for (idx, v) in vertices.iter().enumerate() {
                    if &vertex == v {
                        return GlVertexIndex::Original(idx as u16)
                    }
                }
            }

            for (idx, v) in output_gl_vertices.iter().enumerate() {
                if vertex.equal_eps(v, VERTEX_EPSILON) {
                    return GlVertexIndex::Fixed(idx as u32)
                }
            }

            output_gl_vertices.push(vertex);
            GlVertexIndex::Fixed((output_gl_vertices.len() - 1) as u32)
        }

        fn linedef_for_seg(
            linedefs: &Vec<(IndexedArc<Linedef>, bool)>,
            v1: &FixedVertex,
            v2: &FixedVertex,
        ) -> Option<(IndexedArc<Linedef>, bool)> {
            for (linedef, front) in linedefs {
                if segment_in_segment(
                    v1, v2,
                    &FixedVertex::from(linedef.start_vertex),
                    &FixedVertex::from(linedef.end_vertex),
                ) {
                    return Some((linedef.clone(), *front))
                }
            }
            None
        }

        if !self.open_contours.is_empty() {
            warn!(
                sector_id = ?self.parent_sector.idx,
                count = ?self.open_contours.len(),
                contours = ?self.open_contours,
                "Found incomplete sector contours"
            );
        }

        let mut contour_shapes = vec![];
        for shape in self.closed_shapes {
            shape.into_contour_shapes(&mut contour_shapes);
        }

        let mut convex_contours = vec![];
        for contour_shape in contour_shapes {
            let out = contour_shape.triangulate()
                .into_delaunay()
                .to_convex_polygons();
            convex_contours.extend(out);
        }

        for contour in convex_contours {
            let segs = contour.windows(2)
                .map(|slice| (&slice[0], &slice[1]))
                .chain(std::iter::once((contour.last().unwrap(), contour.first().unwrap())))
                .map(|(v1, v2)| {
                    let vertex_idx = vertex_to_idx(*v1, vertices, output_gl_vertices);
                    if let GlVertexIndex::Original(idx) = &vertex_idx && *idx as u32 > *largest_vertex_idx {
                        *largest_vertex_idx = *idx as u32;
                    }
                    let (linedef, front) = match linedef_for_seg(&self.linedefs, v1, v2) {
                        Some((linedef, front)) => (Some(linedef), front),
                        None => (None, false),
                    };
                    GlSeg {
                        vertex_idx,
                        linedef,
                        front,
                    }
                })
                .collect::<Vec<_>>();
            output_gl_ssectors.push(GlSSector {
                sector: self.parent_sector.clone(),
                segs,
            });
        }
    }
}

#[derive(Educe)]
#[educe(Default)]
pub struct GlNodesBuilder {
    ssectors: HashMap<usize, GlSSectorBuilder>,
}

impl GlNodesBuilder {
    pub fn add_linedef(&mut self, linedef: IndexedRef<Linedef>) {
        match (&linedef.front_sidedef, &linedef.back_sidedef) {
            (Some(front), Some(back)) => {
                self.ssectors.entry(front.sector.idx)
                    .or_insert_with(|| GlSSectorBuilder::new(front.sector.clone()))
                    .add_linedef(linedef);
                self.ssectors.entry(back.sector.idx)
                    .or_insert_with(|| GlSSectorBuilder::new(back.sector.clone()))
                    .add_linedef(linedef);
            },
            (Some(front), None) => {
                self.ssectors.entry(front.sector.idx)
                    .or_insert_with(|| GlSSectorBuilder::new(front.sector.clone()))
                    .add_linedef(linedef);
            },
            (None, Some(back)) => {
                self.ssectors.entry(back.sector.idx)
                    .or_insert_with(|| GlSSectorBuilder::new(back.sector.clone()))
                    .add_linedef(linedef);
            },
            (None, None) => {},
        }
    }

    pub fn build(
        self,
        vertices: Arc<Resource<Vertices>>,
    ) -> GlNodes {
        let mut gl_vertices = vec![];
        let mut gl_ssectors = vec![];

        let mut largest_vertex_idx = 0;
        {
            let vertices_lock = vertices.read();
            for (_, ssector_builder) in self.ssectors {
                ssector_builder.build(
                    &vertices_lock,
                    &mut largest_vertex_idx,
                    &mut gl_vertices,
                    &mut gl_ssectors,
                );
            }
        }

        let gl_vertices = GlVertices {
            original_vertex_count: largest_vertex_idx + 1,
            original_vertices: vertices,
            fixed_vertices: gl_vertices,
        };

        GlNodes {
            gl_vertices,
            gl_ssectors,
        }
    }
}