use crate::geometry::line::{Line, LinearFunction, SlopeInterceptFunction};
use crate::geometry::{Aabb, Axis2D, CompassDirection, Direction2D};
use crate::resource::map::blockmap::{BlockIndex, Blockmap, BlockmapDimensions, BlockmapNum};

use ahash::HashSet;
use std::collections::VecDeque;
use std::iter::FusedIterator;

pub trait BlockmapQuery<const BLOCK_SIZE: u32> {
    type Metadata;
    type IndexQuery: Iterator<Item=(BlockIndex, Self::Metadata)>;

    fn index_query(
        self,
        dims: &BlockmapDimensions<BLOCK_SIZE>,
    ) -> Self::IndexQuery;
}

pub struct QueryResult<'a, const BLOCK_SIZE: u32, T, Q>
where
    Q: BlockmapQuery<BLOCK_SIZE>,
{
    pub(super) blockmap: &'a Blockmap<BLOCK_SIZE, T>,
    pub(super) iter: Q::IndexQuery,
}

impl<'a, const BLOCK_SIZE: u32, T, Q> Iterator for QueryResult<'a, BLOCK_SIZE, T, Q>
where
    Q: BlockmapQuery<BLOCK_SIZE>,
{
    type Item = (&'a T, Q::Metadata);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (idx, metadata) = self.iter.next()?;
            if let Some(block) = self.blockmap.get(&idx) {
                return Some((block, metadata))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, const BLOCK_SIZE: u32, T, Q> ExactSizeIterator for QueryResult<'a, BLOCK_SIZE, T, Q>
where
    Q: BlockmapQuery<BLOCK_SIZE>,
    Q::IndexQuery: ExactSizeIterator,
{}

impl<'a, const BLOCK_SIZE: u32, T, Q> FusedIterator for QueryResult<'a, BLOCK_SIZE, T, Q>
where
    Q: BlockmapQuery<BLOCK_SIZE>,
    Q::IndexQuery: FusedIterator,
{}

// This function assumes that `line.a` is INSIDE `bounds`
fn bounds_crossing<N: BlockmapNum>(bounds: &Aabb<N>, line: &Line<N>) -> (glm::TVec2<N>, CompassDirection) {
    match line.to_function() {
        LinearFunction::Horizontal(line_fn) => {
            if line.b.x > line.a.x {
                (glm::vec2(bounds.max.x, line_fn.y), CompassDirection::X_POSITIVE)
            } else {
                (glm::vec2(bounds.min.x, line_fn.y), CompassDirection::X_NEGATIVE)
            }
        },
        LinearFunction::Vertical(line_fn) => {
            if line.b.y > line.a.y {
                (glm::vec2(line_fn.x, bounds.max.y), CompassDirection::Y_POSITIVE)
            } else {
                (glm::vec2(line_fn.x, bounds.min.y), CompassDirection::Y_NEGATIVE)
            }
        },
        LinearFunction::Sloped(line_fn) => {
            let (crossed, dir) = if line.b.x > line.a.x {
                (glm::vec2(bounds.max.x, line_fn.y(bounds.max.x)), CompassDirection::X_POSITIVE)
            } else {
                (glm::vec2(bounds.min.x, line_fn.y(bounds.min.x)), CompassDirection::X_NEGATIVE)
            };

            if crossed == bounds.max {
                (crossed, CompassDirection::X_POS_Y_POS)
            } else if crossed == bounds.min {
                (crossed, CompassDirection::X_NEG_Y_NEG)
            } else if crossed.x == bounds.max.x && crossed.y == bounds.min.y {
                (crossed, CompassDirection::X_POS_Y_NEG)
            } else if crossed.x == bounds.min.x && crossed.y == bounds.max.y {
                (crossed, CompassDirection::X_NEG_Y_POS)
            } else if crossed.y >= bounds.min.y && crossed.y < bounds.max.y {
                (crossed, dir)
            } else if line.b.y > line.a.y {
                (glm::vec2(line_fn.x(bounds.max.y), bounds.max.y), CompassDirection::Y_POSITIVE)
            } else {
                (glm::vec2(line_fn.x(bounds.min.y), bounds.min.y), CompassDirection::Y_NEGATIVE)
            }
        },
    }
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> BlockmapQuery<BLOCK_SIZE> for Line<N> {
    type Metadata = LineQueryMetadata<N>;
    type IndexQuery = LineQuery<BLOCK_SIZE, N>;

    fn index_query(self, dims: &BlockmapDimensions<BLOCK_SIZE>) -> Self::IndexQuery {
        LineQuery::init(self, *dims)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineQueryMetadata<N> {
    pub segment: Line<N>,
}

pub struct LineQuery<const BLOCK_SIZE: u32, N> {
    dims: BlockmapDimensions<BLOCK_SIZE>,
    line_fn: LinearFunction<N>,
    curr_idx: Option<BlockIndex>,
    last: glm::TVec2<N>,
    end: glm::TVec2<N>,
    flip_output: bool,
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> LineQuery<BLOCK_SIZE, N> {
    fn check_sloped_start_bounds(
        line_fn: &SlopeInterceptFunction<N>,
        start: glm::TVec2<N>,
        min: glm::TVec2<N>,
        max: glm::TVec2<N>,
        block_count: glm::TVec2<u32>,
    ) -> (Option<BlockIndex>, glm::TVec2<N>) {
        // x is considered the "aligned" direction, and y is the "cross" direction
        if start.y >= max.y {
            let start_align = match line_fn.primary_axis() {
                Axis2D::X => line_fn.x(max.y),
                Axis2D::Y => line_fn.y(max.y),
            };
            if start_align >= start.x && start_align < max.x {
                let idx = BlockIndex::new(
                    (start_align - min.x).to_u32().expect("overflow") / BLOCK_SIZE,
                    block_count.y - 1,
                );
                (Some(idx), glm::vec2(start_align, max.y))
            } else {
                (None, start)
            }
        } else if start.y < min.y {
            let start_align = match line_fn.primary_axis() {
                Axis2D::X => line_fn.x(min.y),
                Axis2D::Y => line_fn.y(min.y),
            };
            if start_align >= start.x && start_align < max.x {
                let idx = BlockIndex::new(
                    (start_align - min.x).to_u32().expect("overflow") / BLOCK_SIZE,
                    0,
                );
                (Some(idx), glm::vec2(start_align, min.y))
            } else {
                (None, start)
            }
        } else {
            let idx = (start - min).map(|v| {
                v.to_u32().expect("overflow") / BLOCK_SIZE
            }) as BlockIndex;
            (Some(idx), start)
        }
    }

    fn init(line: Line<N>, dims: BlockmapDimensions<BLOCK_SIZE>) -> Self {
        let bounds = dims.convert_bounds::<N>();
        let line_fn = line.to_function();

        let (curr_idx, last, end, flip_output) = match &line_fn {
            LinearFunction::Vertical(line_fn) => {
                if line_fn.x >= bounds.min.x && line_fn.x < bounds.max.x {
                    let (a, b, flip_output) = line.ordered_points_by_y();
                    if b.y < bounds.min.y || a.y >= bounds.max.y {
                        (None, *a, *b, flip_output)
                    } else if a.y < bounds.min.y {
                        let start = glm::vec2(line_fn.x, bounds.min.y);
                        let curr_idx = dims.get_index(&start);
                        (curr_idx, start, *b, flip_output)
                    } else {
                        let start = glm::vec2(line_fn.x, a.y);
                        let curr_idx = dims.get_index(&start);
                        (curr_idx, start, *b, flip_output)
                    }
                } else {
                    (None, line.a, line.b, false)
                }
            },
            LinearFunction::Horizontal(line_fn) => {
                if line_fn.y >= bounds.min.y && line_fn.y < bounds.max.y {
                    let (a, b, flip_output) = line.ordered_points_by_x();
                    if b.x < bounds.min.x || a.x >= bounds.max.x {
                        (None, *a, *b, flip_output)
                    } else if a.x < bounds.min.x {
                        let start = glm::vec2(bounds.min.x, line_fn.y);
                        let curr_idx = dims.get_index(&start);
                        (curr_idx, start, *b, flip_output)
                    } else {
                        let start = glm::vec2(a.x, line_fn.y);
                        let curr_idx = dims.get_index(&start);
                        (curr_idx, start, *b, flip_output)
                    }
                } else {
                    (None, line.a, line.b, false)
                }
            },
            LinearFunction::Sloped(line_fn) => {
                let primary_axis = line_fn.primary_axis();

                match primary_axis {
                    Axis2D::X => {
                        let (a, b, flip_output) = line.ordered_points_by_x();
                        let (start_idx, start) = if b.x < bounds.min.x {
                            (None, *a)
                        } else if a.x < bounds.min.x {
                            let start_y = line_fn.y(bounds.min.x);
                            Self::check_sloped_start_bounds(
                                line_fn,
                                glm::vec2(bounds.min.x, start_y),
                                bounds.min,
                                bounds.max,
                                *dims.block_count(),
                            )
                        } else if a.x < bounds.max.x {
                            Self::check_sloped_start_bounds(
                                line_fn,
                                *a,
                                bounds.min,
                                bounds.max,
                                *dims.block_count(),
                            )
                        } else {
                            (None, *a)
                        };
                        (start_idx, start, *b, flip_output)
                    },
                    Axis2D::Y => {
                        let (a, b, flip_output) = line.ordered_points_by_y();
                        let (start_idx, start) = if b.y < bounds.min.y {
                            (None, *a)
                        } else if a.y < bounds.min.y {
                            let start_x = line_fn.x(bounds.min.y);
                            // flip axes for this check
                            let (start_idx, start) = Self::check_sloped_start_bounds(
                                line_fn,
                                glm::vec2(bounds.min.y, start_x),
                                bounds.min.yx(),
                                bounds.max.yx(),
                                dims.block_count().yx(),
                            );
                            (start_idx.map(|i| i.yx()), start.yx())
                        } else if a.y < bounds.max.y {
                            // flip axes for this check
                            let (start_idx, start) = Self::check_sloped_start_bounds(
                                line_fn,
                                a.yx(),
                                bounds.min.yx(),
                                bounds.max.yx(),
                                dims.block_count().yx(),
                            );
                            (start_idx.map(|i| i.yx()), start.yx())
                        } else {
                            (None, *a)
                        };
                        (start_idx, start, *b, flip_output)
                    },
                }
            },
        };

        Self {
            dims,
            line_fn,
            curr_idx,
            last,
            end,
            flip_output,
        }
    }
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> Iterator for LineQuery<BLOCK_SIZE, N> {
    type Item = (BlockIndex, LineQueryMetadata<N>);

    fn next(&mut self) -> Option<Self::Item> {
        let curr_idx = self.curr_idx.take()?;
        let curr_bounds = self.dims.bounds_for_index(&curr_idx).unwrap();
        if curr_bounds.contains(&self.end) {
            if self.last != self.end {
                // the current idx is the last one, yield it now
                let (a, b) = match self.flip_output {
                    true => (self.end, self.last),
                    false => (self.last, self.end),
                };
                let metadata = LineQueryMetadata {
                    segment: Line {
                        a,
                        b,
                    }
                };
                return Some((curr_idx, metadata))
            } else {
                // Last item is actually a point; end early
                return None
            }
        }

        let (next_idx, next) = match &self.line_fn {
            LinearFunction::Vertical(line_fn) => {
                (
                    self.dims.checked_index_neighbor(&curr_idx, Direction2D::Y_POSITIVE),
                    glm::vec2(line_fn.x, curr_bounds.max.y),
                )
            },
            LinearFunction::Horizontal(line_fn) => {
                (
                    self.dims.checked_index_neighbor(&curr_idx, Direction2D::X_POSITIVE),
                    glm::vec2(curr_bounds.max.x, line_fn.y),
                )
            },
            LinearFunction::Sloped(line_fn) => {
                match line_fn.primary_axis() {
                    Axis2D::X => {
                        let next_aligned_y = line_fn.y(curr_bounds.max.x);

                        if next_aligned_y >= curr_bounds.max.y {
                            // crossed up before right
                            (
                                self.dims.checked_index_neighbor(&curr_idx, Direction2D::Y_POSITIVE),
                                glm::vec2(line_fn.x(curr_bounds.max.y), curr_bounds.max.y),
                            )
                        } else if next_aligned_y < curr_bounds.min.y {
                            // crossed down before right
                            (
                                self.dims.checked_index_neighbor(&curr_idx, Direction2D::Y_NEGATIVE),
                                glm::vec2(line_fn.x(curr_bounds.min.y), curr_bounds.min.y),
                            )
                        } else {
                            // crossed right
                            (
                                self.dims.checked_index_neighbor(&curr_idx, Direction2D::X_POSITIVE),
                                glm::vec2(curr_bounds.max.x, next_aligned_y),
                            )
                        }
                    },
                    Axis2D::Y => {
                        let next_aligned_x = line_fn.x(curr_bounds.max.y);

                        if next_aligned_x >= curr_bounds.max.x {
                            // crossed right before up
                            (
                                self.dims.checked_index_neighbor(&curr_idx, Direction2D::X_POSITIVE),
                                glm::vec2(curr_bounds.max.x, line_fn.y(curr_bounds.max.x)),
                            )
                        } else if next_aligned_x < curr_bounds.min.x {
                            // crossed left before up
                            (
                                self.dims.checked_index_neighbor(&curr_idx, Direction2D::X_NEGATIVE),
                                glm::vec2(curr_bounds.min.x, line_fn.y(curr_bounds.min.x)),
                            )
                        } else {
                            // crossed up
                            (
                                self.dims.checked_index_neighbor(&curr_idx, Direction2D::Y_POSITIVE),
                                glm::vec2(next_aligned_x, curr_bounds.max.y),
                            )
                        }
                    }
                }
            }
        };

        let (a, b) = match self.flip_output {
            true => (next, self.last),
            false => (self.last, next),
        };
        let metadata = LineQueryMetadata {
            segment: Line {
                a,
                b,
            }
        };

        self.curr_idx = next_idx;

        if self.last == next {
            // we're at an intersection between blocks; this result is useless as it represents the
            // point of that intersection. recurse and grab the next result. this should occur only
            // *once* so recursion should be limited
            self.next()
        } else {
            self.last = next;
            Some((curr_idx, metadata))
        }
    }
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> FusedIterator for LineQuery<BLOCK_SIZE, N> {}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> BlockmapQuery<BLOCK_SIZE> for Vec<glm::TVec2<N>> {
    type Metadata = ContourQueryMetadata<N>;
    type IndexQuery = ContourQuery<BLOCK_SIZE, N>;

    fn index_query(self, dims: &BlockmapDimensions<BLOCK_SIZE>) -> Self::IndexQuery {
        ContourQuery::init(self, *dims)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ContourQueryMetadata<N> {
    pub contour: Vec<glm::TVec2<N>>,
}

pub struct ContourQuery<const BLOCK_SIZE: u32, N> {
    contour: VecDeque<glm::TVec2<N>>,
    dims: BlockmapDimensions<BLOCK_SIZE>,
    visited_indices: HashSet<BlockIndex>,
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> ContourQuery<BLOCK_SIZE, N> {
    fn init(contour: Vec<glm::TVec2<N>>, dims: BlockmapDimensions<BLOCK_SIZE>) -> Self {
        let contour = VecDeque::from(contour);
        //trace!(?contour, "INIT QUERY -------------------------------------------");
        Self {
            contour,
            dims,
            visited_indices: HashSet::default(),
        }
    }
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> Iterator for ContourQuery<BLOCK_SIZE, N> {
    type Item = (BlockIndex, ContourQueryMetadata<N>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.contour.is_empty() {
            return None
        }

        //trace!(contour = ?self.contour, "  ITER_-START");

        let mut found: Option<(usize, BlockIndex, &glm::TVec2<N>)> = None;
        for (i, p) in self.contour.iter().enumerate() {
            let iter = self.dims.get_indices_inclusive(p);
            debug_assert!(
                iter.len() > 0,
                "invalid contour; point ({p:?}) out of bounds {:?}",
                self.dims.bounds(),
            );
            for idx in iter {
                if !self.visited_indices.contains(&idx) {
                    let replace = if let Some((_, block_idx, fp)) = found {
                        if idx.x > block_idx.x || (idx.x == block_idx.x && idx.y > block_idx.y) {
                            p.x > fp.x || (p.x == fp.x && p.y > fp.y)
                        } else {
                            false
                        }
                    } else {
                        true
                    };
                    if replace {
                        found = Some((i, idx, p));
                    }
                }
            }
        }
        let (i, block_idx, _) = found?;

        self.contour.rotate_left(i);
        let mut output = vec![self.contour.pop_front().unwrap()];

        //trace!(?block_idx, first = ?output.first().unwrap(), "  FOUND-FIRST");

        let bounds = self.dims.bounds_for_index(&block_idx).unwrap();
        let scan_count = self.contour.len();
        let mut shift_count = 0;
        let mut last_crossed_dir: Option<Direction2D> = None;
        let mut new_edge = vec![];

        fn finalize_edge<N: BlockmapNum>(
            last_crossed_dir: Direction2D,
            line: Line<N>,
            bounds: &Aabb<N>,
            new_edge: &mut Vec<glm::TVec2<N>>,
            output: &mut Vec<glm::TVec2<N>>,
            contour: &mut VecDeque<glm::TVec2<N>>,
        ) {
            let (crossed, crossed_dir) = bounds_crossing(bounds, &line);

            //trace!(?last_crossed_dir, ?crossed_dir, "    CRSS-BACK");

            if Some(&crossed) != new_edge.last() {
                for dir in last_crossed_dir.cw_to(crossed_dir.to_2d_ccw()) {
                    let corner = match dir {
                        Direction2D::Right => bounds.max,
                        Direction2D::Left => bounds.min,
                        Direction2D::Up => glm::vec2(bounds.min.x, bounds.max.y),
                        Direction2D::Down => glm::vec2(bounds.max.x, bounds.min.y),
                    };
                    //trace!(point = ?corner, ?dir, "    EDGE-ADD_");
                    if Some(&corner) != new_edge.last() {
                        //trace!(point = ?corner, ?dir, "    EDGE-ADD_");
                        new_edge.push(corner);
                    }
                }
            }
            if Some(&crossed) != new_edge.last() {
                //trace!(point = ?crossed, "    EDGE-ADD_");
                new_edge.push(crossed);
            }

            //trace!(edge = ?new_edge, "    EDGE-FNLZ");
            let new_edge_size = new_edge.len();

            output.reserve(new_edge_size);
            for p in new_edge.iter() {
                if Some(p) != output.last() {
                    output.push(*p);
                }
            }

            for p in new_edge.drain(..).rev() {
                if Some(&p) != contour.back() {
                    contour.push_back(p);
                }
            }
        }

        for _ in 0..scan_count {
            let next = self.contour.get(shift_count).unwrap();
            //trace!(?next, contour = ?self.contour, ?shift_count, "      ");
            if bounds.contains_inclusive(next) {
                self.contour.rotate_left(shift_count);
                shift_count = 0;

                let point = self.contour.pop_front().unwrap();

                if let Some(dir) = last_crossed_dir {
                    let prev = self.contour.back().unwrap();
                    finalize_edge(
                        dir,
                        Line::new(point, *prev),
                        &bounds,
                        &mut new_edge,
                        &mut output,
                        &mut self.contour,
                    );
                    last_crossed_dir = None;
                }

                //trace!(?point, "    OUT_-ADD_");
                if Some(&point) != output.last() {
                    output.push(point);
                }
            } else if shift_count == 0 {
                let (crossed, dir) = bounds_crossing(&bounds, &Line::new(
                    *output.last().unwrap(),
                    *self.contour.front().unwrap(),
                ));

                //trace!(?dir, point = ?crossed, "    CRSS-OUT_");

                new_edge.push(crossed);

                shift_count += 1;
                last_crossed_dir = Some(dir.to_2d_cw());
            } else {
                shift_count += 1;
            }
        }

        self.contour.rotate_left(shift_count);

        if let Some(last_crossed_dir) = last_crossed_dir && let Some(last) = self.contour.back() {
            //trace!(contour = ?self.contour, "      ");
            finalize_edge(
                last_crossed_dir,
                Line::new(*output.first().unwrap(), *last),
                &bounds,
                &mut new_edge,
                &mut output,
                &mut self.contour,
            );
        }

        self.visited_indices.insert(block_idx);

        if output.first() == output.last() {
            output.pop();
        }

        if output.len() <= 2 {
            //trace!("  END__-SKIP_");
            self.next()
        } else {
            //trace!(?output, "  END__-OUT__");
            let metadata = ContourQueryMetadata {
                contour: output,
            };
            Some((block_idx, metadata))
        }
    }
}

impl<const BLOCK_SIZE: u32, N: BlockmapNum> FusedIterator for ContourQuery<BLOCK_SIZE, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    use glm::Scalar;
    use rstest::{fixture, rstest};
    use std::cmp::Ordering;
    use std::collections::BTreeMap;
    use test_log::test as test_log;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SortedBlockIndex(BlockIndex);

    impl PartialOrd for SortedBlockIndex {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for SortedBlockIndex {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.0.x.cmp(&other.0.x) {
                Ordering::Equal => self.0.y.cmp(&other.0.y),
                order => order,
            }
        }
    }

    #[fixture]
    fn blockmap<const BLOCK_SIZE: u32>(
        #[default(glm::vec2(-32, -32))]
        offset: glm::TVec2<i32>,
        #[default(glm::vec2(4, 4))]
        block_count: glm::TVec2<u32>,
    ) -> Blockmap<BLOCK_SIZE, SortedBlockIndex> {
        let dims = BlockmapDimensions::new(
            offset,
            block_count * BLOCK_SIZE,
        );
        Blockmap::from_fn(dims, |idx| SortedBlockIndex(idx))
    }

    #[rstest]
    #[case::horiz_below_1(Line::from_raw(-40.0, -40.0, 40.0, -40.0), [])]
    #[case::horiz_below_2(Line::from_raw(12.0, -40.0, 20.0, -40.0), [])]
    #[case::horiz_below_3(Line::from_raw(-40.0, -40.0, 20.0, -40.0), [])]
    #[case::horiz_above_1(Line::from_raw(-40.0, 40.0, 40.0, 40.0), [])]
    #[case::horiz_above_2(Line::from_raw(12.0, 40.0, 40.0, 40.0), [])]
    #[case::horiz_above_3(Line::from_raw(-40.0, 40.0, 20.0, 40.0), [])]
    #[case::horiz_inside_1(Line::from_raw(-30.0, 24.0, -18.0, 24.0), [
        (BlockIndex::new(0, 3), Line::from_raw(-30.0, 24.0, -18.0, 24.0)),
    ])]
    #[case::horiz_inside_2(Line::from_raw(12.0, 24.0, -8.0, 24.0), [
        (BlockIndex::new(2, 3), Line::from_raw(12.0, 24.0, 0.0, 24.0)),
        (BlockIndex::new(1, 3), Line::from_raw(0.0, 24.0, -8.0, 24.0)),
    ])]
    #[case::horiz_inside_3(Line::from_raw(-10.0, -24.0, 30.0, -24.0), [
        (BlockIndex::new(1, 0), Line::from_raw(-10.0, -24.0, 0.0, -24.0)),
        (BlockIndex::new(2, 0), Line::from_raw(0.0, -24.0, 16.0, -24.0)),
        (BlockIndex::new(3, 0), Line::from_raw(16.0, -24.0, 30.0, -24.0)),
    ])]
    #[case::horiz_partial_before_1(Line::from_raw(-25.0, -24.0, -35.0, -24.0), [
        (BlockIndex::new(0, 0), Line::from_raw(-25.0, -24.0, -32.0, -24.0)),
    ])]
    #[case::horiz_partial_before_2(Line::from_raw(-40.0, 0.0, 0.0, 0.0), [
        (BlockIndex::new(0, 2), Line::from_raw(-32.0, 0.0, -16.0, 0.0)),
        (BlockIndex::new(1, 2), Line::from_raw(-16.0, 0.0, 0.0, 0.0)),
    ])]
    #[case::horiz_partial_after(Line::from_raw(20.0, 12.0, 40.0, 12.0), [
        (BlockIndex::new(3, 2), Line::from_raw(20.0, 12.0, 32.0, 12.0)),
    ])]

    #[case::vert_left_1(Line::from_raw(-40.0, -40.0, -40.0, 40.0), [])]
    #[case::vert_left_2(Line::from_raw(-40.0, -24.0, -40.0, 40.0), [])]
    #[case::vert_left_3(Line::from_raw(-40.0, -40.0, -40.0, 24.0), [])]
    #[case::vert_right_1(Line::from_raw(40.0, -40.0, 40.0, 40.0), [])]
    #[case::vert_right_2(Line::from_raw(40.0, -24.0, 40.0, 40.0), [])]
    #[case::vert_right_3(Line::from_raw(40.0, -40.0, 40.0, 24.0), [])]
    #[case::vert_inside_1(Line::from_raw(2.0, 30.0, 2.0, 18.0), [
        (BlockIndex::new(2, 3), Line::from_raw(2.0, 30.0, 2.0, 18.0)),
    ])]
    #[case::vert_inside_2(Line::from_raw(2.0, -14.0, 2.0, 3.0), [
        (BlockIndex::new(2, 1), Line::from_raw(2.0, -14.0, 2.0, 0.0)),
        (BlockIndex::new(2, 2), Line::from_raw(2.0, 0.0, 2.0, 3.0)),
    ])]
    #[case::vert_inside_3(Line::from_raw(-2.0, 20.0, -2.0, -10.0), [
        (BlockIndex::new(1, 3), Line::from_raw(-2.0, 20.0, -2.0, 16.0)),
        (BlockIndex::new(1, 2), Line::from_raw(-2.0, 16.0, -2.0, 0.0)),
        (BlockIndex::new(1, 1), Line::from_raw(-2.0, 0.0, -2.0, -10.0)),
    ])]
    #[case::vert_partial_before_1(Line::from_raw(-2.0, -40.0, -2.0, -24.0), [
        (BlockIndex::new(1, 0), Line::from_raw(-2.0, -32.0, -2.0, -24.0)),
    ])]
    #[case::vert_partial_before_2(Line::from_raw(24.0, -40.0, 24.0, 0.0), [
        (BlockIndex::new(3, 0), Line::from_raw(24.0, -32.0, 24.0, -16.0)),
        (BlockIndex::new(3, 1), Line::from_raw(24.0, -16.0, 24.0, 0.0)),
    ])]
    #[case::vert_partial_after(Line::from_raw(24.0, 40.0, 24.0, 24.0), [
        (BlockIndex::new(3, 3), Line::from_raw(24.0, 32.0, 24.0, 24.0)),
    ])]

    #[case::slope_inside_1(Line::from_raw(20.0, 30.0, 25.0, 18.0), [
        (BlockIndex::new(3, 3), Line::from_raw(20.0, 30.0, 25.0, 18.0)),
    ])]
    #[case::slope_inside_2(Line::from_raw(-4.0, 17.0, 24.0, 10.0), [
        (BlockIndex::new(1, 3), Line::from_raw(-4.0, 17.0, 0.0, 16.0)),
        (BlockIndex::new(2, 2), Line::from_raw(0.0, 16.0, 16.0, 12.0)),
        (BlockIndex::new(3, 2), Line::from_raw(16.0, 12.0, 24.0, 10.0)),
    ])]
    #[case::slope_outward_nw(Line::from_raw(0.0, 0.0, -2.0, 64.0), [
        (BlockIndex::new(1, 2), Line::from_raw(0.0, 0.0, -0.5, 16.0)),
        (BlockIndex::new(1, 3), Line::from_raw(-0.5, 16.0, -1.0, 32.0)),
    ])]
    #[case::slope_outward_ne(Line::from_raw(0.0, 0.0, 50.0, 50.0), [
        (BlockIndex::new(2, 2), Line::from_raw(0.0, 0.0, 16.0, 16.0)),
        (BlockIndex::new(3, 3), Line::from_raw(16.0, 16.0, 32.0, 32.0)),
    ])]
    #[case::slope_outward_se(Line::from_raw(0.0, 0.0, 40.0, -50.0), [
        (BlockIndex::new(2, 1), Line::from_raw(0.0, 0.0, 12.8, -16.0)),
        (BlockIndex::new(2, 0), Line::from_raw(12.8, -16.0, 16.0, -20.0)),
        (BlockIndex::new(3, 0), Line::from_raw(16.0, -20.0, 25.6, -32.0)),
    ])]
    #[case::slope_outward_sw(Line::from_raw(0.0, 0.0, -40.0, -20.0), [
        (BlockIndex::new(1, 1), Line::from_raw(0.0, 0.0, -16.0, -8.0)),
        (BlockIndex::new(0, 1), Line::from_raw(-16.0, -8.0, -32.0, -16.0)),
    ])]
    // TODO: more tests
    #[test_attr(test_log)]
    fn test_line_query<N: BlockmapNum>(
        blockmap: Blockmap<16, SortedBlockIndex>,
        #[case] line: Line<N>,
        #[case] expected: impl IntoIterator<Item=(BlockIndex, Line<N>)>,
    ) {
        let query = blockmap.query(line);

        let actual = query
            .map(|(idx, metadata)| {
                (*idx, metadata)
            })
            .collect::<BTreeMap<_, _>>();

        let expected = expected.into_iter()
            .map(|(idx, segment)| {
                let metadata = LineQueryMetadata {
                    segment,
                };
                (SortedBlockIndex(idx), metadata)
            })
            .collect::<BTreeMap<_, _>>();

        assert_eq!(actual, expected);
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ContourCheck<N> {
        contour: VecDeque<glm::TVec2<N>>,
    }

    impl<N: Scalar + PartialEq> ContourCheck<N> {
        fn align(&mut self, other: &Self) {
            let mut index_pair = None;
            for (i1, p1) in self.contour.iter().enumerate() {
                for (i2, p2) in other.contour.iter().enumerate() {
                    if p1 == p2 {
                        index_pair = Some((i1, i2));
                    }
                }
            }

            if let Some((i1, i2)) = index_pair {
                if i1 > i2 {
                    self.contour.rotate_left(i1 - i2);
                } else if i1 < i2 {
                    self.contour.rotate_right(i2 - i1);
                }
            }
        }
    }

    fn align_contour_maps<N: Scalar + PartialEq>(
        a: &mut BTreeMap<SortedBlockIndex, ContourCheck<N>>,
        b: &BTreeMap<SortedBlockIndex, ContourCheck<N>>,
    ) {
        for (k, v) in a.iter_mut() {
            if let Some(other) = b.get(k) {
                v.align(other);
            }
        }
    }

    #[rstest]
    #[case::single_block(
        [glm::vec2(20.0, -30.0), glm::vec2(24.0, -20.0), glm::vec2(28.0, -30.0)],
        [
            (BlockIndex::new(3, 0), vec![
                glm::vec2(20.0, -30.0), glm::vec2(24.0, -20.0), glm::vec2(28.0, -30.0),
            ]),
        ],
    )]
    #[case::single_block_aligned(
        [glm::vec2(-32.0, 16.0), glm::vec2(-16.0, 16.0), glm::vec2(-16.0, 0.0), glm::vec2(-32.0, 0.0)],
        [
            (BlockIndex::new(0, 2), vec![
                glm::vec2(-32.0, 0.0), glm::vec2(-32.0, 16.0),
                glm::vec2(-16.0, 16.0), glm::vec2(-16.0, 0.0),
            ]),
        ],
    )]
    #[case::double_block(
        [glm::vec2(8.0, -10.0), glm::vec2(8.0, -2.0), glm::vec2(20.0, -2.0), glm::vec2(20.0, -10.0)],
        [
            (BlockIndex::new(2, 1), vec![
                glm::vec2(8.0, -10.0), glm::vec2(8.0, -2.0),
                glm::vec2(16.0, -2.0), glm::vec2(16.0, -10.0),
            ]),
            (BlockIndex::new(3, 1), vec![
                glm::vec2(20.0, -2.0), glm::vec2(20.0, -10.0),
                glm::vec2(16.0, -10.0), glm::vec2(16.0, -2.0),
            ]),
        ],
    )]
    #[case::double_block_aligned(
        [glm::vec2(-16.0, 0.0), glm::vec2(16.0, 0.0), glm::vec2(16.0, -16.0), glm::vec2(-16.0, -16.0)],
        [
            (BlockIndex::new(1, 1), vec![
                glm::vec2(-16.0, -16.0), glm::vec2(-16.0, 0.0),
                glm::vec2(0.0, 0.0), glm::vec2(0.0, -16.0),
            ]),
            (BlockIndex::new(2, 1), vec![
                glm::vec2(0.0, -16.0), glm::vec2(0.0, 0.0),
                glm::vec2(16.0, 0.0), glm::vec2(16.0, -16.0),
            ]),
        ],
    )]
    #[case::quad_block(
        [glm::vec2(-8.0, -8.0), glm::vec2(-8.0, 8.0), glm::vec2(8.0, 8.0), glm::vec2(8.0, -8.0)],
        [
            (BlockIndex::new(1, 1), vec![
                glm::vec2(-8.0, -8.0), glm::vec2(-8.0, 0.0),
                glm::vec2(0.0, 0.0), glm::vec2(0.0, -8.0),
            ]),
            (BlockIndex::new(2, 1), vec![
                glm::vec2(0.0, -8.0), glm::vec2(0.0, 0.0),
                glm::vec2(8.0, 0.0), glm::vec2(8.0, -8.0),
            ]),
            (BlockIndex::new(1, 2), vec![
                glm::vec2(-8.0, 0.0), glm::vec2(-8.0, 8.0),
                glm::vec2(0.0, 8.0), glm::vec2(0.0, 0.0),
            ]),
            (BlockIndex::new(2, 2), vec![
                glm::vec2(0.0, 0.0), glm::vec2(0.0, 8.0),
                glm::vec2(8.0, 8.0), glm::vec2(8.0, 0.0),
            ]),
        ],
    )]
    #[case::quad_block_aligned(
        [glm::vec2(-16.0, -16.0), glm::vec2(-16.0, 16.0), glm::vec2(16.0, 16.0), glm::vec2(16.0, -16.0)],
        [
            (BlockIndex::new(1, 1), vec![
                glm::vec2(-16.0, -16.0), glm::vec2(-16.0, 0.0),
                glm::vec2(0.0, 0.0), glm::vec2(0.0, -16.0),
            ]),
            (BlockIndex::new(2, 1), vec![
                glm::vec2(0.0, -16.0), glm::vec2(0.0, 0.0),
                glm::vec2(16.0, 0.0), glm::vec2(16.0, -16.0),
            ]),
            (BlockIndex::new(1, 2), vec![
                glm::vec2(-16.0, 0.0), glm::vec2(-16.0, 16.0),
                glm::vec2(0.0, 16.0), glm::vec2(0.0, 0.0),
            ]),
            (BlockIndex::new(2, 2), vec![
                glm::vec2(0.0, 0.0), glm::vec2(0.0, 16.0),
                glm::vec2(16.0, 16.0), glm::vec2(16.0, 0.0),
            ]),
        ],
    )]
    #[case::big_triangle(
        [glm::vec2(12.0, 8.0), glm::vec2(21.0, 26.0), glm::vec2(30.0, 17.0)],
        [
            (BlockIndex::new(2, 2), vec![
                glm::vec2(12.0, 8.0), glm::vec2(16.0, 16.0), glm::vec2(16.0, 10.0),
            ]),
            (BlockIndex::new(3, 2), vec![
                glm::vec2(16.0, 10.0), glm::vec2(16.0, 16.0), glm::vec2(28.0, 16.0),
            ]),
            (BlockIndex::new(3, 3), vec![
                glm::vec2(16.0, 16.0), glm::vec2(21.0, 26.0),
                glm::vec2(30.0, 17.0), glm::vec2(28.0, 16.0),
            ]),
        ],
    )]
    #[case::diamond(
        [glm::vec2(-8.0, 8.0), glm::vec2(8.0, 24.0), glm::vec2(24.0, 8.0), glm::vec2(8.0, -8.0)],
        [
            (BlockIndex::new(1, 2), vec![
                glm::vec2(-8.0, 8.0), glm::vec2(0.0, 16.0), glm::vec2(0.0, 0.0),
            ]),
            (BlockIndex::new(2, 3), vec![
                glm::vec2(8.0, 24.0), glm::vec2(16.0, 16.0), glm::vec2(0.0, 16.0),
            ]),
            (BlockIndex::new(3, 2), vec![
                glm::vec2(24.0, 8.0), glm::vec2(16.0, 0.0), glm::vec2(16.0, 16.0),
            ]),
            (BlockIndex::new(2, 1), vec![
                glm::vec2(8.0, -8.0), glm::vec2(0.0, 0.0), glm::vec2(16.0, 0.0),
            ]),
            (BlockIndex::new(2, 2), vec![
                glm::vec2(0.0, 0.0), glm::vec2(0.0, 16.0),
                glm::vec2(16.0, 16.0), glm::vec2(16.0, 0.0),
            ]),
        ],
    )]
    #[case::diamond_enclosing_single(
        [glm::vec2(-12.0, 8.0), glm::vec2(8.0, 28.0), glm::vec2(28.0, 8.0), glm::vec2(8.0, -12.0)],
        [
            (BlockIndex::new(1, 2), vec![
                glm::vec2(-4.0, 0.0), glm::vec2(-12.0, 8.0), glm::vec2(-4.0, 16.0),
                glm::vec2(0.0, 16.0), glm::vec2(0.0, 0.0),
            ]),
            (BlockIndex::new(1, 3), vec![
                glm::vec2(-4.0, 16.0), glm::vec2(0.0, 20.0), glm::vec2(0.0, 16.0),
            ]),
            (BlockIndex::new(2, 3), vec![
                glm::vec2(0.0, 20.0), glm::vec2(8.0, 28.0), glm::vec2(16.0, 20.0),
                glm::vec2(16.0, 16.0), glm::vec2(0.0, 16.0),
            ]),
            (BlockIndex::new(3, 3), vec![
                glm::vec2(16.0, 20.0), glm::vec2(20.0, 16.0), glm::vec2(16.0, 16.0),
            ]),
            (BlockIndex::new(3, 2), vec![
                glm::vec2(20.0, 16.0), glm::vec2(28.0, 8.0), glm::vec2(20.0, 0.0),
                glm::vec2(16.0, 0.0), glm::vec2(16.0, 16.0),
            ]),
            (BlockIndex::new(3, 1), vec![
                glm::vec2(20.0, 0.0), glm::vec2(16.0, -4.0), glm::vec2(16.0, 0.0),
            ]),
            (BlockIndex::new(2, 1), vec![
                glm::vec2(16.0, -4.0), glm::vec2(8.0, -12.0), glm::vec2(0.0, -4.0),
                glm::vec2(0.0, 0.0), glm::vec2(16.0, 0.0),
            ]),
            (BlockIndex::new(1, 1), vec![
                glm::vec2(0.0, -4.0), glm::vec2(-4.0, 0.0), glm::vec2(0.0, 0.0),
            ]),
            (BlockIndex::new(2, 2), vec![
                glm::vec2(0.0, 0.0), glm::vec2(0.0, 16.0),
                glm::vec2(16.0, 16.0), glm::vec2(16.0, 0.0),
            ]),
        ],
    )]
    #[case::thirds_slope_trapezoid(
        [glm::vec2(-4.0, 4.0), glm::vec2(2.0, 22.0), glm::vec2(14.0, 22.0), glm::vec2(20.0, 4.0)],
        [
            (BlockIndex::new(1, 2), vec![
                glm::vec2(-4.0, 4.0), glm::vec2(0.0, 16.0), glm::vec2(0.0, 4.0),
            ]),
            (BlockIndex::new(2, 2), vec![
                glm::vec2(0.0, 4.0), glm::vec2(0.0, 16.0),
                glm::vec2(16.0, 16.0), glm::vec2(16.0, 4.0),
            ]),
            (BlockIndex::new(2, 3), vec![
                glm::vec2(0.0, 16.0), glm::vec2(2.0, 22.0),
                glm::vec2(14.0, 22.0), glm::vec2(16.0, 16.0),
            ]),
            (BlockIndex::new(3, 2), vec![
                glm::vec2(20.0, 4.0), glm::vec2(16.0, 4.0), glm::vec2(16.0, 16.0),
            ]),
        ]
    )]
    #[test_log]
    fn test_contour_query<N: BlockmapNum>(
        blockmap: Blockmap<16, SortedBlockIndex>,
        #[case] contour: impl IntoIterator<Item=glm::TVec2<N>>,
        #[case] expected: impl IntoIterator<Item=(BlockIndex, Vec<glm::TVec2<N>>)>,
    ) {
        let query = blockmap.query(contour.into_iter().collect::<Vec<_>>());

        let mut actual = query
            .map(|(idx, metadata)| {
                (*idx, ContourCheck {
                    contour: VecDeque::from(metadata.contour),
                })
            })
            .collect::<BTreeMap<_, _>>();

        let expected = expected.into_iter()
            .map(|(idx, contour)| {
                (SortedBlockIndex(idx), ContourCheck {
                    contour: VecDeque::from(contour),
                })
            })
            .collect::<BTreeMap<_, _>>();

        align_contour_maps(&mut actual, &expected);
        assert_eq!(actual, expected);
    }
}