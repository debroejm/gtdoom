use glm::Scalar;
use num_traits::{FromPrimitive, NumAssign, Signed, ToPrimitive};
use std::iter::FusedIterator;

use crate::geometry::{Aabb, Direction2D};
use crate::resource::map::blockmap::query::{BlockmapQuery, QueryResult};

pub mod query;

pub trait BlockmapNum: NumAssign + Signed + Scalar + PartialOrd + FromPrimitive + ToPrimitive + Copy {}
impl<T: NumAssign + Signed + Scalar + PartialOrd + FromPrimitive + ToPrimitive + Copy> BlockmapNum for T {}

pub type BlockIndex = glm::TVec2<u32>;

#[derive(Debug, Clone, Copy)]
pub struct BlockmapDimensions<const BLOCK_SIZE: u32> {
    bounds: Aabb<i32>,
    block_count: glm::TVec2<u32>,
}

impl<const BLOCK_SIZE: u32> BlockmapDimensions<BLOCK_SIZE> {
    pub fn new(
        offset: glm::TVec2<i32>,
        size: glm::TVec2<u32>,
    ) -> Self {
        let bounds = Aabb {
            min: offset,
            max: offset + size.cast::<i32>(),
        };
        let block_count = size.map(|dim| {
            ((dim as f32) / (BLOCK_SIZE as f32)).ceil() as u32
        });
        Self {
            bounds,
            block_count,
        }
    }

    #[inline(always)]
    pub fn bounds(&self) -> &Aabb<i32> {
        &self.bounds
    }

    pub fn convert_bounds<N: BlockmapNum>(&self) -> Aabb<N> {
        Aabb {
            min: self.bounds.min.map(|v| N::from_i32(v).expect("overflow")),
            max: self.bounds.max.map(|v| N::from_i32(v).expect("overflow")),
        }
    }

    #[inline(always)]
    pub fn block_count(&self) -> &glm::TVec2<u32> {
        &self.block_count
    }

    #[inline]
    pub fn total_block_count(&self) -> usize {
        self.block_count.x as usize * self.block_count.y as usize
    }

    pub fn get_index<N: BlockmapNum>(&self, pos: &glm::TVec2<N>) -> Option<BlockIndex> {
        let bounds = self.convert_bounds();
        if bounds.contains(pos) {
            let adjusted = pos - bounds.min;
            let block_size = N::from_u32(BLOCK_SIZE).expect("overflow");
            let idx = (adjusted / block_size).map(|v| {
                // cut off any partial bits to get the index
                v.to_u32().expect("overflow")
            });
            Some(idx)
        } else {
            None
        }
    }

    pub fn get_indices_inclusive<N: BlockmapNum>(&self, pos: &glm::TVec2<N>) -> std::vec::IntoIter<BlockIndex> {
        if let Some(idx) = self.get_index(pos) {
            let bounds = self.bounds_for_index(&idx).unwrap();
            let mut indices = vec![idx];
            if pos.x == bounds.min.x && idx.x > 0 {
                indices.push(BlockIndex::new(idx.x - 1, idx.y));
            }
            if pos.y == bounds.min.y && idx.y > 0 {
                indices.push(BlockIndex::new(idx.x, idx.y - 1));
            }
            if pos == &bounds.min && idx.x > 0 && idx.y > 0 {
                indices.push(BlockIndex::new(idx.x - 1, idx.y - 1));
            }
            indices.into_iter()
        } else {
            vec![].into_iter()
        }
    }

    #[inline]
    pub fn contains_index(&self, idx: &BlockIndex) -> bool {
        idx < &self.block_count
    }

    pub fn bounds_for_index<N: BlockmapNum>(&self, idx: &BlockIndex) -> Option<Aabb<N>> {
        if self.contains_index(idx) {
            let min = self.bounds.min + (idx * BLOCK_SIZE).cast::<i32>();
            let max = min + glm::TVec2::<i32>::from_element(BLOCK_SIZE as i32);
            Some(Aabb {
                min: min.map(|v| N::from_i32(v).expect("overflow")),
                max: max.map(|v| N::from_i32(v).expect("overflow")),
            })
        } else {
            None
        }
    }

    pub fn checked_index_neighbor(&self, idx: &BlockIndex, dir: Direction2D) -> Option<BlockIndex> {
        match dir {
            Direction2D::Right => if idx.x < (self.block_count.x - 1) {
                Some(idx + glm::vec2(1, 0))
            } else {
                None
            }
            Direction2D::Left => if idx.x > 0 {
                Some(idx - glm::vec2(1, 0))
            } else {
                None
            }
            Direction2D::Up => if idx.y < (self.block_count.y - 1) {
                Some(idx + glm::vec2(0, 1))
            } else {
                None
            }
            Direction2D::Down => if idx.y > 0 {
                Some(idx - glm::vec2(0, 1))
            } else {
                None
            }
        }
    }
}

pub struct Blockmap<const BLOCK_SIZE: u32, T> {
    dims: BlockmapDimensions<BLOCK_SIZE>,
    data: Vec<T>,
}

impl<const BLOCK_SIZE: u32, T> Blockmap<BLOCK_SIZE, T> {
    pub fn new(dims: BlockmapDimensions<BLOCK_SIZE>) -> Self
    where
        T: Default,
    {
        Self::from_fn(dims, |_| T::default())
    }

    pub fn from_fn<F>(dims: BlockmapDimensions<BLOCK_SIZE>, f: F) -> Self
    where
        F: FnMut(BlockIndex) -> T,
    {
        let mut f = f;
        let capacity = dims.total_block_count();
        // TODO: use Vec::from_fn() when that leaves nightly
        let mut data = Vec::with_capacity(capacity);
        for idx in 0..capacity {
            let idx = BlockIndex::new(
                (idx % dims.block_count.x as usize) as u32,
                (idx / dims.block_count.x as usize) as u32,
            );
            data.push(f(idx));
        }
        Self {
            dims,
            data,
        }
    }

    pub fn from_parts(dims: BlockmapDimensions<BLOCK_SIZE>, data: Vec<T>) -> Self {
        debug_assert!(data.len() == dims.total_block_count());
        Self {
            dims,
            data,
        }
    }

    #[inline(always)]
    pub fn dimensions(&self) -> &BlockmapDimensions<BLOCK_SIZE> {
        &self.dims
    }

    fn get_index(&self, idx: &BlockIndex) -> Option<usize> {
        if idx.x < self.dims.block_count.x && idx.y < self.dims.block_count.y {
            Some(idx.y as usize * self.dims.block_count.x as usize + idx.x as usize)
        } else {
            None
        }
    }

    #[inline]
    pub fn get(&self, idx: &BlockIndex) -> Option<&T> {
        self.get_index(idx).and_then(|idx| self.data.get(idx))
    }

    #[inline]
    pub fn at<N: BlockmapNum>(&self, pos: &glm::TVec2<N>) -> Option<&T> {
        let idx = self.dims.get_index(pos)?;
        self.get(&idx)
    }

    #[inline]
    pub fn get_mut(&mut self, idx: &BlockIndex) -> Option<&mut T> {
        self.get_index(idx).and_then(|idx| self.data.get_mut(idx))
    }

    #[inline]
    pub fn at_mut<N: BlockmapNum>(&mut self, pos: &glm::TVec2<N>) -> Option<&mut T> {
        let idx = self.dims.get_index(pos)?;
        self.get_mut(&idx)
    }

    #[inline]
    pub fn query<Q>(&self, query: Q) -> QueryResult<'_, BLOCK_SIZE, T, Q>
    where
        Q: BlockmapQuery<BLOCK_SIZE>,
    {
        let iter = query.index_query(&self.dims);
        QueryResult {
            blockmap: self,
            iter,
        }
    }

    #[inline]
    pub fn iter(&self) -> BlockmapIter<'_, BLOCK_SIZE, T> {
        self.into_iter()
    }

    #[inline]
    pub fn values(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
}

impl<'a, const BLOCK_SIZE: u32, T> IntoIterator for &'a Blockmap<BLOCK_SIZE, T> {
    type Item = (BlockIndex, &'a T);
    type IntoIter = BlockmapIter<'a, BLOCK_SIZE, T>;

    fn into_iter(self) -> Self::IntoIter {
        BlockmapIter {
            dims: self.dims,
            idx: 0,
            iter: self.data.iter(),
        }
    }
}

#[derive(Clone)]
pub struct BlockmapIter<'a, const BLOCK_SIZE: u32, T> {
    dims: BlockmapDimensions<BLOCK_SIZE>,
    idx: usize,
    iter: std::slice::Iter<'a, T>,
}

impl<'a, const BLOCK_SIZE: u32, T> Iterator for BlockmapIter<'a, BLOCK_SIZE, T> {
    type Item = (BlockIndex, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        let idx = BlockIndex::new(
            (self.idx % self.dims.block_count.x as usize) as u32,
            (self.idx / self.dims.block_count.x as usize) as u32,
        );
        self.idx += 1;
        Some((idx, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, const BLOCK_SIZE: u32, T> ExactSizeIterator for BlockmapIter<'a, BLOCK_SIZE, T> {}
impl<'a, const BLOCK_SIZE: u32, T> FusedIterator for BlockmapIter<'a, BLOCK_SIZE, T> {}

impl<const BLOCK_SIZE: u32, T> IntoIterator for Blockmap<BLOCK_SIZE, T> {
    type Item = (BlockIndex, T);
    type IntoIter = BlockmapIntoIter<BLOCK_SIZE, T>;

    fn into_iter(self) -> Self::IntoIter {
        BlockmapIntoIter {
            dims: self.dims,
            idx: 0,
            iter: self.data.into_iter(),
        }
    }
}

pub struct BlockmapIntoIter<const BLOCK_SIZE: u32, T> {
    dims: BlockmapDimensions<BLOCK_SIZE>,
    idx: usize,
    iter: std::vec::IntoIter<T>,
}

impl<const BLOCK_SIZE: u32, T> Iterator for BlockmapIntoIter<BLOCK_SIZE, T> {
    type Item = (BlockIndex, T);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        let idx = BlockIndex::new(
            (self.idx % self.dims.block_count.x as usize) as u32,
            (self.idx / self.dims.block_count.x as usize) as u32,
        );
        self.idx += 1;
        Some((idx, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<const BLOCK_SIZE: u32, T> ExactSizeIterator for BlockmapIntoIter<BLOCK_SIZE, T> {}
impl<const BLOCK_SIZE: u32, T> FusedIterator for BlockmapIntoIter<BLOCK_SIZE, T> {}