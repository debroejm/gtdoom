use num_traits::bounds::UpperBounded;
use num_traits::AsPrimitive;
use std::fmt::{Debug, Formatter};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use educe::Educe;

#[derive(Educe, Debug, Clone, Copy, PartialEq, Eq)]
#[educe(Deref)]
pub struct Indexed<T, I> {
    #[educe(Deref)]
    inner: T,
    pub idx: I,
}

impl<T, I> Indexed<T, I> {
    pub fn new(inner: T, idx: I) -> Self {
        Self {
            inner,
            idx,
        }
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.inner
    }

    #[inline]
    pub fn into_parts(self) -> (T, I) {
        (self.inner, self.idx)
    }
}

impl<T, I> From<(T, I)> for Indexed<T, I> {
    #[inline]
    fn from(parts: (T, I)) -> Self {
        Self {
            inner: parts.0,
            idx: parts.1,
        }
    }
}

pub struct IndexedVec<T, I = usize> {
    inner: Arc<Vec<T>>,
    _idx_type: PhantomData<I>,
}

impl<T, I> IndexedVec<T, I> {
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, I> IndexedVec<T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    pub fn get(&self, idx: usize) -> Option<IndexedRef<'_, T, I>> {
        if idx < self.inner.len() {
            Some(IndexedRef {
                inner: &self.inner,
                idx: idx.as_(),
            })
        } else {
            None
        }
    }

    #[inline]
    pub fn iter(&self) -> IndexedIter<'_, T, I> {
        IndexedIter {
            indexed: self,
            idx: 0,
            r_idx: self.inner.len(),
        }
    }
}

impl<T, I> From<Vec<T>> for IndexedVec<T, I>
where
    I: AsPrimitive<usize> + UpperBounded,
    usize: AsPrimitive<I>,
{
    #[inline]
    fn from(value: Vec<T>) -> Self {
        {
            let len = value.len();
            let max = I::max_value().as_();
            assert!(len <= max, "input vec too large");
        }
        Self {
            inner: Arc::new(value),
            _idx_type: PhantomData,
        }
    }
}

impl<T, I> From<Arc<Vec<T>>> for IndexedVec<T, I>
where
    I: AsPrimitive<usize> + UpperBounded,
    usize: AsPrimitive<I>,
{
    #[inline]
    fn from(value: Arc<Vec<T>>) -> Self {
        {
            let len = value.len();
            let max = I::max_value().as_();
            assert!(len <= max, "input vec too large");
        }
        Self {
            inner: value,
            _idx_type: PhantomData,
        }
    }
}

#[derive(Educe)]
#[educe(Clone, Copy)]
pub struct IndexedRef<'a, T, I = usize>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    inner: &'a Arc<Vec<T>>,
    idx: I,
}

impl<'a, T, I> Deref for IndexedRef<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner[self.idx.as_()]
    }
}

impl<'a, T, I> Debug for IndexedRef<'a, T, I>
where
    T: Debug,
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedRef")
            .field("idx", &self.idx.as_())
            .field("ref", self.deref())
            .finish()
    }
}

impl<'a, T, I> IndexedRef<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    #[inline]
    pub fn index(this: &Self) -> I {
        this.idx
    }

    #[inline]
    pub fn to_arc(self) -> IndexedArc<T, I> {
        IndexedArc {
            inner: self.inner.clone(),
            idx: self.idx,
        }
    }
}

#[derive(Educe)]
#[educe(Clone)]
pub struct IndexedArc<T, I = usize>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    inner: Arc<Vec<T>>,
    idx: I,
}

impl<T, I> Deref for IndexedArc<T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner[self.idx.as_()]
    }
}

impl<T, I> Debug for IndexedArc<T, I>
where
    T: Debug,
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedArc")
            .field("idx", &self.idx.as_())
            .field("ref", self.deref())
            .finish()
    }
}

impl<'a, T, I> From<IndexedRef<'a, T, I>> for IndexedArc<T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    #[inline]
    fn from(value: IndexedRef<'a, T, I>) -> Self {
        value.to_arc()
    }
}

impl<T, I> IndexedArc<T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    #[inline]
    pub fn index(this: &Self) -> I {
        this.idx
    }

    #[inline]
    pub fn as_ref(&self) -> IndexedRef<'_, T, I> {
        IndexedRef {
            inner: &self.inner,
            idx: self.idx,
        }
    }
}

pub struct IndexedIter<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    indexed: &'a IndexedVec<T, I>,
    idx: usize,
    r_idx: usize,
}

impl<'a, T, I> Iterator for IndexedIter<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    type Item = IndexedRef<'a, T, I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.r_idx {
            let b_ref = IndexedRef {
                inner: &self.indexed.inner,
                idx: self.idx.as_(),
            };
            self.idx += 1;
            Some(b_ref)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.r_idx - self.idx;
        (rem, Some(rem))
    }
}

impl<'a, T, I> DoubleEndedIterator for IndexedIter<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.r_idx > self.idx {
            self.r_idx -= 1;
            let b_ref = IndexedRef {
                inner: &self.indexed.inner,
                idx: self.r_idx.as_(),
            };
            Some(b_ref)
        } else {
            None
        }
    }
}

impl<'a, T, I> ExactSizeIterator for IndexedIter<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{}
impl<'a, T, I> FusedIterator for IndexedIter<'a, T, I>
where
    I: AsPrimitive<usize>,
    usize: AsPrimitive<I>,
{}