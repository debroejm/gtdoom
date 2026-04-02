use glm::Scalar;
use num_traits::{AsPrimitive, Bounded, One, Zero};
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::ops::{BitOr, BitOrAssign, Neg};

#[cfg(debug_assertions)]
use crate::resource::map::vertex::FixedFloat;

pub mod line;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis2D {
    X,
    Y,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction2D {
    Right,
    Left,
    Up,
    Down,
}

impl Direction2D {
    pub const X_POSITIVE: Self = Self::Right;
    pub const X_NEGATIVE: Self = Self::Left;
    pub const Y_POSITIVE: Self = Self::Up;
    pub const Y_NEGATIVE: Self = Self::Down;

    #[inline]
    pub fn axis(self) -> Axis2D {
        match self {
            Self::Right | Self::Left => Axis2D::X,
            Self::Up | Self::Down => Axis2D::Y,
        }
    }

    #[inline]
    pub fn unit_vec<N: One + Zero + Neg<Output=N>>(self) -> glm::TVec2<N> {
        match self {
            Self::Right => glm::TVec2::<N>::new(N::one(), N::zero()),
            Self::Left => glm::TVec2::<N>::new(N::one().neg(), N::zero()),
            Self::Up => glm::TVec2::<N>::new(N::zero(), N::one()),
            Self::Down => glm::TVec2::<N>::new(N::zero(), N::one().neg()),
        }
    }

    #[inline]
    pub fn flip(self) -> Self {
        match self {
            Self::Right => Self::Left,
            Self::Left => Self::Right,
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }

    #[inline]
    pub fn rotate_cw(self) -> Self {
        match self {
            Self::Right => Self::Down,
            Self::Down => Self::Left,
            Self::Left => Self::Up,
            Self::Up => Self::Right,
        }
    }

    #[inline]
    pub fn rotate_ccw(self) -> Self {
        match self {
            Self::Right => Self::Up,
            Self::Up => Self::Left,
            Self::Left => Self::Down,
            Self::Down => Self::Right,
        }
    }

    #[inline]
    pub fn cw_to(self, other: Self) -> CwToIter {
        CwToIter {
            curr: self,
            end: other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CwToIter {
    curr: Direction2D,
    end: Direction2D,
}

impl Iterator for CwToIter {
    type Item = Direction2D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr != self.end {
            self.curr = self.curr.rotate_cw();
            Some(self.curr)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(3))
    }
}

impl FusedIterator for CwToIter {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompassDirection {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl CompassDirection {
    pub const X_POSITIVE: Self = Self::East;
    pub const X_NEGATIVE: Self = Self::West;
    pub const Y_POSITIVE: Self = Self::North;
    pub const Y_NEGATIVE: Self = Self::South;
    pub const X_POS_Y_POS: Self = Self::NorthEast;
    pub const X_POS_Y_NEG: Self = Self::SouthEast;
    pub const X_NEG_Y_POS: Self = Self::NorthWest;
    pub const X_NEG_Y_NEG: Self = Self::SouthWest;

    #[inline]
    pub fn to_2d_cw(self) -> Direction2D {
        match self {
            Self::North | Self::NorthWest => Direction2D::Up,
            Self::East | Self::NorthEast => Direction2D::Right,
            Self::South | Self::SouthEast => Direction2D::Down,
            Self::West | Self::SouthWest => Direction2D::Left,
        }
    }

    #[inline]
    pub fn to_2d_ccw(self) -> Direction2D {
        match self {
            Self::North | Self::NorthEast => Direction2D::Up,
            Self::East | Self::SouthEast => Direction2D::Right,
            Self::South | Self::SouthWest => Direction2D::Down,
            Self::West | Self::NorthWest => Direction2D::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aabb<N> {
    pub min: glm::TVec2<N>,
    pub max: glm::TVec2<N>,
}

impl<N: Scalar> Aabb<N> {
    #[inline]
    pub fn contains(&self, p: &glm::TVec2<N>) -> bool
    where
        N: PartialOrd,
    {
        p >= &self.min && p < &self.max
    }

    #[inline]
    pub fn contains_inclusive(&self, p: &glm::TVec2<N>) -> bool
    where
        N: PartialOrd,
    {
        p >= &self.min && p <= &self.max
    }

    #[inline]
    pub fn contains_exclusive(&self, p: &glm::TVec2<N>) -> bool
    where
        N: PartialOrd,
    {
        p > &self.min && p < &self.max
    }

    #[inline]
    pub fn cast<N2>(&self) -> Aabb<N2>
    where
        N2: Scalar + Copy + 'static,
        N: AsPrimitive<N2>,
    {
        Aabb {
            min: self.min.map(|v| v.as_()),
            max: self.max.map(|v| v.as_()),
        }
    }

    #[inline]
    pub fn map<N2, F>(self, mut f: F) -> Aabb<N2>
    where
        N2: Scalar,
        F: FnMut(N) -> N2,
    {
        Aabb {
            min: self.min.map(&mut f),
            max: self.max.map(&mut f),
        }
    }
}

impl<N: Scalar + PartialOrd + Ord> BitOr for Aabb<N> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            min: self.min.zip_map(&rhs.min, |a, b| a.min(b)),
            max: self.max.zip_map(&rhs.max, |a, b| a.max(b)),
        }
    }
}

impl<'a, N: Scalar + PartialOrd> BitOrAssign<&'a glm::TVec2<N>> for Aabb<N> {
    fn bitor_assign(&mut self, rhs: &'a glm::TVec2<N>) {
        if rhs.x < self.min.x {
            self.min.x = rhs.x.clone();
        }
        if rhs.y < self.min.y {
            self.min.y = rhs.y.clone();
        }
        if rhs.x > self.max.x {
            self.max.x = rhs.x.clone();
        }
        if rhs.y > self.max.y {
            self.max.y = rhs.y.clone();
        }
    }
}

impl<N: Scalar + PartialOrd> BitOrAssign<glm::TVec2<N>> for Aabb<N> {
    #[inline]
    fn bitor_assign(&mut self, rhs: glm::TVec2<N>) {
        *self |= &rhs;
    }
}

impl<'a, N: Scalar + PartialOrd + Bounded> FromIterator<&'a glm::TVec2<N>> for Aabb<N> {
    fn from_iter<T: IntoIterator<Item=&'a glm::TVec2<N>>>(iter: T) -> Self {
        let mut aabb = Aabb {
            min: glm::TVec2::from_element(N::max_value()),
            max: glm::TVec2::from_element(N::min_value()),
        };
        for p in iter.into_iter() {
            aabb |= p;
        }
        debug_assert!(aabb.min < aabb.max);
        aabb
    }
}

impl<N: Scalar + PartialOrd + Bounded> FromIterator<glm::TVec2<N>> for Aabb<N> {
    fn from_iter<T: IntoIterator<Item=glm::TVec2<N>>>(iter: T) -> Self {
        let mut aabb = Aabb {
            min: glm::TVec2::from_element(N::max_value()),
            max: glm::TVec2::from_element(N::min_value()),
        };
        for p in iter.into_iter() {
            aabb |= p;
        }
        debug_assert!(aabb.min < aabb.max);
        aabb
    }
}

#[cfg(debug_assertions)]
pub fn contour_angle<'a>(
    contour: impl IntoIterator<Item=&'a glm::TVec2<FixedFloat>>,
) -> FixedFloat {
    const TWO: FixedFloat = FixedFloat::lit("2.0");

    let mut iter = contour.into_iter();
    let first = match iter.next() {
        Some(v) => v,
        None => return FixedFloat::ZERO
    };
    let second = match iter.next() {
        Some(v) => v,
        None => return FixedFloat::ZERO
    };

    let mut total_angle = FixedFloat::ZERO;

    let mut prev = second;
    let mut prev_angle = cordic::atan2(second.y - first.y, second.x - first.x);
    for next in iter.chain([first, second]) {
        let next_angle = cordic::atan2(next.y - prev.y, next.x - prev.x);
        let mut diff = prev_angle - next_angle;
        if diff < -FixedFloat::PI {
            diff += TWO * FixedFloat::PI;
        } else if diff > FixedFloat::PI {
            diff -= TWO * FixedFloat::PI;
        }
        total_angle += diff;
        prev = next;
        prev_angle = next_angle;
    }

    total_angle
}