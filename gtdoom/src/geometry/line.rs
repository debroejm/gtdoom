use glm::Scalar;
use num_traits::{Num, Signed};

use crate::geometry::Axis2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Line<N> {
    pub a: glm::TVec2<N>,
    pub b: glm::TVec2<N>,
}

impl<N> Line<N> {
    #[inline]
    pub fn new(a: impl Into<glm::TVec2<N>>, b: impl Into<glm::TVec2<N>>) -> Self {
        Self {
            a: a.into(),
            b: b.into(),
        }
    }

    #[inline]
    pub fn from_raw(a_x: N, a_y: N, b_x: N, b_y: N) -> Self {
        Self {
            a: glm::TVec2::<N>::new(a_x, a_y),
            b: glm::TVec2::<N>::new(b_x, b_y),
        }
    }
}

impl<N: PartialOrd + Scalar> Line<N> {
    pub fn ordered_points_by_x(&self) -> (&glm::TVec2<N>, &glm::TVec2<N>, bool) {
        if self.a.x > self.b.x {
            (&self.b, &self.a, true)
        } else {
            (&self.a, &self.b, false)
        }
    }

    pub fn ordered_points_by_y(&self) -> (&glm::TVec2<N>, &glm::TVec2<N>, bool) {
        if self.a.y > self.b.y {
            (&self.b, &self.a, true)
        } else {
            (&self.a, &self.b, false)
        }
    }
}

impl<N: Num + Scalar + Copy> Line<N> {
    pub fn to_function(&self) -> LinearFunction<N> {
        if self.a.x == self.b.x {
            LinearFunction::Vertical(VerticalFunction { x: self.a.x })
        } else if self.a.y == self.b.y {
            LinearFunction::Horizontal(HorizontalFunction { y: self.a.y })
        } else {
            let slope = (self.b.y - self.a.y) / (self.b.x - self.a.x);
            let intercept = self.a.y - (slope * self.a.x);
            LinearFunction::Sloped(SlopeInterceptFunction {
                slope,
                intercept,
            })
        }
    }

    pub fn contains_eps(&self, other: &Self, eps: N) -> bool
    where
        N: PartialOrd + Signed,
    {
        let l1 = self.to_function();
        let l2 = other.to_function();

        if l1.eq_eps(&l2, eps) {
            match l1 {
                // adjust the "inner" line's points by `eps` to perform epsilon comparison
                LinearFunction::Vertical(_) => {
                    let (l1_a, l1_b, _) = self.ordered_points_by_y();
                    let (l2_a, l2_b, _) = other.ordered_points_by_y();
                    (l2_a.y + eps >= l1_a.y) && (l2_b.y - eps <= l1_b.y)
                },
                LinearFunction::Horizontal(_) | LinearFunction::Sloped(_) => {
                    let (l1_a, l1_b, _) = self.ordered_points_by_x();
                    let (l2_a, l2_b, _) = other.ordered_points_by_x();
                    (l2_a.x + eps >= l1_a.x) && (l2_b.x - eps <= l1_b.x)
                },
            }
        } else {
            false
        }
    }

    #[inline]
    pub fn contains(&self, other: &Self) -> bool
    where
        N: PartialOrd + Signed,
    {
        self.contains_eps(other, N::zero())
    }
}

impl<N1: Scalar> Line<N1> {
    pub fn from_line<N2>(line: Line<N2>) -> Self
    where
        N1: From<N2>,
        N2: Scalar,
    {
        let [a_x, a_y] = <[N2; 2]>::from(line.a);
        let [b_x, b_y] = <[N2; 2]>::from(line.b);
        Self {
            a: glm::vec2(a_x.into(), a_y.into()),
            b: glm::vec2(b_x.into(), b_y.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerticalFunction<N> {
    pub x: N,
}

impl<N: Num + Signed + PartialOrd + Copy> VerticalFunction<N> {
    #[inline]
    pub fn eq_eps(&self, other: &Self, eps: N) -> bool {
        (self.x - other.x).abs() < eps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HorizontalFunction<N> {
    pub y: N,
}

impl<N: Num + Signed + PartialOrd + Copy> HorizontalFunction<N> {
    #[inline]
    pub fn eq_eps(&self, other: &Self, eps: N) -> bool {
        (self.y - other.y).abs() < eps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlopeInterceptFunction<N> {
    pub slope: N,
    pub intercept: N,
}

impl<N: Num + Copy> SlopeInterceptFunction<N> {
    #[inline]
    pub fn x(&self, y: N) -> N {
        (y - self.intercept) / self.slope
    }

    #[inline]
    pub fn y(&self, x: N) -> N {
        (self.slope * x) + self.intercept
    }
}

impl<N: Num + Signed + PartialOrd + Copy> SlopeInterceptFunction<N> {
    #[inline]
    pub fn eq_eps(&self, other: &Self, eps: N) -> bool {
        (self.slope - other.slope).abs() < eps
            && (self.intercept - other.intercept).abs() < eps
    }

    #[inline]
    pub fn primary_axis(&self) -> Axis2D {
        if self.slope.abs() < N::one() {
            Axis2D::X
        } else {
            Axis2D::Y
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearFunction<N> {
    Vertical(VerticalFunction<N>),
    Horizontal(HorizontalFunction<N>),
    Sloped(SlopeInterceptFunction<N>),
}

impl<N: Num + Signed + PartialOrd + Copy> LinearFunction<N> {
    #[inline]
    pub fn eq_eps(&self, other: &Self, eps: N) -> bool {
        match (self, other) {
            (Self::Vertical(f1), Self::Vertical(f2))
                => f1.eq_eps(f2, eps),
            (Self::Horizontal(f1), Self::Horizontal(f2))
                => f1.eq_eps(f2, eps),
            (Self::Sloped(f1), Self::Sloped(f2))
                => f1.eq_eps(f2, eps),
            _ => false,
        }
    }
}