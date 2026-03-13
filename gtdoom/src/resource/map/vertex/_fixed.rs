use educe::Educe;
use fixed::traits::LossyFrom;
use fixed::types::extra::{U16, U32};
use fixed::{FixedI32, FixedI64};
use i_triangle::i_overlay::i_float::float::compatible::FloatPointCompatible;
use i_triangle::i_overlay::i_float::float::number::FloatNumber;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::resource::indexed::Indexed;
use crate::resource::map::vertex::{IndexedVertex, Vertex};

pub type FixedFloat = FixedI32<U16>;

#[derive(Educe, Debug, Clone, Copy, PartialEq, Eq)]
#[educe(Deref)]
#[repr(transparent)]
pub struct FixedVertex(glm::TVec2<FixedFloat>);

impl FixedVertex {
    pub const BYTE_COUNT: usize = 8;

    #[inline]
    pub fn from_bytes(bytes: &[u8; Self::BYTE_COUNT]) -> Self {
        Self(glm::vec2(
            FixedFloat::from_le_bytes(bytes[0..4].try_into().unwrap()),
            FixedFloat::from_le_bytes(bytes[4..8].try_into().unwrap()),
        ))
    }

    #[inline]
    pub fn to_3d(self, z: f32) -> glm::TVec3<f32> {
        glm::vec3(
            f32::lossy_from(self.x),
            f32::lossy_from(self.y),
            z,
        )
    }

    #[inline]
    pub fn equal_eps(&self, other: &Self, epsilon: FixedFloat) -> bool {
        (self.x - other.x).abs() < epsilon && (self.y - other.y).abs() < epsilon
    }
}

impl From<Vertex> for FixedVertex {
    #[inline]
    fn from(value: Vertex) -> Self {
        Self(glm::vec2(
            FixedFloat::from(value.x),
            FixedFloat::from(value.y),
        ))
    }
}

impl TryFrom<FixedVertex> for Vertex {
    type Error = ();

    fn try_from(value: FixedVertex) -> Result<Self, Self::Error> {
        const HALF: FixedFloat = FixedFloat::lit("0.5");
        const EPSILON: FixedFloat = FixedFloat::lit("0.0001");

        let whole_vertex = Vertex::from(glm::vec2(
            (value.x + HALF).to_num::<i16>(),
            (value.y + HALF).to_num::<i16>(),
        ));
        if value.equal_eps(&FixedVertex::from(whole_vertex), EPSILON) {
            Ok(whole_vertex)
        } else {
            Err(())
        }
    }
}

type FixedFloatWrapperInner = FixedI64<U32>;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FixedFloatWrapper(FixedFloatWrapperInner);

macro_rules! impl_wrapped_trait_unary {
    ($trait_name:ident, $fn_name:ident) => {
        impl $trait_name for FixedFloatWrapper {
            type Output = Self;

            #[inline]
            fn $fn_name(self) -> Self::Output {
                Self(self.0.$fn_name())
            }
        }
    };
}

macro_rules! impl_wrapped_trait_binary {
    ($trait_name:ident, $fn_name:ident) => {
        impl $trait_name for FixedFloatWrapper {
            type Output = Self;

            #[inline]
            fn $fn_name(self, rhs: Self) -> Self::Output {
                Self(self.0.$fn_name(rhs.0))
            }
        }
    };
}

impl_wrapped_trait_binary!(Mul, mul);
impl_wrapped_trait_binary!(Add, add);
impl_wrapped_trait_binary!(Sub, sub);
impl_wrapped_trait_binary!(Div, div);
impl_wrapped_trait_unary!(Neg, neg);

impl Display for FixedFloatWrapper {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FloatNumber for FixedFloatWrapper {
    const MAX: Self = Self(FixedFloatWrapperInner::MAX);
    const MIN: Self = Self(FixedFloatWrapperInner::MIN);

    #[inline(always)] fn abs(self) -> Self { Self(self.0.abs()) }
    #[inline(always)] fn sqrt(self) -> Self { Self(cordic::sqrt(self.0)) }
    #[inline(always)] fn from_usize(value: usize) -> Self { Self(FixedFloatWrapperInner::from_num(value)) }
    #[inline(always)] fn from_i32(value: i32) -> Self { Self(FixedFloatWrapperInner::from_num(value)) }
    #[inline(always)] fn from_i64(value: i64) -> Self { Self(FixedFloatWrapperInner::from_num(value)) }
    #[inline(always)] fn from_float(value: f64) -> Self { Self(FixedFloatWrapperInner::from_num(value)) }
    #[inline(always)] fn max(self, other: Self) -> Self { Self(self.0.max(other.0)) }
    #[inline(always)] fn min(self, other: Self) -> Self { Self(self.0.min(other.0)) }
    #[inline(always)] fn log2(self) -> Self {
        // TODO: Better log2() function
        Self::from_float(self.to_f64().log2())
    }
    #[inline(always)] fn cos(self) -> Self { Self(cordic::cos(self.0)) }
    #[inline(always)] fn sin(self) -> Self { Self(cordic::sin(self.0)) }
    #[inline(always)] fn tan(self) -> Self { Self(cordic::tan(self.0)) }
    #[inline(always)] fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = cordic::sin_cos(self.0);
        (Self(sin), Self(cos))
    }
    #[inline(always)] fn acos(self) -> Self { Self(cordic::acos(self.0)) }
    #[inline(always)] fn asin(self) -> Self { Self(cordic::asin(self.0)) }
    #[inline(always)] fn to_i32(self) -> i32 { self.0.to_num() }
    #[inline(always)] fn to_usize(self) -> usize { self.0.to_num() }
    #[inline(always)] fn to_f64(self) -> f64 { self.0.to_num() }
    #[inline(always)] fn bit_width() -> u8 { 32 }
}

impl FloatPointCompatible<FixedFloatWrapper> for FixedVertex {
    #[inline]
    fn from_xy(x: FixedFloatWrapper, y: FixedFloatWrapper) -> Self {
        Self(glm::vec2(
            FixedFloat::from_num(x.0),
            FixedFloat::from_num(y.0),
        ))
    }

    #[inline]
    fn x(&self) -> FixedFloatWrapper {
        FixedFloatWrapper(FixedFloatWrapperInner::from(self.x))
    }

    #[inline]
    fn y(&self) -> FixedFloatWrapper {
        FixedFloatWrapper(FixedFloatWrapperInner::from(self.y))
    }
}

pub type IndexedFixedVertex = Indexed<FixedVertex, u32>;

impl From<IndexedVertex> for IndexedFixedVertex {
    #[inline]
    fn from(value: IndexedVertex) -> Self {
        let (inner, idx) = value.into_parts();
        Self::new(FixedVertex::from(inner), idx as u32)
    }
}