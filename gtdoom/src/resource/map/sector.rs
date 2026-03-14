use async_trait::async_trait;
use bitflags::bitflags;
use educe::Educe;
use gtether::resource::id::ResourceId;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{ResourceLoadError, ResourceLoader, ResourceReadData};
use smol::io::AsyncReadExt;
use std::time::Duration;
use tracing::debug;

use crate::resource::indexed::IndexedVec;
use crate::resource::map::MapType;
use crate::resource::string_from_bytes_until_nul_checked;
use crate::resource::texture::TextureId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialAlt {
    Special1,
    Special2,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectorSpecialLight {
    Phased,
    SequenceStart,
    Sequence(SpecialAlt),
    Flicker,
    FireFlicker,
    Strobe {
        duration: f32,
    },
    StrobeSync {
        duration: f32,
    },
    Oscillate,
}

// TODO: Move this somewhere more appropriate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardinalDirection {
    East,
    North,
    South,
    West,
}

// TODO: Move this somewhere more appropriate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardinalDirectionFull {
    East,
    North,
    South,
    West,
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

// TODO: Move this somewhere more appropriate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageType {
    Generic,
    Lava,
    Sludge,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectorSpecialDamage {
    Percent {
        value: f32,
        damage_type: DamageType,
    },
    Strobe,
    End,
    // Scrolls east
    HereticLava,
    Delayed {
        amount: u8,
    },
    InstantDeath,
}

// TODO: Move this somewhere more appropriate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoorAction {
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectorSpecialLightning {
    Outdoor,
    Indoor {
        light: u8,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectorSpecial {
    Light(SectorSpecialLight),
    Stairs(SpecialAlt),
    Wind {
        direction: CardinalDirection,
        speed: u8,
    },
    Damage(SectorSpecialDamage),
    Door {
        action: DoorAction,
        delay: Duration,
    },
    LowFriction,
    Fog,
    Hidden,
    Healing,
    Lightning(SectorSpecialLightning),
    Sky2,
    Scroll {
        direction: CardinalDirectionFull,
        speed: u8,
    },
    Carry {
        direction: CardinalDirection,
        speed: u8,
    },
    StrifeScroll,
}

impl SectorSpecial {
    fn match_carry(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Carry { direction: CardinalDirection::East, speed: 5 }),
            1 => Some(Self::Carry { direction: CardinalDirection::East, speed: 10 }),
            2 => Some(Self::Carry { direction: CardinalDirection::East, speed: 25 }),
            3 => Some(Self::Carry { direction: CardinalDirection::East, speed: 30 }),
            4 => Some(Self::Carry { direction: CardinalDirection::East, speed: 35 }),
            5 => Some(Self::Carry { direction: CardinalDirection::North, speed: 5 }),
            6 => Some(Self::Carry { direction: CardinalDirection::North, speed: 10 }),
            7 => Some(Self::Carry { direction: CardinalDirection::North, speed: 25 }),
            8 => Some(Self::Carry { direction: CardinalDirection::North, speed: 30 }),
            9 => Some(Self::Carry { direction: CardinalDirection::North, speed: 35 }),
            10 => Some(Self::Carry { direction: CardinalDirection::South, speed: 5 }),
            11 => Some(Self::Carry { direction: CardinalDirection::South, speed: 10 }),
            12 => Some(Self::Carry { direction: CardinalDirection::South, speed: 25 }),
            13 => Some(Self::Carry { direction: CardinalDirection::South, speed: 30 }),
            14 => Some(Self::Carry { direction: CardinalDirection::South, speed: 35 }),
            15 => Some(Self::Carry { direction: CardinalDirection::West, speed: 5 }),
            16 => Some(Self::Carry { direction: CardinalDirection::West, speed: 10 }),
            17 => Some(Self::Carry { direction: CardinalDirection::West, speed: 25 }),
            18 => Some(Self::Carry { direction: CardinalDirection::West, speed: 30 }),
            19 => Some(Self::Carry { direction: CardinalDirection::West, speed: 35 }),
            _ => None,
        }
    }

    fn match_wind(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Wind { direction: CardinalDirection::East, speed: 5 }),
            1 => Some(Self::Wind { direction: CardinalDirection::East, speed: 10 }),
            2 => Some(Self::Wind { direction: CardinalDirection::East, speed: 25 }),
            3 => Some(Self::Wind { direction: CardinalDirection::North, speed: 5 }),
            4 => Some(Self::Wind { direction: CardinalDirection::North, speed: 10 }),
            5 => Some(Self::Wind { direction: CardinalDirection::North, speed: 25 }),
            6 => Some(Self::Wind { direction: CardinalDirection::South, speed: 5 }),
            7 => Some(Self::Wind { direction: CardinalDirection::South, speed: 10 }),
            8 => Some(Self::Wind { direction: CardinalDirection::South, speed: 25 }),
            9 => Some(Self::Wind { direction: CardinalDirection::West, speed: 5 }),
            10 => Some(Self::Wind { direction: CardinalDirection::West, speed: 10 }),
            11 => Some(Self::Wind { direction: CardinalDirection::West, speed: 25 }),
            _ => None,
        }
    }

    fn match_scroll(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Scroll { direction: CardinalDirectionFull::North, speed: 5 }),
            1 => Some(Self::Scroll { direction: CardinalDirectionFull::North, speed: 10 }),
            2 => Some(Self::Scroll { direction: CardinalDirectionFull::North, speed: 25 }),
            3 => Some(Self::Scroll { direction: CardinalDirectionFull::East, speed: 5 }),
            4 => Some(Self::Scroll { direction: CardinalDirectionFull::East, speed: 10 }),
            5 => Some(Self::Scroll { direction: CardinalDirectionFull::East, speed: 25 }),
            6 => Some(Self::Scroll { direction: CardinalDirectionFull::South, speed: 5 }),
            7 => Some(Self::Scroll { direction: CardinalDirectionFull::South, speed: 10 }),
            8 => Some(Self::Scroll { direction: CardinalDirectionFull::South, speed: 25 }),
            9 => Some(Self::Scroll { direction: CardinalDirectionFull::West, speed: 5 }),
            10 => Some(Self::Scroll { direction: CardinalDirectionFull::West, speed: 10 }),
            11 => Some(Self::Scroll { direction: CardinalDirectionFull::West, speed: 25 }),
            12 => Some(Self::Scroll { direction: CardinalDirectionFull::NorthWest, speed: 5 }),
            13 => Some(Self::Scroll { direction: CardinalDirectionFull::NorthWest, speed: 10 }),
            14 => Some(Self::Scroll { direction: CardinalDirectionFull::NorthWest, speed: 25 }),
            15 => Some(Self::Scroll { direction: CardinalDirectionFull::NorthEast, speed: 5 }),
            16 => Some(Self::Scroll { direction: CardinalDirectionFull::NorthEast, speed: 10 }),
            17 => Some(Self::Scroll { direction: CardinalDirectionFull::NorthEast, speed: 25 }),
            18 => Some(Self::Scroll { direction: CardinalDirectionFull::SouthEast, speed: 5 }),
            19 => Some(Self::Scroll { direction: CardinalDirectionFull::SouthEast, speed: 10 }),
            20 => Some(Self::Scroll { direction: CardinalDirectionFull::SouthEast, speed: 25 }),
            21 => Some(Self::Scroll { direction: CardinalDirectionFull::SouthWest, speed: 5 }),
            22 => Some(Self::Scroll { direction: CardinalDirectionFull::SouthWest, speed: 10 }),
            23 => Some(Self::Scroll { direction: CardinalDirectionFull::SouthWest, speed: 25 }),
            _ => None,
        }
    }

    pub fn from_raw(value: u16, map_type: MapType) -> Option<Self> {
        match map_type {
            MapType::Doom => {
                match value {
                    1 => Some(Self::Light(SectorSpecialLight::Flicker)),
                    2 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 0.5 })),
                    3 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 1.0 })),
                    4 => Some(Self::Damage(SectorSpecialDamage::Strobe)),
                    5 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.1,
                        damage_type: DamageType::Generic,
                    })),
                    7 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.05,
                        damage_type: DamageType::Generic,
                    })),
                    8 => Some(Self::Light(SectorSpecialLight::Oscillate)),
                    10 => Some(Self::Door {
                        action: DoorAction::Close,
                        delay: Duration::from_secs(30),
                    }),
                    11 => Some(Self::Damage(SectorSpecialDamage::End)),
                    12 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 1.0 })),
                    13 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 0.5 })),
                    14 => Some(Self::Door {
                        action: DoorAction::Open,
                        delay: Duration::from_secs(300),
                    }),
                    16 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.2,
                        damage_type: DamageType::Generic,
                    })),
                    17 => Some(Self::Light(SectorSpecialLight::FireFlicker)),
                    _ => None,
                }
            },
            MapType::Heretic => {
                match value {
                    1 => Some(Self::Light(SectorSpecialLight::Flicker)),
                    2 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 0.5 })),
                    3 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 1.0 })),
                    4 => Some(Self::Damage(SectorSpecialDamage::HereticLava)),
                    5 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.05,
                        damage_type: DamageType::Lava,
                    })),
                    7 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.04,
                        damage_type: DamageType::Sludge,
                    })),
                    8 => Some(Self::Light(SectorSpecialLight::Oscillate)),
                    10 => Some(Self::Door {
                        action: DoorAction::Close,
                        delay: Duration::from_secs(30),
                    }),
                    12 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 1.0 })),
                    13 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 0.5 })),
                    14 => Some(Self::Door {
                        action: DoorAction::Open,
                        delay: Duration::from_secs(300),
                    }),
                    15 => Some(Self::LowFriction),
                    16 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.08,
                        damage_type: DamageType::Lava,
                    })),
                    20..=39 => Self::match_carry(value - 20),
                    40..=51 => Self::match_wind(value - 40),
                    _ => None,
                }
            },
            MapType::Hexen => {
                match value {
                    1 => Some(Self::Light(SectorSpecialLight::Phased)),
                    2 => Some(Self::Light(SectorSpecialLight::SequenceStart)),
                    3 => Some(Self::Light(SectorSpecialLight::Sequence(SpecialAlt::Special1))),
                    4 => Some(Self::Light(SectorSpecialLight::Sequence(SpecialAlt::Special2))),
                    26 => Some(Self::Stairs(SpecialAlt::Special1)),
                    27 => Some(Self::Stairs(SpecialAlt::Special2)),
                    40..=51 => Self::match_wind(value - 40),
                    198 => Some(Self::Lightning(SectorSpecialLightning::Indoor { light: 64 })),
                    199 => Some(Self::Lightning(SectorSpecialLightning::Indoor { light: 32 })),
                    200 => Some(Self::Sky2),
                    201..=224 => Self::match_scroll(value - 200),
                    _ => None,
                }
            },
            MapType::Strife => {
                match value {
                    1 => Some(Self::Light(SectorSpecialLight::Flicker)),
                    2 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 0.5 })),
                    3 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 1.0 })),
                    4 => Some(Self::Damage(SectorSpecialDamage::Strobe)),
                    5 => Some(Self::Damage(SectorSpecialDamage::Delayed { amount: 2 })),
                    7 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.05,
                        damage_type: DamageType::Generic,
                    })),
                    8 => Some(Self::Light(SectorSpecialLight::Oscillate)),
                    10 => Some(Self::Door {
                        action: DoorAction::Close,
                        delay: Duration::from_secs(30),
                    }),
                    11 => Some(Self::Damage(SectorSpecialDamage::End)),
                    12 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 0.5 })),
                    13 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 1.0 })),
                    14 => Some(Self::Door {
                        action: DoorAction::Open,
                        delay: Duration::from_secs(300),
                    }),
                    15 => Some(Self::Damage(SectorSpecialDamage::InstantDeath)),
                    16 => Some(Self::Damage(SectorSpecialDamage::Delayed { amount: 4 })),
                    17 => Some(Self::Light(SectorSpecialLight::FireFlicker)),
                    18 => Some(Self::StrifeScroll),
                    _ => None,
                }
            },
            MapType::ZDoom => {
                match value {
                    1 => Some(Self::Light(SectorSpecialLight::Phased)),
                    2 => Some(Self::Light(SectorSpecialLight::SequenceStart)),
                    3 => Some(Self::Light(SectorSpecialLight::Sequence(SpecialAlt::Special1))),
                    4 => Some(Self::Light(SectorSpecialLight::Sequence(SpecialAlt::Special2))),
                    26 => Some(Self::Stairs(SpecialAlt::Special1)),
                    27 => Some(Self::Stairs(SpecialAlt::Special2)),
                    40..=51 => Self::match_wind(value - 40),
                    65 => Some(Self::Light(SectorSpecialLight::Flicker)),
                    66 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 0.5 })),
                    67 => Some(Self::Light(SectorSpecialLight::Strobe { duration: 1.0 })),
                    68 => Some(Self::Damage(SectorSpecialDamage::Strobe)),
                    69 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.1,
                        damage_type: DamageType::Generic,
                    })),
                    71 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.05,
                        damage_type: DamageType::Generic,
                    })),
                    72 => Some(Self::Light(SectorSpecialLight::Oscillate)),
                    74 => Some(Self::Door {
                        action: DoorAction::Close,
                        delay: Duration::from_secs(30),
                    }),
                    75 => Some(Self::Damage(SectorSpecialDamage::End)),
                    76 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 0.5 })),
                    77 => Some(Self::Light(SectorSpecialLight::StrobeSync { duration: 1.0 })),
                    78 => Some(Self::Door {
                        action: DoorAction::Open,
                        delay: Duration::from_secs(300),
                    }),
                    79 => Some(Self::LowFriction),
                    80 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.2,
                        damage_type: DamageType::Generic,
                    })),
                    81 => Some(Self::Light(SectorSpecialLight::FireFlicker)),
                    82 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.05,
                        damage_type: DamageType::Lava,
                    })),
                    83 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.08,
                        damage_type: DamageType::Lava,
                    })),
                    84 => Some(Self::Damage(SectorSpecialDamage::HereticLava)),
                    85 => Some(Self::Damage(SectorSpecialDamage::Percent {
                        value: 0.04,
                        damage_type: DamageType::Sludge,
                    })),
                    87 => Some(Self::Fog),
                    105 => Some(Self::Damage(SectorSpecialDamage::Delayed { amount: 2 })),
                    115 => Some(Self::Damage(SectorSpecialDamage::InstantDeath)),
                    116 => Some(Self::Damage(SectorSpecialDamage::Delayed { amount: 4 })),
                    118 => Some(Self::StrifeScroll),
                    195 => Some(Self::Hidden),
                    196 => Some(Self::Healing),
                    197 => Some(Self::Lightning(SectorSpecialLightning::Outdoor)),
                    198 => Some(Self::Lightning(SectorSpecialLightning::Indoor { light: 64 })),
                    199 => Some(Self::Lightning(SectorSpecialLightning::Indoor { light: 32 })),
                    200 => Some(Self::Sky2),
                    201..=224 => Self::match_scroll(value - 201),
                    225..=244 => Self::match_carry(value - 225),
                    _ => None,
                }
            },
        }
    }
}

bitflags! {
    #[derive(Debug, Clone)]
    pub struct SectorSpecialFlags: u8 {
        const Damage5Percent = 0b00001;
        const Damage10Percent = 0b00010;
        const Damage20Percent = 0b00011;
        const Secret = 0b00100;
        const Friction = 0b01000;
        const PushPull = 0b10000;
    }
}

impl SectorSpecialFlags {
    pub fn from_raw(value: u16, map_type: MapType) -> Self {
        match map_type {
            MapType::Hexen | MapType::ZDoom => {
                Self::from_bits_truncate((value >> 8) as u8)
            },
            MapType::Doom => {
                Self::from_bits_truncate((value >> 5) as u8)
            },
            _ => Self::empty(),
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct SectorFlags: u16 {
        const Silent = 0b000000000001;
        const NoFallingDamage = 0b000000000010;
        const FloorDrop = 0b000000000100;
        const NoRespawn = 0b000000001000;
        const Friction = 0b000000010000;
        const PushPull = 0b000000100000;
        const SilentMove = 0b000001000000;
        const DamageTerrainEffect = 0b000010000000;
        const DamageEndGodMode = 0b000100000000;
        const DamageEndLevel = 0b001000000000;
        const DamageHazard = 0b010000000000;
        const NoAttack = 0b100000000000;
    }
}

#[derive(Debug, Clone)]
pub struct Sector {
    pub idx: usize,
    pub floor_height: i16,
    pub ceil_height: i16,
    pub floor_tex: TextureId,
    pub ceil_tex: TextureId,
    pub light_level: u8,
    pub special: Option<SectorSpecial>,
    pub special_flags: SectorSpecialFlags,
    pub flags: SectorFlags,
    pub tag: u16,
}

impl Sector {
    pub const BYTE_COUNT: usize = 26;

    async fn load(
        idx: usize,
        bytes: &[u8; Self::BYTE_COUNT],
        map_type: MapType,
    ) -> Result<Self, ResourceLoadError> {
        let floor_height = i16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let ceil_height = i16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let floor_tex_name = string_from_bytes_until_nul_checked(&bytes[4..12]);
        let ceil_tex_name = string_from_bytes_until_nul_checked(&bytes[12..20]);
        let light_level = i16::from_le_bytes(bytes[20..22].try_into().unwrap()) as u8;
        let sector_type = u16::from_le_bytes(bytes[22..24].try_into().unwrap());
        let special = SectorSpecial::from_raw(sector_type, map_type);
        let special_flags = SectorSpecialFlags::from_raw(sector_type, map_type);
        // These are only for UDMF
        let flags = SectorFlags::empty();
        let tag = i16::from_le_bytes(bytes[24..26].try_into().unwrap()) as u16;

        let floor_tex = TextureId::Flat(ResourceId::from(&floor_tex_name));
        let ceil_tex = TextureId::Flat(ResourceId::from(&ceil_tex_name));

        Ok(Self {
            idx,
            floor_height,
            ceil_height,
            floor_tex,
            ceil_tex,
            light_level,
            special,
            special_flags,
            flags,
            tag,
        })
    }
}

#[derive(Educe)]
#[educe(Deref)]
pub struct Sectors(IndexedVec<Sector>);

impl Sectors {
    #[inline]
    pub fn lump_loader(map_type: MapType) -> SectorsLumpLoader {
        SectorsLumpLoader {
            map_type,
        }
    }
}

pub struct SectorsLumpLoader {
    map_type: MapType,
}

#[async_trait]
impl ResourceLoader for SectorsLumpLoader {
    type Output = Sectors;

    async fn load(
        &self,
        mut data: ResourceReadData,
        _ctx: &ResourceLoadContext,
    ) -> Result<Box<Sectors>, ResourceLoadError> {
        let mut bytes = [0; Sector::BYTE_COUNT];
        let mut sectors = Vec::new();
        loop {
            match data.read_exact(&mut bytes).await {
                Ok(()) => sectors.push(Sector::load(
                    sectors.len(),
                    &bytes,
                    self.map_type,
                ).await?),
                Err(error) => match error.kind() {
                    std::io::ErrorKind::UnexpectedEof => break,
                    _ => return Err(ResourceLoadError::from_error(error)),
                }
            }
        }
        debug!(count = ?sectors.len(), "Loaded Sectors");
        Ok(Box::new(Sectors(sectors.into())))
    }
}