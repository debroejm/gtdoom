use async_trait::async_trait;
use bitflags::bitflags;
use educe::Educe;
use gtether::render::Renderer;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::{Resource, ResourceLoadError, ResourceLoader, ResourceReadData};
use smol::io::{AsyncRead, AsyncReadExt};
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

use crate::resource::indexed::{IndexedVec, IndexedArc};
use crate::resource::map::sidedef::{Sidedef, Sidedefs};
use crate::resource::map::vertex::{Vertex, Vertices};
use crate::resource::map::{MapFormat, MapType};

// TODO: Find a better home
#[derive(Debug, Clone)]
pub enum ActionSpecial {

}

#[derive(thiserror::Error, Debug)]
pub enum LinedefLoadError {
    #[error("no vertex found for idx: {0}")]
    MissingVertex(u16),
    #[error("no sidedef found for idx: {0}")]
    MissingSidedef(u16),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

impl From<LinedefLoadError> for ResourceLoadError {
    #[inline]
    fn from(value: LinedefLoadError) -> Self {
        ResourceLoadError::from_error(value)
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct LinedefDoomFlags: u16 {
        const Blocking = 0x0001;
        const BlockMonsters = 0x0002;
        const TwoSided = 0x0004;
        const DontPegTop = 0x0008;
        const DontPegBottom = 0x0010;
        const Secret = 0x0020;
        const SoundBlock = 0x0040;
        const DontDraw = 0x0080;
        const Mapped = 0x0100;
    }

    #[derive(Debug, Clone, Copy)]
    pub struct LinedefStrifeFlags: u16 {
        const Blocking = 0x0001;
        const BlockMonsters = 0x0002;
        const TwoSided = 0x0004;
        const DontPegTop = 0x0008;
        const DontPegBottom = 0x0010;
        const Secret = 0x0020;
        const SoundBlock = 0x0040;
        const DontDraw = 0x0080;
        const Mapped = 0x0100;
        const Railing = 0x0200;
        const BlockFloating = 0x0400;
        const Translucent = 0x0800;
        const Transparent = 0x1000;
    }

    #[derive(Debug, Clone, Copy)]
    pub struct LinedefFlags: u32 {
        const Blocking = 0x00000001;
        const BlockMonsters = 0x00000002;
        const TwoSided = 0x00000004;
        const DontPegTop = 0x00000008;
        const DontPegBottom = 0x00000010;
        const Secret = 0x00000020;
        const SoundBlock = 0x00000040;
        const DontDraw = 0x00000080;
        const Mapped = 0x00000100;
        const RepeatSpecial = 0x00000200;
        const SPAC_Use = 0x00000400;
        const SPAC_MCross = 0x00000800;
        const SPAC_Impact = 0x00000C00;
        const SPAC_Push = 0x00001000;
        const SPAC_PCross = 0x00001400;
        const SPAC_UseThrough = 0x00001800;
        const MonstersCanActivate = 0x00002000;
        const BlockPlayers = 0x00004000;
        const BlockEverything = 0x00008000;
        const SPAC_Cross = 0x00010000;
        const SPAC_AnyCross = 0x00020000;
        const SPAC_MUse = 0x00040000;
        const SPAC_MPush = 0x00080000;
        const FirstSideOnly = 0x00100000;
        const ZoneBoundary = 0x00200000;
        const WrapMidTex = 0x00400000;
        const CheckSwitchRange = 0x00800000;
        const BlockProjectile = 0x01000000;
        const BlockUse = 0x02000000;
        const BlockSight = 0x04000000;
        const BlockHitScan = 0x08000000;
        // TODO: There is also 'midtex3dimpassible'; how is that different from a Strife railing?
        const Railing = 0x10000000;
        const BlockFloating = 0x20000000;
        const BlockLandMonsters = 0x40000000;
    }
}

impl From<LinedefDoomFlags> for LinedefFlags {
    #[inline]
    fn from(value: LinedefDoomFlags) -> Self {
        // Doom flags are 1:1 compatible
        Self::from_bits_truncate(value.bits() as u32)
    }
}

impl From<LinedefStrifeFlags> for LinedefFlags {
    fn from(value: LinedefStrifeFlags) -> Self {
        // Original Doom flags are 1:1 compatible, but the Strife-specific flags need to be re-mapped
        let mut flags = Self::from_bits_truncate((value.bits() & 0x1FF) as u32);
        if value.contains(LinedefStrifeFlags::Railing) {
            flags |= Self::Railing;
        }
        if value.contains(LinedefStrifeFlags::BlockFloating) {
            flags |= Self::BlockFloating;
        }
        flags
    }
}

#[derive(Debug, Clone)]
pub enum Arg0 {
    String(String),
    Int(u8),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RenderStyle {
    Translucent,
    Additive,
}

#[derive(Debug, Clone)]
pub struct LinedefProperties {
    pub alpha: f32,
    pub special: Option<ActionSpecial>,
    pub arg0: Arg0,
    pub arg1: u8,
    pub arg2: u8,
    pub arg3: u8,
    pub arg4: u8,
    pub lock_number: u8,
    pub render_style: RenderStyle,
    pub tags: Vec<u16>,
}

impl Default for LinedefProperties {
    #[inline]
    fn default() -> Self {
        Self {
            alpha: 1.0,
            special: None,
            arg0: Arg0::Int(0),
            arg1: 0,
            arg2: 0,
            arg3: 0,
            arg4: 0,
            lock_number: 0,
            render_style: RenderStyle::Translucent,
            tags: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Linedef {
    pub start_vertex: Vertex,
    pub end_vertex: Vertex,
    pub front_sidedef: Option<IndexedArc<Sidedef>>,
    pub back_sidedef: Option<IndexedArc<Sidedef>>,
    pub flags: LinedefFlags,
    pub props: LinedefProperties,
}

impl Linedef {
    pub const DOOM_BYTE_COUNT: usize = 14;
    pub const HEXEN_BYTE_COUNT: usize = 16;

    fn process_flags(raw: u16, map_type: MapType, props: &mut LinedefProperties) -> LinedefFlags {
        match map_type {
            MapType::Doom | MapType::Heretic => {
                LinedefDoomFlags::from_bits_truncate(raw).into()
            },
            MapType::Strife => {
                let flags = LinedefStrifeFlags::from_bits_truncate(raw);
                if flags.contains(LinedefStrifeFlags::Translucent) {
                    props.alpha = 0.75;
                }
                if flags.contains(LinedefStrifeFlags::Transparent) {
                    props.alpha = 0.25;
                }
                flags.into()
            },
            MapType::Hexen | MapType::ZDoom => {
                LinedefFlags::from_bits_truncate(raw as u32)
            },
        }
    }

    pub fn from_doom_bytes(
        bytes: &[u8; Self::DOOM_BYTE_COUNT],
        map_type: MapType,
        vertices: &Vertices,
        sidedefs: &Sidedefs,
    ) -> Result<Self, LinedefLoadError> {
        let start_vertex = {
            let idx = i16::from_le_bytes(bytes[0..2].try_into().unwrap()) as u16;
            vertices.get(idx as usize)
                .ok_or(LinedefLoadError::MissingVertex(idx))?
                .clone()
        };
        let end_vertex = {
            let idx = i16::from_le_bytes(bytes[2..4].try_into().unwrap()) as u16;
            vertices.get(idx as usize)
                .ok_or(LinedefLoadError::MissingVertex(idx))?
                .clone()
        };

        let mut props = LinedefProperties::default();
        let flags = Self::process_flags(
            i16::from_le_bytes(bytes[4..6].try_into().unwrap()) as u16,
            map_type,
            &mut props,
        );

        props.special = {
            let _idx = i16::from_le_bytes(bytes[6..8].try_into().unwrap());
            let _tag = i16::from_le_bytes(bytes[8..10].try_into().unwrap());
            // TODO: Map Doom linedef types => action special
            None
        };

        let front_sidedef = {
            let idx = i16::from_le_bytes(bytes[10..12].try_into().unwrap()) as u16;
            if idx == 0xFFFF {
                None
            } else {
                Some(sidedefs.get(idx as usize)
                    .ok_or(LinedefLoadError::MissingSidedef(idx))?
                    .to_arc())
            }
        };
        let back_sidedef = {
            let idx = i16::from_le_bytes(bytes[12..14].try_into().unwrap()) as u16;
            if idx == 0xFFFF {
                None
            } else {
                Some(sidedefs.get(idx as usize)
                    .ok_or(LinedefLoadError::MissingSidedef(idx))?
                    .to_arc())
            }
        };

        Ok(Self {
            start_vertex,
            end_vertex,
            front_sidedef,
            back_sidedef,
            flags,
            props,
        })
    }

    pub fn from_hexen_bytes(
        bytes: &[u8; Self::HEXEN_BYTE_COUNT],
        map_type: MapType,
        vertices: &Vertices,
        sidedefs: &Sidedefs,
    ) -> Result<Self, LinedefLoadError> {
        let start_vertex = {
            let idx = i16::from_le_bytes(bytes[0..2].try_into().unwrap()) as u16;
            vertices.get(idx as usize)
                .ok_or(LinedefLoadError::MissingVertex(idx))?
                .clone()
        };
        let end_vertex = {
            let idx = i16::from_le_bytes(bytes[2..4].try_into().unwrap()) as u16;
            vertices.get(idx as usize)
                .ok_or(LinedefLoadError::MissingVertex(idx))?
                .clone()
        };

        let mut props = LinedefProperties::default();
        let flags = Self::process_flags(
            i16::from_le_bytes(bytes[4..6].try_into().unwrap()) as u16,
            map_type,
            &mut props,
        );

        props.special = {
            let _idx = bytes[6];
            // TODO: Define Action Specials enum
            None
        };
        props.arg0 = Arg0::Int(bytes[7]);
        props.arg1 = bytes[8];
        props.arg2 = bytes[9];
        props.arg3 = bytes[10];
        props.arg4 = bytes[11];

        let front_sidedef = {
            let idx = i16::from_le_bytes(bytes[12..14].try_into().unwrap()) as u16;
            if idx == 0xFFFF {
                None
            } else {
                Some(sidedefs.get(idx as usize)
                    .ok_or(LinedefLoadError::MissingSidedef(idx))?
                    .to_arc())
            }
        };
        let back_sidedef = {
            let idx = i16::from_le_bytes(bytes[14..16].try_into().unwrap()) as u16;
            if idx == 0xFFFF {
                None
            } else {
                Some(sidedefs.get(idx as usize)
                    .ok_or(LinedefLoadError::MissingSidedef(idx))?
                    .to_arc())
            }
        };

        Ok(Self {
            start_vertex,
            end_vertex,
            front_sidedef,
            back_sidedef,
            flags,
            props,
        })
    }
}

#[derive(Educe)]
#[educe(Deref)]
pub struct Linedefs {
    #[educe(Deref)]
    linedefs: IndexedVec<Linedef>,
    vertices: Arc<Resource<Vertices>>,
    sidedefs: Arc<Resource<Sidedefs>>,
}

impl Linedefs {
    #[inline]
    pub fn lump_loader(
        renderer: Arc<Renderer>,
        map_type: MapType,
        map_format: MapFormat,
    ) -> LinedefsLumpLoader {
        LinedefsLumpLoader {
            renderer,
            map_type,
            map_format,
        }
    }

    pub async fn read(
        mut reader: impl AsyncRead + Unpin,
        map_type: MapType,
        map_format: MapFormat,
        vertices: Arc<Resource<Vertices>>,
        sidedefs: Arc<Resource<Sidedefs>>,
    ) -> Result<Self, LinedefLoadError> {
        let mut linedefs = Vec::new();

        {
            let vertices_lock = vertices.read();
            let sidedefs_lock = sidedefs.read();
            match map_format {
                MapFormat::Doom => {
                    let mut bytes = [0; Linedef::DOOM_BYTE_COUNT];
                    loop {
                        match reader.read_exact(&mut bytes).await {
                            Ok(()) => linedefs.push(Linedef::from_doom_bytes(
                                &bytes,
                                map_type,
                                &vertices_lock,
                                &sidedefs_lock,
                            )?),
                            Err(error) => match error.kind() {
                                std::io::ErrorKind::UnexpectedEof => break,
                                _ => return Err(error.into()),
                            }
                        }
                    }
                },
                MapFormat::Hexen => {
                    let mut bytes = [0; Linedef::HEXEN_BYTE_COUNT];
                    loop {
                        match reader.read_exact(&mut bytes).await {
                            Ok(()) => linedefs.push(Linedef::from_hexen_bytes(
                                &bytes,
                                map_type,
                                &vertices_lock,
                                &sidedefs_lock,
                            )?),
                            Err(error) => match error.kind() {
                                std::io::ErrorKind::UnexpectedEof => break,
                                _ => return Err(error.into()),
                            }
                        }
                    }
                },
                MapFormat::UDMF => panic!("Can't load binary lump linedefs for UDMF map format"),
            }
        }
        debug!(count = ?linedefs.len(), "Loaded Linedefs");

        Ok(Self {
            linedefs: linedefs.into(),
            vertices,
            sidedefs,
        })
    }
}

pub struct LinedefsLumpLoader {
    renderer: Arc<Renderer>,
    map_type: MapType,
    map_format: MapFormat,
}

#[async_trait]
impl ResourceLoader for LinedefsLumpLoader {
    type Output = Linedefs;

    async fn load(
        &self,
        data: ResourceReadData,
        ctx: &ResourceLoadContext,
    ) -> Result<Box<Linedefs>, ResourceLoadError> {
        let vertices_fut = ctx.get_with_loader(
            PathBuf::from(ctx.id().as_str()).parent().unwrap().join("VERTEXES"),
            Vertices::lump_loader(),
        );
        let sidedefs_fut = ctx.get_with_loader(
            PathBuf::from(ctx.id().as_str()).parent().unwrap().join("SIDEDEFS"),
            Sidedefs::lump_loader(self.renderer.clone(), self.map_type),
        );
        Ok(Box::new(Linedefs::read(
            data,
            self.map_type,
            self.map_format,
            vertices_fut.await?,
            sidedefs_fut.await?,
        ).await?))
    }
}