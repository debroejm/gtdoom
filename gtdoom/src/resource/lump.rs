use ahash::{HashMap, HashMapExt};
use async_trait::async_trait;
use async_zip::base::read::seek::ZipFileReader;
use async_zip::error::ZipError;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use educe::Educe;
use gtether::console::command::{Command, CommandError, ParamCountCheck};
use gtether::resource::id::ResourceId;
use gtether::resource::manager::ResourceLoadContext;
use gtether::resource::source::fs::{ResourceSourceFile, ResourceSourceFilesHandle};
use gtether::resource::source::{ResourceData, ResourceDataResult, ResourceDataSource, ResourceSource, ResourceUpdate};
use gtether::resource::watcher::{ResourceHashProvider, ResourceWatcher};
use gtether::resource::{ResourceDefaultLoader, ResourceLoadError, ResourceLoader, ResourceReadData};
use regex::RegexBuilder;
use sha2::{Digest, Sha256};
use smol::future;
use smol::io::{AsyncRead, AsyncReadExt, AsyncSeekExt, BufReader};
use std::error::Error;
use std::ffi::OsStr;
use std::fmt::{Debug, Display, Formatter};
use std::io::{Error as IoError, ErrorKind as IoErrorKind, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use strum::VariantArray;
use tracing::{debug, error, info, trace, warn};

use crate::resource::string_from_bytes_until_nul_checked;
use crate::resource::texture::TextureFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WadType {
    Iwad,
    Pwad,
    Unknown,
}

impl Display for WadType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Iwad => write!(f, "IWAD"),
            Self::Pwad => write!(f, "PWAD"),
            Self::Unknown => write!(f, "<unknown>"),
        }
    }
}

impl<S: AsRef<str>> From<S> for WadType {
    #[inline]
    fn from(value: S) -> Self {
        match value.as_ref() {
            "IWAD" => Self::Iwad,
            "PWAD" => Self::Pwad,
            _ => Self::Unknown,
        }
    }
}

#[derive(VariantArray, Debug, Clone, Copy, PartialEq, Eq)]
pub enum LumpType {
    Global,
    Marker,
    Patch,
    Sprite,
    Flat,
    Graphic,
    Colormap,
    AcsLibrary,
    Texture,
    HiResTexture,
    Voice,
    Sound,
    Music,
    Voxel,
    Map,
    Unknown,
}

impl Display for LumpType {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Global => write!(f, "<global>"),
            Self::Marker => write!(f, "<marker>"),
            Self::Patch => write!(f, "P:<patch>"),
            Self::Sprite => write!(f, "S:<sprite>"),
            Self::Flat => write!(f, "F:<flat>"),
            Self::Graphic => write!(f, "<graphic>"),
            Self::Colormap => write!(f, "C:<colormap>"),
            Self::AcsLibrary => write!(f, "A:<acs_library>"),
            Self::Texture => write!(f, "TX:<texture>"),
            Self::HiResTexture => write!(f, "HI:<hi-res-texture>"),
            Self::Voice => write!(f, "V:<voice>"),
            Self::Sound => write!(f, "<sound>"),
            Self::Music => write!(f, "<music>"),
            Self::Voxel => write!(f, "VX:<voxel>"),
            Self::Map => write!(f, "<map>"),
            Self::Unknown => write!(f, "<unknown>"),
        }
    }
}

impl From<&str> for LumpType {
    #[inline]
    fn from(value: &str) -> Self {
        match value {
            "P" | "PP" => Self::Patch,
            "S" | "SS" => Self::Sprite,
            "F" | "FF" => Self::Flat,
            "C" => Self::Colormap,
            "A" => Self::AcsLibrary,
            "TX" => Self::Texture,
            "HI" => Self::HiResTexture,
            "V" => Self::Voice,
            "VX" => Self::Voxel,
            _ => Self::Unknown,
        }
    }
}

impl LumpType {
    #[inline]
    pub fn path_prefix(&self) -> Option<PathBuf> {
        match self {
            Self::Patch => Some(PathBuf::from("Patches")),
            Self::Sprite => Some(PathBuf::from("Sprites")),
            Self::Flat => Some(PathBuf::from("Flats")),
            Self::Graphic => Some(PathBuf::from("Graphics")),
            Self::Colormap => Some(PathBuf::from("Colormaps")),
            Self::AcsLibrary => Some(PathBuf::from("ACS")),
            Self::Texture => Some(PathBuf::from("Textures")),
            Self::HiResTexture => Some(PathBuf::from("HiRes")),
            Self::Voice => Some(PathBuf::from("Voices")),
            Self::Sound => Some(PathBuf::from("Sounds")),
            Self::Music => Some(PathBuf::from("Music")),
            Self::Voxel => Some(PathBuf::from("Voxels")),
            _ => None,
        }
    }

    pub fn for_path(path: &Path) -> Self {
        let mut iter = path.iter();
        let first = match iter.next() {
            Some(e) => e,
            None => return Self::Unknown,
        };
        if iter.count() == 0 {
            Self::Global
        } else {
            let first = match first.to_str() {
                Some(s) => s.to_lowercase(),
                None => return Self::Unknown,
            };
            match first.as_str() {
                "patches" => Self::Patch,
                "sprites" => Self::Sprite,
                "flats" => Self::Flat,
                "graphics" => Self::Graphic,
                "colormaps" => Self::Colormap,
                "acs" => Self::AcsLibrary,
                "textures" => Self::Texture,
                "hires" => Self::HiResTexture,
                "voices" => Self::Voice,
                "sounds" => Self::Sound,
                "music" => Self::Music,
                "voxels" => Self::Voxel,
                _ => Self::Unknown,
            }
        }
    }
}

#[inline]
fn lump_data_fmt<T>(_: &T, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(f, "<data>")
}

#[derive(Educe)]
#[educe(Debug)]
pub struct Lump {
    l_type: LumpType,
    #[educe(Debug(method(lump_data_fmt)))]
    data: Box<dyn AsyncRead + Send + Unpin + 'static>,
    hash: String,
}

impl Lump {
    #[inline]
    pub fn new(
        l_type: LumpType,
        data: impl AsyncRead + Send + Unpin + 'static,
        hash: String,
    ) -> Self {
        Self {
            l_type,
            data: Box::new(data),
            hash,
        }
    }

    #[inline]
    pub fn l_type(&self) -> LumpType {
        self.l_type
    }

    #[inline]
    pub fn data(&self) -> &Box<dyn AsyncRead + Send + Unpin + 'static> {
        &self.data
    }

    #[inline]
    pub fn hash(&self) -> &str {
        &self.hash
    }
}

impl From<Lump> for ResourceData {
    fn from(value: Lump) -> Self {
        Self::new(value.data, value.hash)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum LumpFileError {
    #[error("failed to read WAD file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("unknown file type: {0}")]
    UnknownFileType(String),
    #[error("lump not found: {0}")]
    NotFound(ResourceId),
    #[error("start marker ({start}) does not match end marker ({end})")]
    MismatchedMarker {
        start: LumpType,
        end: LumpType,
    },
    #[error(transparent)]
    Other(Box<dyn Error + Send + Sync>),
}

impl From<ZipError> for LumpFileError {
    #[inline]
    fn from(value: ZipError) -> Self {
        Self::Other(Box::new(value))
    }
}

impl From<regex::Error> for LumpFileError {
    #[inline]
    fn from(value: regex::Error) -> Self {
        Self::Other(Box::new(value))
    }
}

impl From<LumpFileError> for ResourceLoadError {
    #[inline]
    fn from(value: LumpFileError) -> Self {
        match value {
            LumpFileError::NotFound(id) => Self::NotFound(id),
            value => Self::from_error(value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LumpDirectoryEntry {
    offset: i32,
    size: i32,
    name: String,
}

impl LumpDirectoryEntry {
    pub const BYTE_COUNT: usize = size_of::<i32>() + size_of::<i32>() + 8;

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() < Self::BYTE_COUNT {
            return Err(IoErrorKind::UnexpectedEof.into())
        }

        let offset = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let size = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let name = string_from_bytes_until_nul_checked(&bytes[8..16]);

        Ok(Self {
            offset,
            size,
            name,
        })
    }

    pub async fn read(mut reader: impl AsyncRead + Unpin) -> Result<Self, IoError> {
        let mut bytes = [0; Self::BYTE_COUNT];
        reader.read_exact(&mut bytes).await?;
        Self::from_bytes(&bytes)
    }

    #[inline]
    pub fn offset(&self) -> u32 {
        self.offset as u32
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size as usize
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
enum WadEntryExtra {
    None,
    StaticPrefix(Vec<u8>),
}

#[derive(Debug)]
struct WadEntry {
    l_type: LumpType,
    size: usize,
    offset: u32,
    hash: String,
    extra: WadEntryExtra,
}

impl WadEntry {
    async fn read_lump(&self, mut file: smol::fs::File) -> Result<Lump, LumpFileError> {
        file.seek(SeekFrom::Start(self.offset as u64)).await?;
        let reader = file.take(self.size as u64);
        match &self.extra {
            WadEntryExtra::None => Ok(Lump::new(self.l_type, reader, self.hash.clone())),
            WadEntryExtra::StaticPrefix(data) => {
                let reader = smol::io::Cursor::new(data.clone()).chain(reader);
                Ok(Lump::new(self.l_type, reader, self.hash.clone()))
            }
        }
    }
}

#[derive(Debug)]
struct Wad {
    #[allow(unused)]
    w_type: WadType,
    data: HashMap<ResourceId, WadEntry>,
}

impl Wad {
    const MAP_LUMP_NAMES: [&'static str; 15] = [
        "TEXTMAP",
        "THINGS",
        "LINEDEFS",
        "SIDEDEFS",
        "VERTEXES",
        "SEGS",
        "SSECTORS",
        "NODES",
        "ZNODES",
        "SECTORS",
        "REJECT",
        "BLOCKMAP",
        "BEHAVIOR",
        "SCRIPTS",
        "ENDMAP",
    ];

    async fn read_textures(
        file: &mut smol::fs::File,
        start_offset: u64,
        data: &mut HashMap<ResourceId, WadEntry>,
    ) -> Result<(), LumpFileError> {
        let orig_pos = file.seek(SeekFrom::Current(0)).await?;
        file.seek(SeekFrom::Start(start_offset)).await?;

        let mut header_buf = [0; 4];
        file.read_exact(&mut header_buf).await?;
        let num_textures = i32::from_le_bytes(header_buf) as u32;

        let mut offsets = Vec::with_capacity(num_textures as usize);
        for _ in 0..num_textures {
            file.read_exact(&mut header_buf).await?;
            offsets.push(i32::from_le_bytes(header_buf) as u32);
        }

        let textures_namespace = PathBuf::from("Textures");

        for offset in offsets {
            file.seek(SeekFrom::Start(start_offset + offset as u64)).await?;
            let mut buf = [0; 18];
            file.read_exact(&mut buf).await?;
            let name = string_from_bytes_until_nul_checked(&buf[0..8]);
            let patch_count = i16::from_le_bytes(buf[16..18].try_into().unwrap());
            let size = if patch_count == 0 {
                // Doom format
                let mut buf = [0; 4];
                file.read_exact(&mut buf).await?;
                let patch_count = i16::from_le_bytes(buf[2..4].try_into().unwrap());
                22 + (patch_count as usize) * 10
            } else {
                // Strife format
                18 + (patch_count as usize) * 6
            };

            let prefix = vec![TextureFormat::Doom.to_byte()];

            data.insert(textures_namespace.join(name).into(), WadEntry {
                l_type: LumpType::Texture,
                size: size - 8, // subtract 8 for the name, which we're dropping
                offset: (start_offset as u32) + offset + 8, // Add 8 for the name
                hash: String::new(),
                extra: WadEntryExtra::StaticPrefix(prefix),
            });
        }

        file.seek(SeekFrom::Start(orig_pos)).await?;
        Ok(())
    }

    async fn read(file: &mut smol::fs::File) -> Result<Self, LumpFileError> {
        file.seek(SeekFrom::Start(0)).await?;
        let (w_type, dir_size, dir_offset) = {
            let mut buffer = [0; 12];
            file.read_exact(&mut buffer).await?;

            let w_type = WadType::from(
                String::from_utf8_lossy(&buffer[0..4]).trim_matches(char::from(0))
            );
            let size = i32::from_le_bytes(buffer[4..8].try_into().unwrap()) as usize;
            let offset = i32::from_le_bytes(buffer[8..12].try_into().unwrap()) as u64;

            (w_type, size, offset)
        };

        enum LumpBuilder {
            Type(LumpType),
            Map(String),
        }
        let map_re = RegexBuilder::new(r"^E\dM\d|MAP\d\d$")
            .case_insensitive(true)
            .build()?;

        file.seek(SeekFrom::Start(dir_offset)).await?;
        let mut lump_builder_stack = vec![LumpBuilder::Type(LumpType::Global)];
        let mut data = HashMap::with_capacity(dir_size);

        fn try_map_pop(stack: &mut Vec<LumpBuilder>) {
            if let Some(top) = stack.pop() {
                match top {
                    LumpBuilder::Map(_) => {},
                    _ => stack.push(top),
                }
            }
        }

        for _ in 0..dir_size {
            let entry = LumpDirectoryEntry::read(&mut *file).await?;
            trace!(?entry, "Processing lump");

            let offset = entry.offset();
            let size = entry.size();
            let name = entry.name().to_string();

            if name.ends_with("_START") {
                try_map_pop(&mut lump_builder_stack);

                let marker = &name[0..(name.len() - 6)];
                let lump_type = LumpType::from(marker);
                if lump_type == LumpType::Unknown {
                    warn!(marker, "Unknown START lump type");
                }

                data.insert(ResourceId::from(&name), WadEntry {
                    l_type: LumpType::Marker,
                    size,
                    offset,
                    hash: name,
                    extra: WadEntryExtra::None,
                });
                lump_builder_stack.push(LumpBuilder::Type(lump_type));
            } else if name.ends_with("_END") {
                try_map_pop(&mut lump_builder_stack);
                let curr_lump_builder = lump_builder_stack.last().unwrap();

                let marker = &name[0..(name.len() - 4)];
                let lump_type = LumpType::from(marker);
                if lump_type == LumpType::Unknown {
                    warn!(marker, "Unknown END lump type");
                }
                match curr_lump_builder {
                    LumpBuilder::Type(curr_lump_type) => {
                        if lump_type != *curr_lump_type {
                            return Err(LumpFileError::MismatchedMarker { start: *curr_lump_type, end: lump_type })
                        }
                    },
                    LumpBuilder::Map { .. } => {
                        return Err(LumpFileError::MismatchedMarker { start: LumpType::Map, end: lump_type })
                    }
                }

                data.insert(ResourceId::from(&name), WadEntry {
                    l_type: LumpType::Marker,
                    size,
                    offset,
                    hash: name,
                    extra: WadEntryExtra::None,
                });
                lump_builder_stack.pop();
            } else if &name == "TEXTURE1" || &name == "TEXTURE2" {
                try_map_pop(&mut lump_builder_stack);
                Self::read_textures(file, offset as u64, &mut data).await?;
            } else if map_re.is_match(&name) { // TODO: Also check map names from MAPINFO
                try_map_pop(&mut lump_builder_stack);
                data.insert(ResourceId::from(&name), WadEntry {
                    l_type: LumpType::Map,
                    size,
                    offset,
                    hash: name.clone(),
                    extra: WadEntryExtra::None,
                });
                lump_builder_stack.push(LumpBuilder::Map(name));
            } else {
                if !Self::MAP_LUMP_NAMES.contains(&name.as_str()) {
                    try_map_pop(&mut lump_builder_stack);
                }
                match lump_builder_stack.last().unwrap() {
                    LumpBuilder::Type(l_type) => {
                        let id = if let Some(prefix) = l_type.path_prefix() {
                            ResourceId::from(prefix.join(&name))
                        } else {
                            ResourceId::from(&name)
                        };
                        data.insert(id, WadEntry {
                            l_type: *l_type,
                            size,
                            offset,
                            hash: String::new(),
                            extra: WadEntryExtra::None,
                        });
                    },
                    LumpBuilder::Map(map_name) => {
                        data.insert(PathBuf::from(map_name).join(name).into(), WadEntry {
                            l_type: LumpType::Map,
                            size,
                            offset,
                            hash: String::new(),
                            extra: WadEntryExtra::None,
                        });
                    }
                }
            }
        }
        try_map_pop(&mut lump_builder_stack);

        for entry in data.values_mut() {
            if entry.hash != "" {
                // Lump already has a hash
                continue
            }
            file.seek(SeekFrom::Start(entry.offset as u64)).await?;
            let mut sha256 = Sha256::new();
            smol::io::copy(
                file.take(entry.size as u64),
                smol::io::AssertAsync::new(&mut sha256),
            ).await?;
            entry.hash = BASE64_STANDARD.encode(sha256.finalize());
        }

        Ok(Self {
            w_type,
            data,
        })
    }
}

#[derive(Debug)]
struct ZipEntry {
    l_type: LumpType,
    path: PathBuf,
    hash: String,
}

impl ZipEntry {
    async fn read_lump(&self, file: smol::fs::File) -> Result<Lump, LumpFileError> {
        let reader = ZipFileReader::new(BufReader::new(file)).await?;
        for idx in 0..reader.file().entries().len() {
            let entry = reader.file().entries().get(idx).unwrap();
            let path = Zip::sanitize_path(entry.filename().as_str()?);
            if path == self.path {
                return Ok(Lump::new(
                    self.l_type,
                    reader.into_entry(idx).await?,
                    self.hash.clone(),
                ))
            }
        }
        Err(LumpFileError::Other(format!("zip file missing entry path: '{}'", self.path.display()).into()))
    }
}

#[derive(Debug)]
struct Zip {
    data: HashMap<ResourceId, ZipEntry>,
}

impl Zip {
    fn sanitize_path(path: &str) -> PathBuf {
        path.replace('\\', "/")
            .split('/')
            .map(sanitize_filename::sanitize)
            .collect()
    }

    async fn read(file: &mut smol::fs::File) -> Result<Self, LumpFileError> {
        let mut reader = ZipFileReader::new(BufReader::new(file)).await?;
        let entry_count = reader.file().entries().len();
        let mut data = HashMap::with_capacity(entry_count);
        for idx in 0..entry_count {
            let entry = reader.file().entries().get(idx).unwrap();
            if entry.dir()? {
                continue
            }
            let path = Self::sanitize_path(entry.filename().as_str()?);

            let l_type = LumpType::for_path(&path);
            if l_type == LumpType::Unknown {
                trace!(lump = ?path, "Ignoring unknown lump type");
                continue
            }

            let id = match path.file_name().and_then(OsStr::to_str) {
                Some(name) => ResourceId::from(name),
                None => continue,
            };

            let mut sha256 = Sha256::new();
            smol::io::copy(
                reader.reader_without_entry(idx).await?,
                smol::io::AssertAsync::new(&mut sha256),
            ).await?;
            let hash = BASE64_STANDARD.encode(sha256.finalize());

            data.insert(id, ZipEntry {
                l_type,
                path,
                hash,
            });
        }

        Ok(Self {
            data,
        })
    }
}

#[derive(Debug)]
enum LumpFileType {
    None,
    Wad(Wad),
    Zip(Zip),
}

impl LumpFileType {
    async fn read(path: &Path) -> Result<Self, LumpFileError> {
        let mut file = smol::fs::File::open(path).await?;
        match path.extension().and_then(OsStr::to_str).map(|s| s.to_lowercase()) {
            Some(ext) => {
                match ext.as_str() {
                    "wad" | "iwad" => {
                        Ok(Self::Wad(Wad::read(&mut file).await?))
                    },
                    "zip" | "pk3" | "pkz" | "pke" | "ipk3" => {
                        Ok(Self::Zip(Zip::read(&mut file).await?))
                    },
                    _ => Err(LumpFileError::UnknownFileType(path.to_string_lossy().to_string())),
                }
            },
            None => Err(LumpFileError::UnknownFileType(path.to_string_lossy().to_string())),
        }
    }

    async fn load_lump(&self, path: &Path, id: &ResourceId) -> Result<Lump, LumpFileError> {
        match self {
            Self::None => Err(LumpFileError::NotFound(id.clone())),
            Self::Wad(wad) => {
                let file = smol::fs::File::open(path).await?;
                wad.data.get(id)
                    .ok_or(LumpFileError::NotFound(id.clone()))?
                    .read_lump(file).await
            },
            Self::Zip(zip) => {
                let file = smol::fs::File::open(path).await?;
                zip.data.get(id)
                    .ok_or(LumpFileError::NotFound(id.clone()))?
                    .read_lump(file).await
            },
        }
    }

    fn hashes(&self) -> LumpFileHashIter<'_> {
        match self {
            Self::None => LumpFileHashIter::None,
            Self::Wad(wad) => LumpFileHashIter::Wad(wad.data.iter()),
            Self::Zip(zip) => LumpFileHashIter::Zip(zip.data.iter()),
        }
    }

    fn hash(&self, id: &ResourceId) -> Option<&String> {
        match self {
            Self::None => None,
            Self::Wad(wad) => wad.data.get(id).map(|e| &e.hash),
            Self::Zip(zip) => zip.data.get(id).map(|e| &e.hash),
        }
    }

    // TODO: create a better watcher iterator for this
    fn diff_hashes(
        &self,
        other: &LumpFileType,
        watcher: &ResourceWatcher,
    ) -> HashMap<ResourceId, ResourceUpdate> {
        let mut diff = HashMap::new();

        for (id, hash) in self.hashes() {
            if watcher.is_watched(id) {
                if let Some(other_hash) = other.hash(id) {
                    if hash != other_hash {
                        diff.insert(id.clone(), ResourceUpdate::Modified(hash.clone()));
                    }
                } else {
                    diff.insert(id.clone(), ResourceUpdate::Added(hash.clone()));
                }
            }
        }

        for (id, _) in other.hashes() {
            if watcher.is_watched(id) {
                if self.hash(id).is_none() {
                    diff.insert(id.clone(), ResourceUpdate::Removed);
                }
            }
        }

        diff
    }
}

enum LumpFileHashIter<'a> {
    None,
    Wad(std::collections::hash_map::Iter<'a, ResourceId, WadEntry>),
    Zip(std::collections::hash_map::Iter<'a, ResourceId, ZipEntry>),
}

impl<'a> Iterator for LumpFileHashIter<'a> {
    type Item = (&'a ResourceId, &'a String);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::Wad(iter) => iter.next()
                .map(|(id, entry)| (id, &entry.hash)),
            Self::Zip(iter) => iter.next()
                .map(|(id, entry)| (id, &entry.hash)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::None => (0, Some(0)),
            Self::Wad(iter) => iter.size_hint(),
            Self::Zip(iter) => iter.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for LumpFileHashIter<'a> {}

impl ResourceHashProvider for LumpFileType {
    #[inline]
    fn hash(&self, id: &ResourceId) -> Option<String> {
        LumpFileType::hash(self, id).cloned()
    }
}

#[derive(Debug)]
pub struct LumpFile {
    path: PathBuf,
    inner: Arc<smol::lock::RwLock<LumpFileType>>,
    watcher: ResourceWatcher,
}

impl LumpFile {
    #[inline]
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, LumpFileError> {
        Self::open_impl(path.as_ref()).await
    }

    async fn open_impl(path: &Path) -> Result<Self, LumpFileError> {
        let lump_file = LumpFileType::read(path).await?;
        let inner = Arc::new(smol::lock::RwLock::new(lump_file));
        let watcher = ResourceWatcher::new(inner.clone());

        Ok(Self {
            path: path.to_path_buf(),
            inner,
            watcher,
        })
    }

    fn set_lump_file(&self, new_lump_file: LumpFileType) {
        let mut inner = self.inner.write_blocking();
        let diff = new_lump_file.diff_hashes(&inner, &self.watcher);
        self.watcher.notify_update(diff.into());
        *inner = new_lump_file;
    }
}

impl ResourceSourceFile for LumpFile {
    #[inline]
    fn path(&self) -> &Path {
        &self.path
    }

    fn reload(&self) {
        let result = future::block_on(LumpFileType::read(&self.path));
        let new_lump_file = match result {
            Ok(new_lump_file) => {
                debug!(
                    path = ?self.path, lump_count = ?new_lump_file.hashes().len(),
                    "Reloaded lump file source"
                );
                new_lump_file
            },
            Err(error) => {
                error!(path = ?self.path, ?error, "Failed to reload lump file source");
                LumpFileType::None
            }
        };

        self.set_lump_file(new_lump_file);
    }

    fn invalidate(&self) {
        self.set_lump_file(LumpFileType::None);
    }
}

#[async_trait]
impl ResourceSource for LumpFile {
    #[inline]
    fn hash(&self, id: &ResourceId) -> Option<ResourceDataSource> {
        self.inner.read_blocking().hash(id).map(|h| ResourceDataSource::new(h.clone()))
    }

    #[inline]
    async fn load(&self, id: &ResourceId) -> ResourceDataResult {
        Ok(self.inner.read().await.load_lump(&self.path, id).await?.into())
    }

    fn watcher(&self) -> &ResourceWatcher {
        &self.watcher
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct ListSourcesCommand {
    #[educe(Debug(ignore))]
    lump_files: ResourceSourceFilesHandle<LumpFile>,
}

impl ListSourcesCommand {
    #[inline]
    pub fn new(lump_files: ResourceSourceFilesHandle<LumpFile>) -> Self {
        Self {
            lump_files,
        }
    }
}

impl Command for ListSourcesCommand {
    fn handle(&self, parameters: &[String]) -> Result<(), CommandError> {
        ParamCountCheck::Equal(0).check(parameters.len() as u32)?;

        info!("Lump Sources:");
        for (idx, lump_file) in self.lump_files.read().iter().enumerate() {
            info!("{idx}: {:?}", lump_file.path());
        }

        Ok(())
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct MoveSourceCommand {
    #[educe(Debug(ignore))]
    lump_files: ResourceSourceFilesHandle<LumpFile>,
}

impl MoveSourceCommand {
    #[inline]
    pub fn new(lump_files: ResourceSourceFilesHandle<LumpFile>) -> Self {
        Self {
            lump_files,
        }
    }
}

impl Command for MoveSourceCommand {
    fn handle(&self, parameters: &[String]) -> Result<(), CommandError> {
        ParamCountCheck::Equal(2).check(parameters.len() as u32)?;

        let src_idx = parameters[0].parse::<usize>()
            .map_err(|_| CommandError::InvalidParameter(parameters[0].clone()))?;
        let dst_idx = parameters[1].parse::<usize>()
            .map_err(|_| CommandError::InvalidParameter(parameters[1].clone()))?;

        let mut lump_files = self.lump_files.write();
        let len = lump_files.len();
        if src_idx >= len {
            return Err(CommandError::InvalidParameter(format!("{src_idx} must be < {len}")))
        }
        if dst_idx >= len {
            return Err(CommandError::InvalidParameter(format!("{dst_idx} must be < {len}")))
        }

        let source = lump_files.remove(src_idx);
        lump_files.insert(dst_idx, source);
        info!("Moved {src_idx} => {dst_idx}");

        Ok(())
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct AddSourceCommand {
    #[educe(Debug(ignore))]
    lump_files: ResourceSourceFilesHandle<LumpFile>,
}

impl AddSourceCommand {
    #[inline]
    pub fn new(lump_files: ResourceSourceFilesHandle<LumpFile>) -> Self {
        Self {
            lump_files,
        }
    }
}

impl Command for AddSourceCommand {
    fn handle(&self, parameters: &[String]) -> Result<(), CommandError> {
        ParamCountCheck::Range { min: 1, max: 2 }.check(parameters.len() as u32)?;

        let (idx, path) = match parameters.len() {
            1 => (None, PathBuf::from(&parameters[0])),
            2 => {
                let idx = parameters[0].parse::<usize>()
                    .map_err(|_| CommandError::InvalidParameter(parameters[0].clone()))?;
                (Some(idx), PathBuf::from(&parameters[1]))
            },
            _ => unreachable!(),
        };

        let lump_file = future::block_on(LumpFile::open(path))
            .map_err(|error| CommandError::CommandFailure(Box::new(error)))?;

        match idx {
            Some(idx) => {
                let mut lump_files = self.lump_files.write();
                let len = lump_files.len();
                if idx > len {
                    return Err(CommandError::InvalidParameter(format!("{idx} must be < {len}")))
                }
                let path = lump_file.path.clone();
                lump_files.insert(idx, lump_file);
                info!("Added {idx}: {path:?}");
            },
            None => {
                let path = lump_file.path.clone();
                self.lump_files.write().push(lump_file);
                info!("Added {path:?}");
            },
        }

        Ok(())
    }
}

#[derive(Educe)]
#[educe(Debug)]
pub struct RemoveSourceCommand {
    #[educe(Debug(ignore))]
    lump_files: ResourceSourceFilesHandle<LumpFile>,
}

impl RemoveSourceCommand {
    #[inline]
    pub fn new(lump_files: ResourceSourceFilesHandle<LumpFile>) -> Self {
        Self {
            lump_files,
        }
    }
}

impl Command for RemoveSourceCommand {
    fn handle(&self, parameters: &[String]) -> Result<(), CommandError> {
        ParamCountCheck::Equal(1).check(parameters.len() as u32)?;

        let idx = parameters[0].parse::<usize>()
            .map_err(|_| CommandError::InvalidParameter(parameters[0].clone()))?;

        let mut lump_files = self.lump_files.write();
        let len = lump_files.len();
        if idx > len {
            return Err(CommandError::InvalidParameter(format!("{idx} must be < {len}")))
        }

        let source = lump_files.remove(idx);
        info!("Removed {idx}: {:?}", source.path);

        Ok(())
    }
}

pub struct MarkerLump {}

impl ResourceDefaultLoader for MarkerLump {
    type Loader = MarkerLumpLoader;

    fn default_loader() -> Self::Loader {
        MarkerLumpLoader(())
    }
}

pub struct MarkerLumpLoader(());

#[async_trait]
impl ResourceLoader for MarkerLumpLoader {
    type Output = MarkerLump;

    async fn load(
        &self,
        _data: ResourceReadData,
        _ctx: &ResourceLoadContext,
    ) -> Result<Box<MarkerLump>, ResourceLoadError> {
        // Marker lumps don't actually have data
        Ok(Box::new(MarkerLump {}))
    }
}