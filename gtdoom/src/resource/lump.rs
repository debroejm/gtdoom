use ahash::{HashMap, HashMapExt};
use async_trait::async_trait;
use async_zip::base::read::seek::ZipFileReader;
use async_zip::error::ZipError;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use educe::Educe;
use gtether::resource::id::ResourceId;
use gtether::resource::source::{ResourceData, ResourceDataResult, ResourceDataSource, ResourceSource, ResourceUpdate, ResourceWatcher, SourceIndex};
use gtether::resource::{ResourceLoadContext, ResourceLoadError, ResourceLoader, ResourceReadData};
use notify::event::{ModifyKind, RenameMode};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer};
use parking_lot::RwLock;
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
use std::time::Duration;
use strum::VariantArray;
use tracing::{debug, error, info, trace, warn};

use crate::resource::string_from_bytes_until_nul_checked;

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
            Self::Patch => Some(PathBuf::from("patches")),
            Self::Sprite => Some(PathBuf::from("sprites")),
            Self::Flat => Some(PathBuf::from("flats")),
            Self::Graphic => Some(PathBuf::from("graphics")),
            Self::Colormap => Some(PathBuf::from("colormaps")),
            Self::AcsLibrary => Some(PathBuf::from("acs")),
            Self::Texture => Some(PathBuf::from("textures")),
            Self::HiResTexture => Some(PathBuf::from("hires")),
            Self::Voice => Some(PathBuf::from("voices")),
            Self::Sound => Some(PathBuf::from("sounds")),
            Self::Music => Some(PathBuf::from("music")),
            Self::Voxel => Some(PathBuf::from("voxels")),
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

impl From<notify::Error> for LumpFileError {
    #[inline]
    fn from(value: notify::Error) -> Self {
        match value.kind {
            notify::ErrorKind::Io(err) => Self::ReadError(err),
            ref _kind => Self::Other(Box::new(value)),
        }
    }
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
struct WadEntry {
    l_type: LumpType,
    size: usize,
    offset: u32,
    hash: String,
}

impl WadEntry {
    async fn read_lump(&self, mut file: smol::fs::File) -> Result<Lump, LumpFileError> {
        file.seek(SeekFrom::Start(self.offset as u64)).await?;
        Ok(Lump::new(self.l_type, file.take(self.size as u64), self.hash.clone()))
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
                });
                lump_builder_stack.pop();
            } else if map_re.is_match(&name) { // TODO: Also check map names from MAPINFO
                try_map_pop(&mut lump_builder_stack);
                data.insert(ResourceId::from(&name), WadEntry {
                    l_type: LumpType::Map,
                    size,
                    offset,
                    hash: name.clone(),
                });
                lump_builder_stack.push(LumpBuilder::Map(name));
            } else {
                if !Self::MAP_LUMP_NAMES.contains(&name.as_str()) {
                    try_map_pop(&mut lump_builder_stack);
                }
                match lump_builder_stack.last().unwrap() {
                    LumpBuilder::Type(l_type) => {
                        data.insert(ResourceId::from(&name), WadEntry {
                            l_type: *l_type,
                            size,
                            offset,
                            hash: String::new(),
                        });
                    },
                    LumpBuilder::Map(map_name) => {
                        data.insert(ResourceId::from(map_name).join(name), WadEntry {
                            l_type: LumpType::Map,
                            size,
                            offset,
                            hash: String::new(),
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
enum LumpFile {
    None,
    Wad(Wad),
    Zip(Zip),
}

impl LumpFile {
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

    fn diff_hashes(&self, other: &LumpFile) -> HashMap<ResourceId, ResourceUpdate> {
        let mut diff = HashMap::new();

        for (id, hash) in self.hashes() {
            if let Some(other_hash) = other.hash(id) {
                if hash != other_hash {
                    diff.insert(id.clone(), ResourceUpdate::Modified(hash.clone()));
                }
            } else {
                diff.insert(id.clone(), ResourceUpdate::Added(hash.clone()));
            }
        }

        for (id, _) in other.hashes() {
            if self.hash(id).is_none() {
                diff.insert(id.clone(), ResourceUpdate::Removed);
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

#[derive(Debug)]
struct State {
    lump_file: RwLock<LumpFile>,
    watchers: RwLock<HashMap<ResourceId, Box<dyn ResourceWatcher>>>,
}

impl State {
    fn set_lump_file(&self, new_lump_file: LumpFile) {
        let watchers = self.watchers.read();
        let mut lump_file = self.lump_file.write();

        let diff = new_lump_file.diff_hashes(&lump_file);
        *lump_file = new_lump_file;
        for (key, update) in diff {
            if let Some(watcher) = watchers.get(&key) {
                watcher.notify_update(&key, update);
            }
        }
    }

    async fn reload(&self, path: &Path) -> Result<(), LumpFileError> {
        let new_lump_file = LumpFile::read(path).await?;
        let lump_count = new_lump_file.hashes().len();
        info!(?path, ?lump_count, "Loaded lump file source");

        self.set_lump_file(new_lump_file);

        Ok(())
    }

    fn clear(&self) {
        self.set_lump_file(LumpFile::None);
    }
}

#[derive(Debug)]
pub struct LumpFileSource {
    file_path: PathBuf,
    #[allow(unused)]
    file_watcher: Debouncer<notify::RecommendedWatcher, notify_debouncer_full::RecommendedCache>,
    state: Arc<State>,
}

impl LumpFileSource {
    #[inline]
    pub async fn open<P>(path: P) -> Result<Self, LumpFileError>
    where
        P: AsRef<Path>
    {
        Self::open_impl(path.as_ref()).await
    }

    async fn open_impl(path: &Path) -> Result<Self, LumpFileError> {
        let lump_file = LumpFile::read(path).await?;
        let lump_count = lump_file.hashes().len();
        info!(?path, ?lump_count, "Loaded lump file source");
        let file_path = path.to_path_buf();

        let state = Arc::new(State {
            lump_file: RwLock::new(lump_file),
            watchers: RwLock::new(HashMap::new()),
        });

        let cb_file_path = file_path.clone();
        let cb_state = state.clone();
        let mut file_watcher = new_debouncer(
            Duration::from_secs(1),
            None,
            move |result: DebounceEventResult| {
                if let Ok(events) = result {
                    for event in events {
                        match event.kind {
                            notify::EventKind::Create(_) |
                            notify::EventKind::Modify(ModifyKind::Data(_)) |
                            notify::EventKind::Modify(ModifyKind::Name(RenameMode::To)) => {
                                info!(path = ?cb_file_path, "Reloading lump file source");
                                if let Err(error) = future::block_on(cb_state.reload(&cb_file_path)) {
                                    error!(path = ?cb_file_path, ?error, "Failed to reload lump file source");
                                }
                            },
                            notify::EventKind::Remove(_) |
                            notify::EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
                                warn!(path = ?cb_file_path, "lump file source removed, falling back to empty source");
                                cb_state.clear();
                            },
                            _ => {}
                        }
                    }
                }
            },
        )?;
        file_watcher.watch(path, notify::RecursiveMode::NonRecursive)?;

        Ok(Self {
            file_path,
            file_watcher,
            state,
        })
    }
}

#[async_trait]
impl ResourceSource for LumpFileSource {
    fn hash(&self, id: &ResourceId) -> Option<ResourceDataSource> {
        let lump_file = self.state.lump_file.read();
        lump_file.hash(id).map(|s| ResourceDataSource::new(s.clone()))
    }

    async fn load(&self, id: &ResourceId) -> ResourceDataResult {
        let lump_file = self.state.lump_file.read();
        let lump = lump_file.load_lump(&self.file_path, id).await?;
        debug!(path = ?self.file_path, %id, "Loading lump");
        Ok(lump.into())
    }

    fn watch(&self, id: ResourceId, watcher: Box<dyn ResourceWatcher>, _sub_idx: Option<SourceIndex>) {
        self.state.watchers.write().insert(id, watcher);
    }

    fn unwatch(&self, id: &ResourceId) {
        self.state.watchers.write().remove(id);
    }
}

pub struct MarkerLump {}

impl MarkerLump {
    #[inline]
    pub fn loader() -> MarkerLumpLoader {
        MarkerLumpLoader {}
    }
}

pub struct MarkerLumpLoader {}

#[async_trait]
impl ResourceLoader<MarkerLump> for MarkerLumpLoader {
    async fn load(
        &self,
        _data: ResourceReadData,
        _ctx: ResourceLoadContext,
    ) -> Result<Box<MarkerLump>, ResourceLoadError> {
        // Marker lumps don't actually have data
        Ok(Box::new(MarkerLump {}))
    }
}