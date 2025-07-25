use std::ffi::CStr;

pub mod lump;
pub mod map;
pub mod patch;

#[inline]
pub fn string_from_bytes_until_nul_checked(bytes: &[u8]) -> String {
    match CStr::from_bytes_until_nul(bytes) {
        Ok(c_str) => c_str.to_string_lossy().to_string(),
        Err(_) => String::from_utf8_lossy(bytes).to_string(),
    }
}