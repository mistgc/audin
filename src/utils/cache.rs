use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::utils::network;

/// Returns the cache directory for model files.
/// Uses platform-specific cache dir (e.g., ~/.cache/audin on Linux).
pub fn cache_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .context("Failed to determine cache directory")?
        .join("audin")
        .join("models");
    std::fs::create_dir_all(&dir).context("Failed to create cache directory")?;
    Ok(dir)
}

/// Ensures a model file is downloaded and cached. Returns the path to the cached file.
pub fn ensure_file(url: &str, filename: &str) -> Result<PathBuf> {
    let dir = cache_dir()?;
    let path = dir.join(filename);

    if !path.exists() {
        log::info!("The {filename} is not exists. Downloading {filename} from {url}");
        network::blocking::download_with_progress(url, &path)?;
    }

    Ok(path)
}
