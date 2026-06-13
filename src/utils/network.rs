use anyhow::Result;
use std::path::Path;

pub async fn download(url: &str, dest: &str) -> Result<()> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    std::fs::write(dest, bytes)?;

    Ok(())
}

pub async fn download_as_bytes(url: &str) -> Result<Vec<u8>> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;

    Ok(bytes.to_vec())
}

pub mod blocking {
    use super::*;
    use anyhow::{Context, Result};
    use indicatif::{ProgressBar, ProgressStyle};
    use std::io::Read;

    pub fn download(url: &str, dest: &str) -> Result<()> {
        let bytes = download_as_bytes(url)?;
        std::fs::write(dest, bytes)?;
        Ok(())
    }

    pub fn download_as_bytes(url: &str) -> Result<Vec<u8>> {
        let response = reqwest::blocking::get(url)
            .context("Failed to send request")?;
        let bytes = response.bytes()?;
        Ok(bytes.to_vec())
    }

    /// Downloads a file with progress bar. If `dest` already exists, skips download.
    pub fn download_with_progress(url: &str, dest: &Path) -> Result<()> {
        if dest.exists() {
            println!("Using cached file: {}", dest.display());
            return Ok(());
        }

        let mut response = reqwest::blocking::get(url)
            .context("Failed to send request")?;

        let total_size = response
            .content_length()
            .unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );

        let mut file = std::fs::File::create(dest)
            .context("Failed to create destination file")?;

        let mut downloaded: u64 = 0;
        let mut buffer = [0u8; 8192];

        loop {
            let n = response.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut file, &buffer[..n])?;
            downloaded += n as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message("Download complete");
        Ok(())
    }
}
