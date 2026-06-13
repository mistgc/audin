use anyhow::{Context, Result};

/// Load a mono 16-bit PCM WAV file as f32 samples normalized to [-1, 1].
///
/// Returns `(samples, sample_rate)` so the caller can resample if needed.
pub fn load_wav_f32(path: &str) -> Result<(Vec<f32>, u32)> {
    let data = std::fs::read(path).context("failed to read file")?;

    // RIFF header: "RIFF" <size> "WAVE"
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        anyhow::bail!("not a valid RIFF/WAV file");
    }

    // Scan for the "fmt " and "data" chunks — they usually appear in this
    // order at the start, but some encoders insert extras (e.g. "LIST").
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, sample_rate, bits_per_sample)
    let mut pcm_data: Option<&[u8]> = None;

    let mut pos = 12;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        let chunk_start = pos + 8;
        let chunk_end = chunk_start + chunk_size;
        if chunk_end > data.len() {
            break;
        }
        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    anyhow::bail!("fmt chunk too small");
                }
                let format = u16::from_le_bytes([data[chunk_start], data[chunk_start + 1]]);
                let channels = u16::from_le_bytes([data[chunk_start + 2], data[chunk_start + 3]]);
                let sample_rate = u32::from_le_bytes([
                    data[chunk_start + 4],
                    data[chunk_start + 5],
                    data[chunk_start + 6],
                    data[chunk_start + 7],
                ]);
                let bits_per_sample =
                    u16::from_le_bytes([data[chunk_start + 14], data[chunk_start + 15]]);
                fmt = Some((format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                pcm_data = Some(&data[chunk_start..chunk_end]);
            }
            _ => {}
        }
        // Chunks are padded to an even number of bytes.
        pos = chunk_end + (chunk_size & 1);
    }

    let (format, channels, sample_rate, bits_per_sample) =
        fmt.context("missing fmt chunk in WAV file")?;
    let pcm = pcm_data.context("missing data chunk in WAV file")?;

    if format != 1 {
        anyhow::bail!("only PCM WAV (format 1) is supported, got format {}", format);
    }
    if bits_per_sample != 16 {
        anyhow::bail!("only 16-bit PCM is supported, got {} bits", bits_per_sample);
    }

    let samples: Vec<f32> = pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();

    // Downmix to mono by averaging channels.
    let channels = channels as usize;
    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, sample_rate))
}

/// Load raw 16-bit signed little-endian PCM bytes as f32 samples normalized
/// to [-1, 1]. Assumes mono; if the stream is multi-channel the caller must
/// downmix separately.
pub fn load_raw_pcm_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect()
}

/// Resample a mono f32 buffer from `from_rate` to `to_rate` using linear
/// interpolation. Good enough for downsampling speech to 16 kHz — not a
/// replacement for a polyphase / sinc resampler when quality matters.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = samples[idx.min(samples.len() - 1)];
        let s1 = samples[(idx + 1).min(samples.len() - 1)];
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}

/// Read a WAV file's header to extract sample rate and data offset/length
/// without loading the full PCM into memory. Useful for streaming reads.
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data_offset: usize,
    pub data_len: usize,
}

pub fn read_wav_header(data: &[u8]) -> Result<WavInfo> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        anyhow::bail!("not a valid RIFF/WAV file");
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data_offset = 0usize;
    let mut data_len = 0usize;

    let mut pos = 12;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        let chunk_start = pos + 8;
        let chunk_end = chunk_start + chunk_size;
        if chunk_end > data.len() {
            break;
        }
        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    anyhow::bail!("fmt chunk too small");
                }
                let format = u16::from_le_bytes([data[chunk_start], data[chunk_start + 1]]);
                let channels = u16::from_le_bytes([data[chunk_start + 2], data[chunk_start + 3]]);
                let sample_rate = u32::from_le_bytes([
                    data[chunk_start + 4],
                    data[chunk_start + 5],
                    data[chunk_start + 6],
                    data[chunk_start + 7],
                ]);
                let bits_per_sample =
                    u16::from_le_bytes([data[chunk_start + 14], data[chunk_start + 15]]);
                fmt = Some((format, channels, sample_rate, bits_per_sample));
            }
            b"data" => {
                data_offset = chunk_start;
                data_len = chunk_size;
            }
            _ => {}
        }
        pos = chunk_end + (chunk_size & 1);
    }

    let (format, channels, sample_rate, bits_per_sample) =
        fmt.context("missing fmt chunk in WAV file")?;
    if format != 1 {
        anyhow::bail!("only PCM WAV (format 1) is supported, got format {}", format);
    }

    Ok(WavInfo {
        sample_rate,
        channels,
        bits_per_sample,
        data_offset,
        data_len,
    })
}

/// Parse a segment of raw PCM data (from a WAV data chunk) into f32 samples,
/// handling channel count and bit depth.
pub fn pcm_to_f32(data: &[u8], channels: u16, bits_per_sample: u16) -> Vec<f32> {
    let channels = channels as usize;
    match bits_per_sample {
        16 => {
            let all_samples: Vec<f32> = data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
            if channels == 1 {
                all_samples
            } else {
                all_samples
                    .chunks(channels)
                    .map(|c| c.iter().sum::<f32>() / channels as f32)
                    .collect()
            }
        }
        _ => Vec::new(), // caller should check
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_raw_pcm_roundtrip() {
        // A single 16-bit sample at max amplitude (32767)
        let raw = 32767i16.to_le_bytes().to_vec();
        let samples = load_raw_pcm_f32(&raw);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn load_raw_pcm_silence() {
        let raw = vec![0u8; 1024]; // 512 samples of silence
        let samples = load_raw_pcm_f32(&raw);
        assert_eq!(samples.len(), 512);
        assert!(samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn resample_identity() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let output = resample_linear(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn resample_downsample() {
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0];
        // Downsample 16k -> 8k → roughly half the length
        let output = resample_linear(&input, 16000, 8000);
        assert!(output.len() <= input.len() / 2 + 1);
    }

    #[test]
    fn read_wav_header_from_real_file() {
        let path = "fixtures/voice-sample.wav";
        let data = std::fs::read(path).expect("failed to read fixture");
        let info = read_wav_header(&data).expect("failed to parse header");
        assert!(info.sample_rate > 0);
        assert!(info.channels >= 1);
        assert_eq!(info.bits_per_sample, 16);
        assert!(info.data_offset >= 44);
        assert!(info.data_len > 0);
    }

    #[test]
    fn pcm_to_f32_stereo_to_mono() {
        // Two sample frames: each frame has 2 channels, 2 bytes per channel
        // Frame 1: left=32767, right=0
        // Frame 2: left=-32768, right=32767
        let mut raw = Vec::new();
        raw.extend_from_slice(&32767i16.to_le_bytes());
        raw.extend_from_slice(&0i16.to_le_bytes());
        raw.extend_from_slice(&(-32768i16).to_le_bytes());
        raw.extend_from_slice(&32767i16.to_le_bytes());

        let mono = pcm_to_f32(&raw, 2, 16);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 0.001);
        assert!((mono[1] - 0.0).abs() < 0.001);
    }
}
