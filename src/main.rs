use anyhow::{Context, Result};
use std::env;

use audin::nn::model::moonshine_base::MoonshineBase;

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-wav-file>", args[0]);
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let (audio, source_rate) = load_wav_f32(wav_path)
        .with_context(|| format!("failed to load WAV file: {}", wav_path))?;

    println!(
        "Loaded {} audio samples at {} Hz ({:.2}s)",
        audio.len(),
        source_rate,
        audio.len() as f64 / source_rate as f64,
    );

    // Resample to 16 kHz if needed — Moonshine expects 16 kHz input.
    let target_rate: u32 = 16_000;
    let audio = if source_rate == target_rate {
        audio
    } else {
        println!("Resampling {} Hz -> {} Hz", source_rate, target_rate);
        resample_linear(&audio, source_rate, target_rate)
    };

    println!(
        "Final audio: {} samples ({:.2}s)",
        audio.len(),
        audio.len() as f64 / target_rate as f64,
    );

    println!("Loading Moonshine base model...");
    let mut model = MoonshineBase::new()?;

    println!("Transcribing...");
    let text = model.transcribe(&audio)?;
    println!("Transcription: {}", text);

    Ok(())
}

/// Load a mono 16-bit PCM WAV file as f32 samples normalized to [-1, 1].
///
/// Returns `(samples, sample_rate)` so the caller can resample if needed.
fn load_wav_f32(path: &str) -> Result<(Vec<f32>, u32)> {
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

/// Resample a mono f32 buffer from `from_rate` to `to_rate` using linear
/// interpolation. Good enough for downsampling speech to 16 kHz — not a
/// replacement for a polyphase / sinc resampler when quality matters.
fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
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
