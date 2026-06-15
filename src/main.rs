use std::env;

use anyhow::{Context, Result};

use audin::nn::model::moonshine_base::MoonshineBase;
use audin::utils::audio;

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(|s| s.as_str()).unwrap_or("audin");

    // Path argument: the last non-flag argument (or first positional).
    let wav_path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(|s| s.to_string());

    if let Some(path) = wav_path {
        run_full_file(&path)
    } else {
        eprintln!("Usage:");
        eprintln!("  {prog} <file.wav>    Transcribe a WAV file");
        std::process::exit(1);
    }
}

fn run_full_file(wav_path: &str) -> Result<()> {
    let (audio, source_rate) =
        audio::load_wav_f32(wav_path).with_context(|| format!("failed to load WAV: {wav_path}"))?;

    println!(
        "Loaded {} audio samples at {} Hz ({:.2}s)",
        audio.len(),
        source_rate,
        audio.len() as f64 / source_rate as f64,
    );

    let target_rate = 16_000u32;
    let audio = if source_rate == target_rate {
        audio
    } else {
        println!("Resampling {source_rate} Hz -> {target_rate} Hz");
        audio::resample_linear(&audio, source_rate, target_rate)
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
    println!("Transcription: {text}");

    Ok(())
}
