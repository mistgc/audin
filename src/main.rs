use std::env;
use std::io::stdin;

use anyhow::{Context, Result};

use audin::nn::model::moonshine_base::MoonshineBase;
use audin::pipeline::stream::{transcribe_stdin_raw, transcribe_wav_stream, StreamConfig, StreamTranscriber};
use audin::utils::audio;

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(|s| s.as_str()).unwrap_or("audin");

    // ---- Parse CLI arguments -----------------------------------------------
    let stream_mode = args.iter().any(|a| a == "--stream");
    let stdin_mode = args.iter().any(|a| a == "--stdin");
    let chunk_ms: u64 = args
        .iter()
        .position(|a| a == "--chunk-ms")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let overlap_ms: u64 = args
        .iter()
        .position(|a| a == "--overlap-ms")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000);

    // Path argument: the last non-flag argument (or first positional).
    let wav_path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(|s| s.to_string());

    // Dispatch mode.
    //   --stdin            : read raw PCM from stdin (16 kHz, 16-bit mono)
    //   --stream <file>    : read WAV in chunks, progressive output
    //   <file>              : transcribe full WAV (existing behaviour)
    if stdin_mode || (stream_mode && wav_path.is_none()) {
        run_stdin_stream(prog, chunk_ms, overlap_ms)
    } else if stream_mode {
        run_file_stream(&wav_path.unwrap(), chunk_ms, overlap_ms)
    } else if let Some(path) = wav_path {
        run_full_file(&path)
    } else {
        eprintln!("Usage:");
        eprintln!("  {prog} <file.wav>                    Transcribe a WAV file (full file)");
        eprintln!("  {prog} --stream <file.wav>           Stream-transcribe a WAV file (progressive)");
        eprintln!("  {prog} --stdin                      Stream-transcribe raw PCM from stdin");
        eprintln!("  {prog} --stdin --chunk-ms 5000      Custom chunk duration (default 10000 ms)");
        eprintln!("  {prog} --stdin --overlap-ms 1000    Custom overlap (default 2000 ms)");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Mode 1 — Full-file transcription (existing behaviour)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Mode 2 — Stream-transcribe a WAV file (chunked, progressive output)
// ---------------------------------------------------------------------------

fn run_file_stream(wav_path: &str, chunk_ms: u64, overlap_ms: u64) -> Result<()> {
    let wav_data =
        std::fs::read(wav_path).with_context(|| format!("failed to read WAV file: {wav_path}"))?;

    let info = audio::read_wav_header(&wav_data)
        .with_context(|| format!("failed to parse WAV header: {wav_path}"))?;

    println!(
        "Stream mode — {} Hz, {} channels, {} bits/sample",
        info.sample_rate, info.channels, info.bits_per_sample
    );
    println!("Chunk: {chunk_ms} ms, Overlap: {overlap_ms} ms");

    // Resample if needed (load the full audio for now — we can read PCM
    // subset later; for streaming we apply resample to each chunk).
    let target_rate = 16_000u32;
    let sample_rate = info.sample_rate;

    println!("Loading Moonshine base model...");
    let model = MoonshineBase::new()?;

    let config = StreamConfig {
        chunk_duration_ms: chunk_ms,
        overlap_ms,
        sample_rate: target_rate,
    };
    let mut transcriber = StreamTranscriber::new(model, config);

    if sample_rate == target_rate {
        // Native rate — stream chunk-by-chunk.
        transcribe_wav_stream(&mut transcriber, &wav_data, |interim| {
            println!("\rInterim: {interim}");
        })?;
    } else {
        // Need to resample — read full PCM, then feed in chunks for
        // progressive output.
        println!("Resampling {sample_rate} Hz -> {target_rate} Hz");
        let (full_audio, _) = audio::load_wav_f32(wav_path)?;
        let audio_16k = audio::resample_linear(&full_audio, sample_rate, target_rate);
        let stride = transcriber.config().stride_samples();
        let mut pos = 0usize;
        while pos < audio_16k.len() {
            let end = usize::min(pos + stride, audio_16k.len());
            if end - pos < stride / 4 {
                break; // leave tiny remainder for finalize
            }
            if let Some(text) = transcriber.push_audio(&audio_16k[pos..end])? {
                println!("Interim: {text}");
            }
            pos = end;
        }
    }

    let final_text = transcriber.finalize()?;
    println!("Transcription: {final_text}");

    Ok(())
}

// ---------------------------------------------------------------------------
// Mode 3 — Stream raw PCM from stdin (16 kHz, 16-bit signed, mono)
// ---------------------------------------------------------------------------

fn run_stdin_stream(prog: &str, chunk_ms: u64, overlap_ms: u64) -> Result<()> {
    eprintln!(
        "{prog}: reading raw 16-bit PCM at 16 kHz from stdin (Ctrl-D to end)"
    );
    eprintln!("Chunk: {chunk_ms} ms, Overlap: {overlap_ms} ms");

    eprintln!("Loading Moonshine base model...");
    let model = MoonshineBase::new()?;

    let config = StreamConfig {
        chunk_duration_ms: chunk_ms,
        overlap_ms,
        sample_rate: 16_000,
    };
    let mut transcriber = StreamTranscriber::new(model, config);

    transcribe_stdin_raw(&mut transcriber, &mut stdin().lock(), |interim| {
        eprint!("\rInterim: {interim}");
    })?;

    let final_text = transcriber.finalize()?;
    eprintln!(); // newline after final interim line
    println!("Transcription: {final_text}");

    Ok(())
}
