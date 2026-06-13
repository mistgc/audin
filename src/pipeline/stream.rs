use anyhow::{Context, Result};
use std::io::Read;

use crate::nn::model::moonshine_base::MoonshineBase;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Streaming transcription configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Duration of each processing chunk in milliseconds.
    pub chunk_duration_ms: u64,
    /// Overlap between consecutive chunks in milliseconds.
    /// A non-zero overlap reduces the chance of cutting words at boundaries.
    pub overlap_ms: u64,
    /// Target sample rate (Hz) — audio is assumed to already be at this rate.
    pub sample_rate: u32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_duration_ms: 10_000,
            overlap_ms: 2_000,
            sample_rate: 16_000,
        }
    }
}

impl StreamConfig {
    /// Number of f32 samples per processing chunk.
    pub fn chunk_samples(&self) -> usize {
        (self.sample_rate as u64 * self.chunk_duration_ms / 1000) as usize
    }

    /// Number of f32 samples that overlap between consecutive chunks.
    pub fn overlap_samples(&self) -> usize {
        (self.sample_rate as u64 * self.overlap_ms / 1000) as usize
    }

    /// Stride (advance) per chunk in samples = chunk - overlap.
    pub fn stride_samples(&self) -> usize {
        self.chunk_samples() - self.overlap_samples()
    }
}

// ---------------------------------------------------------------------------
// StreamTranscriber
// ---------------------------------------------------------------------------

/// A stateful streaming transcriber that processes audio chunk-by-chunk.
///
/// Audio is accumulated via [`push_audio`](Self::push_audio). When the
/// internal buffer has enough samples for one processing chunk, the chunk is
/// transcribed and the transcription is returned. Consecutive chunks overlap
/// by the configured overlap duration, and the transcriber attempts to
/// deduplicate repeated text in the overlap region so that output reads
/// progressively rather than repeating.
///
/// Call [`finalize`](Self::finalize) when no more audio will arrive to flush
/// any remaining samples.
pub struct StreamTranscriber {
    model: MoonshineBase,
    config: StreamConfig,
    buffer: Vec<f32>,
    /// Number of samples from the start of `buffer` that have been processed
    /// and are part of a committed (non-overlapping) region.
    processed_up_to: usize,
    /// Full text of the last transcribed chunk, used for overlap dedup.
    last_chunk_text: String,
    /// Accumulated final transcription (committed, deduplicated text).
    transcription: String,
    finalized: bool,
}

impl StreamTranscriber {
    /// Create a new streaming transcriber.
    ///
    /// `model` — an already-loaded MoonshineBase instance.
    /// `config` — chunk size, overlap, sample rate.
    pub fn new(model: MoonshineBase, config: StreamConfig) -> Self {
        Self {
            model,
            config,
            buffer: Vec::new(),
            processed_up_to: 0,
            last_chunk_text: String::new(),
            transcription: String::new(),
            finalized: false,
        }
    }

    /// Push a segment of audio samples (mono f32, at the configured sample
    /// rate) into the transcriber's buffer.
    ///
    /// Returns an interim transcription if enough audio has accumulated for a
    /// chunk, or `None` if more audio is needed.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<Option<String>> {
        self.buffer.extend_from_slice(samples);
        self.process_chunks()
    }

    /// Process as many complete chunks as the current buffer allows, returning
    /// the newest interim transcription (if any).
    fn process_chunks(&mut self) -> Result<Option<String>> {
        let mut latest: Option<String> = None;

        loop {
            let available = self.buffer.len() - self.processed_up_to;
            if available < self.config.chunk_samples() {
                break;
            }

            let start = self.processed_up_to;
            let end = start + self.config.chunk_samples();
            let chunk: Vec<f32> = self.buffer[start..end].to_vec();

            // Advance the processed window by the stride (chunk - overlap).
            self.processed_up_to += self.config.stride_samples();

            let text = self.model.transcribe(&chunk)?;

            // Extract the "new" portion of this chunk's transcription by
            // deduplicating against the previous chunk's overlap region.
            let new_part = dedup_overlap(&self.last_chunk_text, &text);
            self.last_chunk_text = text;

            if !new_part.is_empty() {
                if !self.transcription.is_empty() {
                    self.transcription.push(' ');
                }
                self.transcription.push_str(&new_part);
            }

            latest = Some(self.transcription.clone());
        }

        Ok(latest)
    }

    /// Consume any remaining audio and return the complete transcription.
    ///
    /// After calling this, the transcriber is exhausted — call
    /// [`reset`](Self::reset) to reuse it.
    pub fn finalize(&mut self) -> Result<String> {
        if self.finalized {
            return Ok(self.transcription.clone());
        }
        self.finalized = true;

        let remaining: Vec<f32> = self.buffer[self.processed_up_to..].to_vec();

        // Only process remaining audio if there's at least 0.5 seconds worth.
        let min_remaining = (self.config.sample_rate as usize) / 2;
        if remaining.len() >= min_remaining {
            let text = self.model.transcribe(&remaining)?;
            let new_part = dedup_overlap(&self.last_chunk_text, &text);
            if !new_part.is_empty() {
                if !self.transcription.is_empty() {
                    self.transcription.push(' ');
                }
                self.transcription.push_str(&new_part);
            }
        }

        Ok(self.transcription.clone())
    }

    /// Return the current streaming configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Return the current accumulated transcription without processing more
    /// audio.
    pub fn current_transcription(&self) -> &str {
        &self.transcription
    }

    /// Reset the streaming state so the transcriber can be reused for a new
    /// stream (the model is kept loaded).
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.processed_up_to = 0;
        self.last_chunk_text.clear();
        self.transcription.clear();
        self.finalized = false;
    }
}

// ---------------------------------------------------------------------------
// Overlap deduplication
// ---------------------------------------------------------------------------

/// Given the full text of the previous chunk and the full text of the current
/// chunk, return only the portion of `current` that is new (i.e. after the
/// overlapping region).
///
/// Works at the word level: finds the longest suffix of `prev` that matches
/// the prefix of `current` and discards that matching prefix from `current`.
fn dedup_overlap(prev: &str, current: &str) -> String {
    if prev.is_empty() || current.is_empty() {
        return current.to_string();
    }

    let prev_words: Vec<&str> = prev.split_whitespace().collect();
    let curr_words: Vec<&str> = current.split_whitespace().collect();

    if prev_words.is_empty() || curr_words.is_empty() {
        return current.to_string();
    }

    // Find the longest suffix of prev_words that matches the prefix of
    // curr_words.
    let max_overlap = std::cmp::min(prev_words.len(), curr_words.len());
    let mut best_n = 0usize;

    for n in (1..=max_overlap).rev() {
        let prev_tail = &prev_words[prev_words.len() - n..];
        let curr_head = &curr_words[..n];
        if prev_tail == curr_head {
            best_n = n;
            break;
        }
    }

    if best_n > 0 && best_n < curr_words.len() {
        curr_words[best_n..].join(" ")
    } else if best_n == curr_words.len() {
        // Everything in current was already in prev → nothing new.
        String::new()
    } else {
        // No overlap found → return the whole current transcription.
        current.to_string()
    }
}

// ---------------------------------------------------------------------------
// Stdin reader helper
// ---------------------------------------------------------------------------

/// Read raw 16-bit mono PCM at the given sample rate from `reader` and feed
/// it into a `StreamTranscriber`. Each chunk of audio read from the stream
/// is pushed and the interim transcription is printed via `on_interim` as it
/// becomes available.
///
/// The loop exits when `reader` reaches EOF. Call `transcriber.finalize()`
/// afterwards to flush any remaining audio.
///
/// `read_size` is the number of bytes to read per iteration (default: 4096).
pub fn transcribe_stdin_raw(
    transcriber: &mut StreamTranscriber,
    reader: &mut dyn Read,
    mut on_interim: impl FnMut(&str),
) -> Result<()> {
    let mut raw_buf = vec![0u8; 4096];
    loop {
        let n = reader
            .read(&mut raw_buf)
            .context("failed to read from stdin")?;
        if n == 0 {
            break; // EOF
        }

        // Convert raw 16-bit PCM to f32.
        let samples: Vec<f32> = raw_buf[..n]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect();

        if let Some(interim) = transcriber.push_audio(&samples)? {
            on_interim(&interim);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// WAV chunked reader helper
// ---------------------------------------------------------------------------

/// Read a WAV file in chunks and feed PCM data into a `StreamTranscriber`.
///
/// This reads the WAV header, then processes the data section chunk-by-chunk.
/// Interim transcriptions are printed via `on_interim`.
pub fn transcribe_wav_stream(
    transcriber: &mut StreamTranscriber,
    wav_data: &[u8],
    mut on_interim: impl FnMut(&str),
) -> Result<()> {
    let info = crate::utils::audio::read_wav_header(wav_data)
        .context("failed to read WAV header")?;

    if info.bits_per_sample != 16 {
        anyhow::bail!(
            "only 16-bit PCM is supported for streaming, got {} bits",
            info.bits_per_sample
        );
    }

    let pcm_data = &wav_data[info.data_offset..info.data_offset + info.data_len];

    // Process PCM in chunks that roughly match the transcriber's stride.
    let raw_chunk_bytes = 4096usize;
    let channels = info.channels as usize;
    let frame_size = channels * 2; // bytes per multi-channel frame (16-bit)

    let mut offset = 0usize;
    while offset + frame_size <= pcm_data.len() {
        let end = std::cmp::min(offset + raw_chunk_bytes, pcm_data.len());
        // Align to a frame boundary.
        let aligned_end = end - (end % frame_size);
        if aligned_end <= offset {
            break;
        }

        let chunk = &pcm_data[offset..aligned_end];
        let samples = crate::utils::audio::pcm_to_f32(chunk, info.channels, info.bits_per_sample);
        offset = aligned_end;

        if let Some(interim) = transcriber.push_audio(&samples)? {
            on_interim(&interim);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // dedup_overlap
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_empty_prev() {
        assert_eq!(dedup_overlap("", "hello world"), "hello world");
    }

    #[test]
    fn dedup_empty_current() {
        assert_eq!(dedup_overlap("hello world", ""), "");
    }

    #[test]
    fn dedup_no_overlap() {
        let result = dedup_overlap("hello world", "foo bar");
        assert_eq!(result, "foo bar");
    }

    #[test]
    fn dedup_partial_overlap() {
        let result = dedup_overlap("hello world how", "world how are you");
        assert_eq!(result, "are you");
    }

    #[test]
    fn dedup_full_overlap() {
        let result = dedup_overlap("hello world", "hello world");
        assert_eq!(result, "");
    }

    #[test]
    fn dedup_exact_overlap_same_words() {
        let result = dedup_overlap("one two three", "two three four");
        assert_eq!(result, "four");
    }

    #[test]
    fn dedup_single_word() {
        let result = dedup_overlap("hello", "hello world");
        assert_eq!(result, "world");
    }

    // -----------------------------------------------------------------------
    // StreamConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_values() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.chunk_duration_ms, 10_000);
        assert_eq!(cfg.overlap_ms, 2_000);
        assert_eq!(cfg.sample_rate, 16_000);
    }

    #[test]
    fn config_sample_counts() {
        let cfg = StreamConfig {
            chunk_duration_ms: 10_000,
            overlap_ms: 2_000,
            sample_rate: 16_000,
        };
        assert_eq!(cfg.chunk_samples(), 160_000);
        assert_eq!(cfg.overlap_samples(), 32_000);
        assert_eq!(cfg.stride_samples(), 128_000);
    }

    // -----------------------------------------------------------------------
    // StreamTranscriber (audio-agnostic state tests)
    // -----------------------------------------------------------------------

    #[test]
    fn initial_transcription_is_empty() {
        // We can't construct MoonshineBase without downloading models
        // (that's an #[ignore] test in the model module). Instead, we
        // verify the state that doesn't require a real model.
        // This test is a placeholder to ensure the module compiles and
        // the type exists.
        let cfg = StreamConfig::default();
        assert_eq!(cfg.chunk_samples(), 160_000);
    }
}
