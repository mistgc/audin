use anyhow::{Context, Result};
use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::utils::cache::ensure_file;

const DECODER_CHECKPOINT_URL: &str = "https://huggingface.co/onnx-community/moonshine-base-ONNX/resolve/main/onnx/decoder_model_fp16.onnx";
const ENCODER_CHECKPOINT_URL: &str = "https://huggingface.co/onnx-community/moonshine-base-ONNX/resolve/main/onnx/encoder_model_fp16.onnx";
const DECODER_FILENAME: &str = "decoder_model_fp16.onnx";
const ENCODER_FILENAME: &str = "encoder_model_fp16.onnx";

/// Helper: convert ort::Error to anyhow::Error by formatting the debug string.
/// This avoids the Send+Sync requirement that `?` tries to enforce on
/// ort::Error<SessionBuilder> (which contains non-Sync FFI pointer types).
fn map_err<T>(e: ort::Error<T>) -> anyhow::Error {
    anyhow::anyhow!("{:?}", e)
}

pub struct MoonshineBase {
    decoder: Session,
    encoder: Session,
}

impl MoonshineBase {
    pub fn new() -> Result<MoonshineBase> {
        let decoder_path = ensure_file(DECODER_CHECKPOINT_URL, DECODER_FILENAME)?;
        let encoder_path = ensure_file(ENCODER_CHECKPOINT_URL, ENCODER_FILENAME)?;

        let decoder_data =
            std::fs::read(&decoder_path).context("Failed to read decoder model file")?;
        let encoder_data =
            std::fs::read(&encoder_path).context("Failed to read encoder model file")?;

        let decoder = Session::builder()
            .map_err(map_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(map_err)?
            .with_intra_threads(4)
            .map_err(map_err)?
            .commit_from_memory(&decoder_data)
            .map_err(map_err)?;
        let encoder = Session::builder()
            .map_err(map_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(map_err)?
            .with_intra_threads(4)
            .map_err(map_err)?
            .commit_from_memory(&encoder_data)
            .map_err(map_err)?;

        Ok(MoonshineBase { decoder, encoder })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_url_points_to_onnx_community() {
        assert!(DECODER_CHECKPOINT_URL.starts_with("https://huggingface.co/onnx-community/"));
        assert!(DECODER_CHECKPOINT_URL.ends_with(".onnx"));
    }

    #[test]
    fn encoder_url_points_to_onnx_community() {
        assert!(ENCODER_CHECKPOINT_URL.starts_with("https://huggingface.co/onnx-community/"));
        assert!(ENCODER_CHECKPOINT_URL.ends_with(".onnx"));
    }

    #[test]
    fn filenames_match_url_suffix() {
        // The cached filenames should correspond to the last path segment of each URL
        // so that `ensure_file` stores them under a predictable name.
        assert!(DECODER_CHECKPOINT_URL.ends_with(DECODER_FILENAME));
        assert!(ENCODER_CHECKPOINT_URL.ends_with(ENCODER_FILENAME));
    }

    #[test]
    fn decoder_and_encoder_are_distinct() {
        assert_ne!(DECODER_CHECKPOINT_URL, ENCODER_CHECKPOINT_URL);
        assert_ne!(DECODER_FILENAME, ENCODER_FILENAME);
    }

    #[test]
    fn map_err_preserves_debug_representation() {
        // Build a real ort::Error by asking the builder to commit from empty bytes
        // (no valid ONNX model can be that small).
        let result = Session::builder()
            .unwrap()
            .commit_from_memory(&[]);
        let err = result.unwrap_err();
        let mapped = map_err(err);

        // The mapped error should carry a non-empty message derived from the
        // original ort::Error's Debug output.
        let msg = format!("{:#}", mapped);
        assert!(!msg.is_empty());
    }

    #[test]
    fn session_builder_can_be_created_and_configured() {
        // Verify that the ort session builder chain used in `new()` doesn't panic
        // up to the point of actually loading a model.
        let mut builder = Session::builder()
            .expect("failed to create session builder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("failed to set optimization level")
            .with_intra_threads(4)
            .expect("failed to set intra threads");

        // Committing from empty bytes should fail (no valid ONNX model), but the
        // builder itself should be usable — that's what we're validating here.
        let result = builder.commit_from_memory(&[]);
        assert!(result.is_err());
    }

    /// Exercises the exact same bytes→session pipeline that `MoonshineBase::new()`
    /// uses, but against an on-disk temp file with bogus content — verifies the
    /// full plumbing (fs::read → builder chain → commit_from_memory) runs and
    /// rejects invalid ONNX data with a sensible error.
    #[test]
    fn load_model_from_file_rejects_invalid_onnx() {
        use std::io::Write;

        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let path = dir.path().join("bogus.onnx");

        // Write some bytes that are emphatically not a valid ONNX model.
        let mut f = std::fs::File::create(&path).expect("failed to create temp file");
        f.write_all(b"not an onnx model").expect("failed to write temp file");
        drop(f);

        // Mirror the load path from `MoonshineBase::new()`.
        let data = std::fs::read(&path).expect("failed to read model file");
        assert_eq!(data, b"not an onnx model");

        let mut builder = Session::builder()
            .expect("failed to create session builder")
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .expect("failed to set optimization level")
            .with_intra_threads(4)
            .expect("failed to set intra threads");

        let err = builder
            .commit_from_memory(&data)
            .expect_err("commit_from_memory should fail for non-ONNX bytes");

        // The error should come through `map_err` cleanly — verify it produces
        // a non-empty anyhow message, same as the real constructor would.
        let anyhow_err = map_err(err);
        let msg = format!("{:#}", anyhow_err);
        assert!(!msg.is_empty(), "expected a non-empty error message");
    }

    /// Full construction requires downloading ~hundreds of MB from HuggingFace
    /// and a working ONNX Runtime installation, so it's gated behind `#[ignore]`.
    /// Run explicitly with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn new_loads_both_sessions() {
        let model = MoonshineBase::new().expect("failed to construct MoonshineBase");
        // If we got here, both sessions were loaded successfully. We can't
        // inspect the private fields directly, but the fact that construction
        // didn't error is the contract we're verifying.
        let _ = model;
    }
}
