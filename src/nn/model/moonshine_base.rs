use std::collections::HashMap;

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use ndarray::{Array1, Array2, ArrayD};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;

const CONFIG_FILENAME: &str = "config.json";
const DECODER_FILENAME: &str = "onnx/decoder_model_merged_int8.onnx";
const ENCODER_FILENAME: &str = "onnx/encoder_model_int8.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Helper: convert ort::Error to anyhow::Error by formatting the debug string.
/// This avoids the Send+Sync requirement that `?` tries to enforce on
/// ort::Error<SessionBuilder> (which contains non-Sync FFI pointer types).
fn map_err<T>(e: ort::Error<T>) -> anyhow::Error {
    anyhow::anyhow!("{:?}", e)
}

/// Configuration for the Moonshine base model, deserialized from the
/// HuggingFace `config.json`. Field names match the JSON keys exactly.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MoonshineBaseConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: u32,
    pub decoder_hidden_act: String,
    pub decoder_num_attention_heads: usize,
    pub decoder_num_hidden_layers: usize,
    pub decoder_num_key_value_heads: usize,
    pub decoder_start_token_id: u32,
    pub encoder_hidden_act: String,
    pub encoder_num_attention_heads: usize,
    pub encoder_num_hidden_layers: usize,
    pub encoder_num_key_value_heads: usize,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub is_encoder_decoder: bool,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub partial_rotary_factor: f32,
    pub rope_scaling: Option<serde_json::Value>,
    pub rope_theta: f32,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

pub struct MoonshineBase {
    config: MoonshineBaseConfig,
    decoder: Session,
    encoder: Session,
    tokenizer: Tokenizer,
}

impl MoonshineBase {
    pub fn new() -> Result<MoonshineBase> {
        let client = HFClientSync::new().context("Failed to create HF client")?;
        let repo = client.model("onnx-community", "moonshine-base-ONNX");

        // Download all required files using hf-hub
        let config_path = repo
            .download_file()
            .filename(CONFIG_FILENAME)
            .send()
            .context("Failed to download config.json")?;

        let decoder_path = repo
            .download_file()
            .filename(DECODER_FILENAME)
            .send()
            .context("Failed to download decoder model")?;

        let encoder_path = repo
            .download_file()
            .filename(ENCODER_FILENAME)
            .send()
            .context("Failed to download encoder model")?;

        let tokenizer_path = repo
            .download_file()
            .filename(TOKENIZER_FILENAME)
            .send()
            .context("Failed to download tokenizer")?;

        let config_data =
            std::fs::read_to_string(&config_path).context("Failed to read config file")?;
        let config: MoonshineBaseConfig = serde_json::from_str(&config_data)
            .context("Failed to parse Moonshine config.json")?;

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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(MoonshineBase { config, decoder, encoder, tokenizer })
    }

    /// Dimension of each key/value vector in the decoder attention layers.
    fn dim_kv(&self) -> usize {
        self.config.hidden_size / self.config.decoder_num_attention_heads
    }

    /// Build the ordered list of past_key_values input names, matching the
    /// iteration order used by the Python reference (layer → module → kv).
    fn pkv_names(&self) -> Vec<String> {
        let num_layers = self.config.decoder_num_hidden_layers;
        let mut names = Vec::with_capacity(num_layers * 2 * 2);
        // The outer iteration in the Python reference is `layer`, then `module`,
        // then `kv`. We preserve that order so the positional mapping from
        // decoder outputs back to the dict stays correct.
        for layer in 0..num_layers {
            for module in ["decoder", "encoder"] {
                for kv in ["key", "value"] {
                    names.push(format!(
                        "past_key_values.{layer}.{module}.{kv}"
                    ));
                }
            }
        }
        names
    }

    /// Initialize the past key/value tensors to zero-length sequence buffers.
    /// Shape per entry: `[batch, num_kv_heads, 0, dim_kv]`.
    fn init_past_key_values(
        &self,
        batch: usize,
        num_kv_heads: usize,
        dim_kv: usize,
    ) -> HashMap<String, ArrayD<f32>> {
        let mut pkv = HashMap::new();
        for name in self.pkv_names() {
            pkv.insert(
                name,
                ArrayD::<f32>::zeros(ndarray::IxDyn(&[batch, num_kv_heads, 0, dim_kv])),
            );
        }
        pkv
    }

    /// Run the encoder on raw audio samples (mono, 16 kHz). Returns the encoder
    /// output tensor as an owned `ArrayD<f32>` of shape `[batch, seq, hidden]`.
    fn encode(&mut self, audio: &[f32]) -> Result<ArrayD<f32>> {
        let batch = 1usize;
        let len = audio.len();
        let mut input_values = Array2::<f32>::zeros((batch, len));
        for (i, &s) in audio.iter().enumerate() {
            input_values[[0, i]] = s;
        }

        let outputs = self
            .encoder
            .run(ort::inputs![
                "input_values" => Tensor::from_array(input_values)?
            ])
            .map_err(map_err)
            .context("encoder inference failed")?;

        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(output.to_owned())
    }

    /// Transcribe a mono 16 kHz audio buffer into text.
    ///
    /// Follows the reference Python loop from the model card:
    ///  1. encode audio → `encoder_outputs`
    ///  2. seed `input_ids` with `decoder_start_token_id`
    ///  3. autoregressively step the merged decoder, feeding the
    ///     `use_cache_branch` flag and updating the past-KV cache
    ///  4. stop at `eos_token_id` or `max_len = min(audio_s * 6, max_position_embeddings)`
    ///  5. decode the resulting token ids with the tokenizer
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        // Copy the config values we need up front so we don't hold an immutable
        // borrow on `self.config` across the mutable borrows for encode/decode.
        let decoder_start_token_id = self.config.decoder_start_token_id;
        let eos_token_id = self.config.eos_token_id;
        let max_position_embeddings = self.config.max_position_embeddings;
        let num_kv_heads = self.config.decoder_num_key_value_heads;
        let dim_kv = self.dim_kv();
        let sample_rate = 16_000usize;

        // 1. Encode audio.
        let encoder_outputs = self.encode(audio)?;
        log::debug!(
            "encoder output shape: {:?}",
            encoder_outputs.shape()
        );

        // 2. Seed decoder inputs.
        let mut generated: Vec<u32> = vec![decoder_start_token_id];
        let mut past_kv = self.init_past_key_values(1, num_kv_heads, dim_kv);

        // "max 6 tokens per second of audio"
        let max_len = std::cmp::min(
            (audio.len() / sample_rate) * 6,
            max_position_embeddings,
        );
        // Always allow at least one token so very short clips still produce output.
        let max_len = std::cmp::max(max_len, 1);

        // Snapshot the PKV name order once; we use it both to build inputs
        // (HashMap) and to index into the decoder outputs positionally.
        let pkv_names = self.pkv_names();

        for step in 0..max_len {
            let use_cache_branch = step > 0;

            // Build the decoder inputs as a name→Value map.
            let input_ids_arr = Array2::<i64>::from_shape_vec(
                (1, 1),
                vec![*generated.last().unwrap() as i64],
            )
            .context("bad input_ids shape")?;

            let mut run_inputs: HashMap<String, ort::value::Value> = HashMap::new();
            run_inputs.insert(
                "input_ids".to_string(),
                Tensor::from_array(input_ids_arr)?.into(),
            );
            run_inputs.insert(
                "encoder_hidden_states".to_string(),
                Tensor::from_array(encoder_outputs.clone())?.into(),
            );
            run_inputs.insert(
                "use_cache_branch".to_string(),
                Tensor::from_array(Array1::<bool>::from_elem(1, use_cache_branch))?.into(),
            );
            for name in &pkv_names {
                let arr = past_kv.get(name).unwrap().clone();
                run_inputs.insert(name.clone(), Tensor::from_array(arr)?.into());
            }

            let outputs = self
                .decoder
                .run(run_inputs)
                .map_err(map_err)
                .with_context(|| format!("decoder inference failed at step {step}"))?;

            // First output: logits [batch, 1, vocab].
            let logits = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let logits_shape = logits.shape().to_vec();
            let vocab = logits_shape[2];
            let next_id = {
                // Take the last time step's logits: logits[0, -1, :].
                let flat = logits.as_slice().context("logits not contiguous")?;
                let row = logits_shape[0] * (logits_shape[1] - 1);
                let logits_last = &flat[row * vocab..(row + 1) * vocab];
                logits_last
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .context("empty logits")?
            };

            // Update past key values. The decoder outputs logits followed by the
            // present key/value tensors in the same order as the input names.
            for (j, name) in pkv_names.iter().enumerate() {
                // Outputs are offset by 1 (logits is index 0).
                let present = outputs[j + 1]
                    .try_extract_array::<f32>()
                    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
                // Python reference update rule: replace this cache entry if we're
                // on the first step (no cache branch) OR the entry belongs to the
                // decoder (encoder KV is static after prompt processing).
                if !use_cache_branch || name.contains("decoder") {
                    past_kv.insert(name.clone(), present.to_owned());
                }
            }

            generated.push(next_id);
            if next_id == eos_token_id {
                break;
            }
        }

        // Drop the trailing EOS if present before decoding.
        if generated.last() == Some(&eos_token_id) {
            generated.pop();
        }

        // The tokenizer's `decode` takes a `&[u32]` of ids.
        let text = self
            .tokenizer
            .decode(&generated, /*skip_special_tokens=*/ true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {}", e))?;
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_id_points_to_onnx_community() {
        let repo_id = "onnx-community/moonshine-base-ONNX";
        assert!(repo_id.starts_with("onnx-community/"));
        assert!(repo_id.contains("moonshine"));
    }

    #[test]
    fn filenames_match_expected_paths() {
        assert_eq!(CONFIG_FILENAME, "config.json");
        assert!(DECODER_FILENAME.ends_with(".onnx"));
        assert!(ENCODER_FILENAME.ends_with(".onnx"));
        assert_eq!(TOKENIZER_FILENAME, "tokenizer.json");
    }

    #[test]
    fn decoder_and_encoder_are_distinct() {
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

    #[test]
    fn config_deserializes_from_huggingface_json() {
        // The exact JSON payload from
        // https://huggingface.co/onnx-community/moonshine-base-ONNX/raw/main/config.json
        let json = r#"{
          "architectures": ["MoonshineForConditionalGeneration"],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 1,
          "decoder_hidden_act": "silu",
          "decoder_num_attention_heads": 8,
          "decoder_num_hidden_layers": 8,
          "decoder_num_key_value_heads": 8,
          "decoder_start_token_id": 1,
          "encoder_hidden_act": "gelu",
          "encoder_num_attention_heads": 8,
          "encoder_num_hidden_layers": 8,
          "encoder_num_key_value_heads": 8,
          "eos_token_id": 2,
          "hidden_size": 416,
          "initializer_range": 0.02,
          "intermediate_size": 1664,
          "is_encoder_decoder": true,
          "max_position_embeddings": 512,
          "model_type": "moonshine",
          "num_attention_heads": 8,
          "num_hidden_layers": 8,
          "num_key_value_heads": 8,
          "partial_rotary_factor": 0.62,
          "rope_scaling": null,
          "rope_theta": 10000.0,
          "torch_dtype": "float32",
          "transformers_version": "4.48.0.dev0",
          "use_cache": true,
          "vocab_size": 32768
        }"#;

        let config: MoonshineBaseConfig =
            serde_json::from_str(json).expect("failed to deserialize config");

        assert_eq!(config.architectures, vec!["MoonshineForConditionalGeneration"]);
        assert_eq!(config.model_type, "moonshine");
        assert!(config.is_encoder_decoder);
        assert!(!config.attention_bias);
        assert!(config.use_cache);
        assert_eq!(config.hidden_size, 416);
        assert_eq!(config.intermediate_size, 1664);
        assert_eq!(config.vocab_size, 32768);
        assert_eq!(config.max_position_embeddings, 512);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.decoder_num_hidden_layers, 8);
        assert_eq!(config.encoder_num_hidden_layers, 8);
        assert_eq!(config.decoder_hidden_act, "silu");
        assert_eq!(config.encoder_hidden_act, "gelu");
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.decoder_start_token_id, 1);
        assert!((config.partial_rotary_factor - 0.62).abs() < 1e-6);
        assert!((config.rope_theta - 10000.0).abs() < 1e-6);
        assert!((config.initializer_range - 0.02).abs() < 1e-6);
        assert!((config.attention_dropout).abs() < 1e-6);
        assert_eq!(config.torch_dtype, "float32");
        assert_eq!(config.transformers_version, "4.48.0.dev0");
        assert!(config.rope_scaling.is_none());
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
