//! ONNX Runtime ModernBERT encoder (CPU Scale-3 path).
//!
//! The typed head / scorer stay on Candle. Only the heavy bidirectional encoder
//! runs in ORT, which is what closes the warm CPU latency gate.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{Device, Result as CandleResult, Tensor};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;

use crate::error::{Error, Result};

/// Intra-op threads for the CPU encoder session.
///
/// `ORT_INTRA_THREADS` wins when it is a positive integer. Otherwise the
/// session uses the host's available parallelism. A fixed bench-machine count
/// would overfit one CPU and starve or oversubscribe every other host.
fn ort_intra_threads() -> usize {
    if let Ok(raw) = std::env::var("ORT_INTRA_THREADS") {
        if let Ok(n) = raw.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// ORT session wrapping an exported ModernBERT encoder (`input_ids`,
/// `attention_mask` → `last_hidden_state`).
pub struct OrtEncoder {
    session: Mutex<Session>,
    hidden_size: usize,
}

impl OrtEncoder {
    /// Load `encoder.onnx` produced by `scripts/ort_encoder_probe.py`.
    pub fn load(path: impl AsRef<Path>, hidden_size: usize) -> Result<Self> {
        let path = path.as_ref();
        let threads = ort_intra_threads();
        let session = Session::builder()
            .map_err(|e| Error::Checkpoint(format!("ort session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| Error::Checkpoint(format!("ort opt level: {e}")))?
            .with_intra_threads(threads)
            .map_err(|e| Error::Checkpoint(format!("ort intra threads: {e}")))?
            .commit_from_file(path)
            .map_err(|e| Error::Checkpoint(format!("ort load {}: {e}", path.display())))?;
        Ok(Self {
            session: Mutex::new(session),
            hidden_size,
        })
    }

    /// Run the encoder from packed host buffers (`ids`/`mask` length `b*l`).
    pub fn forward_packed(
        &self,
        ids: &[u32],
        mask: &[u32],
        b: usize,
        l: usize,
    ) -> CandleResult<Tensor> {
        if ids.len() != b * l || mask.len() != b * l {
            return Err(candle_core::Error::Msg(format!(
                "ort packed len mismatch ids={} mask={} b*l={}",
                ids.len(),
                mask.len(),
                b * l
            )));
        }
        let ids_i64: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
        let mask_i64: Vec<i64> = mask.iter().map(|&v| v as i64).collect();

        let ids_ort = OrtTensor::from_array(([b, l], ids_i64.into_boxed_slice()))
            .map_err(|e| candle_core::Error::Msg(format!("ort ids tensor: {e}")))?;
        let mask_ort = OrtTensor::from_array(([b, l], mask_i64.into_boxed_slice()))
            .map_err(|e| candle_core::Error::Msg(format!("ort mask tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| candle_core::Error::Msg("ort session mutex poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_ort,
                "attention_mask" => mask_ort
            ])
            .map_err(|e| candle_core::Error::Msg(format!("ort run: {e}")))?;

        let hidden = outputs
            .get("last_hidden_state")
            .ok_or_else(|| candle_core::Error::Msg("ort missing last_hidden_state".into()))?;
        let (_shape, data) = hidden
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("ort extract: {e}")))?;
        if data.len() != b * l * self.hidden_size {
            return Err(candle_core::Error::Msg(format!(
                "ort hidden len {} != {}*{}*{}",
                data.len(),
                b,
                l,
                self.hidden_size
            )));
        }
        Tensor::from_vec(data.to_vec(), (b, l, self.hidden_size), &Device::Cpu)
    }

    /// Run the encoder. `ids` / `mask` are `[B, L]` u32 Candle tensors on CPU.
    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> CandleResult<Tensor> {
        let (b, l) = ids.dims2()?;
        let ids_flat = ids.flatten_all()?.to_vec1::<u32>()?;
        let mask_flat = mask.flatten_all()?.to_vec1::<u32>()?;
        self.forward_packed(&ids_flat, &mask_flat, b, l)
    }
}
