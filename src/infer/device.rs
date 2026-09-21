//! Device selection for neural inference.

use candle_core::{DType, Device};

use crate::error::{Error, Result};

/// Host preference for accelerator selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    /// Prefer Metal, then CUDA, then CPU.
    #[default]
    Auto,
    /// Apple Metal (requires `metal` feature).
    Metal,
    /// NVIDIA CUDA (requires `cuda` feature).
    Cuda,
    /// Host CPU (f32).
    Cpu,
}

/// Resolve a concrete Candle [`Device`] from a preference.
pub fn resolve_device(request: DeviceRequest) -> Result<Device> {
    match request {
        DeviceRequest::Cpu => Ok(Device::Cpu),
        DeviceRequest::Metal => new_metal(),
        DeviceRequest::Cuda => new_cuda(),
        DeviceRequest::Auto => {
            if let Ok(dev) = new_metal() {
                return Ok(dev);
            }
            if let Ok(dev) = new_cuda() {
                return Ok(dev);
            }
            Ok(Device::Cpu)
        }
    }
}

fn new_metal() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).map_err(|e| Error::Infer(format!("metal unavailable: {e}")))
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(Error::Infer(
            "metal requested but crate built without `metal` feature".into(),
        ))
    }
}

fn new_cuda() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).map_err(|e| Error::Infer(format!("cuda unavailable: {e}")))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::Infer(
            "cuda requested but crate built without `cuda` feature".into(),
        ))
    }
}

/// Resolve weight / activation dtype for Candle loads.
///
/// CUDA may run `bf16`/`fp16` when the checkpoint (or `APOFASI_DTYPE`) asks for
/// it. CPU and Metal stay on `f32` so logits stay aligned with the MLX f32 path
/// and so host CPUs without fast half matmul do not regress.
pub fn weight_dtype(device: &Device, amp_dtype: &str) -> DType {
    let requested = std::env::var("APOFASI_DTYPE")
        .unwrap_or_else(|_| amp_dtype.to_string())
        .to_ascii_lowercase();
    if device.is_cuda() {
        match requested.as_str() {
            "bf16" | "bfloat16" => DType::BF16,
            "fp16" | "f16" | "half" => DType::F16,
            _ => DType::F32,
        }
    } else {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_weight_dtype_stays_f32_even_when_amp_is_bf16() {
        assert_eq!(weight_dtype(&Device::Cpu, "bf16"), DType::F32);
    }
}
