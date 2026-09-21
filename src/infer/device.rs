//! Device selection for neural inference.

use candle_core::Device;

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
