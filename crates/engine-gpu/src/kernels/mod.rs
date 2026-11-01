//! Poseidon2 mining kernels.
//!
//! Same `mining_main` binding numbers; must stay bit-exact with `pow_core`.
//! Apple uses uniform bindings for inputs 1..4; other kernels use storage.
//! Input word layouts are identical, with dispatch config padded to 16 bytes.
//!
//! - Apple Metal + `SHADER_INT64` → Apple Metal u64 (`mining_u64_apple.wgsl`)
//! - other GPUs + `SHADER_INT64` → native u64 (`mining_u64.wgsl`)
//! - no `SHADER_INT64` → 32-bit fallback (`mining.wgsl`)

use std::borrow::Cow;
use std::sync::LazyLock;

use pow_core::JobContext;
use primitive_types::U512;

static APPLE_SHADER_WITH_HARNESS: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}\n{}",
        include_str!("mining_u64_apple.wgsl"),
        include_str!("mining_u64_apple_test_harness.wgsl")
    )
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    U32,
    Default,
    Apple,
}

impl Kernel {
    pub const fn needs_int64(self) -> bool {
        !matches!(self, Self::U32)
    }

    /// True when the host must byte-swap each target u32 before GPU upload (Apple compare path).
    pub const fn pre_bswap_target_upload(self) -> bool {
        matches!(self, Self::Apple)
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::Default => "u64",
            Self::Apple => "u64-apple",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::U32 => "32-bit",
            Self::Default => "native-u64",
            Self::Apple => "native-u64 Apple Metal",
        }
    }

    /// WGSL compiled into the mining pipeline (production; Apple excludes test harness).
    pub const fn mining_source(self) -> &'static str {
        match self {
            Self::U32 => include_str!("mining.wgsl"),
            Self::Default => include_str!("mining_u64.wgsl"),
            Self::Apple => include_str!("mining_u64_apple.wgsl"),
        }
    }

    /// WGSL for component tests (Apple includes u32 compat harness).
    pub fn test_source(self) -> Cow<'static, str> {
        match self {
            Self::Apple => Cow::Owned(APPLE_SHADER_WITH_HARNESS.clone()),
            other => Cow::Borrowed(other.mining_source()),
        }
    }

    /// Target buffer words for this kernel (16 u32s, little-endian U512 limbs).
    pub fn target_u32s_upload(self, target: U512) -> [u32; 16] {
        let bytes = target.to_little_endian();
        let mut words = [0u32; 16];
        for i in 0..16 {
            let chunk = &bytes[i * 4..(i + 1) * 4];
            words[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        if self.pre_bswap_target_upload() {
            for w in &mut words {
                *w = w.swap_bytes();
            }
        }
        words
    }

    /// Convenience wrapper around [`JobContext::target`].
    pub fn job_target_u32s(self, ctx: &JobContext) -> [u32; 16] {
        self.target_u32s_upload(ctx.target)
    }

    pub fn for_adapter(adapter: &wgpu::Adapter) -> Self {
        Self::for_adapter_info(&adapter.get_info(), adapter.features())
    }

    pub fn for_adapter_info(info: &wgpu::AdapterInfo, features: wgpu::Features) -> Self {
        if !features.contains(wgpu::Features::SHADER_INT64) {
            return Self::U32;
        }
        if info.backend == wgpu::Backend::Metal {
            Self::Apple
        } else {
            Self::Default
        }
    }

    pub const fn all() -> &'static [Self] {
        &[Self::U32, Self::Default, Self::Apple]
    }

    /// Backward-compatible alias for [`Self::test_source`].
    pub fn source(self) -> Cow<'static, str> {
        self.test_source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(backend: wgpu::Backend) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "test".into(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend,
        }
    }

    #[test]
    fn metal_with_int64_selects_apple() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Metal), wgpu::Features::SHADER_INT64),
            Kernel::Apple
        );
    }

    #[test]
    fn vulkan_with_int64_selects_default() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Vulkan), wgpu::Features::SHADER_INT64),
            Kernel::Default
        );
    }

    #[test]
    fn dx12_with_int64_selects_default() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Dx12), wgpu::Features::SHADER_INT64),
            Kernel::Default
        );
    }

    #[test]
    fn no_int64_falls_back_to_u32() {
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Metal), wgpu::Features::empty()),
            Kernel::U32
        );
        assert_eq!(
            Kernel::for_adapter_info(&info(wgpu::Backend::Vulkan), wgpu::Features::empty()),
            Kernel::U32
        );
    }

    #[test]
    fn apple_test_source_includes_harness() {
        let src = Kernel::Apple.test_source();
        assert!(src.contains("fn hash_squeeze_twice"));
        assert!(!Kernel::Apple.mining_source().contains("fn hash_squeeze_twice"));
    }
}
