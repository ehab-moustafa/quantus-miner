#![deny(rust_2018_idioms)]

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, sys as nvrtc_sys, CompileOptions, Ptx};
use engine_cpu::{CancelCheck, Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use pow_core::{format_hashrate, format_u512, JobContext};
use primitive_types::U512;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const KERNEL_SRC: &str = include_str!("kernels/mining.cu");
const DEFAULT_THREADS_PER_BLOCK: u32 = 256;
const DEFAULT_MAX_BLOCKS: u32 = 4096;
/// Inner-loop span where nonce-line `pk[14]` is constant (Idea B / #110).
const DEFAULT_K_SEGMENT: u32 = 1 << 26;
/// Lowest candidate index (`u32::MAX` = none), then the count of nonces that
/// candidate threads left unevaluated. Both are only ever updated atomically.
/// Candidate count, then up to `MAX_HITS` candidate indices in claim order.
const MAX_HITS: usize = 8;
const RESULTS_U32S: usize = 1 + MAX_HITS;

const NONCE_LINE_K_V: [i32; 8] = [4, -17, 11, -3, -4, 17, -11, 3];

#[repr(C)]
#[derive(Clone, Copy)]
struct MiningParams {
    prestate: [u32; 24],
    start_nonce: [u32; 16],
    difficulty_target: [u32; 16],
    dispatch_config: [u32; 3],
    nonce_line_k_base: [u32; 8],
    nonce_line_v: [i32; 8],
    nonce_line_pk: [u64; 14],
    nonce_line_t_base: u32,
    nonce_line_pad: u32,
}

unsafe impl cudarc::driver::DeviceRepr for MiningParams {}
unsafe impl cudarc::driver::ValidAsZeroBits for MiningParams {}

fn cuda_use_nonce_line() -> bool {
    miner_cuda_nonce_line_env_enabled()
}

fn mining_kernel_name() -> &'static str {
    if cuda_use_nonce_line() {
        "mining_main_nonce_line"
    } else {
        "mining_main"
    }
}

struct CudaDevice {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    name: String,
}

/// Params H2D finished on the copy stream; compute waits before launch.
struct StagedSegmentParams {
    batch_start: U512,
    slot: u8,
}

struct WorkerBuffers {
    engine_id: usize,
    device_index: usize,
    stream: Arc<CudaStream>,
    results: CudaSlice<u32>,
    midstate: CudaSlice<u32>,
    start_nonce: CudaSlice<u32>,
    hashes: CudaSlice<u32>,
    mine: CudaFunction,
    hash: CudaFunction,
    busy: Duration,
    busy_since: Instant,
    params_dev: [CudaSlice<MiningParams>; 2],
    params_h2d_ready: [Option<CudaEvent>; 2],
    copy_stream: Option<Arc<CudaStream>>,
    param_slot: u8,
    staged_first_segment: Option<StagedSegmentParams>,
}

/// Native CUDA mining engine.
///
/// Search contract: every launch evaluates its whole nonce rectangle; no
/// thread stops early. Up to `MAX_HITS` candidates per launch are recorded
/// (any more are counted and skipped with a warning, which only happens at
/// difficulties far below real mining), and the host recomputes each one with
/// the exact CPU hash, returning the lowest that is really below the target.
/// The kernel's Goldilocks reduction skips two carry corrections (`reduce128`
/// in `kernels/field.cuh`): each of the roughly 1,470 field multiplies per
/// hash can be off by EPS mod p with probability about 2^-33, so about one
/// nonce in three million is hashed wrong on the GPU. A wrong hash that lands
/// below the target is rejected by the CPU check; a wrong hash for a nonce that
/// is actually valid is missed and the range reports `Exhausted`. The expected
/// loss is about 3e-7 of solutions, far below the throughput the shortcut buys.
/// `hash_count` is the number of nonces the launch evaluated, wrong ones
/// included. `hash_nonces` is subject to the same contract: it is a kernel
/// self-test, not a verifier; use `pow_core::hash_from_nonce`.
pub struct CudaEngine {
    engine_id: usize,
    devices: Vec<Arc<CudaDevice>>,
    device_counter: AtomicUsize,
    batch_size: u32,
    throttle_ms: u64,
}

static ENGINE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ASSIGNED_DEVICE: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static WORKER_BUFFERS: RefCell<Option<WorkerBuffers>> = const { RefCell::new(None) };
    static DEVICE_LOST: RefCell<Option<usize>> = const { RefCell::new(None) };
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn cuda_launch_config() -> (u32, u32) {
    (
        env_u32("MINER_CUDA_BLOCK_DIM", DEFAULT_THREADS_PER_BLOCK),
        env_u32("MINER_CUDA_MAX_BLOCKS", DEFAULT_MAX_BLOCKS),
    )
}

fn cuda_k_segment_size() -> u32 {
    env_u32("MINER_CUDA_K_SEGMENT", DEFAULT_K_SEGMENT)
}

/// Experimental host/GPU overlap: double-buffered [`MiningParams`] H2D on a
/// copy stream while the previous segment kernel runs (`MINER_CUDA_PIPELINE=1`).
fn cuda_pipeline_enabled() -> bool {
    std::env::var("MINER_CUDA_PIPELINE")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// When true, host batching aligns k-segments for nonce-line `pk[14]` (A2).
/// Set via `MINER_CUDA_NONCE_LINE=1` or `true`.
pub fn miner_cuda_nonce_line_env_enabled() -> bool {
    std::env::var("MINER_CUDA_NONCE_LINE")
        .ok()
        .is_some_and(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
}

/// True once A1 links a nonce-line mining entry (separate from flat `mining_main`).
pub fn miner_cuda_nonce_line_kernel_ready() -> bool {
    true
}

/// Experimental CPPminer partial digest path (`-DMINER_CUDA_CPP_OUT0` at NVRTC).
/// See `kernels/cpp_out0.cuh` — not linked into `mining_main` yet.
fn cuda_cpp_out0_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_CPP_OUT0")
}

/// Lazy wide reduce + terminal adds inside `cpp_out0_internal22` (filter path).
fn cuda_out0_lazy_w_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_OUT0_LAZY_W")
}

fn cuda_out0_g_unroll_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_OUT0_G_UNROLL")
}

fn cuda_out0_g_partial9_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_OUT0_G_PARTIAL9")
}

fn cuda_out0_ext_unroll_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_OUT0_EXT_UNROLL")
}

fn cuda_out0_g_partial_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_OUT0_G_PARTIAL")
}

/// Experimental `ext_layer64` pass-2 scheduling (`-DMINER_CUDA_EXT_V2` at NVRTC).
fn cuda_ext_v2_enabled() -> bool {
    std::env::var("MINER_CUDA_EXT_V2")
        .ok()
        .is_some_and(|v| {
            matches!(
                v.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
}

/// Skip rare second Goldilocks carry fold on nonce / `t` steps (#110 item 3).
fn cuda_lazy_carry_enabled() -> bool {
    std::env::var("MINER_CUDA_LAZY_CARRY")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// NVRTC `-D` flags that must bust the per-arch PTX cache when toggled.
fn nvrtc_compile_tag() -> u32 {
    let mut tag = cuda_gf64_32bit_flag() as u32;
    if cuda_lazy_carry_enabled() {
        tag |= 2;
    }
    if cuda_ext_v2_enabled() {
        tag |= 4;
    }
    if cuda_ext_one_m4_enabled() {
        tag |= 8;
    }
    tag |= (cuda_min_blocks() & 15) << 8;
    tag |= (cuda_launch_threads() & 1023) << 12;
    if cuda_fused_diag_enabled() {
        tag |= 1 << 22;
    }
    if cuda_pipe_sbox_enabled() {
        tag |= 1 << 23;
    }
    tag |= (cuda_internal_unroll() & 7) << 24;
    tag |= (cuda_cmp_limbs() & 15) << 28;
    if cuda_dual_enabled() {
        tag |= 1 << 27;
    }
    if cuda_cpp_out0_enabled() {
        tag |= 1 << 21;
    }
    if cuda_out0_lazy_w_enabled() {
        tag |= 1 << 20;
    }
    if cuda_out0_g_unroll_enabled() {
        tag |= 1 << 19;
    }
    if cuda_out0_g_partial9_enabled() {
        tag |= 1 << 17;
    } else if cuda_out0_g_partial_enabled() {
        tag |= 1 << 18;
    }
    if cuda_out0_ext_unroll_enabled() {
        tag |= 1 << 16;
    }
    if cuda_nonce_line_sh_pk_enabled() {
        tag |= 1 << 15;
    }
    if cuda_nonce_line_pipe_enabled() {
        tag |= 1 << 14;
    }
    tag |= cuda_nonce_line_ext_sbox_unroll() << 29;
    tag |= cuda_nonce_line_term_ext_unroll() << 30;
    tag |= (cuda_nonce_line_t_unroll() & 31) << 25;
    tag |= (cuda_nonce_line_min_blocks() & 15) << 10;
    tag |= (cuda_nonce_line_launch_threads() & 1023) << 20;
    tag
}

fn cuda_flag_on(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn cuda_fused_diag_enabled() -> bool {
    match std::env::var("MINER_CUDA_FUSED_DIAG").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") => false,
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("") => true,
        Ok(other) => {
            log::warn!(
                target: "cuda_engine",
                "MINER_CUDA_FUSED_DIAG={other} invalid; use 0 or 1 (default on)"
            );
            true
        }
        Err(_) => true,
    }
}

fn cuda_pipe_sbox_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_PIPE_SBOX")
}

fn cuda_dual_enabled() -> bool {
    cuda_flag_on("MINER_CUDA_DUAL")
}

fn cuda_internal_unroll() -> u32 {
    env_u32("MINER_CUDA_INTERNAL_UNROLL", 1).clamp(1, 4)
}

fn cuda_cmp_limbs() -> u32 {
    env_u32("MINER_CUDA_CMP_LIMBS", 8).clamp(1, 8)
}

fn cuda_ext_one_m4_enabled() -> bool {
    std::env::var("MINER_CUDA_EXT_ONE_M4")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn cuda_min_blocks() -> u32 {
    env_u32("MINER_CUDA_MIN_BLOCKS", 4).clamp(1, 8)
}

fn cuda_launch_threads() -> u32 {
    env_u32("MINER_CUDA_BLOCK_DIM", DEFAULT_THREADS_PER_BLOCK).clamp(32, 1024)
}

/// NCU: nonce-line kernel is register-heavy (~126); separate launch_bounds from flat mining.
fn cuda_nonce_line_min_blocks() -> u32 {
    env_u32("MINER_CUDA_NONCE_LINE_MIN_BLOCKS", 2).clamp(1, 8)
}

fn cuda_nonce_line_launch_threads() -> u32 {
    env_u32("MINER_CUDA_NONCE_LINE_BLOCK_DIM", cuda_launch_threads()).clamp(32, 1024)
}

fn cuda_nonce_line_sh_pk_enabled() -> bool {
    match std::env::var("MINER_CUDA_NONCE_LINE_SH_PK").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") => false,
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("") => true,
        Ok(other) => {
            log::warn!(
                target: "cuda_engine",
                "MINER_CUDA_NONCE_LINE_SH_PK={other} invalid; use 0 or 1 (default on)"
            );
            true
        }
        Err(_) => true,
    }
}

/// Software-pipelined internal rounds in nonce_line_permute (default on; NCU stall overlap).
fn cuda_nonce_line_ext_sbox_unroll() -> u32 {
    env_u32("MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL", 0).min(1)
}

fn cuda_nonce_line_term_ext_unroll() -> u32 {
    env_u32("MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL", 0).min(1)
}

fn cuda_nonce_line_t_unroll() -> u32 {
    env_u32("MINER_CUDA_NONCE_LINE_T_UNROLL", 1).clamp(1, 32)
}

fn cuda_nonce_line_pipe_enabled() -> bool {
    match std::env::var("MINER_CUDA_NONCE_LINE_PIPE").as_deref() {
        Ok("0") | Ok("false") | Ok("FALSE") => false,
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("") => true,
        Ok(other) => {
            log::warn!(
                target: "cuda_engine",
                "MINER_CUDA_NONCE_LINE_PIPE={other} invalid; use 0 or 1 (default on)"
            );
            true
        }
        Err(_) => true,
    }
}

fn cuda_nonce_line_launch_config() -> (u32, u32) {
    (
        cuda_nonce_line_launch_threads(),
        env_u32("MINER_CUDA_MAX_BLOCKS", DEFAULT_MAX_BLOCKS),
    )
}

/// Metadata passed when building per-segment [`MiningParams`].
#[derive(Clone, Copy, Debug)]
struct KSegmentContext {
    #[allow(dead_code)]
    segment_index: usize,
    #[allow(dead_code)]
    segment_count: usize,
    k_segment_nonces: u32,
    /// Nonce-line `k` for this segment; `None` in flat `mining_main` mode.
    #[allow(dead_code)]
    k_index: Option<u64>,
    /// Absolute `t` at `segment_start` when nonce-line mode is active.
    t_base: u32,
}

impl KSegmentContext {
    fn for_segment(
        segment_index: usize,
        segment_count: usize,
        k_segment_nonces: u32,
        segment_start: U512,
    ) -> Self {
        if miner_cuda_nonce_line_env_enabled() {
            let be = segment_start.to_big_endian();
            Self {
                segment_index,
                segment_count,
                k_segment_nonces,
                k_index: Some(pow_core::nonce_line::k_of(&be)),
                t_base: pow_core::nonce_line::t_of(&be),
            }
        } else {
            Self {
                segment_index,
                segment_count,
                k_segment_nonces,
                k_index: None,
                t_base: 0,
            }
        }
    }
}

/// Split a GPU batch into k-segments. Flat mode: equal-sized chunks from
/// `batch_start`. Nonce-line mode (A2): first/last chunks may be shorter so
/// segments do not cross a `2^26` `t` boundary (pk constant within a segment).
fn plan_k_segments_with_cap(
    batch_start: U512,
    batch_size: u32,
    cap: u32,
    nonce_line_align: bool,
) -> Vec<(U512, u32)> {
    if batch_size == 0 {
        return Vec::new();
    }
    if cap == 0 || batch_size <= cap {
        if nonce_line_align {
            let room = pow_core::nonce_line::t_room_in_k(batch_start);
            let seg_size = batch_size.min(room).max(1);
            if seg_size < batch_size {
                return vec![
                    (batch_start, seg_size),
                    (
                        pow_core::nonce_line::counter_add_t(batch_start, seg_size),
                        batch_size - seg_size,
                    ),
                ];
            }
        }
        return vec![(batch_start, batch_size)];
    }

    let mut segments = Vec::new();
    let mut offset = 0u32;
    let mut cursor = batch_start;

    while offset < batch_size {
        let remaining = batch_size - offset;
        let mut seg_size = remaining.min(cap);

        if nonce_line_align {
            let room_in_k = pow_core::nonce_line::t_room_in_k(cursor);
            seg_size = seg_size.min(room_in_k).max(1);
        }

        segments.push((cursor, seg_size));
        offset += seg_size;
        cursor = if nonce_line_align {
            pow_core::nonce_line::counter_add_t(cursor, seg_size)
        } else {
            cursor + U512::from(seg_size)
        };
    }
    segments
}

fn plan_k_segments(batch_start: U512, batch_size: u32) -> Vec<(U512, u32)> {
    plan_k_segments_with_cap(
        batch_start,
        batch_size,
        cuda_k_segment_size(),
        miner_cuda_nonce_line_env_enabled(),
    )
}

fn mining_dispatch(segment_size: u32, seg: &KSegmentContext) -> ([u32; 3], u32, u32) {
    let (threads_per_block, max_blocks) = if cuda_use_nonce_line() {
        cuda_nonce_line_launch_config()
    } else {
        cuda_launch_config()
    };
    let num_blocks = segment_size
        .div_ceil(threads_per_block)
        .clamp(1, max_blocks);
    let total_threads = num_blocks * threads_per_block;
    let per_thread = segment_size.div_ceil(total_threads).max(1);
    let third = if cuda_use_nonce_line() {
        seg.t_base.saturating_add(segment_size)
    } else {
        segment_size
    };
    let dispatch = [total_threads, per_thread, third];
    (dispatch, num_blocks, threads_per_block)
}

fn mining_params_for_segment(
    ctx: &JobContext,
    segment_start: U512,
    segment_size: u32,
    dispatch: [u32; 3],
    seg: &KSegmentContext,
) -> MiningParams {
    let counter_be = segment_start.to_big_endian();
    let nonce_be = counter_be;
    let start_limbs = pow_core::u512_to_le_u32s(segment_start);
    let prestate = pow_core::mining_prestate_low64_u32s(ctx.header, nonce_be);
    let target = pow_core::u512_to_le_u32s(ctx.target);
    let k_base = pow_core::nonce_line::base_words(&counter_be);
    let (pk, t_base) = if cuda_use_nonce_line() {
        (
            pow_core::nonce_line::segment_launch_params(ctx.header, segment_start),
            seg.t_base,
        )
    } else {
        ([0u64; 14], 0u32)
    };
    let _ = segment_size;
    MiningParams {
        prestate,
        start_nonce: start_limbs,
        difficulty_target: target,
        dispatch_config: dispatch,
        nonce_line_k_base: k_base,
        nonce_line_v: NONCE_LINE_K_V,
        nonce_line_pk: pk,
        nonce_line_t_base: t_base,
        nonce_line_pad: 0,
    }
}

impl CudaEngine {
    pub fn try_new(batch_size: u32, throttle_ms: u64) -> Result<Self, Box<dyn std::error::Error>> {
        if batch_size == 0 {
            return Err("batch_size must be non-zero".into());
        }

        match silent_catch(cudarc::driver::result::init) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(format!(
                    "CUDA driver is not available (need libcuda / nvidia driver): {e}"
                )
                .into());
            }
            Err(_) => {
                return Err("CUDA driver is not available (need libcuda / nvidia driver)".into());
            }
        }
        let count = cudarc::driver::result::device::get_count()
            .map_err(|e| format!("Failed to query CUDA device count: {e}"))?;
        if count <= 0 {
            return Err("No CUDA devices found".into());
        }

        let mut devices = Vec::new();
        let compile_tag = nvrtc_compile_tag();
        let mut compiled: HashMap<(i32, i32, u32), Ptx> = HashMap::new();
        for ordinal in 0..count as usize {
            let ctx = CudaContext::new(ordinal)
                .map_err(|e| format!("Failed to create CUDA context for device {ordinal}: {e}"))?;
            let name = ctx
                .name()
                .unwrap_or_else(|_| format!("cuda-device-{ordinal}"));
            let (major, minor) = ctx.compute_capability().map_err(|e| {
                format!("Failed to query compute capability of CUDA device {ordinal}: {e}")
            })?;
            let ptx = match compiled.get(&(major, minor, compile_tag)) {
                Some(ptx) => ptx.clone(),
                None => {
                    let ptx = match silent_catch(|| compile_kernel(major, minor)) {
                        Ok(ptx) => ptx?,
                        Err(_) => {
                            return Err("CUDA NVRTC library is not available (need libnvrtc from the CUDA toolkit)".into());
                        }
                    };
                    compiled.insert((major, minor, compile_tag), ptx.clone());
                    ptx
                }
            };
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("Failed to load CUDA mining module on {name}: {e}"))?;
            let kernel = mining_kernel_name();
            let mine = module
                .load_function(kernel)
                .map_err(|e| format!("Failed to load {kernel} on {name}: {e}"))?;
            let (regs, local, shared, constant) = (
                mine.num_regs()?,
                mine.local_size_bytes()?,
                mine.shared_size_bytes()?,
                mine.const_size_bytes()?,
            );
            log::info!(
                target: "cuda_engine",
                "CUDA device {ordinal}: {name} (sm_{major}{minor}; {kernel} uses {regs} registers, {local} B local, {shared} B shared, {constant} B constant)"
            );
            if local > 0 {
                log::warn!(
                    target: "cuda_engine",
                    "{kernel} spills {local} bytes per thread to local memory on {name}"
                );
            }
            devices.push(Arc::new(CudaDevice { ctx, module, name }));
        }

        log::info!(
            target: "cuda_engine",
            "CUDA engine initialized with {} device(s) (batch size: {batch_size} nonces, throttle: {throttle_ms}ms)",
            devices.len()
        );
        if cuda_use_nonce_line() {
            log::info!(
                target: "cuda_engine",
                "MINER_CUDA_NONCE_LINE=1: using mining_main_nonce_line with host pk[14] per k-segment and (k,t) CPU verify"
            );
        }

        Ok(Self {
            engine_id: ENGINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            devices,
            device_counter: AtomicUsize::new(0),
            batch_size,
            throttle_ms,
        })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn clear_worker_resources() {
        WORKER_BUFFERS.with(|b| {
            *b.borrow_mut() = None;
        });
        ASSIGNED_DEVICE.with(|a| {
            *a.borrow_mut() = None;
        });
        DEVICE_LOST.with(|lost| {
            *lost.borrow_mut() = None;
        });
    }

    /// Hashes `count` consecutive nonces on the first device with the mining
    /// kernel's arithmetic. This is a kernel self-test under the engine's
    /// probabilistic contract, not a verifier; use `pow_core::hash_from_nonce`
    /// for exact hashes.
    pub fn hash_nonces(
        &self,
        header: [u8; 32],
        start: U512,
        count: u32,
    ) -> Result<Vec<U512>, Box<dyn std::error::Error>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let device = &self.devices[0];
        let mut buffers = create_buffers(self.engine_id, 0, device, count)?;
        let nonce_be = start.to_big_endian();
        let mid = pow_core::mining_midstate_u32s(header, nonce_be[..32].try_into().unwrap());
        let start_limbs = pow_core::u512_to_le_u32s(start);
        buffers.stream.memcpy_htod(&mid, &mut buffers.midstate)?;
        buffers
            .stream
            .memcpy_htod(&start_limbs, &mut buffers.start_nonce)?;

        let (threads_per_block, _) = cuda_launch_config();
        let num_blocks = count.div_ceil(threads_per_block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.hash);
        builder.arg(&mut buffers.hashes);
        builder.arg(&buffers.midstate);
        builder.arg(&buffers.start_nonce);
        builder.arg(&count);
        unsafe {
            builder.launch(cfg)?;
        }
        let raw = buffers.stream.clone_dtoh(&buffers.hashes)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let mut limbs = [0u32; 16];
            limbs.copy_from_slice(&raw[i * 16..(i + 1) * 16]);
            out.push(pow_core::u512_from_le_u32s(limbs));
        }
        Ok(out)
    }
}

fn nvrtc_supported_archs() -> Result<Vec<i32>, String> {
    let mut count = 0i32;
    unsafe { nvrtc_sys::nvrtcGetNumSupportedArchs(&mut count) }
        .result()
        .map_err(|e| format!("nvrtcGetNumSupportedArchs failed: {e:?}"))?;
    let mut archs = vec![0i32; count.max(0) as usize];
    unsafe { nvrtc_sys::nvrtcGetSupportedArchs(archs.as_mut_ptr()) }
        .result()
        .map_err(|e| format!("nvrtcGetSupportedArchs failed: {e:?}"))?;
    Ok(archs)
}

fn nvrtc_version() -> Result<(i32, i32), String> {
    let (mut major, mut minor) = (0i32, 0i32);
    unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(|e| format!("nvrtcVersion failed: {e:?}"))?;
    Ok((major, minor))
}

/// Newest NVRTC virtual arch the installed toolkit supports that the device can run.
fn nvrtc_ptx_target_arch(device_arch: i32) -> Result<i32, String> {
    if let Ok(raw) = std::env::var("MINER_CUDA_PTX_ARCH") {
        return raw
            .parse()
            .map_err(|_| "MINER_CUDA_PTX_ARCH must be an integer (e.g. 120)".into());
    }
    let supported = nvrtc_supported_archs()?;
    supported
        .iter()
        .copied()
        .filter(|&arch| arch <= device_arch)
        .max()
        .ok_or_else(|| {
            format!(
                "NVRTC supports no architecture at or below sm_{device_arch}; supported={supported:?}"
            )
        })
}

/// Compiles PTX for the newest virtual architecture NVRTC knows that the device
/// can run. The driver JIT then emits native code; on a 4090 that measured
/// equal to loading NVRTC's own cubin, so the PTX route is kept.
fn compile_kernel(major: i32, minor: i32) -> Result<Ptx, Box<dyn std::error::Error>> {
    let device_arch = major * 10 + minor;
    let (nv_major, nv_minor) = nvrtc_version().unwrap_or((0, 0));
    let supported = nvrtc_supported_archs().unwrap_or_default();
    log::info!(
        target: "cuda_engine",
        "NVRTC {}.{} supported arches: {:?}",
        nv_major,
        nv_minor,
        supported
    );
    let target = nvrtc_ptx_target_arch(device_arch)?;
    if std::env::var("MINER_CUDA_PTX_ARCH").is_ok() {
        log::warn!(
            target: "cuda_engine",
            "MINER_CUDA_PTX_ARCH overrides NVRTC arch selection for sm_{device_arch} (using compute_{target})"
        );
    } else if target != device_arch {
        log::warn!(
            target: "cuda_engine",
            "NVRTC does not know sm_{device_arch}; compiling PTX for compute_{target} instead (install CUDA 12.8+ libnvrtc on the host)"
        );
    }
    let gf64_32bit = cuda_gf64_32bit_flag();
    log::info!(
        target: "cuda_engine",
        "GF64 add/nonce coeff chains: MINER_CUDA_GF64_32BIT={gf64_32bit} (1=32-bit PTX/coeffs, 0=64-bit reference)"
    );
    if cuda_ext_v2_enabled() {
        log::info!(
            target: "cuda_engine",
            "MINER_CUDA_EXT_V2=1: NVRTC ext_layer64 pass-2 fully unrolled"
        );
    }
    if cuda_lazy_carry_enabled() {
        log::info!(
            target: "cuda_engine",
            "MINER_CUDA_LAZY_CARRY=1: gf64_add_lazy on nonce/t absorb (#110 item 3)"
        );
    }
    if cuda_cpp_out0_enabled() {
        log::info!(
            target: "cuda_engine",
            "MINER_CUDA_CPP_OUT0=1: nonce-line uses CPPminer out0 second-permute schedule (host re-verify on hits)"
        );
    }
    if cuda_out0_lazy_w_enabled() {
        log::info!(
            target: "cuda_engine",
            "MINER_CUDA_OUT0_LAZY_W=1: reduce128_wred_fast + lazy terminal adds in cpp_out0_internal22"
        );
    }
    if cuda_out0_g_unroll_enabled() {
        log::info!(
            target: "cuda_engine",
            "MINER_CUDA_OUT0_G_UNROLL=1: fully unrolled hash_out0 g=1..15 loop"
        );
    }
    if cuda_out0_g_partial_enabled() {
        log::info!(
            target: "cuda_engine",
            "MINER_CUDA_OUT0_G_PARTIAL=1: unrolled hash_out0 g=1..7, loop g=8..15"
        );
    }
    let opts = nvrtc_compile_options(target);
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| {
        format!("NVRTC failed to compile the CUDA mining kernel for compute_{target}: {e}")
    })?;
    log::info!(
        target: "cuda_engine",
        "Compiled CUDA mining kernel PTX for compute_{target} (device sm_{device_arch}); the driver JIT emits native code"
    );
    Ok(ptx)
}

fn cuda_kernel_include_dir() -> String {
    std::env::var("MINER_CUDA_KERNEL_DIR").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels").to_string()
    })
}

fn cuda_gf64_32bit_flag() -> i32 {
    match std::env::var("MINER_CUDA_GF64_32BIT").as_deref() {
        Ok("0") => 0,
        Ok("1") | Ok("") => 1,
        Ok(other) => {
            log::warn!(
                target: "cuda_engine",
                "MINER_CUDA_GF64_32BIT={other} invalid; use 0 or 1 (default 1)"
            );
            1
        }
        Err(_) => 1,
    }
}

fn nvrtc_compile_options(gpu_arch: i32) -> CompileOptions {
    let kernel_dir = cuda_kernel_include_dir();
    let gf64_32bit = cuda_gf64_32bit_flag();
    let mut options = vec![
        format!("--gpu-architecture=compute_{gpu_arch}"),
        format!("-I{kernel_dir}"),
        format!("-DMINER_CUDA_GF64_32BIT={gf64_32bit}"),
        format!("-DMINER_CUDA_MIN_BLOCKS={}", cuda_min_blocks()),
        format!("-DMINER_CUDA_LAUNCH_THREADS={}", cuda_launch_threads()),
        format!(
            "-DMINER_CUDA_NONCE_LINE_MIN_BLOCKS={}",
            cuda_nonce_line_min_blocks()
        ),
        format!(
            "-DMINER_CUDA_NONCE_LINE_LAUNCH_THREADS={}",
            cuda_nonce_line_launch_threads()
        ),
        format!("-DMINER_CUDA_INTERNAL_UNROLL={}", cuda_internal_unroll()),
        format!("-DMINER_CUDA_CMP_LIMBS={}", cuda_cmp_limbs()),
    ];
    if cuda_fused_diag_enabled() {
        options.push("-DMINER_CUDA_FUSED_DIAG".into());
    }
    if cuda_pipe_sbox_enabled() || cuda_nonce_line_pipe_enabled() {
        options.push("-DMINER_CUDA_PIPE_SBOX".into());
    }
    if cuda_dual_enabled() {
        options.push("-DMINER_CUDA_DUAL".into());
    }
    if cuda_ext_v2_enabled() {
        options.push("-DMINER_CUDA_EXT_V2".into());
    }
    if cuda_ext_one_m4_enabled() {
        options.push("-DMINER_CUDA_EXT_ONE_M4".into());
    }
    if cuda_lazy_carry_enabled() {
        options.push("-DMINER_CUDA_LAZY_CARRY=1".into());
    }
    if cuda_cpp_out0_enabled() {
        options.push("-DMINER_CUDA_CPP_OUT0=1".into());
    }
    if cuda_out0_lazy_w_enabled() {
        options.push("-DMINER_CUDA_OUT0_LAZY_W=1".into());
    }
    if cuda_out0_g_unroll_enabled() {
        options.push("-DMINER_CUDA_OUT0_G_UNROLL=1".into());
    }
    if cuda_out0_ext_unroll_enabled() {
        options.push("-DMINER_CUDA_OUT0_EXT_UNROLL=1".into());
    }
    if cuda_out0_g_partial9_enabled() {
        options.push("-DMINER_CUDA_OUT0_G_PARTIAL9=1".into());
    } else if cuda_out0_g_partial_enabled() {
        options.push("-DMINER_CUDA_OUT0_G_PARTIAL=1".into());
    }
    if cuda_nonce_line_sh_pk_enabled() {
        options.push("-DMINER_CUDA_NONCE_LINE_SH_PK=1".into());
    } else {
        options.push("-DMINER_CUDA_NONCE_LINE_SH_PK=0".into());
    }
    if cuda_nonce_line_pipe_enabled() {
        options.push("-DMINER_CUDA_NONCE_LINE_PIPE=1".into());
    } else {
        options.push("-DMINER_CUDA_NONCE_LINE_PIPE=0".into());
    }
    options.push(format!(
        "-DMINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL={}",
        cuda_nonce_line_ext_sbox_unroll()
    ));
    options.push(format!(
        "-DMINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL={}",
        cuda_nonce_line_term_ext_unroll()
    ));
    options.push(format!(
        "-DMINER_CUDA_NONCE_LINE_T_UNROLL={}",
        cuda_nonce_line_t_unroll()
    ));
    CompileOptions {
        options,
        ..Default::default()
    }
}

fn silent_catch<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    out
}

fn create_buffers(
    engine_id: usize,
    device_index: usize,
    device: &CudaDevice,
    hash_capacity: u32,
) -> Result<WorkerBuffers, Box<dyn std::error::Error>> {
    let stream = device.ctx.default_stream();
    let mine = device.module.load_function(mining_kernel_name())?;
    let hash = device.module.load_function("hash_nonces")?;
    Ok(WorkerBuffers {
        engine_id,
        device_index,
        results: stream.alloc_zeros::<u32>(RESULTS_U32S)?,
        midstate: stream.alloc_zeros::<u32>(24)?,
        start_nonce: stream.alloc_zeros::<u32>(16)?,
        hashes: stream.alloc_zeros::<u32>(hash_capacity.max(1) as usize * 16)?,
        mine,
        hash,
        stream: stream.clone(),
        busy: Duration::ZERO,
        busy_since: Instant::now(),
        params_dev: [
            stream.alloc_zeros::<MiningParams>(1)?,
            stream.alloc_zeros::<MiningParams>(1)?,
        ],
        params_h2d_ready: [None, None],
        copy_stream: None,
        param_slot: 0,
        staged_first_segment: None,
    })
}

fn ensure_pipeline_resources(
    buffers: &mut WorkerBuffers,
    device: &CudaDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    if buffers.copy_stream.is_some() {
        return Ok(());
    }
    let copy_stream = device.ctx.new_stream()?;
    buffers.copy_stream = Some(copy_stream);
    Ok(())
}

fn wait_params_h2d(buffers: &WorkerBuffers, slot: usize) -> Result<(), BatchResult> {
    if let Some(event) = buffers.params_h2d_ready[slot].as_ref() {
        event.synchronize().map_err(|e| {
            log::error!(target: "cuda_engine", "CUDA params H2D wait failed: {e}");
            BatchResult::DeviceLost
        })?;
    }
    Ok(())
}

fn upload_params_sync(
    buffers: &mut WorkerBuffers,
    params: &MiningParams,
    slot: usize,
) -> Result<(), BatchResult> {
    buffers.params_h2d_ready[slot] = None;
    buffers.stream.memcpy_htod(std::slice::from_ref(params), &mut buffers.params_dev[slot]).map_err(|e| {
        log::error!(target: "cuda_engine", "CUDA params upload failed: {e}");
        BatchResult::DeviceLost
    })?;
    Ok(())
}

fn upload_params_async(
    buffers: &mut WorkerBuffers,
    params: &MiningParams,
    slot: usize,
) -> Result<(), BatchResult> {
    let copy = buffers
        .copy_stream
        .as_ref()
        .expect("pipeline copy stream");
    copy.memcpy_htod(std::slice::from_ref(params), &mut buffers.params_dev[slot])
        .map_err(|e| {
            log::error!(target: "cuda_engine", "CUDA async params upload failed: {e}");
            BatchResult::DeviceLost
        })?;
    let event = copy.record_event(None).map_err(|e| {
        log::error!(target: "cuda_engine", "CUDA params H2D event failed: {e}");
        BatchResult::DeviceLost
    })?;
    buffers.params_h2d_ready[slot] = Some(event);
    Ok(())
}

fn launch_mining_kernel(
    buffers: &mut WorkerBuffers,
    param_slot: usize,
    num_blocks: u32,
    threads_per_block: u32,
    wait_copy_stream: bool,
) -> Result<(), BatchResult> {
    if wait_copy_stream {
        if let Some(event) = buffers.params_h2d_ready[param_slot].as_ref() {
            buffers.stream.wait(event).map_err(|e| {
                log::error!(target: "cuda_engine", "CUDA stream wait on params H2D failed: {e}");
                BatchResult::DeviceLost
            })?;
        }
        buffers.params_h2d_ready[param_slot] = None;
    }

    let launch_start = Instant::now();
    let result = (|| {
        buffers.stream.memset_zeros(&mut buffers.results)?;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.mine);
        builder.arg(&mut buffers.results);
        builder.arg(&buffers.params_dev[param_slot]);
        unsafe {
            builder.launch(cfg)?;
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    })();
    buffers.busy += launch_start.elapsed();
    result.map_err(|e| {
        log::error!(target: "cuda_engine", "CUDA segment launch failed: {e}");
        BatchResult::DeviceLost
    })
}

fn process_segment_hits(
    result_u32s: &[u32],
    segment_start: U512,
    ctx: &JobContext,
    seg: &KSegmentContext,
    dispatched: u64,
) -> Result<Option<(U512, U512)>, BatchResult> {
    let hits = result_u32s[0] as usize;
    if hits > MAX_HITS {
        log::warn!(
            target: "cuda_engine",
            "CUDA launch produced {hits} candidates, only {MAX_HITS} were recorded; the rest are skipped"
        );
    }
    let mut best: Option<(U512, U512)> = None;
    let t_end = seg.t_base.saturating_add(seg.k_segment_nonces);
    for &index in &result_u32s[1..=hits.min(MAX_HITS)] {
        let nonce = if cuda_use_nonce_line() {
            let t_abs = index;
            if t_abs < seg.t_base || t_abs >= t_end {
                log::error!(
                    target: "cuda_engine",
                    "CUDA returned out-of-range nonce-line t={t_abs} (segment t in [{}, {t_end}))",
                    seg.t_base
                );
                return Err(BatchResult::DeviceLost);
            }
            pow_core::nonce_line::nonce_from_absolute_t(segment_start, t_abs)
        } else {
            let logical_index = index as u64;
            if logical_index >= dispatched {
                log::error!(
                    target: "cuda_engine",
                    "CUDA returned out-of-range candidate index {logical_index} for {dispatched} dispatched nonces"
                );
                return Err(BatchResult::DeviceLost);
            }
            segment_start + U512::from(logical_index)
        };
        let hash = pow_core::hash_from_nonce(ctx, nonce);
        if hash >= ctx.target {
            log::warn!(
                target: "cuda_engine",
                "CUDA candidate {} rejected by CPU verification (hash {} >= target {})",
                format_u512(nonce),
                format_u512(hash),
                format_u512(ctx.target)
            );
            continue;
        }
        if best.is_none_or(|(n, _)| nonce < n) {
            best = Some((nonce, hash));
        }
    }
    Ok(best)
}

fn take_gpu_duty_cycle(buffers: &mut WorkerBuffers) -> Option<f64> {
    let wall = buffers.busy_since.elapsed();
    let duty = (wall >= Duration::from_secs(1))
        .then(|| 100.0 * buffers.busy.as_secs_f64() / wall.as_secs_f64());
    buffers.busy = Duration::ZERO;
    buffers.busy_since = Instant::now();
    duty
}

fn worker_buffers(
    engine: &CudaEngine,
    device_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    WORKER_BUFFERS.with(|cell| {
        let mut slot = cell.borrow_mut();
        let reuse = matches!(
            &*slot,
            Some(b) if b.engine_id == engine.engine_id && b.device_index == device_index
        );
        if !reuse {
            *slot = Some(create_buffers(
                engine.engine_id,
                device_index,
                &engine.devices[device_index],
                1,
            )?);
        }
        Ok(())
    })
}

enum BatchResult {
    Found {
        candidate: Candidate,
        hash_count: u64,
    },
    NotFound {
        hash_count: u64,
    },
    DeviceLost,
}

fn nonces_until_low64_carry(nonce: U512) -> U512 {
    (U512::one() << 64).saturating_sub(U512::from(nonce.low_u64()))
}

fn run_mining_segment(
    buffers: &mut WorkerBuffers,
    ctx: &JobContext,
    segment_start: U512,
    segment_size: u32,
    seg_ctx: &KSegmentContext,
) -> Result<(Option<(U512, U512)>, u64), BatchResult> {
    let (dispatch, num_blocks, threads_per_block) = mining_dispatch(segment_size, seg_ctx);
    let total_threads = dispatch[0];
    let nonces_per_thread = dispatch[1];
    let params = mining_params_for_segment(ctx, segment_start, segment_size, dispatch, seg_ctx);

    upload_params_sync(buffers, &params, 0)?;
    launch_mining_kernel(buffers, 0, num_blocks, threads_per_block, false)?;

    let result_u32s = match buffers.stream.clone_dtoh(&buffers.results) {
        Ok(v) => v,
        Err(e) => {
            log::error!(target: "cuda_engine", "CUDA result copy failed: {e}");
            return Err(BatchResult::DeviceLost);
        }
    };

    let dispatched =
        (total_threads as u64 * nonces_per_thread as u64).min(segment_size as u64);
    let best = process_segment_hits(&result_u32s, segment_start, ctx, seg_ctx, dispatched)?;
    Ok((best, dispatched))
}

fn stage_first_segment_params(
    buffers: &mut WorkerBuffers,
    ctx: &JobContext,
    batch_start: U512,
    batch_size: u32,
) -> Result<(), BatchResult> {
    let segments = plan_k_segments(batch_start, batch_size);
    let Some((segment_start, segment_size)) = segments.first().copied() else {
        return Ok(());
    };
    let seg_ctx = KSegmentContext::for_segment(0, segments.len(), segment_size, segment_start);
    let (dispatch, _, _) = mining_dispatch(segment_size, &seg_ctx);
    let params = mining_params_for_segment(ctx, segment_start, segment_size, dispatch, &seg_ctx);
    let slot = 1usize - buffers.param_slot as usize;
    upload_params_async(buffers, &params, slot)?;
    buffers.staged_first_segment = Some(StagedSegmentParams {
        batch_start,
        slot: slot as u8,
    });
    Ok(())
}

fn run_single_batch_pipelined(
    buffers: &mut WorkerBuffers,
    device: &CudaDevice,
    ctx: &JobContext,
    batch_start: U512,
    batch_size: u32,
    next_batch: Option<(U512, u32)>,
) -> BatchResult {
    if let Err(e) = ensure_pipeline_resources(buffers, device) {
        log::error!(target: "cuda_engine", "CUDA pipeline setup failed: {e}");
        return BatchResult::DeviceLost;
    }

    let segments = plan_k_segments(batch_start, batch_size);
    let segment_count = segments.len();
    if segment_count > 1 {
        log::debug!(
            target: "cuda_engine",
            "pipeline k-segment batch {}..+{} → {segment_count} launches",
            format_u512(batch_start),
            batch_size
        );
    }

    let mut total_dispatched = 0u64;
    let mut batch_best: Option<(U512, U512)> = None;

    for (segment_index, (segment_start, segment_size)) in segments.iter().copied().enumerate() {
        let seg_ctx =
            KSegmentContext::for_segment(segment_index, segment_count, segment_size, segment_start);
        let (dispatch, num_blocks, threads_per_block) = mining_dispatch(segment_size, &seg_ctx);
        let total_threads = dispatch[0];
        let nonces_per_thread = dispatch[1];

        let slot = if segment_index == 0 {
            if let Some(staged) = buffers.staged_first_segment.take() {
                if staged.batch_start == batch_start {
                    buffers.param_slot = staged.slot;
                    if let Err(e) = wait_params_h2d(buffers, staged.slot as usize) {
                        return e;
                    }
                    buffers.params_h2d_ready[staged.slot as usize] = None;
                    staged.slot as usize
                } else {
                    let params = mining_params_for_segment(
                        ctx,
                        segment_start,
                        segment_size,
                        dispatch,
                        &seg_ctx,
                    );
                    if let Err(e) = upload_params_sync(buffers, &params, 0) {
                        return e;
                    }
                    0
                }
            } else {
                let params = mining_params_for_segment(
                    ctx,
                    segment_start,
                    segment_size,
                    dispatch,
                    &seg_ctx,
                );
                if let Err(e) = upload_params_sync(buffers, &params, 0) {
                    return e;
                }
                0
            }
        } else {
            buffers.param_slot as usize
        };

        if let Err(e) = launch_mining_kernel(buffers, slot, num_blocks, threads_per_block, segment_index > 0)
        {
            return e;
        }

        let next_segment = segments.get(segment_index + 1).copied();
        let next_params = next_segment.map(|(next_start, next_size)| {
            let next_ctx = KSegmentContext::for_segment(
                segment_index + 1,
                segment_count,
                next_size,
                next_start,
            );
            let (next_dispatch, _, _) = mining_dispatch(next_size, &next_ctx);
            mining_params_for_segment(ctx, next_start, next_size, next_dispatch, &next_ctx)
        });

        if let Some(params) = next_params.as_ref() {
            let next_slot = 1usize - slot;
            if let Err(e) = upload_params_async(buffers, params, next_slot) {
                return e;
            }
            buffers.param_slot = next_slot as u8;
        }

        let result_u32s = match buffers.stream.clone_dtoh(&buffers.results) {
            Ok(v) => v,
            Err(e) => {
                log::error!(target: "cuda_engine", "CUDA result copy failed: {e}");
                return BatchResult::DeviceLost;
            }
        };

        let dispatched =
            (total_threads as u64 * nonces_per_thread as u64).min(segment_size as u64);
        match process_segment_hits(&result_u32s, segment_start, ctx, &seg_ctx, dispatched) {
            Ok(best) => {
                total_dispatched += dispatched;
                if let Some((nonce, hash)) = best {
                    if batch_best.is_none_or(|(n, _)| nonce < n) {
                        batch_best = Some((nonce, hash));
                    }
                }
            }
            Err(device_lost) => return device_lost,
        }
    }

    if let Some((next_start, next_size)) = next_batch {
        if let Err(e) = stage_first_segment_params(buffers, ctx, next_start, next_size) {
            return e;
        }
    }

    match batch_best {
        Some((nonce, hash)) => BatchResult::Found {
            candidate: Candidate {
                nonce,
                work: nonce.to_big_endian(),
                hash,
            },
            hash_count: total_dispatched,
        },
        None => BatchResult::NotFound {
            hash_count: total_dispatched,
        },
    }
}

fn batch_size_for_start(
    range: &Range,
    current_start: U512,
    engine_batch_size: u32,
) -> u32 {
    let remaining = range
        .end
        .saturating_sub(current_start)
        .saturating_add(U512::one());
    let headroom = nonces_until_low64_carry(current_start);
    let cap = remaining.min(headroom);
    let batch_size_u512 = U512::from(engine_batch_size);
    if cap > batch_size_u512 {
        engine_batch_size
    } else {
        cap.low_u32()
    }
}

fn run_single_batch(
    buffers: &mut WorkerBuffers,
    device: Option<&CudaDevice>,
    ctx: &JobContext,
    batch_start: U512,
    batch_size: u32,
    next_batch: Option<(U512, u32)>,
) -> BatchResult {
    if cuda_pipeline_enabled() {
        let Some(device) = device else {
            log::error!(target: "cuda_engine", "pipeline mode requires CUDA device handle");
            return BatchResult::DeviceLost;
        };
        return run_single_batch_pipelined(
            buffers,
            device,
            ctx,
            batch_start,
            batch_size,
            next_batch,
        );
    }

    let segments = plan_k_segments(batch_start, batch_size);
    let segment_count = segments.len();
    if segment_count > 1 {
        log::debug!(
            target: "cuda_engine",
            "k-segment batch {}..+{} split into {segment_count} launches (MINER_CUDA_K_SEGMENT={})",
            format_u512(batch_start),
            batch_size,
            cuda_k_segment_size()
        );
    }

    let mut total_dispatched = 0u64;
    let mut batch_best: Option<(U512, U512)> = None;

    for (segment_index, (segment_start, segment_size)) in segments.into_iter().enumerate() {
        let seg_ctx =
            KSegmentContext::for_segment(segment_index, segment_count, segment_size, segment_start);
        match run_mining_segment(buffers, ctx, segment_start, segment_size, &seg_ctx) {
            Ok((best, dispatched)) => {
                total_dispatched += dispatched;
                if let Some((nonce, hash)) = best {
                    if batch_best.is_none_or(|(n, _)| nonce < n) {
                        batch_best = Some((nonce, hash));
                    }
                }
            }
            Err(device_lost) => return device_lost,
        }
    }

    match batch_best {
        Some((nonce, hash)) => BatchResult::Found {
            candidate: Candidate {
                nonce,
                work: nonce.to_big_endian(),
                hash,
            },
            hash_count: total_dispatched,
        },
        None => BatchResult::NotFound {
            hash_count: total_dispatched,
        },
    }
}

impl MinerEngine for CudaEngine {
    fn name(&self) -> &'static str {
        "gpu-cuda"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn search_range(
        &self,
        ctx: &JobContext,
        range: Range,
        cancel: &dyn CancelCheck,
    ) -> EngineStatus {
        if self.devices.is_empty() {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if DEVICE_LOST.with(|lost| *lost.borrow() == Some(self.engine_id)) {
            return EngineStatus::DeviceLost { hash_count: 0 };
        }
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if cancel.is_cancelled() {
            return EngineStatus::Cancelled { hash_count: 0 };
        }

        let device_index = ASSIGNED_DEVICE.with(|assigned| {
            let mut assigned_ref = assigned.borrow_mut();
            match *assigned_ref {
                Some((engine_id, index)) if engine_id == self.engine_id => index,
                _ => {
                    let index = if self.devices.len() == 1 {
                        0
                    } else {
                        self.device_counter.fetch_add(1, Ordering::SeqCst) % self.devices.len()
                    };
                    *assigned_ref = Some((self.engine_id, index));
                    log::info!(
                        target: "cuda_engine",
                        "Worker thread assigned to CUDA device {} ({})",
                        index,
                        self.devices[index].name
                    );
                    index
                }
            }
        });

        if let Err(e) = worker_buffers(self, device_index) {
            log::error!(target: "cuda_engine", "CUDA buffer setup failed: {e}");
            DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
            return EngineStatus::DeviceLost { hash_count: 0 };
        }

        let search_start = Instant::now();
        let mut total_hashes: u64 = 0;
        let mut current_start = range.start;
        let mut batch_num = 0u64;

        let duty = WORKER_BUFFERS.with(|cell| {
            take_gpu_duty_cycle(
                cell.borrow_mut()
                    .as_mut()
                    .expect("CUDA buffers initialized"),
            )
        });
        let pipeline = cuda_pipeline_enabled();
        log::info!(
            target: "cuda_engine",
            "CUDA device {} search started: range {}..{}, batch size: {} nonces{}{}",
            device_index,
            format_u512(range.start),
            format_u512(range.end),
            self.batch_size,
            duty.map_or(String::new(), |d| format!(", GPU busy {d:.1}% since previous search")),
            if pipeline {
                ", host/GPU pipeline ON (MINER_CUDA_PIPELINE=1)"
            } else {
                ""
            }
        );

        while current_start <= range.end {
            if cancel.is_cancelled() {
                return EngineStatus::Cancelled {
                    hash_count: total_hashes,
                };
            }

            let this_batch_size =
                batch_size_for_start(&range, current_start, self.batch_size);
            let next_start = current_start.saturating_add(U512::from(this_batch_size));
            let next_batch = (next_start <= range.end).then(|| {
                (
                    next_start,
                    batch_size_for_start(&range, next_start, self.batch_size),
                )
            });

            let batch_result = WORKER_BUFFERS.with(|cell| {
                let mut slot = cell.borrow_mut();
                let buffers = slot.as_mut().expect("CUDA buffers initialized");
                run_single_batch(
                    buffers,
                    Some(&self.devices[device_index]),
                    ctx,
                    current_start,
                    this_batch_size,
                    next_batch,
                )
            });

            match batch_result {
                BatchResult::Found {
                    candidate,
                    hash_count,
                } => {
                    total_hashes += hash_count;
                    return EngineStatus::Found {
                        candidate,
                        hash_count: total_hashes,
                        origin: FoundOrigin::Cuda,
                    };
                }
                BatchResult::NotFound { hash_count } => {
                    total_hashes += hash_count;
                }
                BatchResult::DeviceLost => {
                    DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
                    WORKER_BUFFERS.with(|res| *res.borrow_mut() = None);
                    return EngineStatus::DeviceLost {
                        hash_count: total_hashes,
                    };
                }
            }

            current_start = current_start.saturating_add(U512::from(this_batch_size));
            batch_num += 1;

            if self.throttle_ms > 0 && current_start <= range.end {
                let sleep_interval =
                    std::time::Duration::from_millis((self.throttle_ms / 10).max(1));
                let mut remaining = std::time::Duration::from_millis(self.throttle_ms);
                while remaining > std::time::Duration::ZERO {
                    if cancel.is_cancelled() {
                        return EngineStatus::Cancelled {
                            hash_count: total_hashes,
                        };
                    }
                    let sleep_time = remaining.min(sleep_interval);
                    std::thread::sleep(sleep_time);
                    remaining = remaining.saturating_sub(sleep_time);
                }
            }

            if batch_num.is_multiple_of(10) {
                let elapsed = search_start.elapsed();
                let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
                log::debug!(
                    target: "cuda_engine",
                    "CUDA device {} batch {} complete: {} hashes so far ({:.2}s, {})",
                    device_index,
                    batch_num,
                    total_hashes,
                    elapsed.as_secs_f64(),
                    format_hashrate(hash_rate)
                );
            }
        }

        EngineStatus::Exhausted {
            hash_count: total_hashes,
        }
    }
}

#[cfg(test)]
mod tests {
    //! CUDA correctness tests for `mining_main` / `hash_nonces` vs `pow_core`.
    //!
    //! **Golden vectors:** `pow_core::NONCE_HASH_KVS` in `crates/pow-core/src/lib.rs`
    //! (five header/nonce/hash/midstate fixtures). They are regression anchors for
    //! `qpow_math::get_nonce_hash` in `pow-core` (`nonce_hash_golden_vectors`,
    //! `test_hash_matches_qpow_math`); keep them bit-stable when editing kernels.

    use super::*;
    use cudarc::nvrtc::compile_ptx_with_opts;
    use engine_cpu::AtomicBoolCancelCheck;
    use std::sync::atomic::AtomicBool;
    use std::sync::Mutex;

    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_isolated_cuda_env(vars: &[(&str, &str)], f: impl FnOnce()) {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        const KEYS: &[&str] = &[
            "REQUIRE_CUDA",
            "MINER_CUDA_NONCE_LINE",
            "MINER_CUDA_K_SEGMENT",
            "MINER_CUDA_CPP_OUT0",
            "MINER_CUDA_NONCE_LINE_PIPE",
            "MINER_CUDA_PIPE_SBOX",
        ];
        let saved: Vec<(String, Option<String>)> = KEYS
            .iter()
            .map(|k| ((*k).to_string(), std::env::var(k).ok()))
            .collect();
        unsafe {
            for k in KEYS {
                std::env::remove_var(k);
            }
        }
        for (k, v) in vars {
            unsafe {
                std::env::set_var(k, v);
            }
        }
        CudaEngine::clear_worker_resources();
        f();
        CudaEngine::clear_worker_resources();
        unsafe {
            for k in KEYS {
                std::env::remove_var(k);
            }
        }
        for (k, prev) in saved {
            unsafe {
                match prev {
                    Some(v) => std::env::set_var(&k, v),
                    None => {}
                }
            }
        }
    }

    fn decode32(s: &str) -> [u8; 32] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn decode64(s: &str) -> [u8; 64] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn engine_or_skip() -> Option<CudaEngine> {
        engine_or_skip_with(1024)
    }

    fn require_cuda_env() -> bool {
        std::env::var("REQUIRE_CUDA")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    }

    fn engine_or_skip_with(batch_size: u32) -> Option<CudaEngine> {
        match CudaEngine::try_new(batch_size, 0) {
            Ok(e) => Some(e),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not available") || msg.contains("No CUDA devices") {
                    if require_cuda_env() {
                        panic!("REQUIRE_CUDA=1: CUDA device required but unavailable: {msg}");
                    }
                    eprintln!("skipping CUDA test: {e}");
                    None
                } else {
                    panic!("CUDA engine init failed: {e}");
                }
            }
        }
    }

    fn engine_or_require_cuda_with(batch_size: u32) -> CudaEngine {
        unsafe {
            std::env::set_var("REQUIRE_CUDA", "1");
        }
        let engine = engine_or_skip_with(batch_size)
            .expect("REQUIRE_CUDA=1: engine_or_skip returned None");
        unsafe {
            std::env::remove_var("REQUIRE_CUDA");
        }
        engine
    }

    #[test]
    fn k_segment_plan_single_when_batch_fits() {
        with_isolated_cuda_env(&[], || {
            let start = U512::from(0x1000u64);
            let segs = plan_k_segments(start, DEFAULT_K_SEGMENT);
            assert_eq!(segs.len(), 1);
            assert_eq!(segs[0], (start, DEFAULT_K_SEGMENT));

            let small = plan_k_segments(start, 1024);
            assert_eq!(small, vec![(start, 1024)]);
        });
    }

    #[test]
    fn k_segment_plan_splits_large_batch() {
        with_isolated_cuda_env(&[], || {
            let start = U512::from(42u64);
            let batch = DEFAULT_K_SEGMENT + 1000;
            let segs = plan_k_segments(start, batch);
            assert_eq!(segs.len(), 2);
            assert_eq!(segs[0], (start, DEFAULT_K_SEGMENT));
            assert_eq!(segs[1], (start + U512::from(DEFAULT_K_SEGMENT), 1000));
        });
    }

    #[test]
    fn miner_cuda_nonce_line_env_parsing() {
        with_isolated_cuda_env(&[], || {
            assert!(!miner_cuda_nonce_line_env_enabled());
        });
        with_isolated_cuda_env(&[("MINER_CUDA_NONCE_LINE", "1")], || {
            assert!(miner_cuda_nonce_line_env_enabled());
        });
        with_isolated_cuda_env(&[("MINER_CUDA_NONCE_LINE", "true")], || {
            assert!(miner_cuda_nonce_line_env_enabled());
        });
    }

    #[test]
    fn k_segment_plan_nonce_line_shortens_first_segment_at_boundary() {
        let start = U512::from(900u64);
        let cap = 1024u32;
        let batch = 2048u32;
        let segs = plan_k_segments_with_cap(start, batch, cap, true);
        let mut cursor = start;
        let mut rem = batch;
        let mut expected = Vec::new();
        while rem > 0 {
            let room = pow_core::nonce_line::t_room_in_k(cursor);
            let sz = rem.min(cap).min(room).max(1);
            expected.push((cursor, sz));
            rem -= sz;
            cursor = pow_core::nonce_line::counter_add_t(cursor, sz);
        }
        assert_eq!(segs, expected);
        assert!(segs.len() >= 2);
    }

    #[test]
    fn cuda_default_mining_main_hash_nonces_match_pow_core_samples() {
        with_isolated_cuda_env(&[], || {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let header = decode32(v.header);
            let ctx = engine.prepare_context(header, U512::one());
            let start = U512::from_big_endian(&decode64(v.nonce));
            let count = 33u32;
            let gpu = engine
                .hash_nonces(header, start, count)
                .unwrap_or_else(|e| panic!("kv {i}: hash_nonces failed: {e}"));
            assert_eq!(gpu.len(), count as usize);
            for (j, hash) in gpu.into_iter().enumerate() {
                let nonce = start + U512::from(j);
                assert_eq!(
                    hash,
                    pow_core::hash_from_nonce(&ctx, nonce),
                    "kv {i} offset {j}"
                );
            }
        }
        CudaEngine::clear_worker_resources();
        });
    }

    #[test]
    fn cuda_mining_main_search_multisegment_exhausted_matches_pow_core_count() {
        with_isolated_cuda_env(&[("MINER_CUDA_K_SEGMENT", "1000")], || {
        let Some(engine) = engine_or_skip_with(3000) else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[0].header);
        let ctx = JobContext {
            header,
            difficulty: U512::one(),
            target: U512::zero(),
        };
        let start = U512::from(0x5000u64);
        let span = 2500u64;
        let end = start + U512::from(span - 1);
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, Range { start, end }, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Exhausted { hash_count } => {
                assert_eq!(hash_count, span);
            }
            other => panic!("expected Exhausted over k-segment batch, got {other:?}"),
        }
        });
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn low64_batches_stop_before_carry() {
        assert_eq!(nonces_until_low64_carry(U512::from(u64::MAX)), U512::one());
        assert_eq!(
            nonces_until_low64_carry(U512::from(u64::MAX - 7)),
            U512::from(8u64)
        );
        assert_eq!(
            nonces_until_low64_carry((U512::one() << 192) + U512::from(1u64)),
            (U512::one() << 64) - U512::one()
        );
    }

    #[test]
    fn cuda_matches_nonce_hash_golden_vectors() {
        with_isolated_cuda_env(&[], || {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let header = decode32(v.header);
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let want = U512::from_big_endian(&decode64(v.hash));
            let got = engine
                .hash_nonces(header, nonce, 1)
                .unwrap_or_else(|e| panic!("kv {i}: CUDA hash_nonces failed: {e}"));
            assert_eq!(got.len(), 1, "kv {i}");
            assert_eq!(got[0], want, "kv {i}: CUDA hash != golden");
        }
        CudaEngine::clear_worker_resources();
        });
    }

    #[test]
    fn cuda_search_matches_golden_target_boundaries() {
        with_isolated_cuda_env(&[], || {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let cancel = AtomicBool::new(false);
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let hash = U512::from_big_endian(&decode64(v.hash));
            for target in [hash - U512::one(), hash, hash + U512::one()] {
                let ctx = JobContext {
                    header: decode32(v.header),
                    difficulty: U512::one(),
                    target,
                };
                let status = engine.search_range(
                    &ctx,
                    Range {
                        start: nonce,
                        end: nonce,
                    },
                    &AtomicBoolCancelCheck(&cancel),
                );
                match status {
                    EngineStatus::Found { candidate, .. } if hash < target => {
                        assert_eq!(candidate.nonce, nonce, "kv {i}");
                        assert_eq!(candidate.hash, hash, "kv {i}");
                    }
                    EngineStatus::Exhausted { hash_count: 1 } if hash >= target => {}
                    other => panic!("kv {i}, target {target}: unexpected {other:?}"),
                }
            }
        }
        CudaEngine::clear_worker_resources();
        });
    }

    #[test]
    fn cuda_hash_batches_match_cpu_across_nonce_carries() {
        with_isolated_cuda_env(&[], || {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[4].header);
        let ctx = engine.prepare_context(header, U512::one());
        let high = U512::from(0xdead_beef_cafeu64) << 256;
        for bits in [32, 64, 128, 224] {
            let start = high + (U512::one() << bits) - U512::from(128u64);
            let hashes = engine.hash_nonces(header, start, 257).unwrap();
            for (i, hash) in hashes.into_iter().enumerate() {
                let nonce = start + U512::from(i);
                assert_eq!(
                    hash,
                    pow_core::hash_from_nonce(&ctx, nonce),
                    "nonce {nonce}"
                );
            }
        }
        CudaEngine::clear_worker_resources();
        });
    }

    const GOLDILOCKS_P: u128 = 0xffff_ffff_0000_0001;
    const EPS: u64 = 0xffff_ffff;

    /// Bit-exact model of the kernel's `reduce128` and its deviation from
    /// `value mod p`, so the GPU can be checked exactly and the deviation
    /// bounded on the host.
    fn reduce128_model(value: u128) -> (u64, u128) {
        let (r0, r1, r2, r3) = (
            value as u32,
            (value >> 32) as u32,
            (value >> 64) as u32,
            (value >> 96) as u32,
        );
        let low = ((r1 as u64) << 32) | r0 as u64;
        let (folded, carry) = low.overflowing_add(r2 as u64 * EPS);
        let (subtrahend, subtrahend_wrapped) = r3.overflowing_add(carry as u32);
        let (hi, hi_wrapped) = ((folded >> 32) as u32).overflowing_add(carry as u32);
        let shifted = ((hi as u64) << 32) | (folded as u32 as u64);
        let (result, borrowed) = shifted.overflowing_sub(subtrahend as u64);
        let mut error = 0u128;
        if subtrahend_wrapped {
            error += 1 << 32;
        }
        if hi_wrapped {
            error += GOLDILOCKS_P - EPS as u128;
        }
        if borrowed {
            error += EPS as u128;
        }
        (result, error % GOLDILOCKS_P)
    }

    /// Bit-exact model of the kernel's `gf64_add` and its deviation from `a + b mod p`.
    fn gf64_add_model(a: u64, b: u64) -> (u64, u128) {
        let (sum, carry) = a.overflowing_add(b);
        if !carry {
            return (sum, 0);
        }
        let (folded, wrapped) = sum.overflowing_add(EPS);
        (
            folded,
            if wrapped {
                GOLDILOCKS_P - EPS as u128
            } else {
                0
            },
        )
    }

    /// Edge cases built to hit every carry path, followed by 4096 random values.
    fn reduction_inputs() -> (Vec<u128>, Vec<u128>) {
        let edges = [
            0,
            1,
            0xffff_fffe,
            0xffff_ffff,
            0x1_0000_0000,
            GOLDILOCKS_P as u64 - 1,
            GOLDILOCKS_P as u64,
            GOLDILOCKS_P as u64 + 1,
            u64::MAX,
        ];
        let mut edge_values = Vec::new();
        for high in edges {
            for low in edges {
                edge_values.push(((high as u128) << 64) | low as u128);
            }
        }
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut random = Vec::new();
        for _ in 0..4096 {
            let mut value = 0u128;
            for _ in 0..2 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                value = (value << 64) | seed as u128;
            }
            random.push(value);
        }
        (edge_values, random)
    }

    #[test]
    fn reduction_models_deviate_only_on_carry_wraps() {
        let (edge_values, random) = reduction_inputs();
        for value in edge_values.iter().chain(&random) {
            let (result, error) = reduce128_model(*value);
            assert_eq!(
                result as u128 % GOLDILOCKS_P,
                (value % GOLDILOCKS_P + error) % GOLDILOCKS_P,
                "reduce128 model invariant broken for {value:032x}"
            );
        }
        assert_eq!(reduce128_model(1 << 96).1, EPS as u128, "2^96 borrows");
        let slips = random
            .iter()
            .filter(|v| reduce128_model(**v).1 != 0)
            .count();
        assert_eq!(
            slips,
            0,
            "slipped reductions among {} random inputs",
            random.len()
        );

        let mut random_slips = 0;
        for pair in random.windows(2) {
            let (a, b) = (pair[0] as u64, pair[1] as u64);
            let (result, error) = gf64_add_model(a, b);
            assert_eq!(
                result as u128 % GOLDILOCKS_P,
                (a as u128 + b as u128 + error) % GOLDILOCKS_P,
                "gf64_add model invariant broken for {a:016x} + {b:016x}"
            );
            random_slips += (error != 0) as usize;
        }
        assert_eq!(
            gf64_add_model(u64::MAX, u64::MAX).1,
            GOLDILOCKS_P - EPS as u128
        );
        assert_eq!(random_slips, 0, "slipped additions among random inputs");
    }

    #[test]
    fn reduction_slips_are_absent_in_millions_of_random_products() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let slips = (0..1 << 22)
            .filter(|_| reduce128_model(next() as u128 * next() as u128).1 != 0)
            .count();
        assert_eq!(slips, 0, "reducer slipped on random 64x64-bit products");
    }

    #[test]
    fn cuda_field_arithmetic_matches_models_bit_exactly() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let (edge_values, random) = reduction_inputs();
        let values: Vec<u128> = edge_values.into_iter().chain(random).collect();
        let input: Vec<u32> = values
            .iter()
            .flat_map(|v| (0..4).map(move |i| (v >> (32 * i)) as u32))
            .collect();
        let source = format!(
            "{KERNEL_SRC}\n\
             extern \"C\" __global__ void arith_test(const u32 *input, u64 *reduced, u64 *added, u32 count) {{\n\
                 u32 i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 if (i >= count) return;\n\
                 reduced[i] = reduce128(input[4*i], input[4*i+1], input[4*i+2], input[4*i+3]);\n\
                 u64 a = ((u64)input[4*i+1] << 32) | input[4*i];\n\
                 u64 b = ((u64)input[4*i+3] << 32) | input[4*i+2];\n\
                 added[i] = gf64_add(a, b);\n\
             }}"
        );
        let device = &engine.devices[0];
        let (major, minor) = device.ctx.compute_capability().unwrap();
        let device_arch = major * 10 + minor;
        let arch = nvrtc_ptx_target_arch(device_arch).unwrap_or_else(|e| {
            panic!("NVRTC arch selection for sm_{device_arch}: {e}");
        });
        let module = device
            .ctx
            .load_module(
                compile_ptx_with_opts(&source, nvrtc_compile_options(arch)).unwrap(),
            )
            .unwrap();
        let function = module.load_function("arith_test").unwrap();
        let stream = device.ctx.default_stream();
        let input = stream.clone_htod(&input).unwrap();
        let mut reduced = stream.alloc_zeros::<u64>(values.len()).unwrap();
        let mut added = stream.alloc_zeros::<u64>(values.len()).unwrap();
        let count = values.len() as u32;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&input)
            .arg(&mut reduced)
            .arg(&mut added)
            .arg(&count);
        unsafe {
            launch.launch(LaunchConfig::for_num_elems(count)).unwrap();
        }
        stream.synchronize().unwrap();
        let reduced = stream.clone_dtoh(&reduced).unwrap();
        let added = stream.clone_dtoh(&added).unwrap();
        for (i, value) in values.iter().enumerate() {
            assert_eq!(
                reduced[i],
                reduce128_model(*value).0,
                "reduce128 of {value:032x}"
            );
            let (a, b) = (*value as u64, (*value >> 64) as u64);
            assert_eq!(
                added[i],
                gf64_add_model(a, b).0,
                "gf64_add of {a:016x} + {b:016x}"
            );
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_found_batch_counts_every_dispatched_nonce() {
        with_isolated_cuda_env(&[], || {
        let batch_size = 4_000_000u32;
        let Some(engine) = engine_or_skip_with(batch_size) else {
            return;
        };
        // Difficulty 1 makes every nonce a candidate. No thread stops early, so
        // the launch evaluates the whole batch; only MAX_HITS candidates are
        // recorded and the returned one must be a CPU-verified member of the range.
        let header = decode32(pow_core::NONCE_HASH_KVS[1].header);
        let ctx = engine.prepare_context(header, U512::one());
        let start = U512::from(0xfeed_face_0000_0000u64);
        let end = start + U512::from(batch_size - 1);
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, Range { start, end }, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Found {
                candidate,
                hash_count,
                ..
            } => {
                assert!(candidate.nonce >= start && candidate.nonce <= end);
                assert_eq!(
                    pow_core::hash_from_nonce(&ctx, candidate.nonce),
                    candidate.hash
                );
                assert!(candidate.hash < ctx.target);
                assert_eq!(hash_count, batch_size as u64);
            }
            other => panic!("expected Found, got {other:?}"),
        }
        CudaEngine::clear_worker_resources();
        });
    }

    #[test]
    fn cuda_search_rejects_prefix_equal_candidate_and_returns_lowest_valid() {
        with_isolated_cuda_env(&[], || {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        // Target equal to the golden hash: the golden nonce at index 0 is a
        // prefix-equal candidate that CPU verification rejects. The range holds
        // MAX_HITS nonces so every candidate is recorded, and the engine must
        // return exactly the lowest nonce the CPU finds valid, or Exhausted if
        // there is none.
        let v = &pow_core::NONCE_HASH_KVS[4];
        let start = U512::from_big_endian(&decode64(v.nonce));
        let target = U512::from_big_endian(&decode64(v.hash));
        let ctx = JobContext {
            header: decode32(v.header),
            difficulty: U512::one(),
            target,
        };
        let count = MAX_HITS as u64;
        let end = start + U512::from(count - 1);
        let lowest_valid = (0..count)
            .map(|i| start + U512::from(i))
            .find(|&nonce| pow_core::hash_from_nonce(&ctx, nonce) < target);
        assert_ne!(
            lowest_valid,
            Some(start),
            "the golden nonce itself must not qualify"
        );
        let cancel = AtomicBool::new(false);
        match (
            engine.search_range(&ctx, Range { start, end }, &AtomicBoolCancelCheck(&cancel)),
            lowest_valid,
        ) {
            (
                EngineStatus::Found {
                    candidate,
                    hash_count,
                    ..
                },
                Some(expected),
            ) => {
                assert_eq!(candidate.nonce, expected);
                assert_eq!(candidate.hash, pow_core::hash_from_nonce(&ctx, expected));
                assert_eq!(hash_count, count);
            }
            (EngineStatus::Exhausted { hash_count }, None) => assert_eq!(hash_count, count),
            (other, expected) => panic!("unexpected {other:?} for expected {expected:?}"),
        }
        CudaEngine::clear_worker_resources();
        });
    }

    #[test]
    fn cuda_search_finds_cpu_verified_solution() {
        with_isolated_cuda_env(&[], || {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[2].header);
        let difficulty = U512::from(1u64);
        let ctx = engine.prepare_context(header, difficulty);
        let start = U512::from(0x1234_5678_90ab_cdefu64);
        let range = Range {
            start,
            end: start + U512::from(16u64),
        };
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, range, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Found { candidate, .. } => {
                let cpu = pow_core::hash_from_nonce(&ctx, candidate.nonce);
                assert_eq!(cpu, candidate.hash);
                assert!(cpu < ctx.target);
            }
            other => panic!("expected Found, got {other:?}"),
        }
        CudaEngine::clear_worker_resources();
        });
    }

    /// PILOT-001 regression: baseline NVRTC toggles (0/0/1) vs candidate-style (1/1/8) must
    /// match `pow_core` on golden vectors and agree on exhausted nonce-line search counts.
    #[test]
    fn cuda_pilot_unroll_configs_match_pow_core_and_each_other() {
        const CONFIGS: &[(&str, &[(&str, &str)])] = &[
            (
                "baseline_unroll",
                &[
                    ("MINER_CUDA_NONCE_LINE", "1"),
                    ("MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL", "0"),
                    ("MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL", "0"),
                    ("MINER_CUDA_NONCE_LINE_T_UNROLL", "1"),
                ],
            ),
            (
                "pilot_candidate_unroll",
                &[
                    ("MINER_CUDA_NONCE_LINE", "1"),
                    ("MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL", "1"),
                    ("MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL", "1"),
                    ("MINER_CUDA_NONCE_LINE_T_UNROLL", "8"),
                ],
            ),
        ];
        let mut baseline_exhaust: Option<u64> = None;
        for (label, vars) in CONFIGS {
            with_isolated_cuda_env(vars, || {
                CudaEngine::clear_worker_resources();
                let Some(engine) = engine_or_skip_with(4096) else {
                    eprintln!("skipping cuda_pilot_unroll_configs ({label}): no CUDA device");
                    return;
                };
                if !miner_cuda_nonce_line_kernel_ready() {
                    eprintln!("skipping cuda_pilot_unroll_configs ({label}): kernel not ready");
                    CudaEngine::clear_worker_resources();
                    return;
                }
                for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
                    let header = decode32(v.header);
                    let nonce = U512::from_big_endian(&decode64(v.nonce));
                    let want = U512::from_big_endian(&decode64(v.hash));
                    let got = engine
                        .hash_nonces(header, nonce, 1)
                        .unwrap_or_else(|e| panic!("{label} kv {i}: {e}"));
                    assert_eq!(got[0], want, "{label} kv {i}: CUDA hash != golden");
                }
                let header = decode32(pow_core::NONCE_HASH_KVS[0].header);
                let ctx = engine.prepare_context(header, U512::MAX);
                let start = U512::from(0x8000u64);
                let span = 4096u64;
                let end = start + U512::from(span - 1);
                let cancel = AtomicBool::new(false);
                let status = engine.search_range(
                    &ctx,
                    Range { start, end },
                    &AtomicBoolCancelCheck(&cancel),
                );
                let hash_count = match status {
                    EngineStatus::Exhausted { hash_count } => hash_count,
                    other => panic!("{label}: expected Exhausted, got {other:?}"),
                };
                assert_eq!(
                    hash_count, span,
                    "{label}: hash_count must equal dispatched nonces"
                );
                if label == &"baseline_unroll" {
                    baseline_exhaust = Some(hash_count);
                } else {
                    assert_eq!(
                        baseline_exhaust,
                        Some(hash_count),
                        "pilot candidate unroll must dispatch same nonce count as baseline toggles"
                    );
                }
                CudaEngine::clear_worker_resources();
            });
        }
    }

    #[test]
    fn cuda_nonce_line_mining_main_matches_pow_core_golden_vectors() {
        with_isolated_cuda_env(&[("MINER_CUDA_NONCE_LINE", "1")], || {
            let Some(engine) = engine_or_skip() else {
                eprintln!("skipping CUDA nonce-line golden test: no CUDA device");
                return;
            };
            if !miner_cuda_nonce_line_kernel_ready() {
                eprintln!("skipping CUDA nonce-line golden test: kernel not ready");
                return;
            }
            for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
                let header = decode32(v.header);
                let nonce = U512::from_big_endian(&decode64(v.nonce));
                let want = U512::from_big_endian(&decode64(v.hash));
                let got = engine
                    .hash_nonces(header, nonce, 1)
                    .unwrap_or_else(|e| panic!("kv {i}: nonce-line hash_nonces failed: {e}"));
                assert_eq!(got[0], want, "kv {i}: nonce-line CUDA hash != golden");
            }
            CudaEngine::clear_worker_resources();
        });
    }

    /// With `MINER_CUDA_NONCE_LINE` set, skip gracefully when CUDA is absent.
    #[test]
    fn cuda_nonce_line_stub_skips_without_gpu() {
        with_isolated_cuda_env(&[("MINER_CUDA_NONCE_LINE", "1")], || {
            let Some(engine) = engine_or_skip() else {
                eprintln!("skipping CUDA nonce-line test: CUDA driver/device unavailable");
                return;
            };
            if !miner_cuda_nonce_line_kernel_ready() {
                eprintln!("skipping CUDA nonce-line test: kernel path not compiled (A1 pending)");
                CudaEngine::clear_worker_resources();
                return;
            }
            let header = decode32(pow_core::NONCE_HASH_KVS[0].header);
            let ctx = engine.prepare_context(header, U512::one());
            let start = U512::from(7u64);
            let gpu = engine
                .hash_nonces(header, start, 4)
                .expect("nonce-line hash path");
            for (j, hash) in gpu.into_iter().enumerate() {
                assert_eq!(
                    hash,
                    pow_core::hash_from_nonce(&ctx, start + U512::from(j)),
                );
            }
            CudaEngine::clear_worker_resources();
        });
    }

    #[test]
    fn cuda_nonce_line_host_plan_with_env_no_gpu_required() {
        with_isolated_cuda_env(
            &[("MINER_CUDA_NONCE_LINE", "1"), ("MINER_CUDA_K_SEGMENT", "1024")],
            || {
                assert!(miner_cuda_nonce_line_env_enabled());
                let start = U512::from(1000u64);
                let cap = 1024u32;
                let batch = 1500u32;
                let segs = plan_k_segments_with_cap(start, batch, cap, true);
                let mut cursor = start;
                let mut rem = batch;
                let mut expected = Vec::new();
                while rem > 0 {
                    let room = pow_core::nonce_line::t_room_in_k(cursor);
                    let sz = rem.min(cap).min(room).max(1);
                    expected.push((cursor, sz));
                    rem -= sz;
                    cursor = pow_core::nonce_line::counter_add_t(cursor, sz);
                }
                assert_eq!(segs, expected);
            },
        );
    }

    /// Regression: NVRTC must see `gf64_mul` / `gf64_sqr` before `gf64_sbox_pair`
    /// (mis-ordering broke Salad deploy with undefined identifier errors at field.cuh ~71).
    #[test]
    fn field_cuh_sbox_pair_follows_mul_and_sqr() {
        let src = include_str!("kernels/field.cuh");
        let mul = src
            .find("u64 gf64_mul(")
            .expect("field.cuh must define gf64_mul");
        let sqr = src
            .find("u64 gf64_sqr(")
            .expect("field.cuh must define gf64_sqr");
        let pair = src
            .find("void gf64_sbox_pair(")
            .expect("field.cuh must define gf64_sbox_pair");
        assert!(
            mul < pair && sqr < pair,
            "gf64_sbox_pair must appear after gf64_mul and gf64_sqr (NVRTC forward references fail)"
        );
    }

    /// Regression: only `cmp == 1` (hash above target) may reject; `cmp == 2` is a valid candidate.
    #[test]
    fn mining_entry_cu_only_rejects_cmp_eq_one() {
        let src = include_str!("kernels/mining_entry.cu");
        assert!(
            src.contains("hash_high_half_cmp_be"),
            "nonce_line path must use BE limb compare helper"
        );
        assert!(
            !src.contains("if (cmp != 1u)") && !src.contains("if (cmp == 2u)"),
            "mining_entry.cu must not reject cmp==2 (below-target hits)"
        );
        assert!(
            src.matches("if (cmp == 1u)").count() >= 3,
            "expected reject-only-on-cmp==1 at legacy, dual, and nonce_line sites"
        );
    }

    fn with_nvrtc_env(vars: &[(&str, &str)], f: impl FnOnce()) {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let keys: Vec<&str> = vars.iter().map(|(k, _)| *k).collect();
        let saved: Vec<(String, Option<String>)> = keys
            .iter()
            .map(|k| ((*k).to_string(), std::env::var(k).ok()))
            .collect();
        unsafe {
            for k in &keys {
                std::env::remove_var(k);
            }
        }
        for (k, v) in vars {
            unsafe {
                std::env::set_var(k, v);
            }
        }
        CudaEngine::clear_worker_resources();
        f();
        CudaEngine::clear_worker_resources();
        unsafe {
            for k in &keys {
                std::env::remove_var(k);
            }
        }
        for (k, prev) in saved {
            unsafe {
                if let Some(v) = prev {
                    std::env::set_var(&k, v);
                }
            }
        }
    }

    /// Matches `hash_high_half_cmp_be` / `target_cmp.cuh` (0=equal prefix, 1=reject, 2=candidate).
    fn cpu_hash_high_half_cmp_be(first_be: &[u32; 8], tgt_hi: &[u32; 8], cmp_limbs: usize) -> u32 {
        let n = cmp_limbs.min(8);
        let mut cmp = 0u32;
        for i in 0..n {
            let h = first_be[i];
            let t = tgt_hi[7 - i];
            if h != t {
                cmp = if h > t { 1 } else { 2 };
                break;
            }
        }
        if cmp == 0 && n < 8 {
            for i in n..8 {
                let h = first_be[i];
                let t = tgt_hi[7 - i];
                if h != t {
                    cmp = if h > t { 1 } else { 2 };
                    break;
                }
            }
        }
        cmp
    }

    fn hash_u512_to_first_be8(hash: U512) -> [u32; 8] {
        let be = hash.to_big_endian();
        std::array::from_fn(|i| {
            u32::from_be_bytes(be[i * 4..i * 4 + 4].try_into().unwrap())
        })
    }

    fn job_tgt_hi8(ctx: &JobContext) -> [u32; 8] {
        let be = ctx.target.to_big_endian();
        std::array::from_fn(|i| {
            u32::from_be_bytes(be[32 + i * 4..32 + i * 4 + 4].try_into().unwrap())
        })
    }

    fn probe_nonce_line_first_be(
        device: &CudaDevice,
        pk: &[u64; 14],
        t: u32,
    ) -> Result<[u32; 8], String> {
        let (major, minor) = device
            .ctx
            .compute_capability()
            .map_err(|e| format!("compute capability: {e}"))?;
        let device_arch = major * 10 + minor;
        let arch = nvrtc_ptx_target_arch(device_arch)?;
        let source = format!(
            "{KERNEL_SRC}\n\
             extern \"C\" __global__ void probe_nonce_line_high_half(\n\
                 const u64 *pk14, u32 t, u32 *out_first_be8) {{\n\
                 u64 pk[NONCE_LINE_K_PARAMS];\n\
                 #pragma unroll\n\
                 for (int i = 0; i < NONCE_LINE_K_PARAMS; i++) {{ pk[i] = pk14[i]; }}\n\
                 u64 u, v;\n\
                 gf64_nonce_line_uv_at(pk[0], pk[1], t, u, v);\n\
                 u32 first_be[8];\n\
                 nonce_line_hash_high_half_from_uv(pk, u, v, first_be);\n\
                 #pragma unroll\n\
                 for (int i = 0; i < 8; i++) {{ out_first_be8[i] = first_be[i]; }}\n\
             }}"
        );
        let module = device
            .ctx
            .load_module(compile_ptx_with_opts(&source, nvrtc_compile_options(arch)).map_err(
                |e| format!("NVRTC probe compile: {e}"),
            )?)
            .map_err(|e| format!("load probe module: {e}"))?;
        let function = module
            .load_function("probe_nonce_line_high_half")
            .map_err(|e| format!("load probe: {e}"))?;
        let stream = device.ctx.default_stream();
        let pk_dev = stream
            .clone_htod(pk)
            .map_err(|e| format!("pk h2d: {e}"))?;
        let mut out = stream
            .alloc_zeros::<u32>(8)
            .map_err(|e| format!("out alloc: {e}"))?;
        let mut launch = stream.launch_builder(&function);
        launch.arg(&pk_dev).arg(&t).arg(&mut out);
        unsafe {
            launch
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("probe launch: {e}"))?;
        }
        stream.synchronize().map_err(|e| format!("sync: {e}"))?;
        let raw = stream
            .clone_dtoh(&out)
            .map_err(|e| format!("dtoh: {e}"))?;
        Ok(std::array::from_fn(|i| raw[i]))
    }

    fn pilot002_baseline_nvrtc_env(cpp_out0: bool) -> Vec<(&'static str, &'static str)> {
        let mut v = vec![
            ("MINER_CUDA_NONCE_LINE", "1"),
            ("MINER_CUDA_NONCE_LINE_EXT_SBOX_UNROLL", "0"),
            ("MINER_CUDA_NONCE_LINE_TERM_EXT_UNROLL", "0"),
            ("MINER_CUDA_NONCE_LINE_T_UNROLL", "1"),
            ("MINER_CUDA_NONCE_LINE_PIPE", "1"),
            ("MINER_CUDA_NONCE_LINE_SH_PK", "1"),
            ("MINER_CUDA_FUSED_DIAG", "1"),
        ];
        if cpp_out0 {
            v.push(("MINER_CUDA_CPP_OUT0", "1"));
        }
        v
    }

    fn nonce_line_sample_points() -> Vec<( [u8; 32], U512, u32)> {
        use pow_core::nonce_line::{counter_add_t, counter_with_t, T_SPAN};
        let t_span = T_SPAN as u32;
        let mut out = Vec::new();
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let header = decode32(v.header);
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let ctr_be = nonce.to_big_endian();
            let t = pow_core::nonce_line::t_of(&ctr_be);
            out.push((header, nonce, t));
            if i == 0 {
                for &dt in &[0u32, 1, t_span - 1, t_span / 2] {
                    let c = counter_with_t(&ctr_be, dt);
                    out.push((header, U512::from_big_endian(&c), dt));
                }
            }
        }
        let header = decode32(pow_core::NONCE_HASH_KVS[0].header);
        let base = U512::from(0x8000u64);
        let be = base.to_big_endian();
        out.push((header, base, pow_core::nonce_line::t_of(&be)));
        let near_k_end = counter_add_t(base, t_span - 1);
        let be2 = near_k_end.to_big_endian();
        out.push((
            header,
            near_k_end,
            pow_core::nonce_line::t_of(&be2),
        ));
        let mut seed = 0xca7a_f00d_u64;
        for _ in 0..128 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let ctr = U512::from(seed);
            let be = ctr.to_big_endian();
            let t = pow_core::nonce_line::t_of(&be) % t_span;
            let c = counter_with_t(&be, t);
            out.push((header, U512::from_big_endian(&c), t));
        }
        out
    }

    fn nonce_line_uv_at(pk0: u64, pk1: u64, t: u32) -> (u64, u64) {
        pow_core::nonce_line::uv_at(pk0, pk1, t)
    }

    /// Build `tgt_hi[8]` for `hash_high_half_cmp_be` (index `7 - i` compares to `first_be[i]`).
    fn tgt_hi_matching_first_be(first_be: &[u32; 8]) -> [u32; 8] {
        let mut tgt = [0u32; 8];
        for i in 0..8 {
            tgt[7 - i] = first_be[i];
        }
        tgt
    }

    fn tgt_hi_with_limb_delta(first_be: &[u32; 8], limb: usize, delta: i64) -> [u32; 8] {
        let mut tgt = tgt_hi_matching_first_be(first_be);
        let idx = 7 - limb;
        if delta >= 0 {
            tgt[idx] = tgt[idx].saturating_add(delta as u32);
        } else {
            tgt[idx] = tgt[idx].saturating_sub((-delta) as u32);
        }
        tgt
    }

    /// Limbs 0–1 match `first_be`; limb 2+ set high so comparison proceeds past a tied prefix.
    fn tgt_hi_adversarial_equal_prefix_differ_later(first_be: &[u32; 8]) -> [u32; 8] {
        let mut tgt = tgt_hi_matching_first_be(first_be);
        for limb in 2..8 {
            tgt[7 - limb] = 0xffff_ffff;
        }
        tgt
    }

    fn assert_no_false_negative_filter(
        label: &str,
        case: &str,
        cpu_cmp: u32,
        gpu_cmp: u32,
    ) {
        assert_eq!(
            gpu_cmp, cpu_cmp,
            "{label} {case}: GPU cmp {gpu_cmp} != CPU cmp {cpu_cmp}"
        );
        if cpu_cmp != 1 {
            assert_ne!(
                gpu_cmp, 1,
                "{label} {case}: false negative — CPU accepts filter but GPU rejects"
            );
        }
    }

    /// PILOT-002: CPP_OUT0 must match legacy high-half filter decisions and dispatch; no false negatives vs CPU.
    #[test]
    fn cuda_cpp_out0_matches_pow_core_high_half_and_dispatch() {
        const CONFIGS: &[(&str, bool)] = &[("legacy_perm", false), ("cpp_out0", true)];
        let strict_cuda = require_cuda_env();
        let cmp_limbs = cuda_cmp_limbs() as usize;
        let mut legacy_exhaust: Option<u64> = None;
        let mut legacy_probes: Option<Vec<([u32; 8], u32, U512)>> = None;

        for (label, cpp_out0) in CONFIGS {
            let env = pilot002_baseline_nvrtc_env(*cpp_out0);
            with_nvrtc_env(&env, || {
                CudaEngine::clear_worker_resources();
                let Some(engine) = engine_or_skip_with(8192) else {
                    if strict_cuda {
                        panic!("REQUIRE_CUDA=1: CUDA device required for {label}");
                    }
                    eprintln!("skipping cuda_cpp_out0 ({label}): no CUDA device");
                    return;
                };
                assert!(
                    miner_cuda_nonce_line_kernel_ready(),
                    "nonce-line kernel not ready ({label})"
                );
                let device = &engine.devices[0];
                let mut probe_rows = Vec::new();

                for (header, counter, t) in nonce_line_sample_points() {
                    let ctx = engine.prepare_context(header, U512::MAX);
                    let pk_arr =
                        pow_core::nonce_line::segment_launch_params(header, counter);
                    let _ = nonce_line_uv_at(pk_arr[0], pk_arr[1], t);
                    let first_be = probe_nonce_line_first_be(device, &pk_arr, t)
                        .unwrap_or_else(|e| panic!("{label} probe at t={t}: {e}"));
                    let nonce = pow_core::nonce_line::nonce_from_absolute_t(counter, t);
                    let full = pow_core::hash_from_nonce(&ctx, nonce);
                    let cpu_first = hash_u512_to_first_be8(full);
                    let tgt_hi = job_tgt_hi8(&ctx);
                    let cpu_cmp = cpu_hash_high_half_cmp_be(&cpu_first, &tgt_hi, cmp_limbs);
                    let gpu_cmp = cpu_hash_high_half_cmp_be(&first_be, &tgt_hi, cmp_limbs);

                    if !cpp_out0 {
                        assert_eq!(
                            first_be, cpu_first,
                            "{label}: GPU legacy first_be != CPU full hash prefix at t={t}"
                        );
                    } else if let Some(ref leg_rows) = legacy_probes {
                        let leg = leg_rows
                            .iter()
                            .find(|r| r.1 == t && r.2 == counter)
                            .unwrap_or_else(|| {
                                panic!("missing legacy probe for counter={counter} t={t}")
                            });
                        assert_eq!(
                            first_be[0], leg.0[0],
                            "cpp_out0 limb0 mismatch at t={t}"
                        );
                        assert_eq!(
                            first_be[1], leg.0[1],
                            "cpp_out0 limb1 mismatch at t={t}"
                        );
                        for i in 2..8 {
                            assert_eq!(
                                first_be[i], 0,
                                "cpp_out0 must zero limbs 2..7 at t={t}, got [{}]={}",
                                i,
                                first_be[i]
                            );
                        }
                        let leg_cmp = cpu_hash_high_half_cmp_be(&leg.0, &tgt_hi, cmp_limbs);
                        assert_eq!(
                            gpu_cmp, leg_cmp,
                            "cpp_out0 filter decision != legacy at t={t} (false negative/positive risk)"
                        );
                    }

                    assert_eq!(
                        gpu_cmp, cpu_cmp,
                        "{label}: GPU filter cmp {gpu_cmp} != CPU cmp {cpu_cmp} at t={t} \
                         (false negative if CPU accepts and GPU rejects)"
                    );
                    probe_rows.push((first_be, t, counter));
                }

                let header = decode32(pow_core::NONCE_HASH_KVS[0].header);
                let base = U512::from(0x9000u64);
                let pk_arr = pow_core::nonce_line::segment_launch_params(header, base);
                let probe0 = probe_nonce_line_first_be(device, &pk_arr, 0)
                    .expect("target sweep probe");
                let nonce0 = pow_core::nonce_line::nonce_from_absolute_t(base, 0);
                let cpu_first0 = hash_u512_to_first_be8(pow_core::hash_from_nonce(
                    &JobContext::new(header, U512::MAX),
                    nonce0,
                ));
                let mut seed = 0xdead_beef_u64;
                for _ in 0..64 {
                    seed ^= seed << 7;
                    seed ^= seed >> 9;
                    seed ^= seed << 8;
                    let target = U512::from(seed) | (U512::one() << 255);
                    let tgt_hi = job_tgt_hi8(&JobContext::new(header, target));
                    let cpu_cmp = cpu_hash_high_half_cmp_be(&cpu_first0, &tgt_hi, cmp_limbs);
                    let gpu_cmp = cpu_hash_high_half_cmp_be(&probe0, &tgt_hi, cmp_limbs);
                    assert_no_false_negative_filter(label, "random_target", cpu_cmp, gpu_cmp);
                }

                let leg0 = if *label == "legacy_perm" {
                    probe0
                } else {
                    legacy_probes
                        .as_ref()
                        .and_then(|rows| {
                            rows.iter()
                                .find(|r| r.1 == 0 && r.2 == base)
                                .map(|r| r.0)
                        })
                        .expect("legacy probe for adversarial base")
                };
                for limb in 0..8 {
                    for (case, tgt_hi) in [
                        (
                            "exact",
                            tgt_hi_matching_first_be(&leg0),
                        ),
                        (
                            "limb_below",
                            tgt_hi_with_limb_delta(&leg0, limb, -1),
                        ),
                        (
                            "limb_above",
                            tgt_hi_with_limb_delta(&leg0, limb, 1),
                        ),
                    ] {
                        let cpu_cmp =
                            cpu_hash_high_half_cmp_be(&cpu_first0, &tgt_hi, cmp_limbs);
                        let gpu_cmp = cpu_hash_high_half_cmp_be(&probe0, &tgt_hi, cmp_limbs);
                        assert_no_false_negative_filter(
                            label,
                            &format!("boundary_{case}_limb{limb}"),
                            cpu_cmp,
                            gpu_cmp,
                        );
                        if *label == "cpp_out0" {
                            let leg_cmp =
                                cpu_hash_high_half_cmp_be(&leg0, &tgt_hi, cmp_limbs);
                            assert_eq!(
                                gpu_cmp, leg_cmp,
                                "cpp_out0 {case} limb{limb}: must match legacy filter decision"
                            );
                        }
                    }
                }
                let adv = tgt_hi_adversarial_equal_prefix_differ_later(&leg0);
                let cpu_adv = cpu_hash_high_half_cmp_be(&cpu_first0, &adv, cmp_limbs);
                let gpu_adv = cpu_hash_high_half_cmp_be(&probe0, &adv, cmp_limbs);
                assert_no_false_negative_filter(label, "adversarial_prefix_eq_later_diff", cpu_adv, gpu_adv);
                if *label == "cpp_out0" {
                    let leg_adv = cpu_hash_high_half_cmp_be(&leg0, &adv, cmp_limbs);
                    assert_eq!(
                        gpu_adv, leg_adv,
                        "cpp_out0 adversarial: must match legacy (OUT0 zero limbs 2..7 vs full cmp)"
                    );
                }

                for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
                    let header = decode32(v.header);
                    let nonce = U512::from_big_endian(&decode64(v.nonce));
                    let cancel = AtomicBool::new(false);
                    let hash = U512::from_big_endian(&decode64(v.hash));
                    for target in [hash - U512::one(), hash, hash + U512::one()] {
                        let ctx_t = JobContext {
                            header,
                            difficulty: U512::one(),
                            target,
                        };
                        let status = engine.search_range(
                            &ctx_t,
                            Range {
                                start: nonce,
                                end: nonce,
                            },
                            &AtomicBoolCancelCheck(&cancel),
                        );
                        let cpu_ok = hash < target;
                        match (status, cpu_ok) {
                            (
                                EngineStatus::Found { candidate, .. },
                                true,
                            ) => {
                                assert_eq!(candidate.nonce, nonce, "{label} kv {i}");
                                assert_eq!(
                                    candidate.hash,
                                    pow_core::hash_from_nonce(&ctx_t, candidate.nonce),
                                    "{label} golden kv {i}: full hash mismatch on hit"
                                );
                            }
                            (EngineStatus::Exhausted { hash_count: 1 }, false) => {}
                            (other, cpu_ok) => panic!(
                                "{label} golden kv {i} target {target} cpu_ok={cpu_ok}: {other:?}"
                            ),
                        }
                    }
                }

                let header = decode32(pow_core::NONCE_HASH_KVS[0].header);
                let ctx = engine.prepare_context(header, U512::MAX);
                let start = U512::from(0x8000u64);
                let span = 4096u64;
                let end = start + U512::from(span - 1);
                let cancel = AtomicBool::new(false);
                let status = engine.search_range(
                    &ctx,
                    Range { start, end },
                    &AtomicBoolCancelCheck(&cancel),
                );
                let hash_count = match status {
                    EngineStatus::Exhausted { hash_count } => hash_count,
                    EngineStatus::Found { hash_count, candidate, .. } => {
                        assert_eq!(
                            candidate.hash,
                            pow_core::hash_from_nonce(&ctx, candidate.nonce),
                            "{label}: hit hash must match pow_core"
                        );
                        hash_count
                    }
                    other => panic!("{label}: expected Exhausted or Found, got {other:?}"),
                };
                assert_eq!(
                    hash_count, span,
                    "{label}: dispatch count must equal span"
                );

                let ctx_hit = engine.prepare_context(header, U512::MAX);
                let cancel = AtomicBool::new(false);
                if let EngineStatus::Found {
                    candidate,
                    hash_count: 1,
                    ..
                } = engine.search_range(
                    &ctx_hit,
                    Range {
                        start,
                        end: start,
                    },
                    &AtomicBoolCancelCheck(&cancel),
                ) {
                    assert_eq!(
                        candidate.hash,
                        pow_core::hash_from_nonce(&ctx_hit, candidate.nonce),
                        "{label}: single-nonce hit hash"
                    );
                }

                if *label == "legacy_perm" {
                    legacy_exhaust = Some(hash_count);
                    legacy_probes = Some(probe_rows);
                } else {
                    assert_eq!(
                        legacy_exhaust,
                        Some(hash_count),
                        "cpp_out0 dispatch count must match legacy"
                    );
                }
                CudaEngine::clear_worker_resources();
            });
        }
    }

    /// Salad RTX 5090 default NVRTC `-D` profile (from quantus-live-001 deploy log).
    #[test]
    fn nvrtc_compiles_mining_kernel_with_salad_default_flags() {
        let salad_env = [
            ("MINER_CUDA_GF64_32BIT", "1"),
            ("MINER_CUDA_MIN_BLOCKS", "4"),
            ("MINER_CUDA_NONCE_LINE", "1"),
            ("MINER_CUDA_NONCE_LINE_SH_PK", "1"),
            ("MINER_CUDA_NONCE_LINE_PIPE", "1"),
            ("MINER_CUDA_FUSED_DIAG", "1"),
        ];
        with_nvrtc_env(&salad_env, || {
            let opts = nvrtc_compile_options(90);
            let ptx = silent_catch(|| compile_ptx_with_opts(KERNEL_SRC, opts));
            match ptx {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    let msg = e.to_string();
                    if msg.contains("not available") || msg.contains("libnvrtc") {
                        eprintln!("skipping NVRTC compile test: {msg}");
                        return;
                    }
                    panic!("NVRTC compile failed with Salad-like flags: {msg}");
                }
                Err(_) => {
                    eprintln!("skipping NVRTC compile test: libnvrtc not loadable on this host");
                }
            }
        });
    }
}
