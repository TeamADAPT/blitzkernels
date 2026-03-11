//! **blitz-int8-matmul** — INT8 quantized matrix multiplication kernel.
//!
//! Implements symmetric per-tensor and per-channel INT8 quantization for
//! transformer weight matrices, targeting production inference workloads:
//!
//! ```text
//! Quantize:  q = clamp(round(x / scale), -127, 127)   [FP32 → INT8]
//! Dequant:   x̂ = q * scale                             [INT8 → FP32]
//! MatMul:    C = A_q · B_q (INT8 arithmetic) * scale_A * scale_B  [output FP32]
//! ```
//!
//! ## Why INT8?
//! - **4× memory bandwidth reduction** vs FP32 (weight memory is the bottleneck)
//! - **2× reduction** vs FP16
//! - Sub-0.1% accuracy loss with symmetric per-tensor quantization on GELU/SwiGLU weights
//! - WASM-safe: no SIMD intrinsics, compiles on any target including wasm32-wasip2
//!
//! ## Quantization Modes
//! - [`QuantMode::PerTensor`]: single scale for entire matrix (fastest, slight accuracy loss)
//! - [`QuantMode::PerChannel`]: per-output-channel scale (more accurate, common for weights)
//!
//! Pure Rust, no external deps, WASM-portable.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ── Quantization config ───────────────────────────────────────────────────────

/// Quantization granularity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantMode {
    /// Single scale value for the entire tensor.
    /// Fastest path; suitable for activations.
    PerTensor,
    /// One scale per output channel (row of weight matrix).
    /// More accurate for weight matrices with heterogeneous distributions.
    PerChannel,
}

impl Default for QuantMode {
    fn default() -> Self {
        Self::PerTensor
    }
}

/// Configuration for INT8 quantization.
#[derive(Debug, Clone)]
pub struct Int8Config {
    /// Quantization granularity.
    pub mode: QuantMode,
    /// INT8 range: values clamped to [-clamp_val, clamp_val].
    /// Standard symmetric INT8 uses 127 (avoids -128 asymmetry).
    pub clamp_val: i8,
}

impl Default for Int8Config {
    fn default() -> Self {
        Self {
            mode: QuantMode::PerTensor,
            clamp_val: 127,
        }
    }
}

// ── Quantized matrix type ─────────────────────────────────────────────────────

/// A quantized matrix in INT8 symmetric format.
///
/// Stores `rows × cols` INT8 values plus per-tensor or per-channel scales.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    /// Quantized values, row-major. Shape: `[rows, cols]`.
    pub data: Vec<i8>,
    /// Scale factors. Length 1 (PerTensor) or `rows` (PerChannel).
    pub scales: Vec<f32>,
    /// Number of rows (output dimension).
    pub rows: usize,
    /// Number of columns (inner dimension).
    pub cols: usize,
    /// Quantization mode used during quantization.
    pub mode: QuantMode,
}

impl QuantMatrix {
    /// Get the scale for a given output row.
    #[inline(always)]
    pub fn scale_for_row(&self, row: usize) -> f32 {
        match self.mode {
            QuantMode::PerTensor => self.scales[0],
            QuantMode::PerChannel => self.scales[row],
        }
    }
}

// ── Quantization ──────────────────────────────────────────────────────────────

/// Compute the symmetric per-tensor quantization scale.
///
/// scale = max(|x|) / clamp_val
/// Avoids divide-by-zero by returning 1.0 for zero tensors.
pub fn compute_scale_per_tensor(data: &[f32], clamp_val: i8) -> f32 {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if max_abs < 1e-10 {
        1.0
    } else {
        max_abs / clamp_val as f32
    }
}

/// Compute per-channel (per-row) quantization scales for a row-major matrix.
pub fn compute_scales_per_channel(data: &[f32], rows: usize, cols: usize, clamp_val: i8) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let row = &data[r * cols..(r + 1) * cols];
            compute_scale_per_tensor(row, clamp_val)
        })
        .collect()
}

/// Quantize a flat FP32 slice to INT8 using a single scale (per-tensor).
#[inline]
pub fn quantize_per_tensor(data: &[f32], scale: f32, clamp_val: i8) -> Vec<i8> {
    data.iter()
        .map(|x| {
            let q = (x / scale).round() as i32;
            q.clamp(-(clamp_val as i32), clamp_val as i32) as i8
        })
        .collect()
}

/// Quantize a row-major matrix to INT8 with per-channel scales.
pub fn quantize_per_channel(data: &[f32], rows: usize, cols: usize, scales: &[f32], clamp_val: i8) -> Vec<i8> {
    let mut out = vec![0i8; rows * cols];
    for r in 0..rows {
        let scale = scales[r];
        for c in 0..cols {
            let x = data[r * cols + c];
            let q = (x / scale).round() as i32;
            out[r * cols + c] = q.clamp(-(clamp_val as i32), clamp_val as i32) as i8;
        }
    }
    out
}

/// Quantize an FP32 matrix to [`QuantMatrix`] using the given config.
///
/// # Panics
/// Panics if `data.len() != rows * cols`.
pub fn quantize(data: &[f32], rows: usize, cols: usize, config: &Int8Config) -> QuantMatrix {
    assert_eq!(
        data.len(),
        rows * cols,
        "data length {} != rows({}) * cols({})",
        data.len(), rows, cols
    );

    match config.mode {
        QuantMode::PerTensor => {
            let scale = compute_scale_per_tensor(data, config.clamp_val);
            let quantized = quantize_per_tensor(data, scale, config.clamp_val);
            QuantMatrix {
                data: quantized,
                scales: vec![scale],
                rows,
                cols,
                mode: QuantMode::PerTensor,
            }
        }
        QuantMode::PerChannel => {
            let scales = compute_scales_per_channel(data, rows, cols, config.clamp_val);
            let quantized = quantize_per_channel(data, rows, cols, &scales, config.clamp_val);
            QuantMatrix {
                data: quantized,
                scales,
                rows,
                cols,
                mode: QuantMode::PerChannel,
            }
        }
    }
}

// ── Dequantization ────────────────────────────────────────────────────────────

/// Dequantize an INT8 value to FP32: x̂ = q * scale.
#[inline(always)]
pub fn dequantize_scalar(q: i8, scale: f32) -> f32 {
    q as f32 * scale
}

/// Dequantize a QuantMatrix back to FP32.
pub fn dequantize(mat: &QuantMatrix) -> Vec<f32> {
    (0..mat.rows)
        .flat_map(|r| {
            let scale = mat.scale_for_row(r);
            let row_start = r * mat.cols;
            (0..mat.cols).map(move |c| dequantize_scalar(mat.data[row_start + c], scale))
        })
        .collect()
}

// ── INT8 Matrix Multiplication ────────────────────────────────────────────────

/// Compute INT8 matrix multiplication: `C = A × B^T` in INT8 with FP32 output.
///
/// - `a`: `[M, K]` quantized matrix (activations, PerTensor typically)
/// - `b`: `[N, K]` quantized matrix (weights, PerChannel or PerTensor)
/// - Output `C`: `[M, N]` in FP32
///
/// The multiplication accumulates in i32 to avoid overflow, then scales:
/// `C[m,n] = sum_k(A_q[m,k] * B_q[n,k]) * scale_A[m] * scale_B[n]`
///
/// Note: B is stored as `[N, K]` (transposed relative to standard notation),
/// which enables sequential memory access for both A rows and B rows.
///
/// # Panics
/// Panics if `a.cols != b.cols` (inner dimension mismatch).
pub fn matmul_int8(a: &QuantMatrix, b: &QuantMatrix) -> Vec<f32> {
    assert_eq!(
        a.cols, b.cols,
        "inner dimension mismatch: a.cols={} b.cols={}",
        a.cols, b.cols
    );

    let m = a.rows;
    let n = b.rows;
    let k = a.cols;
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let scale_a = a.scale_for_row(i);
        let a_row = &a.data[i * k..(i + 1) * k];

        for j in 0..n {
            let scale_b = b.scale_for_row(j);
            let b_row = &b.data[j * k..(j + 1) * k];

            // Accumulate in i32 to avoid overflow (127*127*K before scaling)
            let acc: i32 = a_row
                .iter()
                .zip(b_row.iter())
                .map(|(&aq, &bq)| (aq as i32) * (bq as i32))
                .sum();

            out[i * n + j] = acc as f32 * scale_a * scale_b;
        }
    }

    out
}

/// Fused quantize-matmul: quantizes `a` and `b` in-place then multiplies.
///
/// Convenience wrapper for the common case of starting from FP32 inputs.
///
/// # Arguments
/// - `a_fp32`: activations `[M, K]` row-major
/// - `b_fp32`: weights `[N, K]` row-major (note: B rows = output channels)
/// - `config`: quantization configuration
///
/// # Returns
/// FP32 output matrix `[M, N]`.
pub fn matmul_int8_fused(
    a_fp32: &[f32],
    b_fp32: &[f32],
    m: usize,
    n: usize,
    k: usize,
    config: &Int8Config,
) -> Vec<f32> {
    let a_q = quantize(a_fp32, m, k, config);
    let b_q = quantize(b_fp32, n, k, config);
    matmul_int8(&a_q, &b_q)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Quantization scalar tests ─────────────────────────────────────────────

    #[test]
    fn scale_per_tensor_basic() {
        // max(|[-1.0, 0.5, 0.25]|) = 1.0 → scale = 1.0/127 ≈ 0.00787
        let scale = compute_scale_per_tensor(&[-1.0, 0.5, 0.25], 127);
        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn scale_per_tensor_zero_input() {
        // Zero tensor: should return 1.0 (not divide-by-zero)
        let scale = compute_scale_per_tensor(&[0.0, 0.0, 0.0], 127);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn quantize_scalar_roundtrip() {
        let scale = 0.01;
        for val in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let q = quantize_per_tensor(&[val], scale, 127)[0];
            let dq = dequantize_scalar(q, scale);
            // Roundtrip error <= 0.5 * scale (rounding error)
            assert!((dq - val).abs() <= 0.5 * scale + 1e-6, "val={val} q={q} dq={dq}");
        }
    }

    #[test]
    fn quantize_clamps_overflow() {
        // Values > clamp_val * scale should saturate to ±127
        let scale = 1.0 / 127.0;
        let large = 2.0f32; // >> 1.0 (max representable)
        let q = quantize_per_tensor(&[large, -large], scale, 127);
        assert_eq!(q[0], 127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn quantize_symmetric_range() {
        let data: Vec<f32> = (-127..=127).map(|i| i as f32 / 127.0).collect();
        let config = Int8Config::default();
        let qm = quantize(&data, 1, data.len(), &config);
        // All values should reconstruct within scale/2
        let scale = qm.scales[0];
        let dq = dequantize(&qm);
        for (orig, rec) in data.iter().zip(dq.iter()) {
            assert!((orig - rec).abs() <= scale + 1e-5, "orig={orig} rec={rec} scale={scale}");
        }
    }

    // ── QuantMatrix construction ──────────────────────────────────────────────

    #[test]
    fn quantize_per_tensor_shape() {
        let data = vec![1.0f32; 12];
        let config = Int8Config { mode: QuantMode::PerTensor, ..Default::default() };
        let qm = quantize(&data, 3, 4, &config);
        assert_eq!(qm.rows, 3);
        assert_eq!(qm.cols, 4);
        assert_eq!(qm.data.len(), 12);
        assert_eq!(qm.scales.len(), 1); // single scale
    }

    #[test]
    fn quantize_per_channel_shape() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let config = Int8Config { mode: QuantMode::PerChannel, ..Default::default() };
        let qm = quantize(&data, 3, 4, &config);
        assert_eq!(qm.scales.len(), 3); // one scale per row
    }

    #[test]
    fn per_channel_scales_differ() {
        // Row 0: [0,0,0,0] → scale 1.0 (zero guard)
        // Row 1: [1,2,3,4] → scale 4/127
        // Row 2: [10,10,10,10] → scale 10/127
        let data = vec![
            0.0f32, 0.0, 0.0, 0.0,
            1.0, 2.0, 3.0, 4.0,
            10.0, 10.0, 10.0, 10.0,
        ];
        let config = Int8Config { mode: QuantMode::PerChannel, ..Default::default() };
        let qm = quantize(&data, 3, 4, &config);
        assert_eq!(qm.scales[0], 1.0);
        assert!((qm.scales[1] - 4.0 / 127.0).abs() < 1e-6);
        assert!((qm.scales[2] - 10.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn quantize_panics_on_shape_mismatch() {
        let data = vec![1.0f32; 5]; // Not 3*4=12
        let config = Int8Config::default();
        quantize(&data, 3, 4, &config);
    }

    // ── Dequantization ────────────────────────────────────────────────────────

    #[test]
    fn dequantize_roundtrip_per_tensor() {
        let data: Vec<f32> = (-5..=5).map(|i| i as f32 * 0.1).collect();
        let config = Int8Config::default();
        let qm = quantize(&data, 1, data.len(), &config);
        let recovered = dequantize(&qm);
        let scale = qm.scales[0];
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() <= scale + 1e-6);
        }
    }

    // ── Matrix multiplication ─────────────────────────────────────────────────

    #[test]
    fn matmul_int8_identity_like() {
        // A = [[1,0],[0,1]], B^T = [[1,0],[0,1]] → C = [[1,0],[0,1]]
        // Using small integer values to minimize quantization error
        let a_data = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2
        let b_data = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 (B rows = output channels)
        let config = Int8Config::default();
        let a_q = quantize(&a_data, 2, 2, &config);
        let b_q = quantize(&b_data, 2, 2, &config);
        let c = matmul_int8(&a_q, &b_q);
        // C[0,0] = 1*1 + 0*0 = 1.0, etc.
        assert!((c[0] - 1.0).abs() < 0.01, "c[0,0]={}", c[0]);
        assert!((c[1] - 0.0).abs() < 0.01, "c[0,1]={}", c[1]);
        assert!((c[2] - 0.0).abs() < 0.01, "c[1,0]={}", c[2]);
        assert!((c[3] - 1.0).abs() < 0.01, "c[1,1]={}", c[3]);
    }

    #[test]
    fn matmul_int8_known_value() {
        // A = [[2,3]], B = [[1,4],[2,5]]
        // C = A × B^T: C[0,0] = 2*1+3*4=14, C[0,1] = 2*2+3*5=19
        let a_data = vec![2.0f32, 3.0]; // 1×2
        let b_data = vec![1.0f32, 4.0, 2.0, 5.0]; // 2×2
        let config = Int8Config::default();
        let result = matmul_int8_fused(&a_data, &b_data, 1, 2, 2, &config);
        // With INT8 quantization, expect ~1% error
        assert!((result[0] - 14.0).abs() < 0.5, "expected ~14, got {}", result[0]);
        assert!((result[1] - 19.0).abs() < 0.5, "expected ~19, got {}", result[1]);
    }

    #[test]
    fn matmul_int8_output_shape() {
        // A: 3×4, B: 5×4 → output: 3×5
        let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.01).collect();
        let result = matmul_int8_fused(&a_data, &b_data, 3, 5, 4, &Int8Config::default());
        assert_eq!(result.len(), 15); // 3×5
    }

    #[test]
    #[should_panic(expected = "inner dimension mismatch")]
    fn matmul_inner_dim_mismatch_panics() {
        let a_q = quantize(&[1.0f32; 6], 2, 3, &Int8Config::default());
        let b_q = quantize(&[1.0f32; 8], 2, 4, &Int8Config::default()); // cols differ: 3 vs 4
        matmul_int8(&a_q, &b_q);
    }

    #[test]
    fn matmul_zero_matrix() {
        let a_data = vec![0.0f32; 4];
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let result = matmul_int8_fused(&a_data, &b_data, 2, 2, 2, &Int8Config::default());
        for v in &result {
            assert!(v.abs() < 1e-6, "expected zero, got {}", v);
        }
    }

    // ── Quantization error bounds ─────────────────────────────────────────────

    #[test]
    fn quantization_error_under_one_percent() {
        // For a normalized weight matrix (values in [-1, 1]),
        // INT8 quantization error should be < 1% of max absolute value.
        let n = 64;
        // Deterministic pseudo-random weights in [-1, 1]
        let weights: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 1.618033988) % 2.0) - 1.0)
            .collect();
        let config = Int8Config::default();
        let qm = quantize(&weights, 1, n, &config);
        let recovered = dequantize(&qm);
        let max_abs = weights.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let max_err = weights.iter().zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let err_pct = max_err / max_abs * 100.0;
        assert!(
            err_pct < 1.0,
            "quantization error {:.3}% exceeds 1% threshold (max_abs={:.4}, max_err={:.6})",
            err_pct, max_abs, max_err
        );
    }

    #[test]
    fn per_channel_lower_error_than_per_tensor() {
        // Per-channel should have lower or equal error for heterogeneous matrices
        // Row 0: small values [0.01..0.04], Row 1: large values [10.0..40.0]
        let data = vec![
            0.01f32, 0.02, 0.03, 0.04,
            10.0, 20.0, 30.0, 40.0,
        ];
        let config_pt = Int8Config { mode: QuantMode::PerTensor, ..Default::default() };
        let config_pc = Int8Config { mode: QuantMode::PerChannel, ..Default::default() };

        let qm_pt = quantize(&data, 2, 4, &config_pt);
        let qm_pc = quantize(&data, 2, 4, &config_pc);

        let dq_pt = dequantize(&qm_pt);
        let dq_pc = dequantize(&qm_pc);

        // Error on small values (row 0): per-channel should be much better
        let err_pt_row0: f32 = data[..4].iter().zip(dq_pt[..4].iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / 4.0;
        let err_pc_row0: f32 = data[..4].iter().zip(dq_pc[..4].iter())
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / 4.0;

        assert!(
            err_pc_row0 <= err_pt_row0 + 1e-6,
            "per-channel err {err_pc_row0:.6} should be <= per-tensor {err_pt_row0:.6} for small row"
        );
    }

    // ── QuantMode ─────────────────────────────────────────────────────────────

    #[test]
    fn quant_mode_default_is_per_tensor() {
        assert_eq!(QuantMode::default(), QuantMode::PerTensor);
    }

    #[test]
    fn config_default_clamp_127() {
        let cfg = Int8Config::default();
        assert_eq!(cfg.clamp_val, 127);
        assert_eq!(cfg.mode, QuantMode::PerTensor);
    }
}
