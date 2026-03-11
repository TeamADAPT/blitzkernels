//! **blitz-bf16-matmul** — BF16 (bfloat16) matrix multiplication kernel.
//!
//! BF16 is the dominant mixed-precision dtype for LLM training and inference
//! on H100/A100 GPUs and Google TPUs:
//!
//! ```text
//! BF16 layout: [sign:1][exponent:8][mantissa:7]  (same exponent as FP32)
//! FP32 layout: [sign:1][exponent:8][mantissa:23]
//! ```
//!
//! ## Why BF16?
//! - **Same dynamic range as FP32** — identical 8-bit exponent, no overflow on LLM activations
//! - **2× memory reduction** vs FP32; better numerical stability than FP16
//! - **Trivial conversion**: BF16→FP32 = append 16 zero mantissa bits
//! - **LLaMA / Mistral / Gemma default training dtype**
//!
//! ## Implementation
//! BF16 is represented as `u16` bit-patterns (no native Rust type).
//! All arithmetic is performed in FP32 (convert → multiply → accumulate → round back).
//! This is identical to how hardware BF16 tensor cores work: store in BF16,
//! compute in FP32 accumulators.
//!
//! Pure Rust, no external deps, WASM-portable (wasm32-wasip2).

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ── BF16 scalar type ─────────────────────────────────────────────────────────

/// BF16 scalar stored as a `u16` bit-pattern.
///
/// The bit layout mirrors the upper 16 bits of a 32-bit IEEE float:
/// `[sign:1][exponent:8][mantissa:7]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// Convert FP32 to BF16 by truncating the lower 16 mantissa bits.
    ///
    /// Uses round-to-nearest-even (RNE) to minimize truncation error.
    #[inline(always)]
    pub fn from_f32(f: f32) -> Self {
        let bits = f.to_bits();
        // Extract rounding bit and sticky bits
        let round_bit = (bits >> 15) & 1;
        let sticky = bits & 0x7FFF;
        let truncated = bits >> 16;
        // RNE: round up if round_bit=1 and (sticky != 0 OR lsb of result = 1)
        let lsb = truncated & 1;
        let rounded = if round_bit == 1 && (sticky != 0 || lsb == 1) {
            truncated.wrapping_add(1)
        } else {
            truncated
        };
        Bf16(rounded as u16)
    }

    /// Convert BF16 to FP32 by zero-extending the mantissa.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// BF16 representation of zero.
    pub const ZERO: Self = Bf16(0x0000);

    /// BF16 representation of one.
    pub const ONE: Self = Bf16(0x3F80);

    /// BF16 representation of negative infinity (useful for masking).
    pub const NEG_INF: Self = Bf16(0xFF80);
}

impl From<f32> for Bf16 {
    fn from(f: f32) -> Self {
        Bf16::from_f32(f)
    }
}

impl From<Bf16> for f32 {
    fn from(b: Bf16) -> f32 {
        b.to_f32()
    }
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Convert a slice of FP32 values to BF16.
pub fn f32_to_bf16_vec(data: &[f32]) -> Vec<Bf16> {
    data.iter().map(|&x| Bf16::from_f32(x)).collect()
}

/// Convert a slice of BF16 values back to FP32.
pub fn bf16_to_f32_vec(data: &[Bf16]) -> Vec<f32> {
    data.iter().map(|x| x.to_f32()).collect()
}

// ── BF16 Matrix type ──────────────────────────────────────────────────────────

/// A matrix stored in BF16 precision, row-major.
#[derive(Debug, Clone)]
pub struct Bf16Matrix {
    /// BF16 elements, row-major. Length = `rows * cols`.
    pub data: Vec<Bf16>,
    pub rows: usize,
    pub cols: usize,
}

impl Bf16Matrix {
    /// Create a BF16 matrix from FP32 data.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    pub fn from_f32(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(), rows * cols,
            "data length {} != rows({}) * cols({})", data.len(), rows, cols
        );
        Self {
            data: f32_to_bf16_vec(data),
            rows,
            cols,
        }
    }

    /// Convert the BF16 matrix back to FP32.
    pub fn to_f32(&self) -> Vec<f32> {
        bf16_to_f32_vec(&self.data)
    }

    /// Access element at (row, col).
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> Bf16 {
        self.data[row * self.cols + col]
    }
}

// ── BF16 Matrix Multiplication ────────────────────────────────────────────────

/// Compute `C = A × B^T` with BF16 inputs, FP32 accumulation, BF16 output.
///
/// - `a`: `[M, K]` BF16 matrix (activations)
/// - `b`: `[N, K]` BF16 matrix (weights, stored transposed for cache efficiency)
/// - Output: `[M, N]` BF16 matrix
///
/// Each inner product accumulates in FP32 (matching H100 tensor core behavior),
/// then rounds back to BF16 for the output.
///
/// # Panics
/// Panics if `a.cols != b.cols`.
pub fn matmul_bf16(a: &Bf16Matrix, b: &Bf16Matrix) -> Bf16Matrix {
    assert_eq!(
        a.cols, b.cols,
        "inner dimension mismatch: a.cols={} b.cols={}",
        a.cols, b.cols
    );

    let m = a.rows;
    let n = b.rows;
    let k = a.cols;
    let mut out = vec![Bf16::ZERO; m * n];

    for i in 0..m {
        for j in 0..n {
            // FP32 accumulation (matches H100 tensor core semantics)
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a.get(i, l).to_f32() * b.get(j, l).to_f32();
            }
            out[i * n + j] = Bf16::from_f32(acc);
        }
    }

    Bf16Matrix { data: out, rows: m, cols: n }
}

/// Fused FP32→BF16→matmul→FP32 convenience wrapper.
///
/// Converts inputs to BF16, multiplies, returns FP32 output.
/// Models the standard mixed-precision inference pipeline.
pub fn matmul_bf16_fused(
    a_fp32: &[f32],
    b_fp32: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let a = Bf16Matrix::from_f32(a_fp32, m, k);
    let b = Bf16Matrix::from_f32(b_fp32, n, k);
    matmul_bf16(&a, &b).to_f32()
}

/// Measure BF16 quantization error vs FP32 reference.
///
/// Returns mean absolute error across all elements.
pub fn bf16_roundtrip_error(data: &[f32]) -> f32 {
    let recovered: Vec<f32> = data.iter()
        .map(|&x| Bf16::from_f32(x).to_f32())
        .collect();
    let total_err: f32 = data.iter().zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    total_err / data.len() as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Conversion ─────────────────────────────────────────────────────────────

    #[test]
    fn bf16_zero_roundtrip() {
        assert_eq!(Bf16::from_f32(0.0).to_f32(), 0.0);
    }

    #[test]
    fn bf16_one_roundtrip() {
        let b = Bf16::from_f32(1.0);
        assert_eq!(b, Bf16::ONE);
        assert_eq!(b.to_f32(), 1.0);
    }

    #[test]
    fn bf16_neg_one_roundtrip() {
        let b = Bf16::from_f32(-1.0);
        assert_eq!(b.to_f32(), -1.0);
    }

    #[test]
    fn bf16_small_positive() {
        // 0.5 is exactly representable in BF16 (exponent -1, mantissa 0)
        let b = Bf16::from_f32(0.5);
        assert_eq!(b.to_f32(), 0.5);
    }

    #[test]
    fn bf16_from_f32_trait() {
        let b: Bf16 = 2.0f32.into();
        let f: f32 = b.into();
        assert_eq!(f, 2.0);
    }

    #[test]
    fn bf16_large_value_no_overflow() {
        // BF16 has same exponent range as FP32 (max ~3.4e38)
        let large = 1e30f32;
        let b = Bf16::from_f32(large);
        let recovered = b.to_f32();
        // Allow ~1% relative error (7 mantissa bits)
        let rel_err = (recovered - large).abs() / large;
        assert!(rel_err < 0.01, "large value rel_err={rel_err:.4}");
    }

    #[test]
    fn bf16_neg_inf_constant() {
        let recovered = Bf16::NEG_INF.to_f32();
        assert!(recovered.is_infinite() && recovered < 0.0);
    }

    #[test]
    fn f32_to_bf16_vec_roundtrip() {
        let data = vec![1.0f32, -1.0, 0.5, 0.0, 100.0];
        let bf16 = f32_to_bf16_vec(&data);
        let back = bf16_to_f32_vec(&bf16);
        for (a, b) in data.iter().zip(back.iter()) {
            // Relative error < 1% for values in normal range
            if a.abs() > 1e-6 {
                assert!((a - b).abs() / a.abs() < 0.01, "a={a} b={b}");
            }
        }
    }

    // ── Bf16Matrix ────────────────────────────────────────────────────────────

    #[test]
    fn matrix_from_f32_shape() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let m = Bf16Matrix::from_f32(&data, 3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 12);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn matrix_shape_mismatch_panics() {
        Bf16Matrix::from_f32(&[1.0f32; 5], 3, 4);
    }

    #[test]
    fn matrix_get_element() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let m = Bf16Matrix::from_f32(&data, 2, 2);
        assert_eq!(m.get(0, 0).to_f32(), 1.0);
        assert_eq!(m.get(0, 1).to_f32(), 2.0);
        assert_eq!(m.get(1, 0).to_f32(), 3.0);
        assert_eq!(m.get(1, 1).to_f32(), 4.0);
    }

    // ── Matrix multiplication ─────────────────────────────────────────────────

    #[test]
    fn matmul_bf16_identity() {
        // A = I₂, B = I₂ → C = I₂
        let eye = vec![1.0f32, 0.0, 0.0, 1.0];
        let a = Bf16Matrix::from_f32(&eye, 2, 2);
        let b = Bf16Matrix::from_f32(&eye, 2, 2);
        let c = matmul_bf16(&a, &b).to_f32();
        assert!((c[0] - 1.0).abs() < 0.01);
        assert!((c[1] - 0.0).abs() < 0.01);
        assert!((c[2] - 0.0).abs() < 0.01);
        assert!((c[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn matmul_bf16_known_value() {
        // A = [[2, 3]], B = [[1, 4],[2, 5]]
        // C[0,0] = 2*1 + 3*4 = 14, C[0,1] = 2*2 + 3*5 = 19
        let a_data = vec![2.0f32, 3.0];
        let b_data = vec![1.0f32, 4.0, 2.0, 5.0];
        let result = matmul_bf16_fused(&a_data, &b_data, 1, 2, 2);
        assert!((result[0] - 14.0).abs() < 0.3, "expected ~14 got {}", result[0]);
        assert!((result[1] - 19.0).abs() < 0.3, "expected ~19 got {}", result[1]);
    }

    #[test]
    fn matmul_bf16_output_shape() {
        // A: 3×4, B: 5×4 → output 3×5
        let a: Vec<f32> = (0..12).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..20).map(|i| i as f32 * 0.01).collect();
        let out = matmul_bf16_fused(&a, &b, 3, 5, 4);
        assert_eq!(out.len(), 15);
    }

    #[test]
    fn matmul_zero_input() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = matmul_bf16_fused(&a, &b, 2, 2, 2);
        for v in &out {
            assert!(v.abs() < 0.001, "expected zero, got {v}");
        }
    }

    #[test]
    #[should_panic(expected = "inner dimension mismatch")]
    fn matmul_dim_mismatch_panics() {
        let a = Bf16Matrix::from_f32(&[1.0f32; 6], 2, 3);
        let b = Bf16Matrix::from_f32(&[1.0f32; 8], 2, 4);
        matmul_bf16(&a, &b);
    }

    // ── Error bounds ──────────────────────────────────────────────────────────

    #[test]
    fn bf16_roundtrip_error_under_one_percent() {
        // Normalized weights in [-1, 1]: BF16 roundtrip error < 1%
        let data: Vec<f32> = (0..64)
            .map(|i| ((i as f32 * 1.618033988) % 2.0) - 1.0)
            .collect();
        let err = bf16_roundtrip_error(&data);
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err_pct = err / max_abs * 100.0;
        assert!(
            rel_err_pct < 1.0,
            "BF16 roundtrip error {rel_err_pct:.3}% exceeds 1% threshold"
        );
    }

    #[test]
    fn matmul_bf16_vs_fp32_accuracy() {
        // BF16 matmul should be within 2% of FP32 reference for unit-scale inputs
        let a_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 8.0).collect();
        let b_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) / 8.0).collect();
        // BF16 result (compute first to avoid ownership issues)
        let bf16_out = matmul_bf16_fused(&a_data, &b_data, 4, 4, 4);
        // FP32 reference (borrow after bf16 call)
        let fp32_out: Vec<f32> = (0..4).flat_map(|i| {
            let a = &a_data;
            let b = &b_data;
            (0..4).map(move |j| {
                (0..4).map(|k| a[i * 4 + k] * b[j * 4 + k]).sum::<f32>()
            })
        }).collect();
        let max_ref = fp32_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_ref > 1e-6 {
            for (fp, bf) in fp32_out.iter().zip(bf16_out.iter()) {
                let rel_err = (fp - bf).abs() / max_ref;
                assert!(rel_err < 0.02, "fp32={fp:.4} bf16={bf:.4} rel_err={rel_err:.4}");
            }
        }
    }
}
