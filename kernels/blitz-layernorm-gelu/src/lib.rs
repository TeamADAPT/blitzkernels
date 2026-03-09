// WASM-portable: drop std when built without the "std" feature.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;
pub use config::LayerNormGeluConfig;

// ── Math helpers (std vs libm) ────────────────────────────────────────────────

#[inline(always)]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.sqrt() }
    #[cfg(not(feature = "std"))]
    { libm::sqrtf(x) }
}

#[inline(always)]
fn tanh_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.tanh() }
    #[cfg(not(feature = "std"))]
    { libm::tanhf(x) }
}

/// Polynomial approximation of erf, accurate to ~1.5e-7 (Abramowitz & Stegun §7.1.26).
/// No external deps — compatible with both std and no_std targets.
#[inline(always)]
fn erf_f32(x: f32) -> f32 {
    let sign: f32 = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1_f32 * ax);
    let poly = t * (0.254_829_592_f32
        + t * (-0.284_496_736_f32
        + t * (1.421_413_741_f32
        + t * (-1.453_152_027_f32
        + t * 1.061_405_429_f32))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

// ── GELU implementations ──────────────────────────────────────────────────────

const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/π)
const GELU_COEFF: f32 = 0.044_715;
const SQRT_2_INV: f32 = 0.707_106_77;    // 1/sqrt(2)

/// Approximate GELU: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// Matches PyTorch `F.gelu(x, approximate='tanh')` and is standard in GPT-2 / BERT-large.
#[inline(always)]
fn gelu_approx(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    0.5 * x * (1.0 + tanh_f32(inner))
}

/// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
#[inline(always)]
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_f32(x * SQRT_2_INV))
}

// ── Fused kernel ──────────────────────────────────────────────────────────────

/// Apply **fused LayerNorm + GELU** to `input`.
///
/// # Layout
/// `input` is row-major with shape `[batch, d_model]` (i.e. `batch = input.len() / d_model`).
/// `gamma` and `beta` are learnable affine parameters of shape `[d_model]`.
///
/// ## Steps (fused per row)
/// 1. Compute row mean `μ` and variance `σ²`.
/// 2. Normalise: `x̂ = (x − μ) / sqrt(σ² + ε)`.
/// 3. Affine: `y = γ ⊙ x̂ + β`.
/// 4. GELU: `out = GELU(y)`.
///
/// # Panics
/// Panics if `gamma.len() != d_model`, `beta.len() != d_model`, or
/// `input.len()` is not a multiple of `d_model`.
///
/// # Example
/// ```
/// use blitz_layernorm_gelu::{fused_layer_norm_gelu, LayerNormGeluConfig};
///
/// let cfg = LayerNormGeluConfig::new(4);
/// let input = vec![1.0_f32, 2.0, 3.0, 4.0];   // batch=1, d_model=4
/// let gamma = vec![1.0_f32; 4];
/// let beta  = vec![0.0_f32; 4];
///
/// let out = fused_layer_norm_gelu(&input, &gamma, &beta, &cfg);
/// assert_eq!(out.len(), 4);
/// // After LN the values are mean-0; GELU(0) ≈ 0, so negative normalised
/// // values trend toward 0 and positive ones are preserved.
/// assert!(out[3] > out[0]);
/// ```
pub fn fused_layer_norm_gelu(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    config: &LayerNormGeluConfig,
) -> Vec<f32> {
    let d = config.d_model;
    assert_eq!(gamma.len(), d, "gamma must have length d_model");
    assert_eq!(beta.len(), d, "beta must have length d_model");
    assert_eq!(input.len() % d, 0, "input length must be a multiple of d_model");

    let batch = input.len() / d;
    let mut out = vec![0.0_f32; batch * d];
    let gelu_fn: fn(f32) -> f32 = if config.approximate_gelu { gelu_approx } else { gelu_exact };

    for b in 0..batch {
        let row = &input[b * d..(b + 1) * d];

        // Mean
        let mean: f32 = row.iter().sum::<f32>() / d as f32;

        // Variance (biased)
        let var: f32 = row.iter().map(|&x| {
            let diff = x - mean;
            diff * diff
        }).sum::<f32>() / d as f32;

        let inv_std = 1.0 / sqrt_f32(var + config.eps);

        for i in 0..d {
            let normed = (row[i] - mean) * inv_std;
            let affine = gamma[i] * normed + beta[i];
            out[b * d + i] = gelu_fn(affine);
        }
    }

    out
}

/// Apply LayerNorm only (no GELU), returning the normalised + affine output.
///
/// Useful for ablation tests and verifying the LN step independently.
pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    config: &LayerNormGeluConfig,
) -> Vec<f32> {
    let d = config.d_model;
    assert_eq!(gamma.len(), d);
    assert_eq!(beta.len(), d);
    assert_eq!(input.len() % d, 0);

    let batch = input.len() / d;
    let mut out = vec![0.0_f32; batch * d];

    for b in 0..batch {
        let row = &input[b * d..(b + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|&x| { let d = x - mean; d * d }).sum::<f32>() / d as f32;
        let inv_std = 1.0 / sqrt_f32(var + config.eps);
        for i in 0..d {
            out[b * d + i] = gamma[i] * (row[i] - mean) * inv_std + beta[i];
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(d: usize) -> LayerNormGeluConfig { LayerNormGeluConfig::new(d) }
    fn cfg_exact(d: usize) -> LayerNormGeluConfig { LayerNormGeluConfig::new(d).with_exact_gelu() }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    // ── LayerNorm tests ───────────────────────────────────────────────────────

    #[test]
    fn ln_zero_mean_unit_var() {
        // LN output should have mean≈0 and var≈1 (before affine, i.e. gamma=1,beta=0).
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];
        let out = layer_norm(&input, &gamma, &beta, &cfg(4));

        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-5), "mean={mean}");
        assert!(approx_eq(var, 1.0, 1e-3), "var={var}");
    }

    #[test]
    fn ln_constant_row_produces_zero() {
        // All-same input → variance=0 → normalised=0 → affine with beta=0 → 0.
        let input = vec![5.0_f32; 8];
        let gamma = vec![1.0_f32; 8];
        let beta = vec![0.0_f32; 8];
        let out = layer_norm(&input, &gamma, &beta, &cfg(8));
        for &v in &out {
            assert!(approx_eq(v, 0.0, 1e-5), "expected 0, got {v}");
        }
    }

    #[test]
    fn ln_affine_scale_shift() {
        // gamma=2, beta=1 should double the normalised value and shift by 1.
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let g1 = vec![1.0_f32; 4];
        let g2 = vec![2.0_f32; 4];
        let b0 = vec![0.0_f32; 4];
        let b1 = vec![1.0_f32; 4];

        let base = layer_norm(&input, &g1, &b0, &cfg(4));
        let scaled = layer_norm(&input, &g2, &b1, &cfg(4));

        for i in 0..4 {
            let expected = 2.0 * base[i] + 1.0;
            let si = scaled[i];
            assert!(approx_eq(si, expected, 1e-5), "i={i}: {si} != {expected}");
        }
    }

    #[test]
    fn ln_batch_independence() {
        // Two identical rows should produce identical outputs regardless of batch position.
        let row = vec![1.0_f32, 2.0, 3.0, 4.0];
        let input: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];
        let out = layer_norm(&input, &gamma, &beta, &cfg(4));
        for i in 0..4 {
            assert!(approx_eq(out[i], out[4 + i], 1e-6));
        }
    }

    // ── GELU tests ────────────────────────────────────────────────────────────

    #[test]
    fn gelu_approx_zero_input() {
        // GELU(0) = 0 exactly.
        assert!(approx_eq(gelu_approx(0.0), 0.0, 1e-7));
    }

    #[test]
    fn gelu_approx_large_positive() {
        // GELU(x) ≈ x for large positive x (passes through).
        let x = 10.0_f32;
        assert!(approx_eq(gelu_approx(x), x, 1e-3));
    }

    #[test]
    fn gelu_approx_large_negative() {
        // GELU(x) ≈ 0 for large negative x (suppressed).
        let x = -10.0_f32;
        assert!(approx_eq(gelu_approx(x), 0.0, 1e-3));
    }

    #[test]
    fn gelu_approx_monotone_positive() {
        // Monotone on positive reals.
        let xs = [0.1_f32, 0.5, 1.0, 2.0, 5.0];
        for w in xs.windows(2) {
            assert!(gelu_approx(w[1]) > gelu_approx(w[0]));
        }
    }

    #[test]
    fn gelu_exact_zero_input() {
        assert!(approx_eq(gelu_exact(0.0), 0.0, 1e-7));
    }

    #[test]
    fn gelu_approx_vs_exact_close() {
        // Approximate and exact GELU agree to within 0.5% for x in [-3, 3].
        let xs: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
        for x in xs {
            let diff = (gelu_approx(x) - gelu_exact(x)).abs();
            assert!(diff < 0.005, "x={x}: approx={:.6} exact={:.6} diff={diff:.6}",
                gelu_approx(x), gelu_exact(x));
        }
    }

    // ── Fused kernel tests ────────────────────────────────────────────────────

    #[test]
    fn fused_output_length() {
        let input = vec![0.0_f32; 16]; // batch=4, d=4
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];
        let out = fused_layer_norm_gelu(&input, &gamma, &beta, &cfg(4));
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn fused_equals_sequential() {
        // fused(input) should equal GELU(LN(input)) computed sequentially.
        // batch=1, d_model=8 so gamma/beta/input all have length 8.
        let input = vec![0.5_f32, 1.5, -0.5, 2.0, -1.0, 0.0, 1.0, -2.0];
        let gamma = vec![1.0_f32, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3];
        let beta = vec![0.1_f32, -0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.1];

        let ln_out = layer_norm(&input, &gamma, &beta, &cfg_exact(8));
        let expected: Vec<f32> = ln_out.iter().map(|&x| gelu_exact(x)).collect();
        let fused = fused_layer_norm_gelu(&input, &gamma, &beta, &cfg_exact(8));

        for i in 0..8 {
            assert!(approx_eq(fused[i], expected[i], 1e-5),
                "i={i}: fused={} expected={}", fused[i], expected[i]);
        }
    }

    #[test]
    fn fused_approx_output_in_range() {
        // All outputs should be in (-inf, input_max] since GELU <= identity.
        let input: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];
        let out = fused_layer_norm_gelu(&input, &gamma, &beta, &cfg(4));
        let input_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &v in &out {
            assert!(v <= input_max + 1e-3, "output {v} exceeds input max {input_max}");
        }
    }

    #[test]
    fn fused_negative_bias_suppressed() {
        // With strong negative beta, output should be near 0 (GELU suppresses negatives).
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0_f32; 4];
        let beta = vec![-100.0_f32; 4]; // push everything very negative
        let out = fused_layer_norm_gelu(&input, &gamma, &beta, &cfg(4));
        for &v in &out {
            assert!(v.abs() < 1e-3, "expected ~0 but got {v}");
        }
    }

    #[test]
    #[should_panic]
    fn panics_on_gamma_len_mismatch() {
        let input = vec![1.0_f32; 4];
        let gamma = vec![1.0_f32; 3]; // wrong length
        let beta = vec![0.0_f32; 4];
        fused_layer_norm_gelu(&input, &gamma, &beta, &cfg(4));
    }

    #[test]
    #[should_panic]
    fn panics_on_input_len_not_multiple() {
        let input = vec![1.0_f32; 5]; // 5 is not divisible by 4
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];
        fused_layer_norm_gelu(&input, &gamma, &beta, &cfg(4));
    }
}
