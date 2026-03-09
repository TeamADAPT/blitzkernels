//! **blitz-fused-mlp** — Fused MLP kernel for transformer FFN blocks.
//!
//! Fuses the entire feed-forward network into one pass:
//!
//! ```text
//! input ──▶ Linear(W_up) ──▶ LayerNorm ──▶ GELU ──▶ Linear(W_down) ──▶ output
//! [B, D]     [B, F]           [B, F]       [B, F]      [B, D]
//! ```
//!
//! Eliminates 3 intermediate allocations compared to calling each op separately.
//! Pure Rust, no external deps, WASM-portable.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;
pub use config::FusedMlpConfig;

// ── Math helpers (std vs libm) ───────────────────────────────────────────────

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

/// Polynomial approximation of erf (Abramowitz & Stegun §7.1.26).
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

// ── GELU implementations ─────────────────────────────────────────────────────

const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/π)
const GELU_COEFF: f32 = 0.044_715;
const SQRT_2_INV: f32 = 0.707_106_77;    // 1/sqrt(2)

#[inline(always)]
fn gelu_approx(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    0.5 * x * (1.0 + tanh_f32(inner))
}

#[inline(always)]
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_f32(x * SQRT_2_INV))
}

// ── Core operations ──────────────────────────────────────────────────────────

/// Fused MLP: Linear(W_up) → LayerNorm → GELU → Linear(W_down).
///
/// # Weights layout (row-major)
/// - `w_up`:   `[d_model, d_ff]` — up-projection weight matrix
/// - `b_up`:   `[d_ff]` — up-projection bias
/// - `gamma`:  `[d_ff]` — LayerNorm scale
/// - `beta`:   `[d_ff]` — LayerNorm shift
/// - `w_down`: `[d_ff, d_model]` — down-projection weight matrix
/// - `b_down`: `[d_model]` — down-projection bias
/// - `input`:  `[batch, d_model]` (batch = input.len() / d_model)
///
/// # Returns
/// Output tensor `[batch, d_model]`.
///
/// # Example
/// ```
/// use blitz_fused_mlp::{fused_mlp, FusedMlpConfig};
///
/// let cfg = FusedMlpConfig::with_d_ff(4, 8);
/// let w_up   = vec![0.1_f32; 4 * 8];   // [d_model=4, d_ff=8]
/// let b_up   = vec![0.0_f32; 8];
/// let gamma  = vec![1.0_f32; 8];
/// let beta   = vec![0.0_f32; 8];
/// let w_down = vec![0.1_f32; 8 * 4];   // [d_ff=8, d_model=4]
/// let b_down = vec![0.0_f32; 4];
/// let input  = vec![1.0_f32; 4];        // batch=1
///
/// let out = fused_mlp(&input, &w_up, &b_up, &gamma, &beta, &w_down, &b_down, &cfg);
/// assert_eq!(out.len(), 4);
/// ```
///
/// # Panics
/// Panics if weight/bias dimensions don't match `d_model` and `d_ff`.
pub fn fused_mlp(
    input: &[f32],
    w_up: &[f32],
    b_up: &[f32],
    gamma: &[f32],
    beta: &[f32],
    w_down: &[f32],
    b_down: &[f32],
    config: &FusedMlpConfig,
) -> Vec<f32> {
    let d = config.d_model;
    let f = config.d_ff;

    // Validate dimensions
    assert_eq!(w_up.len(), d * f, "w_up must be [d_model, d_ff]");
    assert_eq!(b_up.len(), f, "b_up must have length d_ff");
    assert_eq!(gamma.len(), f, "gamma must have length d_ff");
    assert_eq!(beta.len(), f, "beta must have length d_ff");
    assert_eq!(w_down.len(), f * d, "w_down must be [d_ff, d_model]");
    assert_eq!(b_down.len(), d, "b_down must have length d_model");
    assert_eq!(input.len() % d, 0, "input length must be a multiple of d_model");

    let batch = input.len() / d;
    let gelu_fn: fn(f32) -> f32 = if config.approximate_gelu { gelu_approx } else { gelu_exact };

    let mut output = vec![0.0_f32; batch * d];

    // Per-row scratch buffer for the intermediate hidden state [d_ff]
    let mut hidden = vec![0.0_f32; f];

    for b in 0..batch {
        let in_row = &input[b * d..(b + 1) * d];

        // Step 1: Linear up-projection — hidden[j] = input @ W_up[j] + b_up[j]
        for j in 0..f {
            let mut acc = b_up[j];
            for i in 0..d {
                acc += in_row[i] * w_up[i * f + j];
            }
            hidden[j] = acc;
        }

        // Step 2: LayerNorm over the hidden dimension
        let mean: f32 = hidden.iter().sum::<f32>() / f as f32;
        let var: f32 = hidden.iter().map(|&x| {
            let diff = x - mean;
            diff * diff
        }).sum::<f32>() / f as f32;
        let inv_std = 1.0 / sqrt_f32(var + config.eps);

        // Step 3: Normalise + affine + GELU (in-place on hidden)
        for j in 0..f {
            let normed = (hidden[j] - mean) * inv_std;
            let affine = gamma[j] * normed + beta[j];
            hidden[j] = gelu_fn(affine);
        }

        // Step 4: Linear down-projection — output[i] = hidden @ W_down[i] + b_down[i]
        let out_row = &mut output[b * d..(b + 1) * d];
        for i in 0..d {
            let mut acc = b_down[i];
            for j in 0..f {
                acc += hidden[j] * w_down[j * d + i];
            }
            out_row[i] = acc;
        }
    }

    output
}

/// Run only the up-projection + LayerNorm + GELU portion (no down-projection).
/// Returns the intermediate hidden state `[batch, d_ff]`.
///
/// Useful for debugging, ablation, and verifying individual stages.
pub fn up_ln_gelu(
    input: &[f32],
    w_up: &[f32],
    b_up: &[f32],
    gamma: &[f32],
    beta: &[f32],
    config: &FusedMlpConfig,
) -> Vec<f32> {
    let d = config.d_model;
    let f = config.d_ff;

    assert_eq!(w_up.len(), d * f);
    assert_eq!(b_up.len(), f);
    assert_eq!(gamma.len(), f);
    assert_eq!(beta.len(), f);
    assert_eq!(input.len() % d, 0);

    let batch = input.len() / d;
    let gelu_fn: fn(f32) -> f32 = if config.approximate_gelu { gelu_approx } else { gelu_exact };

    let mut output = vec![0.0_f32; batch * f];

    for b in 0..batch {
        let in_row = &input[b * d..(b + 1) * d];
        let out_row = &mut output[b * f..(b + 1) * f];

        // Linear up
        for j in 0..f {
            let mut acc = b_up[j];
            for i in 0..d {
                acc += in_row[i] * w_up[i * f + j];
            }
            out_row[j] = acc;
        }

        // LayerNorm
        let mean: f32 = out_row.iter().sum::<f32>() / f as f32;
        let var: f32 = out_row.iter().map(|&x| {
            let diff = x - mean;
            diff * diff
        }).sum::<f32>() / f as f32;
        let inv_std = 1.0 / sqrt_f32(var + config.eps);

        // Normalise + affine + GELU
        for j in 0..f {
            let normed = (out_row[j] - mean) * inv_std;
            let affine = gamma[j] * normed + beta[j];
            out_row[j] = gelu_fn(affine);
        }
    }

    output
}

/// Run only the down-projection: output = hidden @ W_down + b_down.
/// `hidden` is `[batch, d_ff]`, output is `[batch, d_model]`.
pub fn down_project(
    hidden: &[f32],
    w_down: &[f32],
    b_down: &[f32],
    config: &FusedMlpConfig,
) -> Vec<f32> {
    let d = config.d_model;
    let f = config.d_ff;

    assert_eq!(w_down.len(), f * d);
    assert_eq!(b_down.len(), d);
    assert_eq!(hidden.len() % f, 0);

    let batch = hidden.len() / f;
    let mut output = vec![0.0_f32; batch * d];

    for b in 0..batch {
        let h_row = &hidden[b * f..(b + 1) * f];
        let out_row = &mut output[b * d..(b + 1) * d];
        for i in 0..d {
            let mut acc = b_down[i];
            for j in 0..f {
                acc += h_row[j] * w_down[j * d + i];
            }
            out_row[i] = acc;
        }
    }

    output
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    fn identity_weights(d: usize) -> Vec<f32> {
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        w
    }

    // ── Output shape tests ───────────────────────────────────────────────────

    #[test]
    fn output_shape_single_batch() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        let out = fused_mlp(
            &vec![1.0; 4], &vec![0.1; 32], &vec![0.0; 8],
            &vec![1.0; 8], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4], &cfg,
        );
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn output_shape_multi_batch() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        let out = fused_mlp(
            &vec![1.0; 12], &vec![0.1; 32], &vec![0.0; 8],
            &vec![1.0; 8], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4], &cfg,
        );
        assert_eq!(out.len(), 12); // batch=3, d_model=4
    }

    // ── Fused = sequential test ──────────────────────────────────────────────

    #[test]
    fn fused_equals_sequential() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8).with_exact_gelu();
        let w_up = vec![0.1_f32, -0.2, 0.3, 0.05, -0.1, 0.2, -0.15, 0.25,
                        0.15, 0.1, -0.3, 0.2, 0.05, -0.2, 0.1, 0.3,
                        -0.1, 0.25, 0.15, -0.05, 0.2, 0.1, -0.2, 0.05,
                        0.3, -0.15, 0.1, 0.2, -0.25, 0.05, 0.15, -0.1];
        let b_up = vec![0.01; 8];
        let gamma = vec![1.0, 0.9, 1.1, 0.95, 1.05, 1.0, 0.85, 1.15];
        let beta = vec![0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1];
        let w_down: Vec<f32> = (0..32).map(|i| ((i as f32 * 0.07) - 0.5).sin() * 0.2).collect();
        let b_down = vec![0.01; 4];
        let input = vec![1.0_f32, -0.5, 0.3, 2.0];

        // Sequential
        let h = up_ln_gelu(&input, &w_up, &b_up, &gamma, &beta, &cfg);
        let expected = down_project(&h, &w_down, &b_down, &cfg);

        // Fused
        let fused = fused_mlp(&input, &w_up, &b_up, &gamma, &beta, &w_down, &b_down, &cfg);

        for i in 0..4 {
            assert!(approx_eq(fused[i], expected[i], 1e-5),
                "i={i}: fused={} expected={}", fused[i], expected[i]);
        }
    }

    // ── Zero input tests ─────────────────────────────────────────────────────

    #[test]
    fn zero_input_with_zero_bias_gives_near_zero() {
        // Zero input + zero bias → up-proj = 0 → LN(constant) → 0 → GELU(beta) → down-proj
        // With beta=0, GELU(0)=0, so output = b_down only.
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        let out = fused_mlp(
            &vec![0.0; 4], &vec![0.1; 32], &vec![0.0; 8],
            &vec![1.0; 8], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4], &cfg,
        );
        for &v in &out {
            assert!(v.abs() < 1e-3, "expected ~0, got {v}");
        }
    }

    // ── Bias pass-through test ───────────────────────────────────────────────

    #[test]
    fn b_down_adds_to_output() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        let w_up = vec![0.1; 32];
        let b_up = vec![0.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let w_down = vec![0.1; 32];
        let input = vec![1.0; 4];

        let out_no_bias = fused_mlp(&input, &w_up, &b_up, &gamma, &beta, &w_down, &vec![0.0; 4], &cfg);
        let out_with_bias = fused_mlp(&input, &w_up, &b_up, &gamma, &beta, &w_down, &vec![1.0; 4], &cfg);

        for i in 0..4 {
            assert!(approx_eq(out_with_bias[i] - out_no_bias[i], 1.0, 1e-5),
                "bias offset wrong at i={i}");
        }
    }

    // ── Batch independence test ──────────────────────────────────────────────

    #[test]
    fn batch_rows_independent() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        let w_up = vec![0.1; 32];
        let b_up = vec![0.01; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let w_down = vec![0.1; 32];
        let b_down = vec![0.0; 4];

        let row = vec![1.0_f32, 2.0, 3.0, 4.0];
        let single = fused_mlp(&row, &w_up, &b_up, &gamma, &beta, &w_down, &b_down, &cfg);

        let doubled: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let batched = fused_mlp(&doubled, &w_up, &b_up, &gamma, &beta, &w_down, &b_down, &cfg);

        for i in 0..4 {
            assert!(approx_eq(single[i], batched[i], 1e-6), "row 0 mismatch at {i}");
            assert!(approx_eq(single[i], batched[4 + i], 1e-6), "row 1 mismatch at {i}");
        }
    }

    // ── up_ln_gelu shape test ────────────────────────────────────────────────

    #[test]
    fn up_ln_gelu_output_shape() {
        let cfg = FusedMlpConfig::with_d_ff(4, 16);
        let h = up_ln_gelu(
            &vec![1.0; 8], &vec![0.1; 64], &vec![0.0; 16],
            &vec![1.0; 16], &vec![0.0; 16], &cfg,
        );
        assert_eq!(h.len(), 32); // batch=2, d_ff=16
    }

    // ── down_project shape test ──────────────────────────────────────────────

    #[test]
    fn down_project_output_shape() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        let out = down_project(&vec![1.0; 16], &vec![0.1; 32], &vec![0.0; 4], &cfg);
        assert_eq!(out.len(), 8); // batch=2, d_model=4
    }

    // ── Identity up-projection test ──────────────────────────────────────────

    #[test]
    fn identity_up_proj_preserves_structure() {
        // If d_model == d_ff and W_up = I, b_up = 0 → up-proj = input.
        // Then LN normalises, GELU activates.
        let d = 4;
        let cfg = FusedMlpConfig::with_d_ff(d, d);
        let w_up = identity_weights(d);
        let b_up = vec![0.0; d];
        let gamma = vec![1.0; d];
        let beta = vec![0.0; d];

        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let h = up_ln_gelu(&input, &w_up, &b_up, &gamma, &beta, &cfg);

        // After LN: mean-centred. Positive normalised values > 0 after GELU.
        // Last element (4.0) has largest normalised value → largest GELU output.
        assert!(h[3] > h[0], "largest input should produce largest activation");
        assert_eq!(h.len(), 4);
    }

    // ── GELU activation property ─────────────────────────────────────────────

    #[test]
    fn gelu_suppresses_negatives() {
        assert!(gelu_approx(-5.0).abs() < 1e-3);
        assert!(gelu_exact(-5.0).abs() < 1e-3);
    }

    #[test]
    fn gelu_passes_positives() {
        assert!((gelu_approx(5.0) - 5.0).abs() < 0.01);
        assert!((gelu_exact(5.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn gelu_approx_vs_exact_agreement() {
        let xs: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.1).collect();
        for x in xs {
            let diff = (gelu_approx(x) - gelu_exact(x)).abs();
            assert!(diff < 0.005, "x={x}: diff={diff}");
        }
    }

    // ── LayerNorm within fused path ──────────────────────────────────────────

    #[test]
    fn ln_normalises_hidden() {
        // After up-proj with uniform weights, all hidden values same → LN → 0 → GELU(beta)
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        // Uniform W_up → all hidden values will be equal → variance=0 → LN output = beta
        let w_up = vec![0.25; 32]; // each hidden unit sums all inputs equally
        let b_up = vec![0.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let h = up_ln_gelu(&input, &w_up, &b_up, &gamma, &beta, &cfg);

        // All hidden values should be GELU(beta=0) = 0
        for j in 0..8 {
            assert!(h[j].abs() < 1e-3, "j={j}: expected ~0, got {}", h[j]);
        }
    }

    // ── Panics on dimension mismatch ─────────────────────────────────────────

    #[test]
    #[should_panic]
    fn panics_w_up_wrong_size() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        fused_mlp(
            &vec![1.0; 4], &vec![0.1; 16], &vec![0.0; 8], // w_up too small
            &vec![1.0; 8], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4], &cfg,
        );
    }

    #[test]
    #[should_panic]
    fn panics_input_not_multiple() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        fused_mlp(
            &vec![1.0; 5], &vec![0.1; 32], &vec![0.0; 8], // 5 not divisible by 4
            &vec![1.0; 8], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4], &cfg,
        );
    }

    #[test]
    #[should_panic]
    fn panics_gamma_wrong_size() {
        let cfg = FusedMlpConfig::with_d_ff(4, 8);
        fused_mlp(
            &vec![1.0; 4], &vec![0.1; 32], &vec![0.0; 8],
            &vec![1.0; 4], &vec![0.0; 8], // gamma=4, should be 8
            &vec![0.1; 32], &vec![0.0; 4], &cfg,
        );
    }

    // ── Default d_ff = 4x test ───────────────────────────────────────────────

    #[test]
    fn default_d_ff_is_4x() {
        let cfg = FusedMlpConfig::new(64);
        assert_eq!(cfg.d_ff, 256);
        assert_eq!(cfg.d_model, 64);
    }

    // ── Approximate vs exact GELU in full pipeline ───────────────────────────

    #[test]
    fn approx_and_exact_gelu_close_in_fused() {
        let cfg_approx = FusedMlpConfig::with_d_ff(4, 8);
        let cfg_exact = FusedMlpConfig::with_d_ff(4, 8).with_exact_gelu();
        let w_up: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
        let b_up = vec![0.01; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let w_down: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
        let b_down = vec![0.0; 4];
        let input = vec![1.0, -0.5, 0.3, 2.0];

        let out_approx = fused_mlp(&input, &w_up, &b_up, &gamma, &beta, &w_down, &b_down, &cfg_approx);
        let out_exact = fused_mlp(&input, &w_up, &b_up, &gamma, &beta, &w_down, &b_down, &cfg_exact);

        for i in 0..4 {
            let diff = (out_approx[i] - out_exact[i]).abs();
            assert!(diff < 0.01, "i={i}: approx={} exact={} diff={diff}",
                out_approx[i], out_exact[i]);
        }
    }
}
