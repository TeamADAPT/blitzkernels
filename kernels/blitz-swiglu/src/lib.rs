//! **blitz-swiglu** — SwiGLU activation kernel for modern transformer FFN blocks.
//!
//! Implements the gating mechanism from PaLM / LLaMA / Mistral / Gemma:
//!
//! ```text
//! gate   = x · W_gate + b_gate          [B, F]
//! up     = x · W_up   + b_up            [B, F]
//! hidden = Swish(gate) ⊙ up             [B, F]  (element-wise)
//! output = hidden · W_down + b_down      [B, D]
//! ```
//!
//! Swish(x) = x · σ(βx), where σ is the sigmoid function.
//! When β=1, this is SiLU (Sigmoid Linear Unit).
//!
//! The fused kernel eliminates intermediate allocations.
//! Pure Rust, no external deps, WASM-portable.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;
pub use config::SwiGluConfig;

// ── Math helpers ─────────────────────────────────────────────────────────────

#[inline(always)]
fn exp_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    { x.exp() }
    #[cfg(not(feature = "std"))]
    { libm::expf(x) }
}

/// Sigmoid: σ(x) = 1 / (1 + exp(-x))
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-x))
}

/// Swish activation: x · σ(βx). When β=1, this is SiLU.
#[inline(always)]
fn swish(x: f32, beta: f32) -> f32 {
    x * sigmoid(beta * x)
}

// ── Core operations ──────────────────────────────────────────────────────────

/// Fused SwiGLU FFN: gate + up projections → Swish gating → down projection.
///
/// # Weights layout (row-major)
/// - `w_gate`:  `[d_model, d_ff]` — gate projection
/// - `b_gate`:  `[d_ff]` — gate bias
/// - `w_up`:    `[d_model, d_ff]` — up projection
/// - `b_up`:    `[d_ff]` — up bias
/// - `w_down`:  `[d_ff, d_model]` — down projection
/// - `b_down`:  `[d_model]` — down bias
/// - `input`:   `[batch, d_model]`
///
/// # Returns
/// Output tensor `[batch, d_model]`.
///
/// # Example
/// ```
/// use blitz_swiglu::{fused_swiglu, SwiGluConfig};
///
/// let cfg = SwiGluConfig::with_d_ff(4, 8);
/// let w_gate = vec![0.1_f32; 4 * 8];
/// let b_gate = vec![0.0_f32; 8];
/// let w_up   = vec![0.1_f32; 4 * 8];
/// let b_up   = vec![0.0_f32; 8];
/// let w_down = vec![0.1_f32; 8 * 4];
/// let b_down = vec![0.0_f32; 4];
/// let input  = vec![1.0_f32; 4];
///
/// let out = fused_swiglu(&input, &w_gate, &b_gate, &w_up, &b_up, &w_down, &b_down, &cfg);
/// assert_eq!(out.len(), 4);
/// ```
///
/// # Panics
/// Panics if dimensions don't match config.
pub fn fused_swiglu(
    input: &[f32],
    w_gate: &[f32],
    b_gate: &[f32],
    w_up: &[f32],
    b_up: &[f32],
    w_down: &[f32],
    b_down: &[f32],
    config: &SwiGluConfig,
) -> Vec<f32> {
    let d = config.d_model;
    let f = config.d_ff;
    let beta = config.swish_beta;

    assert_eq!(w_gate.len(), d * f, "w_gate must be [d_model, d_ff]");
    assert_eq!(b_gate.len(), f, "b_gate must have length d_ff");
    assert_eq!(w_up.len(), d * f, "w_up must be [d_model, d_ff]");
    assert_eq!(b_up.len(), f, "b_up must have length d_ff");
    assert_eq!(w_down.len(), f * d, "w_down must be [d_ff, d_model]");
    assert_eq!(b_down.len(), d, "b_down must have length d_model");
    assert_eq!(input.len() % d, 0, "input length must be a multiple of d_model");

    let batch = input.len() / d;
    let mut output = vec![0.0_f32; batch * d];

    // Scratch buffers for gate and up projections
    let mut gate = vec![0.0_f32; f];
    let mut up = vec![0.0_f32; f];

    for b in 0..batch {
        let in_row = &input[b * d..(b + 1) * d];

        // Step 1: Parallel gate + up projections
        for j in 0..f {
            let mut g_acc = b_gate[j];
            let mut u_acc = b_up[j];
            for i in 0..d {
                let x = in_row[i];
                g_acc += x * w_gate[i * f + j];
                u_acc += x * w_up[i * f + j];
            }
            gate[j] = g_acc;
            up[j] = u_acc;
        }

        // Step 2: Swish(gate) ⊙ up → hidden (reuse gate buffer)
        for j in 0..f {
            gate[j] = swish(gate[j], beta) * up[j];
        }

        // Step 3: Down projection
        let out_row = &mut output[b * d..(b + 1) * d];
        for i in 0..d {
            let mut acc = b_down[i];
            for j in 0..f {
                acc += gate[j] * w_down[j * d + i];
            }
            out_row[i] = acc;
        }
    }

    output
}

/// Compute only the SwiGLU activation (no down projection).
/// Returns `Swish(gate) ⊙ up` with shape `[batch, d_ff]`.
///
/// Useful for debugging, ablation, and intermediate inspection.
pub fn swiglu_activation(
    input: &[f32],
    w_gate: &[f32],
    b_gate: &[f32],
    w_up: &[f32],
    b_up: &[f32],
    config: &SwiGluConfig,
) -> Vec<f32> {
    let d = config.d_model;
    let f = config.d_ff;
    let beta = config.swish_beta;

    assert_eq!(w_gate.len(), d * f);
    assert_eq!(b_gate.len(), f);
    assert_eq!(w_up.len(), d * f);
    assert_eq!(b_up.len(), f);
    assert_eq!(input.len() % d, 0);

    let batch = input.len() / d;
    let mut output = vec![0.0_f32; batch * f];

    for b in 0..batch {
        let in_row = &input[b * d..(b + 1) * d];
        let out_row = &mut output[b * f..(b + 1) * f];

        for j in 0..f {
            let mut g_acc = b_gate[j];
            let mut u_acc = b_up[j];
            for i in 0..d {
                let x = in_row[i];
                g_acc += x * w_gate[i * f + j];
                u_acc += x * w_up[i * f + j];
            }
            out_row[j] = swish(g_acc, beta) * u_acc;
        }
    }

    output
}

/// Apply the Swish (SiLU) activation element-wise.
/// `swish_elementwise(x, beta)` = `x * sigmoid(beta * x)` for each element.
pub fn swish_elementwise(input: &[f32], beta: f32) -> Vec<f32> {
    input.iter().map(|&x| swish(x, beta)).collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    // ── Swish / sigmoid unit tests ───────────────────────────────────────────

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!(approx_eq(sigmoid(0.0), 0.5, 1e-7));
    }

    #[test]
    fn sigmoid_large_positive_is_one() {
        assert!(approx_eq(sigmoid(10.0), 1.0, 1e-4));
    }

    #[test]
    fn sigmoid_large_negative_is_zero() {
        assert!(approx_eq(sigmoid(-10.0), 0.0, 1e-4));
    }

    #[test]
    fn sigmoid_is_monotone() {
        let xs = [-5.0_f32, -1.0, 0.0, 1.0, 5.0];
        for w in xs.windows(2) {
            assert!(sigmoid(w[1]) > sigmoid(w[0]),
                "sigmoid({}) = {} should be > sigmoid({}) = {}",
                w[1], sigmoid(w[1]), w[0], sigmoid(w[0]));
        }
    }

    #[test]
    fn swish_at_zero_is_zero() {
        assert!(approx_eq(swish(0.0, 1.0), 0.0, 1e-7));
    }

    #[test]
    fn swish_positive_is_positive() {
        // For x > 0, swish(x) > 0
        for &x in &[0.1_f32, 1.0, 5.0, 10.0] {
            assert!(swish(x, 1.0) > 0.0, "swish({x}) should be > 0");
        }
    }

    #[test]
    fn swish_large_positive_approaches_identity() {
        // swish(x) ≈ x for large positive x (sigmoid → 1)
        let x = 10.0_f32;
        assert!(approx_eq(swish(x, 1.0), x, 0.01));
    }

    #[test]
    fn swish_large_negative_approaches_zero() {
        // swish(x) ≈ 0 for large negative x (sigmoid → 0)
        assert!(swish(-10.0, 1.0).abs() < 1e-3);
    }

    #[test]
    fn swish_beta_sharpens() {
        // Higher beta → sharper transition. At x=0.5, higher beta gives less activation.
        let s1 = swish(0.5, 1.0);
        let s10 = swish(0.5, 10.0);
        // Both positive, but with very high beta, sigmoid(beta*x) → 1 faster,
        // so swish ≈ x for positive x with high beta.
        assert!(s10 > s1, "higher beta should give higher swish for positive x");
    }

    // ── Output shape tests ───────────────────────────────────────────────────

    #[test]
    fn output_shape_single_batch() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let out = fused_swiglu(
            &vec![1.0; 4],
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4],
            &cfg,
        );
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn output_shape_multi_batch() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let out = fused_swiglu(
            &vec![1.0; 12], // batch=3
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4],
            &cfg,
        );
        assert_eq!(out.len(), 12);
    }

    // ── Fused = sequential test ──────────────────────────────────────────────

    #[test]
    fn fused_equals_sequential() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let w_gate: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.17).sin() * 0.3).collect();
        let b_gate = vec![0.01; 8];
        let w_up: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.23).cos() * 0.2).collect();
        let b_up = vec![-0.01; 8];
        let w_down: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.07).sin() * 0.15).collect();
        let b_down = vec![0.02; 4];
        let input = vec![1.0_f32, -0.5, 0.3, 2.0];

        // Sequential: activation then down-project
        let hidden = swiglu_activation(&input, &w_gate, &b_gate, &w_up, &b_up, &cfg);
        let mut expected = vec![0.0_f32; 4];
        for i in 0..4 {
            let mut acc = b_down[i];
            for j in 0..8 {
                acc += hidden[j] * w_down[j * 4 + i];
            }
            expected[i] = acc;
        }

        // Fused
        let fused = fused_swiglu(&input, &w_gate, &b_gate, &w_up, &b_up, &w_down, &b_down, &cfg);

        for i in 0..4 {
            assert!(approx_eq(fused[i], expected[i], 1e-5),
                "i={i}: fused={} expected={}", fused[i], expected[i]);
        }
    }

    // ── Zero input test ──────────────────────────────────────────────────────

    #[test]
    fn zero_input_zero_bias_gives_zero() {
        // x=0, bias=0 → gate=0, up=0 → swish(0)*0 = 0 → output = b_down
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let out = fused_swiglu(
            &vec![0.0; 4],
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4],
            &cfg,
        );
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    // ── Bias passthrough test ────────────────────────────────────────────────

    #[test]
    fn b_down_adds_to_output() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let w_g = vec![0.1; 32];
        let b_g = vec![0.0; 8];
        let w_u = vec![0.1; 32];
        let b_u = vec![0.0; 8];
        let w_d = vec![0.1; 32];
        let input = vec![1.0; 4];

        let out_no_bias = fused_swiglu(&input, &w_g, &b_g, &w_u, &b_u, &w_d, &vec![0.0; 4], &cfg);
        let out_with_bias = fused_swiglu(&input, &w_g, &b_g, &w_u, &b_u, &w_d, &vec![1.0; 4], &cfg);

        for i in 0..4 {
            assert!(approx_eq(out_with_bias[i] - out_no_bias[i], 1.0, 1e-5),
                "bias offset wrong at i={i}");
        }
    }

    // ── Batch independence ───────────────────────────────────────────────────

    #[test]
    fn batch_rows_independent() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let w_g = vec![0.1; 32];
        let b_g = vec![0.01; 8];
        let w_u = vec![0.1; 32];
        let b_u = vec![0.0; 8];
        let w_d = vec![0.1; 32];
        let b_d = vec![0.0; 4];
        let row = vec![1.0_f32, 2.0, 3.0, 4.0];

        let single = fused_swiglu(&row, &w_g, &b_g, &w_u, &b_u, &w_d, &b_d, &cfg);
        let doubled: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let batched = fused_swiglu(&doubled, &w_g, &b_g, &w_u, &b_u, &w_d, &b_d, &cfg);

        for i in 0..4 {
            assert!(approx_eq(single[i], batched[i], 1e-6), "row 0 mismatch at {i}");
            assert!(approx_eq(single[i], batched[4 + i], 1e-6), "row 1 mismatch at {i}");
        }
    }

    // ── Activation shape test ────────────────────────────────────────────────

    #[test]
    fn activation_output_shape() {
        let cfg = SwiGluConfig::with_d_ff(4, 16);
        let h = swiglu_activation(
            &vec![1.0; 8], // batch=2
            &vec![0.1; 64], &vec![0.0; 16],
            &vec![0.1; 64], &vec![0.0; 16],
            &cfg,
        );
        assert_eq!(h.len(), 32); // batch=2, d_ff=16
    }

    // ── Swish elementwise test ───────────────────────────────────────────────

    #[test]
    fn swish_elementwise_correct() {
        let input = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let out = swish_elementwise(&input, 1.0);
        assert_eq!(out.len(), 5);
        assert!(approx_eq(out[2], 0.0, 1e-7)); // swish(0) = 0
        assert!(out[3] > 0.0); // swish(1) > 0
        assert!(out[4] > out[3]); // monotone for positive
    }

    // ── Default d_ff (LLaMA-style) ───────────────────────────────────────────

    #[test]
    fn default_d_ff_llama_style() {
        // LLaMA: d_ff ≈ (8/3)*d_model, rounded to multiple of 8
        let cfg = SwiGluConfig::new(512);
        assert_eq!(cfg.d_ff % 8, 0, "d_ff should be multiple of 8");
        // 8*512/3 = 1365.33 → round up to 1368
        assert_eq!(cfg.d_ff, 1368);
        assert_eq!(cfg.swish_beta, 1.0);
    }

    // ── Panics on dimension mismatch ─────────────────────────────────────────

    #[test]
    #[should_panic]
    fn panics_w_gate_wrong_size() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        fused_swiglu(
            &vec![1.0; 4],
            &vec![0.1; 16], &vec![0.0; 8], // w_gate too small
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4],
            &cfg,
        );
    }

    #[test]
    #[should_panic]
    fn panics_input_not_multiple() {
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        fused_swiglu(
            &vec![1.0; 5], // not divisible by 4
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 8],
            &vec![0.1; 32], &vec![0.0; 4],
            &cfg,
        );
    }

    // ── Gate controls output magnitude ───────────────────────────────────────

    #[test]
    fn zero_gate_suppresses_output() {
        // If w_gate=0, b_gate=0 → gate=0 → swish(0)=0 → hidden=0 → output=b_down
        let cfg = SwiGluConfig::with_d_ff(4, 8);
        let out = fused_swiglu(
            &vec![5.0; 4],
            &vec![0.0; 32], &vec![0.0; 8], // zero gate weights
            &vec![0.5; 32], &vec![0.0; 8], // nonzero up weights
            &vec![0.5; 32], &vec![0.0; 4],
            &cfg,
        );
        for &v in &out {
            assert!(v.abs() < 1e-6, "with zero gate, output should be ~0, got {v}");
        }
    }
}
