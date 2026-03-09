// WASM-portable when built without the "std" feature.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ── Math helpers ──────────────────────────────────────────────────────────────

#[inline(always)]
fn sin_cos_f32(x: f32) -> (f32, f32) {
    #[cfg(feature = "std")]
    { (x.sin(), x.cos()) }
    #[cfg(not(feature = "std"))]
    { (libm::sinf(x), libm::cosf(x)) }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for RoPE.
#[derive(Clone, Debug)]
pub struct RopeConfig {
    /// Head dimension (must be even; typically 64 or 128).
    pub head_dim: usize,
    /// RoPE base frequency (default 10_000.0; LLaMA 3 uses 500_000.0).
    pub base: f32,
    /// Maximum sequence length to precompute frequencies for.
    pub max_seq_len: usize,
}

impl RopeConfig {
    /// Standard config matching LLaMA 2 / Mistral (base=10_000).
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even");
        Self { head_dim, base: 10_000.0, max_seq_len }
    }

    /// Config with custom base (e.g. LLaMA 3 uses 500_000.0).
    pub fn with_base(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even");
        Self { head_dim, base, max_seq_len }
    }
}

// ── Frequency precomputation ──────────────────────────────────────────────────

/// Precomputed (cos, sin) table for RoPE application.
///
/// Shape: `[max_seq_len, head_dim/2]` for each of cos and sin.
pub struct RopeFreqs {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub seq_len: usize,
    pub half_dim: usize,
}

impl RopeFreqs {
    /// Precompute RoPE frequencies for `[0..max_seq_len]` positions.
    ///
    /// # Example
    /// ```
    /// use blitz_rope::{RopeConfig, RopeFreqs};
    /// let cfg = RopeConfig::new(64, 512);
    /// let freqs = RopeFreqs::precompute(&cfg);
    /// assert_eq!(freqs.cos.len(), 512 * 32); // seq_len * half_dim
    /// ```
    pub fn precompute(config: &RopeConfig) -> Self {
        let half = config.head_dim / 2;
        let seq = config.max_seq_len;
        let mut cos_table = vec![0.0_f32; seq * half];
        let mut sin_table = vec![0.0_f32; seq * half];

        for pos in 0..seq {
            for i in 0..half {
                // θ_i = pos / base^(2i / head_dim)
                let theta = pos as f32
                    / config.base.powf(2.0 * i as f32 / config.head_dim as f32);
                let (s, c) = sin_cos_f32(theta);
                cos_table[pos * half + i] = c;
                sin_table[pos * half + i] = s;
            }
        }

        Self { cos: cos_table, sin: sin_table, seq_len: seq, half_dim: half }
    }
}

// ── Core rotation ─────────────────────────────────────────────────────────────

/// Apply RoPE to query or key tensors in-place.
///
/// Rotates each head's embedding using precomputed (cos, sin) frequencies.
///
/// # Layout
/// `tensor` is row-major `[batch, num_heads, seq_len, head_dim]`.
/// `positions` maps each sequence position to its absolute index into `freqs`
/// (allows packed/KV-cache offset addressing).
///
/// # Panics
/// Panics if `tensor.len()` is not divisible by `head_dim`, if any position
/// in `positions` is out of bounds for `freqs`, or if `positions.len() != seq_len`.
///
/// # Example
/// ```
/// use blitz_rope::{RopeConfig, RopeFreqs, apply_rope};
///
/// let cfg = RopeConfig::new(4, 8);
/// let freqs = RopeFreqs::precompute(&cfg);
///
/// // batch=1, num_heads=2, seq_len=3, head_dim=4
/// let mut q = vec![1.0_f32; 1 * 2 * 3 * 4];
/// let positions: Vec<usize> = (0..3).collect();
/// apply_rope(&mut q, &freqs, &positions, 2, &cfg);
/// assert_eq!(q.len(), 24);
/// ```
pub fn apply_rope(
    tensor: &mut [f32],
    freqs: &RopeFreqs,
    positions: &[usize],
    _num_heads: usize,
    config: &RopeConfig,
) {
    let d = config.head_dim;
    let half = d / 2;
    let seq_len = positions.len();

    assert_eq!(
        tensor.len() % d, 0,
        "tensor length must be divisible by head_dim"
    );

    // total tokens = batch * num_heads * seq_len
    let total_tokens = tensor.len() / d;
    // num of (batch * num_heads) groups
    let bh = total_tokens / seq_len;
    assert_eq!(bh * seq_len, total_tokens, "tensor length inconsistent with seq_len");

    for b in 0..bh {
        for s in 0..seq_len {
            let pos = positions[s];
            assert!(pos < freqs.seq_len, "position {pos} out of freqs range");

            let base = (b * seq_len + s) * d;
            for i in 0..half {
                let x0 = tensor[base + i];
                let x1 = tensor[base + half + i];
                let c = freqs.cos[pos * freqs.half_dim + i];
                let s_val = freqs.sin[pos * freqs.half_dim + i];
                // RoPE rotation: [x0, x1] -> [x0*cos - x1*sin, x1*cos + x0*sin]
                tensor[base + i] = x0 * c - x1 * s_val;
                tensor[base + half + i] = x1 * c + x0 * s_val;
            }
        }
    }
}

/// Apply RoPE and return a new Vec (non-mutating variant).
pub fn rope(
    tensor: &[f32],
    freqs: &RopeFreqs,
    positions: &[usize],
    _num_heads: usize,
    config: &RopeConfig,
) -> Vec<f32> {
    let mut out = tensor.to_vec();
    apply_rope(&mut out, freqs, positions, _num_heads, config);
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    fn cfg(d: usize) -> RopeConfig { RopeConfig::new(d, 64) }

    // ── Frequency precomputation ──────────────────────────────────────────────

    #[test]
    fn freqs_table_size() {
        let cfg = RopeConfig::new(64, 512);
        let f = RopeFreqs::precompute(&cfg);
        assert_eq!(f.cos.len(), 512 * 32);
        assert_eq!(f.sin.len(), 512 * 32);
    }

    #[test]
    fn freqs_position_zero_cos_one_sin_zero() {
        // At position=0, θ=0 for all i → cos=1, sin=0.
        let cfg = RopeConfig::new(8, 4);
        let f = RopeFreqs::precompute(&cfg);
        for i in 0..4 {
            assert!(approx_eq(f.cos[i], 1.0, 1e-6), "cos[0,{i}]={}", f.cos[i]);
            assert!(approx_eq(f.sin[i], 0.0, 1e-6), "sin[0,{i}]={}", f.sin[i]);
        }
    }

    #[test]
    fn freqs_cos_sin_unit_circle() {
        // cos²+sin²=1 for all positions.
        let cfg = RopeConfig::new(8, 16);
        let f = RopeFreqs::precompute(&cfg);
        for i in 0..f.cos.len() {
            let norm = f.cos[i] * f.cos[i] + f.sin[i] * f.sin[i];
            assert!(approx_eq(norm, 1.0, 1e-5), "unit circle violated at i={i}: norm={norm}");
        }
    }

    // ── RoPE application ──────────────────────────────────────────────────────

    #[test]
    fn rope_position_zero_is_identity() {
        // At position=0, cos=1 sin=0 → rotation is identity.
        let c = cfg(4);
        let f = RopeFreqs::precompute(&c);
        let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // batch=1, heads=1, seq=1, d=4
        let out = rope(&input, &f, &[0], 1, &c);
        for i in 0..4 {
            assert!(approx_eq(out[i], input[i], 1e-6), "i={i}: {}", out[i]);
        }
    }

    #[test]
    fn rope_output_length_preserved() {
        let c = cfg(8);
        let f = RopeFreqs::precompute(&c);
        let input = vec![0.5_f32; 2 * 4 * 8]; // batch=2, heads=4, seq=1, d=8
        let out = rope(&input, &f, &[0], 4, &c);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation is orthogonal → it preserves the L2 norm of each head vector.
        let c = cfg(8);
        let f = RopeFreqs::precompute(&c);
        let input: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let out = rope(&input, &f, &[5], 1, &c);

        let norm_in: f32 = input.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_out: f32 = out.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(norm_in, norm_out, 1e-4),
            "norm not preserved: in={norm_in} out={norm_out}");
    }

    #[test]
    fn rope_double_rotation_invertible() {
        // Rotating by +θ then -θ should recover the original (approximate).
        // Equivalent: applying pos=p then negating sin gives identity.
        let c = cfg(4);
        let f = RopeFreqs::precompute(&c);
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];

        // Rotate forward at pos=3
        let rotated = rope(&input, &f, &[3], 1, &c);

        // Build inverse freqs (negate sin)
        let mut inv_freqs = RopeFreqs::precompute(&c);
        for s in inv_freqs.sin.iter_mut() { *s = -*s; }

        // Rotate back
        let recovered = rope(&rotated, &inv_freqs, &[3], 1, &c);
        for i in 0..4 {
            assert!(approx_eq(recovered[i], input[i], 1e-5),
                "i={i}: recovered={} original={}", recovered[i], input[i]);
        }
    }

    #[test]
    fn rope_different_positions_differ() {
        let c = cfg(8);
        let f = RopeFreqs::precompute(&c);
        let input = vec![1.0_f32; 8];
        let out0 = rope(&input, &f, &[0], 1, &c);
        let out5 = rope(&input, &f, &[5], 1, &c);
        // Different positions should produce different results (except pos=0 which is identity).
        assert_ne!(out0, out5);
    }

    #[test]
    fn rope_multi_seq_positions() {
        // seq_len=3, positions=[0,1,2], batch*heads=1
        let c = cfg(4);
        let f = RopeFreqs::precompute(&c);
        let input = vec![1.0_f32; 3 * 4]; // seq=3, d=4
        let out = rope(&input, &f, &[0, 1, 2], 1, &c);
        assert_eq!(out.len(), 12);
        // pos=0 is identity
        for i in 0..4 {
            assert!(approx_eq(out[i], 1.0, 1e-6), "pos0 i={i}: {}", out[i]);
        }
    }

    #[test]
    fn rope_custom_base() {
        // LLaMA-3 style base=500_000 should not panic and produce valid unit-circle freqs.
        let c = RopeConfig::with_base(8, 16, 500_000.0);
        let f = RopeFreqs::precompute(&c);
        for i in 0..f.cos.len() {
            let norm = f.cos[i] * f.cos[i] + f.sin[i] * f.sin[i];
            assert!(approx_eq(norm, 1.0, 1e-5));
        }
    }

    #[test]
    fn rope_inplace_matches_functional() {
        let c = cfg(8);
        let f = RopeFreqs::precompute(&c);
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 + 0.1).collect();
        let positions = vec![3usize];

        let functional = rope(&input, &f, &positions, 1, &c);
        let mut inplace = input.clone();
        apply_rope(&mut inplace, &f, &positions, 1, &c);
        assert_eq!(functional, inplace);
    }

    #[test]
    #[should_panic]
    fn panics_odd_head_dim() {
        RopeConfig::new(7, 64); // must be even
    }

    #[test]
    #[should_panic]
    fn panics_position_out_of_bounds() {
        let c = cfg(4);
        let f = RopeFreqs::precompute(&c);
        let input = vec![1.0_f32; 4];
        rope(&input, &f, &[999], 1, &c); // max_seq_len=64, 999 out of bounds
    }
}
