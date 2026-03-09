// Portable when built without the "std" feature (wasm64-unknown-unknown target).
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;

pub use config::AttentionConfig;

/// Output of a fused attention computation.
pub struct AttentionOutput {
    /// Attention result tensor, row-major: [batch, num_heads, seq_len, head_dim]
    pub data: Vec<f32>,
    /// Log-sum-exp per query position, row-major: [batch, num_heads, seq_len]
    /// Useful for numerically-stable chunked / ring attention corrections.
    pub logsumexp: Vec<f32>,
}

/// Compute fused multi-head attention in a single pass.
///
/// Implements the online-softmax (log-sum-exp) algorithm to avoid
/// materialising the full `[seq_len × seq_len]` attention matrix.
/// Supports grouped-query attention (GQA) when `config.num_kv_heads < config.num_heads`.
///
/// # Layout (row-major)
/// - `q`: `[batch, num_heads, seq_len, head_dim]`
/// - `k`: `[batch, num_kv_heads, seq_len, head_dim]`
/// - `v`: `[batch, num_kv_heads, seq_len, head_dim]`
/// - `mask`: optional additive bias `[seq_len, seq_len]` applied identically to all batches/heads
///
/// # Panics
/// Panics if slice lengths are inconsistent with the inferred batch size,
/// or if `num_heads` is not divisible by `num_kv_heads`.
///
/// # Example
/// ```
/// use blitz_attention::{fused_attention, AttentionConfig};
///
/// let cfg = AttentionConfig::new(4, 2, 3);
/// let n_q  = cfg.num_heads    * cfg.seq_len * cfg.head_dim;
/// let n_kv = cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
///
/// let q = vec![0.1_f32; n_q];
/// let k = vec![0.1_f32; n_kv];
/// let v = vec![1.0_f32; n_kv];
///
/// let out = fused_attention(&q, &k, &v, None, &cfg);
/// assert_eq!(out.data.len(),     n_q);
/// assert_eq!(out.logsumexp.len(), cfg.num_heads * cfg.seq_len);
/// ```
pub fn fused_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    config: &AttentionConfig,
) -> AttentionOutput {
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let seq_len = config.seq_len;
    let scale = config.scale;

    assert!(
        num_heads % num_kv_heads == 0,
        "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    );
    let gqa_ratio = num_heads / num_kv_heads;

    let q_per_batch = num_heads * seq_len * head_dim;
    assert!(
        q_per_batch > 0 && q.len() % q_per_batch == 0,
        "q length {} is not a multiple of num_heads*seq_len*head_dim={}",
        q.len(),
        q_per_batch,
    );
    let batch = q.len() / q_per_batch;

    let kv_per_batch = num_kv_heads * seq_len * head_dim;
    assert_eq!(k.len(), batch * kv_per_batch);
    assert_eq!(v.len(), batch * kv_per_batch);
    if let Some(m) = mask {
        assert_eq!(m.len(), seq_len * seq_len);
    }

    let mut data = vec![0.0_f32; batch * num_heads * seq_len * head_dim];
    let mut logsumexp = vec![0.0_f32; batch * num_heads * seq_len];

    for b in 0..batch {
        for h in 0..num_heads {
            let kh = h / gqa_ratio;

            for i in 0..seq_len {
                let attend_to = if config.causal { i + 1 } else { seq_len };

                let q_base = (b * num_heads + h) * seq_len * head_dim + i * head_dim;
                let q_row = &q[q_base..q_base + head_dim];

                // Pass 1 — dot products + running maximum for numerical stability
                let mut running_max = f32::NEG_INFINITY;
                let mut scores = vec![0.0_f32; attend_to];
                for j in 0..attend_to {
                    let k_base = (b * num_kv_heads + kh) * seq_len * head_dim + j * head_dim;
                    let k_row = &k[k_base..k_base + head_dim];
                    let mut dot = 0.0_f32;
                    for d in 0..head_dim {
                        dot += q_row[d] * k_row[d];
                    }
                    let s = dot * scale + mask.map_or(0.0, |m| m[i * seq_len + j]);
                    scores[j] = s;
                    if s > running_max {
                        running_max = s;
                    }
                }
                if running_max == f32::NEG_INFINITY {
                    running_max = 0.0;
                }

                // Pass 2 — stable softmax weights
                let mut sum_exp = 0.0_f32;
                let mut weights = vec![0.0_f32; attend_to];
                for j in 0..attend_to {
                    let e = (scores[j] - running_max).exp();
                    weights[j] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp.max(f32::MIN_POSITIVE);

                // Accumulate V
                let out_base = (b * num_heads + h) * seq_len * head_dim + i * head_dim;
                for j in 0..attend_to {
                    let w = weights[j] * inv_sum;
                    let v_base = (b * num_kv_heads + kh) * seq_len * head_dim + j * head_dim;
                    let v_row = &v[v_base..v_base + head_dim];
                    for d in 0..head_dim {
                        data[out_base + d] += w * v_row[d];
                    }
                }

                // log(Σ exp(s_j)) = log(sum_exp) + running_max (Boyen-Koller identity)
                let lse_idx = (b * num_heads + h) * seq_len + i;
                logsumexp[lse_idx] = sum_exp.ln() + running_max;
            }
        }
    }

    AttentionOutput { data, logsumexp }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_shape_single_batch() {
        let cfg = AttentionConfig::new(8, 2, 4);
        let n_q = cfg.num_heads * cfg.seq_len * cfg.head_dim;
        let n_kv = cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
        let out = fused_attention(&vec![0.1; n_q], &vec![0.1; n_kv], &vec![1.0; n_kv], None, &cfg);
        assert_eq!(out.data.len(), n_q);
        assert_eq!(out.logsumexp.len(), cfg.num_heads * cfg.seq_len);
    }

    #[test]
    fn test_output_shape_multi_batch() {
        let batch = 3;
        let cfg = AttentionConfig::new(4, 2, 5);
        let n_q = batch * cfg.num_heads * cfg.seq_len * cfg.head_dim;
        let n_kv = batch * cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
        let out = fused_attention(&vec![0.5; n_q], &vec![0.5; n_kv], &vec![2.0; n_kv], None, &cfg);
        assert_eq!(out.data.len(), n_q);
        assert_eq!(out.logsumexp.len(), batch * cfg.num_heads * cfg.seq_len);
    }

    #[test]
    fn test_uniform_v_output_equals_v_value() {
        // If all V rows are identical, output must equal that value regardless of weights.
        let cfg = AttentionConfig::new(4, 1, 3);
        let c = 3.7_f32;
        let n = cfg.num_heads * cfg.seq_len * cfg.head_dim;
        let out = fused_attention(&vec![1.0; n], &vec![1.0; n], &vec![c; n], None, &cfg);
        for &x in &out.data {
            assert!((x - c).abs() < 1e-5, "expected {c}, got {x}");
        }
    }

    #[test]
    fn test_causal_mask_first_query_only_sees_token0() {
        // With causal masking, query at i=0 can only attend to k=0 → output = V[0].
        let cfg = AttentionConfig::new(4, 1, 4);
        let hd = cfg.head_dim;
        let seq = cfg.seq_len;
        let mut v = vec![0.0_f32; seq * hd];
        for d in 0..hd { v[d] = 9.0; }
        let out = fused_attention(&vec![1.0; seq * hd], &vec![1.0; seq * hd], &v, None, &cfg);
        for d in 0..hd {
            assert!((out.data[d] - 9.0).abs() < 1e-5, "d={d}: expected 9.0 got {}", out.data[d]);
        }
    }

    #[test]
    fn test_non_causal_uniform_qk_mean_of_v() {
        // Without causal mask, uniform Q/K → uniform attention → output = mean(V rows).
        let mut cfg = AttentionConfig::new(4, 1, 3);
        cfg.causal = false;
        let hd = cfg.head_dim;
        let seq = cfg.seq_len;
        let mut v = vec![0.0_f32; seq * hd];
        for d in 0..hd {
            v[d]          = 3.0;  // row 0
            v[hd + d]     = 6.0;  // row 1
            v[2 * hd + d] = 9.0;  // row 2
        }
        let out = fused_attention(&vec![0.0; seq * hd], &vec![0.0; seq * hd], &v, None, &cfg);
        for &x in &out.data {
            assert!((x - 6.0).abs() < 1e-4, "expected 6.0, got {x}");
        }
    }

    #[test]
    fn test_gqa_4q_2kv_heads() {
        let cfg = AttentionConfig::new(8, 4, 3).with_gqa(2);
        let n_q = cfg.num_heads * cfg.seq_len * cfg.head_dim;
        let n_kv = cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
        let out = fused_attention(&vec![0.1; n_q], &vec![0.1; n_kv], &vec![1.0; n_kv], None, &cfg);
        assert_eq!(out.data.len(), n_q);
    }

    #[test]
    fn test_additive_mask_blocks_key_position() {
        // neg-inf mask on j=1 → both queries only attend j=0 → output = V[0] = 5.
        let mut cfg = AttentionConfig::new(4, 1, 2);
        cfg.causal = false;
        let hd = cfg.head_dim;
        let seq = cfg.seq_len;
        let mut v = vec![0.0_f32; seq * hd];
        for d in 0..hd { v[d] = 5.0; v[hd + d] = 10.0; }
        let mut mask = [0.0_f32; 4];
        mask[1] = f32::NEG_INFINITY;
        mask[3] = f32::NEG_INFINITY;
        let out = fused_attention(&vec![1.0; seq * hd], &vec![1.0; seq * hd], &v, Some(&mask), &cfg);
        for &x in &out.data {
            assert!((x - 5.0).abs() < 1e-5, "expected 5.0, got {x}");
        }
    }

    #[test]
    fn test_logsumexp_finite_nonzero() {
        let cfg = AttentionConfig::new(4, 2, 3);
        let n_q = cfg.num_heads * cfg.seq_len * cfg.head_dim;
        let n_kv = cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
        let out = fused_attention(&vec![0.3; n_q], &vec![0.3; n_kv], &vec![1.0; n_kv], None, &cfg);
        for &lse in &out.logsumexp {
            assert!(lse.is_finite(), "logsumexp must be finite, got {lse}");
        }
    }
}
