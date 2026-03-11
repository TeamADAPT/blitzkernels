//! **blitz-flash-attention** — Flash Attention v2 kernel.
//!
//! Implements tiled block attention with online softmax normalization.
//! Avoids materialising the full O(N²) attention score matrix — each Q-tile
//! iterates over all K/V-tiles, accumulating with running `m` (max) and `ℓ`
//! (sum-exp) statistics that are rescaled across tiles.
//!
//! ## Algorithm (Flash Attention v2 forward pass)
//!
//! ```text
//! For each Q-tile [i*Br..(i+1)*Br]:
//!   Initialize: O = 0, ℓ = 0, m = -∞
//!   For each KV-tile [j*Bc..(j+1)*Bc]:
//!     S = Q_tile @ K_tile^T * scale            # [Br, Bc] scores
//!     If causal: mask positions where k_col > q_row
//!     m_new = max(m, rowmax(S))
//!     P = exp(S - m_new)                       # stable numerator
//!     ℓ_new = exp(m - m_new) * ℓ + rowsum(P)  # rescale accumulator
//!     O = (exp(m - m_new) * O + P @ V_tile)   # rescale and accumulate
//!     m = m_new; ℓ = ℓ_new
//!   O = O / ℓ                                  # normalize
//!   logsumexp = m + log(ℓ)                     # for backward / ring-attention
//! ```
//!
//! ## Memory complexity
//!
//! - Standard attention: O(N²) to store the score matrix
//! - Flash Attention: O(N) — only Br×Bc tile in SRAM at a time
//!
//! ## Features
//!
//! - **Causal masking** — decoder-only models (LLaMA, Mistral, Gemma)
//! - **Grouped Query Attention (GQA)** — when `num_kv_heads < num_heads`
//! - **Configurable tile sizes** — tune Br/Bc for your memory budget
//! - **Pure Rust, no external deps, WASM-portable**
//!
//! ## Example
//!
//! ```rust
//! use blitz_flash_attention::{flash_attention, FlashConfig};
//!
//! let cfg = FlashConfig::new(4, 2, 8, 16).with_causal();
//! let n_q  = cfg.num_heads    * cfg.seq_len * cfg.head_dim;
//! let n_kv = cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
//!
//! let q = vec![0.1_f32; n_q];
//! let k = vec![0.1_f32; n_kv];
//! let v = vec![1.0_f32; n_kv];
//!
//! let out = flash_attention(&q, &k, &v, &cfg);
//! assert_eq!(out.output.len(), n_q);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;

pub use config::FlashConfig;

// ── Output ────────────────────────────────────────────────────────────────────

/// Result of a Flash Attention forward pass.
pub struct FlashOutput {
    /// Attention output tensor, row-major: `[batch, num_heads, seq_len, head_dim]`.
    pub output: Vec<f32>,
    /// Log-sum-exp per query position: `[batch, num_heads, seq_len]`.
    ///
    /// Used for numerically-stable chunked / ring attention corrections and the
    /// backward pass (gradient through softmax).
    pub logsumexp: Vec<f32>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Compute Flash Attention v2 forward pass.
///
/// # Inputs (row-major)
/// - `q`: `[batch, num_heads,    seq_len, head_dim]`
/// - `k`: `[batch, num_kv_heads, seq_len, head_dim]`
/// - `v`: `[batch, num_kv_heads, seq_len, head_dim]`
///
/// # Returns
/// [`FlashOutput`] containing the output tensor and per-position log-sum-exp.
///
/// # Panics
/// Panics if slice lengths are inconsistent with the configuration, or if
/// `num_heads % num_kv_heads != 0`.
pub fn flash_attention(q: &[f32], k: &[f32], v: &[f32], cfg: &FlashConfig) -> FlashOutput {
    let h  = cfg.num_heads;
    let hkv = cfg.num_kv_heads;
    let n  = cfg.seq_len;
    let d  = cfg.head_dim;

    assert!(h % hkv == 0, "num_heads ({h}) must be divisible by num_kv_heads ({hkv})");

    let q_per_batch  = h   * n * d;
    let kv_per_batch = hkv * n * d;

    assert!(q_per_batch > 0 && q.len() % q_per_batch == 0,
        "q length {} is not a multiple of num_heads*seq_len*head_dim={}", q.len(), q_per_batch);
    let batch = q.len() / q_per_batch;
    assert_eq!(k.len(), batch * kv_per_batch, "k length mismatch");
    assert_eq!(v.len(), batch * kv_per_batch, "v length mismatch");

    let total_out  = batch * h * n * d;
    let total_lse  = batch * h * n;
    let mut output    = vec![0.0_f32; total_out];
    let mut logsumexp = vec![0.0_f32; total_lse];

    let gqa = h / hkv;   // heads per kv group

    for b in 0..batch {
        let q_base  = b * q_per_batch;
        let kv_base = b * kv_per_batch;

        for head in 0..h {
            let kv_head = head / gqa;

            let q_head_off  = q_base  + head   * n * d;
            let k_head_off  = kv_base + kv_head * n * d;
            let v_head_off  = kv_base + kv_head * n * d;
            let out_head_off = (b * h + head) * n * d;
            let lse_head_off = (b * h + head) * n;

            flash_attention_single_head(
                &q[q_head_off..q_head_off + n * d],
                &k[k_head_off..k_head_off + n * d],
                &v[v_head_off..v_head_off + n * d],
                &mut output[out_head_off..out_head_off + n * d],
                &mut logsumexp[lse_head_off..lse_head_off + n],
                n, d, cfg,
            );
        }
    }

    FlashOutput { output, logsumexp }
}

// ── Core: single-head Flash Attention v2 ─────────────────────────────────────

/// Flash Attention v2 forward pass for a single head.
///
/// Q-tiles of size `Br` are iterated in the outer loop. For each Q-tile,
/// all K/V-tiles of size `Bc` are processed in the inner loop, maintaining
/// running `m` (max) and `ℓ` (normalizer) statistics for stable online softmax.
fn flash_attention_single_head(
    q: &[f32],     // [seq_len, head_dim]
    k: &[f32],     // [seq_len, head_dim]
    v: &[f32],     // [seq_len, head_dim]
    out: &mut [f32],        // [seq_len, head_dim]
    lse: &mut [f32],        // [seq_len]
    n: usize,
    d: usize,
    cfg: &FlashConfig,
) {
    let br  = cfg.block_q.min(n);
    let bc  = cfg.block_kv.min(n);
    let scale = cfg.scale;
    let causal = cfg.causal;

    // Scratch buffers for the current Q-tile's running state.
    let mut m_buf  = vec![f32::NEG_INFINITY; br]; // running row-max
    let mut l_buf  = vec![0.0_f32; br];           // running normalizer
    let mut o_buf  = vec![0.0_f32; br * d];       // running output accumulator

    // Score tile: [br, bc]
    let mut s_tile = vec![0.0_f32; br * bc];

    // Number of complete+partial Q-tiles.
    let n_tiles_q  = n.div_ceil(br);
    let n_tiles_kv = n.div_ceil(bc);

    for tile_i in 0..n_tiles_q {
        let q_start = tile_i * br;
        let q_end   = (q_start + br).min(n);
        let actual_br = q_end - q_start;

        // Reset tile accumulators.
        for x in m_buf[..actual_br].iter_mut() { *x = f32::NEG_INFINITY; }
        for x in l_buf[..actual_br].iter_mut() { *x = 0.0; }
        for x in o_buf[..actual_br * d].iter_mut() { *x = 0.0; }

        for tile_j in 0..n_tiles_kv {
            let kv_start = tile_j * bc;
            let kv_end   = (kv_start + bc).min(n);
            let actual_bc = kv_end - kv_start;

            // ── Compute S = Q_tile @ K_tile^T * scale ────────────────────
            for qi in 0..actual_br {
                let q_row = &q[(q_start + qi) * d..(q_start + qi) * d + d];
                for kj in 0..actual_bc {
                    let k_row = &k[(kv_start + kj) * d..(kv_start + kj) * d + d];
                    let mut dot = 0.0_f32;
                    for l in 0..d {
                        dot += q_row[l] * k_row[l];
                    }
                    s_tile[qi * bc + kj] = dot * scale;
                }
            }

            // ── Causal masking ────────────────────────────────────────────
            // Mask out positions where k_col > q_row (future tokens).
            if causal {
                for qi in 0..actual_br {
                    let q_pos = q_start + qi;
                    for kj in 0..actual_bc {
                        let k_pos = kv_start + kj;
                        if k_pos > q_pos {
                            s_tile[qi * bc + kj] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            // ── Online softmax update ─────────────────────────────────────
            // For each query row in the tile:
            //   m_new = max(m_old, rowmax(S))
            //   P = exp(S - m_new)                  [unnormalised]
            //   l_new = exp(m_old - m_new)*l_old + rowsum(P)
            //   O = exp(m_old - m_new)*O_old + P @ V_tile
            for qi in 0..actual_br {
                let row_start = qi * bc;

                // Row-max of this KV tile.
                let mut row_max = f32::NEG_INFINITY;
                for kj in 0..actual_bc {
                    let s = s_tile[row_start + kj];
                    if s > row_max { row_max = s; }
                }

                let m_old = m_buf[qi];
                let m_new = if row_max > m_old { row_max } else { m_old };

                // Rescaling factor for existing accumulator.
                let rescale = (m_old - m_new).exp();

                // Accumulate P = exp(S - m_new) and update l.
                let mut row_sum = 0.0_f32;
                for kj in 0..actual_bc {
                    let p = (s_tile[row_start + kj] - m_new).exp();
                    s_tile[row_start + kj] = p;  // reuse buffer as P
                    row_sum += p;
                }

                let l_old = l_buf[qi];
                let l_new = rescale * l_old + row_sum;

                // Rescale existing output accumulator.
                let o_row_start = qi * d;
                for ld in 0..d {
                    o_buf[o_row_start + ld] *= rescale;
                }

                // Accumulate P @ V_tile into O.
                for kj in 0..actual_bc {
                    let p = s_tile[row_start + kj];
                    let v_row = &v[(kv_start + kj) * d..(kv_start + kj) * d + d];
                    for ld in 0..d {
                        o_buf[o_row_start + ld] += p * v_row[ld];
                    }
                }

                m_buf[qi] = m_new;
                l_buf[qi] = l_new;
            }
        } // end KV tiles

        // ── Write normalized output for this Q-tile ───────────────────────
        for qi in 0..actual_br {
            let l = l_buf[qi];
            let m = m_buf[qi];
            let inv_l = if l.abs() > 1e-30 { 1.0 / l } else { 0.0 };

            let out_row = (q_start + qi) * d;
            for ld in 0..d {
                out[out_row + ld] = o_buf[qi * d + ld] * inv_l;
            }

            // log-sum-exp = m + log(ℓ)  [numerically stable form]
            lse[q_start + qi] = m + l.ln();
        }
    } // end Q tiles
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(h: usize, hkv: usize, n: usize, d: usize) -> FlashConfig {
        FlashConfig::new(h, hkv, n, d)
    }

    // ── Basic correctness ────────────────────────────────────────────────

    #[test]
    fn output_shape_mha() {
        let cfg = make_cfg(4, 4, 8, 16);
        let q = vec![0.1_f32; 4 * 8 * 16];
        let k = vec![0.1_f32; 4 * 8 * 16];
        let v = vec![1.0_f32; 4 * 8 * 16];
        let out = flash_attention(&q, &k, &v, &cfg);
        assert_eq!(out.output.len(), 4 * 8 * 16);
        assert_eq!(out.logsumexp.len(), 4 * 8);
    }

    #[test]
    fn output_shape_gqa() {
        // 8 Q heads, 2 KV heads (GQA 4:1)
        let cfg = make_cfg(8, 2, 16, 32);
        let q  = vec![0.1_f32; 8 * 16 * 32];
        let k  = vec![0.1_f32; 2 * 16 * 32];
        let v  = vec![1.0_f32; 2 * 16 * 32];
        let out = flash_attention(&q, &k, &v, &cfg);
        assert_eq!(out.output.len(), 8 * 16 * 32);
        assert_eq!(out.logsumexp.len(), 8 * 16);
    }

    #[test]
    fn output_shape_batch() {
        let cfg = make_cfg(2, 2, 4, 8);
        let q = vec![0.1_f32; 3 * 2 * 4 * 8]; // batch=3
        let k = vec![0.1_f32; 3 * 2 * 4 * 8];
        let v = vec![1.0_f32; 3 * 2 * 4 * 8];
        let out = flash_attention(&q, &k, &v, &cfg);
        assert_eq!(out.output.len(), 3 * 2 * 4 * 8);
        assert_eq!(out.logsumexp.len(), 3 * 2 * 4);
    }

    // ── Uniform attention ─────────────────────────────────────────────────

    /// When all keys are identical, each query attends equally to every position.
    /// The output should equal the value vector (since softmax weights are uniform).
    #[test]
    fn uniform_attention_equals_value() {
        let n = 4; let d = 4; let h = 1;
        let cfg = make_cfg(h, h, n, d);
        let q = vec![1.0_f32; h * n * d];
        let k = vec![1.0_f32; h * n * d];
        // Constant value row = [1, 2, 3, 4] repeated
        let v: Vec<f32> = (0..h * n * d).map(|i| (i % d + 1) as f32).collect();
        let out = flash_attention(&q, &k, &v, &cfg);
        // With uniform attention over n=4 identical rows, output = avg of v rows = v[0]
        let expected: Vec<f32> = (0..d).map(|i| (i + 1) as f32).collect();
        for qi in 0..n {
            for di in 0..d {
                let got = out.output[qi * d + di];
                let exp = expected[di];
                assert!(
                    (got - exp).abs() < 1e-4,
                    "pos={qi} dim={di}: got={got:.5} expected={exp:.5}"
                );
            }
        }
    }

    // ── Causal mask ────────────────────────────────────────────────────────

    /// With causal masking, position 0 attends only to position 0.
    /// For uniform Q/K, the output at position 0 must equal V[0].
    #[test]
    fn causal_first_position_attends_self() {
        let n = 8; let d = 8; let h = 1;
        let cfg = make_cfg(h, h, n, d).with_causal();
        let q = vec![1.0_f32; h * n * d];
        let k = vec![1.0_f32; h * n * d];
        let mut v = vec![0.0_f32; h * n * d];
        // V[0] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in 0..d { v[i] = (i + 1) as f32 * 10.0; }
        let out = flash_attention(&q, &k, &v, &cfg);
        // position 0 only sees position 0 → output = v[0]
        for di in 0..d {
            let got = out.output[di];
            let exp = (di + 1) as f32 * 10.0;
            assert!(
                (got - exp).abs() < 1e-3,
                "dim={di}: got={got:.4} expected={exp:.4}"
            );
        }
    }

    /// With causal masking, the last position attends to all positions.
    #[test]
    fn causal_last_position_attends_all() {
        let n = 4; let d = 4; let h = 1;
        let cfg = make_cfg(h, h, n, d).with_causal();
        let q = vec![1.0_f32; h * n * d];
        let k = vec![1.0_f32; h * n * d];
        // All V rows identical → output must equal V row
        let v_val = vec![3.0_f32; h * n * d];
        let out = flash_attention(&q, &k, &v_val, &cfg);
        let last_row = &out.output[(n - 1) * d..n * d];
        for &x in last_row {
            assert!((x - 3.0).abs() < 1e-4, "last row output should be 3.0, got {x}");
        }
    }

    // ── Tile boundary cases ───────────────────────────────────────────────

    /// Test with seq_len not divisible by block_q / block_kv.
    #[test]
    fn odd_seq_len_no_panic() {
        let cfg = FlashConfig::new(2, 2, 7, 8)
            .with_block_q(4)
            .with_block_kv(3);
        let q = vec![0.5_f32; 2 * 7 * 8];
        let k = vec![0.5_f32; 2 * 7 * 8];
        let v = vec![1.0_f32; 2 * 7 * 8];
        let out = flash_attention(&q, &k, &v, &cfg);
        assert_eq!(out.output.len(), 2 * 7 * 8);
    }

    /// Test with block_q = block_kv = 1 (extreme tiling).
    #[test]
    fn single_element_tiles() {
        let cfg = FlashConfig::new(1, 1, 4, 4)
            .with_block_q(1)
            .with_block_kv(1);
        let q = vec![1.0_f32; 4 * 4];
        let k = vec![1.0_f32; 4 * 4];
        let v: Vec<f32> = (0..4 * 4).map(|i| i as f32).collect();
        let out = flash_attention(&q, &k, &v, &cfg);
        assert_eq!(out.output.len(), 4 * 4);
        // All outputs should be finite
        for &x in &out.output { assert!(x.is_finite(), "output contains non-finite: {x}"); }
    }

    /// Test with block_q larger than seq_len (single tile covers everything).
    #[test]
    fn block_larger_than_seq() {
        let cfg = FlashConfig::new(1, 1, 3, 4)
            .with_block_q(16)
            .with_block_kv(16);
        let q = vec![1.0_f32; 3 * 4];
        let k = vec![1.0_f32; 3 * 4];
        let v = vec![2.0_f32; 3 * 4];
        let out = flash_attention(&q, &k, &v, &cfg);
        for &x in &out.output {
            assert!((x - 2.0).abs() < 1e-4, "expected 2.0 got {x}");
        }
    }

    // ── Numerical stability ───────────────────────────────────────────────

    /// Large score values should not produce NaN/Inf.
    #[test]
    fn stable_with_large_scores() {
        let n = 8; let d = 4; let h = 1;
        let mut cfg = FlashConfig::new(h, h, n, d);
        cfg.scale = 10.0; // will produce large scores
        let q = vec![3.0_f32; h * n * d];
        let k = vec![3.0_f32; h * n * d];
        let v = vec![1.0_f32; h * n * d];
        let out = flash_attention(&q, &k, &v, &cfg);
        for &x in &out.output {
            assert!(x.is_finite(), "expected finite output with large scores, got {x}");
        }
    }

    /// All-negative scores (extreme causal mask) should produce finite output.
    #[test]
    fn stable_all_masked_except_self() {
        let n = 4; let d = 4; let h = 1;
        let cfg = FlashConfig::new(h, h, n, d).with_causal();
        // Position 0 sees only itself
        let q = vec![0.0_f32; h * n * d];
        let k = vec![0.0_f32; h * n * d];
        let v = vec![5.0_f32; h * n * d];
        let out = flash_attention(&q, &k, &v, &cfg);
        for &x in &out.output {
            assert!(x.is_finite(), "output should be finite, got {x}");
        }
    }

    // ── Log-sum-exp ───────────────────────────────────────────────────────

    #[test]
    fn logsumexp_shape_correct() {
        let cfg = make_cfg(4, 2, 12, 16);
        let q  = vec![0.1_f32; 4 * 12 * 16];
        let k  = vec![0.1_f32; 2 * 12 * 16];
        let v  = vec![1.0_f32; 2 * 12 * 16];
        let out = flash_attention(&q, &k, &v, &cfg);
        assert_eq!(out.logsumexp.len(), 4 * 12);
        for &lse in &out.logsumexp {
            assert!(lse.is_finite(), "lse should be finite");
        }
    }

    // ── Single token (seq_len=1) ──────────────────────────────────────────

    #[test]
    fn single_token_sequence() {
        let cfg = make_cfg(2, 2, 1, 8);
        let q = vec![1.0_f32; 2 * 1 * 8];
        let k = vec![1.0_f32; 2 * 1 * 8];
        let v = vec![7.0_f32; 2 * 1 * 8];
        let out = flash_attention(&q, &k, &v, &cfg);
        for &x in &out.output {
            assert!((x - 7.0).abs() < 1e-4, "single token: expected 7.0 got {x}");
        }
    }

    // ── Config builder ────────────────────────────────────────────────────

    #[test]
    fn config_builder_chain() {
        let cfg = FlashConfig::new(8, 4, 64, 32)
            .with_causal()
            .with_block_q(16)
            .with_block_kv(32);
        assert!(cfg.causal);
        assert_eq!(cfg.block_q, 16);
        assert_eq!(cfg.block_kv, 32);
    }

    #[test]
    fn default_scale_is_inverse_sqrt_dim() {
        let cfg = FlashConfig::new(1, 1, 4, 64);
        let expected = 1.0 / 8.0_f32; // 1/sqrt(64)
        assert!((cfg.scale - expected).abs() < 1e-6);
    }
}
