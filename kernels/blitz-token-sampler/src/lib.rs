//! **blitz-token-sampler** — LLM token sampling kernel.
//!
//! Converts raw logits from a language model into sampled token indices.
//!
//! ## Strategies
//!
//! | Strategy | Description | Use case |
//! |----------|-------------|----------|
//! | **Greedy** | Always picks argmax | Deterministic, code generation |
//! | **Top-k** | Sample from top k logits | Creative text, controlled diversity |
//! | **Top-p (nucleus)** | Sample from smallest set covering p mass | Open-ended generation |
//!
//! Temperature scales logits before softmax — `<1.0` sharpens, `>1.0` flattens.
//!
//! ## Example
//!
//! ```rust
//! use blitz_token_sampler::{sample_greedy, sample_top_k, sample_top_p, SampleConfig};
//!
//! let logits = vec![1.0f32, 2.0, 0.5, 3.0, 1.5];
//! assert_eq!(sample_greedy(&logits), 3);
//!
//! let cfg = SampleConfig::new(42).with_temperature(0.8);
//! let tok = sample_top_k(&logits, 3, &cfg);
//! assert!(tok < 5);
//!
//! let cfg = SampleConfig::new(7);
//! let tok = sample_top_p(&logits, 0.9, &cfg);
//! assert!(tok < 5);
//! ```
//!
//! Pure Rust, no external deps, WASM-portable (wasm32-wasip2).

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Configuration for stochastic sampling strategies.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    pub seed: u64,
    pub temperature: f32,
}

impl SampleConfig {
    pub fn new(seed: u64) -> Self {
        Self { seed, temperature: 1.0 }
    }
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        assert!(temperature > 0.0, "temperature must be > 0");
        self.temperature = temperature;
        self
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rand_f32(state: &mut u64) -> f32 {
    let bits = (xorshift64(state) >> 41) as u32;
    f32::from_bits(0x3f80_0000 | bits) - 1.0
}

/// Apply temperature scaling and compute softmax probabilities.
pub fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(temperature > 0.0, "temperature must be > 0");
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Greedy decoding: always return the argmax token.
///
/// ```rust
/// use blitz_token_sampler::sample_greedy;
/// assert_eq!(sample_greedy(&[0.1f32, 0.9, 0.3]), 1);
/// ```
pub fn sample_greedy(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "logits must not be empty");
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

/// Top-k sampling: sample from the k highest-probability tokens.
///
/// ```rust
/// use blitz_token_sampler::{sample_top_k, SampleConfig};
/// let logits = vec![1.0f32, 2.0, 0.5, 3.0, 1.5];
/// let cfg = SampleConfig::new(42);
/// assert!(sample_top_k(&logits, 3, &cfg) < 5);
/// ```
pub fn sample_top_k(logits: &[f32], k: usize, cfg: &SampleConfig) -> usize {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(k > 0, "k must be > 0");
    let k = k.min(logits.len());
    let probs = softmax_with_temperature(logits, cfg.temperature);
    let mut sorted_probs: Vec<f32> = probs.clone();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    let threshold = sorted_probs[k - 1];
    let mut candidates: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= threshold)
        .map(|(i, &p)| (i, p))
        .collect();
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    for (_, p) in candidates.iter_mut() { *p /= total; }
    let mut state = cfg.seed.wrapping_add(1);
    let r = rand_f32(&mut state);
    let mut cumulative = 0.0f32;
    for (idx, prob) in &candidates {
        cumulative += prob;
        if r <= cumulative { return *idx; }
    }
    candidates.last().map(|(i, _)| *i).unwrap_or(0)
}

/// Top-p (nucleus) sampling: sample from the smallest set of tokens whose
/// cumulative probability mass is at least `p`.
///
/// ```rust
/// use blitz_token_sampler::{sample_top_p, SampleConfig};
/// let logits = vec![1.0f32, 2.0, 0.5, 3.0, 1.5];
/// let cfg = SampleConfig::new(99);
/// assert!(sample_top_p(&logits, 0.9, &cfg) < 5);
/// ```
pub fn sample_top_p(logits: &[f32], p: f32, cfg: &SampleConfig) -> usize {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
    let probs = softmax_with_temperature(logits, cfg.temperature);
    let mut sorted: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    let mut cumulative = 0.0f32;
    for (idx, prob) in &sorted {
        nucleus.push((*idx, *prob));
        cumulative += prob;
        if cumulative >= p { break; }
    }
    let total: f32 = nucleus.iter().map(|(_, p)| p).sum();
    for (_, p) in nucleus.iter_mut() { *p /= total; }
    let mut state = cfg.seed.wrapping_add(7);
    let r = rand_f32(&mut state);
    let mut cum = 0.0f32;
    for (idx, prob) in &nucleus {
        cum += prob;
        if r <= cum { return *idx; }
    }
    nucleus.last().map(|(i, _)| *i).unwrap_or(0)
}

/// Greedy sampling over a batch of logit vectors (flat, shape [batch, vocab]).
pub fn batch_greedy(logits: &[f32], batch_size: usize, vocab_size: usize) -> Vec<usize> {
    assert_eq!(logits.len(), batch_size * vocab_size);
    (0..batch_size)
        .map(|b| sample_greedy(&logits[b * vocab_size..(b + 1) * vocab_size]))
        .collect()
}

/// Top-p sampling over a batch with per-item seeds derived from base_seed.
pub fn batch_top_p(logits: &[f32], batch_size: usize, vocab_size: usize, p: f32, base_seed: u64) -> Vec<usize> {
    assert_eq!(logits.len(), batch_size * vocab_size);
    (0..batch_size)
        .map(|b| {
            let cfg = SampleConfig::new(base_seed.wrapping_add(b as u64));
            sample_top_p(&logits[b * vocab_size..(b + 1) * vocab_size], p, &cfg)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        assert_eq!(sample_greedy(&[0.1f32, 0.9, 0.3, 0.7, 0.5]), 1);
    }

    #[test]
    fn greedy_handles_negative_logits() {
        assert_eq!(sample_greedy(&[-5.0f32, -1.0, -3.0, -0.5, -2.0]), 3);
    }

    #[test]
    fn greedy_single_element() {
        assert_eq!(sample_greedy(&[42.0f32]), 0);
    }

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax_with_temperature(&[1.0f32, 2.0, 3.0, 4.0], 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    }

    #[test]
    fn softmax_temperature_sharpens() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let hot = softmax_with_temperature(&logits, 2.0);
        let cold = softmax_with_temperature(&logits, 0.5);
        assert!(cold[2] > hot[2], "low temp should sharpen top token");
    }

    #[test]
    fn top_k_returns_valid_token() {
        let logits = vec![1.0f32, 2.0, 0.5, 3.0, 1.5];
        let tok = sample_top_k(&logits, 3, &SampleConfig::new(42));
        assert!(tok < 5, "token {tok} out of range");
    }

    #[test]
    fn top_k_eq_1_is_greedy() {
        let logits = vec![0.1f32, 5.0, 0.2, 0.3];
        assert_eq!(sample_top_k(&logits, 1, &SampleConfig::new(123)), 1);
    }

    #[test]
    fn top_k_clips_to_vocab_size() {
        let tok = sample_top_k(&[1.0f32, 2.0, 3.0], 100, &SampleConfig::new(1));
        assert!(tok < 3);
    }

    #[test]
    fn top_p_returns_valid_token() {
        let logits = vec![1.0f32, 2.0, 0.5, 3.0, 1.5];
        let tok = sample_top_p(&logits, 0.9, &SampleConfig::new(99));
        assert!(tok < 5, "token {tok} out of range");
    }

    #[test]
    fn top_p_full_mass_covers_all() {
        let tok = sample_top_p(&[1.0f32, 1.0, 1.0, 1.0], 1.0, &SampleConfig::new(7));
        assert!(tok < 4);
    }

    #[test]
    fn top_p_very_peaked_picks_argmax() {
        let logits = vec![-10.0f32, -10.0, 10.0, -10.0];
        assert_eq!(sample_top_p(&logits, 0.5, &SampleConfig::new(5)), 2);
    }

    #[test]
    fn batch_greedy_correct() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 4.0, 1.0];
        let res = batch_greedy(&logits, 2, 3);
        assert_eq!(res, vec![2, 1]);
    }

    #[test]
    fn batch_top_p_correct_shape() {
        let logits = vec![1.0f32, 2.0, 3.0, 1.0, 5.0, 0.5];
        let res = batch_top_p(&logits, 2, 3, 0.9, 42);
        assert_eq!(res.len(), 2);
        for &t in &res { assert!(t < 3); }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let logits = vec![1.0f32, 2.0, 0.5, 3.0, 1.5];
        assert_eq!(
            sample_top_k(&logits, 3, &SampleConfig::new(314)),
            sample_top_k(&logits, 3, &SampleConfig::new(314))
        );
        assert_eq!(
            sample_top_p(&logits, 0.9, &SampleConfig::new(314)),
            sample_top_p(&logits, 0.9, &SampleConfig::new(314))
        );
    }

    #[test]
    fn different_seeds_produce_valid_tokens() {
        let logits = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        for seed in 0u64..5 {
            let tok = sample_top_k(&logits, 3, &SampleConfig::new(seed));
            assert!(tok < 5);
        }
    }
}
