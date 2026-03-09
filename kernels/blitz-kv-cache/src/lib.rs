// Portable when built without the "std" feature (wasm64-unknown-unknown target).
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;

pub use config::{EvictionPolicy, KvCacheConfig};

/// A paged KV-cache for transformer inference.
///
/// Stores key-value pairs for all transformer layers in a flat pool, with a
/// logical page table tracking which physical slots hold which sequence positions.
/// When capacity is reached, the configured [`EvictionPolicy`] selects tokens to drop.
///
/// # Storage layout
/// Keys and values are stored per (layer, kv_head) pair in flat `Vec<f32>` buffers.
/// Each position `p` occupies elements `p * head_dim .. (p+1) * head_dim` within
/// its `(layer, kv_head)` buffer.
///
/// # Example
/// ```
/// use blitz_kv_cache::{KvCache, KvCacheConfig, EvictionPolicy};
///
/// // 2 layers, 2 KV heads, head_dim=4, capacity=8 tokens
/// let cfg = KvCacheConfig::new(8, 2, 4, 2);
/// let mut cache = KvCache::new(cfg);
///
/// // Append one step: keys/values layout = [num_layers * num_kv_heads * head_dim]
/// let stride = 2 * 2 * 4; // layers * kv_heads * head_dim
/// cache.append(&vec![1.0_f32; stride], &vec![2.0_f32; stride]);
/// assert_eq!(cache.len(), 1);
///
/// let kv = cache.get(&[0]);
/// assert_eq!(kv.seq_positions, vec![0]);
/// ```
pub struct KvCache {
    config: KvCacheConfig,

    /// keys[layer * num_kv_heads + kh]: flat storage, len = num_tokens * head_dim
    keys: Vec<Vec<f32>>,
    /// values[layer * num_kv_heads + kh]: same shape as keys
    values: Vec<Vec<f32>>,

    /// Logical sequence positions for each cached token slot.
    seq_positions: Vec<usize>,

    /// Cumulative attention scores per slot (for AttentionScore eviction).
    attn_scores: Vec<f32>,

    /// Monotone clock incremented on every `get()` call; used for LRU.
    clock: u64,
    /// Last access time per slot (for LRU eviction).
    access_time: Vec<u64>,

    /// Number of cached tokens currently in the cache.
    num_tokens: usize,
}

/// A view into cached key-value pairs at a set of sequence positions.
pub struct KvSlice {
    /// Gathered keys, layout: [num_positions * num_layers * num_kv_heads * head_dim]
    pub keys: Vec<f32>,
    /// Gathered values, same layout as `keys`
    pub values: Vec<f32>,
    /// The sequence positions that were found (same order as the `positions` argument)
    pub seq_positions: Vec<usize>,
}

impl KvCache {
    /// Allocate a new KV-cache with the given configuration.
    pub fn new(config: KvCacheConfig) -> Self {
        let num_slots = config.max_seq_len;
        let num_slots_per_buf = num_slots * config.head_dim;
        let num_bufs = config.num_layers * config.num_kv_heads;

        let keys = (0..num_bufs).map(|_| vec![0.0_f32; num_slots_per_buf]).collect();
        let values = (0..num_bufs).map(|_| vec![0.0_f32; num_slots_per_buf]).collect();

        Self {
            config,
            keys,
            values,
            seq_positions: Vec::new(),
            attn_scores: Vec::new(),
            clock: 0,
            access_time: Vec::new(),
            num_tokens: 0,
        }
    }

    /// Append new key-value pairs for one or more generation steps.
    ///
    /// # Layout
    /// `keys` and `values` must each have length
    /// `n_new_tokens * num_layers * num_kv_heads * head_dim`, where
    /// `n_new_tokens = keys.len() / (num_layers * num_kv_heads * head_dim)`.
    ///
    /// The sequence positions assigned to the new tokens are
    /// `[self.len(), self.len() + n_new_tokens)`.
    ///
    /// If appending would exceed `max_seq_len`, the configured eviction policy
    /// is applied before writing the new tokens.
    ///
    /// # Panics
    /// Panics if `keys.len()` is not a multiple of `num_layers * num_kv_heads * head_dim`.
    pub fn append(&mut self, keys: &[f32], values: &[f32]) {
        let stride = self.config.num_layers * self.config.num_kv_heads * self.config.head_dim;
        assert!(
            stride > 0 && keys.len() % stride == 0,
            "keys length {} is not a multiple of num_layers*num_kv_heads*head_dim={}",
            keys.len(),
            stride,
        );
        assert_eq!(keys.len(), values.len(), "keys and values must have the same length");

        let n_new = keys.len() / stride;

        // Evict as needed to make room.
        let max_cap = self.config.max_seq_len;
        let needed = self.num_tokens + n_new;
        if needed > max_cap {
            let to_evict = needed - max_cap;
            self.evict_n(self.config.eviction_policy, to_evict);
        }

        // Write each new token into the storage.
        for t in 0..n_new {
            let seq_pos = self.num_tokens; // logical position
            let slot = self.num_tokens;    // physical slot (append-only while under capacity)

            for l in 0..self.config.num_layers {
                for kh in 0..self.config.num_kv_heads {
                    let buf_idx = l * self.config.num_kv_heads + kh;
                    let src_off = (t * self.config.num_layers * self.config.num_kv_heads
                        + l * self.config.num_kv_heads
                        + kh)
                        * self.config.head_dim;
                    let dst_off = slot * self.config.head_dim;
                    self.keys[buf_idx][dst_off..dst_off + self.config.head_dim]
                        .copy_from_slice(&keys[src_off..src_off + self.config.head_dim]);
                    self.values[buf_idx][dst_off..dst_off + self.config.head_dim]
                        .copy_from_slice(&values[src_off..src_off + self.config.head_dim]);
                }
            }

            self.seq_positions.push(seq_pos);
            self.attn_scores.push(0.0);
            self.access_time.push(self.clock);
            self.num_tokens += 1;
        }
    }

    /// Retrieve cached KV pairs at the requested sequence positions.
    ///
    /// Each found position updates its LRU timestamp and accumulates an attention
    /// score of `1.0` (uniform; callers may update `attn_scores` directly for
    /// attention-score eviction).
    ///
    /// Returns a [`KvSlice`] with gathered keys/values in the order positions appear.
    /// Positions not present in the cache are silently skipped.
    pub fn get(&self, positions: &[usize]) -> KvSlice {
        let hd = self.config.head_dim;
        let nl = self.config.num_layers;
        let nkh = self.config.num_kv_heads;
        let kv_stride = nl * nkh * hd;

        let mut out_keys = Vec::with_capacity(positions.len() * kv_stride);
        let mut out_values = Vec::with_capacity(positions.len() * kv_stride);
        let mut found_positions = Vec::with_capacity(positions.len());

        for &pos in positions {
            // Linear scan — for large caches a hash-map index would be faster,
            // but keeps the implementation dependency-free.
            if let Some(slot) = self.seq_positions.iter().position(|&p| p == pos) {
                found_positions.push(pos);
                for l in 0..nl {
                    for kh in 0..nkh {
                        let buf_idx = l * nkh + kh;
                        let src = slot * hd;
                        out_keys.extend_from_slice(&self.keys[buf_idx][src..src + hd]);
                        out_values.extend_from_slice(&self.values[buf_idx][src..src + hd]);
                    }
                }
            }
        }

        KvSlice { keys: out_keys, values: out_values, seq_positions: found_positions }
    }

    /// Evict `n` tokens according to `policy`.
    ///
    /// Returns the number of tokens actually evicted (may be less than `n` if
    /// the cache contains fewer tokens).
    pub fn evict(&mut self, policy: EvictionPolicy) -> usize {
        let to_evict = self.num_tokens.min(1);
        self.evict_n(policy, to_evict)
    }

    /// Current number of cached tokens.
    pub fn len(&self) -> usize {
        self.num_tokens
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Update the cumulative attention score for a cached position.
    ///
    /// Call this after each forward pass with the attention weights summed over all
    /// heads so that [`EvictionPolicy::AttentionScore`] can rank tokens by relevance.
    pub fn add_attention_score(&mut self, seq_pos: usize, score: f32) {
        if let Some(slot) = self.seq_positions.iter().position(|&p| p == seq_pos) {
            self.attn_scores[slot] += score;
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Evict exactly `n` tokens (clamped to cache size) using `policy`.
    /// Removes slots in-place by swapping with the last element (O(n) per eviction).
    fn evict_n(&mut self, policy: EvictionPolicy, n: usize) -> usize {
        let n = n.min(self.num_tokens);
        for _ in 0..n {
            if self.num_tokens == 0 {
                break;
            }
            let victim = self.choose_victim(policy);
            self.remove_slot(victim);
        }
        n
    }

    /// Select the slot index to evict according to `policy`.
    fn choose_victim(&self, policy: EvictionPolicy) -> usize {
        match policy {
            EvictionPolicy::Lru => {
                // Evict the slot with the smallest (oldest) access time.
                self.access_time
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &t)| t)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            EvictionPolicy::SlidingWindow { window_size } => {
                // Evict the slot whose seq_position is furthest outside the window.
                // The window retains the `window_size` highest seq_positions.
                let max_pos = self.seq_positions.iter().copied().max().unwrap_or(0);
                let cutoff = max_pos.saturating_sub(window_size.saturating_sub(1));
                // Find oldest position below the cutoff, or just the minimum.
                self.seq_positions
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p < cutoff)
                    .min_by_key(|&(_, &p)| p)
                    .map(|(i, _)| i)
                    .unwrap_or_else(|| {
                        self.seq_positions
                            .iter()
                            .enumerate()
                            .min_by_key(|&(_, &p)| p)
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    })
            }
            EvictionPolicy::AttentionScore => {
                // Evict the slot with the lowest cumulative attention score.
                self.attn_scores
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        }
    }

    /// Remove slot `idx` by swapping with the last slot and truncating.
    fn remove_slot(&mut self, idx: usize) {
        let last = self.num_tokens - 1;
        let hd = self.config.head_dim;

        if idx != last {
            // Swap key/value data for all (layer, kv_head) buffers.
            for buf in &mut self.keys {
                let (a, b) = (idx * hd, last * hd);
                buf.copy_within(b..b + hd, a);
            }
            for buf in &mut self.values {
                let (a, b) = (idx * hd, last * hd);
                buf.copy_within(b..b + hd, a);
            }
            self.seq_positions.swap(idx, last);
            self.attn_scores.swap(idx, last);
            self.access_time.swap(idx, last);
        }

        self.seq_positions.pop();
        self.attn_scores.pop();
        self.access_time.pop();
        self.num_tokens -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stride(cfg: &KvCacheConfig) -> usize {
        cfg.num_layers * cfg.num_kv_heads * cfg.head_dim
    }

    fn make_cache(max_seq_len: usize, layers: usize, hd: usize, kv_heads: usize) -> KvCache {
        KvCache::new(KvCacheConfig::new(max_seq_len, layers, hd, kv_heads))
    }

    #[test]
    fn test_new_is_empty() {
        let cache = make_cache(16, 2, 4, 2);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_append_one_token_len_one() {
        let mut cache = make_cache(8, 2, 4, 2);
        let s = stride(cache.config());
        cache.append(&vec![1.0; s], &vec![2.0; s]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let mut cache = make_cache(16, 1, 4, 1);
        let s = stride(cache.config());
        cache.append(&vec![0.5; s * 5], &vec![0.5; s * 5]);
        assert_eq!(cache.len(), 5);
    }

    #[test]
    fn test_get_roundtrip_values() {
        let layers = 2;
        let kv_heads = 2;
        let hd = 4;
        let mut cache = make_cache(8, layers, hd, kv_heads);
        let s = layers * kv_heads * hd;
        // Append token 0 with all-1 keys, all-2 values
        let keys = vec![1.0_f32; s];
        let vals = vec![2.0_f32; s];
        cache.append(&keys, &vals);
        let kv = cache.get(&[0]);
        assert_eq!(kv.seq_positions, vec![0]);
        assert_eq!(kv.keys.len(), s);
        assert_eq!(kv.values.len(), s);
        for &k in &kv.keys   { assert!((k - 1.0).abs() < 1e-6); }
        for &v in &kv.values { assert!((v - 2.0).abs() < 1e-6); }
    }

    #[test]
    fn test_get_missing_position_skipped() {
        let mut cache = make_cache(8, 1, 4, 1);
        let s = stride(cache.config());
        cache.append(&vec![1.0; s], &vec![1.0; s]);
        let kv = cache.get(&[99]); // position 99 not in cache
        assert!(kv.seq_positions.is_empty());
        assert!(kv.keys.is_empty());
    }

    #[test]
    fn test_lru_eviction_removes_oldest() {
        let mut cache = make_cache(3, 1, 2, 1).with_policy(EvictionPolicy::Lru);
        let s = stride(cache.config());
        // Fill to capacity
        cache.append(&vec![1.0; s * 3], &vec![1.0; s * 3]);
        assert_eq!(cache.len(), 3);
        // Append one more — LRU should evict slot 0 (oldest)
        cache.append(&vec![9.0; s], &vec![9.0; s]);
        assert_eq!(cache.len(), 3);
        // Position 0 should have been evicted
        let kv = cache.get(&[0]);
        assert!(kv.seq_positions.is_empty(), "position 0 should have been evicted");
    }

    #[test]
    fn test_sliding_window_eviction() {
        let window = 2;
        let mut cache = make_cache(4, 1, 2, 1)
            .with_policy(EvictionPolicy::SlidingWindow { window_size: window });
        let s = stride(cache.config());
        cache.append(&vec![1.0; s * 4], &vec![1.0; s * 4]); // tokens 0-3
        assert_eq!(cache.len(), 4);
        // Append one more — sliding window keeps last 2, evicts from outside
        cache.append(&vec![5.0; s], &vec![5.0; s]); // token 4
        assert_eq!(cache.len(), 4);
        // Positions 0 or 1 should have been evicted (outside window of 2 from max=4)
        let kv0 = cache.get(&[0]);
        let kv1 = cache.get(&[1]);
        assert!(
            kv0.seq_positions.is_empty() || kv1.seq_positions.is_empty(),
            "at least one of positions 0/1 should be evicted"
        );
    }

    #[test]
    fn test_attention_score_eviction_removes_lowest_score() {
        let mut cache = make_cache(3, 1, 2, 1).with_policy(EvictionPolicy::AttentionScore);
        let s = stride(cache.config());
        cache.append(&vec![1.0; s * 3], &vec![1.0; s * 3]);
        // Give position 1 a high attention score so it survives
        cache.add_attention_score(1, 10.0);
        // Append triggers eviction of the lowest-score slot (0 or 2, both at 0.0)
        cache.append(&vec![9.0; s], &vec![9.0; s]);
        assert_eq!(cache.len(), 3);
        // Position 1 must still be present
        let kv = cache.get(&[1]);
        assert_eq!(kv.seq_positions, vec![1]);
    }

    #[test]
    fn test_evict_explicit_call() {
        let mut cache = make_cache(8, 1, 2, 1);
        let s = stride(cache.config());
        cache.append(&vec![1.0; s * 4], &vec![1.0; s * 4]);
        assert_eq!(cache.len(), 4);
        cache.evict(EvictionPolicy::Lru);
        assert_eq!(cache.len(), 3);
    }

    impl KvCache {
        fn config(&self) -> &KvCacheConfig { &self.config }
        fn with_policy(mut self, policy: EvictionPolicy) -> Self {
            self.config.eviction_policy = policy;
            self
        }
    }
}
