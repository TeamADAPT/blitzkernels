/// Eviction policy for bounded KV-cache.
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Evict least-recently-used tokens first.
    Lru,
    /// Fixed sliding window — only keep the last N tokens.
    SlidingWindow { window_size: usize },
    /// Evict tokens with the lowest cumulative attention scores.
    /// Requires attention score tracking to be enabled.
    AttentionScore,
}

/// Configuration for the KV-cache.
pub struct KvCacheConfig {
    /// Maximum sequence length the cache can hold before eviction.
    pub max_seq_len: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Number of KV heads (for GQA this is less than num_query_heads).
    pub num_kv_heads: usize,
    /// Page size in tokens — each page is a contiguous allocation unit.
    pub page_size: usize,
    /// Default eviction policy when cache is full.
    pub eviction_policy: EvictionPolicy,
}

impl KvCacheConfig {
    pub fn new(max_seq_len: usize, num_layers: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        Self {
            max_seq_len,
            num_layers,
            head_dim,
            num_kv_heads,
            page_size: 256,
            eviction_policy: EvictionPolicy::Lru,
        }
    }

    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }

    pub fn with_eviction(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }
}
