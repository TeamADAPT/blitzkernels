/// Configuration for fused multi-head attention.
pub struct AttentionConfig {
    /// Dimension per attention head (typically 64, 80, or 128)
    pub head_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for grouped-query attention; equals num_heads for MHA)
    pub num_kv_heads: usize,
    /// Maximum sequence length
    pub seq_len: usize,
    /// Whether to apply causal (autoregressive) masking
    pub causal: bool,
    /// Softmax scale factor (typically 1/sqrt(head_dim))
    pub scale: f32,
    /// SRAM tile size in elements — controls on-chip blocking for H200
    pub tile_size: usize,
}

impl AttentionConfig {
    pub fn new(head_dim: usize, num_heads: usize, seq_len: usize) -> Self {
        Self {
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            seq_len,
            causal: true,
            scale: 1.0 / (head_dim as f32).sqrt(),
            tile_size: 128,
        }
    }

    /// Configure for grouped-query attention (GQA) as used in Llama 3, etc.
    pub fn with_gqa(mut self, num_kv_heads: usize) -> Self {
        assert!(
            self.num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads"
        );
        self.num_kv_heads = num_kv_heads;
        self
    }
}
