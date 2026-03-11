//! Flash Attention v2 configuration.

/// Configuration for a Flash Attention computation.
///
/// # Tile sizes
/// Block sizes `block_q` (Br) and `block_kv` (Bc) control the SRAM tile sizes.
/// Smaller tiles use less memory; larger tiles amortize loop overhead.
/// Typical values: Br = Bc = 64 or 128 for SRAM-limited hardware.
/// For WASM execution (no hardware SRAM), these still control loop blocking.
#[derive(Debug, Clone)]
pub struct FlashConfig {
    /// Number of query/output heads.
    pub num_heads: usize,
    /// Number of key/value heads (≤ num_heads). Enables GQA when < num_heads.
    pub num_kv_heads: usize,
    /// Sequence length (same for Q, K, V in this implementation).
    pub seq_len: usize,
    /// Head dimension (d_k).
    pub head_dim: usize,
    /// Query block tile size (Br). Rows of Q processed per outer tile.
    pub block_q: usize,
    /// Key/Value block tile size (Bc). Rows of K/V processed per inner tile.
    pub block_kv: usize,
    /// Softmax scale factor. Defaults to 1/sqrt(head_dim) if not set.
    pub scale: f32,
    /// If true, apply causal mask (each position only attends to positions ≤ itself).
    pub causal: bool,
}

impl FlashConfig {
    /// Create a new config with sensible defaults.
    ///
    /// - `block_q` = 64, `block_kv` = 64
    /// - `scale` = 1/sqrt(head_dim)
    /// - `causal` = false
    pub fn new(num_heads: usize, num_kv_heads: usize, seq_len: usize, head_dim: usize) -> Self {
        assert!(head_dim > 0, "head_dim must be > 0");
        Self {
            num_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            block_q: 64.min(seq_len),
            block_kv: 64.min(seq_len),
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
        }
    }

    /// Enable causal masking (autoregressive / decoder-only models).
    pub fn with_causal(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Override the query tile size.
    pub fn with_block_q(mut self, block_q: usize) -> Self {
        assert!(block_q > 0, "block_q must be > 0");
        self.block_q = block_q;
        self
    }

    /// Override the key/value tile size.
    pub fn with_block_kv(mut self, block_kv: usize) -> Self {
        assert!(block_kv > 0, "block_kv must be > 0");
        self.block_kv = block_kv;
        self
    }
}
