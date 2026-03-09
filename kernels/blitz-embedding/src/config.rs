/// Quantization mode for embedding output.
#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    /// Full 32-bit float (no quantization).
    None,
    /// 16-bit float — halves memory, minimal quality loss.
    Fp16,
    /// 8-bit integer with per-channel scale factors.
    /// Typically <1% quality degradation on retrieval benchmarks.
    Int8,
}

/// Configuration for batched embedding computation.
pub struct EmbeddingConfig {
    /// Model embedding dimension (e.g., 768 for BERT-base, 1024 for large models).
    pub model_dim: usize,
    /// Maximum tokens per input in the batch. Shorter inputs are padded.
    pub max_tokens: usize,
    /// Maximum batch size. Larger batches amortize kernel launch overhead.
    pub max_batch: usize,
    /// Output quantization mode.
    pub quantization: Quantization,
    /// Whether to L2-normalize output embeddings (standard for similarity search).
    pub normalize: bool,
}

impl EmbeddingConfig {
    pub fn new(model_dim: usize) -> Self {
        Self {
            model_dim,
            max_tokens: 512,
            max_batch: 256,
            quantization: Quantization::None,
            normalize: true,
        }
    }

    pub fn with_quantization(mut self, q: Quantization) -> Self {
        self.quantization = q;
        self
    }

    pub fn with_max_batch(mut self, max_batch: usize) -> Self {
        self.max_batch = max_batch;
        self
    }
}
