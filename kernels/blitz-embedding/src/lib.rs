// no_std when built without the "std" feature (i.e., the BlitzKernels pipeline
// passes --no-default-features for the wasm64-unknown-unknown target).
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

mod config;

pub use config::{EmbeddingConfig, Quantization};

/// Weight table for token embeddings.
///
/// In production use, initialize from your model checkpoint. The table is
/// separate from the kernel so that weights stay resident across batches.
///
/// # Example
/// ```
/// use blitz_embedding::EmbeddingTable;
/// let table = EmbeddingTable::new(32000, 768); // e.g., BERT-base vocab
/// ```
pub struct EmbeddingTable {
    /// Flat weight matrix stored row-major: weights[token_id * model_dim..][..model_dim]
    pub weights: Vec<f32>,
    /// Number of distinct tokens in the vocabulary.
    pub vocab_size: usize,
    /// Embedding dimension (must match EmbeddingConfig::model_dim).
    pub model_dim: usize,
}

impl EmbeddingTable {
    /// Create a table with deterministic synthetic weights — useful for
    /// benchmarking throughput and testing kernel correctness without a
    /// model checkpoint.
    pub fn new(vocab_size: usize, model_dim: usize) -> Self {
        let n = vocab_size * model_dim;
        let mut weights = Vec::with_capacity(n);
        // Hash-based init: unique per (vocab_idx, dim_idx), reproducible.
        // Mix with large primes to break periodicity.
        for vocab_idx in 0..vocab_size {
            for dim_idx in 0..model_dim {
                // Knuth multiplicative hash mix
                let h = (vocab_idx.wrapping_mul(2_654_435_761)
                    ^ dim_idx.wrapping_mul(40_503_usize))
                    .wrapping_mul(2_246_822_519);
                let phase = (h as f32) / (usize::MAX as f32) * core::f32::consts::TAU;
                weights.push(phase.sin() * 0.02);
            }
        }
        Self { weights, vocab_size, model_dim }
    }

    /// Initialize from a pre-loaded weight matrix.
    ///
    /// `weights` must be row-major with length `vocab_size * model_dim`.
    ///
    /// # Panics
    /// Panics if `weights.len() != vocab_size * model_dim`.
    pub fn from_weights(weights: Vec<f32>, vocab_size: usize, model_dim: usize) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * model_dim,
            "weights length {} != vocab_size {} * model_dim {}",
            weights.len(),
            vocab_size,
            model_dim
        );
        Self { weights, vocab_size, model_dim }
    }

    /// Look up the embedding row for a token id.
    /// Out-of-range ids are clamped to vocab_size-1 (UNK token behavior).
    #[inline]
    fn lookup(&self, token_id: u32) -> &[f32] {
        let idx = (token_id as usize).min(self.vocab_size - 1);
        let start = idx * self.model_dim;
        &self.weights[start..start + self.model_dim]
    }
}

/// A batch of computed embedding vectors.
pub struct EmbeddingBatch {
    /// Flat embedding data stored row-major: [batch_size, model_dim]
    pub data: Vec<f32>,
    /// Number of embeddings produced (may be <= input batch size).
    pub batch_size: usize,
    /// Dimensionality of each embedding vector.
    pub dim: usize,
}

impl EmbeddingBatch {
    /// Get a slice view of the embedding at position `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.batch_size`.
    pub fn get(&self, index: usize) -> &[f32] {
        assert!(index < self.batch_size, "index {} out of range (batch_size={})", index, self.batch_size);
        let start = index * self.dim;
        &self.data[start..start + self.dim]
    }
}

/// Compute embeddings for a batch of tokenized inputs.
///
/// **Pipeline (fused in one pass):**
/// 1. Token embedding lookup — gather rows from `table` by token id
/// 2. Mean pooling — average over sequence length (pad-free)
/// 3. Layer normalization — zero-mean, unit-variance, ε = 1e-5
/// 4. L2 normalization — unit-length output for cosine similarity (if `config.normalize`)
///
/// Sequences longer than `config.max_tokens` are truncated. Batches larger
/// than `config.max_batch` are truncated. Empty sequences produce zero vectors.
///
/// # Arguments
/// * `token_ids` — Batch of token id sequences (ragged — no padding needed)
/// * `table` — Embedding weight table (keep resident; load once from checkpoint)
/// * `config` — Kernel configuration (dimension, batch limits, normalization)
///
/// # Example
/// ```
/// use blitz_embedding::{batch_embed, EmbeddingConfig, EmbeddingTable};
///
/// let table = EmbeddingTable::new(1000, 128);
/// let config = EmbeddingConfig::new(128);
/// let tokens = vec![vec![1u32, 2, 3], vec![4u32, 5, 6, 7]];
/// let batch = batch_embed(&tokens, &table, &config);
/// assert_eq!(batch.batch_size, 2);
/// assert_eq!(batch.dim, 128);
/// ```
pub fn batch_embed(
    token_ids: &[Vec<u32>],
    table: &EmbeddingTable,
    config: &EmbeddingConfig,
) -> EmbeddingBatch {
    debug_assert_eq!(
        config.model_dim, table.model_dim,
        "config.model_dim must match table.model_dim"
    );

    let batch_size = token_ids.len().min(config.max_batch);
    let dim = config.model_dim;
    let mut data = vec![0.0f32; batch_size * dim];

    for (b, ids) in token_ids.iter().take(batch_size).enumerate() {
        let out = &mut data[b * dim..(b + 1) * dim];
        let n_tokens = ids.len().min(config.max_tokens);

        if n_tokens == 0 {
            // Empty sequence — leave as zero vector
            continue;
        }

        // Step 1: Sum token embeddings (mean pool)
        for &tid in ids.iter().take(n_tokens) {
            let emb = table.lookup(tid);
            for (o, &e) in out.iter_mut().zip(emb.iter()) {
                *o += e;
            }
        }
        let inv_n = 1.0 / n_tokens as f32;
        out.iter_mut().for_each(|x| *x *= inv_n);

        // Step 2: Layer normalization — zero mean, unit variance, ε = 1e-5
        let mean: f32 = out.iter().sum::<f32>() / dim as f32;
        out.iter_mut().for_each(|x| *x -= mean);
        let var: f32 = out.iter().map(|x| x * x).sum::<f32>() / dim as f32;
        let inv_std = (var + 1e-5_f32).sqrt().recip();
        out.iter_mut().for_each(|x| *x *= inv_std);

        // Step 3: L2 normalization for cosine similarity
        if config.normalize {
            let norm_sq: f32 = out.iter().map(|x| x * x).sum();
            if norm_sq > 0.0 {
                let inv_norm = norm_sq.sqrt().recip();
                out.iter_mut().for_each(|x| *x *= inv_norm);
            }
        }
    }

    EmbeddingBatch { data, batch_size, dim }
}

/// Compute cosine similarity between two embedding vectors.
///
/// Both vectors must have the same length. Returns a value in [-1, 1].
/// Returns 0.0 if either vector is the zero vector.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
///
/// # Example
/// ```
/// use blitz_embedding::cosine_similarity;
/// let a = vec![1.0f32, 0.0, 0.0];
/// let b = vec![1.0f32, 0.0, 0.0];
/// assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "embedding dimensions must match");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table(dim: usize) -> EmbeddingTable {
        EmbeddingTable::new(1000, dim)
    }

    fn make_config(dim: usize) -> EmbeddingConfig {
        EmbeddingConfig::new(dim)
    }

    #[test]
    fn test_batch_shape() {
        let batch = batch_embed(
            &[vec![1u32, 2, 3], vec![4u32, 5]],
            &make_table(64),
            &make_config(64),
        );
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.dim, 64);
        assert_eq!(batch.data.len(), 128);
    }

    #[test]
    fn test_l2_normalized_unit_norm() {
        let batch = batch_embed(
            &[vec![10u32, 20, 30, 40]],
            &make_table(128),
            &make_config(128),
        );
        let emb = batch.get(0);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "expected unit-norm embedding, got norm={:.8}",
            norm
        );
    }

    #[test]
    fn test_deterministic_output() {
        let table = make_table(64);
        let config = make_config(64);
        let tokens = vec![vec![1u32, 2, 3]];
        let b1 = batch_embed(&tokens, &table, &config);
        let b2 = batch_embed(&tokens, &table, &config);
        assert_eq!(b1.data, b2.data, "same input must produce same output");
    }

    #[test]
    fn test_empty_sequence_produces_zero() {
        let batch = batch_embed(&[vec![]], &make_table(32), &make_config(32));
        assert_eq!(batch.batch_size, 1);
        let emb = batch.get(0);
        assert!(emb.iter().all(|&x| x == 0.0), "empty sequence must yield zero vector");
    }

    #[test]
    fn test_empty_batch() {
        let batch = batch_embed(&[], &make_table(32), &make_config(32));
        assert_eq!(batch.batch_size, 0);
        assert_eq!(batch.data.len(), 0);
    }

    #[test]
    fn test_max_batch_truncation() {
        let mut config = make_config(32);
        config.max_batch = 2;
        let tokens: Vec<Vec<u32>> = (0..5).map(|i| vec![i as u32]).collect();
        let batch = batch_embed(&tokens, &make_table(32), &config);
        assert_eq!(batch.batch_size, 2, "batch should be truncated to max_batch");
    }

    #[test]
    fn test_max_tokens_truncation() {
        let mut config = make_config(32);
        config.max_tokens = 2;
        // Supply 100 tokens — only first 2 should be used
        let tokens = vec![(0..100u32).collect::<Vec<_>>()];
        let batch_truncated = batch_embed(&tokens, &make_table(32), &config);
        config.max_tokens = 512;
        let tokens_short = vec![vec![0u32, 1]];
        let batch_short = batch_embed(&tokens_short, &make_table(32), &config);
        // Both see the same first 2 tokens — embeddings should match
        assert_eq!(
            batch_truncated.get(0),
            batch_short.get(0),
            "truncated sequence must match equivalent short sequence"
        );
    }

    #[test]
    fn test_from_weights_roundtrip() {
        let dim = 16;
        let vocab = 4;
        let weights: Vec<f32> = (0..(vocab * dim)).map(|i| i as f32 * 0.01).collect();
        let table = EmbeddingTable::from_weights(weights.clone(), vocab, dim);
        // Token 0 should return first dim weights
        let row = table.lookup(0);
        assert_eq!(row, &weights[..dim]);
        // Token 2 should return weights[2*dim..]
        let row2 = table.lookup(2);
        assert_eq!(row2, &weights[2 * dim..3 * dim]);
    }

    #[test]
    fn test_out_of_range_token_clamped() {
        let table = EmbeddingTable::new(10, 32);
        // Token id 9999 >> vocab_size=10, should clamp to 9 (not panic)
        let _row = table.lookup(9999);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![0.1f32, 0.5, -0.3, 0.8];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let z = vec![0.0f32, 0.0];
        let v = vec![1.0f32, 0.0];
        assert_eq!(cosine_similarity(&z, &v), 0.0);
        assert_eq!(cosine_similarity(&v, &z), 0.0);
    }

    #[test]
    fn test_different_tokens_produce_different_embeddings() {
        let table = make_table(64);
        let config = make_config(64);
        let b1 = batch_embed(&[vec![1u32]], &table, &config);
        let b2 = batch_embed(&[vec![500u32]], &table, &config);
        // Different tokens should yield different embeddings
        let sim = cosine_similarity(b1.get(0), b2.get(0));
        assert!(sim < 0.99, "different tokens must not produce identical embeddings (sim={})", sim);
    }

    #[test]
    fn test_large_batch_throughput() {
        // Smoke test: 256 sequences of 64 tokens each in a 768-dim space
        let table = EmbeddingTable::new(32000, 768);
        let config = EmbeddingConfig::new(768);
        let tokens: Vec<Vec<u32>> = (0..256)
            .map(|b| (0..64u32).map(|t| (b * 64 + t) % 32000).collect())
            .collect();
        let batch = batch_embed(&tokens, &table, &config);
        assert_eq!(batch.batch_size, 256);
        // All output embeddings should be unit-norm
        for i in 0..256 {
            let emb = batch.get(i);
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "embedding {} has non-unit norm: {:.8}",
                i,
                norm
            );
        }
    }
}
