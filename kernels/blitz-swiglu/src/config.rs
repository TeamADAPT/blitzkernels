/// Configuration for the SwiGLU activation kernel.
///
/// SwiGLU is the gating mechanism used in LLaMA, Mistral, Gemma, and PaLM:
///
/// ```text
/// SwiGLU(x) = Swish(x · W_gate + b_gate) ⊙ (x · W_up + b_up)
/// ```
///
/// Where Swish(x) = x · σ(x) and ⊙ is element-wise multiplication.
/// The gate and up projections go from `d_model` to `d_ff`.
#[derive(Clone, Debug)]
pub struct SwiGluConfig {
    /// Input dimension (model width).
    pub d_model: usize,
    /// Intermediate dimension (typically 2/3 * 4 * d_model for LLaMA-style).
    pub d_ff: usize,
    /// Swish beta parameter. Standard is 1.0. Higher values sharpen the gate.
    pub swish_beta: f32,
}

impl SwiGluConfig {
    /// Create with LLaMA-style defaults: d_ff ≈ (8/3) * d_model, rounded to nearest multiple of 8.
    pub fn new(d_model: usize) -> Self {
        // LLaMA uses 2/3 of the 4x expansion, rounded for hardware alignment
        let raw = (8 * d_model) / 3;
        let d_ff = ((raw + 7) / 8) * 8; // round up to multiple of 8
        Self {
            d_model,
            d_ff,
            swish_beta: 1.0,
        }
    }

    /// Create with explicit intermediate dimension.
    pub fn with_d_ff(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_model,
            d_ff,
            swish_beta: 1.0,
        }
    }

    /// Set the Swish beta parameter (default 1.0).
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.swish_beta = beta;
        self
    }
}
