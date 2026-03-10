/// Configuration for the fused LayerNorm + GELU kernel.
#[derive(Clone, Debug)]
pub struct LayerNormGeluConfig {
    /// Feature dimension (size of each row to normalise).
    pub d_model: usize,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
    /// Use the approximate tanh-based GELU (matches PyTorch `gelu_approximate='tanh'`).
    /// When `false`, uses the exact erf-based GELU.
    pub approximate_gelu: bool,
}

impl LayerNormGeluConfig {
    /// Create a config with standard defaults (eps=1e-5, approximate GELU).
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            eps: 1e-5,
            approximate_gelu: true,
        }
    }

    /// Create a config with a custom eps.
    pub fn with_eps(d_model: usize, eps: f32) -> Self {
        Self {
            d_model,
            eps,
            approximate_gelu: true,
        }
    }

    /// Use exact erf-based GELU instead of the tanh approximation.
    pub fn with_exact_gelu(mut self) -> Self {
        self.approximate_gelu = false;
        self
    }
}
