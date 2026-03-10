/// Configuration for the fused MLP kernel.
///
/// Represents a standard transformer FFN block:
///   hidden = GELU(LayerNorm(input @ W_up + b_up))
///   output = hidden @ W_down + b_down
///
/// Where W_up projects from `d_model` to `d_ff` and W_down projects back.
#[derive(Clone, Debug)]
pub struct FusedMlpConfig {
    /// Input / output dimension (model width).
    pub d_model: usize,
    /// Feed-forward intermediate dimension (typically 4 * d_model).
    pub d_ff: usize,
    /// Epsilon for LayerNorm numerical stability.
    pub eps: f32,
    /// Use the tanh-approximate GELU (matches PyTorch `gelu_approximate='tanh'`).
    pub approximate_gelu: bool,
}

impl FusedMlpConfig {
    /// Create with standard defaults: d_ff = 4 * d_model, eps=1e-5, approximate GELU.
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            d_ff: 4 * d_model,
            eps: 1e-5,
            approximate_gelu: true,
        }
    }

    /// Create with explicit intermediate dimension.
    pub fn with_d_ff(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_model,
            d_ff,
            eps: 1e-5,
            approximate_gelu: true,
        }
    }

    /// Use exact erf-based GELU.
    pub fn with_exact_gelu(mut self) -> Self {
        self.approximate_gelu = false;
        self
    }

    /// Set custom eps.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}
