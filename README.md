# BlitzKernels

Production WASM inference kernels for edge AI — built in Rust, compiled to `wasm32-wasip2`, deployable on Cloudflare Workers, wasmCloud, and any WASI runtime.

## Kernels

| Kernel | Description | Status |
|--------|-------------|--------|
| `blitz-embedding` | Batched token embedding with mean-pool + L2 norm | Available |
| `blitz-attention` | Multi-head attention (MHA) block | Available |
| `blitz-kv-cache` | KV-cache for autoregressive inference | Available |

## Purchase

Single kernel: **$1,500** | Bundle of 3: **$3,500**

Landing page: https://blitzkernels.pages.dev  
Contact: hello@teamadapt.dev

## Quick Example (blitz-embedding)

```rust
use blitz_embedding::{batch_embed, EmbeddingConfig, EmbeddingTable};

let config = EmbeddingConfig::default();
let table = EmbeddingTable::new(config.vocab_size, config.hidden_dim);
let token_ids = vec![1u32, 42, 100, 7];
let embeddings = batch_embed(&token_ids, &table, &config);
// embeddings: Vec<Vec<f32>>, L2-normalized, ready for cosine similarity
```

## Architecture

- **Target**: `wasm32-wasip2` (WASI Preview 2 + Component Model)
- **No runtime dependencies** — pure Rust, no external crates at inference time  
- **Integration**: 30-minute architecture call included with purchase
- **Delivery**: Private repo access or zip within 24h of payment

## License

Source code delivered under commercial license. Contact hello@teamadapt.dev.

---

Built by [TeamADAPT](https://teamadapt.dev) — production Rust AI infrastructure.
