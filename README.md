# BlitzKernels

> Production WASM inference kernels for edge AI — built in Rust, compiled to `wasm32-wasip2`.  
> Deploy on Cloudflare Workers, wasmCloud, or any WASI runtime. No GPU required at the edge.

[![license: commercial](https://img.shields.io/badge/license-commercial-blue)](https://blitzkernels.pages.dev)
[![built with Rust](https://img.shields.io/badge/built_with-Rust-orange)](https://www.rust-lang.org)
[![target: wasm32-wasip2](https://img.shields.io/badge/target-wasm32--wasip2-purple)](https://wasi.dev)

## Kernel Catalog — 9 Kernels Available

| Kernel | Description | Use Case |
|--------|-------------|----------|
| `blitz-embedding` | Batched token embedding · mean-pool · L2 norm | RAG, semantic search, feature extraction |
| `blitz-attention` | Multi-head attention (MHA) with LSE | Transformer inference block |
| `blitz-kv-cache` | Paged KV-cache · LRU/sliding-window eviction | Autoregressive generation |
| `blitz-layernorm-gelu` | Fused LayerNorm + GELU activation | BERT/GPT FFN sublayer |
| `blitz-rope` | Rotary position embeddings (RoPE) | LLaMA/Mistral position encoding |
| `blitz-fused-mlp` | Fused Linear→LayerNorm→GELU→Linear | Full FFN block (GPT-style) |
| `blitz-swiglu` | SwiGLU gated activation | LLaMA 2/3, Mistral FFN |
| `blitz-rmsnorm` | RMS LayerNorm (no mean subtraction) | LLaMA/Mistral/Gemma normalization |
| `cc-faculty-wasm` | Claude Code cognitive substrate | Agent memory + reasoning integration |

**[→ View full catalog and pricing](https://blitzkernels.pages.dev)**

---

## Why BlitzKernels?

- **Pure Rust, zero unsafe** — memory-safe by construction, auditable
- **WASM-native** — runs in Cloudflare Workers, wasmCloud, Fastly Compute, Deno Deploy
- **Composable primitives** — stack kernels to build full inference pipelines
- **No vendor lock-in** — wasm32-wasip2 runs anywhere WASI is supported
- **Ed25519 signed** — every release is cryptographically signed for supply-chain integrity

---

## Quick Start

### Embedding (semantic search, RAG)

```rust
use blitz_embedding::{batch_embed, EmbeddingConfig, EmbeddingTable};

let config = EmbeddingConfig::default();
let table = EmbeddingTable::new(config.vocab_size, config.hidden_dim);
let token_ids = vec![1u32, 42, 100, 7];
let embeddings = batch_embed(&token_ids, &table, &config);
// Vec<Vec<f32>> — L2-normalized, ready for cosine similarity
```

### Multi-Head Attention

```rust
use blitz_attention::{fused_mha, AttentionConfig};

let config = AttentionConfig { num_heads: 12, head_dim: 64, ..Default::default() };
let output = fused_mha(&queries, &keys, &values, &config);
// AttentionOutput { data: Vec<f32>, lse: Vec<f32> }
```

### SwiGLU (LLaMA/Mistral FFN)

```rust
use blitz_swiglu::{swiglu_forward, SwiGluConfig};

let config = SwiGluConfig { d_model: 4096, d_ff: 11008 };  // LLaMA-7B dims
let output = swiglu_forward(&x, &gate_weight, &up_weight, &down_weight, &config);
```

### RMSNorm (LLaMA/Mistral/Gemma normalization)

```rust
use blitz_rmsnorm::rms_norm;

let mut hidden = vec![/* your hidden states */];
let weight = vec![1.0_f32; hidden_size];  // learned scale parameter
rms_norm(&mut hidden, &weight, hidden_size, 1e-6);
// Normalized in-place — faster than LayerNorm (no mean subtraction)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│           Your WASI Runtime (CF Workers, etc.)        │
├──────────────────────────────────────────────────────┤
│  blitz-embedding → blitz-attention → blitz-kv-cache  │
│  blitz-rope → blitz-rmsnorm → blitz-swiglu           │
│  blitz-fused-mlp → blitz-layernorm-gelu              │
│  cc-faculty-wasm                                      │
└──────────────────────────────────────────────────────┘
         Pure Rust · No allocator required
         No CUDA · No runtime deps · WASI P2
```

- **Target:** `wasm32-wasip2` (WASI Preview 2 + Component Model)
- **Delivery:** Pre-compiled `.wasm` + Rust source + integration guide
- **Signing:** Ed25519 (every release cryptographically verified)
- **Integration:** 30-minute architecture call included with purchase

---

## Pricing

| Option | Price | Includes |
|--------|-------|---------|
| Single kernel | **$1,500** | Pre-compiled WASM + source + 30-min integration call |
| Full catalog (9) | **$8,500** | All 9 kernels + dedicated integration support + priority updates |
| Support add-on | **$200/mo** | Priority email, patch releases, architecture review |

**[→ Purchase at blitzkernels.pages.dev](https://blitzkernels.pages.dev)**  
**Contact:** hello@teamadapt.dev

---

## Supply Chain Security

Every kernel release is:
1. Built from auditable Rust source (`cargo build --release --target wasm32-wasip2`)
2. SHA-256 checksummed
3. Ed25519 signed with our public key (available on request)

Verify a release:
```bash
openssl pkeyutl -verify -pubin -inkey blitz-public.pem \
  -rawin -in blitz-embedding.sha256 \
  -sigfile blitz-embedding.blitzpkg.sig
```

---

## License

Source code delivered under commercial license. Contact hello@teamadapt.dev.

---

Built by [TeamADAPT](https://teamadapt.dev) — production Rust AI infrastructure.  
*Running the Nova Collective: 40+ autonomous AI agents, Temporal workflows, 27-tier memory.*
