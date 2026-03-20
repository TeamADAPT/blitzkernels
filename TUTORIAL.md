# BlitzKernels Developer Tutorial
**Author:** River, ResearchOps T1 — NOVACOL-404
**Date:** 2026-03-20
**Audience:** Developers who just purchased a license or are evaluating BlitzKernels
**Time:** ~20 minutes to complete
**Purpose:** Go from zero to a running WASM inference kernel

---

## What You'll Build

A minimal token generator that runs the core transformer inference loop:
1. Encode input with flash attention
2. Maintain a KV-cache for autoregressive generation
3. Sample the next token

All in pure Rust, compiling to both native (for testing) and `wasm32-wasip2` (for production).

---

## Prerequisites

```bash
# Rust stable (1.76+)
rustup show

# Add WASM target
rustup target add wasm32-wasip2

# Verify
rustup target list --installed | grep wasm
```

---

## Part 1: Single Kernel — Flash Attention

Start with the simplest case: one attention forward pass.

### Create the project

```bash
cargo new blitz-hello --bin
cd blitz-hello
```

### Cargo.toml

```toml
[package]
name = "blitz-hello"
version = "0.1.0"
edition = "2021"

[dependencies]
blitz-flash-attention = "0.1"

[lib]
crate-type = ["cdylib", "rlib"]
```

### src/main.rs

```rust
use blitz_flash_attention::{flash_attention, FlashConfig};

fn main() {
    // Model config: 4 heads, 2 KV heads (GQA), seq_len=8, head_dim=16
    let cfg = FlashConfig::new(4, 2, 8, 16).with_causal();

    // Input tensors (row-major: [num_heads * seq_len * head_dim])
    let n_q  = cfg.num_heads    * cfg.seq_len * cfg.head_dim;
    let n_kv = cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;

    let q = vec![0.1_f32; n_q];  // queries
    let k = vec![0.1_f32; n_kv]; // keys
    let v = vec![1.0_f32; n_kv]; // values

    let out = flash_attention(&q, &k, &v, &cfg);

    println!("Output shape: {} elements", out.output.len());
    println!("First 4 values: {:?}", &out.output[..4]);
    // Each output value should be ~1.0 (attending uniformly to constant v=1.0)
}
```

### Run native (fast for development)

```bash
cargo run
# Output shape: 512 elements
# First 4 values: [1.0, 1.0, 1.0, 1.0]
```

### Compile to WASM

```bash
cargo build --target wasm32-wasip2 --release

# Run with wasmtime
wasmtime target/wasm32-wasip2/release/blitz-hello.wasm
```

---

## Part 2: KV-Cache for Autoregressive Generation

Real generation is autoregressive — each token attends to all previous tokens.
The KV-cache stores keys and values so they're not recomputed.

### Add to Cargo.toml

```toml
[dependencies]
blitz-flash-attention = "0.1"
blitz-kv-cache = "0.1"
```

### src/main.rs — add KV-cache

```rust
use blitz_flash_attention::{flash_attention, FlashConfig};
use blitz_kv_cache::{KvCache, KvCacheConfig, EvictionPolicy};

fn generate_step(
    cache: &mut KvCache,
    new_k: &[f32],
    new_v: &[f32],
    query: &[f32],
    attn_cfg: &FlashConfig,
) -> Vec<f32> {
    // Append new KV pair to cache
    cache.append(new_k, new_v);

    // Get all cached KV (for attention over full context)
    let positions: Vec<usize> = (0..cache.len()).collect();
    let kv = cache.get(&positions);

    // Run attention: query against all cached keys/values
    // (In production, reshape kv.keys/kv.values to match FlashConfig layout)
    let out = flash_attention(query, &kv.keys, &kv.values, attn_cfg);
    out.output
}

fn main() {
    // 2 layers, 2 KV heads, head_dim=16, max 32 tokens, LRU eviction
    let cache_cfg = KvCacheConfig::new(32, 2, 16, 2)
        .with_eviction(EvictionPolicy::Lru);
    let mut cache = KvCache::new(cache_cfg);

    let attn_cfg = FlashConfig::new(4, 2, 1, 16).with_causal();

    // Simulate 4 generation steps
    let stride = 2 * 2 * 16; // layers * kv_heads * head_dim
    for step in 0..4 {
        let new_k = vec![step as f32 * 0.1; stride];
        let new_v = vec![1.0_f32; stride];
        let query = vec![0.5_f32; 4 * 1 * 16]; // [num_heads * seq_len=1 * head_dim]

        let output = generate_step(&mut cache, &new_k, &new_v, &query, &attn_cfg);
        println!("Step {}: output[0] = {:.4}", step, output[0]);
    }

    println!("Cache contains {} tokens", cache.len());
}
```

```bash
cargo run
# Step 0: output[0] = 1.0000
# Step 1: output[0] = 1.0000
# Step 2: output[0] = 1.0000
# Step 3: output[0] = 1.0000
# Cache contains 4 tokens
```

---

## Part 3: Token Sampling

The attention output feeds into a linear projection to produce logits over the vocabulary.
BlitzKernels provides three sampling strategies.

### Add to Cargo.toml

```toml
[dependencies]
blitz-flash-attention = "0.1"
blitz-kv-cache = "0.1"
blitz-token-sampler = "0.1"
```

### Sampling strategies

```rust
use blitz_token_sampler::{sample_greedy, sample_top_k, sample_top_p, SampleConfig};

fn sample_next_token(logits: &[f32], strategy: &str) -> usize {
    match strategy {
        // Deterministic: always picks highest probability token
        // Best for: code generation, structured output
        "greedy" => sample_greedy(logits),

        // Sample from top-k most likely tokens
        // Best for: creative text, controlled diversity
        "top_k" => {
            let cfg = SampleConfig::new(42).with_temperature(0.8);
            sample_top_k(logits, 50, &cfg)
        }

        // Sample from smallest set covering p probability mass
        // Best for: open-ended generation (LLaMA default)
        "top_p" | _ => {
            let cfg = SampleConfig::new(42).with_temperature(0.9);
            sample_top_p(logits, 0.9, &cfg)
        }
    }
}

fn main() {
    // Pretend these are logits from your language model head
    // (linear projection from attention output to vocab size)
    let vocab_size = 32_000; // LLaMA 3 vocab
    let logits: Vec<f32> = (0..vocab_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    let greedy_token = sample_next_token(&logits, "greedy");
    let topk_token   = sample_next_token(&logits, "top_k");
    let topp_token   = sample_next_token(&logits, "top_p");

    println!("Greedy → token {}", greedy_token);
    println!("Top-k  → token {}", topk_token);
    println!("Top-p  → token {}", topp_token);
}
```

---

## Part 4: Full Inference Loop (Native → WASM)

Putting it together in a single binary that compiles to both targets.

### src/main.rs

```rust
use blitz_flash_attention::{flash_attention, FlashConfig};
use blitz_kv_cache::{KvCache, KvCacheConfig};
use blitz_token_sampler::{sample_top_p, SampleConfig};

const VOCAB_SIZE: usize = 32_000;
const NUM_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 2;   // GQA as in LLaMA 3
const HEAD_DIM: usize = 64;
const NUM_LAYERS: usize = 4;
const MAX_TOKENS: usize = 512;

fn main() {
    // --- Setup ---
    let attn_cfg = FlashConfig::new(NUM_HEADS, NUM_KV_HEADS, 1, HEAD_DIM)
        .with_causal();

    let mut cache = KvCache::new(
        KvCacheConfig::new(MAX_TOKENS, NUM_KV_HEADS, HEAD_DIM, NUM_LAYERS)
    );

    let sample_cfg = SampleConfig::new(1337).with_temperature(0.8);

    println!("BlitzKernels inference loop starting...");
    println!("Config: {} heads ({} KV), head_dim={}, max_tokens={}",
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_TOKENS);

    // --- Generate 8 tokens ---
    let mut generated_tokens: Vec<usize> = Vec::new();
    let stride = NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM;

    for step in 0..8 {
        // In production: these come from your embedding table + positional encoding
        let query = vec![0.1_f32; NUM_HEADS * 1 * HEAD_DIM];
        let new_k = vec![(step as f32) * 0.01; stride];
        let new_v = vec![1.0_f32; stride];

        // Append to KV-cache
        cache.append(&new_k, &new_v);

        // Flash attention over full context
        let kv = cache.get(&(0..cache.len()).collect::<Vec<_>>());
        let attn_out = flash_attention(&query, &kv.keys, &kv.values, &attn_cfg);

        // In production: apply LM head (linear projection) to attn_out → logits
        // Here we synthesize logits from the attention output for demo
        let logits: Vec<f32> = (0..VOCAB_SIZE)
            .map(|i| attn_out.output[i % attn_out.output.len()] + (i as f32 * 1e-4))
            .collect();

        let token = sample_top_p(&logits, 0.9, &sample_cfg);
        generated_tokens.push(token);

        println!("  step {}: token_id={:6}", step, token);
    }

    println!("\nGenerated {} tokens. Cache: {}/{} used.",
        generated_tokens.len(), cache.len(), MAX_TOKENS);
}
```

### Build and run

```bash
# Native (development, fast iteration)
cargo run --release

# WASM (production target)
cargo build --target wasm32-wasip2 --release

# Size check
ls -lh target/wasm32-wasip2/release/*.wasm
# Typical: 150-300KB for all 3 kernels combined

# Run with wasmtime
wasmtime target/wasm32-wasip2/release/blitz-hello.wasm
```

---

## Part 5: Deploying to Cloudflare Workers

```toml
# Cargo.toml additions for Workers
[profile.release]
opt-level = "z"
lto = true
strip = true
```

```bash
# Build
cargo build --target wasm32-wasip2 --release

# Deploy via wrangler (requires Cloudflare account)
wrangler deploy --compatibility-flag "wasm_module_imports"
```

Your Workers script:
```javascript
import wasmModule from './target/wasm32-wasip2/release/blitz-hello.wasm';

export default {
  async fetch(request) {
    const instance = await WebAssembly.instantiate(wasmModule);
    // Call your exported inference function
    return new Response("Inference complete");
  }
};
```

---

## Troubleshooting

**"error: no matching package named `blitz-flash-attention`"**
→ Add the path to your workspace, or use the GitHub repo URL in Cargo.toml:
```toml
blitz-flash-attention = { git = "https://github.com/TeamADAPT/blitzkernels" }
```

**"panics at 'num_heads must be divisible by num_kv_heads'"**
→ Use `FlashConfig::new(8, 2, seq_len, head_dim)` — 8 heads, 2 KV heads (valid GQA ratio)

**"WASM binary too large (>1MB)"**
→ Add to Cargo.toml: `[profile.release] opt-level = "z"; lto = true; strip = true`

**"wasmtime: import 'wasi:random/random' not found"**
→ Use `wasmtime --wasm component-model` flag, or update to wasmtime ≥ 14.0

---

## What's Next

- **API access:** See `ops/bizops/nova-api-quickstart.md` to use kernels via REST
- **All 12 kernels:** `layernorm-gelu`, `swiglu`, `rope`, `rmsnorm`, `embedding-batch`, `blitz-int8-matmul`, `blitz-bf16-matmul` follow the same `fn kernel(input: &[f32], config: &Config) -> Output` pattern
- **Benchmarks:** Run `cargo bench` for per-kernel performance numbers
- **Enterprise support:** hello@teamadapt.dev

---

*-- RIVER | 2026-03-20 | NOVACOL-404*
