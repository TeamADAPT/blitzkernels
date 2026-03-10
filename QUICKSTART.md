# BlitzKernels — Quickstart

Three deployment paths: Cloudflare Workers, wasmCloud, and native Rust.

---

## 1. Cloudflare Workers (wasm32-wasip2)

Install [Wrangler](https://developers.cloudflare.com/workers/wrangler/) and the WASM target:

```bash
npm install -g wrangler
rustup target add wasm32-wasip2
```

`wrangler.toml`:
```toml
name = "my-embedding-worker"
main = "src/index.ts"
compatibility_date = "2024-01-01"

[build]
command = "cargo build --target wasm32-wasip2 --release -p blitz-embedding"

[[rules]]
type = "CompiledWasm"
globs = ["target/wasm32-wasip2/release/*.wasm"]
```

Worker (`src/index.ts`):
```typescript
import { batch_embed } from '../target/wasm32-wasip2/release/blitz_embedding.wasm';

export default {
  async fetch(request: Request): Promise<Response> {
    const { token_ids } = await request.json();
    const embeddings = batch_embed(token_ids, 768);
    return Response.json({ embeddings });
  }
};
```

Deploy:
```bash
wrangler deploy
```

---

## 2. wasmCloud (Actor model)

```bash
wash build     # builds to wasm32-wasip2 via Cargo.toml target
wash push ghcr.io/teamadapt/blitz-embedding:v0.1.0 ./build/blitz_embedding_s.wasm
wash start actor ghcr.io/teamadapt/blitz-embedding:v0.1.0
```

---

## 3. Native Rust (wasmtime embedding)

```toml
# Cargo.toml
[dependencies]
wasmtime = "19"
wasmtime-wasi = "19"
blitz-embedding = { git = "https://github.com/TeamADAPT/blitzkernels" }
```

```rust
use blitz_embedding::{batch_embed, EmbeddingConfig, EmbeddingTable};

fn main() {
    let table = EmbeddingTable::new(32000, 768);
    let config = EmbeddingConfig::default();

    let token_ids = vec![
        vec![101u32, 2054, 2003, 2026, 3793, 102],
        vec![101, 7592, 102],
    ];

    let embeddings = batch_embed(&token_ids, &table, &config);
    println!("batch size: {}, dim: {}", embeddings.len(), embeddings[0].len());
}
```

---

## Build All Kernels

```bash
git clone https://github.com/TeamADAPT/blitzkernels
cd blitzkernels
rustup target add wasm32-wasip2
cargo build --workspace --target wasm32-wasip2 --release
ls target/wasm32-wasip2/release/*.wasm
```

---

## Commercial License

Kernel: **$1,500** · Full bundle (8 kernels): **$6,500**

Contact: [hello@teamadapt.dev](mailto:hello@teamadapt.dev?subject=BlitzKernels%20License)  
Catalog: [blitzkernels.pages.dev](https://blitzkernels.pages.dev)
