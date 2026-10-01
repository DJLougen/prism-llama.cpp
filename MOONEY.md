# MOONEY — what this branch is

Branch `lbf/flashnext-ternary` is the **llama.cpp runtime for
[Qwen3.8-Flash-Next-Mooney](https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney)** —
a 180B-parameter MoE checkpoint compressed to 92 GB on disk / about 39 GiB in
memory, running on **one NVIDIA DGX Spark** (GB10).

Mooney's GGUF shards use a `PQ2_0` ternary GGML type (id 142) for the routed
experts plus `lowbitflash.rot.*` rotation metadata — stock llama.cpp cannot
load them. This branch adds:

- `PQ2_0` loading and CUDA kernels for the ternary routed experts;
- `lowbitflash.rot.*` KV parsing (fail-closed: version, names, pow2 segments,
  ±1 signs) and the fused segmented-FWHT·sign·int8 rotation path;
- Q8_0 PLE n-gram table support — the 54.4 GB table is read lazily from SSD
  (`--load-mode mmap --tensor-read-lazy on -ot per_layer_token_embd=CPU`);
- the corrected QSA `compress_ratios` schedule (12 sparse layers);
- vision via the BF16 `mmproj` file — **images work here**.

Measured on a DGX Spark: 27.8 tok/s decode (short) · 25.8 (4k) · 17.6 (30k),
≈39 GiB resident, model load 99 s. See the model card for full numbers.

## Easiest path: one-command setup

[`DJLougen/mooney-spark`](https://github.com/DJLougen/mooney-spark) builds this
branch, downloads the model with per-file sha256 verification, and writes a
launcher:

```bash
git clone https://github.com/DJLougen/mooney-spark && cd mooney-spark
./setup_spark.sh --runtime llama.cpp
~/mooney-spark/launch/serve_llamacpp.sh   # OpenAI-compatible, http://127.0.0.1:8089/v1
```

## Build manually on a DGX Spark (GB10)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real -DLLAMA_BUILD_TESTS=OFF
ninja -C build -j12 llama-server
```

Run (the exact argv behind the model card's measured numbers):

```bash
llama-server \
  -m Qwen3.8-Flash-Next-Mooney-00001-of-00004.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Mooney-BF16.gguf \
  --load-mode mmap --tensor-read-lazy on \
  -ot per_layer_token_embd=CPU \
  -ngl all -fa on -np 1 \
  --no-cache-prompt --cache-ram 0 -c 32768 --reasoning auto \
  --host 127.0.0.1 --port 8089
```

## Credits and license

MIT, unchanged: llama.cpp (ggml-org) + the PrismML fork (`prism` branch) that
this branch builds on. Quantization and evaluation compute for the Mooney
release was generously provided by [Lambda](https://lambda.ai) — thanks to
[Zach Mueller](https://x.com/TheZachMueller).
