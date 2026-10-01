# GPT-2 GPU Inference Engine (from scratch)

I built GPT-2 inference by hand on a GPU without `model.generate()`and HuggingFace's
attention module. Every step (loading weights, attention, KV cache, custom GPU
kernels, quantization) is written by me and checked against HuggingFace's own
output, so I actually know it's correct instead of just assuming it.

Built entirely on Kaggle's free T4 GPU -- I don't have one of my own.

## Why

I wanted to understand what frameworks like vLLM hide from you: how memory
actually moves around a GPU, why a kernel can be slow even when it reads less
data, and what breaks when you write the GPU code yourself instead of calling
someone else's.

## Sprints

| # | Topic | Status |
|---|---|---|
| 0 | GPU fundamentals -- why/when a GPU is actually faster | coming soon |
| 1-2 | GPT-2 built by hand, checked line by line against HuggingFace | coming soon |
| 3 | KV cache, prefill/decode, TTFT/TPOT | coming soon |
| 4 | First GPU kernel -- fused LayerNorm in Triton | coming soon |
| 5 | Fused attention kernel -- found two real bugs | coming soon |
| 6 | INT8 quantization -- and an honest "it's still slower" result | coming soon |
| 7 | Sampling -- temperature, top-k, top-p | coming soon |

Each folder will have its own short README -- what I built, and what I
actually found. This table updates as each sprint gets pushed.

## Running it

```bash
pip install -r requirements.txt
```

## License

MIT
