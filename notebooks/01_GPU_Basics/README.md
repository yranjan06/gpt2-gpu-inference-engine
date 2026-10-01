# Sprint 0: GPU Fundamentals

## Goal
Understand why a GPU is faster than a CPU -- and, just as importantly, when it isn't.

## What I built
- Checked GPU access and read hardware specs (SM count, VRAM) directly from PyTorch.
- Measured CPU-to-GPU transfer bandwidth vs the GPU's own VRAM bandwidth.
- Demonstrated the "async timing trap" -- unsynchronized GPU timing gives a fake,
  physically-impossible result.
- Swept matrix size from 32 to 4096 to find where a GPU stops losing to a CPU.
- Measured raw VRAM bandwidth with a pure memory-bound op (`y = x + 1`).
- Measured how batching lets a GPU reuse weights already loaded in memory.

## What I found
- GPU ops are asynchronous. Timing without `torch.cuda.synchronize()` measured
  kernel *launch* time, not run time -- a fake 261.8 TFLOPS reading on a GPU
  whose fp32 peak is ~8 TFLOPS.
- Small GPU kernels have a fixed ~30 microsecond floor, no matter how little
  work they do. Below matrix size ~128, this T4 actually loses to a 4-core CPU.
- T4's achievable VRAM bandwidth is ~230-245 GB/s (72-75% of the 320 GB/s
  datasheet peak).
- Batching reuses weights almost for free: multiplying an 8192x8192 fp16 matrix
  by 64 vectors took only ~1.36x longer than by 1 vector -- 64x the work for
  1.36x the time, because the weight matrix is read from VRAM once and reused.

## Key numbers

| Experiment | Result |
| --- | --- |
| Hardware | Tesla T4, 40 SMs, 2560 CUDA cores, 15.64 GB VRAM |
| CPU -> GPU transfer (pageable) | ~4.8 GB/s |
| VRAM bandwidth (measured) | ~230-245 GB/s |
| Fake vs real TFLOPS (no sync vs sync) | 261.8 (fake) vs 4.2 (real) |
| CPU vs GPU matmul crossover | n ~ 128 |
| Matmul speedup at n=4096 | 17.2x (CPU: 248 GFLOPS, GPU: 4258 GFLOPS) |
| Batch 1 -> 64 time increase (8192x8192 fp16) | ~1.36x, for 64x more work |
