# Talechime candle fork

This repository is a fork of [huggingface/candle](https://github.com/huggingface/candle)
used by [yexiyue/talechime](https://github.com/yexiyue/talechime) through a workspace
`[patch.crates-io]` entry.

## Base

- Upstream repository: `https://github.com/huggingface/candle`
- Base commit: `31f35b147389700ed2a178ee66a91c3cc25cc80d` (upstream tag `0.11.0`,
  "Bump candle version to 0.11.0 (#3658)")
- Branch used by Talechime: `talechime-perf`
- Crate versions are intentionally **unchanged** (`0.11.0`) so Talechime's
  `=0.11.0` requirements keep resolving to this fork.
- Licensing is unchanged: dual Apache-2.0 / MIT, see `LICENSE-APACHE` and
  `LICENSE-MIT`.

## Patches

### `candle-metal-kernels`: 32-row GEMM tile for M up to 32

- Upstream: huggingface/candle#3313 (Metal GEMM Dynamic Tile Selection) added
  `select_tile_config` with a `if m < 16 { return TILE_32_32_16_2_2 }` guard.
- Change: raise the guard to `m <= 32`. For `16 <= m <= 32` both `bm = 32` and
  `bm = 64` need a single M tile, so the 64-row tile computed 32..42 padding rows
  for free but for nothing.
- Effect: VoxCPM LocDiT (`m = 22` per CFM step) on Apple M4 Pro, 21.71 -> 18.99
  ms per frame (-12.5%) with bit-identical WAV output. Shapes with `m < 16`
  (LocEnc `m = 5`, BaseLM decode `m = 1`) are unchanged.
- Behaviour-only, no public API change.

Each patch must be listed here with its upstream issue/PR, and should be kept
small enough to upstream. Prefer behaviour-only changes that do not add, remove
or alter public API: Talechime still publishes the candle-consuming crates to
crates.io, where `[patch]` is stripped and downstream builds resolve the
registry release. If a change ever requires new candle API, the affected
Talechime crates must be marked `publish = false` until the change lands
upstream.

### `candle-core` + `candle-metal-kernels`: fused shared-tap single-channel conv1d (Metal)

- Motivation: anti-aliased FIR filters (BigVGAN vocoder) issue conv1d as a
  batch of `(b*c, 1, L)` rows against one `(1, 1, k <= 12)` shared kernel.
  The generic Metal path materializes a k-times-larger im2col buffer and then
  runs `b*c` tiny GEMMs, which is pure overhead for K-tap dot products.
- Change: when `padding == 0`, dtype `F32`, `c_in == c_out == 1`,
  `k <= 64` and both layouts are contiguous, dispatch to a fused
  one-thread-per-output kernel (`conv1d_shared_f32`); every other shape keeps
  the generic path.
- Effect: IndexTTS BigVGAN FIR stages on Apple M4 Pro, offline probe:
  fir_down ~6.6x, fir_up ~4x faster; fixed-input deviation vs the generic
  path <= 9.5e-7. Behaviour-only, no public API change.

### `candle-core` + `candle-metal-kernels`: row-chunk copy for rank-4 transpose(1,2) views (Metal)

- Motivation: contiguous-izing a `transpose(1, 2)` view — dims
  `(b, x1, x2, d)` with strides `(x1*x2*d, d, x1*d, 1)`, the attention
  `(B,T,H,D) <-> (B,H,T,D)` materializations — falls through
  `copy_strided_src` to the element-wise strided gather because
  `strided_blocks()` yields `MultipleBlocks` (the wrap stride between `x1`
  rows breaks uniformity). Measured 11.6 GB/s effective bandwidth on M4 Pro
  (~4% of available) for a 4 MB copy.
- Change: when the stride signature above matches exactly and dtype is F32
  with `d % 4 == 0` and both offsets 16B aligned, dispatch to
  `transpose2_contiguous_f32`: grid `(d/4, rows)`, one thread moves one
  float4 chunk of a contiguous d-row. The kernel reads exactly the offsets
  the checked strides imply, so any matching layout is handled correctly;
  every other shape keeps the existing path. Pure copy, bit-identical by
  construction (unit-tested against the CPU contiguous path).
- Effect: CosyVoice3/IndexTTS attention split + output transposes.
  Behaviour-only, no public API change in `candle-core` (the launcher lives
  in `candle-metal-kernels`, an internal dependency).

### `candle-core` + `candle-metal-kernels`: broadcast fast path for binary ops (Metal)

- Motivation: a binary op with a contiguous lhs and a broadcast rhs —
  `(B,1,D)->(B,T,D)` gating/modulation or `(D,)->(B,T,D)` bias — dispatches
  to the generic strided kernel, which computes a multi-dim div/mod index
  chain per element for the rhs. Measured ~0.9 GB/s effective bandwidth on
  M4 Pro for the CosyVoice3 DiT gate chain.
- Change: when the lhs is contiguous, both are f32 3-D with matching
  broadcast shapes, and the rhs strides are `[s, 0, 1]` / `[0, 0, s]` with
  `s` a multiple of `D` (covers narrow views of modulation tensors), dispatch to `binary_<op>_f32_rbcast`: the rhs index collapses
  to `(i / group_len) * src_stride + i % block_len`, matching the checked
  strides by construction. Every other shape keeps the existing path.
  Elementwise, so bit-identical by construction (unit-tested against the
  CPU broadcast path for both shapes).
- Effect: CosyVoice3 DiT gate broadcasts. Behaviour-only, no public API
  change in `candle-core` (launcher lives in `candle-metal-kernels`).

### `candle-metal-kernels`: dense direct conv1d kernel (Metal, NOT dispatched)

> Tried for the CosyVoice3 grouped positional convs and rejected: see the
> note in `metal_backend/mod.rs` conv1d. Kept (kernel + launcher + gates)
> as dormant infrastructure.



- Motivation: stride-1/dilation-1/valid conv1d goes through an im2col
  materialization `(b, l_out, c_in*k)` plus a GEMM. Grouped convs issue one
  such call per group from the tensor-level loop, so the im2col traffic —
  not the math — dominates: the CosyVoice3 positional conv (k=31, groups=16,
  c=1024, T~534) writes ~136 MB per call for ~4.3 GFLOP of actual work.
- Change: when `padding == 0`, `stride == 1`, `dilation == 1`, dtype F32,
  `k <= 64`, `c_in_per_group * k <= 4096` (any strides — the kernel is
  stride-parametric, which the grouped conv's non-contiguous per-group chunk
  views require), dispatch to `conv1d_dense_f32`: one thread per output
  element computing
  the full `(c_in x k)` dot product directly. Every other shape keeps the
  im2col path. Summation order differs from im2col+GEMM, gated at <= 1e-5
  against the CPU reference (incl. the grouped positional-conv geometry).
- Effect: CosyVoice3 DiT input embedding conv. Behaviour-only, no public
  API change in `candle-core` (launcher lives in `candle-metal-kernels`).
- v2: one thread per 8-wide output tile along `l` (contiguous src window
  reads, k row reused across the thread group). The naive one-thread-per-
  output variant was memory-latency bound (~9 GB of redundant scattered
  reads per conv) and slower than im2col; the tiled variant is not.

## Removing the fork

Delete the `[patch.crates-io]` block in the Talechime workspace `Cargo.toml`,
refresh `Cargo.lock`, and delete this repository once every patch listed above
is either merged upstream or dropped.
