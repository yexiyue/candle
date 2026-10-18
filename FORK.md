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

## Removing the fork

Delete the `[patch.crates-io]` block in the Talechime workspace `Cargo.toml`,
refresh `Cargo.lock`, and delete this repository once every patch listed above
is either merged upstream or dropped.
