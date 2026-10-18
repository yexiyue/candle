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

None yet. The branch is currently byte-identical to the base commit apart from
this file.

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
