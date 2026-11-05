# Ethereus changes to Locals

This is a private fork of [Locals](https://github.com/Generated-Computers/Locals)
(Apache-2.0) ported from Qwenputer (Qwen3.5-4B, `calc` firmware) to Ethereus-9B
untrained (Qwen3.5-9B, `eth` firmware). Upstream `LICENSE`, `NOTICE` and
`THIRD_PARTY_NOTICES.md` are kept unchanged.

## What changed

- `locals/models/qwenputer.py`: parameterised for both models. The computer's
  dims, layout and vocabulary come from the checkpoint's own `chipformer.json`
  instead of hard-coded Qwen3.5-4B constants (`D_QWEN=2560`, `REPO`, `BASE`,
  `rv1` only). Supports the `rv1-store` layout (object-store families after
  the controls, 23 ballast dims, store id bits in text rows) and Ethereus-9B
  (`ETH_BASE`, `ETH_D_QWEN=4096`, `ETH_FIRMWARE=eth-bridge.elf`, `ETH_TOOL`).
  MLX import is optional so the module parses on Linux; pack save/load uses
  torch/safetensors there. `firmware_states` builds the circuit with the
  checkpoint's store config when present.
- `locals/models/ethereus_cpu.py` (new): the chip pack run on CPU with torch
  (no MLX), following `Chip` step by step, for exactness on Linux.
- `locals/models/qwenputer_walk.py`, `qwenputer_cpu.py`: parse on Linux
  (MLX-gated); the C stepper's ballast count comes from the pack, not
  `#define PRE 62`. (Mac-only execution; see `docs/mac-test.md`.)
- `locals/defaults.py`: `ethereus-9b` model id (Qwen's part
  `mlx-community/Qwen3.5-9B-MLX-4bit`, computer from the local Ethereus-9B
  checkpoint; `is_qwenputer` covers it so serving, parking and the computer
  thread apply; internal tool `eth` instead of `calc`).
- `tests/test_ethereus.py` (new): the chip pack vs the reference machine
  (`ref.py` via `bridge.Session`) token for token on 50 `eth` tool calls
  (keccak, ABI encode/decode, EIP-712 type hashes, address checks).
- `docs/mac-test.md` (new): exact Mac commands for the MLX/C parts that
  cannot run on Linux.
- `locals/models/eth-bridge.elf` (new, 245 KB): the parked `eth` firmware
  (bridge build) the pack's snapshot is made from. The non-bridge gate-1
  image `eth.elf` lives alongside the checkpoint under `/workspace` and is
  not shipped (it cannot serve tool calls from one snapshot; see report).

## Checkpoint and firmware (read from `/workspace`, not guessed)

- Checkpoint: `/workspace/dist-store/q9-store-rv` (Ethereus-9B untrained,
  `rv1-store`, hidden 5888 = 4096 Qwen + 1792 computer, 12 computer layers
  + 32 Qwen layers, chip vocab 236469, 23 ballast dims).
- Firmware: `/workspace/ethereus-main/experimental/riscv/tests/data/rv/eth-bridge.elf`
  (tool-call image; `eth.elf` is the non-bridge gate-1 image).
- Reference: `/workspace/ethereus-main/experimental/riscv/chipformer/rv/ref.py`
  via `bridge.Session`.

## Pack

Built on CPU with `qwenputer.convert` + `firmware_states`/`add_snapshot`:
`chip.safetensors` (189 MB, 83M dense compacted params, R=1781),
`snapshot.safetensors` (19 MB), `chip.json`, `eth-bridge.elf` — ~208 MB total
(Qwenputer-4B's pack is 123 MB).
