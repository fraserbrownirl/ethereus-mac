# Testing Ethereus-9B on a Mac

The Mac-only parts — the MLX computer (`models/qwenputer.py` `Chip`,
`models/qwenputer_walk.py` walkers) and the C stepper
(`models/qwenputer_cpu.py`, built with clang) — cannot run on Linux. This
page gives the exact commands to test them on an Apple-silicon Mac. All
commands run from the repo root (`~/ethereus-mac`). There are two packs:
`eth` (`~/.locals/ethereus/untrained9b`) and Solana (`~/.locals/ethereus/sol9b`),
same chip, different firmware snapshots.

## 0. Prereqs

```sh
xcode-select --install            # clang for the C stepper (once)
curl -LsSf astral.sh/uv/install.sh | sh   # uv (once); then restart the shell
```

Apple silicon with 32 GB RAM or more. Each pack is ~208 MB (the chip weights
are identical; only the snapshot and firmware differ); Qwen's part
(`mlx-community/Qwen3.5-9B-MLX-4bit`, 4-bit) downloads on first use (~6 GB
in the Hugging Face cache).

## 1. Install

```sh
uv sync
```

## 2. The chip packs

Copy the packs built on Linux (preferred; building needs the 37 GB
checkpoint). From the Mac, with the Linux host reachable as `linux`:

```sh
mkdir -p ~/.locals/ethereus/untrained9b ~/.locals/ethereus/sol9b
scp 'linux:/tmp/eth-pack/{chip.safetensors,chip.json,snapshot.safetensors,eth-bridge.elf}' ~/.locals/ethereus/untrained9b/
scp 'linux:/tmp/sol-pack/{chip.safetensors,chip.json,snapshot.safetensors,sol-bridge.elf}' ~/.locals/ethereus/sol9b/
ls -lh ~/.locals/ethereus/untrained9b/ ~/.locals/ethereus/sol9b/
```

Or build them from the checkpoint (needs `ETH_SRC` with
`chipformer.json`, `config.json`, `tokenizer.json`,
`model.safetensors.index.json` and the 15 shards, plus
`LOCALS_CHIPFORMER` with the chipformer package):

```sh
ETH_SRC=/path/to/q9-store-rv LOCALS_CHIPFORMER=/path/to/chipformer \
  uv run python -c "from locals.models.qwenputer import ensure_pack_eth; print(ensure_pack_eth())"
ETH_SRC=/path/to/q9-store-rv LOCALS_CHIPFORMER=/path/to/chipformer \
  uv run python -c "from locals.models.qwenputer import ensure_pack_sol; print(ensure_pack_sol())"
```

(`ensure_pack_sol` reuses the eth pack's chip when present — same checkpoint —
and only snapshots the sol firmware.)

## 3. Tokenizer for the reference machine

Exactness compares the pack against the reference machine, which needs
a Qwen3.5 tokenizer. Any Qwen3.5 `tokenizer.json` works (only the base
vocabulary is used); the MLX model's is the easiest:

```sh
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3.5-9B-MLX-4bit', allow_patterns='tokenizer.json'))"
export ETH_SRC=$(dirname $(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3.5-9B-MLX-4bit', allow_patterns='tokenizer.json'))"))
```

(`ETH_SRC` only needs `tokenizer.json` for the tests below; the pack
itself is already in `~/.locals/ethereus/untrained9b`.)

The reference machine also needs the chipformer package:

```sh
export LOCALS_CHIPFORMER=/path/to/chipformer   # experimental/riscv, with chipformer/rv/{ref,bridge,elf,spec}.py
```

## 4. Exactness: packs vs reference

```sh
uv run pytest tests/test_ethereus.py tests/test_sol.py tests/test_store_calls.py -q
```

This runs each pack's computer on the GPU (MLX `Chip` + `FusedStep`)
against `bridge.Session` on its firmware, token for token: 50 `eth`
calls (keccak, ABI encode/decode, EIP-712 type hashes, address checks),
68 `sol` calls (SHA-256, base58, amounts, u64/u128, discriminators,
borsh, instruction building, fees, rent, key/pda/seed/ATA), and the
files calls (store put, list, get in one session against
`store.StoreSession`). Expect `119 passed` (plus the categories
checks). On an M2 Pro about 15–30 minutes; the long keccak and
ed25519 runs dominate.

Single short calls, to smoke-test each setup first:

```sh
uv run pytest tests/test_ethereus.py -q -k "0xZZ"
SOL_PACK=~/.locals/ethereus/sol9b uv run pytest tests/test_sol.py -q -k "u64 and 1+2"
uv run pytest tests/test_store_calls.py -q
```

## 5. C stepper

The C stepper (`qwenputer_cpu`, built with clang) runs the same pack
on CPU cores. `scripts/ethereus_c_check.py` runs each eth call three ways from
the same state, the reference (`bridge.Session`), the GPU walker
(`Computer.step`, as served) and `CPU.run`, and requires all three to be the
same tokens. (A run must start from a token the computer predicted: after
arbitrary prompt ids it predicts nothing, `read` gives -1, and `CPU.run`
now refuses that instead of reading before its tables.) On an Apple-silicon
Mac: keccak 0xZZ 3,834 tokens, C 1.46 s, walker 3.39 s; keccak 0x6161 53,917
tokens, C 26.77 s, walker 19.87 s; all equal to the reference.
(It compares tokens, not the state buffers bit for bit.)

```sh
LOCALS_CHIPFORMER=... ETH_SRC=... PYTHONPATH=$LOCALS_CHIPFORMER \
  uv run python scripts/ethereus_c_check.py "keccak 0xZZ" "keccak 0x6161" storelist
```

(`storelist` runs the files calls — put, list, get in one store
session — the same three ways against `store.StoreSession`: it covers
the walkers' store-family decode and any-print paths.)

A full C-vs-GPU exactness test for Ethereus (the 50 calls through
`CPU.run`, asserting the reference run and bit-identical state) is
future work; `storelist` already covers the walkers' store paths
(store-family decode, any-print) through both. Qwenputer's C tests
still run:

```sh
uv run pytest tests/test_qwenputer_c.py tests/test_qwenputer.py -q
```

## 6. Serving (manual)

Serve Ethereus-9B (Qwen's part downloads on first use):

```sh
uv run locals-server --model ethereus-9b        # the eth computer
uv run locals-server --model ethereus-9b-sol    # the sol computer (same chip, sol firmware)
```

Then ask for a tool call, e.g. "What is keccak of 0x6161? Use the
eth tool." or "What is sha256 of abc? Use the sol tool." The reply's
`<tool_response>` carries the exact value.

## What was verified on Linux

Both packs were built from `/workspace/dist-store/q9-store-rv` on CPU
(the sol pack reuses the eth pack's chip bytes: same checkpoint, same
`convert`; only the snapshot differs) and reproduce `ref.py` token for
token — 50 `eth` calls, 68 `sol` calls and the files calls (store
put, list, get in one session; torch runner,
`locals/models/ethereus_cpu.py`; see `ETHEREUS_CHANGES.md`). On Linux
the 68 are 62 full runs plus 8k-token prefixes of the 6 ed25519-longest
(ATA, PDA, key, seed, names); the Mac test runs all 68 fully. The MLX
walkers and the C build only parse on Linux (their C/Python table
order is checked there); their store-family decode and any-print
paths are new and untested until steps 4–5 above pass on a Mac.
