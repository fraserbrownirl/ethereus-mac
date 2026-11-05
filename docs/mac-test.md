# Testing Ethereus-9B on a Mac

The Mac-only parts — the MLX computer (`models/qwenputer.py` `Chip`,
`models/qwenputer_walk.py` walkers) and the C stepper
(`models/qwenputer_cpu.py`, built with clang) — cannot run on Linux. This
page gives the exact commands to test them on an Apple-silicon Mac. All
commands run from the repo root (`~/ethereus-mac`).

## 0. Prereqs

```sh
xcode-select --install            # clang for the C stepper (once)
curl -LsSf astral.sh/uv/install.sh | sh   # uv (once); then restart the shell
```

Apple silicon with 32 GB RAM or more. The Ethereus-9B pack is ~208 MB;
Qwen's part (`mlx-community/Qwen3.5-9B-MLX-4bit`, 4-bit) downloads on
first use (~6 GB in the Hugging Face cache).

## 1. Install

```sh
uv sync
```

## 2. The chip pack

Copy the pack built on Linux (preferred; building needs the 37 GB
checkpoint). From the Mac, with the Linux host reachable as `linux`:

```sh
mkdir -p ~/.locals/ethereus/untrained9b
scp 'linux:/tmp/eth-pack/{chip.safetensors,chip.json,snapshot.safetensors,eth-bridge.elf}' ~/.locals/ethereus/untrained9b/
ls -lh ~/.locals/ethereus/untrained9b/
```

Or build it from the checkpoint (needs `ETH_SRC` with
`chipformer.json`, `config.json`, `tokenizer.json`,
`model.safetensors.index.json` and the 15 shards, plus
`LOCALS_CHIPFORMER` with the chipformer package):

```sh
ETH_SRC=/path/to/q9-store-rv LOCALS_CHIPFORMER=/path/to/chipformer \
  uv run python -c "from locals.models.qwenputer import ensure_pack_eth; print(ensure_pack_eth())"
```

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

## 4. Exactness: pack vs reference, 50 eth tool calls

```sh
uv run pytest tests/test_ethereus.py -q
```

This runs the pack's computer on the GPU (MLX `Chip` + `FusedStep`)
against `bridge.Session` on `eth-bridge.elf`, token for token, on 50
`eth` calls (keccak, ABI encode/decode, EIP-712 type hashes, address
checks). Expect `50 passed` (plus the categories check). On an M2 Pro
about 10–20 minutes; the long keccak runs dominate.

A single short call, to smoke-test the setup first:

```sh
uv run pytest tests/test_ethereus.py -q -k "0xZZ"
```

## 5. C stepper

The C stepper (`qwenputer_cpu`, built with clang) runs the same pack
on CPU cores. Its Ethereus check mirrors Qwenputer's: a run on a core
is the GPU's, bit for bit (state, convolutions, masks), on host
copies and in place on the device's buffers:

```sh
uv run pytest tests/test_ethereus.py -q -k "0xZZ"   # (uses the GPU path; the C check below is separate)
uv run python - <<'EOF'
import numpy as np
from locals.models import qwenputer as Q
from locals.models.qwenputer_walk import Computer, Device
from locals.models.qwenputer_cpu import CPU, shared
import tempfile
chip = Q.Chip(Q.pack_dir_eth())
dev = Device(chip)
comp = Computer(dev)
comp.start([1, 2, 3])
first = int(comp.read([4, 5, 6])[-1])
s, c, mk = (np.array(x) for x in (comp.s, comp.c, comp.mk))
cpu = CPU.make(dev, tempfile.mkdtemp())
assert cpu is not None, "no C compiler (xcode-select --install)"
rest, _ = cpu.run(s.copy(), c.copy(), mk.copy(), first)
print("C stepper ran", len(rest), "tokens")
EOF
```

A full C-vs-GPU exactness test for Ethereus (the 50 calls through
`CPU.run`, asserting the reference run and bit-identical state) is
future work; the walkers' store embedding (`IDBOF`/`DET_IDS` tables)
and the ballast count (`PRE` from the pack) are new in this fork and
worth covering there. Qwenputer's C tests still run:

```sh
uv run pytest tests/test_qwenputer_c.py tests/test_qwenputer.py -q
```

## 6. Serving (manual)

Serve Ethereus-9B (Qwen's part downloads on first use):

```sh
uv run locals-server --model ethereus-9b
```

Then ask for an `eth` tool call, e.g. "What is keccak of 0x6161?
Use the eth tool." The reply's `<tool_response>` carries the exact
value.

## What was verified on Linux

The pack was built from `/workspace/dist-store/q9-store-rv` on CPU
and reproduces `ref.py` token for token on all 50 calls (torch runner,
`locals/models/ethereus_cpu.py`; see `ETHEREUS_CHANGES.md`). The MLX
walkers and the C build only parse on Linux; their store support
(`rv1-store` text rows, 23 ballast dims) is new and untested until
step 4 above passes on a Mac.
