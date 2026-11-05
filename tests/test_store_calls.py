"""Ethereus-9B files calls: the pack's computer answers the object store's put/list/get
token for token, as the reference machine with the store (``store.StoreSession``) does.

The store's answers hold its chip tokens (SP/SN/SO/SL/SG: the walkers decode their
families) and any-printed text (the id code names the token, e.g. 547 "ok"): without
either, a files call stops early. The three calls run in one session, the store
accumulating as served.
"""

import os
from pathlib import Path

import pytest

from locals.models import qwenputer as Q

OPS = [
    ("put", "note", "hello store"),
    ("list", None, None),
    ("get", "note", None),
]


def _pack():
    p = os.environ.get("ETH_PACK")
    return Path(p) if p else Q.pack_dir_eth()


@pytest.fixture(scope="module")
def chip():
    pack = _pack()
    if not (pack / "chip.safetensors").exists() or not (pack / "snapshot.safetensors").exists():
        pytest.skip(f"no Ethereus chip pack in {pack} (ETH_PACK names it)")
    if Q.HAVE_MLX:  # (the Mac: the MLX computer, as served)
        return Q.Chip(pack)
    pytest.importorskip("torch")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from locals.models.ethereus_cpu import CpuChip

    return CpuChip(pack)


@pytest.fixture(scope="module")
def session():
    if Q.chipformer() is None:
        pytest.skip("exactness needs the chipformer package (LOCALS_CHIPFORMER)")
    from chipformer.rv import bridge, elf, ref, store

    pack = _pack()
    fw = pack / Q.ETH_FIRMWARE
    if not fw.exists():
        pytest.skip(f"no firmware {fw} in the pack")
    import json

    info = json.loads((pack / "chip.json").read_text())
    src = Path(os.environ.get("ETH_SRC", info.get("source", "")))
    tj = src / "tokenizer.json" if src and (src / "tokenizer.json").exists() else None
    if tj is None:
        pytest.skip("exactness needs the checkpoint tokenizer (ETH_SRC)")
    T = bridge.Text(str(tj))
    m = ref.Machine.from_program(elf.program(str(fw)))
    bridge.park(m)
    return T, store.StoreSession(T, m)


def _run(chip, st, first: int, max_tokens: int):
    """Free-run from ``first`` until Qwen predicts (both backends)."""
    if type(chip).__name__ == "CpuChip":  # (Linux)
        return chip.run(st, first, max_tokens=max_tokens)
    # (MLX Chip on the Mac: stepped on the GPU, as served)
    return Q.FusedStep(chip).run(st, [first], max_tokens=max_tokens)[0]


def test_files_calls_reproduce_reference(chip, session):
    """put, list and get in one session: the pack's runs are the store's, token for token."""
    from chipformer.rv import bridge
    from chipformer.rv import spec as S
    from chipformer.rv import store as ST

    T, ses = session
    prompt = T.encode("<|im_start|>user\nDo it.<|im_end|>\n<|im_start|>assistant\n")
    ses.history(prompt)
    st = chip.new_state(1)
    chip(chip.embed([prompt]), st)
    chip.inject(st)
    wants = {}
    for op, name, text in OPS:
        call = ST.StoreSession.call_tokens(op,
                                           T.encode(name) if name is not None else None,
                                           T.encode(text) if text is not None else None)
        n0 = len(ses.toks)
        ses.qwen(call)
        want = [chip.v0 + t if not S.is_text(t) else t - S.TEXT_TOKEN for t in ses.toks[n0 + len(call):]]
        wants[op] = want
        preds = chip.read(call, st)
        assert int(preds[-1]) == want[0], f"{op}: first {preds[-1]} != {want[0]}"
        assert all(int(p) == -1 for p in preds[:-1]), f"{op}: a call position predicts"
        got = [int(preds[-1])] + _run(chip, st, int(preds[-1]), max_tokens=len(want) + 10)
        assert got == want, f"{op}: first diff at {next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)}"
        echo = T.decode_bytes([t for t in want if t < chip.v0]).decode(errors="replace")
        assert "<tool_response>" in echo, f"{op}: no tool response in echo"
    # the test exercises both store paths: a store-family chip (family decode) and the
    # any-printed "ok" (id code), without which the put run stops after its header.
    assert any(t >= chip.v0 + Q.CHIP_VOCAB for t in wants["put"]), "no store token in the put run"
    assert 547 in wants["put"], "no any-printed ok in the put run"
