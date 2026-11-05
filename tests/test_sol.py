"""Ethereus-9B Solana (locals/models/qwenputer.py + ethereus_cpu.py): the same checkpoint's
chip with the sol firmware as the parked snapshot, running the model's sol tool.

With the Solana pack made (``SOL_PACK`` names its folder; on a Mac ``ensure_pack_sol`` makes it
from the eth pack or ``ETH_SRC``): the pack's computer, run on CPU (``ethereus_cpu.CpuChip``),
makes the run the reference machine made (``bridge.Session`` on ``sol-bridge.elf``) for 68 sol
tool calls, token for token, covering SHA-256, base58, amounts, u64/u128, discriminators
(TransferChecked), borsh encode/decode, instruction building (system, token
TransferChecked, memo, compute, ATA), fees, rent, key/pda/seed/ATA derivation.
"""

import os
from pathlib import Path

import pytest

from locals.models import qwenputer as Q

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SYS = "11111111111111111111111111111111"

# 68 sol tool calls: (cmd, category).
CALLS = [
    # sha256
    ("sha256 0x", "sha256"),
    ("sha256 'abc'", "sha256"),
    ("sha256 0x6161", "sha256"),
    ("sha256 'hello world'", "sha256"),
    ("sha256 0xZZ", "sha256"),
    ("sha256 ''", "sha256"),
    ("sha256 'solana'", "sha256"),
    ("sha256 0x123", "sha256"),
    # base58
    ("b58 enc 0x68656c6c6f", "base58"),
    ("b58 enc 0x", "base58"),
    ("b58 enc 0x00", "base58"),
    ("b58 enc 0xZZ", "base58"),
    ("b58 dec Cn8eVZg", "base58"),
    ("b58 dec 1", "base58"),
    ("b58 dec !!!!", "base58"),
    ("b58 dec 0OIl", "base58"),
    # amounts
    ("amount 1.5 6", "amounts"),
    ("amount 0.000001 6", "amounts"),
    ("amount 1 9", "amounts"),
    ("ui 1500000 6", "amounts"),
    ("ui 1 9", "amounts"),
    ("sol 1000000000", "amounts"),
    ("lamports 0.5", "amounts"),
    ("sol abc", "amounts"),
    ("amount abc 6", "amounts"),
    ("lamports 1", "amounts"),
    # u64/u128
    ("u64 1+2", "u64"),
    ("u64 18446744073709551615", "u64"),
    ("u128 340282366920938463463374607431768211455", "u64"),
    ("u64 7 / 0", "u64"),
    # discriminators (TransferChecked encode side; token ix building is bad op in this firmware)
    ("disc ix transfer_checked", "disc"),
    ("disc ix TransferChecked", "disc"),
    ("disc ix transfer", "disc"),
    ("disc account Mint", "disc"),
    ("disc event Transfer", "disc"),
    # borsh encode/decode (instruction-data fields; the u64 decodes are the
    # amount fields of the transfer_checked data below: 0x0c<u64le><dec>)
    ("borsh enc u8 255", "borsh"),
    ("borsh enc u32 16909060", "borsh"),
    ("borsh enc u64 100", "borsh"),
    ("borsh enc bool true", "borsh"),
    ("borsh dec u32 0x02000000", "borsh"),
    ("borsh dec u64 0x0100000000000000", "borsh"),
    ("borsh dec u64 0x68866102c4070000", "borsh"),
    ("borsh dec u8 0xff", "borsh"),
    ("borsh enc str hello", "borsh"),
    # instruction building (arg orders per the sol oracle: transfer_checked
    # takes source dest mint owner amount decimals; memo takes message signer)
    ("ix system.transfer system usdc 1", "ix"),
    ("ix system.transfer system usdc 0", "ix"),
    ("ix system.transfer a b 1", "ix"),
    ("ix token.transfer_checked system system usdc system 1 6", "ix"),
    ("ix token.transfer_checked system system usdc system 8538434930280 9", "ix"),
    ("ix memo hello system", "ix"),
    ("ix compute.set_compute_unit_limit 200000", "ix"),
    ("ix system.create_account system usdc 100 165 token", "ix"),
    ("ix token.nope a b c", "ix"),
    # fees and rent
    ("fee 2", "fee"),
    ("fee 2 200000 5000", "fee"),
    ("rent 165", "rent"),
    ("rent 10240", "rent"),
    # key/pda/seed/ATA/names (valid ones are ed25519-long; errors and refusals are short)
    ("key usdc", "key"),
    ("key nota*key", "key"),
    ("pda system str:vault", "pda"),
    ("pda badprog s", "pda"),
    ("seed system hello usdc", "seed"),
    ("seed a b c", "seed"),
    ("ata system usdc", "ata"),
    ("ata system usdc token22", "ata"),
    ("ata short mint", "ata"),
    (f"ata {SYS} {USDC}", "ata"),
    ("names", "names"),
]
assert len(CALLS) == 68, len(CALLS)


def _pack():
    p = os.environ.get("SOL_PACK")
    return Path(p) if p else Q.pack_dir_sol()


@pytest.fixture(scope="module")
def chip():
    pack = _pack()
    if not (pack / "chip.safetensors").exists() or not (pack / "snapshot.safetensors").exists():
        pytest.skip(f"no Solana chip pack in {pack} (SOL_PACK names it)")
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
    from chipformer.rv import bridge, elf, ref

    pack = _pack()
    fw = pack / Q.SOL_FIRMWARE
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
    return T, m


def _tool_call_text(cmd: str) -> str:
    return ("<tool_call>\n<function=sol>\n"
            f"<parameter=cmd>\n{cmd}\n</parameter>\n"
            "</function>\n</tool_call>")


def _run(chip, st, first: int, max_tokens: int):
    """Free-run from ``first`` until Qwen predicts (both backends)."""
    if type(chip).__name__ == "CpuChip":  # (Linux)
        return chip.run(st, first, max_tokens=max_tokens)
    # (MLX Chip on the Mac: stepped on the GPU, as served)
    return Q.FusedStep(chip).run(st, [first], max_tokens=max_tokens)[0]


def test_categories_covered():
    cats = {c for _, c in CALLS}
    assert {"sha256", "base58", "amounts", "disc", "borsh", "ix", "fee", "rent", "ata"} <= cats, cats


@pytest.mark.parametrize("cmd,cat", CALLS)
def test_pack_reproduces_reference(chip, session, cmd, cat):
    """The pack's run for one sol tool call is the reference machine's, token for token."""
    from chipformer.rv import bridge
    from chipformer.rv import spec as S

    T, m0 = session
    import copy

    m = copy.deepcopy(m0)
    ses = bridge.Session(T, m)
    prompt = T.encode("<|im_start|>user\nDo it.<|im_end|>\n<|im_start|>assistant\n")
    call = T.encode(_tool_call_text(cmd))
    ses.history(prompt)
    n0 = len(ses.toks)
    ses.qwen(call)
    want = [chip.v0 + t if not S.is_text(t) else t - S.TEXT_TOKEN for t in ses.toks[n0 + len(call):]]
    st = chip.new_state(1)
    chip(chip.embed([prompt]), st)
    chip.inject(st)
    preds = chip.read(call, st)
    assert int(preds[-1]) == want[0], f"{cmd}: first {preds[-1]} != {want[0]}"
    assert all(int(p) == -1 for p in preds[:-1]), f"{cmd}: a call position predicts"
    got = [int(preds[-1])] + _run(chip, st, int(preds[-1]), max_tokens=len(want) + 10)
    assert got == want, f"{cmd}: first diff at {next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)}"
    echo = T.decode_bytes([t for t in want if t < chip.v0]).decode(errors="replace")
    assert "<tool_response>" in echo, f"{cmd}: no tool response in echo"
