"""Ethereus-9B (locals/models/qwenputer.py + ethereus_cpu.py): Qwen3.5-9B with the eth
computer (rv1-store) in 12 layers before Qwen's, which runs the model's eth tool.

With the chip pack made (``ETH_PACK`` names its folder; on a Mac
``ensure_pack_eth`` makes it from ``ETH_SRC``): the pack's computer, run on CPU
(``ethereus_cpu.CpuChip``), makes the run the authors' reference machine made
(``bridge.Session`` on ``eth-bridge.elf``) for 50 eth tool calls, token for
token, covering keccak, ABI encode and decode, EIP-712 type hashes and address
checks.
"""

import os
from pathlib import Path

import pytest

from locals.models import qwenputer as Q

ADDR1 = "0x000000000000000000000000000000000000dEaD"
ADDR2 = "0x0000000000000000000000000000000000000001"
ADDR_LOWER = "0x000000000000000000000000000000000000dead"
ADDR_ZERO = "0x0000000000000000000000000000000000000000"
ADDR_BAD = "0x1234"

# 50 eth tool calls: (cmd, category). Categories: keccak, encode, decode, eip712, address.
# eip712 calls are keccak/encode of EIP-712 type strings and field types (the firmware has no
# dedicated eip712 command; EIP-712 hashes are keccak over ABI-encoded structs).
# address calls validate addresses (valid checksummed/lower/zero, invalid short) via encode,
# decode-window refusal, and create/create2.
CALLS = [
    # keccak (valid)
    ("keccak 0x", "keccak"),
    ("keccak 'abc'", "keccak"),
    ("keccak 0x6161", "keccak"),
    # eip712 (type hashes are keccak of the canonical type strings)
    ("keccak 'EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)'", "eip712"),
    ("keccak 'Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)'", "eip712"),
    ("keccak 'Transfer(address from,address to,uint256 value)'", "eip712"),
    ("keccak 0x1901", "eip712"),
    # keccak (errors)
    ("keccak 0xZZ", "keccak"),
    ("keccak 0x123", "keccak"),
    ("keccak 0xGG", "keccak"),
    # selector (cache hits; miss goes through keccak like encode misses do)
    ("selector transfer(address,uint256)", "encode"),
    ("selector balanceOf(address)", "encode"),
    ("selector approve(address,uint256)", "encode"),
    # encode valid (hits: short; misses: full selector keccak + encoding)
    (f"encode transfer(address,uint256) {ADDR1} 100", "encode"),
    (f"encode balanceOf(address) {ADDR2}", "encode"),
    (f"encode approve(address,uint256) {ADDR1} 1000", "encode"),
    ("encode f(uint8) 255", "encode"),
    ("encode f(bytes32) 0x" + "ab" * 32, "encode"),
    ("encode f(string) hello", "encode"),
    # encode errors + address checks
    (f"encode transfer(address,uint256) {ADDR_BAD} 100", "address"),
    (f"encode transfer(address,uint256) {ADDR_LOWER} 100", "address"),
    (f"encode transfer(address,uint256) {ADDR_ZERO} 0", "address"),
    ("encode f(uint8) 256", "encode"),
    ("encode f(bool) maybe", "encode"),
    ("encode f(uint7) 1", "encode"),
    ("encode f() 1", "encode"),
    ("encode 9lives(uint256) 1", "encode"),
    ("encode f(uint256,) 1", "encode"),
    # decode (32-byte values exceed the 80-token window: the bridge refuses with
    # "too long", which the computer must reproduce; short ones fit and run)
    ("decode uint8 0x1234", "decode"),
    ("decode nope", "decode"),
    ("decode uint8 0x" + "00" * 31 + "ff", "decode"),
    ("decode bool 0x" + "00" * 31 + "01", "decode"),
    ("decode address 0x" + "00" * 12 + ADDR1[2:], "address"),
    ("decode uint256 0x" + "00" * 31 + "2a", "decode"),
    ("decode bytes32 0x" + "11" * 32, "decode"),
    ("decode uint8 0xZZ", "decode"),
    ("decode address 0x1234", "address"),
    # u256 (exact 256-bit arithmetic the eth tool also runs)
    ("u256 1+2", "encode"),
    ("u256 7 / 0", "encode"),
    ("u256 255 hex", "encode"),
    ("u256 10 % 3", "encode"),
    ("u256 (1+2)*3", "encode"),
    ("u256 0 hex", "encode"),
    # create / create2 (address derivation; create2's inline hashes exceed the window)
    (f"create {ADDR1} 7", "address"),
    (f"create {ADDR_BAD} 7", "address"),
    (f"create {ADDR1} -1", "address"),
    (f"create2 {ADDR1} 1 0x" + "00" * 32, "address"),
    ("foo", "encode"),
    ("keccak 'hello world'", "keccak"),
    ("encode f(bool) true", "encode"),
]
assert len(CALLS) == 50, len(CALLS)


def _pack():
    p = os.environ.get("ETH_PACK")
    return Path(p) if p else Q.pack_dir_eth()


@pytest.fixture(scope="module")
def chip():
    pack = _pack()
    if not (pack / "chip.safetensors").exists() or not (pack / "snapshot.safetensors").exists():
        pytest.skip(f"no Ethereus-9B chip pack in {pack} (ETH_PACK names it)")
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
    fw = pack / Q.ETH_FIRMWARE
    if not fw.exists():
        pytest.skip(f"no firmware {fw} in the pack")
    # tokenizer from the pack's source checkpoint if recorded, else ETH_SRC
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
    return ("<tool_call>\n<function=eth>\n"
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
    assert {"keccak", "encode", "decode", "eip712", "address"} <= cats, cats


@pytest.mark.parametrize("cmd,cat", CALLS)
def test_pack_reproduces_reference(chip, session, cmd, cat):
    """The pack's run for one eth tool call is the reference machine's, token for token."""
    from chipformer.rv import bridge
    from chipformer.rv import spec as S

    T, m0 = session
    # fresh parked machine per call (ref machines are cheap; the chip state is fresh too)
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
    # the answer the echo carries (a spot check that the run is meaningful, not just matching)
    echo = T.decode_bytes([t for t in want if t < chip.v0]).decode(errors="replace")
    assert "<tool_response>" in echo, f"{cmd}: no tool response in echo"
