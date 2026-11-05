"""Ethereus-9B computer on CPU (torch): the chip pack run without MLX.

Loads ``chip.safetensors``/``chip.json`` (made by ``qwenputer.convert``) with
safetensors/torch and runs the computer layers in float32, following
``qwenputer.Chip`` (which follows ``modeling_qwen3_5``) step by step. The
recurrence is the reference gated-delta update (transformers'
``torch_recurrent_gated_delta_rule``); the snapshot is stored in MLX layout
(value rows, key columns) and transposed to torch layout (key rows, value
columns) on load.

Used for exactness on Linux, where MLX is unavailable: the pack must reproduce
the reference machine token for token (``tests/test_ethereus.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from safetensors import safe_open

HEAD = 128
CONV = 4
EPS = 1e-6


def _silu_f32(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))


def _softplus_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20, x, torch.log1p(torch.exp(torch.minimum(x, torch.tensor(20.0)))))


def _rms_f32(x: torch.Tensor, n: int) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) / n + EPS)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


class CpuState:
    """Per-layer conv (B,3,C) bf16 and recurrent (B,nh,128,128) f32 in torch layout (key rows, value cols)."""

    def __init__(self, conv: list, rec: list):
        self.conv, self.rec = conv, rec


class CpuChip:
    """A chip pack's computer on CPU."""

    def __init__(self, path):
        self._long_cache: Dict[str, torch.Tensor] = {}
        self._float_cache: Dict[str, torch.Tensor] = {}
        from . import qwenputer as Q

        path = Path(path)
        self.info = json.loads((path / "chip.json").read_text())
        with safe_open(str(path / "chip.safetensors"), framework="pt", device="cpu") as f:
            self.W = {k: f.get_tensor(k) for k in f.keys()}
        self.n_layers = self.info["layers"]
        self.v0 = self.info["chip_token_offset"]
        self.H = self.info["hidden"]
        self.code = self.W["dims"]  # placeholder, replaced below
        self.code_idx = list(self.info["code"])
        self.nfam, self.npay = self.info["nfam"], self.info["npay"]
        self.printable = self.info["printable"]
        self.out_names = self.info.get("out_names", Q.OUT_NAMES)
        self.has_store = bool(self.info.get("store_families"))
        snap = path / "snapshot.safetensors"
        self.snapshot = None
        if snap.exists():
            with safe_open(str(snap), framework="pt", device="cpu") as f:
                # MLX layout (nh,V,K) -> torch layout (nh,K,V)
                self.snapshot = {k: f.get_tensor(k).float().transpose(-1, -2).contiguous() for k in f.keys()}
        # embedding tables on host (f32 numpy)
        e: Dict[str, np.ndarray] = {}
        for k, v in self.W.items():
            if not k.startswith("emb."):
                continue
            name = k[4:]
            if v.dtype == torch.bfloat16:
                e[name] = v.float().numpy()
            else:
                e[name] = np.array(v)
        self._emb = e
        self._special = {int(t): i for i, t in enumerate(e["special_ids"])}
        self._explicit = {int(t): i for i, t in enumerate(e["explicit_ids"])}
        self._det = {int(t): i for i, t in enumerate(e["det_ids"])} if "det_ids" in e else {}
        cvocab = self.info.get("chip_vocab", Q.CHIP_VOCAB)
        self._fam_of = np.zeros(cvocab, np.int32)
        for i, (base, size) in enumerate(Q.FAMILIES.values()):
            self._fam_of[base:base + size] = i
        self._fam_base = np.array([b for b, _ in Q.FAMILIES.values()])
        self._Q = Q
        self._cache: Dict[int, np.ndarray] = {}

    def _row(self, t: int) -> np.ndarray:
        e = self._emb
        if t < self.v0:
            i = self._special.get(t)
            if i is not None:
                return e["special_rows"][i]
            j = self._det.get(t)
            if j is not None:
                return e["det_rows"][j]
            r = e["plain"].copy()
            r[e["bitcols"].astype(np.int64)] = np.unpackbits(e["bits"][t])[: len(e["bitcols"])]
            if "id_xb" in e and t < 248044:
                for k, c in enumerate(e["id_xb"].astype(np.int64)):
                    r[c] = (t >> k) & 1
                for k, c in enumerate(e["id_yc"].astype(np.int64)):
                    r[c] = 128.0 * (2 * ((t >> k) & 1) - 1)
            return r
        c = t - self.v0
        i = self._explicit.get(c)
        if i is not None:
            return e["explicit_rows"][i]
        f = self._fam_of[c]
        p = c - self._fam_base[f]
        bits = (p >> np.arange(16)) & 1
        return e["fam_base"][f] + bits.astype(np.float32) @ e["fam_delta"][f]

    def embed(self, ids: List[List[int]]) -> torch.Tensor:
        rows = []
        for seq in ids:
            for t in seq:
                r = self._cache.get(t)
                if r is None:
                    r = self._row(t)
                    if len(self._cache) < 1 << 16:
                        self._cache[t] = r
                rows.append(r)
        x = np.stack(rows).reshape(len(ids), len(ids[0]), -1)
        return torch.from_numpy(x).to(torch.bfloat16)

    def new_state(self, B: int) -> CpuState:
        conv, rec = [], []
        for L in range(self.n_layers):
            w = self.W.get(f"L{L}.conv")
            if w is None:
                conv.append(None)
                rec.append(None)
                continue
            nh = self.W[f"L{L}.kmap"].shape[0]
            C = w.shape[0]
            conv.append(torch.zeros((B, CONV - 1, C), dtype=torch.bfloat16))
            rec.append(torch.zeros((B, nh, HEAD, HEAD), dtype=torch.float32))
        return CpuState(conv, rec)

    def inject(self, st: CpuState, rows: Optional[List[int]] = None) -> None:
        if self.snapshot is None:
            raise RuntimeError("this chip pack has no firmware snapshot")
        for L in range(self.n_layers):
            if st.rec[L] is None:
                continue
            s = self.snapshot[f"L{L}"]
            B = st.rec[L].shape[0]
            if rows is None or len(rows) == B:
                st.rec[L] = s.unsqueeze(0).expand(B, *s.shape).clone()
            else:
                want = set(rows)
                for i in range(B):
                    if i in want:
                        st.rec[L][i] = s

    def _long(self, k: str) -> torch.Tensor:
        t = self._long_cache.get(k)
        if t is None:
            t = self.W[k].long()
            self._long_cache[k] = t
        return t

    def _float(self, k: str) -> torch.Tensor:
        t = self._float_cache.get(k)
        if t is None:
            t = self.W[k].float()
            self._float_cache[k] = t
        return t

    def _lin(self, L: int, k: str, h: torch.Tensor) -> torch.Tensor:
        idx = self._long(f"L{L}.{k}.in")
        w = self._float(f"L{L}.{k}.w")
        return (h.float()[:, :, idx] @ w.T).to(torch.bfloat16)

    def _add(self, L: int, k: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        to = self._long(f"L{L}.{k}.to")
        out = x.clone()
        out[:, :, to] = (x[:, :, to].float() + y.float()).to(torch.bfloat16)
        return out

    def _attn(self, L: int, x: torch.Tensor, st: CpuState) -> torch.Tensor:
        W = self.W
        if f"L{L}.conv" not in W:
            return x
        B, T, _ = x.shape
        h = _bf16(_rms_f32(x.float(), self.H))
        qkv = self._lin(L, "qkv", h)
        nh = W[f"L{L}.kmap"].shape[0]
        if f"L{L}.z.w" in W:
            z = self._lin(L, "z", h)
        else:
            z = torch.zeros((B, T, nh * HEAD), dtype=torch.bfloat16)
        if f"L{L}.b.w" in W:
            b = self._lin(L, "b", h)
        else:
            b = torch.zeros((B, T, nh), dtype=torch.bfloat16)
        if f"L{L}.a.w" in W:
            a = self._lin(L, "a", h)
        else:
            a = torch.zeros((B, T, nh), dtype=torch.bfloat16)
        # conv (f32, bf16 out, silu)
        seq = torch.cat([st.conv[L], qkv], dim=1)
        st.conv[L] = seq[:, -(CONV - 1):].clone()
        w = self._float(f"L{L}.conv")
        sf = seq.float()
        conv = sum(sf[:, j:j + T] * w[:, j] for j in range(CONV)).to(torch.bfloat16)
        conv = _bf16(_silu_f32(conv.float()))
        kmap = self._long(f"L{L}.kmap")
        C = conv.shape[-1]
        nk = (C - nh * HEAD) // (2 * HEAD)
        q = conv[:, :, : nk * HEAD].reshape(B, T, nk, HEAD).float()
        k = conv[:, :, nk * HEAD: 2 * nk * HEAD].reshape(B, T, nk, HEAD).float()
        v = conv[:, :, 2 * nk * HEAD:].reshape(B, T, nh, HEAD).float()
        q, k = q[:, :, kmap], k[:, :, kmap]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q / (HEAD ** 0.5)
        beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
        g = -torch.exp(self._float(f"L{L}.A_log")) * _softplus_f32(a.float() + self._float(f"L{L}.dt_bias"))
        # recurrence, one token at a time (torch layout: S is (B,nh,K,V))
        S = st.rec[L]
        outs = []
        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            gt = g[:, t] if g.dim() == 3 else g
            decay = torch.exp(gt).view(B, nh, 1, 1)
            S = S * decay
            bt = beta[:, t] if beta.dim() == 3 else beta
            kv = (S * kt.unsqueeze(-1)).sum(-2)
            delta = (vt - kv) * bt.unsqueeze(-1)
            S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((S * qt.unsqueeze(-1)).sum(-2))
        y = torch.stack(outs, dim=1).to(torch.bfloat16).float()
        st.rec[L] = S
        n = (y * torch.rsqrt((y * y).mean(-1, keepdim=True) + EPS)).to(torch.bfloat16)
        n = self._float(f"L{L}.norm") * n.float()
        n = (n * _silu_f32(z.reshape(B, T, nh, HEAD).float())).to(torch.bfloat16)
        return self._add(L, "out", x, (n.reshape(B, T, nh * HEAD).float() @ self._float(f"L{L}.out.w").T).to(torch.bfloat16))

    def _mlp(self, L: int, x: torch.Tensor) -> torch.Tensor:
        if f"L{L}.gate.w" not in self.W:
            return x
        h = _bf16(_rms_f32(x.float(), self.H))
        g = _bf16(_silu_f32(self._lin(L, "gate", h).float()))
        up = self._lin(L, "up", h)
        return self._add(L, "down", x, ((g.float() * up.float()) @ self._float(f"L{L}.down.w").T).to(torch.bfloat16))

    def __call__(self, x: torch.Tensor, st: CpuState) -> torch.Tensor:
        with torch.no_grad():
            for L in range(self.n_layers):
                x = self._attn(L, x, st)
                x = self._mlp(L, x)
        return x

    def predict(self, h: torch.Tensor) -> np.ndarray:
        code = h.float().numpy()[:, :, self.code_idx]
        set_ = (code != 0).any(-1)
        bits = (code > 0).astype(np.int64)
        fid = (bits[..., : self.nfam] << np.arange(self.nfam)).sum(-1)
        pay = (bits[..., self.nfam:] << np.arange(self.npay)).sum(-1)
        out = np.full(code.shape[:-1], -1, np.int64)
        for i in zip(*np.nonzero(set_)):
            out[i] = self.decode(int(fid[i]), int(pay[i]))
        return out

    def decode(self, fid: int, p: int) -> int:
        Q = self._Q
        if not 0 <= fid < len(self.out_names):
            return -2
        name = self.out_names[fid]
        if name == "Q":
            return self.printable[p] if p < len(self.printable) else -2
        if name in Q.CONTROL_NAMES:
            return self.v0 + Q.CTRL + Q.CONTROL_NAMES.index(name)
        fams = {**Q.FAMILIES, **Q.STORE_FAMILIES} if self.has_store else Q.FAMILIES
        base, size = fams[name]
        return self.v0 + base + p if p < size else -2

    def name(self, t: int) -> str:
        Q = self._Q
        c = t - self.v0
        if self.has_store and c >= Q.CHIP_VOCAB:
            for n, (b, s) in Q.STORE_FAMILIES.items():
                if b <= c < b + s:
                    return f"{n}({c - b})"
            return f"?({c})"
        if c >= Q.CTRL:
            return Q.CONTROL_NAMES[c - Q.CTRL]
        f = int(self._fam_of[c])
        return f"{Q.FAMILY_SIZES[f][0]}({c - self._fam_base[f]})"

    def read(self, toks: List[int], st: CpuState) -> np.ndarray:
        """Teacher-forced predictions for known tokens (one block)."""
        h = self(self.embed([toks]), st)
        return self.predict(h)[0]

    def run(self, st: CpuState, first: int, max_tokens: int = 60000) -> List[int]:
        """Free-run from ``first`` until Qwen predicts (or a failure code)."""
        out = []
        cur = first
        for _ in range(max_tokens):
            h = self(self.embed([[cur]]), st)
            nxt = int(self.predict(h)[0, 0])
            if nxt < 0:
                if nxt != -1:
                    out.append(nxt)
                break
            out.append(nxt)
            if nxt < self.v0:
                # echoed text token: the run continues (BYTE echoes are text)
                pass
            cur = nxt
        return out
