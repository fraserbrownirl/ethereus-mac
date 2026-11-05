# docs/mac-test.md step 5: the C stepper on the Ethereus pack, from a real eth call.
# Per call: the reference run (bridge.Session), the GPU walker's run (Computer.step) and the C stepper's run
# (CPU.run) from the same state; all three must be the same tokens. "storelist" runs the files calls
# (put, list, get in one store session) the same three ways against store.StoreSession.
import copy, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
from locals.models import qwenputer as Q
from locals.models.qwenputer_walk import Computer, Device
from locals.models.qwenputer_cpu import CPU
from chipformer.rv import bridge, elf, ref
from chipformer.rv import spec as S

pack = Q.pack_dir_eth()
T = bridge.Text(str(Path(os.environ["ETH_SRC"]) / "tokenizer.json"))
m0 = ref.Machine.from_program(elf.program(str(pack / Q.ETH_FIRMWARE))); bridge.park(m0)
t0 = time.time(); chip = Q.Chip(pack); dev = Device(chip); cpu = CPU.make(dev, tempfile.mkdtemp())
assert cpu is not None, "no C compiler"
print(f"setup {time.time() - t0:.1f}s", flush=True)
prompt = T.encode("<|im_start|>user\nDo it.<|im_end|>\n<|im_start|>assistant\n")
bad = 0


def check(comp, ses, call, label):
    """One call three ways from the same state; True when C, walker and reference agree."""
    global bad
    n0 = len(ses.toks)
    ses.qwen(call)
    want = [chip.v0 + t if not S.is_text(t) else t - S.TEXT_TOKEN for t in ses.toks[n0 + len(call):]]
    first = int(comp.read(call)[-1])
    s, c, mk = (np.array(x) for x in (comp.s, comp.c, comp.mk))
    t1 = time.time(); crun, _ = cpu.run(s, c, mk, first); tc = time.time() - t1
    t1 = time.time(); grun = comp.step(first); tg = time.time() - t1
    ok_c, ok_g = [first] + list(crun) == want, [first] + list(grun) == want
    bad += not (ok_c and ok_g)
    print(f"{'PASS' if ok_c and ok_g else 'FAIL'} {label}: {len(want)} tokens; C {'=' if ok_c else '!='} ref in {tc:.2f}s, "
          f"GPU walker {'=' if ok_g else '!='} ref in {tg:.2f}s", flush=True)


for cmd in sys.argv[1:]:
    if cmd == "storelist":
        from chipformer.rv import store as ST

        ses = ST.StoreSession(T, copy.deepcopy(m0))
        ses.history(prompt)
        comp = Computer(dev)
        comp.start(prompt)
        ops = [("put", "note", "hello store"), ("list", None, None), ("get", "note", None)]
        for op, name, text in ops:
            call = ST.StoreSession.call_tokens(op, T.encode(name) if name is not None else None,
                                               T.encode(text) if text is not None else None)
            check(comp, ses, call, f"store {op}")
        continue
    call = T.encode("<tool_call>\n<function=eth>\n" f"<parameter=cmd>\n{cmd}\n</parameter>\n" "</function>\n</tool_call>")
    ses = bridge.Session(T, copy.deepcopy(m0)); ses.history(prompt)
    comp = Computer(dev); comp.start(prompt)
    check(comp, ses, call, repr(cmd))
sys.exit(1 if bad else 0)
