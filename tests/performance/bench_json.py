"""json.h vs Python's built-in json module: parse and write throughput,
timed separately (json_parse alone, json_write alone against an
already-parsed tree) the same way json.loads/json.dumps are naturally
used one at a time. Two shapes: a flat array of numbers (wide, shallow)
and an array of small record objects (narrower, one level deeper) -
picked so both land near ~1e3 and ~1e5 total scalar values, not because
either shape is special to json.h's parser."""
import ctypes
import json
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libjson.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libjson.so"))
C = ctypes.c_char_p
lib.c_json_parse_only.argtypes = [C]
lib.c_json_parse_only.restype = None
lib.c_json_build_from_text.argtypes = [C]
lib.c_json_build_from_text.restype = None
lib.c_json_write_to_buf.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.c_json_write_to_buf.restype = ctypes.c_int
lib.c_json_write_only.argtypes = []
lib.c_json_write_only.restype = None
lib.c_json_close.restype = None

REPEATS = 3


def bench(fn):
    fn()
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        runs = 0
        while time.perf_counter() - t0 < 1.0:
            fn()
            runs += 1
        ms = (time.perf_counter() - t0) / runs * 1000
        if ms < best:
            best = ms
    return best


def header(title):
    print(f"\n{title}")
    print(f"{'shape':>18}  {'ours ms':>9}  {'py json ms':>10}  {'ratio':>7}")
    print("-" * 50)


def row(label, ours, ref):
    print(f"{label:>18}  {ours:>9.3f}  {ref:>10.3f}  {ours / ref:>7.2f}")


rng = np.random.default_rng(42)


def flat_array(n):
    return [round(float(x), 6) for x in rng.standard_normal(n)]


def nested_records(m):
    return [
        {"id": i, "name": f"item_{i}", "value": round(float(rng.standard_normal()), 6),
         "active": bool(i % 2)}
        for i in range(m)
    ]


cases = [
    ("flat n=1000", flat_array(1_000)),
    ("flat n=100000", flat_array(100_000)),
    ("records m=200", nested_records(200)),
    ("records m=20000", nested_records(20_000)),
]

header("json_parse (vs json.loads)")
for label, obj in cases:
    text = json.dumps(obj).encode()
    lib.c_json_parse_only(text)  # warmup
    ours = bench(lambda: lib.c_json_parse_only(text))
    ref = bench(lambda: json.loads(text))
    row(label, ours, ref)

header("json_write (vs json.dumps; tree/object already built, write only)")
for label, obj in cases:
    text = json.dumps(obj).encode()
    lib.c_json_build_from_text(text)

    buf = ctypes.create_string_buffer(max(len(text) * 3, 4096))
    n = lib.c_json_write_to_buf(buf, len(buf))
    assert 0 < n < len(buf)
    back = json.loads(buf.value.decode())
    assert len(back) == len(obj)

    ours = bench(lib.c_json_write_only)
    ref = bench(lambda: json.dumps(obj))
    row(label, ours, ref)
    lib.c_json_close()
