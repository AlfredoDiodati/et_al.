"""Checks frame/npz.h against a live numpy, in both directions.

Every other suite under tests/correctness/ runs without Python, and this
one deliberately does not join make test for that reason - numpy is a
development-tier dependency (see README's Installation tiers), and the
shipped suite has to build and pass with OpenBLAS and a C compiler alone.
What test_npz.c can do instead is embed archives numpy produced once, and
that is what it does. The gap this file closes is that an embedded
fixture proves the reader against one numpy version at one moment; it
cannot prove the writer against numpy at all, since nothing in a
Python-free suite can open a file with np.load.

    make test-npz-python

Four things are checked, and the first is the one that only works here:

  1. numpy reads what this library writes. df_write_npz's archive is
     opened with np.load, and every value, dtype, shape, key and key
     order is compared against the C side's own dump of the same frame.
     Numeric values are compared as raw IEEE754 bits, not as decimals.
  2. This library reads what numpy writes. np.savez and
     np.savez_compressed archives - 1D and 2D numeric members, a unicode
     string member, a row-label member - are read by df_read_npz and its
     dump compared against the arrays numpy started from.
  3. The archives embedded in test_npz.c are real. They are extracted
     from the C source, handed to np.load, and required to hold the
     values that file asserts, so an embedded fixture cannot drift into
     being merely self-consistent.
  4. zlib accepts what gzip.h's DEFLATE encoder produces. The C suite can
     only round-trip the encoder through the decoder beside it, which
     proves the two halves agree but not that what they agree on is
     DEFLATE. This is also where the ratio-against-zlib figures in
     docs/GZIP_DOCUMENTATION.md come from.
"""

import os
import random
import re
import subprocess
import sys
import tempfile
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DRIVER = os.path.join(HERE, "npz_python_interop")

failures = []


def check(condition, message):
    if condition:
        return
    failures.append(message)
    print("  FAIL " + message)


def run_driver(*args):
    result = subprocess.run([DRIVER, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"driver failed: {args}\n{result.stdout}{result.stderr}")
    return result.stdout


def parse_dump(path):
    """The driver's flat dump into something comparable with numpy."""
    frame = {"columns": [], "numeric": {}, "strings": {}, "row_names": None}
    for line in open(path, encoding="utf-8"):
        kind, _, rest = line.rstrip("\n").partition(" ")
        if kind == "precision":
            frame["precision"] = rest
        elif kind == "rows":
            frame["rows"] = int(rest)
        elif kind == "cols":
            frame["cols"] = int(rest)
        elif kind == "rownames":
            if rest == "present":
                frame["row_names"] = []
        elif kind == "rowname":
            _, _, value = rest.partition(" ")
            frame["row_names"].append(value)
        elif kind == "col":
            _, coltype, name = rest.split(" ", 2)
            frame["columns"].append((name, coltype))
            (frame["numeric"] if coltype == "numeric" else frame["strings"])[name] = []
        elif kind == "num":
            name, _, rest2 = rest.partition(" ")
            frame["numeric"][name].append(int(rest2.split(" ", 1)[1], 16))
        elif kind == "str":
            name, _, rest2 = rest.partition(" ")
            frame["strings"][name].append(rest2.split(" ", 1)[1])
    return frame


def bits_of(array):
    """Raw IEEE754 bits of each element, as Python ints."""
    width = "u4" if array.dtype == np.float32 else "u8"
    return [int(v) for v in array.view(width).ravel()]


def check_numpy_reads_what_c_wrote(tmp, dtype, compressed=False):
    writer = "df_write_npz_compressed" if compressed else "df_write_npz"
    print(f"numpy reads what {writer} wrote")
    npz = os.path.join(tmp, "from_c.npz")
    dump = os.path.join(tmp, "from_c.txt")
    run_driver("write_compressed" if compressed else "write", npz, dump)
    written = parse_dump(dump)

    with zipfile.ZipFile(npz) as archive:
        check(archive.testzip() is None, "archive fails zipfile's own CRC check")
        members = [info.filename for info in archive.infolist()]
        methods = {info.compress_type for info in archive.infolist()}
        if compressed:
            # a member that does not actually shrink is stored instead, so both
            # are legal here; what must not happen is nothing being deflated
            check(methods <= {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                  f"unexpected compression methods {methods}")
            check(zipfile.ZIP_DEFLATED in methods,
                  "no member was deflated, so the encoder never ran")
        else:
            check(methods == {zipfile.ZIP_STORED},
                  "members are not all stored, which is what np.savez emits")

    loaded = np.load(npz)
    expected_members = [name + ".npy" for name, _ in written["columns"]] + ["__row_names__.npy"]
    check(members == expected_members,
          f"member order {members} does not match declaration order {expected_members}")
    check(list(loaded.files) == [name for name, _ in written["columns"]] + ["__row_names__"],
          f"np.load keys {list(loaded.files)} do not match the frame's columns")

    for name, coltype in written["columns"]:
        array = loaded[name]
        check(array.shape == (written["rows"],), f"{name}: numpy shape {array.shape} is not 1D of {written['rows']}")
        if coltype == "numeric":
            check(array.dtype == dtype, f"{name}: numpy dtype {array.dtype}, expected {dtype}")
            check(bits_of(array) == written["numeric"][name],
                  f"{name}: values differ from what C wrote, bit for bit")
        else:
            check(array.dtype.kind == "U", f"{name}: numpy dtype {array.dtype} is not a unicode string dtype")
            check([str(v) for v in array] == written["strings"][name],
                  f"{name}: strings differ from what C wrote")

    check([str(v) for v in loaded["__row_names__"]] == written["row_names"],
          "row labels differ from what C wrote")

    frame = pd.DataFrame({k: loaded[k] for k in loaded.files if k != "__row_names__"},
                         index=loaded["__row_names__"])
    check(list(frame.columns) == [name for name, _ in written["columns"]],
          "pandas does not reconstruct the frame's columns in order")
    check(frame.shape == (written["rows"], written["cols"]),
          f"pandas reconstructs {frame.shape}, expected {(written['rows'], written['cols'])}")


def check_c_reads_what_numpy_wrote(tmp, dtype):
    print("df_read_npz reads what numpy wrote (np.savez and np.savez_compressed)")
    vec = np.array([0.1, -1.0 / 3.0, 0.0, 1e30, 1e-30], dtype=dtype)
    grid = np.arange(10, dtype=dtype).reshape(5, 2)
    label = np.array(["alpha", "", "caffè", "中文", "\U0001F600"])
    rows = np.array(["r0", "r1", "r2", "r3", "r4"])

    for tag, save in (("np.savez", np.savez), ("np.savez_compressed", np.savez_compressed)):
        npz = os.path.join(tmp, "from_py.npz")
        dump = os.path.join(tmp, "from_py.txt")
        save(npz, vec=vec, grid=grid, label=label, __row_names__=rows)
        run_driver("read", npz, dump)
        got = parse_dump(dump)

        check(got["rows"] == 5, f"{tag}: read {got['rows']} rows, expected 5")
        check([name for name, _ in got["columns"]] == ["vec", "grid0", "grid1", "label"],
              f"{tag}: columns {got['columns']} - a 2D member should expand to grid0/grid1")
        check(got["numeric"]["vec"] == bits_of(vec), f"{tag}: vec differs bit for bit")
        check(got["numeric"]["grid0"] == bits_of(np.ascontiguousarray(grid[:, 0])),
              f"{tag}: grid column 0 differs bit for bit")
        check(got["numeric"]["grid1"] == bits_of(np.ascontiguousarray(grid[:, 1])),
              f"{tag}: grid column 1 differs bit for bit")
        check(got["strings"]["label"] == [str(v) for v in label], f"{tag}: label strings differ")
        check(got["row_names"] == [str(v) for v in rows], f"{tag}: row labels differ")

    print("a pandas frame survives the same trip")
    npz = os.path.join(tmp, "from_pandas.npz")
    dump = os.path.join(tmp, "from_pandas.txt")
    frame = pd.DataFrame({"gdp": np.array([1.5, 2.5, 3.5], dtype=dtype),
                          "region": ["north", "south", "east"]})
    np.savez(npz, **{name: frame[name].to_numpy() if name != "region"
                     else frame[name].to_numpy().astype(str) for name in frame.columns})
    run_driver("read", npz, dump)
    got = parse_dump(dump)
    check([name for name, _ in got["columns"]] == ["gdp", "region"], "pandas columns did not survive")
    check(got["numeric"]["gdp"] == bits_of(frame["gdp"].to_numpy()), "pandas gdp differs bit for bit")
    check(got["strings"]["region"] == list(frame["region"]), "pandas region strings differ")


FIXTURE_PATTERN = re.compile(
    r"static const unsigned char (numpy_\w+)\[\d+\] = \{(.*?)\};", re.DOTALL)
UTF8_FIXTURE_LABELS = ["alpha", "b", "", "caffè 中\U0001F600"]


def check_embedded_fixtures_are_real(tmp, dtype):
    print("the archives embedded in test_npz.c are genuine numpy output")
    source = open(os.path.join(HERE, "test_npz.c"), encoding="utf-8").read()
    fixtures = dict(FIXTURE_PATTERN.findall(source))
    suffix = "f64" if dtype == np.float64 else "f32"
    wanted = [f"numpy_stored_{suffix}", f"numpy_deflated_{suffix}"]
    check(set(wanted) <= set(fixtures), f"test_npz.c is missing one of {wanted}")

    for name in wanted:
        if name not in fixtures:
            continue
        raw = bytes(int(b, 16) for b in re.findall(r"0x([0-9A-Fa-f]{2})", fixtures[name]))
        path = os.path.join(tmp, name + ".npz")
        open(path, "wb").write(raw)
        with zipfile.ZipFile(path) as archive:
            check(archive.testzip() is None, f"{name}: fails zipfile's own CRC check")
        loaded = np.load(path)
        check(list(loaded.files) == ["vec", "grid", "label"], f"{name}: keys are {list(loaded.files)}")
        check(loaded["vec"].dtype == dtype, f"{name}: vec dtype is {loaded['vec'].dtype}")
        check(loaded["vec"].tolist() == [1.5, -2.25, 300.0, 0.0], f"{name}: vec values changed")
        check(loaded["grid"].shape == (4, 2), f"{name}: grid shape is {loaded['grid'].shape}")
        check(loaded["grid"].tolist() == [[10, 11], [12, 13], [14, 15], [16, 17]],
              f"{name}: grid values changed")
        check([str(v) for v in loaded["label"]] == UTF8_FIXTURE_LABELS,
              f"{name}: label values changed")


_NOISE = random.Random(3)

DEFLATE_CASES = {
    "empty": b"",
    "one byte": b"A",
    "100k identical": b"a" * 100000,
    "repeated english": b"the quick brown fox jumps over the lazy dog. " * 7000,
    "incompressible": _NOISE.randbytes(150000),
    "skewed": bytes((i % 251) if i % 4001 == 0 else 0 for i in range(300000)),
}


def check_zlib_accepts_our_deflate(tmp):
    """The encoder half of gzip.h, judged by an implementation that is not ours.

    tests/correctness/gzip_deflate.c can only round-trip its output through the
    decoder beside it, which proves the two halves agree but not that what they
    agree on is DEFLATE. zlib is the arbiter, and it is also the reference the
    ratio figures in docs/GZIP_DOCUMENTATION.md are measured against.
    """
    print("zlib accepts what gzip_deflate_raw produced, and at what ratio")
    src = os.path.join(tmp, "deflate_in.bin")
    dst = os.path.join(tmp, "deflate_out.bin")
    ours_total = zlib_total = 0
    for name, data in DEFLATE_CASES.items():
        open(src, "wb").write(data)
        run_driver("deflate", src, dst)  # default level
        stream = open(dst, "rb").read()
        try:
            back = zlib.decompress(stream, -15)
        except zlib.error as failure:
            check(False, f"{name}: zlib rejected our stream ({failure})")
            continue
        check(back == data, f"{name}: zlib decoded our stream to different bytes")
        reference = len(zlib.compress(data, 6))
        ours_total += len(stream)
        zlib_total += reference
        ratio = len(stream) / reference if reference else 1.0
        print(f"    {name:<18} {len(data):>8} -> {len(stream):>8}   zlib-6 {reference:>8}   {ratio:.3f}")
    if zlib_total:
        print(f"    {'total':<18} {'':>8}    {ours_total:>8}   zlib-6 {zlib_total:>8}   "
              f"{ours_total / zlib_total:.4f}")

    # every level, against the matching zlib level, on one structured input
    print("  every level, against the same zlib level")
    structured = bytes(
        (ord("a") + i % 26) if i % 100 < 55 else (32 if i % 7 else 10) if i % 100 < 80
        else _NOISE.getrandbits(8)
        for i in range(400000))
    open(src, "wb").write(structured)
    sizes = {}
    for level in range(10):
        run_driver(f"deflate{level}", src, dst)
        stream = open(dst, "rb").read()
        try:
            check(zlib.decompress(stream, -15) == structured,
                  f"level {level}: zlib decoded our stream to different bytes")
        except zlib.error as failure:
            check(False, f"level {level}: zlib rejected our stream ({failure})")
            continue
        reference = len(zlib.compress(structured, level))
        sizes[level] = len(stream)
        print(f"    level {level}: {len(stream):>8}   zlib -{level} {reference:>8}   "
              f"{len(stream) / reference:.3f}")
    # Adjacent levels are deliberately not required to be ordered - zlib's
    # own are not either on this input, its level 4 landing above its
    # level 3. What must hold is that the range buys something and that
    # level 0 really stores.
    check(sizes[9] < sizes[1], f"level 9 ({sizes[9]}) is no better than level 1 ({sizes[1]})")
    check(sizes[0] > len(structured), f"level 0 ({sizes[0]}) compressed instead of storing")


def main():
    if not os.path.exists(DRIVER):
        raise SystemExit(f"driver not built: run `make {os.path.relpath(DRIVER, ROOT)}` first")

    precision = run_driver("precision").strip()
    dtype = np.float64 if precision == "<f8" else np.float32
    print(f"frame/npz.h built at {precision}, numpy {np.__version__}\n")

    with tempfile.TemporaryDirectory() as tmp:
        check_numpy_reads_what_c_wrote(tmp, dtype)
        check_numpy_reads_what_c_wrote(tmp, dtype, compressed=True)
        check_c_reads_what_numpy_wrote(tmp, dtype)
        check_embedded_fixtures_are_real(tmp, dtype)
        check_zlib_accepts_our_deflate(tmp)

    if failures:
        print(f"\nnpz_python_interop: {len(failures)} failed")
        return 1
    print("\nnpz_python_interop: all passed")
    return 0


if __name__ == "__main__":
    try:
        import numpy as np
        import pandas as pd
    except ImportError as missing:
        print(f"npz_python_interop: skipped ({missing})")
        sys.exit(0)
    sys.exit(main())
