"""Checks df_read_csv's missing-value markers against pandas, polars and R.

    make test-csv-na-python     (PYTHON=... for an interpreter with numpy, pandas and polars; Rscript for the R arm)

Outside make test, since it needs Python with pandas and polars, and R.

References: pandas 2.3.3, read_csv(na_values=markers,
keep_default_na=False); polars 1.38.1, read_csv(null_values=markers); R
4.6.1, read.csv(na.strings=markers). Files: 300 random CSVs from numpy's
default_rng(2026), 1 to 30 rows and 1 to 6 columns, each column numeric
(integers and decimals), numeric with markers mixed in, text, text with
markers, or markers only; markers "NA" and, in half the files, also the
empty field, quoted in some cells.

Compared: the number of rows (blank lines are skipped by all four), each
column's type (numeric or text) and, for a numeric column, every value,
NaN matching a missing value. Known differences, kept:
- an empty field that is not a listed marker makes a text column here, as
  in pandas, while polars and R read it as missing in a numeric column; the
  files where that happens are compared with pandas only;
- polars types a column with no value at all as String, where pandas and R
  make it numeric and so does this library; such a column counts as numeric
  on polars' side;
- polars keeps a blank line of a one-column file as a null row, where the
  other three skip it; those files are not compared with polars;
- a file left with no data rows has numeric columns here, every value in
  them being vacuously a number, and object columns in pandas; only the
  number of rows is compared then.
"""
import os
import subprocess
import sys
import tempfile
import ctypes

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libcsvna_f64.so"], cwd=ROOT, check=True)
lib = ctypes.CDLL(os.path.join(ROOT, "libcsvna_f64.so"))
lib.c_read_csv_with_markers.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
lib.c_read_csv_with_markers.restype = ctypes.c_int

rng = np.random.default_rng(2026)
failures, comparisons = [], 0


def cell(kind, markers, rng):
    if kind in ("numeric_na", "text_na", "na") and (kind == "na" or rng.random() < 0.3):
        marker = markers[rng.integers(len(markers))]
        return f'"{marker}"' if marker and rng.random() < 0.3 else marker
    if kind in ("numeric", "numeric_na"):
        return str(int(rng.integers(-50, 50))) if rng.random() < 0.5 else repr(round(float(rng.normal() * 10.0 ** rng.integers(-3, 4)), 6))
    return "t" + str(int(rng.integers(0, 100)))


def ours(path, markers, capacity, cols):
    """types, values (cols x rows read) and the number of rows read, blank lines skipped"""
    types = (ctypes.c_int * cols)()
    values = (ctypes.c_double * (capacity * cols))()
    rows = ctypes.c_int()
    n = lib.c_read_csv_with_markers(path.encode(), "\n".join(markers).encode(), len(markers), capacity, ctypes.byref(rows), types, values)
    assert n == cols
    return [bool(t) for t in types], np.array(values[:]).reshape(cols, capacity)[:, :rows.value], rows.value


def check(name, label, our_types, our_values, their_types, their_values, our_rows, their_rows):
    global comparisons
    comparisons += 1
    if our_rows != their_rows or len(our_types) != len(their_types):
        failures.append(label)
        print(f"  FAIL {label} ({name}): {our_rows} rows and {len(our_types)} columns here, {their_rows} and {len(their_types)} there")
        return
    if our_rows == 0:
        return
    for j, (a, b) in enumerate(zip(our_types, their_types)):
        if a != b:
            failures.append(label)
            print(f"  FAIL {label} ({name}): column {j} numeric here {a}, there {b}")
            return
        if a:
            x, y = our_values[j], np.asarray(their_values[j], dtype=float)
            if not (np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])):
                failures.append(label)
                print(f"  FAIL {label} ({name}): column {j} values differ")
                return


with tempfile.TemporaryDirectory() as tmp:
    for f in range(300):
        rows, cols = int(rng.integers(1, 31)), int(rng.integers(1, 7))
        markers = ["NA", ""] if f % 2 else ["NA"]
        kinds = [["numeric", "numeric_na", "text", "text_na", "na"][rng.integers(5)] for _ in range(cols)]
        header = ",".join(f"c{j}" for j in range(cols))
        body = [[cell(k, markers, rng) for k in kinds] for _ in range(rows)]
        path = os.path.join(tmp, f"f{f}.csv")
        with open(path, "w") as out:
            out.write(header + "\n" + "\n".join(",".join(r) for r in body) + "\n")
        our_types, our_values, our_rows = ours(path, markers, rows, cols)
        label = f"file {f}"

        frame = pd.read_csv(path, na_values=markers, keep_default_na=False)
        types = [pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns]
        check("pandas", label, our_types, our_values, types, [frame[c].to_numpy() if t else None for c, t in zip(frame.columns, types)],
              our_rows, len(frame))

        has_unlisted_empty = "" not in markers and any(v == "" for r in body for v in r)
        if has_unlisted_empty:
            continue
        # polars reads a blank line in a one-column file as a null row where the
        # other three skip it, and types a column with no value at all as
        # String; both are polars' conventions, compared around
        if any(",".join(r) == "" for r in body):
            continue
        frame = pl.read_csv(path, null_values=markers, infer_schema_length=None)
        types = [frame[c].dtype.is_numeric() or frame[c].null_count() == len(frame) for c in frame.columns]
        check("polars", label, our_types, our_values, types,
              [frame[c].cast(pl.Float64).fill_null(np.nan).to_numpy() if t else None for c, t in zip(frame.columns, types)], our_rows, len(frame))

        out_path = os.path.join(tmp, f"r{f}.csv")
        na_strings = ", ".join(f'"{m}"' for m in markers)
        script = (f"d <- read.csv('{path}', na.strings = c({na_strings})); "
                  f"numeric <- sapply(d, function(x) is.numeric(x) || all(is.na(x))); "
                  f"write.csv(data.frame(type = numeric), '{out_path}', row.names = FALSE); "
                  f"for (j in which(numeric)) writeLines(ifelse(is.na(d[[j]]), 'NA', sprintf('%.17g', as.numeric(d[[j]]))), paste0('{out_path}.', j))")
        subprocess.run(["Rscript", "-e", script], check=True, capture_output=True)
        r_types = [t == "TRUE" for t in open(out_path).read().split()[1:]]
        r_values = []
        for j, t in enumerate(r_types):
            if t:
                text = open(f"{out_path}.{j + 1}").read().split()
                r_values.append(np.array([np.nan if v == "NA" else float(v) for v in text]))
            else:
                r_values.append(None)
        r_rows = len(r_values[0]) if r_types and r_types[0] else None
        if r_rows is None:
            r_rows = int(subprocess.run(["Rscript", "-e", f"cat(nrow(read.csv('{path}', na.strings = c({na_strings}))))"],
                                        check=True, capture_output=True, text=True).stdout)
        check("R", label, our_types, our_values, r_types, r_values, our_rows, r_rows)

if failures:
    print(f"csv_missing_values_reference_agreement: {len(failures)} of {comparisons} comparisons failed")
    sys.exit(1)
print(f"csv_missing_values_reference_agreement: all {comparisons} comparisons agree with pandas 2.3.3, polars 1.38.1 and R 4.6.1")
