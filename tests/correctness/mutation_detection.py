"""Do the correctness tests catch deliberate faults in the code they test.

    make test-mutations     (PYTHON=... for an interpreter with scipy, which one check needs)

Each mutation below replaces one exact piece of source text with a faulty
version, in a copy of the sources under out/mutation_detection/, rebuilds the
named tests there with the repository's own Makefile and runs them. A mutation
is caught when a test exits nonzero, whether by a failed check or an abort.
Nothing outside out/ is modified.

Subjects:
- the banded Cholesky (linalg/factor.h, linalg/decomp.h), against
  tests/correctness/band_chol_correctness.c at float64 and at float32;
- solver/brent.h, against tests/correctness/brent_correctness.c, and for one
  mutation against tests/correctness/brent_reference_agreement.py, whose build
  without -ffast-math compares every point a search evaluates with scipy's;
- filter/kalman.h, against tests/correctness/kalman_correctness.c: one term
  dropped from each method's derivative recursion and from the prediction's,
  and one from two of the value recursions.

One mutation is expected to go undetected: removing the finiteness test on the
parabola in brent_bounded. With GCC on x86-64 the comparisons after it already
reject a NaN or an infinity at run time, so no test can tell the difference;
the test stays in the code because -ffast-math allows the compiler to assume
no such value exists. It is run so that the report shows whether that is still
true, and its outcome does not fail this script.

The script exits nonzero when a mutation expected to be caught is not, or
when a pattern no longer occurs exactly once in the source, which means this
file has fallen behind the code.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "out", "mutation_detection")
PYTHON = os.environ.get("PYTHON", sys.executable)

# Copied into each mutated tree: what the tests below include, and the
# Makefile that builds them.
TREE = ["linalg", "random", "solver", "filter", "ss", "json.h", "tests/check.h", "Makefile",
        "tests/correctness/kalman_correctness.c",
        "tests/correctness/band_chol_correctness.c",
        "tests/correctness/brent_correctness.c",
        "tests/correctness/brent_reference_agreement.c",
        "tests/correctness/brent_reference_agreement.py"]

BAND_CHOL = [("band_chol_correctness", "1"), ("band_chol_correctness", "")]
BRENT = [("brent_correctness", "1")]
BRENT_PATH = [("brent_reference_agreement", None)]
KALMAN = [("kalman_correctness", "1")]

# name, file, original text, faulty text, checks, expected to be caught
MUTATIONS = [
    ("rank-one update drops its last entry", "linalg/factor.h",
     "for (int i = r; i <= below; i++) target[i - r] -= factor * column[i];",
     "for (int i = r; i < below; i++) target[i - r] -= factor * column[i];", BAND_CHOL, True),
    ("column scaling skips its last entry", "linalg/factor.h",
     "for (int i = 1; i <= below; i++) column[i] *= inverse;",
     "for (int i = 1; i < below; i++) column[i] *= inverse;", BAND_CHOL, True),
    ("backward solve drops a term", "linalg/factor.h",
     "for (int i = 1; i <= below; i++) total -= column[i] * b[j + i];",
     "for (int i = 1; i < below; i++) total -= column[i] * b[j + i];", BAND_CHOL, True),
    ("backward solve for many right-hand sides drops a term", "linalg/factor.h",
     "for (int r = 0; r < nrhs; r++) current[r] -= coefficient * later[r];",
     "if (i < below) for (int r = 0; r < nrhs; r++) current[r] -= coefficient * later[r];", BAND_CHOL, True),
    ("no numerical-zero test on the pivot", "linalg/decomp.h",
     "if (pivot * pivot <= tolerance * (double)AT(band, k, 0))",
     "if (pivot * pivot <= 0 * tolerance * (double)AT(band, k, 0))", BAND_CHOL, True),
    ("log-determinant without its factor 2", "linalg/decomp.h",
     "return (mreal)(2 * total);", "return (mreal)(total);", BAND_CHOL, True),
    ("no NaN test on the pivot", "linalg/factor.h",
     "if (!(pivot > 0) || MISNAN(pivot)) return j + 1;", "if (!(pivot > 0)) return j + 1;", BAND_CHOL, True),
    ("entries past the end of the matrix not zeroed", "linalg/decomp.h",
     "for (int i = below + 1; i <= kd; i++) AT(l, j, i) = 0;",
     "for (int i = below + 1; i <= kd; i++) AT(l, j, i) = AT(band, j, i);", BAND_CHOL, True),
    ("blocked: A13 copied back without its diagonal", "linalg/factor.h",
     "for (int c = 0; c <= r && c < i3; c++)", "for (int c = 0; c < r && c < i3; c++)", BAND_CHOL, True),
    ("blocked: A23 update skipped", "linalg/factor.h",
     "            if (i2 > 0)\n                MBLAS(gemm)", "            if (0)\n                MBLAS(gemm)",
     BAND_CHOL, True),
    ("blocked: A33 update skipped", "linalg/factor.h",
     "            MBLAS(syrk)(CblasRowMajor, CblasUpper, CblasTrans, i3, ib,",
     "            if (0) MBLAS(syrk)(CblasRowMajor, CblasUpper, CblasTrans, i3, ib,", BAND_CHOL, True),
    ("blocked: A13 one column short", "linalg/factor.h",
     "int i3 = ib < n - j - kd ? ib : n - j - kd;", "int i3 = ib < n - j - kd ? ib - 1 : n - j - kd - 1;",
     BAND_CHOL, True),
    ("blocked: A12 one column short", "linalg/factor.h",
     "int i2 = kd - ib < n - j - ib ? kd - ib : n - j - ib;",
     "int i2 = kd - ib < n - j - ib ? kd - ib - 1 : n - j - ib;", BAND_CHOL, True),
    ("blocked: A12 solved with U11 instead of its transpose", "linalg/factor.h",
     "CblasTrans, CblasNonUnit,\n                        ib, i2, 1, base, ld, base + ib, ld);",
     "CblasNoTrans, CblasNonUnit,\n                        ib, i2, 1, base, ld, base + ib, ld);", BAND_CHOL, True),
    ("blocked: work array not zeroed above its triangle", "linalg/factor.h",
     "work[r * i3 + c] = r >= c ? base[(size_t)r * ld + kd + c] : 0;",
     "work[r * i3 + c] = r >= c ? base[(size_t)r * ld + kd + c] : 1;", BAND_CHOL, True),
    ("blocked: status not offset by the block start", "linalg/factor.h",
     "if (info) return j + info;\n        if (j + ib >= n) break;",
     "if (info) return info;\n        if (j + ib >= n) break;", BAND_CHOL, True),
    ("golden-section steps only", "solver/brent.h",
     "        if (fabs(e) > tol1) {", "        if (0 && fabs(e) > tol1) {", BRENT, True),
    ("bracket update reversed", "solver/brent.h",
     "            if (x >= xf) a = xf;\n            else b = xf;",
     "            if (x >= xf) b = xf;\n            else a = xf;", BRENT, True),
    ("NaN not converted to infinity", "solver/brent.h",
     "return mat_isnan_f64(value) ? INFINITY : value;", "return value;", BRENT, True),
    ("evaluation cap tested one evaluation late", "solver/brent.h",
     "if (num >= options.max_evaluations) {", "if (num > options.max_evaluations) {", BRENT, True),
    ("step floor tol1 dropped", "solver/brent.h",
     "x = xf + _brent_sign(rat) * fmax(fabs(rat), tol1);", "x = xf + rat;", BRENT, True),
    ("not-finite status never set", "solver/brent.h",
     "if (!_brent_is_finite(fx)) status = BRENT_NOT_FINITE;", "(void)0;", BRENT, True),
    ("zero step given a negative sign", "solver/brent.h",
     "return v < 0 ? -1.0 : 1.0;", "return v <= 0 ? -1.0 : 1.0;", BRENT + BRENT_PATH, True),
    ("no finiteness test on the parabola", "solver/brent.h",
     "int finite = _brent_is_finite(p) && _brent_is_finite(q);", "int finite = 1;", BRENT + BRENT_PATH, False),
    ("woodbury derivative: trace of S Inner^-1 dP dropped", "filter/kalman.h",
     "dlogdet += (double)h * trace_GdP;", "dlogdet += 0;", KALMAN, True),
    ("univariate derivative: curvature term dropped", "filter/kalman.h",
     "+ Pz[b] * Pz[c] * curvature;", "+ 0;", KALMAN, True),
    ("dense derivative: K dF K' dropped", "filter/kalman.h",
     "- w->mm_a[(size_t)b * m + a] + w->mm_b[(size_t)a * m + b];", "- w->mm_a[(size_t)b * m + a];", KALMAN, True),
    ("prediction derivative: transpose of dT Ptt T' dropped", "filter/kalman.h",
     "dP[(size_t)a * m + b] += w->mm_b[(size_t)a * m + b] + w->mm_b[(size_t)b * m + a];",
     "dP[(size_t)a * m + b] += w->mm_b[(size_t)a * m + b];", KALMAN, True),
    ("woodbury value: log det Inner dropped", "filter/kalman.h",
     "*logdet_out = (double)n * log((double)sigma2) + logdet_inner;",
     "*logdet_out = (double)n * log((double)sigma2);", KALMAN, True),
    ("univariate value: gain not divided by F", "filter/kalman.h",
     "for (int b = 0; b < m; b++) a[b] += Pz[b] * F_reciprocal * v;",
     "for (int b = 0; b < m; b++) a[b] += Pz[b] * v;", KALMAN, True),
]


def copy_tree(tree):
    shutil.rmtree(tree, ignore_errors=True)
    for path in TREE:
        source, target = os.path.join(ROOT, path), os.path.join(tree, path)
        if os.path.isdir(source):
            shutil.copytree(source, target)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy(source, target)


def run_check(tree, check):
    """True when the check fails on the mutated tree, None when it cannot run."""
    name, double = check
    if name == "brent_reference_agreement":
        if subprocess.run([PYTHON, "-c", "import scipy"], capture_output=True).returncode:
            return None
        run = subprocess.run(["make", "-s", "test-brent-python", f"PYTHON={PYTHON}"], cwd=tree,
                             capture_output=True, text=True)
        return run.returncode != 0
    build = subprocess.run(["make", "-s", "-B", f"MAT_DOUBLE={double}", f"tests/correctness/{name}"],
                           cwd=tree, capture_output=True, text=True)
    if build.returncode:
        return True
    run = subprocess.run([f"./tests/correctness/{name}"], cwd=tree, capture_output=True, text=True)
    return run.returncode != 0


def label(check):
    name, double = check
    if double is None:
        return name
    return f"{name} {'float64' if double else 'float32'}"


problems = 0
lines = []
for name, path, original, faulty, checks, expected in MUTATIONS:
    text = open(os.path.join(ROOT, path)).read()
    if text.count(original) != 1:
        lines.append(f"{name}: the original text occurs {text.count(original)} times in {path}")
        problems += 1
        continue
    tree = os.path.join(SCRATCH, "tree")
    copy_tree(tree)
    target = os.path.join(tree, path)
    with open(target, "w") as f:
        f.write(text.replace(original, faulty))
    outcomes = [(check, run_check(tree, check)) for check in checks]
    caught = [label(c) for c, failed in outcomes if failed]
    skipped = [label(c) for c, failed in outcomes if failed is None]
    if caught:
        verdict = "caught by " + ", ".join(caught)
    else:
        verdict = "not caught"
    if skipped:
        verdict += "; skipped, no scipy: " + ", ".join(skipped)
    if expected and not caught:
        verdict += "  EXPECTED TO BE CAUGHT"
        problems += 1
    if not expected:
        verdict += "  (expected to go undetected)"
    lines.append(f"{name}: {verdict}")
shutil.rmtree(SCRATCH, ignore_errors=True)

for line in lines:
    print(line)
print(f"{len(MUTATIONS)} mutations, {problems} problems")
sys.exit(1 if problems else 0)
