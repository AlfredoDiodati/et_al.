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
  and one from two of the value recursions; in the Woodbury value on reduced
  observations, the residual e dropped and the stored Z_o'Z_o left unused; in
  the derivatives batched over the coordinates, the transpose in T dP T'
  skipped and the last coordinate left out of the stacked dP S P product;
  in the exact initial filter, a term dropped from the update of P_* and of
  its derivative, P_infinity left unreduced by an entry or unpredicted, the
  diffuse likelihood term taken from F_* instead of F_infinity, the
  diffuse periods never ending or their count starting one short of the
  rank of P_infinity, a term dropped from the derivatives of the likelihood
  and of P_infinity, and the Woodbury method's reduced observations read in
  a diffuse period;
- filter/hp.h's causal filter, against
  tests/correctness/hp_filter_causal_correctness.c at float64: lambda in place
  of 1 / lambda as the trend's variance, one of the two state elements not
  diffuse, and the trend read from the wrong state element;
- hp_estimate and hp_log_likelihood, against
  tests/correctness/hp_filter_lambda_estimation.c: a wrong second
  difference, a wrong diagonal in M, sigma2_c divided by n instead of n - 2,
  log det M dropped, and the full banded solve in place of its forward half,
  which tests/correctness/band_chol_correctness.c checks too;
- frame/gzip.h, against tests/correctness/gzip_inflate.c and
  tests/correctness/gzip_deflate.c: a CRC32 table built with the wrong
  shift, one carry-less folding constant off by one, and a Huffman
  subtable indexed one bit too far;
- frame/npz.h, against tests/correctness/test_npz.c: the subset reader
  selecting every member, or reading every member and dropping the unasked
  ones; a member's CRC32 not checked; codepoints up to 0xFF decoded as one
  byte; single columns copied from the band's own rows instead of the
  frame's; and, on the threaded paths alone, every member unpacked against
  the first member's directory entry and the last row of every band left
  uncopied.

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
TREE = ["linalg", "random", "solver", "filter", "ss", "frame", "json.h", "tests/check.h", "Makefile",
        "tests/correctness/kalman_correctness.c",
        "tests/correctness/hp_filter_causal_correctness.c",
        "tests/correctness/hp_filter_lambda_estimation.c",
        "tests/correctness/band_chol_correctness.c",
        "tests/correctness/brent_correctness.c",
        "tests/correctness/brent_reference_agreement.c",
        "tests/correctness/brent_reference_agreement.py",
        "tests/correctness/gzip_inflate.c",
        "tests/correctness/gzip_deflate.c",
        "tests/correctness/test_npz.c"]

BAND_CHOL = [("band_chol_correctness", "1"), ("band_chol_correctness", "")]
BRENT = [("brent_correctness", "1")]
BRENT_PATH = [("brent_reference_agreement", None)]
KALMAN = [("kalman_correctness", "1")]
HP_CAUSAL = [("hp_filter_causal_correctness", "1")]
HP_ESTIMATE = [("hp_filter_lambda_estimation", "1")]
GZIP_INFLATE = [("gzip_inflate", "")]
GZIP_DEFLATE = [("gzip_deflate", "")]
NPZ = [("test_npz", "")]

# A pass over every directory entry before the selection, reading and
# unpacking each member and throwing it away.
NPZ_READ_ALL = """    for (int i = 0; i < n_entries; i++) {
        FrameNpzMember extra = frame_npz_read_member(&archive, directory + entry[i],
                                                     (const char*)(directory + entry[i] + 46), key_length[i]);
        frame_npz_unpack_member(&extra, directory + entry[i]);
        free(extra.name);
        free(extra.image);
    }
"""

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
    ("reduced observations: residual e dropped from v'v", "filter/kalman.h",
     "vv = (mreal)(w->prepared_e[prepared] + (double)_kalman_dot(k, vo, vo));",
     "vv = (mreal)((double)_kalman_dot(k, vo, vo));", KALMAN, True),
    ("reduced observations: stored Z_o'Z_o not used", "filter/kalman.h",
     "        S = w->prepared_S + (size_t)prepared * mm;", "", KALMAN, True),
    ("batched prediction derivative: transpose in T dP T' skipped", "filter/kalman.h",
     "X_transposed[(size_t)i * mm + (size_t)r * m + c] = X[(size_t)i * mm + (size_t)c * m + r];",
     "X_transposed[(size_t)i * mm + (size_t)r * m + c] = X[(size_t)i * mm + (size_t)r * m + c];", KALMAN, True),
    ("batched update derivative: last coordinate left out of dP S P", "filter/kalman.h",
     "mat_gemm(0, 0, k * m, m, m, -h, w->dP, m, SP, m, 1, Y_stack, m);",
     "mat_gemm(0, 0, (k - 1) * m, m, m, -h, w->dP, m, SP, m, 1, Y_stack, m);", KALMAN, True),
    ("diagonal T: a full matrix taken as diagonal", "filter/kalman.h",
     "if (a != b && A.d[(size_t)a * A.stride + b] != 0) return 0;", "(void)0;", KALMAN, True),
    ("diagonal T value: T_i in place of T_j in T Ptt T'", "filter/kalman.h",
     "P[ij] = RQR[ij] + TP[ij] * T[(size_t)j * T_diagonal_step];",
     "P[ij] = RQR[ij] + TP[ij] * T_i;", KALMAN, True),
    ("diagonal T derivative: T_a in place of T_b in T dP T'", "filter/kalman.h",
     "dP[(size_t)a * m + b] = T_a * dPtt[(size_t)a * m + b] * T[(size_t)b * T_diagonal_step];",
     "dP[(size_t)a * m + b] = T_a * dPtt[(size_t)a * m + b] * T_a;", KALMAN, True),
    ("diagonal dT derivative: transpose of dT Ptt T' dropped", "filter/kalman.h",
     "+ ds->T.d[(size_t)b * dT_diagonal_step] * w->TP[(size_t)a * m + b];", ";", KALMAN, True),
    ("diagonal dT derivative: dT a_{t|t} dropped", "filter/kalman.h",
     "da[a] += ds->T.d[(size_t)a * dT_diagonal_step] * w->att[a];", "(void)0;", KALMAN, True),
    ("diffuse update: K0 K0' F_* dropped from P_*", "filter/kalman.h",
     "P[(size_t)b * m + c] += K0_b * K0_c * Fstar - K0_b * Mstar[c] - Mstar[b] * K0_c;",
     "P[(size_t)b * m + c] += -K0_b * Mstar[c] - Mstar[b] * K0_c;", KALMAN, True),
    ("diffuse update: P_infinity not reduced by the entry", "filter/kalman.h",
     "Pinf[(size_t)b * m + c] -= K0_b * Minf[c];", "(void)0;", KALMAN, True),
    ("diffuse likelihood: log F_* in place of log F_infinity", "filter/kalman.h",
     "logdet += log((double)Finf);", "logdet += log((double)Fstar);", KALMAN, True),
    ("diffuse periods never end", "filter/kalman.h",
     "            if (paths) paths->diffuse_periods = t + 1;\n            diffuse = diffuse_remaining > 0;",
     "            if (paths) paths->diffuse_periods = t + 1;\n            diffuse = 1;", KALMAN, True),
    ("diffuse count starts one short of the rank", "filter/kalman.h",
     "diffuse_remaining = w->Pinf_rank;", "diffuse_remaining = w->Pinf_rank - 1;", KALMAN, True),
    ("diffuse prediction: P_infinity not predicted", "filter/kalman.h",
     "            _kalman_predict_diffuse(w, s, with_tangents);\n", "", KALMAN, True),
    ("diffuse derivative: d log F_infinity dropped", "filter/kalman.h",
     "w->dl[p] += -0.5 * (double)dFinf * F_reciprocal;", "(void)0;", KALMAN, True),
    ("diffuse derivative: K0 K0' dF_* dropped from dP_*", "filter/kalman.h",
     "(dK0_b * K0_c + K0_b * dK0_c) * Fstar + K0_b * K0_c * dFstar", "(dK0_b * K0_c + K0_b * dK0_c) * Fstar", KALMAN, True),
    ("diffuse derivative: dP_infinity not reduced by the entry", "filter/kalman.h",
     "dPinf[(size_t)b * m + c] -= dK0_b * Minf[c] + K0_b * dMinf[c];", "(void)0;", KALMAN, True),
    ("diffuse prediction derivative: transpose of dT P_infinity T' dropped", "filter/kalman.h",
     "dPinf[(size_t)a * m + b] += w->mm_b[(size_t)a * m + b] + w->mm_b[(size_t)b * m + a];",
     "dPinf[(size_t)a * m + b] += w->mm_b[(size_t)a * m + b];", KALMAN, True),
    ("diffuse period reads the reduced observations", "filter/kalman.h",
     "int n = use_prepared && !diffuse ? w->prepared_n[t] : _kalman_gather_observed(w, y, t);",
     "int n = use_prepared ? w->prepared_n[t] : _kalman_gather_observed(w, y, t);", KALMAN, True),
    ("causal HP: lambda in place of 1 / lambda as the trend's variance", "filter/hp.h",
     "AT(system.Q, 0, 0) = (mreal)(1 / lambda);", "AT(system.Q, 0, 0) = (mreal)(lambda);", HP_CAUSAL, True),
    ("causal HP: tau_0 not diffuse", "filter/hp.h",
     "AT(P1_infinity, 0, 0) = 1; AT(P1_infinity, 1, 1) = 1;", "AT(P1_infinity, 0, 0) = 1;", HP_CAUSAL, True),
    ("causal HP: trend read from tau_{t-1}", "filter/hp.h",
     "? AT(paths.att, 0, t) : (mreal)NAN;", "? AT(paths.att, 1, t) : (mreal)NAN;", HP_CAUSAL, True),
    ("HP estimate: first difference in place of the second", "filter/hp.h",
     "z.d[t] = y2 - 2 * y1 + y0;", "z.d[t] = y2 - y1;", HP_ESTIMATE, True),
    ("HP estimate: 6 + 2 q on the diagonal of M", "filter/hp.h",
     "AT(band, j, 0) = (mreal)(6 + q);", "AT(band, j, 0) = (mreal)(6 + 2 * q);", HP_ESTIMATE, True),
    ("HP estimate: sigma2_c divided by n instead of n - 2", "filter/hp.h",
     "*sigma2_out = squares / m;", "*sigma2_out = squares / (m + 2);", HP_ESTIMATE, True),
    ("HP estimate: log det M dropped", "filter/hp.h",
     "return -0.5 * (m * (log(2 * M_PI * sigma2) + 1) + logdet);",
     "return -0.5 * (m * (log(2 * M_PI * sigma2) + 1) + 0 * logdet);", HP_ESTIMATE, True),
    ("forward banded solve runs the full solve", "linalg/solver.h",
     "_pbtrs_forward(factor.d, factor.r, factor.c - 1, factor.stride, x.d);",
     "_pbtrs(factor.d, factor.r, factor.c - 1, factor.stride, x.d);", BAND_CHOL + HP_ESTIMATE, True),
    ("CRC32: a table built with the wrong shift", "frame/gzip.h",
     "table[k * 256 + i] = (previous >> 8) ^ table[previous & 0xFF];",
     "table[k * 256 + i] = (previous >> 7) ^ table[previous & 0xFF];", GZIP_INFLATE, True),
    ("CRC32 fold: one folding constant off by one", "frame/gzip.h",
     "0x01751997d0LL", "0x01751997d1LL", GZIP_INFLATE, True),
    ("Huffman: subtable indexed one bit too far", "frame/gzip.h",
     "((peek >> GZIP_ROOT_BITS) & ((1u << h->sub_bits) - 1))",
     "((peek >> (GZIP_ROOT_BITS + 1)) & ((1u << h->sub_bits) - 1))", GZIP_DEFLATE + NPZ, True),
    ("npz subset: every member selected", "frame/npz.h",
     "    if (!keys) {\n        for (int i = 0; i < n_entries; i++) selected[n_selected++] = i;",
     "    if (1) {\n        for (int i = 0; i < n_entries; i++) selected[n_selected++] = i;", NPZ, True),
    ("npz subset: every member read, the unasked ones dropped", "frame/npz.h",
     "    /* The entries to read, in the order their columns will appear. */\n",
     "    /* The entries to read, in the order their columns will appear. */\n" + NPZ_READ_ALL, NPZ, True),
    ("npz: a member's CRC32 not checked", "frame/npz.h",
     '    assert(gzip_crc32(member->image, uncompressed) == crc && "frame: .npz member failed its CRC32 check");',
     "    (void)crc;", NPZ, True),
    ("npz: codepoints up to 0xFF decoded as one byte", "frame/npz.h",
     "        if (v < 0x80) {", "        if (v < 0x100) {", NPZ, True),
    ("npz: single columns copied from the band's own rows, not the frame's", "frame/npz.h",
     "data + (size_t)i * sizeof(mreal), sizeof(mreal));", "data + (size_t)(i - start) * sizeof(mreal), sizeof(mreal));",
     NPZ, True),
    ("npz threaded unpack: every member against the first member's entry", "frame/npz.h",
     "        #pragma omp parallel for schedule(dynamic, 1)\n#endif\n"
     "        for (int m = 0; m < n_selected; m++) frame_npz_unpack_member(&members[m], directory + entry[selected[m]]);",
     "        #pragma omp parallel for schedule(dynamic, 1)\n#endif\n"
     "        for (int m = 0; m < n_selected; m++) frame_npz_unpack_member(&members[m], directory + entry[selected[0]]);",
     NPZ, True),
    ("npz threaded band copy: the last row of every band left uncopied", "frame/npz.h",
     "            #pragma omp parallel for schedule(static)\n#endif\n"
     "            for (int b = 0; b < n_bands; b++) frame_npz_copy_band(df.numeric, b * band, band, sources,",
     "            #pragma omp parallel for schedule(static)\n#endif\n"
     "            for (int b = 0; b < n_bands; b++) frame_npz_copy_band(df.numeric, b * band, band - 1, sources,",
     NPZ, True),
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
