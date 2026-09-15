# basis/poly.h and basis/spline.h vs R's stats::poly and the splines
# package: how long one basis costs.
#
# What is being compared is not C against R-the-language. R's splineDesign
# calls compiled C for the recursion itself, and stats::poly calls compiled
# LINPACK for its QR. What is left in R around those calls is what this
# measures: splineDesign's C returns the ord non-zero values per point plus
# a column offset, and R then builds two index vectors with outer() and
# rep.int() and scatters the values into a dense matrix through an index
# matrix, which is an allocation and a pass over nx * ord elements per call
# that this implementation does not make because it writes into the dense
# matrix directly. bs() and ns() add another layer of the same: a
# quantile(), several matrix subsets that each copy, and for ns() a full
# qr.qty() against the complete orthogonal factor where two Householder
# reflectors applied in place would do.
#
# Two of our timings are reported. "ours" is the whole .C() call, which
# includes R allocating the result vector and copying the answer back
# across the interface - the number a caller in R would actually see.
# "kernel" runs the identical computation inside C and returns only the
# elapsed time, so the difference between them is the boundary cost rather
# than the algorithm's.
#
# The run fails if any case is slower than R on the kernel timing. That is
# the claim the file exists to hold: the reimplementation is faster at
# every shape it is used at, not merely competitive on average.
#
# R is a development-tier dependency, so this is not part of bench.sh
# (which drives the Python comparison suites). Run:
#   make bench-basis

suppressPackageStartupMessages(library(splines))

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- if (length(script_path) == 1L)
    normalizePath(file.path(dirname(script_path), "..", "..")) else normalizePath(".")

stopifnot(system2("make", c("-C", root, "libbasisbench.so"), stdout = FALSE) == 0L)
dyn.load(file.path(root, "libbasisbench.so"))

mreal_bytes <- .C("c_mreal_bytes", out = integer(1))$out
build <- if (mreal_bytes == 8L) "float64" else "float32"

# Every case is timed for at least this long, and the best of three such
# rounds is reported, so a scheduling hiccup in one round cannot inflate
# the result.
budget_seconds <- 1.0
rounds <- 3L

best_milliseconds <- function(f) {
    f()
    best <- Inf
    for (round in seq_len(rounds)) {
        started <- proc.time()[["elapsed"]]
        runs <- 0L
        repeat {
            f()
            runs <- runs + 1L
            if (proc.time()[["elapsed"]] - started >= budget_seconds) break
        }
        elapsed <- (proc.time()[["elapsed"]] - started) / runs * 1000
        if (elapsed < best) best <- elapsed
    }
    best
}

# The kernel entry points time themselves in C over a repeat count chosen
# so the loop runs for about the same budget, and return seconds per call.
kernel_milliseconds <- function(symbol, arguments, guess_milliseconds) {
    repeats <- max(3L, as.integer(budget_seconds * 1000 / max(guess_milliseconds, 1e-4)))
    repeats <- min(repeats, 200000L)
    best <- Inf
    for (round in seq_len(rounds)) {
        result <- do.call(".C", c(list(symbol), arguments,
                                  list(as.integer(repeats), seconds = double(1))))
        milliseconds <- result$seconds * 1000
        if (milliseconds < best) best <- milliseconds
    }
    best
}

lines <- character(0)
say <- function(...) { text <- sprintf(...); lines <<- c(lines, text); invisible(NULL) }

say("basis/poly.h and basis/spline.h vs R %s, %s build", getRversion(), build)
say("best of %d rounds, each at least %.1f s; kernel timings exclude the .C() copy",
    rounds, budget_seconds)
say("")

slower <- character(0)
record <- function(subject, shape, theirs, ours, kernel) {
    say("%-26s %12s %10.4f %10.4f %10.4f %8.1fx %8.1fx",
        subject, shape, theirs, ours, kernel, theirs / ours, theirs / kernel)
    if (kernel >= theirs) slower <<- c(slower, sprintf("%s %s", subject, shape))
}
header <- function() {
    say("%-26s %12s %10s %10s %10s %9s %9s",
        "subject", "shape", "R ms", "ours ms", "kernel ms", "speedup", "kernel")
    say(strrep("-", 92))
}

header()

# --- stats::poly
for (n in c(1000L, 100000L)) for (degree in c(3L, 10L)) {
    x <- as.double(seq_len(n)) + 0.5
    buffer <- double(as.double(n) * degree)
    theirs <- best_milliseconds(function() poly(x, degree))
    ours <- best_milliseconds(function()
        .C("c_poly", n, degree, x, basis = buffer)$basis)
    kernel <- kernel_milliseconds("kernel_poly", list(n, degree, x), ours)
    record("poly", sprintf("%d, deg %d", n, degree), theirs, ours, kernel)
}

# --- splines::splineDesign
for (n in c(1000L, 100000L)) for (ord in c(4L, 6L)) {
    knots <- c(rep(0, ord), seq_len(19), rep(20, ord))
    nk <- length(knots)
    x <- seq(0, 20, length.out = n)
    buffer <- double(as.double(n) * (nk - ord))
    theirs <- best_milliseconds(function() splineDesign(knots, x, ord = ord))
    ours <- best_milliseconds(function()
        .C("c_spline_design", nk, knots, n, x, ord, design = buffer)$design)
    kernel <- kernel_milliseconds("kernel_spline_design", list(nk, knots, n, x, ord), ours)
    record("splineDesign", sprintf("%d, ord %d", n, ord), theirs, ours, kernel)
}

# --- splines::bs and splines::ns
for (n in c(1000L, 100000L)) for (df in c(7L, 20L)) {
    x <- seq(0, 20, length.out = n)
    buffer <- double(as.double(n) * df)
    theirs <- best_milliseconds(function() bs(x, df = df))
    ours <- best_milliseconds(function()
        .C("c_bs", n, x, 3L, df, basis = buffer)$basis)
    kernel <- kernel_milliseconds("kernel_bs", list(n, x, 3L, df), ours)
    record("bs", sprintf("%d, df %d", n, df), theirs, ours, kernel)

    theirs <- best_milliseconds(function() ns(x, df = df))
    ours <- best_milliseconds(function()
        .C("c_ns", n, x, df, basis = buffer)$basis)
    kernel <- kernel_milliseconds("kernel_ns", list(n, x, df), ours)
    record("ns", sprintf("%d, df %d", n, df), theirs, ours, kernel)
}

# --- splines::interpSpline
# The collocation system is banded, and both sides know it: R's sparse =
# TRUE hands it to the Matrix package, and interp_spline builds it in band
# storage and solves it with vec_band_solve. R's default is the dense
# solve, which is what the dense column below is, so the two R columns are
# the same computation done two ways and the margin over the dense one is
# expected to grow with n rather than shrink. See
# docs/SPLINE_BASIS_DOCUMENTATION.md.
for (n in c(50L, 200L, 800L, 3200L, 12800L)) {
    x <- as.double(seq_len(n))
    y <- sin(x / 7) + x / 40
    theirs <- best_milliseconds(function() interpSpline(x, y))
    sparse <- if (requireNamespace("Matrix", quietly = TRUE))
        best_milliseconds(function() interpSpline(x, y, sparse = TRUE)) else NA_real_
    ours <- best_milliseconds(function()
        .C("c_interp_spline", n, x, y, knots = double(n),
           coefficients = double(as.double(n) * 4))$coefficients)
    kernel <- kernel_milliseconds("kernel_interp_spline", list(n, x, y), ours)
    record("interpSpline", sprintf("%d points", n), theirs, ours, kernel)
    if (is.finite(sparse))
        say("%-26s %12s %10.4f %10s %10s %8s %8.1fx", "  vs R sparse = TRUE",
            sprintf("%d points", n), sparse, "", "", "", sparse / kernel)
}

# --- evaluating a fitted spline
for (m in c(1000L, 100000L)) {
    n <- 200L
    x <- as.double(seq_len(n))
    y <- sin(x / 7) + x / 40
    at <- seq(1, n, length.out = m)
    fitted <- interpSpline(x, y)
    theirs <- best_milliseconds(function() predict(fitted, at))
    ours <- best_milliseconds(function()
        .C("c_spline_predict", n, x, y, m, at, out = double(m))$out)
    kernel <- kernel_milliseconds("kernel_spline_predict", list(n, x, y, m, at), ours)
    record("predict(polySpline)", sprintf("%d points", m), theirs, ours, kernel)
}

say("")
if (length(slower)) {
    say("SLOWER THAN R on the kernel timing: %s", paste(slower, collapse = ", "))
} else {
    say("faster than R on the kernel timing at every shape above")
}

dir.create(file.path(root, "out"), showWarnings = FALSE, recursive = TRUE)
report <- file.path(root, "out", "bench_basis_report.txt")
writeLines(lines, report)
cat(paste(lines, collapse = "\n"), "\n", sep = "")
cat("report written to ", report, "\n", sep = "")
quit(status = if (length(slower)) 1L else 0L)
