# basis/poly.h and basis/spline.h against a live R: do the two compute the
# same numbers?
#
# Unlike tests/correctness/lhs_r_agreement.R next to it, which can only
# compare distributions because the two generators differ, every function
# compared here is a deterministic function of its input. Both sides are
# handed byte-identical inputs and the comparison is element by element.
# What a passing run establishes is therefore much stronger: not that the
# reimplementation is of the same family, but that it is the same
# function to floating-point tolerance.
#
# R is a development-tier dependency. `make test` never runs this file;
# `make test-basis-r` does. The shared object it drives is built from
# tests/correctness/basis_r_agreement.c and exposes plain C symbols, so
# nothing in the library knows about R.
#
# Usage:
#   make test-basis-r
#   Rscript tests/correctness/basis_r_agreement.R       # same thing
#   MAT_DOUBLE=1 make test-basis-r                      # float64 build

suppressPackageStartupMessages(library(splines))

# The repository root, taken from the script's own path so the run does not
# depend on the working directory it was started from.
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- if (length(script_path) == 1L)
    normalizePath(file.path(dirname(script_path), "..", "..")) else normalizePath(".")

shared_object <- file.path(root, "libbasisagreement.so")
stopifnot(system2("make", c("-C", root, "libbasisagreement.so"), stdout = FALSE) == 0L)
dyn.load(shared_object)

mreal_bytes <- .C("c_mreal_bytes", out = integer(1))$out
build <- if (mreal_bytes == 8L) "float64" else "float32"

# The tolerance every comparison is judged against, relative to the larger
# of the two magnitudes and falling back to absolute below one. It is set
# by the storage precision of the build and nothing else: the inputs
# themselves are rounded to mreal on the way in, which is the floor on how
# closely two implementations of the same function can agree.
tolerance <- if (mreal_bytes == 8L) 1e-11 else 2e-5

failures <- 0L
comparisons <- 0L
worst_overall <- 0

# Every comparison in this file goes through here, so that what counts as
# agreement is decided in one place and the report always names the size
# of the disagreement rather than only its verdict.
compare <- function(label, got, want, tol = tolerance) {
    comparisons <<- comparisons + 1L
    got <- as.vector(got); want <- as.vector(want)
    if (length(got) != length(want)) {
        cat(sprintf("  FAIL %-46s length %d, want %d\n", label, length(got), length(want)))
        failures <<- failures + 1L
        return(invisible(NULL))
    }
    finite_both <- is.finite(got) & is.finite(want)
    if (any(is.finite(got) != is.finite(want))) {
        cat(sprintf("  FAIL %-46s non-finite values in different places\n", label))
        failures <<- failures + 1L
        return(invisible(NULL))
    }
    scale <- pmax(abs(got[finite_both]), abs(want[finite_both]), 1)
    relative <- if (any(finite_both)) max(abs(got[finite_both] - want[finite_both]) / scale) else 0
    worst_overall <<- max(worst_overall, relative)
    if (relative > tol) {
        cat(sprintf("  FAIL %-46s relative %.3g > %.3g\n", label, relative, tol))
        failures <<- failures + 1L
    } else {
        cat(sprintf("  ok   %-46s relative %.3g\n", label, relative))
    }
    invisible(NULL)
}

cat(sprintf("basis/poly.h and basis/spline.h against R %s, %s build\n",
            getRversion(), build))
cat(sprintf("tolerance %.3g relative, floor 1 in the denominator\n\n", tolerance))

# --- basis/poly.h

cat("stats::poly\n")

# The last two cases carry their own float32 tolerance. Both are deliberately
# adversarial - a sample sitting a million from the origin, and one spread over
# 1e-4 - and what limits agreement there is not the algorithm but the storage:
# the basis is the sample centred and raised to powers, and at float32 the
# information those powers depend on is already gone by the time either
# implementation sees the input. Measured on this build, the largest relative
# disagreement is 6.6e-3 for the offset sample and 8.8e-5 for the narrow one;
# at float64 both are at the file's own tolerance and the field is unused.
poly_cases <- list(
    list(name = "1:10, degree 3",          x = as.double(1:10),               degree = 3L),
    list(name = "1:50, degree 6",          x = as.double(1:50),               degree = 6L),
    list(name = "unequally spaced, deg 4", x = c(0.1, 0.3, 0.35, 1.2, 4, 9, 9.5, 11, 20, 31, 44), degree = 4L),
    list(name = "negative and zero, deg 5",x = seq(-7, 5, length.out = 40),   degree = 5L),
    list(name = "large offset, degree 3",  x = 1e6 + (1:30),                  degree = 3L, tol32 = 2e-2),
    list(name = "tiny spread, degree 2",   x = 1 + 1e-4 * (1:25),             degree = 2L, tol32 = 5e-4)
)

for (case in poly_cases) {
    case_tolerance <- if (mreal_bytes == 4L && !is.null(case$tol32)) case$tol32 else tolerance
    n <- length(case$x); degree <- case$degree
    got <- .C("c_poly", as.integer(n), as.integer(degree), as.double(case$x),
              basis = double(n * degree), alpha = double(degree), norm2 = double(degree + 2))
    reference <- poly(case$x, degree)
    compare(sprintf("poly basis: %s", case$name),
            matrix(got$basis, n, degree, byrow = TRUE), unclass(reference),
            tol = case_tolerance)
    compare(sprintf("poly alpha: %s", case$name), got$alpha, attr(reference, "coefs")$alpha)
    compare(sprintf("poly norm2: %s", case$name), got$norm2, attr(reference, "coefs")$norm2)

    # prediction at points the fit never saw, which is the whole reason
    # the coefs exist: a fresh fit on new data would be a different basis
    newx <- seq(min(case$x) - 0.3 * diff(range(case$x)),
                max(case$x) + 0.3 * diff(range(case$x)), length.out = 17)
    predicted <- .C("c_poly_predict", as.integer(degree), as.double(got$alpha),
                    as.double(got$norm2), as.integer(length(newx)), as.double(newx),
                    basis = double(length(newx) * degree))$basis
    compare(sprintf("poly predict: %s", case$name),
            matrix(predicted, length(newx), degree, byrow = TRUE),
            unclass(predict(reference, newx)), tol = case_tolerance)

    raw_basis <- .C("c_poly_raw", as.integer(n), as.integer(degree), as.double(case$x),
                    basis = double(n * degree))$basis
    compare(sprintf("poly raw: %s", case$name),
            matrix(raw_basis, n, degree, byrow = TRUE),
            unclass(poly(case$x, degree, raw = TRUE)))
}

# Only up to 20 levels. At 24 and above R's qr() finds the Vandermonde of
# the centred scores numerically rank deficient - rank 22 of 24, 25 of 30,
# 42 of 95 - and pivots its columns, so what comes back is an orthonormal
# basis in an order that is no longer by polynomial degree. Both sides stay
# orthonormal there, which is what the check below tests instead; neither
# is the linear/quadratic/cubic contrast it is named after.
cat("\nstats::contr.poly\n")
# The tolerance here is not the file's, and it is measured rather than derived.
# The two sides factor the same Vandermonde of the centred scores with two
# backward-stable QRs, so what separates their answers is that matrix's
# condition number, which grows fast with the number of levels; and one side is
# always R at float64, so a float32 build is also comparing across precisions.
# The measured largest relative disagreement, and the tolerance each is checked
# against - about ten times the measurement, so the check still has teeth:
#
#   levels      2       3       4       7      12      20
#   float64  1e-16   1e-16   2e-16   3e-15   1e-13   3e-11
#   float32  7e-8    7e-8    6e-8    7e-7    2e-5    5e-2 (not checked)
#
# Twenty levels is dropped at float32: a degree-19 Vandermonde of scores
# running to 19 has no float32 left in it, the disagreement is 5 per cent, and
# a tolerance wide enough to pass it would not be testing anything. What holds
# there instead is orthonormality, checked below.
contr_levels <- if (mreal_bytes == 8L) c(2L, 3L, 4L, 7L, 12L, 20L) else c(2L, 3L, 4L, 7L, 12L)
contr_tolerances <- if (mreal_bytes == 8L) {
    c(1e-15, 1e-15, 2e-15, 3e-14, 1e-12, 3e-10)
} else {
    c(1e-6, 1e-6, 1e-6, 1e-5, 2e-4)
}
for (case in seq_along(contr_levels)) {
    levels <- contr_levels[case]
    got <- .C("c_contr_poly", as.integer(levels), as.double(seq_len(levels)),
              as.integer(1L), out = double(levels * (levels - 1L)))$out
    compare(sprintf("contr.poly(%d)", levels),
            matrix(got, levels, levels - 1L, byrow = TRUE), contr.poly(levels),
            tol = contr_tolerances[case])
}

# Orthonormality holds where agreement no longer does, and is the property
# contrasts are built for, so it is checked well past the pivoting point.
#
# How far past depends on the build, and the ceiling is arithmetic rather than
# a choice. contr.poly at n levels factors the centred scores raised to the
# power n - 1; the centred scores reach (n - 1) / 2, so the top column reaches
# ((n-1)/2)^(n-1), which passes float32's largest value of 3.4e38 at about
# n = 33 and float64's 1.8e308 at about n = 108. Past its own ceiling the
# column comes back as an infinity and the fit asserts, which is the intended
# behaviour for a degree the storage cannot hold.
for (levels in if (mreal_bytes == 8L) c(24L, 40L, 95L) else c(20L, 26L, 30L)) {
    got <- .C("c_contr_poly", as.integer(levels), as.double(seq_len(levels)),
              as.integer(1L), out = double(levels * (levels - 1L)))$out
    ours <- matrix(got, levels, levels - 1L, byrow = TRUE)
    compare(sprintf("contr.poly(%d) orthonormal", levels),
            crossprod(ours), diag(levels - 1L))
}

scores <- c(1, 2, 4, 8, 16)
got <- .C("c_contr_poly", 5L, as.double(scores), 1L, out = double(5 * 4))$out
compare("contr.poly(5, scores = 2^(0:4))",
        matrix(got, 5, 4, byrow = TRUE), contr.poly(5, scores = scores))
got <- .C("c_contr_poly", 5L, as.double(scores), 0L, out = double(25))$out
compare("contr.poly(5, scores, contrasts = FALSE)",
        matrix(got, 5, 5, byrow = TRUE), contr.poly(5, scores = scores, contrasts = FALSE))

cat("\nstats::polym\n")
set.seed(4)
for (case in list(list(nvars = 2L, degree = 3L, n = 24L),
                  list(nvars = 3L, degree = 2L, n = 31L))) {
    n <- case$n; nvars <- case$nvars; degree <- case$degree
    x <- matrix(round(rnorm(n * nvars), 4), n, nvars)
    upper <- choose(degree + nvars, nvars)      # generous: the grid minus the constant
    got <- .C("c_polym", as.integer(n), as.integer(nvars), as.integer(degree),
              as.double(t(x)), ncol = integer(1),
              powers = double(upper * nvars), basis = double(n * upper))
    ncol <- got$ncol
    powers <- matrix(got$powers[seq_len(ncol * nvars)], ncol, nvars, byrow = TRUE)
    basis <- matrix(got$basis[seq_len(n * ncol)], n, ncol, byrow = TRUE)
    reference <- do.call(polym, c(split(x, col(x)), degree = degree))
    # match columns by exponent tuple, so a different enumeration order
    # would be a reordering rather than a failure
    ours <- apply(powers, 1L, paste, collapse = ".")
    compare(sprintf("polym %d vars degree %d", nvars, degree),
            basis, unclass(reference)[, match(ours, colnames(reference)), drop = FALSE])

    newx <- matrix(round(rnorm(9 * nvars), 4), 9, nvars)
    predicted <- .C("c_polym_predict", as.integer(n), as.integer(nvars), as.integer(degree),
                    as.double(t(x)), 9L, as.double(t(newx)),
                    basis = double(9 * ncol))$basis
    reference_predicted <- do.call(polym, c(split(newx, col(newx)),
                                            degree = degree,
                                            list(coefs = attr(reference, "coefs"))))
    compare(sprintf("polym predict %d vars degree %d", nvars, degree),
            matrix(predicted, 9, ncol, byrow = TRUE),
            reference_predicted[, match(ours, colnames(reference_predicted)), drop = FALSE])
}

# --- basis/spline.h: splineDesign

cat("\nsplines::splineDesign\n")

design_cases <- list(
    list(name = "man page knots, ord 4", knots = c(1, 1.8, 3:5, 6.5, 7, 8.1, 9.2, 10),
         x = seq(4, 7, length.out = 23), ord = 4L, derivs = 0L, outer = 0L),
    list(name = "man page knots, ord 3", knots = c(1, 1.8, 3:5, 6.5, 7, 8.1, 9.2, 10),
         x = seq(3, 8.1, length.out = 19), ord = 3L, derivs = 0L, outer = 0L),
    list(name = "man page knots, ord 1", knots = c(1, 1.8, 3:5, 6.5, 7, 8.1, 9.2, 10),
         x = seq(1, 10, length.out = 29), ord = 1L, derivs = 0L, outer = 1L),
    list(name = "PR#16549 kn8.01", knots = c(0, 0, 0, 0, 1, 1, 1, 1),
         x = c(-1, 0, 0.5, 1, 2), ord = 4L, derivs = 0L, outer = 1L),
    list(name = "repeated interior knots", knots = c(0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3),
         x = seq(0, 3, length.out = 37), ord = 4L, derivs = 0L, outer = 0L),
    list(name = "outer.ok, x well outside", knots = c(-3, -3, -2, 0, 2, 3, 3),
         x = seq(-4, 4, by = 0.25), ord = 4L, derivs = 0L, outer = 1L),
    list(name = "first derivative", knots = c(0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4),
         x = seq(0, 4, length.out = 41), ord = 4L, derivs = 1L, outer = 0L),
    list(name = "second derivative", knots = c(0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4),
         x = seq(0, 4, length.out = 41), ord = 4L, derivs = 2L, outer = 0L),
    list(name = "derivs recycled 0,1,2", knots = c(0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4),
         x = seq(0, 4, length.out = 42), ord = 4L, derivs = c(0L, 1L, 2L), outer = 0L)
)

for (case in design_cases) {
    nk <- length(case$knots); nx <- length(case$x)
    got <- .C("c_spline_design", as.integer(nk), as.double(case$knots),
              as.integer(nx), as.double(case$x), as.integer(case$ord),
              as.integer(length(case$derivs)), as.integer(case$derivs),
              as.integer(case$outer), ncol = integer(1), design = double(nx * nk))
    ncol <- got$ncol
    reference <- splineDesign(case$knots, case$x, ord = case$ord, derivs = case$derivs,
                              outer.ok = as.logical(case$outer))
    compare(sprintf("splineDesign: %s", case$name),
            matrix(got$design[seq_len(nx * ncol)], nx, ncol, byrow = TRUE), reference)
}

# Random knot vectors, biased towards ties and towards x on the knots,
# which is where every bug this function has ever had lived.
set.seed(17)
worst_fuzz <- 0
fuzz_failures <- 0L
for (replicate in 1:300) {
    knots <- sort(round(4 * rnorm(4 + rpois(1, lambda = 4))) / 2)
    ord <- sample(1:4, 1L)
    if (length(knots) < ord + 1L) next
    x <- sort(c(unique(knots), runif(20, min(knots) - 1, max(knots) + 1)))
    nk <- length(knots); nx <- length(x)
    got <- .C("c_spline_design", as.integer(nk), as.double(knots), as.integer(nx),
              as.double(x), as.integer(ord), 1L, 0L, 1L,
              ncol = integer(1), design = double(nx * nk))
    ncol <- got$ncol
    if (ncol == 0L) next
    reference <- splineDesign(knots, x, ord = ord, outer.ok = TRUE)
    ours <- matrix(got$design[seq_len(nx * ncol)], nx, ncol, byrow = TRUE)
    both <- is.finite(ours) & is.finite(reference)
    if (any(is.finite(ours) != is.finite(reference))) { fuzz_failures <- fuzz_failures + 1L; next }
    if (any(both))
        worst_fuzz <- max(worst_fuzz, max(abs(ours[both] - reference[both]) /
                                          pmax(abs(reference[both]), 1)))
}
comparisons <- comparisons + 1L
worst_overall <- max(worst_overall, worst_fuzz)
if (fuzz_failures > 0L || worst_fuzz > tolerance) {
    cat(sprintf("  FAIL %-46s relative %.3g, %d shape mismatches\n",
                "splineDesign: 300 random knot vectors", worst_fuzz, fuzz_failures))
    failures <- failures + 1L
} else {
    cat(sprintf("  ok   %-46s relative %.3g\n",
                "splineDesign: 300 random knot vectors", worst_fuzz))
}

# --- basis/spline.h: bs and ns

cat("\nsplines::bs\n")

x_regular <- seq(1.5, 8.5, by = 0.25)
set.seed(13)
x_skewed <- c(rep(0, 44), 1:10, 2 * (6:15), round(1.25^(15:22)))

bs_cases <- list(
    list(name = "knots = 4, degree 3",  x = x_regular, degree = 3L, df = 0L,
         knots = 4, intercept = 0L, boundary = NULL),
    list(name = "intercept, knots = 4", x = x_regular, degree = 3L, df = 0L,
         knots = 4, intercept = 1L, boundary = NULL),
    list(name = "degree 1",             x = x_regular, degree = 1L, df = 0L,
         knots = c(3, 5, 7), intercept = 0L, boundary = NULL),
    list(name = "degree 5",             x = x_regular, degree = 5L, df = 0L,
         knots = c(3, 5, 7), intercept = 0L, boundary = NULL),
    list(name = "df = 7, quantile knots", x = x_regular, degree = 3L, df = 7L,
         knots = NULL, intercept = 0L, boundary = NULL),
    list(name = "df = 7 on a skewed sample", x = as.double(x_skewed), degree = 3L, df = 7L,
         knots = NULL, intercept = 0L, boundary = NULL),
    list(name = "boundary inside the data", x = x_regular, degree = 3L, df = 0L,
         knots = 4, intercept = 0L, boundary = c(2, 8)),
    list(name = "boundary outside the data", x = x_regular, degree = 3L, df = 0L,
         knots = 4, intercept = 0L, boundary = c(1, 9)),
    list(name = "single observation", x = pi, degree = 3L, df = 0L,
         knots = NULL, intercept = 0L, boundary = NULL)
)

for (case in bs_cases) {
    nx <- length(case$x)
    n_knots <- if (is.null(case$knots)) -1L else length(case$knots)
    knots_in <- if (is.null(case$knots)) 0 else case$knots
    has_boundary <- if (is.null(case$boundary)) 0L else 1L
    boundary_in <- if (is.null(case$boundary)) c(0, 0) else case$boundary
    got <- .C("c_bs", as.integer(nx), as.double(case$x), as.integer(case$degree),
              as.integer(case$df), as.integer(n_knots), as.double(knots_in),
              as.integer(case$intercept), as.integer(has_boundary), as.double(boundary_in),
              ncol = integer(1), basis = double(nx * 64L),
              n_iknots = integer(1), iknots = double(64L))
    arguments <- list(x = case$x, degree = case$degree, intercept = as.logical(case$intercept))
    if (!is.null(case$knots)) arguments$knots <- case$knots
    if (case$df > 0L) arguments$df <- case$df
    if (!is.null(case$boundary)) arguments$Boundary.knots <- case$boundary
    reference <- suppressWarnings(do.call(bs, arguments))
    ncol <- got$ncol
    compare(sprintf("bs: %s", case$name),
            matrix(got$basis[seq_len(nx * ncol)], nx, ncol, byrow = TRUE), unclass(reference))
    reference_knots <- attr(reference, "knots")
    compare(sprintf("bs knots: %s", case$name),
            got$iknots[seq_len(got$n_iknots)],
            if (is.null(reference_knots)) numeric(0) else as.vector(reference_knots))

    newx <- seq(min(case$x) - 2, max(case$x) + 2, length.out = 31)
    predicted <- .C("c_bs_predict", as.integer(nx), as.double(case$x), as.integer(case$degree),
                    as.integer(case$df), as.integer(n_knots), as.double(knots_in),
                    as.integer(case$intercept), as.integer(has_boundary), as.double(boundary_in),
                    as.integer(length(newx)), as.double(newx),
                    ncol = integer(1), basis = double(length(newx) * 64L))
    compare(sprintf("bs predict: %s", case$name),
            matrix(predicted$basis[seq_len(length(newx) * predicted$ncol)],
                   length(newx), predicted$ncol, byrow = TRUE),
            unclass(suppressWarnings(predict(reference, newx))))
}

cat("\nsplines::ns\n")

ns_cases <- list(
    list(name = "df = 4",              x = x_regular, df = 4L, knots = NULL, intercept = 0L, boundary = NULL),
    list(name = "df = 4, intercept",   x = x_regular, df = 4L, knots = NULL, intercept = 1L, boundary = NULL),
    list(name = "explicit knots",      x = x_regular, df = 0L, knots = c(3, 5, 7), intercept = 0L, boundary = NULL),
    list(name = "no interior knots",   x = x_regular, df = 0L, knots = NULL, intercept = 0L, boundary = NULL),
    list(name = "boundary outside",    x = x_regular, df = 0L, knots = c(3, 5, 7), intercept = 0L, boundary = c(1, 9)),
    list(name = "boundary inside",     x = x_regular, df = 0L, knots = c(3, 5, 7), intercept = 0L, boundary = c(2, 8)),
    list(name = "df = 5 on a skewed sample", x = as.double(x_skewed), df = 5L, knots = NULL, intercept = 0L, boundary = NULL),
    list(name = "single observation",  x = pi, df = 0L, knots = NULL, intercept = 0L, boundary = NULL)
)

for (case in ns_cases) {
    nx <- length(case$x)
    n_knots <- if (is.null(case$knots)) -1L else length(case$knots)
    knots_in <- if (is.null(case$knots)) 0 else case$knots
    has_boundary <- if (is.null(case$boundary)) 0L else 1L
    boundary_in <- if (is.null(case$boundary)) c(0, 0) else case$boundary
    got <- .C("c_ns", as.integer(nx), as.double(case$x), as.integer(case$df),
              as.integer(n_knots), as.double(knots_in), as.integer(case$intercept),
              as.integer(has_boundary), as.double(boundary_in),
              ncol = integer(1), basis = double(nx * 64L),
              n_iknots = integer(1), iknots = double(64L))
    arguments <- list(x = case$x, intercept = as.logical(case$intercept))
    if (!is.null(case$knots)) arguments$knots <- case$knots
    if (case$df > 0L) arguments$df <- case$df
    if (!is.null(case$boundary)) arguments$Boundary.knots <- case$boundary
    reference <- suppressWarnings(do.call(ns, arguments))
    ncol <- got$ncol
    compare(sprintf("ns: %s", case$name),
            matrix(got$basis[seq_len(nx * ncol)], nx, ncol, byrow = TRUE), unclass(reference))
    reference_knots <- attr(reference, "knots")
    compare(sprintf("ns knots: %s", case$name),
            got$iknots[seq_len(got$n_iknots)],
            if (is.null(reference_knots)) numeric(0) else as.vector(reference_knots))

    newx <- seq(min(case$x) - 2, max(case$x) + 2, length.out = 31)
    predicted <- .C("c_ns_predict", as.integer(nx), as.double(case$x), as.integer(case$df),
                    as.integer(n_knots), as.double(knots_in), as.integer(case$intercept),
                    as.integer(has_boundary), as.double(boundary_in),
                    as.integer(length(newx)), as.double(newx),
                    ncol = integer(1), basis = double(length(newx) * 64L))
    compare(sprintf("ns predict: %s", case$name),
            matrix(predicted$basis[seq_len(length(newx) * predicted$ncol)],
                   length(newx), predicted$ncol, byrow = TRUE),
            unclass(suppressWarnings(predict(reference, newx))))
}

# --- basis/spline.h: the spline objects

cat("\nsplines::interpSpline, backSpline, periodicSpline\n")

# The interpolating splines carry a build-aware tolerance, and only at float32.
# Their coefficient table holds the value, slope, half the curvature and a sixth
# of the third derivative at each knot, all read out of one dense solve of an
# (n+2) x (n+2) collocation system; each derivative order is one more round of
# differencing on top of that solve, and at float32 each one costs about a
# digit. Measured on this build, the largest relative disagreement with R over
# the four cases below is 3.6e-6 on values, 7.7e-6 on first derivatives and
# 1.0e-5 to 6.1e-5 on second derivatives. At float64 every one of them is at
# the file's own tolerance and none of this applies.
derivative_tolerance <- function(deriv)
    if (mreal_bytes == 8L) tolerance else tolerance * 10^deriv
coefficient_tolerance <- if (mreal_bytes == 8L) tolerance else 1e-4

interp_cases <- list(
    list(name = "women", x = as.double(women$height), y = as.double(women$weight)),
    list(name = "four points", x = c(0, 1, 3, 4), y = c(1, 2, 0.5, 3)),
    list(name = "unequal spacing", x = c(0, 0.1, 0.5, 2, 6, 6.2, 11),
         y = c(3, 1, 4, 1, 5, 9, 2)),
    list(name = "monotone cubic", x = seq(-2, 2, length.out = 12),
         y = seq(-2, 2, length.out = 12)^3 + 10 * seq(-2, 2, length.out = 12))
)

for (case in interp_cases) {
    n <- length(case$x)
    reference_b <- interpSpline(case$x, case$y, bSpline = TRUE)
    got <- .C("c_interp_spline", as.integer(n), as.double(case$x), as.double(case$y),
              knots = double(n + 6L), coefficients = double(n + 2L))
    compare(sprintf("interpSpline knots: %s", case$name), got$knots, reference_b$knots)
    compare(sprintf("interpSpline coefficients: %s", case$name),
            got$coefficients, reference_b$coefficients)

    reference_p <- interpSpline(case$x, case$y)
    got_p <- .C("c_interp_spline_poly", as.integer(n), as.double(case$x), as.double(case$y),
                knots = double(n), coefficients = double(n * 4L))
    compare(sprintf("polySpline knots: %s", case$name), got_p$knots, reference_p$knots)
    compare(sprintf("polySpline coefficients: %s", case$name),
            matrix(got_p$coefficients, n, 4L, byrow = TRUE), reference_p$coefficients,
            tol = coefficient_tolerance)

    at <- seq(min(case$x) - 1, max(case$x) + 1, length.out = 41)
    for (deriv in 0:2) {
        values <- .C("c_interp_spline_predict", as.integer(n), as.double(case$x),
                     as.double(case$y), as.integer(length(at)), as.double(at),
                     as.integer(deriv), out = double(length(at)))$out
        compare(sprintf("polySpline predict deriv %d: %s", deriv, case$name),
                values, predict(reference_p, at, deriv = deriv)$y,
                tol = derivative_tolerance(deriv))

        values_b <- .C("c_bspline_predict", as.integer(n), as.double(case$x),
                       as.double(case$y), as.integer(length(at)), as.double(at),
                       as.integer(deriv), out = double(length(at)))$out
        compare(sprintf("bSpline predict deriv %d: %s", deriv, case$name),
                values_b, predict(reference_b, at, deriv = deriv)$y,
                tol = derivative_tolerance(deriv))
    }
}

# backSpline needs a monotone spline, so only the monotone case above
monotone <- interp_cases[[4L]]
n <- length(monotone$x)
reference_back <- backSpline(interpSpline(monotone$x, monotone$y))
got_back <- .C("c_back_spline", as.integer(n), as.double(monotone$x), as.double(monotone$y),
               knots = double(n), coefficients = double(n * 4L))
compare("backSpline knots", got_back$knots, reference_back$knots)
compare("backSpline coefficients",
        matrix(got_back$coefficients, n, 4L, byrow = TRUE), reference_back$coefficients)

periodic_x <- seq(0, 2 * pi, length.out = 13)[-13L]
periodic_y <- sin(periodic_x) + 0.3 * cos(3 * periodic_x)
for (ord in c(2L, 4L, 6L)) {
    n <- length(periodic_x)
    reference_periodic <- periodicSpline(periodic_x, periodic_y, period = 2 * pi, ord = ord)
    got <- .C("c_periodic_spline", as.integer(n), as.double(periodic_x), as.double(periodic_y),
              as.double(2 * pi), as.integer(ord),
              knots = double(n + 2L * ord - 1L), coefficients = double(n + ord - 1L))
    compare(sprintf("periodicSpline knots: ord %d", ord), got$knots, reference_periodic$knots)
    compare(sprintf("periodicSpline coefficients: ord %d", ord),
            got$coefficients, reference_periodic$coefficients)

    at <- seq(-2 * pi, 4 * pi, length.out = 51)
    values <- .C("c_periodic_spline_predict", as.integer(n), as.double(periodic_x),
                 as.double(periodic_y), as.double(2 * pi), as.integer(ord),
                 as.integer(length(at)), as.double(at), 0L, out = double(length(at)))$out
    compare(sprintf("periodicSpline predict: ord %d", ord),
            values, predict(reference_periodic, at)$y)
}

cat(sprintf("\n%s, %d comparisons, %d failures, worst relative disagreement %.3g\n",
            if (failures) "FAILED" else "PASSED", comparisons, failures, worst_overall))
quit(status = if (failures) 1L else 0L)
