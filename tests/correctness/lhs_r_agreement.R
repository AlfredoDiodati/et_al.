# lhs.h against R's lhs package: do the two produce the same distribution
# of designs?
#
# They cannot produce the same designs. The generators differ (PCG64 here,
# Mersenne-Twister in R) and the permutation is drawn differently (Fisher-
# Yates here, sorted uniforms there), so a seed does not carry across and
# nothing here compares two matrices for equality. What is compared is the
# law: every statistic of a design whose distribution the construction
# fixes must have the same distribution on both sides.
#
# R is a development-tier dependency. `make test` never runs this file;
# `make test-lhs-r` does. The shared object it drives is built from
# tests/correctness/lhs_r_agreement.c and exposes plain C symbols, so
# nothing in the library knows about R.
#
# Every statistic below is computed by the R code in this file, applied to
# both arms, so a disagreement is a disagreement about the designs and not
# about how the statistic was implemented. Sample sizes, seeds and the
# rejection threshold are fixed, and the report names all of them.
#
# Usage:
#   make test-lhs-r
#   Rscript tests/correctness/lhs_r_agreement.R            # same thing
#   MAT_DOUBLE=1 make test-lhs-r                           # float64 build

suppressPackageStartupMessages(library(lhs))

# The repository root, taken from the script's own path so the run does not
# depend on the working directory it was started from.
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
if (length(script_path) == 1L) {
  root <- normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  root <- normalizePath(".")
}

shared_object <- file.path(root, "liblhsagreement.so")
stopifnot(system2("make", c("-C", root, "liblhsagreement.so"), stdout = FALSE) == 0L)
dyn.load(shared_object)

mreal_bytes <- .C("c_mreal_bytes", out = integer(1))$out
build <- if (mreal_bytes == 8L) "float64" else "float32"

# The threshold every p-value in this file is judged against. The null is
# exact equality of the two laws, so a passing statistic has a p-value
# drawn from U(0,1) and this rejects one run in ten thousand per test; the
# alternatives these tests exist to catch (a permutation that is not
# uniform, jitter that is not, a stratum holding two points) land many
# orders of magnitude below it.
alpha <- 1e-4

# Number of designs behind each comparison. Chosen so the count tests have
# hundreds of designs per cell and the KS comparisons have thousands of
# replicates, while the whole file still runs in under a minute.
n_pooled_designs <- 500L      # 50 x 4 designs, pooled coordinates
n_permutation_designs <- 48000L  # 4 x 1 designs, all 24 orderings counted
n_pairing_designs <- 40000L   # 2 x 2 designs, all 4 column pairings counted
n_joint_designs <- 2000L      # 20 x 5 designs, one summary statistic each
n_structure_designs <- 20L    # 1000 x 9 designs, the size the design table uses

seed_ours <- 20240501L
seed_r <- 20240502L

ours <- function(seed, designs, n, k) {
  flat <- .C("c_lhs_random_many",
             as.integer(seed), as.integer(designs), as.integer(n), as.integer(k),
             out = double(as.double(designs) * n * k))$out
  # C writes row-major within a design; R fills column-major, so the array
  # arrives transposed and is put back with aperm.
  aperm(array(flat, dim = c(k, n, designs)), c(2, 1, 3))
}

theirs <- function(seed, designs, n, k) {
  set.seed(seed)
  out <- array(0, dim = c(n, k, designs))
  for (d in seq_len(designs)) out[, , d] <- randomLHS(n, k)
  out
}

# --- statistics, each applied identically to both arms

strata_of <- function(design, n) floor(design * n)

is_latin <- function(design) {
  n <- nrow(design)
  s <- strata_of(design, n)
  all(apply(s, 2, function(column) identical(sort(column), as.double(seq_len(n) - 1))))
}

jitter_of <- function(design) {
  n <- nrow(design)
  design * n - strata_of(design, n)
}

min_pairwise_distance <- function(design) min(dist(design))

max_abs_column_correlation <- function(design) {
  correlations <- cor(design)
  max(abs(correlations[upper.tri(correlations)]))
}

# Hickernell's centered L2 discrepancy: the standard scalar measure of how
# far a design is from covering the box uniformly, and a summary of the
# whole design rather than of one column.
centered_discrepancy <- function(design) {
  n <- nrow(design); k <- ncol(design)
  offset <- abs(design - 0.5)
  single <- (2 / n) * sum(apply(1 + 0.5 * offset - 0.5 * offset^2, 1, prod))
  double_sum <- 0
  for (i in seq_len(n)) {
    row_offset <- matrix(offset[i, ], n, k, byrow = TRUE)
    row_value <- matrix(design[i, ], n, k, byrow = TRUE)
    terms <- 1 + 0.5 * row_offset + 0.5 * offset - 0.5 * abs(row_value - design)
    double_sum <- double_sum + sum(apply(terms, 1, prod))
  }
  sqrt(max((13 / 12)^k - single + double_sum / n^2, 0))
}

apply_designs <- function(cube, f) apply(cube, 3, function(design) f(design))

# --- the report

lines <- character()
failures <- 0L

# Echoed as it is produced as well as collected, so a run that dies part
# way through still shows what it had established. A whole mutation check
# was once lost to the opposite arrangement: a jitter fixed at the stratum
# midpoint made one set of quantile bin edges collapse onto each other,
# cut() stopped the script, and the report that would have named the cause
# had never been written.
say <- function(...) {
  line <- paste0(...)
  lines <<- c(lines, line)
  cat(line, "\n", sep = "")
}

record <- function(name, pvalue, detail = "") {
  ok <- is.finite(pvalue) && pvalue > alpha
  if (!ok) failures <<- failures + 1L
  say(sprintf("  %-46s p = %-10.4g %s%s", name, pvalue,
              if (ok) "ok" else "FAIL", if (nzchar(detail)) paste0("  [", detail, "]") else ""))
  invisible(ok)
}

record_exact <- function(name, ok, detail = "") {
  if (!ok) failures <<- failures + 1L
  say(sprintf("  %-46s %-14s %s%s", name, "", if (ok) "ok" else "FAIL",
              if (nzchar(detail)) paste0("  [", detail, "]") else ""))
  invisible(ok)
}

# A statistic that cannot be computed at all is reported as p = 0 rather
# than allowed to stop the run: every way these tests fail to evaluate -
# a sample with no spread, bin edges that collapse together, a table with
# an empty row - is itself a disagreement with R, and the report has to say
# which test it was.
p_value_or_zero <- function(expr) {
  value <- tryCatch(suppressWarnings(expr), error = function(e) 0)
  if (is.numeric(value) && length(value) == 1L && is.finite(value)) value else 0
}

# Two-sample Kolmogorov-Smirnov. Ties are possible once one arm is stored
# at float32, and ks.test warns rather than failing on them; the warning is
# suppressed because the binned tests beside every KS here are tie-free and
# carry the same claim.
ks_uniform <- function(x) p_value_or_zero(ks.test(x, "punif")$p.value)

ks_two_sample <- function(x, y) p_value_or_zero(ks.test(x, y)$p.value)

binned_homogeneity <- function(x, y, breaks) {
  breaks <- unique(breaks)
  if (length(breaks) < 3L) return(0)
  p_value_or_zero(chisq.test(rbind(table(cut(x, breaks)), table(cut(y, breaks))))$p.value)
}

count_homogeneity <- function(left_counts, right_counts) {
  p_value_or_zero(chisq.test(rbind(left_counts, right_counts))$p.value)
}

count_uniformity <- function(counts) p_value_or_zero(chisq.test(counts)$p.value)

independence <- function(x, y) p_value_or_zero(cor.test(x, y)$p.value)

say(sprintf("lhs.h against R's lhs %s: agreement of the design distribution",
            as.character(packageVersion("lhs"))))
say("")
say(sprintf("  run          %s", format(Sys.time())))
say(sprintf("  R            %s", R.version.string))
say(sprintf("  build        %s (mreal is %d bytes)", build, mreal_bytes))
say(sprintf("  seeds        ours %d, R %d", seed_ours, seed_r))
say(sprintf("  threshold    reject at p < %g", alpha))
say("")
say("Nothing here compares two designs for equality: the generators differ, so")
say("only the distribution of a design can agree. Each statistic is computed by")
say("this script for both arms and the two samples of it are compared.")
say("")

# --- 1. structure, at the size a parameter design table actually uses

say(sprintf("1. structure of %d designs per arm at n = 1000, k = 9", n_structure_designs))
our_structure <- ours(seed_ours, n_structure_designs, 1000L, 9L)
their_structure <- theirs(seed_r, n_structure_designs, 1000L, 9L)

for (arm in c("ours", "R")) {
  cube <- if (arm == "ours") our_structure else their_structure
  record_exact(sprintf("%s: every column is a Latin permutation", arm),
               all(apply_designs(cube, is_latin)))
  record_exact(sprintf("%s: every coordinate in (0, 1)", arm),
               all(cube > 0 & cube < 1))
  record_exact(sprintf("%s: no duplicated rows", arm),
               all(apply_designs(cube, function(d) anyDuplicated(as.data.frame(d)) == 0L)))
}
say("")

# --- 2. pooled coordinates and within-stratum jitter

say(sprintf("2. pooled coordinates and jitter, %d designs per arm at n = 50, k = 4",
            n_pooled_designs))
our_pooled <- ours(seed_ours + 1L, n_pooled_designs, 50L, 4L)
their_pooled <- theirs(seed_r + 1L, n_pooled_designs, 50L, 4L)

our_values <- as.vector(our_pooled)
their_values <- as.vector(their_pooled)
our_jitter <- as.vector(apply(our_pooled, 3, jitter_of))
their_jitter <- as.vector(apply(their_pooled, 3, jitter_of))

say(sprintf("   %d coordinates per arm", length(our_values)))
record("coordinates: ours vs U(0,1), KS", ks_uniform(our_values))
record("coordinates: R vs U(0,1), KS", ks_uniform(their_values))
record("coordinates: ours vs R, KS", ks_two_sample(our_values, their_values))
record("coordinates: ours vs R, 20 bins",
       binned_homogeneity(our_values, their_values, seq(0, 1, length.out = 21)))
record("jitter: ours vs U(0,1), KS", ks_uniform(our_jitter))
record("jitter: R vs U(0,1), KS", ks_uniform(their_jitter))
record("jitter: ours vs R, KS", ks_two_sample(our_jitter, their_jitter))
record("jitter: ours vs R, 20 bins",
       binned_homogeneity(our_jitter, their_jitter, seq(0, 1, length.out = 21)))

# The jitter must not depend on which stratum it landed in - the two are
# drawn independently, so their correlation is zero on both sides.
our_strata <- as.vector(apply(our_pooled, 3, function(d) strata_of(d, 50)))
their_strata <- as.vector(apply(their_pooled, 3, function(d) strata_of(d, 50)))
record("stratum vs jitter independence: ours", independence(our_strata, our_jitter))
record("stratum vs jitter independence: R", independence(their_strata, their_jitter))
say("")

# --- 3. the permutation law itself, counted exactly

say(sprintf("3. all 4! orderings, %d designs per arm at n = 4, k = 1",
            n_permutation_designs))
ordering_index <- function(cube) {
  strata <- apply(cube, 3, function(d) strata_of(d, 4))
  apply(strata, 2, function(p) paste(p, collapse = ""))
}
our_orderings <- table(factor(ordering_index(ours(seed_ours + 2L, n_permutation_designs, 4L, 1L))))
their_orderings <- table(factor(ordering_index(theirs(seed_r + 2L, n_permutation_designs, 4L, 1L))))

record_exact("both arms produced all 24 orderings",
             length(our_orderings) == 24L && length(their_orderings) == 24L,
             sprintf("ours %d, R %d", length(our_orderings), length(their_orderings)))
common <- union(names(our_orderings), names(their_orderings))
our_counts <- as.vector(our_orderings[common]); our_counts[is.na(our_counts)] <- 0L
their_counts <- as.vector(their_orderings[common]); their_counts[is.na(their_counts)] <- 0L
record("orderings: ours uniform over 24", count_uniformity(our_counts))
record("orderings: R uniform over 24", count_uniformity(their_counts))
record("orderings: ours vs R", count_homogeneity(our_counts, their_counts))
say("")

# --- 4. columns permuted independently of each other

say(sprintf("4. all 4 column pairings, %d designs per arm at n = 2, k = 2",
            n_pairing_designs))
pairing_index <- function(cube) {
  apply(cube, 3, function(d) paste(strata_of(d, 2)[1, ], collapse = ""))
}
our_pairings <- table(factor(pairing_index(ours(seed_ours + 3L, n_pairing_designs, 2L, 2L)),
                             levels = c("00", "01", "10", "11")))
their_pairings <- table(factor(pairing_index(theirs(seed_r + 3L, n_pairing_designs, 2L, 2L)),
                               levels = c("00", "01", "10", "11")))
record("pairings: ours uniform over 4", count_uniformity(as.vector(our_pairings)))
record("pairings: R uniform over 4", count_uniformity(as.vector(their_pairings)))
record("pairings: ours vs R", count_homogeneity(as.vector(our_pairings), as.vector(their_pairings)))
say("")

# --- 5. whole-design summary statistics

say(sprintf("5. design summary statistics, %d designs per arm at n = 20, k = 5",
            n_joint_designs))
our_joint <- ours(seed_ours + 4L, n_joint_designs, 20L, 5L)
their_joint <- theirs(seed_r + 4L, n_joint_designs, 20L, 5L)

summaries <- list(
  "minimum pairwise distance" = min_pairwise_distance,
  "maximum absolute column correlation" = max_abs_column_correlation,
  "centered L2 discrepancy" = centered_discrepancy,
  "mean of the first column" = function(d) mean(d[, 1]),
  "standard deviation of the first column" = function(d) sd(d[, 1])
)

for (name in names(summaries)) {
  our_values <- apply_designs(our_joint, summaries[[name]])
  their_values <- apply_designs(their_joint, summaries[[name]])
  record(sprintf("%s, KS", name), ks_two_sample(our_values, their_values),
         sprintf("means %.5f vs %.5f", mean(our_values), mean(their_values)))
  record(sprintf("%s, 10 bins", name),
         binned_homogeneity(our_values, their_values,
                            quantile(c(our_values, their_values),
                                     probs = seq(0, 1, length.out = 11))))
}
say("")

# --- 6. lhs_scale against the two sweep() calls it replaces

say("6. lhs_scale against sweep(), on one design handed to both")
# Nine parameter ranges, deliberately awkward rather than round: one
# negative and narrow, several wide, none symmetric about zero.
lower <- c(0.05, -1.50, 0.10, 0.10, 0.10, 1.00, 0.00, 0.50, 0.50)
upper <- c(0.25, -1.25, 0.50, 0.50, 0.50, 1.50, 0.50, 0.95, 0.95)
unit <- ours(seed_ours + 5L, 1L, 1000L, 9L)[, , 1]

scaled_in_c <- matrix(
  .C("c_lhs_scale", as.integer(nrow(unit)), as.integer(ncol(unit)),
     as.double(t(unit)), as.double(lower), as.double(upper),
     out = double(length(unit)))$out,
  nrow = nrow(unit), ncol = ncol(unit), byrow = TRUE)

scaled_in_r <- sweep(sweep(unit, 2, upper - lower, "*"), 2, lower, "+")

# One rounding of the affine map at the build's own precision, relative to
# the width of each parameter's range.
tolerance <- if (mreal_bytes == 8L) 1e-12 else 1e-6
relative_gap <- max(abs(scaled_in_c - scaled_in_r) / rep(upper - lower, each = nrow(unit)))
record_exact("lhs_scale agrees with sweep()", relative_gap <= tolerance,
             sprintf("largest relative gap %.3g, tolerance %g", relative_gap, tolerance))
record_exact("scaled design stays inside its bounds",
             all(sweep(scaled_in_c, 2, lower, ">=") & sweep(scaled_in_c, 2, upper, "<=")))
say("")

# --- verdict

say(sprintf("%s, %d failures", if (failures == 0L) "PASSED" else "FAILED", failures))

dir.create(file.path(root, "out"), showWarnings = FALSE, recursive = TRUE)
report <- file.path(root, "out", "lhs_r_agreement_report.txt")
writeLines(lines, report)

cat("report written to ", report, "\n", sep = "")

if (failures > 0L) quit(status = 1L)
