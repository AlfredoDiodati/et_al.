# Does inference/seri_mcs.h find what the authors' own R code finds?
#
# The reference is tests/correctness/seri_mcs_authors_code.R, the code Seri,
# Martinoli, Secchi and Centorrino (2021) published with the paper, from
# their repository https://github.com/mmartinoli87/Model-Confidence-Set; the
# data, examples/datasets/seri_2021_simulation_runs.npz, holds that
# repository's Data.csv as float32 members (scripts/seri_runs_to_npz.c), and
# read_runs below returns the data frame read.csv returns for Data.csv. Its
# function definitions (fnComplete, fnTest, fnElim, fnMCS) and its
# construction of the losses from the simulated runs are evaluated as they
# are written. Three things in that file are not run: reading Data.csv
# (read_runs reads the .npz instead), attach(), and the plot. The one line that needs the data.table package,
# the mean and the variance of each configuration's losses, is replaced by
# base R's mean() and var(), the functions that line calls, grouped with
# tapply(); the configurations come out in the same order, 1 to 16.
#
# What is compared, in the float64 and the float32 build of the shared
# object built from tests/correctness/seri_mcs_r_agreement.c; in the
# float32 build the authors' functions are given the losses rounded to
# single precision, which is what that build stores:
#
# 1. The paper's example as the authors run it: their losses, variance with
#    divisor n - 1 (R's var), the whole elimination path (dA = 1). Their
#    order of elimination against seri_mcs's, which uses divisor n as the
#    paper states; the p-values differ by that divisor and are reported.
# 2. The paper's example and random designs with the variance given to the
#    authors' functions with divisor n, so both sides compute the same
#    quantity. Asserted: the same elimination order; every round's W (from
#    fnTest, which solves the paper's matrix form) and p-value to a
#    tolerance; each model's mean and variance; seri_mcs_statistic against
#    fnTest on sets that are not suffixes of the elimination order; and at
#    alpha = 0.01, 0.05 and 0.10 the models fnMCS eliminates, their MCS
#    p-values and the set.
#
# Two behaviours of fnMCS are counted and reported rather than asserted:
# - when the test accepts at the first or the second round, fnMCS stops
#   with an error ("incorrect number of dimensions"), since it subsets its
#   matrix of results to one row or none and then indexes a column;
# - when every test rejects down to one model, it returns each round's own
#   p-value without the running maximum that makes it an MCS p-value. The
#   comparison then uses seri_mcs's round p-values, and how often the two
#   differ from the MCS p-values is counted.
# fnMCS also holds its results in a matrix of 16 rows, so it errors past 16
# models; the random designs have 2 to 16. Between equal means, which.max
# in fnElim drops the smaller column index and seri_mcs the larger; the
# random designs have continuous losses, and a tied case is reported.
#
# Usage:
#   make test-seri-mcs-r
# Writes out/seri_mcs_r_agreement_report.txt and exits with status 1 if an
# assertion fails.

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- if (length(script_path) == 1L) normalizePath(file.path(dirname(script_path), "..", "..")) else normalizePath(".")
stopifnot(system2("make", c("-C", root, "libserimcs_f64.so", "libserimcs_f32.so"), stdout = FALSE) == 0L)
builds <- c(float64 = "libserimcs_f64", float32 = "libserimcs_f32")
for (b in builds) dyn.load(file.path(root, paste0(b, ".so")))

# The simulated runs, examples/datasets/seri_2021_simulation_runs.npz, read
# through the float64 shared object, since R has no .npz reader. The columns
# are integers, so they are returned as integers, the types read.csv gives
# the authors' Data.csv.
read_runs <- function(root) {
  path <- file.path(root, "examples", "datasets", "seri_2021_simulation_runs.npz")
  rows <- .C("c_seri_runs_rows", path, rows = integer(1), PACKAGE = "libserimcs_f64")$rows
  runs <- .C("c_seri_runs", path, as.integer(rows), COP = double(rows), run = double(rows), step = double(rows),
             solved = double(rows), PACKAGE = "libserimcs_f64")
  data.frame(COP = as.integer(runs$COP), run = as.integer(runs$run), step = as.integer(runs$step),
             pr.solved2 = as.integer(runs$solved))
}

dir.create(file.path(root, "out"), showWarnings = FALSE)
report_path <- file.path(root, "out", "seri_mcs_r_agreement_report.txt")
report <- file(report_path, "w")
say <- function(...) writeLines(sprintf(...), report)
failures <- 0L
fail <- function(...) {
  failures <<- failures + 1L
  say(paste("  FAIL", sprintf(...)))
}

# The authors' file, expression by expression.
authors <- new.env()
authors$mData <- read_runs(root)
skipped <- character(0)
for (expression in parse(file.path(root, "tests", "correctness", "seri_mcs_authors_code.R"), keep.source = FALSE)) {
  text <- paste(deparse(expression), collapse = " ")
  if (grepl("read.csv|attach\\(|library\\(|pdf\\(|plot\\(|axis\\(|segments\\(|abline\\(|dev.off", text)) {
    skipped <- c(skipped, substr(text, 1, 40))
  } else if (grepl("setDT", text)) {
    with(authors, mD <- data.frame(COP = 1:16, mean = as.vector(tapply(mDist$Dist, mDist$COP, mean)),
                                   var = as.vector(tapply(mDist$Dist, mDist$COP, var))))
  } else {
    eval(expression, authors)
  }
}
# read.csv, attach, library, pdf, plot, two axis calls, the loop of
# segments, two ablines and dev.off.
stopifnot(length(skipped) == 11L)

# seri_mcs through the shared object of one build.
ours <- function(build, losses, alpha) {
  n <- nrow(losses)
  m <- ncol(losses)
  out <- .C("c_seri_mcs", as.integer(n), as.integer(m), as.double(losses), as.double(alpha),
            L_bar = double(m), sigma2 = double(m), round_eliminated = integer(m - 1),
            round_statistic = double(m - 1), round_pvalue = double(m - 1), pvalue = double(m),
            in_set = integer(m), PACKAGE = builds[[build]])
  out$round_eliminated <- out$round_eliminated + 1L
  out
}

# The authors' functions on means and variances, with the global sample
# size fnMCS reads.
theirs_path <- function(L_bar, sigma2, n) {
  authors$iN <- n
  m <- length(L_bar)
  path <- authors$fnMCS(L_bar, sigma2, 1:m, 1, verbose = 0)[1:m, , drop = FALSE]
  # W of every round's set, from fnTest on that set.
  W <- numeric(m - 1)
  surviving <- 1:m
  for (k in 1:(m - 1)) {
    W[k] <- authors$fnTest(L_bar[surviving], sigma2[surviving], n, 1)$test
    surviving <- surviving[surviving != path[k, 1]]
  }
  list(order = path[, 1], p = path[, 2], W = W)
}

relative_error <- function(a, b) max(abs(a - b) / pmax(1, abs(b)))

# Tolerances, relative to the larger of 1 and the reference value. The
# moments: the float32 build rounds each variance to single precision. The
# procedure, W and the p-values, gets the same means and variances on both
# sides in either build.
tolerance <- c(float64 = 1e-12, float32 = 1e-6)
procedure_tolerance <- 1e-12
largest <- list()
note_error <- function(build, what, value) {
  key <- paste(build, what)
  largest[[key]] <<- max(if (is.null(largest[[key]])) 0 else largest[[key]], value)
}

# Every loss rounded to single precision, as the float32 build stores it.
to_single <- function(x) {
  rounded <- readBin(writeBin(as.vector(x), raw(), size = 4), "double", size = 4, n = length(x))
  matrix(rounded, nrow(x), ncol(x))
}

# Compares one dataset; label names it in a failure. In the float32 build
# both sides get the losses rounded to single precision, so they compare
# the same data: on the design with means at 1000 within 1e-3 of each
# other, single precision moves the losses by more than the means differ.
compare <- function(build, losses, label) {
  if (build == "float32") losses <- to_single(losses)
  n <- nrow(losses)
  m <- ncol(losses)
  R_L_bar <- colMeans(losses)
  R_sigma2 <- apply(losses, 2, var) * (n - 1) / n
  mine <- ours(build, losses, 0.05)
  tol <- tolerance[[build]]
  e <- relative_error(mine$L_bar, R_L_bar); note_error(build, "mean", e)
  if (e > tol) fail("%s %s: means differ by %.3g", build, label, e)
  e <- relative_error(mine$sigma2, R_sigma2); note_error(build, "variance", e)
  if (e > tol) fail("%s %s: variances differ by %.3g", build, label, e)
  # The authors' functions get the means and variances seri_mcs computed,
  # so what is compared below is the procedure on identical inputs.
  L_bar <- mine$L_bar
  sigma2 <- mine$sigma2
  # Equal means are broken differently by the two sides (section 3), and
  # fnMCS at dA = 1 stops at a set whose means are all equal, where W = 0.
  if (anyDuplicated(L_bar) || anyDuplicated(R_L_bar)) {
    counts$tied[[build]] <<- counts$tied[[build]] + 1L
    return(invisible(NULL))
  }
  theirs <- theirs_path(L_bar, sigma2, n)
  # Reported only: the authors' functions on R's own means and variances.
  # colMeans adds in extended precision and seri_mcs in double, and where
  # the means are close to each other and far from zero W magnifies that.
  theirs_own <- theirs_path(R_L_bar, R_sigma2, n)
  if (identical(theirs_own$order, theirs$order)) note_error(build, "W, R's own moments", relative_error(mine$round_statistic, theirs_own$W))
  else counts$own_moments_order[[build]] <<- counts$own_moments_order[[build]] + 1L
  if (!identical(as.integer(theirs$order[1:(m - 1)]), mine$round_eliminated)) {
    fail("%s %s: elimination order %s against the authors' %s", build, label,
         paste(mine$round_eliminated, collapse = " "), paste(theirs$order[1:(m - 1)], collapse = " "))
    return(invisible(NULL))
  }
  e <- relative_error(mine$round_statistic, theirs$W); note_error(build, "W", e)
  if (e > procedure_tolerance) fail("%s %s: W differs by %.3g", build, label, e)
  e <- max(abs(mine$round_pvalue - theirs$p[1:(m - 1)])); note_error(build, "round p-value", e)
  if (e > procedure_tolerance) fail("%s %s: round p-values differ by %.3g", build, label, e)

  # seri_mcs_statistic on sets that are not suffixes of the elimination
  # order: the odd-numbered models, the even-numbered ones, and all of them
  # listed in reverse, against fnTest on the same set.
  for (set in list(seq(1L, m, 2L), seq(2L, m, 2L), m:1)) {
    if (length(set) < 2L) next
    W_set <- .C("c_seri_mcs_statistic", as.integer(n), L_bar, sigma2, as.integer(set), as.integer(length(set)),
                W = double(1), PACKAGE = builds[[build]])$W
    W_reference <- authors$fnTest(L_bar[set], sigma2[set], n, 1)$test
    e <- relative_error(W_set, W_reference); note_error(build, "W of other sets", e)
    if (e > procedure_tolerance) fail("%s %s: seri_mcs_statistic differs from fnTest by %.3g", build, label, e)
  }

  for (alpha in c(0.01, 0.05, 0.10)) {
    mine <- ours(build, losses, alpha)
    authors$iN <- n
    result <- tryCatch(authors$fnMCS(L_bar, sigma2, 1:m, alpha, verbose = 0), error = function(err) err)
    if (inherits(result, "error")) {
      counts$reference_error <<- counts$reference_error + 1L
      # The error is expected only when the set is decided at round 1 or 2.
      decided <- which(mine$round_pvalue >= alpha)[1]
      if (is.na(decided) || decided > 2) fail("%s %s alpha %.2f: fnMCS failed with the set decided at round %s",
                                              build, label, alpha, decided)
      next
    }
    counts$sets <<- counts$sets + 1L
    every_round_rejects <- all(mine$round_pvalue < alpha)
    eliminated <- if (every_round_rejects) result[1:(m - 1), 1] else result[, 1]
    their_p <- if (every_round_rejects) result[1:(m - 1), 2] else result[, 2]
    k <- length(eliminated)
    if (!identical(as.integer(eliminated), mine$round_eliminated[1:k])) {
      fail("%s %s alpha %.2f: eliminated %s against the authors' %s", build, label, alpha,
           paste(mine$round_eliminated[1:k], collapse = " "), paste(eliminated, collapse = " "))
      next
    }
    reference_p <- if (every_round_rejects) mine$round_pvalue else mine$pvalue[eliminated]
    e <- max(abs(reference_p - their_p)); note_error(build, "MCS p-value", e)
    if (e > procedure_tolerance) fail("%s %s alpha %.2f: MCS p-values differ by %.3g", build, label, alpha, e)
    if (every_round_rejects) {
      counts$every_round_rejects <<- counts$every_round_rejects + 1L
      if (any(abs(mine$pvalue[eliminated] - their_p) > procedure_tolerance)) counts$without_running_maximum <<- counts$without_running_maximum + 1L
    }
    their_set <- setdiff(1:m, eliminated)
    if (!identical(which(mine$in_set == 1L), their_set))
      fail("%s %s alpha %.2f: set {%s} against the authors' {%s}", build, label, alpha,
           paste(which(mine$in_set == 1L), collapse = ", "), paste(their_set, collapse = ", "))
  }
}

say("inference/seri_mcs.h against the authors' R code, tests/correctness/seri_mcs_authors_code.R")
say("%s; %s", R.version.string, format(Sys.time(), "%Y-%m-%d %H:%M"))
say("")

# The paper's example, the losses built by the authors' code.
paper_losses <- matrix(authors$mDist$Dist, nrow = 200, ncol = 16)
stopifnot(all(matrix(authors$mDist$COP, 200, 16) == rep(1:16, each = 200)))
paper_n_minus_1 <- authors$mPValue
say("1. The paper's example as the authors run it: variance divisor n - 1, the whole path (dA = 1)")
say("   round, dropped by the authors, their p-value, dropped by seri_mcs (divisor n), its p-value")
for (build in names(builds)) {
  mine <- ours(build, paper_losses, 0.05)
  if (!identical(as.integer(paper_n_minus_1[1:15, 1]), mine$round_eliminated))
    fail("%s paper example: elimination order differs from the authors' run", build)
  if (!identical(which(mine$in_set == 1L), c(8L, 13L, 16L)))
    fail("%s paper example: set at 95%% is {%s}, the paper's is {8, 13, 16}", build, paste(which(mine$in_set == 1L), collapse = ", "))
  if (build == "float64")
    for (k in 1:15) say("   %2d, %2d, %.6g, %2d, %.6g", k, paper_n_minus_1[k, 1], paper_n_minus_1[k, 2],
                        mine$round_eliminated[k], mine$round_pvalue[k])
}
say("   last model: the authors' %d; set at 95%% from seri_mcs: {8, 13, 16} in both builds unless a FAIL line says otherwise", paper_n_minus_1[16, 1])
say("")

counts <- list(designs = 0L, sets = 0L, reference_error = 0L, every_round_rejects = 0L, without_running_maximum = 0L,
               tied = list(float64 = 0L, float32 = 0L), own_moments_order = list(float64 = 0L, float32 = 0L))

say("2. The authors' functions with variance divisor n, against seri_mcs")
for (build in names(builds)) compare(build, paper_losses, "paper example")
authors$iN <- 200
paper_reference <- theirs_path(colMeans(paper_losses), apply(paper_losses, 2, var) * 199 / 200, 200)
say("   paper example, every round from the authors' fnTest and fnMCS (the numbers tests/correctness/seri_mcs_correctness.c checks):")
say("   round, set size, W, p-value, dropped")
for (k in 1:15) say("   %2d, %2d, %.17g, %.17g, %2d", k, 17 - k, paper_reference$W[k], paper_reference$p[k], paper_reference$order[k])

# Random designs. Losses of model g have mean mu_g and their own scale;
# the means are spread so that sets of every size occur.
families <- list(
  normal = function(n, mu, scale) rnorm(n, mu, scale),
  exponential = function(n, mu, scale) mu + scale * (rexp(n) - 1),
  student_t3 = function(n, mu, scale) mu + scale * rt(n, 3) / sqrt(3),
  close_means_far_from_zero = function(n, mu, scale) 1000 + 1e-3 * mu + scale * rnorm(n)
)
sizes <- c(10L, 50L, 200L, 1000L)
seed <- 20211001L
set.seed(seed)
for (family in names(families)) {
  for (n in sizes) {
    for (m in 2:16) {
      for (replication in 1:3) {
        scale <- exp(runif(m, -1, 1))
        mu <- sort(runif(m, 0, 1)) * (4 / sqrt(n)) * mean(scale) * replication
        if (family == "close_means_far_from_zero") scale <- scale * 1e-3 / sqrt(n)
        losses <- sapply(1:m, function(g) families[[family]](n, mu[g], scale[g]))
        counts$designs <- counts$designs + 1L
        label <- sprintf("%s n %d m %d replication %d", family, n, m, replication)
        for (build in names(builds)) compare(build, losses, label)
      }
    }
  }
}
say("   random designs: %d (families %s; n %s; 2 to 16 models; 3 replications each), seed %d, both builds",
    counts$designs, paste(names(families), collapse = ", "), paste(sizes, collapse = ", "), seed)
say("   skipped for exactly equal means: %d in float64, %d in float32 (losses rounded to single precision)",
    counts$tied$float64, counts$tied$float32)
say("   sets compared at alpha 0.01, 0.05, 0.10: %d", counts$sets)
say("   designs where the authors' functions on R's own moments eliminate in another order: %d in float64, %d in float32",
    counts$own_moments_order$float64, counts$own_moments_order$float32)
say("   fnMCS stopped with an error, the set decided at round 1 or 2: %d", counts$reference_error)
say("   every round rejected down to one model: %d, of which fnMCS's p-values lack the running maximum and differ from the MCS p-values: %d",
    counts$every_round_rejects, counts$without_running_maximum)
say("   largest difference, relative to the larger of 1 and the reference value, and the tolerance:")
for (key in names(largest)) {
  if (grepl("own moments", key)) say("     %-32s %.3g (reported, not asserted)", key, largest[[key]])
  else if (grepl("mean|variance", key)) say("     %-32s %.3g (tolerance %.0e)", key, largest[[key]], tolerance[[sub(" .*", "", key)]])
  else say("     %-32s %.3g (tolerance %.0e)", key, largest[[key]], procedure_tolerance)
}
say("")

# Tied means: two identical columns.
say("3. Tied means, reported: two identical columns among four")
set.seed(7L)
tied <- matrix(rnorm(400), 100, 4)
tied[, 3] <- tied[, 4] <- tied[, 2] + 0.5
authors$iN <- 100
tied_theirs <- authors$fnMCS(colMeans(tied), apply(tied, 2, var) * 0.99, 1:4, 1, verbose = 0)
tied_mine <- ours("float64", tied, 0.05)
say("   the authors drop %s, seri_mcs drops %s (columns 3 and 4 are equal; which.max takes the first)",
    paste(tied_theirs[1:3, 1], collapse = " "), paste(tied_mine$round_eliminated, collapse = " "))
say("")

say("%s, %d failures", if (failures == 0L) "PASSED" else "FAILED", failures)
close(report)
cat(readLines(report_path), sep = "\n")
quit(status = if (failures == 0L) 0L else 1L)
