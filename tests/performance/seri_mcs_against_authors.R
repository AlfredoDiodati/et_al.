# How long does seri_mcs() take against the authors' R code, on the same
# losses?
#
# Arms, each timed on the same loss matrix:
# - authors: the per-model mean and variance (mean() and var(), the
#   functions the authors' data.table line calls) and their whole
#   elimination path, fnMCS with dA = 1, from
#   tests/correctness/seri_mcs_authors_code.R. fnMCS holds its results in 16
#   rows and errors past 16 models, so above 16 models the arm runs
#   fnMCS's loop body (fnTest, then fnElim) for every round without that
#   matrix, which is the same arithmetic. The authors' code and the data
#   are from https://github.com/mmartinoli87/Model-Confidence-Set.
# - seri_mcs from R: one .C() call per run, which copies the losses into a
#   DataFrame and runs seri_mcs (tests/correctness/seri_mcs_r_agreement.c,
#   c_seri_mcs), in the float64 and the float32 build.
# - seri_mcs in C: seri_mcs called in a loop inside C on a DataFrame built
#   once (c_seri_mcs_time), the cost of the library call alone.
# Every arm computes every round's test, as seri_mcs always does.
#
# Losses: the paper's example (200 runs by 16 configurations, built by the
# authors' code from examples/datasets/seri_2021_simulation_runs.npz), and
# normal losses with model g's mean 1 + 0.5 g / m and standard deviation 1,
# seed 20211002, at 1000 x 16, 100000 x 16, 1000 x 64 and 1000 x 256.
#
# Each arm runs in batches of at least 50 ms (one call if a call is longer).
# 2 rounds are discarded, then 10 are kept; the arm order is reversed every
# other round. The report gives the median time per call over the kept
# rounds, and the ratio of the authors' arm to each seri_mcs arm.
#
# Usage:
#   make bench-seri_mcs_against_authors
# Writes out/seri_mcs_against_authors_report.txt.

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

# The authors' file, as tests/correctness/seri_mcs_r_agreement.R evaluates it.
authors <- new.env()
authors$mData <- read_runs(root)
for (expression in parse(file.path(root, "tests", "correctness", "seri_mcs_authors_code.R"), keep.source = FALSE)) {
  text <- paste(deparse(expression), collapse = " ")
  if (grepl("read.csv|attach\\(|library\\(|pdf\\(|plot\\(|axis\\(|segments\\(|abline\\(|dev.off|setDT|mD\\$|fnMCS\\(vMean", text)) next
  eval(expression, authors)
}
paper_losses <- matrix(authors$mDist$Dist, nrow = 200, ncol = 16)

authors_arm <- function(losses) {
  n <- nrow(losses)
  m <- ncol(losses)
  vMean <- apply(losses, 2, mean)
  vVar <- apply(losses, 2, var)
  authors$iN <- n
  if (m <= 16) return(authors$fnMCS(vMean, vVar, 1:m, 1, verbose = 0)[1:(m - 1), 1])
  vIndex <- 1:m
  eliminated <- integer(m - 1)
  for (i in 1:(m - 1)) {
    lTest <- authors$fnTest(vMean, vVar, n, 1)
    lElim <- authors$fnElim(vMean, vVar, vIndex)
    vMean <- lElim$mean
    vVar <- lElim$var
    vIndex <- lElim$indices
    eliminated[i] <- lElim$elim
  }
  eliminated
}

from_r_arm <- function(build, losses) {
  n <- nrow(losses)
  m <- ncol(losses)
  .C("c_seri_mcs", as.integer(n), as.integer(m), losses, 0.05,
     L_bar = double(m), sigma2 = double(m), round_eliminated = integer(m - 1),
     round_statistic = double(m - 1), round_pvalue = double(m - 1), pvalue = double(m),
     in_set = integer(m), PACKAGE = builds[[build]])$round_eliminated + 1L
}

in_c_per_call <- function(build, losses, calls) {
  .C("c_seri_mcs_time", as.integer(nrow(losses)), as.integer(ncol(losses)), losses, 0.05,
     as.integer(calls), elapsed = double(1), checksum = double(1), PACKAGE = builds[[build]])$elapsed / calls
}

# Seconds per call of an R arm, over a batch of calls.
per_call <- function(f, calls) {
  start <- as.numeric(Sys.time())
  for (i in seq_len(calls)) f()
  (as.numeric(Sys.time()) - start) / calls
}

# The calls a batch of an arm needs to last at least 50 ms, doubling from one.
calls_for_batch <- function(arm) {
  calls <- 1L
  while (arm(calls) * calls < 0.05) calls <- 2L * calls
  calls
}

set.seed(20211002L)
normal_losses <- function(n, m) sapply(1:m, function(g) rnorm(n, 1 + 0.5 * g / m, 1))
shapes <- list(
  "200 x 16, the paper's example" = paper_losses,
  "1000 x 16" = normal_losses(1000, 16),
  "100000 x 16" = normal_losses(100000, 16),
  "1000 x 64" = normal_losses(1000, 64),
  "1000 x 256" = normal_losses(1000, 256)
)

arm_names <- c("authors", "seri_mcs from R, float64", "seri_mcs from R, float32",
               "seri_mcs in C, float64", "seri_mcs in C, float32")
discarded <- 2L
kept <- 10L
results <- list()
for (shape in names(shapes)) {
  losses <- shapes[[shape]]
  # The authors' divisor n - 1 and seri_mcs's n give the same order here.
  stopifnot(identical(as.integer(authors_arm(losses)), from_r_arm("float64", losses)))
  run <- list(
    "authors" = function(calls) per_call(function() authors_arm(losses), calls),
    "seri_mcs from R, float64" = function(calls) per_call(function() from_r_arm("float64", losses), calls),
    "seri_mcs from R, float32" = function(calls) per_call(function() from_r_arm("float32", losses), calls),
    "seri_mcs in C, float64" = function(calls) in_c_per_call("float64", losses, calls),
    "seri_mcs in C, float32" = function(calls) in_c_per_call("float32", losses, calls)
  )
  calls <- sapply(arm_names, function(a) calls_for_batch(run[[a]]))
  times <- matrix(NA_real_, discarded + kept, length(arm_names), dimnames = list(NULL, arm_names))
  for (round in 1:(discarded + kept)) {
    order <- if (round %% 2L == 1L) arm_names else rev(arm_names)
    for (a in order) times[round, a] <- run[[a]](calls[[a]])
  }
  results[[shape]] <- apply(times[-(1:discarded), , drop = FALSE], 2, median)
}

format_time <- function(seconds) {
  if (seconds >= 1) sprintf("%.2f s", seconds)
  else if (seconds >= 1e-3) sprintf("%.3g ms", seconds * 1e3)
  else sprintf("%.3g us", seconds * 1e6)
}

dir.create(file.path(root, "out"), showWarnings = FALSE)
report_path <- file.path(root, "out", "seri_mcs_against_authors_report.txt")
report <- file(report_path, "w")
say <- function(...) writeLines(sprintf(...), report)
cpu <- grep("^model name", readLines("/proc/cpuinfo"), value = TRUE)[1]
cflags <- system2("make", c("-C", root, "-s", "-n", "-B", "libserimcs_f64.so"), stdout = TRUE)
say("seri_mcs() against the authors' R code (tests/performance/seri_mcs_against_authors.R)")
say("%s; BLAS %s; %s", R.version.string, sessionInfo()$BLAS, format(Sys.time(), "%Y-%m-%d %H:%M"))
say("%s, %d hardware threads; seri_mcs is serial", sub("^model name\\s*:\\s*", "", cpu), parallel::detectCores())
say("C build: %s", paste(cflags, collapse = " "))
say("median time per call over %d kept rounds (%d discarded), batches of at least 50 ms, arm order reversed every other round", kept, discarded)
say("")
for (shape in names(results)) {
  r <- results[[shape]]
  say("%s", shape)
  for (a in arm_names) {
    if (a == "authors") say("  %-26s %10s", a, format_time(r[[a]]))
    else say("  %-26s %10s   authors / this %.3g", a, format_time(r[[a]]), r[["authors"]] / r[[a]])
  }
}
close(report)
cat(readLines(report_path), sep = "\n")
