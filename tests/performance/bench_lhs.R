# lhs.h vs R's lhs package: how long one n x k design takes.
#
# The two produce the same distribution of designs (checked in
# tests/correctness/lhs_r_agreement.R) by different algorithms, and the
# algorithms are what this times. Per column, R's randomLHS draws n
# uniforms and sorts them to get a permutation, an O(n log n) comparison
# sort; lhs_random draws n-1 bounded integers and shuffles in place, O(n).
# Both then draw n uniforms of within-stratum jitter.
#
# Two of our timings are reported. "ours" is the whole .C() call, which
# includes R allocating an n*k double vector and copying the result back
# across the interface - the number a caller in R would actually see.
# "kernel" runs the identical sampler and skips only that copy, so the
# difference between them is the boundary cost rather than the sampler's.
#
# R is a development-tier dependency, so this is not part of bench.sh
# (which drives the Python comparison suites). Run:
#   make bench-lhs

suppressPackageStartupMessages(library(lhs))

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- if (length(script_path) == 1L) {
  normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  normalizePath(".")
}

stopifnot(system2("make", c("-C", root, "liblhs.so"), stdout = FALSE) == 0L)
dyn.load(file.path(root, "liblhs.so"))

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

seed <- 20240501L

cases <- list(
  c(n = 100, k = 5),
  c(n = 1000, k = 9),
  c(n = 10000, k = 9),
  c(n = 100000, k = 10)
)

lines <- character()
say <- function(...) lines <<- c(lines, paste0(...))

say(sprintf("lhs.h vs R lhs %s: one n x k random Latin hypercube design",
            as.character(packageVersion("lhs"))))
say("")
say(sprintf("  run       %s", format(Sys.time())))
say(sprintf("  R         %s", R.version.string))
say(sprintf("  timing    best of %d rounds, each at least %.1f s of repeats",
            rounds, budget_seconds))
say("")
say("  ours    the whole .C() call, result copied back into an R vector")
say("  kernel  the same sampler without that copy")
say("")
say(sprintf("%12s %12s %12s %12s %10s %10s", "design", "R ms", "ours ms",
            "kernel ms", "speedup", "kernel x"))
say(strrep("-", 74))

for (case in cases) {
  n <- as.integer(case[["n"]]); k <- as.integer(case[["k"]])
  buffer <- double(as.double(n) * k)

  theirs <- best_milliseconds(function() randomLHS(n, k))
  ours <- best_milliseconds(function()
    .C("c_lhs_random", seed, n, k, out = buffer)[[4]])
  kernel <- best_milliseconds(function()
    .C("c_lhs_random_only", seed, n, k, out = double(1))[[4]])

  say(sprintf("%12s %12.4f %12.4f %12.4f %9.2fx %9.2fx",
              sprintf("%d x %d", n, k), theirs, ours, kernel,
              theirs / ours, theirs / kernel))
}

say("")

# The scaling step the design table needs after the unit design: two
# sweep() calls in R against one pass in C.
# Nine parameter ranges, deliberately awkward rather than round: one
# negative and narrow, several wide, none symmetric about zero.
lower <- c(0.05, -1.50, 0.10, 0.10, 0.10, 1.00, 0.00, 0.50, 0.50)
upper <- c(0.25, -1.25, 0.50, 0.50, 0.50, 1.50, 0.50, 0.95, 0.95)

say(sprintf("%12s %12s %12s %10s", "scale", "R ms", "ours ms", "speedup"))
say(strrep("-", 50))
for (n in c(1000L, 100000L)) {
  k <- length(lower)
  unit <- matrix(runif(n * k), n, k)
  flat <- as.double(t(unit))
  buffer <- double(as.double(n) * k)

  theirs <- best_milliseconds(function()
    sweep(sweep(unit, 2, upper - lower, "*"), 2, lower, "+"))
  ours <- best_milliseconds(function()
    .C("c_lhs_scale", n, k, flat, lower, upper, out = buffer)[[6]])

  say(sprintf("%12s %12.4f %12.4f %9.2fx", sprintf("%d x %d", n, k),
              theirs, ours, theirs / ours))
}

dir.create(file.path(root, "out"), showWarnings = FALSE, recursive = TRUE)
report <- file.path(root, "out", "bench_lhs_report.txt")
writeLines(lines, report)

cat(paste(lines, collapse = "\n"), "\n", sep = "")
cat("report written to ", report, "\n", sep = "")
