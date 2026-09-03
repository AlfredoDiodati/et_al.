# Simulation of the DSK stock-flow-consistent model over a Latin hypercube
# design of parameter configurations (CoPs).
#
# The executable accepts one JSON parameter file and one seed per execution, so
# the design table is never handed to the model directly. The workflow is: one
# design row -> one modified JSON -> n_mc seeded executions -> extract the five
# series this project uses -> save one 5 x 401 x n_mc array. One array per
# design row, so the whole design is 1000 arrays: the 1000 x 1000 experiment,
# 1000 configurations of 1000 replications each.
#
# This is the last step of the pipeline that is not C. Everything downstream
# reads the arrays through applications/abm_system.h, so each file is written
# in the exact shape that header asserts on: an R workspace holding one object,
# named estimation, a double array whose first dimension carries the variable
# names it looks up and whose second carries the model period each column is
# on. That is also why the arrays are saved with save() rather
# than saveRDS(): et_al's reader parses the workspace wrapper, and a
# single-object .rds file is a different container it does not accept.
#
# Usage:
#   Rscript applications/abm_system_simulate.R design DESIGN.csv [N_COP] [SEED]
#   Rscript applications/abm_system_simulate.R run WHICH DESIGN.csv BASE.json EXECUTABLE RESULT_DIR [N_MC]
#
# WHICH selects the design rows to simulate: "all" for the whole design, "7"
# for one row, "1-250" for a range. One row per job is what a cluster array
# wants; "all" is what a single machine wants.
#   Rscript applications/abm_system_simulate.R inspect ARRAY.Rdata [OUT_PREFIX]
#
# Add --force to overwrite a design or an array that is already on disk;
# without it both modes keep what they find, so a re-submitted job array only
# simulates the CoPs that are still missing.

bounds <- data.frame(
  parameter = c(
    "Gamma", "chi", "psi1", "psi3", "alfa",
    "taylor1", "taylor2", "taylor", "kappa"
  ),
  lower = c(0.05, -1.50, 0.10, 0.10, 0.10, 1.00, 0.00, 0.50, 0.50),
  upper = c(0.25, -1.25, 0.50, 0.50, 0.50, 1.50, 0.50, 0.95, 0.95),
  stringsAsFactors = FALSE
)

parameter_names <- bounds$parameter

n_steps <- 600L
n_model_columns <- 83L

# Only the periods the auxiliary model is ever fitted on are kept. That window
# starts at ABM_SYSTEM_BURN_IN + 1 = 201 in applications/abm_system.h, and
# period 200 comes with it because period 201's first difference is anchored on
# it. A million runs of raw output are not stored, so this window cannot be
# widened later without simulating them again.
first_stored_step <- 200L
stored_steps <- first_stored_step:n_steps

# Positions in the model's own results file, which has no header, under the
# names applications/abm_system.h looks its variables up by. "Gross inflation"
# is the price level cpi(1), which is what that header already reads it as;
# column 6 is cpi(1)/cpi(5), a four-period gross inflation factor, and is a
# different series. "Interest rate" is the central bank's policy rate, not the
# average commercial loan rate, which is column 82. The model's consumption,
# investment and emissions series are not extracted: the auxiliary model has no
# slot for them and the real data has no counterpart series to compare them
# against.
output_columns <- c(
  GDP = 2L,
  `Employment rate` = 5L,
  `Gross inflation` = 34L,
  `Interest rate` = 50L,
  Energy = 8L
)

# Units are the model's own and nothing is rescaled here: the employment rate
# stays a 0-1 proportion and the interest rate a decimal, matching the existing
# simulated files, since abm_system.h does every transformation itself.

build_design <- function(design_path, n_cop, seed, force) {
  if (file.exists(design_path) && !force) {
    message("Design already on disk, kept: ", design_path)
    return(invisible(design_path))
  }

  library(lhs)
  set.seed(seed)

  u <- randomLHS(n_cop, nrow(bounds))

  Sample <- sweep(u, 2, bounds$upper - bounds$lower, "*")
  Sample <- sweep(Sample, 2, bounds$lower, "+")
  colnames(Sample) <- parameter_names

  stopifnot(anyDuplicated(as.data.frame(Sample)) == 0L)

  dir.create(dirname(design_path), recursive = TRUE, showWarnings = FALSE)
  write.csv(Sample, design_path, row.names = FALSE, quote = FALSE)

  message("Wrote ", design_path, ": ", n_cop, " CoPs, seed ", seed)
  invisible(design_path)
}

# "all", one row number, or "first-last".
parse_cop_ids <- function(which, n_cop) {
  if (which == "all") return(seq_len(n_cop))

  if (grepl("^[0-9]+-[0-9]+$", which)) {
    ends <- as.integer(strsplit(which, "-", fixed = TRUE)[[1]])
    stopifnot(ends[1] >= 1L, ends[2] <= n_cop, ends[1] <= ends[2])
    return(ends[1]:ends[2])
  }

  id <- suppressWarnings(as.integer(which))
  stopifnot(!is.na(id), id >= 1L, id <= n_cop)
  id
}

simulate_cop <- function(cop_id, design_path, base_json, source_exe,
                         result_dir, n_mc, force) {
  library(jsonlite)

  array_file <- file.path(result_dir, sprintf("cop_%04d.Rdata", cop_id))

  if (file.exists(array_file) && !force) {
    message("Array already on disk, kept: ", array_file)
    return(invisible(array_file))
  }

  design <- read.csv(design_path, check.names = FALSE)

  stopifnot(
    cop_id >= 1L,
    cop_id <= nrow(design),
    all(parameter_names %in% names(design))
  )

  cop <- design[cop_id, parameter_names, drop = FALSE]

  dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(result_dir, "errors"), recursive = TRUE, showWarnings = FALSE)

  # The model writes its output beside its own executable, so each task gets a
  # private copy on node-local scratch: otherwise parallel jobs collide in one
  # output directory and hammer a single shared filesystem.
  scratch_root <- Sys.getenv("SLURM_TMPDIR", unset = tempdir())
  job_dir <- tempfile(pattern = sprintf("dsk_cop%04d_", cop_id), tmpdir = scratch_root)

  dir.create(job_dir, recursive = TRUE)
  on.exit(unlink(job_dir, recursive = TRUE, force = TRUE), add = TRUE)
  dir.create(file.path(job_dir, "output", "errors"), recursive = TRUE, showWarnings = FALSE)

  local_exe <- file.path(job_dir, "dsk_SFC")
  stopifnot(file.copy(source_exe, local_exe))
  Sys.chmod(local_exe, mode = "0755")
  stopifnot(file.access(local_exe, mode = 1) == 0L)

  input <- fromJSON(base_json, simplifyVector = FALSE)

  for (p in parameter_names) {
    input$params[[1]][[p]] <- as.numeric(cop[[p]])
  }

  input$params[[1]][["T"]] <- n_steps

  json_path <- file.path(job_dir, sprintf("inputs_cop%04d.json", cop_id))
  write_json(input, json_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)

  # The period labels are what tells the C side which model periods this file
  # holds, so a file with its transient already removed and one with every
  # period from the first produce the same series downstream. They are built
  # with sprintf because as.character() of an integer sequence returns a
  # deferred-string ALTREP vector, which et_al's reader rejects by design.
  estimation <- array(
    NA_real_,
    dim = c(length(output_columns), length(stored_steps), n_mc),
    dimnames = list(names(output_columns), sprintf("%d", stored_steps), NULL)
  )

  attr(estimation, "CoP_ID") <- cop_id
  attr(estimation, "parameters") <- unlist(cop, use.names = TRUE)
  attr(estimation, "output_columns") <- output_columns
  attr(estimation, "seed_scheme") <- "MC seed 1:n_mc, repeated across CoPs"

  failures <- vector("list", 0L)
  run_name <- sprintf("cop%04d", cop_id)

  for (mc in seq_len(n_mc)) {
    # Reusing the same seeds for every CoP is common random numbers: the
    # comparison across configurations is what this design exists for, and
    # correlating their noise reduces the variance of the difference.
    seed <- mc

    status <- suppressWarnings(
      system2(
        local_exe,
        args = c(
          shQuote(json_path),
          "-r", run_name,
          "-s", seed,
          "-f", 0,
          "-c", 0,
          "-v", 0
        ),
        stdout = FALSE,
        stderr = FALSE
      )
    )

    raw_file <- file.path(job_dir, "output", sprintf("results_%s_%d.txt", run_name, seed))
    error_file <- file.path(job_dir, "output", "errors", sprintf("Errors_%s_%d.txt", run_name, seed))

    error_size <- if (file.exists(error_file)) file.info(error_file)$size else NA_real_
    error_nonempty <- isTRUE(error_size > 0)

    values <- if (file.exists(raw_file)) {
      tryCatch(scan(raw_file, what = double(), quiet = TRUE), error = function(e) numeric())
    } else {
      numeric()
    }

    valid_length <- length(values) == n_steps * n_model_columns
    reasons <- character()

    if (status != 0L) {
      reasons <- c(reasons, sprintf("exit status %d", status))
    }
    if (error_nonempty) {
      reasons <- c(reasons, "non-empty model error log")
    }
    if (!valid_length) {
      reasons <- c(reasons, sprintf(
        "expected %d values, obtained %d",
        n_steps * n_model_columns, length(values)
      ))
    }

    if (length(reasons) == 0L) {
      model_output <- matrix(values, nrow = n_steps, ncol = n_model_columns, byrow = TRUE)
      selected <- t(model_output[stored_steps, unname(output_columns), drop = FALSE])

      if (all(is.finite(selected))) {
        estimation[, , mc] <- selected
      } else {
        reasons <- c(reasons, "non-finite model output")
      }
    }

    if (length(reasons) > 0L) {
      failures[[length(failures) + 1L]] <- data.frame(
        seed = seed,
        reason = paste(reasons, collapse = "; "),
        stringsAsFactors = FALSE
      )

      if (error_nonempty) {
        file.copy(
          error_file,
          file.path(result_dir, "errors", sprintf("cop%04d_seed%04d.txt", cop_id, seed)),
          overwrite = TRUE
        )
      }
    }

    # About 3 MB of raw 83-column output per run, none of it needed once the
    # eight series are in the array.
    unlink(c(raw_file, error_file), force = TRUE)

    if (mc %% 25L == 0L || mc == n_mc) {
      message(sprintf(
        "CoP %d: completed %d/%d runs; failures=%d",
        cop_id, mc, n_mc, length(failures)
      ))
    }
  }

  # gzip is the one compression et_al's reader accepts; xz, which save() will
  # use if asked, produces a file it rejects.
  save(estimation, file = array_file, compress = "gzip")

  if (length(failures) > 0L) {
    write.csv(
      do.call(rbind, failures),
      file.path(result_dir, sprintf("cop_%04d_failures.csv", cop_id)),
      row.names = FALSE
    )
  }

  message("Saved ", array_file)
  invisible(array_file)
}

inspect_cop <- function(array_path, out_prefix) {
  loaded <- new.env()
  load(array_path, envir = loaded)
  x <- get("estimation", envir = loaded)
  cop_id <- attr(x, "CoP_ID")
  variables <- dimnames(x)[[1]]

  dir.create(dirname(out_prefix), recursive = TRUE, showWarnings = FALSE)

  complete <- apply(x, 3L, function(run) all(is.finite(run)))

  report_path <- sprintf("%s_cop%04d_summary.txt", out_prefix, cop_id)
  plot_path <- sprintf("%s_cop%04d_paths.pdf", out_prefix, cop_id)

  lines <- c(
    sprintf("Array: %s", normalizePath(array_path)),
    sprintf("CoP: %d", cop_id),
    sprintf("Dimensions: %s", paste(dim(x), collapse = " x ")),
    sprintf("Complete MC runs: %d of %d", sum(complete), dim(x)[3L]),
    sprintf("Seed scheme: %s", attr(x, "seed_scheme")),
    "",
    "Parameters:",
    capture.output(print(attr(x, "parameters")))
  )

  if (any(complete)) {
    lines <- c(
      lines,
      "",
      "Per-variable summary over completed runs and all steps:",
      capture.output(print(t(apply(
        x[, , complete, drop = FALSE], 1L,
        function(v) c(
          min = min(v), median = median(v), mean = mean(v), max = max(v)
        )
      ))))
    )
  }

  writeLines(lines, report_path)

  if (!any(complete)) {
    message("No completed run in ", array_path, "; wrote ", report_path, " only")
    return(invisible(report_path))
  }

  pdf(plot_path, width = 9, height = 11)
  on.exit(dev.off(), add = TRUE)
  par(mfrow = c(4, 2), mar = c(4, 4, 2, 1))

  for (v in variables) {
    paths <- matrix(x[v, , complete, drop = FALSE], nrow = dim(x)[2L])

    matplot(
      paths,
      type = "l", lty = 1, col = adjustcolor("black", alpha.f = 0.25),
      xlab = "step", ylab = v, main = v
    )
  }

  message("Wrote ", report_path, " and ", plot_path)
  invisible(report_path)
}

args <- commandArgs(trailingOnly = TRUE)
force <- "--force" %in% args
args <- args[args != "--force"]

mode <- if (length(args) >= 1L) args[1] else ""

if (mode == "design") {
  if (length(args) < 2L) {
    stop("Usage: design DESIGN.csv [N_COP] [SEED]", call. = FALSE)
  }

  build_design(
    design_path = args[2],
    n_cop = if (length(args) >= 3L) as.integer(args[3]) else 1000L,
    seed = if (length(args) >= 4L) as.integer(args[4]) else 1L,
    force = force
  )
} else if (mode == "run") {
  if (length(args) < 6L) {
    stop(
      "Usage: run WHICH DESIGN.csv BASE.json EXECUTABLE RESULT_DIR [N_MC]",
      call. = FALSE
    )
  }

  design_path <- normalizePath(args[3], mustWork = TRUE)
  cop_ids <- parse_cop_ids(args[2], nrow(read.csv(design_path, check.names = FALSE)))

  for (cop_id in cop_ids) {
    simulate_cop(
      cop_id = cop_id,
      design_path = design_path,
      base_json = normalizePath(args[4], mustWork = TRUE),
      source_exe = normalizePath(args[5], mustWork = TRUE),
      result_dir = args[6],
      n_mc = if (length(args) >= 7L) as.integer(args[7]) else 1000L,
      force = force
    )
  }

  message("Done: ", length(cop_ids), " of ", nrow(read.csv(design_path)), " CoPs")
} else if (mode == "inspect") {
  if (length(args) < 2L) {
    stop("Usage: inspect ARRAY.Rdata [OUT_PREFIX]", call. = FALSE)
  }

  inspect_cop(
    array_path = normalizePath(args[2], mustWork = TRUE),
    out_prefix = if (length(args) >= 3L) args[3] else "out/abm_system_simulate"
  )
} else {
  stop(
    paste(
      "Usage:",
      "  Rscript applications/abm_system_simulate.R design DESIGN.csv [N_COP] [SEED]",
      "  Rscript applications/abm_system_simulate.R run WHICH DESIGN.csv BASE.json EXECUTABLE RESULT_DIR [N_MC]",
      "  Rscript applications/abm_system_simulate.R inspect ARRAY.Rdata [OUT_PREFIX]",
      sep = "\n"
    ),
    call. = FALSE
  )
}
