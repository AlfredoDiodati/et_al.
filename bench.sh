#!/bin/bash

# Run every performance benchmark and write full output to bench_report.txt.
# Terminal shows only a one-line done/FAILED per suite - these are
# benchmarks, not tests, so there's no pass/fail signal beyond "did it run
# to completion" (a nonzero exit means the benchmark process itself
# crashed or one of its own internal sanity checks fired, not a
# performance regression - read the report for actual numbers).
#
# Each bench_*.py self-builds its own lib*.so via `make` (see any
# individual script), so unlike check.sh there is no separate build step
# here. The interpreter needs numpy/scipy/pandas/jax available (see each
# bench_*.py's own imports for which it specifically needs) - set PYTHON
# to a venv interpreter that has them if plain python3 on PATH doesn't:
#   PYTHON=/path/to/venv/bin/python ./bench.sh

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
REPORT="bench_report.txt"
OK=0
FAILED=0
FAILED_NAMES=""

: > "$REPORT"
printf "bench run: %s (interpreter: %s)\n\n" "$(date)" "$PYTHON" >> "$REPORT"

SUITES="bench_mat bench_decomp bench_dist bench_ad bench_frame bench_random bench_stats bench_adam bench_special bench_json"

for s in $SUITES; do
    printf "  %-16s" "$s"
    if output=$("$PYTHON" "tests/performance/$s.py" 2>&1); then
        printf "done\n"
        printf "=== %s: done ===\n%s\n\n" "$s" "$output" >> "$REPORT"
        OK=$((OK + 1))
    else
        printf "FAILED\n"
        printf "=== %s: FAILED ===\n%s\n\n" "$s" "$output" >> "$REPORT"
        FAILED=$((FAILED + 1))
        FAILED_NAMES="$FAILED_NAMES $s"
    fi
done

printf "\n=== summary ===\n%d ran to completion, %d failed\n" "$OK" "$FAILED" >> "$REPORT"

if [ "$FAILED" -eq 0 ]; then
    printf "\nall %d benchmarks ran to completion — report: %s\n" "$OK" "$REPORT"
    exit 0
else
    printf "\n%d of %d failed:%s — see %s\n" "$FAILED" "$((OK + FAILED))" "$FAILED_NAMES" "$REPORT"
    exit 1
fi
