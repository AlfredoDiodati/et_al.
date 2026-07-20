#!/bin/bash

# Run every test suite and write full output to test_report.txt.
# Terminal shows only a one-line PASS/FAIL per suite.
# Exits 0 if all pass, 1 if any fail.

cd "$(dirname "$0")"

REPORT="test_report.txt"
PASS=0
FAIL=0

: > "$REPORT"
printf "test run: %s\n\n" "$(date)" >> "$REPORT"

run() {
    local label=$1
    shift
    printf "  %-24s" "$label"
    if output=$("$@" 2>&1); then
        printf "PASS\n"
        printf "=== %s: PASS ===\n%s\n\n" "$label" "$output" >> "$REPORT"
        PASS=$((PASS + 1))
    else
        printf "FAIL\n"
        printf "=== %s: FAIL ===\n%s\n\n" "$label" "$output" >> "$REPORT"
        FAIL=$((FAIL + 1))
    fi
}

printf "building...\n"
printf "=== build ===\n" >> "$REPORT"
if ! make tests/correctness/test_mat tests/correctness/test_mat_special tests/correctness/test_decomp tests/correctness/test_solver >> "$REPORT" 2>&1; then
    printf "build failed — see %s\n" "$REPORT"
    exit 1
fi
printf "\n" >> "$REPORT"

run "test_mat"          ./tests/correctness/test_mat
run "test_mat_special"  ./tests/correctness/test_mat_special
run "test_decomp"       ./tests/correctness/test_decomp
run "test_solver"       ./tests/correctness/test_solver

printf "\n=== summary ===\n%d passed, %d failed\n" "$PASS" "$FAIL" >> "$REPORT"

if [ "$FAIL" -eq 0 ]; then
    printf "\nall %d passed — report: %s\n" "$PASS" "$REPORT"
    exit 0
else
    printf "\n%d of %d failed — see %s\n" "$FAIL" "$((PASS + FAIL))" "$REPORT"
    exit 1
fi
