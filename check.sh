#!/bin/bash

# Run every correctness test suite and write full output to test_report.txt.
# Terminal shows only a one-line PASS/FAIL per suite.
# Exits 0 if all pass, 1 if any fail.
#
# Set STRESS=1 in the environment to run every suite's stress sweep too
# (the same fixed-seed randomized/independent-reference checks `make
# test-stress` runs) - off by default since it's noticeably slower.

cd "$(dirname "$0")"

REPORT="test_report.txt"
PASS=0
FAIL=0

: > "$REPORT"
printf "test run: %s%s\n\n" "$(date)" "${STRESS:+ (STRESS=1)}" >> "$REPORT"

run() {
    local label=$1
    shift
    printf "  %-24s" "$label"
    # Only actually set STRESS in the child's environment when the caller
    # asked for it - the test binaries check getenv("STRESS") for a non-NULL
    # pointer, so STRESS="" (unset-but-defined) would wrongly enable it.
    local ok
    if [ -n "$STRESS" ]; then
        output=$(STRESS=1 "$@" 2>&1); ok=$?
    else
        output=$("$@" 2>&1); ok=$?
    fi
    if [ "$ok" -eq 0 ]; then
        printf "PASS\n"
        printf "=== %s: PASS ===\n%s\n\n" "$label" "$output" >> "$REPORT"
        PASS=$((PASS + 1))
    else
        printf "FAIL\n"
        printf "=== %s: FAIL ===\n%s\n\n" "$label" "$output" >> "$REPORT"
        FAIL=$((FAIL + 1))
    fi
}

# Every correctness suite `make test` runs, plus test_mat_special (built
# separately - see the Makefile's test-special rule - since it deliberately
# skips -ffast-math to get IEEE-defined NaN/Inf semantics).
SUITES="test_mat test_mat_special test_decomp test_solver test_special test_stats test_random test_broadcast test_gauss test_student test_mvgauss test_mvstudent test_ad test_adam test_optimizer test_mlp test_frame test_csv test_txt test_npy test_json test_sql"

printf "building...\n"
printf "=== build ===\n" >> "$REPORT"
BUILD_TARGETS=""
for s in $SUITES; do BUILD_TARGETS="$BUILD_TARGETS tests/correctness/$s"; done
if ! make $BUILD_TARGETS >> "$REPORT" 2>&1; then
    printf "build failed — see %s\n" "$REPORT"
    exit 1
fi
printf "\n" >> "$REPORT"

for s in $SUITES; do
    run "$s" "./tests/correctness/$s"
done

printf "\n=== summary ===\n%d passed, %d failed\n" "$PASS" "$FAIL" >> "$REPORT"

if [ "$FAIL" -eq 0 ]; then
    printf "\nall %d passed — report: %s\n" "$PASS" "$REPORT"
    exit 0
else
    printf "\n%d of %d failed — see %s\n" "$FAIL" "$((PASS + FAIL))" "$REPORT"
    exit 1
fi
