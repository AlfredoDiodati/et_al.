#!/bin/bash

# Run every correctness and integration test suite and write full output to
# test_report.txt. Terminal shows only a one-line PASS/FAIL per suite.
# Exits 0 if all pass, 1 if any fail.
#
# The examples are built here too, and their build failing fails the run. They
# are documentation that has to keep compiling, and no target named them
# together until `make examples` existed; a rule for one of them had already
# outlived the file it referred to.
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
    printf "  %-34s" "$label"
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
#
# The statistical test and model suites at the end of the list are built at
# float64 whatever MAT_DOUBLE says, through the Makefile's STAT_CFLAGS - see
# the note above it for why that is mandatory rather than advisable.
SUITES="test_mat test_mat_special test_decomp test_solver test_special test_stats test_random test_mcs test_mcs_variance test_broadcast test_gauss test_student test_mvgauss test_mvstudent test_matgauss test_matgauss_recovery test_ad test_tape_reset test_adam test_optimizer test_cluster test_mlp test_frame test_csv test_txt test_npy test_json test_sql test_join gzip_inflate rdata_array_read adf_correctness kpss_correctness dfgls_correctness otto_correctness hlt_union_correctness hlt_break_correctness hhlt_correctness zivot_andrews_correctness johansen_correctness engle_granger_correctness maki_correctness qlr_test_correctness lbfgs_correctness score_driven_location_correctness qvarma_correctness qvarma_analytic_agreement qvarma_gaussian_limit qvarma_identification"

# tests/integration/ answers a different question from tests/correctness/: not
# "is this module correct" but "does the hand-off between two of them hold".
# Each of these binaries includes headers from at least two directories. See
# README.md's "Testing and benchmarking" for the split.
INTEGRATION="frame_to_model join_missing_values distributed_simulation optimizer_swap pipeline_ownership header_composition header_composition_f32"

printf "building...\n"
printf "=== build ===\n" >> "$REPORT"
BUILD_TARGETS=""
for s in $SUITES; do BUILD_TARGETS="$BUILD_TARGETS tests/correctness/$s"; done
for s in $INTEGRATION; do BUILD_TARGETS="$BUILD_TARGETS tests/integration/$s"; done
if ! make $BUILD_TARGETS examples >> "$REPORT" 2>&1; then
    printf "build failed — see %s\n" "$REPORT"
    exit 1
fi
printf "\n" >> "$REPORT"

for s in $SUITES; do
    run "$s" "./tests/correctness/$s"
done

printf "\nintegration\n"
printf "=== integration ===\n" >> "$REPORT"
for s in $INTEGRATION; do
    run "$s" "./tests/integration/$s"
done

printf "\n=== summary ===\n%d passed, %d failed\n" "$PASS" "$FAIL" >> "$REPORT"

if [ "$FAIL" -eq 0 ]; then
    printf "\nall %d passed — report: %s\n" "$PASS" "$REPORT"
    exit 0
else
    printf "\n%d of %d failed — see %s\n" "$FAIL" "$((PASS + FAIL))" "$REPORT"
    exit 1
fi
