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
SUITES="poly_correctness spline_design_correctness spline_basis_correctness spline_objects_correctness test_mat test_tensor test_tensor_serial cumsum_correctness rolling_mean_correctness test_mat_special test_decomp test_solver lstsq_rank_deficiency chol_singularity singularity_rule_comparison ols_pseudo_inverse_fallback lag_matrix_layout rank_rule_consistency test_special test_stats test_random test_lhs test_mcs test_mcs_variance mcs_primitives mcs_size_and_power test_broadcast test_gauss test_student test_mvgauss test_mvstudent test_matgauss test_matgauss_recovery mv_density_dispatch test_ad ad_tensor_gradients test_tape_reset test_adam test_optimizer test_cluster test_mlp test_frame test_csv test_txt test_npy test_npz test_json test_sql test_join gzip_inflate gzip_deflate rdata_array_read adf_correctness kpss_correctness dfgls_correctness otto_correctness hlt_union_correctness hlt_break_correctness hhlt_correctness zivot_andrews_correctness johansen_correctness engle_granger_correctness maki_correctness qlr_test_correctness lbfgs_correctness score_driven_location_correctness qvarma_correctness qvarma_analytic_agreement qvarma_gaussian_limit qvarma_identification qvarma_fixed_parameter_fit var_correctness var_recovery"

# tests/integration/ answers a different question from tests/correctness/: not
# "is this module correct" but "does the hand-off between two of them hold".
# Each of these binaries includes headers from at least two directories. See
# README.md's "Testing and benchmarking" for the split.
INTEGRATION="basis_to_regression frame_to_model frame_to_tensor tensor_to_optimizer join_missing_values distributed_simulation optimizer_swap pipeline_ownership npz_to_statistics header_composition header_composition_f32"

# Build output belongs on disk, never in the index: a tracked binary is
# rewritten by every build, shows up as a change in every commit, and runs
# stale if a checkout gives it a newer timestamp than the sources make
# compares it against. Two ways one gets in: git tracks a file whatever
# .gitignore later says about it, and a binary nobody added a pattern for is
# not ignored at all. Both are checked, the second by the ELF signature in
# the first four bytes.
tracked_build_output() {
    local found
    found=$( {
        git ls-files -ci --exclude-standard
        git ls-files -z | while IFS= read -r -d '' f; do
            [ -f "$f" ] && [ "$(head -c 4 "$f" | od -An -tx1 | tr -d ' ')" = "7f454c46" ] && printf "%s\n" "$f"
        done
    } | sort -u)
    if [ -n "$found" ]; then
        printf "tracked build output, remove with git rm --cached and add to .gitignore:\n%s\n" "$found"
        return 1
    fi
    printf "no build output is tracked\n"
}

run "tracked_build_output" tracked_build_output

# `make test` and this script are two runners over the same suites, and a
# suite one of them skips passes unnoticed: the test recipe once handed
# mcs_size_and_power to mcs_primitives as an argument instead of running it.
# The recipe now runs every prerequisite, and this checks that its list and
# the lists here name the same suites. test_mat_special is the one expected
# difference: it is built without -ffast-math and only this script runs it.
same_suites_as_make_test() {
    local from_make from_here
    from_make=$(make -n test 2>/dev/null | grep '^for t in' | head -1 \
        | sed 's/^for t in //; s/; do.*//' | tr ' ' '\n' | sed 's|^tests/[a-z]*/||' | grep -v '^$' | sort)
    from_here=$(printf '%s\n' $SUITES $INTEGRATION | grep -vx test_mat_special | sort)
    if [ -z "$from_make" ]; then
        printf "could not read the suites from make -n test\n"
        return 1
    fi
    if [ "$from_make" != "$from_here" ]; then
        printf "make test and check.sh run different suites:\n"
        diff <(printf '%s\n' "$from_make") <(printf '%s\n' "$from_here") | grep '^[<>]' \
            | sed 's/^</  only in make test:/; s/^>/  only in check.sh:/'
        return 1
    fi
    printf "make test and check.sh run the same %d suites\n" "$(printf '%s\n' "$from_make" | wc -l)"
}

run "same_suites_as_make_test" same_suites_as_make_test

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

# Every integration suite runs from a clean output directory, the state of a
# fresh clone: frame_to_tensor once wrote into tests/integration/out without
# creating it, and passed only on machines where an earlier run had.
rm -rf tests/integration/out

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
