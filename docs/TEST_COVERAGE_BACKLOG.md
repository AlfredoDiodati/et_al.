# Test coverage backlog

Public functions identified as undertested against `tests/correctness/*.c`,
audited 2026-08-14 against `./check.sh`'s 25 passing suites (all suites
passed at audit time - these are coverage gaps, not known bugs). `check.sh`
now runs 47, the difference being `test_mcs`, `test_join` and the fifteen
statistical test and model suites; nothing below has been re-audited against
those, so this file understates current coverage rather than overstating it.
Update this file as items are picked up: check the box, note the test added
and where.

This file is about coverage of individual functions. The separate question of
whether the hand-off between two modules holds is `tests/integration/`'s, and
its own backlog is the "What this directory does not cover" section of
`docs/INTEGRATION_TESTS_DOCUMENTATION.md`. Three items below are now partly
covered from there rather than here, and say so in place: `ad_leaf` with a
strided input, `mlp_fit` with a second optimizer, and `mlp_forecast` on a model
no Adam state touched. Partly, not fully - an integration test drives a
function through a real caller, which is not the same as the direct
known-output call this file asks for, so none of the three boxes is checked.

Coverage bar, from `README.md`'s "Testing requirements": every function
needs at minimum a known-output case or invariant check, a view/strided-input
case, and one adversarial input. "Test the public API, not `static inline`
internals" - internal (`_`-prefixed) helpers are out of scope here except
where called out explicitly below.

## Zero coverage

- [ ] `vec_triangular_solve` - `linalg/solver.h:106` - partially covered
      since `ad_triangular_solve` was added: `tests/correctness/test_ad.c`
      checks the forward residual `L*x - b` and `L^T*x - b` at `uplo` 'L',
      `diag` 'N', and `sd/score_driven_location.h` exercises it every period
      of every filter. Still missing: `uplo` 'U', `diag` 'U', a strided/view
      input, and a near-zero-diagonal adversarial case.
- [ ] `mat_isnan_f32` / `mat_isnan_f64` - `linalg/mat.h:67,71` - only reached
      via the `MISNAN` macro inside `mat_max`/`mat_min`, never called
      directly. Needs direct calls on NaN, -NaN, and finite boundary values.
- [ ] `mat_isinf_f32` / `mat_isinf_f64` - `linalg/mat.h:75,79` - never called
      by name anywhere. Needs direct calls on +inf, -inf, FLT_MAX/DBL_MAX
      (must be false), NaN (must be false).
- [ ] `mat_ipow` - `linalg/mat.h:307` - only exponent=2 is exercised (via
      `mat_pow`). Needs direct calls at exp = -1, 0, 1, 3, and >3 (e.g. 10),
      checked against `pow()`.
- [ ] `mat_print` - `linalg/mat.h:638` - called but asserts nothing (smoke
      test only). Needs stdout capture and a format check.
- [ ] `frame_npy_check_fortran_order` - `frame/npy.h:40` - no test ever
      writes `fortran_order: True`. Needs a header with that flag set,
      expecting rejection.
- [ ] `json_as_string` - `json.h:116` - never called; tests read `v->string`
      directly instead. Needs a known-output call on a parsed string value
      plus a fork/SIGABRT case on a non-string value.
- [ ] `mlp_to_json` / `mlp_from_json` - `nn/mlp.h:119,148` - only reached
      transitively via `mlp_save`/`mlp_load`; shape-mismatch asserts never
      triggered. Needs a hand-built tree checked for exact recovery, plus a
      fork/SIGABRT case for a shape mismatch.
- [ ] `mlp_rand_uniform` - `nn/mlp.h:61` - no test checks its output
      range/distribution, only used internally by `mlp_init`. Needs a
      fixed-seed draw of many samples, asserting every value in [-a,a) and
      mean near zero.
- [ ] `frame_trim` - `frame/frame.h:288` - dead code, no callers anywhere in
      the repo. Either add a caller and a test, or remove it.
- [ ] `_orgtr` (internal, `linalg/factor.h:1565`) - accumulates Q from
      symmetric tridiagonal reduction. Only call site anywhere is a
      performance bench (`tests/performance/eigsym_lapack_removal.c`), which
      doesn't count as correctness coverage. Flagged despite being internal
      since it's load-bearing and absent from every `*_blas_only.c` test.
- [ ] `_orgbr_p` (internal, `linalg/factor.h:2677`) - forms the right-
      orthogonal (P) factor from bidiagonalization. Zero call sites anywhere
      in the repo, including benches. Check whether it backs a live SVD path
      before writing a test - may be dead code.

## Thin / indirect coverage

Exercised only as a side effect of testing something else; no assertion
targets the function itself.

- [ ] `gzip_inflate_raw` - `frame/gzip.h:309` - the `expected_size == 0`
      (grow-on-demand) branch is never exercised, only `expected_size > 0`.
      Needs a direct call with `expected_size = 0` against a known small
      DEFLATE stream.
- [ ] `gzip_build_dynamic_trees` - `frame/gzip.h:247` - happy path covered via real
      `.RData` files; its three malformed-input asserts (repeat code with no
      previous length, invalid code-length symbol, bad HLIT/HDIST/HCLEN)
      are only hit incidentally by fuzzing, never targeted. Needs a
      hand-built dynamic-block header per assert.
- [ ] `gzip_decode_symbol` - `frame/gzip.h:152` - "invalid huffman code" assert
      only reachable incidentally through fuzzing. Needs a targeted
      corrupted-code test.
- [ ] `json_parse_file` - `json.h:290` - only exercised on a valid path.
      Needs a fork/SIGABRT case on a nonexistent path.
- [ ] `json_write_file` - `json.h:397` - only happy-path. Lower priority
      (OS I/O failure, not malformed input) - needs an unwritable-path case.
- [ ] `frame_npy_parse_shape` - `npy.h:52` - 1D/2D shapes tested; the
      `ndim < 2` (>2D array) assert, the 0-d `()` assert, and a genuinely
      malformed shape token are never exercised.
- [ ] `df_read_npy` - `npy.h:81` - dtype mismatch/truncated/bad-magic/
      undersized all covered; a header missing `descr`/`shape`/
      `fortran_order`, `fortran_order=True`, or a >2D shape is never fed in.
- [ ] `df_from_rvalue` RVALUE_LOGICAL branch - `frame/rdata.h:627` - no
      fixture has a real R logical (TRUE/FALSE) column; only reached via the
      identical-layout RVALUE_INTEGER path. Needs a real logical-column
      fixture.
- [ ] `rvalue_to_mat` RVALUE_LOGICAL branch - `frame/rdata.h:563` - same gap,
      no standalone logical vector/matrix is ever converted.
- [ ] `frame_read_file` - `frame/frame.h:232` - success path only,
      transitively, through every loader; its open-failure assert path is
      never targeted directly.
- [ ] `df_print` - `frame/frame.h:204` - smoke test only, no assertion on
      output content.
- [ ] `frame_strdup`, `strlist_init`/`strlist_push`/`strlist_free`
      (`frame/frame.h:52,274,275,282`), `frame_try_parse_numeric` (`:303`),
      `frame_build_from_rows` (`:322`), `frame_rows_to_dataframe` (`:356`) -
      never called by name in any test; solid indirect coverage through
      `df_read_csv`/`df_read_txt` round-trips only. Note: the header's own
      comment (frame.h:218-224) marks these "not part of the public API"
      despite lacking a leading underscore - see naming-convention note
      below before writing direct tests for these.
- [ ] `frame_col_lookup` - `frame/frame.h:164` - comment says "Private:"
      despite no underscore; only reached indirectly. Same naming-convention
      caveat as above.
- [ ] `df_write_txt`'s all-numeric contract assert - `frame/txt.h:94-96` -
      `test_txt.c` has no fork/SIGABRT machinery at all (unlike csv/npy/
      frame/rdata tests). Needs a DataFrame with a string column fed in to
      confirm the abort.
- [ ] `rng_u64` - `random.h:47` - PCG64 output step. Only tested for
      reproducibility and stream/seed divergence, never against a
      known-output reference vector - a wrong rotate amount or byte order
      that still mixes well would still pass. Needs a hardcoded expected
      `uint64_t` for a fixed (seed, stream) from an independent reference.
- [ ] `rng_new` - `random.h:70` - state init via SplitMix64. Same gap, no
      known-output vector.
- [ ] `rng_splitmix64` - `random.h:58` - seed-expansion step. Zero direct
      call sites, exercised only as a black box through `rng_new`. Needs a
      direct known-output check.
- [ ] `student_bcast_shape` - `dist/student.h:34` - no direct shape test
      with `nu` contributing a non-1 dimension (unlike `gauss_bcast_shape`,
      which has one).
- [ ] `mvgauss_diff_t` - `dist/mv/gauss.h:43` - never isolated, only reached
      inside logpdf/gradient functions that do cover it.
- [ ] `mvgauss_check` - `dist/mv/gauss.h:33` - the `loc.r` branch (must be 1
      or n) is never driven to fail, only the cov-dimension-mismatch branch
      is tested.
- [ ] `student_lognorm` - `dist/student.h:48` - no direct call, covered
      indirectly across nu in {1, 3, fractional, 1e6}.
- [ ] `student_dlognorm_dnu` - `dist/student.h:176` - indirect only, via
      `student_dlogpdf_nu`.
- [ ] `student_draw` - `dist/student.h:245` - no isolated value check, only
      through `student_sample`'s moment tests.
- [ ] `mvstudent_lognorm` - `dist/mv/student.h:41` - same pattern as
      `student_lognorm`.
- [ ] `mvstudent_dlognorm_dnu` - `dist/mv/student.h:134` - indirect only, via
      `mvstudent_dlogpdf_nu`.
- [ ] `ad_leaf` - `ad.h:269` - every fixture across ad.h/dist/mlp tests was
      contiguous (`mat_lit`/`mat_new`/`mat_fill`); none passed a strided view
      (`mat_slice`) into it. Now covered from `tests/integration/frame_to_model.c`,
      which drives it through `sd/qvarma.h`'s filter on a `y` that is a column
      range of a wider matrix, and through `nn/mlp.h` on a windowed design
      matrix, requiring the same answer as a contiguous copy in both cases. A
      direct call with a stride check on the copy is still worth adding here.
- [ ] `ad_log` - `ad.h:439` - no standalone known-output/finite-difference
      test, only exercised compositionally inside gauss/student/mv gradient
      checks, where local cancellation could mask a bug. Needs a direct
      known-output + FD test plus a near-zero-argument adversarial case,
      matching the treatment `ad_exp`/`ad_tanh` already get.
- [ ] `adam_optimizer_step` / `adam_optimizer_free` - `solver/adam.h:127,130`
      - never called by name, only via `opt.step(...)`/`opt.free(...)`. Low
      risk (test_optimizer.c cross-checks the result bit-for-bit against
      direct `adam_step` over 200 iterations) but no test targets these two
      adapter functions by name.
- [ ] `mlp_fit` - `nn/mlp.h:297` - only exercised via XOR (n=4) and an
      epoch-callback test; no isolated call with n=1 or a constant/
      degenerate target. The optimizer half of this gap is closed:
      `tests/integration/optimizer_swap.c` fits through a non-Adam
      implementation of the `Optimizer` interface and counts the instances
      created, their shapes, the steps taken and the frees.
- [ ] `mlp_forecast` - `nn/mlp.h:366` - covered via the same XOR/save-load
      tests; no isolated single-sample/single-feature case (`mlp_forward`,
      which it wraps, does have one). It is now also driven from
      `tests/integration/pipeline_ownership.c` after its training data has
      been freed, and from `optimizer_swap.c` on a model no Adam state
      touched.

## Systemic gap (not per-function)

- [ ] `linalg/decomp.h` (all 10 functions: `mat_chol`, `mat_lu`, `mat_qr`,
      `mat_eig_sym`, `mat_svd`, `mat_det`, `mat_inv`, `mat_cond`, `mat_rank`,
      `mat_eig`) and `linalg/solver.h`'s 6 solve functions (`vec_solve`,
      `vec_solve_sym`, `vec_lu_solve`, `vec_chol_solve`, `mat_lstsq`,
      `mat_lstsq_rd`) all meet known-output + view + single-element
      adversarial coverage, but none is tested against a near-singular or
      badly-scaled input (condition number ~1e6-1e8) - the adversarial
      category README calls out as most relevant to factorizations and
      solves. Needs a Hilbert-like or rank-1-perturbed matrix at that
      condition number, checked against a size-scaled tolerance.

## Naming-convention inconsistency (not a coverage gap)

Several headers have non-underscore-prefixed functions that their own doc
comments mark private: `frame.h`'s `frame_strdup`/`strlist_*`/
`frame_col_lookup` cluster ("not part of the public API" / "Private:"),
`sql.h`'s ~39 tokenizer/parser/evaluator helpers ("private implementation
detail"), and `ad.h`'s 6 tape/node primitives (`tape_alloc`, `ad_node_new`,
`ad_node_new_pooled`, `ad_accum`, `ad_accum_neg`, `ad_tape_push` - confirmed
zero call sites outside their own header). These were excluded from the
list above per the documented intent rather than the naming convention, but
either add the leading underscore or drop the "private" comment so the two
signals agree.

## Well-covered (verified during audit, no action needed)

`frame/csv.h`, `frame/sql.h`'s 2 public functions, `stats.h` (all 20,
including `stats_hac_var`/`stats_hac_var_centered`), `mcs.h` (all 18 public
functions, including the report/export writers and `dm_test`), `random.h`'s `rng_below`, `special.h`'s
`special_digamma` and `special_norm_cdf`, `dist/gauss.h`/`dist/student.h`/
`dist/broadcast.h`/`dist/mv/gauss.h`/`dist/mv/student.h`'s main pdf/logpdf/
gradient/sample functions, `solver/adam.h`'s `adam_step` and init/free
functions, `solver/optimizer.h` (no public functions, interface exercised
via a hand-rolled SGD in `test_optimizer.c`), `nn/mlp.h`'s `mlp_init`/
`mlp_free`/`mlp_forward`/`mlp_save`/`mlp_load`/`mlp_fit_free`,
`linalg/mat.h`'s other 26 public functions, `frame/rdata.h` (strongest-
tested module overall - real R 4.3.3 fixtures, every R type, NA in every
scalar type, 60-trial fuzz - only the logical-column gap above), and
`frame/npy.h`/`frame/txt.h`'s main entry points.
