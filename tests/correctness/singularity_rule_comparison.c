/*
Does the tolerance rule catch the singular matrices the exact-zero rule misses,
and where does each of them break?

Two rules decide whether a factorization is singular. The exact-zero rule is
what factor.h's kernels return on their own: _gels reports a diagonal entry of
R that is exactly zero, _potrf a pivot that is not positive. The tolerance
rule is what mat_lstsq and mat_chol add on top: a column whose part outside
the span of the earlier ones is at most 10 * sqrt(m) * MEPS of its length, a
pivot with L[k][k]^2 at most 10 * sqrt(n) * MEPS * a[k][k]. Both are computed
here on the same input, in one binary.

Every singular matrix below is built from small integers, so every entry and
every combination of entries is stored exactly. The stored matrix is then
singular in exact arithmetic, and a nonzero pivot on it can only be rounding.
That is what makes the comparison a test of the rounding argument rather than
of how the input happened to be generated.

Sections, what each is trying to break, and what is required:

  exact       exactly singular designs in six patterns, 4 to 10000 rows, 50
              draws each. The tolerance rule has to report the first exactly
              dependent column, found by integer elimination modulo two primes,
              in every draw; the exact-zero rule has to miss most of them, the
              premise the tolerance exists for. It missed about 1420 of 1500 in
              both builds, and the solutions it let through had coefficients up
              to 1e21 on a problem whose valid solutions are of order 10.
  sweep       a design whose second column is 2^k times the first plus a
              small integer vector, so its independence shrinks like 2^-k while
              the true coefficients stay known exactly. The tolerance rule has
              to accept every draw whose exact sine is above three times the
              tolerance and reject every draw below a third of it, and every
              solution the exact-zero rule lets through has to obey the rounding
              model, error at most 2 sqrt(m) MEPS / sine. What the table shows
              beyond that is the price: just below the tolerance the rejected
              designs still had coefficients accurate to about 0.1 per cent,
              and at a sine of MEPS to within about 40 per cent.
  adversarial dependence that runs through a nearly collinear pair, which both
              rules miss and mat_lstsq_rd reports; the same columns in the other
              order, which the tolerance rule catches; entries at 1e+-200 in
              float64 and 1e+-30 in float32, where the tolerance rule has to
              accept the independent design and catch the dependent one; an
              infinite and a NaN entry, which it has to reject; subnormal
              entries; 200000 rows; a 40 x 40 square
  cholesky    the exact patterns on Gram matrices: the tolerance rule has to
              report pivot 4 every time and the exact-zero rule has to miss
              some; and the same scaled by powers of two up to 2^+-900, where a
              singular matrix has to be caught and a regular one accepted

The full tables go to out/singularity_rule_comparison_float32_report.txt or the
float64 name.
Run with make tests/correctness/singularity_rule_comparison.
*/

#include "../check.h"
#include "../../linalg/solver.h"
#include <sys/stat.h>

static FILE *report_file;

static double rule_tolerance(int length) {
    return 10.0 * sqrt((double)length) * (double)MEPS;
}

/* The exact-zero rule for least squares: _gels on a copy. Returns its info
   and, through x_out when given, the solution it computed when info is 0. */
static int exact_rule_lstsq(Mat a, Mat b, Mat *x_out) {
    Mat qr = mat_copy(a), work = mat_copy(b);
    int info = _gels(a.r, a.c, b.c, qr.d, qr.stride, work.d, work.stride);
    if (x_out) {
        *x_out = mat_new(a.c, b.c);
        for (int i = 0; i < a.c; i++)
            for (int j = 0; j < b.c; j++) AT(*x_out, i, j) = AT(work, i, j);
    }
    mat_free(qr); mat_free(work);
    return info;
}

static int tolerance_rule_lstsq(Mat a, Mat b) {
    int status;
    Mat x = mat_lstsq(a, b, &status);
    mat_free(x);
    return status;
}

static int exact_rule_chol(Mat a) {
    Mat l = mat_copy(a);
    int info = _potrf(l.d, a.r, l.stride);
    mat_free(l);
    return info;
}

static int tolerance_rule_chol(Mat a) {
    int status;
    Mat l = mat_chol(a, &status);
    mat_free(l);
    return status;
}

static double small_int(Rng *rng, int bound) {
    return (double)((int)rng_below(rng, (uint64_t)(2 * bound + 1)) - bound);
}

typedef enum {
    COMBINATION,      /* c3 = c0 + 2 c1 - c2 */
    DUPLICATE,        /* c3 = c1 */
    POWER_OF_TWO,     /* c3 = 4 c1 */
    THREE_TIMES,      /* c3 = 3 c1 */
    STUCK_SERIES,     /* c0 = 1, c3 = 0.25: a constant next to an intercept */
    WIDE_RANGE,       /* c3 = 2^s c0 + c1, s as large as exactness allows */
    N_PATTERNS
} Pattern;

static const char *pattern_names[] = {
    "c3 = c0 + 2 c1 - c2", "c3 = c1", "c3 = 4 c1", "c3 = 3 c1",
    "c0 = 1, c3 = 0.25", "c3 = 2^s c0 + c1"
};

/* Four columns, the fourth exactly dependent on the first three, all entries
   integers small enough that every one of them and every combination is
   stored exactly in mreal. */
static Mat exact_singular_design(Rng *rng, Pattern pattern, int m) {
    int wide_shift = sizeof(mreal) == sizeof(double) ? 20 : 8;
    Mat a = mat_new(m, 4);
    for (int i = 0; i < m; i++) {
        double c0 = small_int(rng, 8), c1 = small_int(rng, 8), c2 = small_int(rng, 8), c3 = 0;
        switch (pattern) {
        case COMBINATION: c3 = c0 + 2 * c1 - c2; break;
        case DUPLICATE: c3 = c1; break;
        case POWER_OF_TWO: c3 = 4 * c1; break;
        case THREE_TIMES: c3 = 3 * c1; break;
        case STUCK_SERIES: c0 = 1; c3 = 0.25; break;
        case WIDE_RANGE: c3 = ldexp(c0, wide_shift) + c1; break;
        default: break;
        }
        AT(a, i, 0) = (mreal)c0; AT(a, i, 1) = (mreal)c1;
        AT(a, i, 2) = (mreal)c2; AT(a, i, 3) = (mreal)c3;
    }
    return a;
}

static Mat integer_response(Rng *rng, int m) {
    Mat b = mat_new(m, 1);
    for (int i = 0; i < m; i++) AT(b, i, 0) = (mreal)small_int(rng, 8);
    return b;
}

static double max_abs(Mat x) {
    double worst = 0;
    for (int i = 0; i < x.r * x.c; i++)
        if (fabs((double)x.d[i]) > worst) worst = fabs((double)x.d[i]);
    return worst;
}

static int exact_missed_total = 0, exact_total = 0;
static double smallest_let_through = INFINITY;

static void test_exact(Rng *rng) {
    puts("exactly singular designs");
    fprintf(report_file, "Exactly singular designs, 4 columns, the 4th dependent; 50 draws per row\n");
    fprintf(report_file, "%-22s %6s %12s %16s %22s\n", "pattern", "rows", "exact rule", "tolerance rule",
            "largest |x| let through");
    int sizes[] = { 4, 16, 100, 1000, 10000 };
    int draws = 50;
    for (int p = 0; p < N_PATTERNS; p++)
        for (int s = 0; s < 5; s++) {
            int m = sizes[s], exact_flags = 0, tolerance_flags = 0, right_column = 0, earlier = 0;
            double largest_through = 0;
            for (int draw = 0; draw < draws; draw++) {
                Mat a = exact_singular_design(rng, (Pattern)p, m);
                Mat b = integer_response(rng, m);
                Mat x;
                int exact = exact_rule_lstsq(a, b, &x);
                int tolerance = tolerance_rule_lstsq(a, b);
                if (exact) exact_flags++;
                else {
                    if (max_abs(x) > largest_through) largest_through = max_abs(x);
                    if (max_abs(x) < smallest_let_through) smallest_let_through = max_abs(x);
                }
                int expected = check_first_dependent_exact(a);
                if (expected < 4) earlier++;
                if (tolerance) tolerance_flags++;
                if (tolerance == expected) right_column++;
                mat_free(a); mat_free(b); mat_free(x);
            }
            fprintf(report_file, "%-22s %6d %9d/%d %13d/%d %22.3g%s\n", pattern_names[p], m,
                    exact_flags, draws, tolerance_flags, draws, largest_through,
                    earlier ? "  (some draws dependent before column 4)" : "");
            CHECK(tolerance_flags == draws, "%s, %d rows: tolerance rule flagged %d of %d",
                  pattern_names[p], m, tolerance_flags, draws);
            CHECK(right_column == draws, "%s, %d rows: the first exactly dependent column reported in %d of %d",
                  pattern_names[p], m, right_column, draws);
            exact_missed_total += draws - exact_flags;
            exact_total += draws;
        }
    fprintf(report_file, "\n");
    printf("  exact-zero rule missed %d of %d exactly singular designs\n", exact_missed_total, exact_total);
    printf("  smallest largest |x| it let through: %.3g\n", smallest_let_through);
    CHECK(exact_missed_total > exact_total / 2,
          "the exact-zero rule is expected to miss most exactly singular designs (missed %d of %d)",
          exact_missed_total, exact_total);
    fprintf(report_file, "smallest largest |x| over every design the exact-zero rule let through: %.3g\n\n",
            smallest_let_through);
}

/* c0 = 2^k v, c1 = 2^k v + u with v, u small integers, b = 3 c0 - 2 c1, so the
   true coefficients are exactly (3, -2) and the residual is exactly zero. The
   sine of the angle between c1 and c0 is computed from the integers:
   ||u - (u.v / v.v) v||^2 / ||c1||^2. */
typedef struct { double sine, error; int exact, tolerance; } SweepCell;

static SweepCell sweep_cell(Rng *rng, int m, int k) {
    long double vv = 0, uu = 0, uv = 0;
    Mat a = mat_new(m, 2), b = mat_new(m, 1);
    long double scale = ldexpl(1.0L, k);
    for (int i = 0; i < m; i++) {
        double v = small_int(rng, 8), u = small_int(rng, 3);
        if (v == 0) v = 1;
        vv += v * v; uu += u * u; uv += u * v;
        double c0 = (double)(scale * v), c1 = (double)(scale * v + u);
        AT(a, i, 0) = (mreal)c0;
        AT(a, i, 1) = (mreal)c1;
        AT(b, i, 0) = (mreal)(3 * c0 - 2 * c1);
    }
    long double perp = uu - uv * uv / vv;
    long double c1_norm = scale * scale * vv + 2 * scale * uv + uu;
    SweepCell cell;
    cell.sine = perp > 0 ? (double)sqrtl(perp / c1_norm) : 0.0;
    Mat x;
    cell.exact = exact_rule_lstsq(a, b, &x);
    cell.tolerance = tolerance_rule_lstsq(a, b);
    cell.error = cell.exact ? NAN : fmax(fabs((double)AT(x, 0, 0) - 3), fabs((double)AT(x, 1, 0) + 2)) / 3;
    mat_free(a); mat_free(b); mat_free(x);
    return cell;
}

static double worst_model_ratio = 0;

static void test_sweep(Rng *rng) {
    puts("independence shrinking like 2^-k, true coefficients known");
    int mantissa = sizeof(mreal) == sizeof(double) ? 53 : 24;
    int sizes[] = { 10, 100, 1000 };
    int draws = 20;
    fprintf(report_file, "Two columns, c1 = 2^k c0 + small integers, b = 3 c0 - 2 c1 exactly; %d draws per row\n", draws);
    fprintf(report_file, "sine is the exact sine of the angle between c1 and c0; error is max |x - (3, -2)| / 3\n");
    fprintf(report_file, "of the solution _gels returned, over the draws the exact-zero rule let through\n");
    fprintf(report_file, "%6s %4s %11s %11s %12s %16s %13s %13s\n", "rows", "k", "sine", "tolerance",
            "exact rule", "tolerance rule", "median error", "worst error");
    for (int s = 0; s < 3; s++) {
        int m = sizes[s];
        for (int k = 0; k <= mantissa - 4; k++) {
            SweepCell cells[20];
            int exact_flags = 0, tolerance_flags = 0;
            double errors[20];
            int n_errors = 0;
            double sine_sum = 0;
            for (int d = 0; d < draws; d++) {
                cells[d] = sweep_cell(rng, m, k);
                sine_sum += cells[d].sine;
                exact_flags += cells[d].exact != 0;
                tolerance_flags += cells[d].tolerance != 0;
                if (!cells[d].exact) errors[n_errors++] = cells[d].error;
            }
            double median = NAN, worst = NAN;
            if (n_errors) {
                for (int i = 1; i < n_errors; i++)
                    for (int j = i; j > 0 && errors[j] < errors[j - 1]; j--) {
                        double t = errors[j]; errors[j] = errors[j - 1]; errors[j - 1] = t;
                    }
                median = errors[n_errors / 2];
                worst = errors[n_errors - 1];
            }
            fprintf(report_file, "%6d %4d %11.3g %11.3g %9d/%d %13d/%d %13.3g %13.3g\n", m, k, sine_sum / draws,
                    rule_tolerance(m), exact_flags, draws, tolerance_flags, draws, median, worst);

            /* Checked per draw rather than per row, since the sine varies with the draw. */
            for (int d = 0; d < draws; d++) {
                SweepCell c = cells[d];
                /* The rounding model the tolerance rests on: a Householder QR
                   solution carries relative error of about sqrt(m) * MEPS
                   times the condition number, and for two columns at angle
                   theta that is about 2 / sin(theta). */
                if (!c.exact && c.sine > 0) {
                    double model = c.error * c.sine / (double)MEPS;
                    if (model > worst_model_ratio) worst_model_ratio = model;
                    CHECK(model <= 2 * sqrt((double)m),
                          "%d rows, k = %d: error %.3g at sine %.3g exceeds 2 sqrt(m) MEPS / sine", m, k, c.error, c.sine);
                }
                if (c.sine > 3 * rule_tolerance(m))
                    CHECK(c.tolerance == 0, "%d rows, k = %d: sine %.3g, three times the tolerance, was rejected",
                          m, k, c.sine);
                if (c.sine < rule_tolerance(m) / 3)
                    CHECK(c.tolerance == 2, "%d rows, k = %d: sine %.3g, a third of the tolerance, was accepted",
                          m, k, c.sine);
            }
        }
    }
    fprintf(report_file, "worst error * sine / MEPS over every draw the exact-zero rule let through: %.3g\n\n",
            worst_model_ratio);
    printf("  worst error * sine / MEPS: %.3g\n", worst_model_ratio);
}

static void report_pair(const char *label, Mat a, int *exact_out, int *tolerance_out) {
    Mat b = mat_new(a.r, 1);
    for (int i = 0; i < a.r; i++) AT(b, i, 0) = 1;
    Mat x;
    int exact = exact_rule_lstsq(a, b, &x);
    int tolerance = tolerance_rule_lstsq(a, b);
    fprintf(report_file, "%-58s exact rule %d   tolerance rule %d   largest |x| %.3g\n", label, exact, tolerance,
            exact ? 0.0 : max_abs(x));
    printf("  %-58s exact %d, tolerance %d\n", label, exact, tolerance);
    if (exact_out) *exact_out = exact;
    if (tolerance_out) *tolerance_out = tolerance;
    mat_free(b); mat_free(x);
}

static void test_adversarial(Rng *rng) {
    puts("trying to break either rule");
    fprintf(report_file, "Adversarial designs; a verdict is the 1-based column flagged, 0 for accepted\n");
    int m = 100;
    int exact, tolerance;

    /* Dependence through a nearly collinear pair: c0 = K v, c1 = K v + w,
       c2 = c1 - c0 = w exactly. c1 is independent of c0, only barely; c2 is
       exactly dependent on the two, but QR finds that out by subtracting two
       numbers of size K that agree to within w, so the rounding left in its
       diagonal is of order MEPS * K relative to |w|. */
    int shift = sizeof(mreal) == sizeof(double) ? 30 : 12;
    Mat a = mat_new(m, 3), reordered = mat_new(m, 3);
    for (int i = 0; i < m; i++) {
        double v = small_int(rng, 8), w = small_int(rng, 3);
        double big = ldexp(v, shift);
        AT(a, i, 0) = (mreal)big; AT(a, i, 1) = (mreal)(big + w); AT(a, i, 2) = (mreal)w;
        AT(reordered, i, 0) = (mreal)w; AT(reordered, i, 1) = (mreal)big; AT(reordered, i, 2) = (mreal)(big + w);
    }
    report_pair("c2 = c1 - c0 with c1 nearly collinear with c0", a, &exact, &tolerance);
    int rank;
    Mat b = mat_new(m, 1);
    for (int i = 0; i < m; i++) AT(b, i, 0) = 1;
    Mat x_rd = mat_lstsq_rd(a, b, &rank);
    fprintf(report_file, "%-58s mat_lstsq_rd rank %d of 3\n", "  same design", rank);
    /* Its cutoff is a fixed 10 * FLT_EPSILON relative to the largest singular
       value, so at float64 it also counts the barely independent c1 as
       dependent and reports rank 1: rank deficient either way. */
    CHECK(rank < 3, "mat_lstsq_rd reports the rank deficiency the column rules miss (rank %d)", rank);
    mat_free(x_rd); mat_free(b);
    report_pair("the same columns ordered w, K v, K v + w", reordered, &exact, &tolerance);
    CHECK(tolerance == 3, "reordered, the dependent column is found (status %d)", tolerance);
    mat_free(a); mat_free(reordered);

    /* Independent columns at the ends of the range. The design is well
       conditioned once each column is scaled to unit length, so both rules
       have to accept it. */
    double magnitudes_double[] = { 1e200, 1e-200, 1e150, 1e-150 };
    double magnitudes_float[] = { 1e30, 1e-30, 1e18, 1e-18 };
    double *magnitudes = sizeof(mreal) == sizeof(double) ? magnitudes_double : magnitudes_float;
    for (int t = 0; t < 4; t++) {
        a = mat_new(m, 3);
        for (int i = 0; i < m; i++) {
            AT(a, i, 0) = (mreal)(magnitudes[t] * small_int(rng, 8));
            AT(a, i, 1) = (mreal)small_int(rng, 8);
            AT(a, i, 2) = (mreal)(magnitudes[t] * small_int(rng, 8));
        }
        AT(a, 0, 0) = (mreal)(magnitudes[t] * 9);
        char label[96];
        snprintf(label, sizeof label, "independent, columns 1 and 3 at %g", magnitudes[t]);
        report_pair(label, a, &exact, &tolerance);
        CHECK(tolerance == 0, "%s: tolerance rule rejected a well-conditioned design (status %d)", label, tolerance);
        CHECK(exact == 0, "%s: exact rule rejected a well-conditioned design (info %d)", label, exact);
        mat_free(a);

        a = mat_new(m, 3);
        for (int i = 0; i < m; i++) {
            double c0 = small_int(rng, 8), c1 = small_int(rng, 8);
            AT(a, i, 0) = (mreal)(magnitudes[t] * c0);
            AT(a, i, 1) = (mreal)(magnitudes[t] * c1);
            AT(a, i, 2) = (mreal)(magnitudes[t] * (c0 - c1));
        }
        snprintf(label, sizeof label, "c2 = c0 - c1, all at %g", magnitudes[t]);
        report_pair(label, a, &exact, &tolerance);
        CHECK(tolerance == 3, "%s: tolerance rule missed the dependent column (status %d)", label, tolerance);
        mat_free(a);
    }

    /* Below the smallest normal number. -ffast-math sets flush-to-zero, so
       these entries read as zeros and both rules see a zero column. */
    a = mat_new(m, 2);
    for (int i = 0; i < m; i++) {
        AT(a, i, 0) = (mreal)small_int(rng, 8);
        AT(a, i, 1) = (mreal)((sizeof(mreal) == sizeof(double) ? 1e-310 : 1e-40) * (1 + i % 3));
    }
    report_pair("second column subnormal", a, NULL, NULL);
    mat_free(a);

    /* Non-finite entries. The tolerance rule has to reject them rather than
       return a solution made of NaN; the report shows what the exact-zero
       rule does with them. */
    a = mat_new(m, 3);
    for (int i = 0; i < m * 3; i++) a.d[i] = (mreal)small_int(rng, 8);
    AT(a, 5, 1) = check_non_finite(1);
    CHECK(check_stored_non_finite(&AT(a, 5, 1), 1), "the stored entry really is infinite");
    report_pair("one infinite entry in column 2", a, NULL, &tolerance);
    CHECK(tolerance != 0, "a design with an infinite entry is rejected (status %d)", tolerance);
    AT(a, 5, 1) = check_non_finite(0);
    CHECK(check_stored_non_finite(&AT(a, 5, 1), 0), "the stored entry really is NaN");
    report_pair("one NaN entry in column 2", a, NULL, &tolerance);
    CHECK(tolerance != 0, "a design with a NaN entry is rejected (status %d)", tolerance);
    mat_free(a);

    /* A long sample: the tolerance grows like sqrt(m), and so does the
       rounding it has to cover. */
    int long_m = 200000;
    a = exact_singular_design(rng, COMBINATION, long_m);
    char label[96];
    snprintf(label, sizeof label, "c3 = c0 + 2 c1 - c2, %d rows", long_m);
    report_pair(label, a, &exact, &tolerance);
    CHECK(tolerance == 4, "%s: tolerance rule missed (status %d)", label, tolerance);
    mat_free(a);

    /* Square and exactly singular. */
    int n = 40;
    a = mat_new(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) AT(a, i, j) = (mreal)small_int(rng, 8);
        AT(a, i, n - 1) = AT(a, i, 0) - AT(a, i, 1) + AT(a, i, 2);
    }
    report_pair("40 x 40, last column c0 - c1 + c2", a, &exact, &tolerance);
    CHECK(tolerance == n, "40 x 40: tolerance rule missed (status %d)", tolerance);
    mat_free(a);
    fprintf(report_file, "\n");
}

/* Gram matrices of the exact patterns: integer entries, exactly singular,
   stored exactly. */
static void test_cholesky(Rng *rng) {
    puts("Cholesky on exactly singular Gram matrices");
    fprintf(report_file, "Cholesky on X^T X of the exact patterns, 50 draws per row; then scaled by 2^s\n");
    fprintf(report_file, "%-22s %6s %12s %16s\n", "pattern", "rows", "exact rule", "tolerance rule");
    int sizes[] = { 16, 1000 };
    int draws = 50;
    int missed = 0, total = 0;
    for (int p = 0; p < N_PATTERNS; p++) {
        if (p == WIDE_RANGE) continue; /* X^T X of it is not stored exactly in float32 */
        for (int s = 0; s < 2; s++) {
            int m = sizes[s], exact_flags = 0, tolerance_flags = 0;
            for (int draw = 0; draw < draws; draw++) {
                Mat x = exact_singular_design(rng, (Pattern)p, m);
                Mat xt = mat_T(x), gram = mat_mul(xt, x);
                if (exact_rule_chol(gram)) exact_flags++;
                if (tolerance_rule_chol(gram) == 4) tolerance_flags++;
                mat_free(x); mat_free(xt); mat_free(gram);
            }
            fprintf(report_file, "%-22s %6d %9d/%d %13d/%d\n", pattern_names[p], m, exact_flags, draws,
                    tolerance_flags, draws);
            CHECK(tolerance_flags == draws, "Cholesky, %s, %d rows: pivot 4 reported in %d of %d",
                  pattern_names[p], m, tolerance_flags, draws);
            missed += draws - exact_flags;
            total += draws;
        }
    }
    printf("  exact-zero rule missed %d of %d exactly singular Gram matrices\n", missed, total);
    CHECK(missed > 0, "the exact-zero rule is expected to miss some exactly singular Gram matrices (missed %d)", missed);

    /* The same, scaled by powers of two, which keeps every entry exact. */
    int shifts_double[] = { 900, -900, 400, -400 };
    int shifts_float[] = { 100, -100, 50, -50 };
    int *shifts = sizeof(mreal) == sizeof(double) ? shifts_double : shifts_float;
    for (int t = 0; t < 4; t++) {
        Mat x = exact_singular_design(rng, COMBINATION, 100);
        Mat xt = mat_T(x), gram = mat_mul(xt, x);
        Mat independent = mat_copy(gram);
        for (int k = 0; k < 4; k++) AT(independent, k, k) += 1;
        for (int i = 0; i < 16; i++) {
            gram.d[i] = (mreal)ldexp((double)gram.d[i], shifts[t]);
            independent.d[i] = (mreal)ldexp((double)independent.d[i], shifts[t]);
        }
        int exact = exact_rule_chol(gram), tolerance = tolerance_rule_chol(gram);
        int exact_ind = exact_rule_chol(independent), tolerance_ind = tolerance_rule_chol(independent);
        fprintf(report_file, "scaled by 2^%-5d singular: exact rule %d, tolerance rule %d; "
                "plus identity before scaling: exact rule %d, tolerance rule %d\n",
                shifts[t], exact, tolerance, exact_ind, tolerance_ind);
        printf("  Cholesky scaled by 2^%d: singular exact %d tolerance %d, regular exact %d tolerance %d\n",
               shifts[t], exact, tolerance, exact_ind, tolerance_ind);
        CHECK(tolerance == 4, "Cholesky scaled by 2^%d: singular matrix not reported at pivot 4 (%d)", shifts[t], tolerance);
        CHECK(tolerance_ind == 0, "Cholesky scaled by 2^%d: regular matrix rejected (%d)", shifts[t], tolerance_ind);
        mat_free(x); mat_free(xt); mat_free(gram); mat_free(independent);
    }
    fprintf(report_file, "\n");
}

int main(void) {
    check_banner("the tolerance rule against the exact-zero rule");
    mkdir("out", 0755);
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/singularity_rule_comparison_float64_report.txt"
                     : "out/singularity_rule_comparison_float32_report.txt";
    report_file = fopen(path, "w");
    if (!report_file) { perror(path); return 1; }
    fprintf(report_file, "Tolerance rule against exact-zero rule, %s build\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");

    Rng rng = rng_new(20260924, 4);
    test_exact(&rng);
    test_sweep(&rng);
    test_adversarial(&rng);
    test_cholesky(&rng);

    fclose(report_file);
    printf("\nfull tables in %s\n", path);
    return check_report();
}
