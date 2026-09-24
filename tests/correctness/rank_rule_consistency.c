/*
Do the package's rank decisions agree with one another?

Every rank or singularity decision in the library uses one rule,
mat_rank_tolerance(length) in linalg/decomp.h, applied to a different
quantity: mat_lstsq to each column's part outside the span of the earlier
ones, mat_chol to each pivot, mat_rank and mat_lstsq_rd to the singular
values. The rule's documentation states what that makes consistent and what
it does not; this file checks the statements, on the same designs, one
design at a time:

  implication     when mat_lstsq flags a design, mat_rank and mat_lstsq_rd
                  report a rank below its column count, and mat_chol flags its
                  Gram matrix X^T X
  same rule       mat_rank and mat_lstsq_rd, two different SVD routines, report
                  the same rank, and both count a singular value at three times
                  the tolerance and drop one at a third of it
  converse        the reverse implication fails, as documented, for a column in
                  much smaller units than the rest: mat_lstsq, which is unchanged
                  by rescaling a column, accepts it, and the singular-value rank
                  drops

The random designs mix exactly dependent columns, columns at a controlled
angle to the earlier ones from far above to far below the tolerance, and
columns rescaled by up to 1e3 either way. A draw is left out of a comparison
when the quantity that comparison turns on lies within a factor 3 of the
tolerance, where rounding in the last digits may put two computations on
opposite sides; how many were left out is printed.

Run with make tests/correctness/rank_rule_consistency. STRESS=1 widens the
sweep from 2000 to 20000 designs.
*/

#include "../check.h"
#include "../../linalg/solver.h"

static double smallest_sine_long_double(Mat a) {
    int m = a.r, n = a.c;
    long double *basis = malloc((size_t)m * n * sizeof *basis), *column = malloc((size_t)m * sizeof *column);
    double smallest = 1;
    for (int j = 0; j < n; j++) {
        long double norm_sq = 0;
        for (int i = 0; i < m; i++) { column[i] = AT(a, i, j); norm_sq += column[i] * column[i]; }
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < j; k++) {
                long double dot = 0;
                for (int i = 0; i < m; i++) dot += basis[(size_t)k * m + i] * column[i];
                for (int i = 0; i < m; i++) column[i] -= dot * basis[(size_t)k * m + i];
            }
        long double rest_sq = 0;
        for (int i = 0; i < m; i++) rest_sq += column[i] * column[i];
        double sine = norm_sq > 0 ? (double)sqrtl(rest_sq / norm_sq) : 0;
        if (sine < smallest) smallest = sine;
        long double rest = sqrtl(rest_sq);
        for (int i = 0; i < m; i++) basis[(size_t)j * m + i] = rest > 0 ? column[i] / rest : 0;
    }
    free(basis); free(column);
    return smallest;
}

/* Singular values of a through mat_svd, relative to the largest: whether any
   lies within a factor 3 of the tolerance. */
static int singular_values_near_cut(Mat a, double tolerance) {
    Mat u, vt;
    Vec s;
    mat_svd(a, &u, &s, &vt);
    int near = 0;
    for (int i = 0; i < s.r; i++) {
        double ratio = (double)AT(s, i, 0) / (double)AT(s, 0, 0);
        if (ratio > tolerance / 3 && ratio < 3 * tolerance) near = 1;
    }
    mat_free(u); mat_free(vt); mat_free(s);
    return near;
}

/* Columns are either fresh or a copy of an earlier column plus sine times
   fresh noise at that column's size, and then every column is rescaled by
   its own factor, so dependence and units vary independently. */
static Mat design(Rng *rng, int m, int n) {
    Mat a = mat_new(m, n);
    double sines[] = { 0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1 };
    double *unscaled = malloc((size_t)m * n * sizeof *unscaled);
    for (int j = 0; j < n; j++) {
        int dependent = j > 0 && rng_uniform(rng) < 0.4;
        int source = dependent ? (int)rng_below(rng, (uint64_t)j) : 0;
        double sine = sines[rng_below(rng, 7)];
        for (int i = 0; i < m; i++) {
            double fresh = rng_normal(rng);
            unscaled[i * n + j] = dependent ? unscaled[i * n + source] + sine * fresh : fresh;
        }
    }
    for (int j = 0; j < n; j++) {
        double scale = pow(10.0, 6 * rng_uniform(rng) - 3);
        for (int i = 0; i < m; i++) AT(a, i, j) = (mreal)(scale * unscaled[i * n + j]);
    }
    free(unscaled);
    return a;
}

static Mat gram_long_double(Mat a) {
    int m = a.r, n = a.c;
    Mat g = mat_new(n, n);
    for (int p = 0; p < n; p++)
        for (int q = 0; q < n; q++) {
            long double s = 0;
            for (int i = 0; i < m; i++) s += (long double)AT(a, i, p) * AT(a, i, q);
            AT(g, p, q) = (mreal)s;
        }
    return g;
}

static void test_random_designs(Rng *rng) {
    int draws = getenv("STRESS") ? 20000 : 2000;
    printf("random designs, %d draws\n", draws);
    int flagged = 0, implication_compared = 0, same_rule_compared = 0, converse = 0;
    for (int draw = 0; draw < draws; draw++) {
        int n = 2 + (int)rng_below(rng, 9);
        int m = n + (int)rng_below(rng, 200);
        Mat a = design(rng, m, n), b = mat_new(m, 1);
        for (int i = 0; i < m; i++) AT(b, i, 0) = (mreal)rng_normal(rng);
        double column_tolerance = mat_rank_tolerance(m);
        double sine = smallest_sine_long_double(a);

        int status;
        Mat x = mat_lstsq(a, b, &status);
        mat_free(x);
        int rank = mat_rank(a), rd_rank;
        Mat x_rd = mat_lstsq_rd(a, b, &rd_rank);
        mat_free(x_rd);

        if (!singular_values_near_cut(a, mat_rank_tolerance(m))) {
            CHECK(rank == rd_rank, "draw %d, %d x %d: mat_rank %d, mat_lstsq_rd %d", draw, m, n, rank, rd_rank);
            same_rule_compared++;
        }
        if (!(sine > column_tolerance / 3 && sine < 3 * column_tolerance)) {
            implication_compared++;
            if (status) {
                flagged++;
                CHECK(rank < n, "draw %d, %d x %d: mat_lstsq flagged column %d, mat_rank says %d", draw, m, n, status, rank);
                CHECK(rd_rank < n, "draw %d, %d x %d: mat_lstsq flagged column %d, mat_lstsq_rd says %d", draw, m, n, status, rd_rank);
                Mat g = gram_long_double(a);
                int chol_status;
                Mat l = mat_chol(g, &chol_status);
                mat_free(l);
                CHECK(chol_status > 0, "draw %d, %d x %d: mat_lstsq flagged column %d, mat_chol accepted X^T X", draw, m, n, status);
                mat_free(g);
            } else if (rank < n) {
                converse++;
            }
        }
        mat_free(a); mat_free(b);
    }
    printf("  implication compared on %d draws, %d of them flagged by mat_lstsq\n", implication_compared, flagged);
    printf("  same-rule comparison on %d draws\n", same_rule_compared);
    printf("  %d draws accepted by mat_lstsq with a lower singular-value rank\n", converse);
    CHECK(flagged > implication_compared / 10, "the implication is exercised (%d flagged)", flagged);
    CHECK(implication_compared > draws / 2 && same_rule_compared > draws / 2, "most draws are compared");
}

/* Singular values placed exactly: a 1000 x 2 matrix whose two columns are
   e_1 and ratio * e_2, so its singular values are 1 and ratio. At three
   times the tolerance both SVD routines have to count two, at a third of it
   one. 1000 rows is where the rule differs from NumPy's max(m, n) * MEPS by
   more than that factor 3: the old threshold would have counted one. */
static void test_singular_value_threshold(void) {
    puts("singular values either side of the tolerance");
    int m = 1000;
    double tolerance = mat_rank_tolerance(m);
    double ratios[] = { 3 * tolerance, tolerance / 3 };
    int expected[] = { 2, 1 };
    for (int t = 0; t < 2; t++) {
        Mat a = mat_new(m, 2), b = mat_new(m, 1);
        for (int i = 0; i < m * 2; i++) a.d[i] = 0;
        for (int i = 0; i < m; i++) AT(b, i, 0) = 1;
        AT(a, 0, 0) = 1;
        AT(a, 1, 1) = (mreal)ratios[t];
        int rd_rank;
        Mat x = mat_lstsq_rd(a, b, &rd_rank);
        CHECK(mat_rank(a) == expected[t], "ratio %.3g: mat_rank %d, expected %d", ratios[t], mat_rank(a), expected[t]);
        CHECK(rd_rank == expected[t], "ratio %.3g: mat_lstsq_rd %d, expected %d", ratios[t], rd_rank, expected[t]);
        mat_free(x); mat_free(a); mat_free(b);
    }
}

/* A well-conditioned design with one column in units 1e-15 of the others
   (1e-7 in float32), below the singular-value cutoff: its singular values
   span that factor, so the singular-value rank drops, while every column's
   part outside the others is most of the column. */
static void test_converse_by_scaling(Rng *rng) {
    puts("the converse fails for a column in much smaller units");
    double small = sizeof(mreal) == sizeof(double) ? 1e-15 : 1e-7;
    int m = 50, n = 3;
    Mat a = mat_new(m, n), b = mat_new(m, 1);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) AT(a, i, j) = (mreal)rng_normal(rng);
        AT(a, i, 2) *= (mreal)small;
        AT(b, i, 0) = (mreal)rng_normal(rng);
    }
    int status;
    Mat x = mat_lstsq(a, b, &status);
    CHECK(status == 0, "mat_lstsq accepts it (status %d)", status);
    CHECK(mat_rank(a) == 2, "mat_rank drops to 2 (got %d)", mat_rank(a));
    mat_free(x); mat_free(a); mat_free(b);
}

int main(void) {
    check_banner("the package's rank decisions against one another");
    Rng rng = rng_new(20260924, 6);
    test_random_designs(&rng);
    test_singular_value_threshold();
    test_converse_by_scaling(&rng);
    return check_report();
}
