#pragma once
#include "../special.h"

/* The chi-squared distribution's upper tail, the p-value of a Wald,
   likelihood ratio or portmanteau statistic.

   Scalar and in double, like dist/gauss.h's gauss_standard_cdf, and
   including nothing but special.h: stats.h's Ljung-Box test needs this tail
   and sits directly above linalg/mat.h, below the linalg chain the other
   dist/ files are built on, so a file that pulled in linalg/ would pull stats.h
   up with it. */

/* P(X > x) for X ~ chi-squared(df), as Q(df/2, x/2) since chi-squared(df) is
   Gamma(df/2, 2). Evaluated from Q directly rather than as 1 - P, so a small
   tail keeps its relative accuracy; special.h documents the incomplete gamma
   function and its accuracy at large df. A negative x or a df that is not
   positive returns not-a-number, as special_gammainc_q does: a degenerate
   statistic is a result to report, not programmer error. */
static inline double chi_squared_sf(double x, double df) {
    return special_gammainc_q(df * 0.5, x * 0.5);
}
