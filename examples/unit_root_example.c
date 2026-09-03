#include <sys/stat.h>
#include "frame/csv.h"
#include "inference/unit_root.h"
#include "inference/cointegration.h"

/* Is each of these US macro series stationary, and are any two of them
   co-integrated - the question every one of these tests exists to answer,
   asked once on a real quarterly sample.

   The data is examples/datasets/us_real.csv: 192 quarters of US real GDP,
   consumption, CPI, investment, unemployment and the effective federal funds
   rate, 1973Q1 onward, one row per quarter.

   The point of running several tests rather than one is that they disagree,
   and the disagreement is the result. ADF's null is a unit root and KPSS's is
   stationarity, so a series both agree on is settled and one they split on is
   a modelling choice that has to be stated. The break tests then ask whether
   an apparent unit root is really a trend that shifted once.

   Every critical value in this file is at the 5 per cent level, which is index
   1 in ADF's and KPSS's arrays, index 1 in the simulated ones, and
   HHLT_LEVEL_05 for the break tests. Where a test simulates its own critical
   values the draw count is 2000, which is enough for a 5 per cent quantile and
   deliberately not enough for a 1 per cent one - see each header's own note.

   Results go to examples/out/, never to the terminal. */

enum { DRAWS = 2000 };

/* One row of the unit root table: four tests applied to one series, with the
   verdict each returns spelled out rather than left as a comparison for the
   reader to make. */
static void write_unit_root_row(FILE *out, const char *name, Mat series, unsigned long long seed) {
    int n = stats_series_length(series);
    int lags = adf_max_lags(n);
    int first = lags + 1;

    AdfResult constant = adf(series, lags, first);
    AdfResult trend = adf_with_deterministic(series, lags, first, ADF_CONSTANT_TREND);
    mreal trend_critical = adf_critical_value_for(trend.observations, 1, ADF_CONSTANT_TREND);
    KpssResult level = kpss_level(series, kpss_bandwidth(n));
    DfglsResult gls = dfgls(series, lags, DFGLS_CONSTANT_TREND);
    DfglsCritical gls_critical =
        dfgls_critical(gls.observations, lags, DFGLS_CONSTANT_TREND, DRAWS, seed);

    fprintf(out, "%-13s", name);
    fprintf(out, " %7.3f %7.3f %-11s", (double)constant.statistic, (double)constant.critical[1],
            constant.statistic < constant.critical[1] ? "stationary" : "unit root");
    fprintf(out, " %7.3f %7.3f %-11s", (double)trend.statistic, (double)trend_critical,
            trend.statistic < trend_critical ? "stationary" : "unit root");
    /* KPSS rejects by being large, the other three by being small */
    fprintf(out, " %7.3f %7.3f %-11s", (double)level.statistic, (double)level.critical[1],
            level.statistic > level.critical[1] ? "unit root" : "stationary");
    fprintf(out, " %7.3f %7.3f %-11s\n", (double)gls.statistic, (double)gls_critical.critical[1],
            gls.statistic < gls_critical.critical[1] ? "stationary" : "unit root");
}

/* The break tests, which ask a different question: is what looks like a unit
   root actually a trend that shifted once at a date nobody supplied. */
static void write_break_row(FILE *out, const char *name, Mat series, unsigned long long seed) {
    int n = stats_series_length(series);
    int lags = adf_max_lags(n);

    ZivotAndrewsResult za = zivot_andrews(series, lags, ZA_BOTH, (mreal)0.15);
    ZivotAndrewsCritical za_critical =
        zivot_andrews_critical(n, lags, ZA_BOTH, (mreal)0.15, DRAWS, seed);
    HltBreakResult trend_break =
        hlt_break(series, 0, 1, (mreal)0.15, (mreal)0.85, (mreal)10, (mreal)20);
    HhltResult hh = hhlt(series, lags, (mreal)6, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);

    fprintf(out, "%-13s", name);
    fprintf(out, " %7.3f %7.3f %-11s q%-3d", (double)za.statistic,
            (double)za_critical.critical[1],
            za.statistic < za_critical.critical[1] ? "stationary" : "unit root",
            za.break_index);
    fprintf(out, " %8.3f %7.3f %-9s", (double)trend_break.statistic,
            (double)trend_break.critical,
            trend_break.rejects ? "break" : "no break");
    fprintf(out, " %8.3f %7.3f %-11s %s\n", (double)hh.statistic, (double)hh.critical,
            hh.rejects ? "stationary" : "unit root",
            hh.allows_break ? "with a break" : "no break allowed");
}

int main(void) {
    mkdir("examples/out", 0755);

    DataFrame frame = df_read_csv("examples/datasets/us_real.csv", csv_read_options_default());
    const char *names[] = { "GDP", "Consumption", "Cpi", "Investment",
                            "Unemployment", "Fed_rate" };
    enum { N_SERIES = 6 };

    /* The levels are what the unit root tests want. GDP, consumption and
       investment are taken in logs, where a constant growth rate is a linear
       trend rather than an exponential one; the CPI index and the two rate
       series are left as they are. */
    Mat columns[N_SERIES];
    for (int k = 0; k < N_SERIES; k++) {
        Mat raw = df_col_numeric(&frame, names[k]);
        int n = raw.r;
        int take_log = (k == 0 || k == 1 || k == 3);
        columns[k] = mat_new(1, n);
        for (int t = 0; t < n; t++)
            AT(columns[k], 0, t) = take_log ? (mreal)log((double)AT(raw, t, 0))
                                            : AT(raw, t, 0);
    }
    int periods = columns[0].c;

    FILE *out = fopen("examples/out/unit_root_example_report.txt", "w");
    assert(out && "cannot open examples/out/unit_root_example_report.txt for writing");

    fprintf(out, "US quarterly macro series, %d quarters from 1973Q1\n", periods);
    fprintf(out, "examples/datasets/us_real.csv; GDP, consumption and investment in logs\n");
    fprintf(out, "every verdict at the 5 per cent level; simulated critical values "
                 "from %d draws\n\n", DRAWS);

    fprintf(out, "unit root tests\n");
    fprintf(out, "%-13s %7s %7s %-11s %7s %7s %-11s %7s %7s %-11s %7s %7s %-11s\n",
            "series", "ADF", "5pc", "verdict", "ADF+t", "5pc", "verdict",
            "KPSS", "5pc", "verdict", "DFGLS", "5pc", "verdict");
    for (int k = 0; k < N_SERIES; k++)
        write_unit_root_row(out, names[k], columns[k], 20260101ull + k);

    fprintf(out, "\nADF and KPSS have opposite nulls, so their two verdict columns are\n"
                 "independent evidence rather than the same answer twice. A series they\n"
                 "disagree on is a modelling choice, not a measurement. ADF+t adds a\n"
                 "linear trend to the regression and DFGLS detrends first, which is what\n"
                 "buys it power against a persistent alternative.\n");

    fprintf(out, "\nbreak tests: is the unit root really a trend that shifted once\n");
    fprintf(out, "%-13s %7s %7s %-11s %-4s %8s %7s %-9s %8s %7s %-11s %s\n",
            "series", "ZA", "5pc", "verdict", "at", "HLT", "5pc", "verdict",
            "HHLT", "5pc", "verdict", "regressor");
    for (int k = 0; k < N_SERIES; k++)
        write_break_row(out, names[k], columns[k], 20260201ull + k);

    fprintf(out, "\nZivot-Andrews takes a minimum over every candidate date and pays for it\n"
                 "in critical value whether or not a break exists; HHLT decides first and\n"
                 "pays only when it decides a break is there. HLT tests for the break\n"
                 "itself and says nothing about a unit root.\n");

    /* Co-integration between the two series the score-driven literature pairs:
       the price level and the federal funds rate. Both directions of
       Engle-Granger, because it fixes a normalization and is not symmetric,
       then Johansen on the pair, which fixes none. */
    Mat pair = mat_new(2, periods);
    for (int t = 0; t < periods; t++) {
        AT(pair, 0, t) = AT(columns[2], 0, t);  /* Cpi */
        AT(pair, 1, t) = AT(columns[5], 0, t);  /* Fed_rate */
    }
    int lags = 4;

    fprintf(out, "\nco-integration between the CPI and the federal funds rate, %d lags\n", lags);
    for (int dependent = 0; dependent < 2; dependent++) {
        EngleGrangerResult eg = engle_granger(pair, dependent, lags, 0);
        EngleGrangerCritical eg_critical =
            engle_granger_critical(2, eg.observations, lags, 0, DRAWS, 424242ull + dependent);
        fprintf(out, "Engle-Granger, %-8s on the other: %8.3f against %8.3f, %s\n",
                dependent == 0 ? "Cpi" : "Fed_rate",
                (double)eg.statistic, (double)eg_critical.critical[1],
                eg.statistic < eg_critical.critical[1]
                    ? "co-integrated" : "not co-integrated");
        engle_granger_result_free(&eg);
    }
    fprintf(out, "The two rows are different tests, not one test run twice: each fixes a\n"
                 "different normalization, and on a borderline relation that choice decides\n"
                 "the answer.\n\n");

    JohansenResult jo = johansen(pair, lags);
    fprintf(out, "Johansen on the same pair, %d effective observations\n", jo.observations);
    for (int r = 0; r < jo.n; r++) {
        JohansenCritical critical =
            johansen_critical(jo.n - r, jo.observations, DRAWS, 909090ull + r);
        fprintf(out, "  rank at most %d: trace %8.3f against %8.3f %-9s"
                     "  max eigenvalue %8.3f against %8.3f %s\n",
                r, (double)AT(jo.trace_statistic, r, 0), (double)critical.trace[1],
                AT(jo.trace_statistic, r, 0) > critical.trace[1] ? "reject" : "no reject",
                (double)AT(jo.max_statistic, r, 0), (double)critical.max[1],
                AT(jo.max_statistic, r, 0) > critical.max[1] ? "reject" : "no reject");
    }
    fprintf(out, "Read the trace column downward and stop at the first rank it fails to\n"
                 "reject: that rank is the answer.\n");
    johansen_result_free(&jo);

    /* Maki, for the case the relation itself breaks. The two tests above
       assume it does not, and a relation whose intercept shifted looks like no
       relation at all to them. */
    MakiResult mk = maki(pair, 0, 2, 3, lags, (mreal)0.15);
    MakiCritical mk_critical = maki_critical(2, periods, 2, 3, lags, (mreal)0.15, 200, 5150ull);
    fprintf(out, "\nMaki, model 2, up to 3 breaks: %8.3f against %8.3f, %s\n",
            (double)mk.statistic, (double)mk_critical.critical[1],
            mk.statistic < mk_critical.critical[1]
                ? "co-integrated once breaks are allowed" : "not co-integrated");
    fprintf(out, "%d breaks found over %d candidate dates:", mk.n_breaks, mk.candidates);
    for (int b = 0; b < mk.n_breaks; b++)
        fprintf(out, " q%d", mk.breaks[b]);
    fprintf(out, "\nIts critical values come from only 200 draws here rather than %d, since\n"
                 "every draw runs one regression and one ADF per candidate date per pass -\n"
                 "the most expensive simulation in the package.\n", DRAWS);

    fclose(out);

    mat_free(pair);
    for (int k = 0; k < N_SERIES; k++) mat_free(columns[k]);
    df_free(&frame);
    return 0;
}
