/* Entry point to brent_bounded for tests/correctness/brent_reference_agreement.py:
   the objectives are written here, selected by number, and every point the
   search evaluates is recorded, so the script can compare the whole path with
   scipy's rather than only the answer. The same objectives are written in
   Python there with the same constants. */
#include "../../solver/brent.h"

typedef struct {
    int which;
    double c0, c1, c2;
    double *points;
    int capacity, count;
} Objective;

static double evaluate(int which, double x, double c0, double c1, double c2) {
    switch (which) {
        case 0: return (x - c0) * (x - c0);
        case 1: return exp(x) - c0 * x;
        case 2: return x * log(x);
        case 3: return cos(x);
        case 4: { double d = x - c0; return d * d * d * d; }
        case 5: return fabs(x - c0);
        case 6: return x;
        case 7: return -x;
        case 8: return sin(c0 * x) + c1 * x * x;
        case 9: return x < c0 ? INFINITY : (x - c1) * (x - c1);
        /* a random smooth function with several minima */
        case 10: return c0 * sin(x) + c1 * cos(2 * x) + c2 * x * x;
        /* a likelihood-shaped curve over log q: n log(1 + e^x) - x, flat at
           the left end and linear at the right */
        case 11: return c0 * log1p(exp(x)) - c1 * x;
    }
    return NAN;
}

static double recorded(double x, void *context) {
    Objective *o = (Objective*)context;
    if (o->count < o->capacity) o->points[o->count] = x;
    o->count++;
    return evaluate(o->which, x, o->c0, o->c1, o->c2);
}

/* Returns the number of points written to points; result in out[0..3]:
   x, value, nfev, status. */
int c_brent(int which, double c0, double c1, double c2, double lower, double upper,
            double xatol, int max_evaluations, double *points, int capacity, double *out) {
    Objective o = { which, c0, c1, c2, points, capacity, 0 };
    BrentOptions options = { xatol, max_evaluations };
    BrentResult r = brent_bounded(recorded, &o, lower, upper, options);
    out[0] = r.x;
    out[1] = r.value;
    out[2] = r.nfev;
    out[3] = r.status;
    return o.count < capacity ? o.count : capacity;
}
