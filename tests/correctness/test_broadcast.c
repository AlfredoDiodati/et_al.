#include "../../dist/broadcast.h"
#include <stdio.h>

int main(void) {
    puts("dim resolution");
    assert(dist_bcast_dim(3, 3) == 3);
    assert(dist_bcast_dim(1, 5) == 5);
    assert(dist_bcast_dim(5, 1) == 5);
    assert(dist_bcast_dim(1, 1) == 1);

    puts("broadcast reads");
    /* scalar: reads (0,0) at every broadcast position */
    Mat s = mat_lit(1, 1, 7.0f);
    assert(dist_bcast_at(s, 0, 0) == 7.0f);
    assert(dist_bcast_at(s, 4, 9) == 7.0f);
    /* row vector: row index collapses, column index passes through */
    Mat row = mat_lit(1, 3, 1.0f, 2.0f, 3.0f);
    assert(dist_bcast_at(row, 5, 1) == 2.0f);
    /* column vector: the mirror case */
    Mat col = mat_lit(3, 1, 4.0f, 5.0f, 6.0f);
    assert(dist_bcast_at(col, 2, 7) == 6.0f);
    /* full matrix: both indices pass through */
    Mat full = mat_lit(2, 2, 1.0f, 2.0f, 3.0f, 4.0f);
    assert(dist_bcast_at(full, 1, 0) == 3.0f);

    /* strided view: a 1x2 row sliced from a wider parent (stride != c)
       must still read the sliced values, not the parent's layout */
    Mat parent = mat_lit(2, 3, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
    Mat v = mat_slice(parent, 0, 1, 1, 3);
    assert(v.stride != v.c);
    assert(dist_bcast_at(v, 8, 1) == 3.0f);

    mat_free(s); mat_free(row); mat_free(col); mat_free(full); mat_free(parent);
    puts("test_broadcast: all passed");
    return 0;
}
