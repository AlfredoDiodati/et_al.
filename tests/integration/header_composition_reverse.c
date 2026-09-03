/*
The same headers as header_composition.c, included in the opposite order and
compiled as a separate translation unit that is linked with it.

Two things this catches that one unit cannot. A header that compiles only
because another one came first passes in one order and fails in the other, and
a symbol with external linkage defined in two headers is a duplicate the linker
rejects only when two units both pull it in. Everything about why this exists
is written in header_composition.c; this file is the other half of it and
carries no checks of its own beyond compiling and linking.
*/

#include "../../sd/score_driven_location.h"
#include "../../sd/qvarma.h"
#include "../../nn/mlp.h"
#include "../../cluster/cluster.h"
#include "../../frame/join.h"
#include "../../frame/sql.h"
#include "../../frame/rdata.h"
#include "../../frame/npz.h"
#include "../../frame/npy.h"
#include "../../frame/txt.h"
#include "../../frame/csv.h"
#include "../../frame/frame.h"
#include "../../solver/lbfgs.h"
#include "../../solver/adam.h"
#include "../../solver/optimizer.h"
#include "../../dist/mv/matgauss.h"
#include "../../dist/mv/student.h"
#include "../../dist/mv/gauss.h"
#include "../../dist/student.h"
#include "../../dist/gauss.h"
#include "../../dist/broadcast.h"
#include "../../qlr_test.h"
#include "../../cointegration.h"
#include "../../unit_root.h"
#include "../../mcs.h"
#include "../../frame/gzip.h"
#include "../../json.h"
#include "../../stats.h"
#include "../../random.h"
#include "../../special.h"
#include "../../ad.h"
#include "../../linalg/solver.h"
#include "../../linalg/decomp.h"
#include "../../linalg/factor.h"
#include "../../linalg/mat.h"

/* Called from header_composition.c's main so the linker cannot drop this unit,
   which is the entire point of it being a separate one. Returns a problem
   count for the same reason every check there does. */
int reverse_order_unit_compiles(void);

int reverse_order_unit_compiles(void) {
    int problems = 0;
    Mat m = mat_eye(3);
    if (m.r != 3 || MABS(AT(m, 1, 1) - 1.f) > 1e-4f) problems++;
    mat_free(m);
    return problems;
}
