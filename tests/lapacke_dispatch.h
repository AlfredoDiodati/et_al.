#pragma once
#include <lapacke.h>

/* MLAPACK is the LAPACKE half of linalg/mat.h's precision switch, and lives
   here because the comparison arms are the only code that calls LAPACKE.
   It mirrors MBLAS: MLAPACK(potrf) is LAPACKE_spotrf under the default
   float build and LAPACKE_dpotrf under -DMAT_DOUBLE, so a test can time or
   cross-check a kernel against the routine it replaced under either
   precision without spelling the prefix at every call site.

   Include this instead of <lapacke.h> directly. It is reachable only from
   tests/, so the library keeps linking against OpenBLAS alone; the targets
   that compile these files are the ones that add -llapacke, and they are
   listed on their own in the Makefile. */
#ifdef MAT_DOUBLE
#define MLAPACK(fn) LAPACKE_d##fn
#else
#define MLAPACK(fn) LAPACKE_s##fn
#endif
