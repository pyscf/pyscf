#ifndef HAVE_DEFINED_GRID_UTILS_H
#define HAVE_DEFINED_GRID_UTILS_H

extern void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc);
#endif
