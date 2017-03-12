#include <stdlib.h>
#include <string.h>
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))

/*
 * numpy.dot may call unoptimized blas
 */
void NPdgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double *a, double *b, double *c,
             const double alpha, const double beta)
{
        a += offseta;
        b += offsetb;
        c += offsetc;

        if ((k/m) > 3 && (k/n) > 3) { // parallelize k

                int i, j;
                if (beta == 0) {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*ldc+j] = 0;
                                }
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*ldc+j] *= beta;
                                }
                        }
                }

#pragma omp parallel default(none) shared(a, b, c) \
        private(i, j)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((k+nthread-1) / nthread, 1);
                double D0 = 0;
                double *cpriv = malloc(sizeof(double) * m * n);
                int ij, di;
                size_t stride, stride_b;
                if (trans_a == 'N') {
                        stride = lda;
                } else {
                        stride = 1;
                }
                if (trans_b == 'N') {
                        stride_b = 1;
                } else {
                        stride_b = ldb;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, k-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &m, &n, &di,
                                       &alpha, a+stride*i*nblk, &lda,
                                       b+stride_b*i*nblk, &ldb,
                                       &D0, cpriv, &m);
                        }
                }
#pragma omp critical
                if (di > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                                for (j = 0; j < m; j++, ij++) {
                                        c[i*ldc+j] += cpriv[ij];
                                }
                        }
                }
                free(cpriv);
}

        } else if (m > n+4) { // parallelize m

#pragma omp parallel default(none) shared(a, b, c)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((m+nthread-1) / nthread, 1);
                nthread = (m+nblk-1) / nblk;
                int i, di;
                size_t stride;
                if (trans_a == 'N') {
                        stride = 1;
                } else {
                        stride = lda;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, m-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &di, &n, &k,
                                       &alpha, a+stride*i*nblk, &lda, b, &ldb,
                                       &beta, c+i*nblk, &ldc);
                        }
                }
}

        } else { // parallelize n

#pragma omp parallel default(none) shared(a, b, c)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((n+nthread-1) / nthread, 1);
                nthread = (n+nblk-1) / nblk;
                int i, di;
                size_t stride;
                if (trans_b == 'N') {
                        stride = ldb;
                } else {
                        stride = 1;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, n-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &m, &di, &k,
                                       &alpha, a, &lda, b+stride*i*nblk, &ldb,
                                       &beta, c+ldc*i*nblk, &ldc);
                        }
                }
}
        }
}


void NPzgemm(const char trans_a, const char trans_b,
             const int m, const int n, const int k,
             const int lda, const int ldb, const int ldc,
             const int offseta, const int offsetb, const int offsetc,
             double complex *a, double complex *b, double complex *c,
             const double complex *alpha, const double complex *beta)
{
        a += offseta;
        b += offsetb;
        c += offsetc;

        if ((k/m) > 3 && (k/n) > 3) { // parallelize k

                int i, j;
                if (creal(*beta) == 0 && cimag(*beta) == 0) {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*ldc+j] = 0;
                                }
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++) {
                                        c[i*ldc+j] *= beta[0];
                                }
                        }
                }

#pragma omp parallel default(none) shared(a, b, c, alpha) \
        private(i, j)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((k+nthread-1) / nthread, 1);
                double complex Z0 = 0;
                double complex *cpriv = malloc(sizeof(double complex) * m * n);
                int ij, di;
                size_t stride, stride_b;
                if (trans_a == 'N') {
                        stride = lda;
                } else {
                        stride = 1;
                }
                if (trans_b == 'N') {
                        stride_b = 1;
                } else {
                        stride_b = ldb;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, k-i*nblk);
                        if (di > 0) {
                                zgemm_(&trans_a, &trans_b, &m, &n, &di,
                                       alpha, a+stride*i*nblk, &lda,
                                       b+stride_b*i*nblk, &ldb,
                                       &Z0, cpriv, &m);
                        }
                }
#pragma omp critical
                if (di > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                                for (j = 0; j < m; j++, ij++) {
                                        c[i*ldc+j] += cpriv[ij];
                                }
                        }
                }
                free(cpriv);
}

        } else if (m > n+4) { // parallelize m

#pragma omp parallel default(none) shared(a, b, c, alpha, beta)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((m+nthread-1) / nthread, 1);
                nthread = (m+nblk-1) / nblk;
                int i, di;
                size_t stride;
                if (trans_a == 'N') {
                        stride = 1;
                } else {
                        stride = lda;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, m-i*nblk);
                        if (di > 0) {
                                zgemm_(&trans_a, &trans_b, &di, &n, &k,
                                       alpha, a+stride*i*nblk, &lda, b, &ldb,
                                       beta, c+i*nblk, &ldc);
                        }
                }
}

        } else { // parallelize n

#pragma omp parallel default(none) shared(a, b, c, alpha, beta)
{
                int nthread = omp_get_num_threads();
                int nblk = MAX((n+nthread-1) / nthread, 1);
                nthread = (n+nblk-1) / nblk;
                int i, di;
                size_t stride;
                if (trans_b == 'N') {
                        stride = ldb;
                } else {
                        stride = 1;
                }
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, n-i*nblk);
                        if (di > 0) {
                                zgemm_(&trans_a, &trans_b, &m, &di, &k,
                                       alpha, a, &lda, b+stride*i*nblk, &ldb,
                                       beta, c+ldc*i*nblk, &ldc);
                        }
                }
}
        }
}
