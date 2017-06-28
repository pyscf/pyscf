#include <stdlib.h>
#include <string.h>
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
        const size_t dimc = ldc;
        int i, j;
        if (m == 0 || n == 0) {
                return;
        } else if (k == 0) {
                for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                        c[i*dimc+j] = 0;
                } }
                return;
        }
        a += offseta;
        b += offsetb;
        c += offsetc;

        size_t stride;
        int nthread = 1;
        int di, nblk;

        if ((k/m) > 3 && (k/n) > 3) { // parallelize k

                const double D0 = 0;
                double *cpriv;
                int ij, stride_b;
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
                if (beta == 0) {
                        for (i = 0; i < n; i++) {
                                memset(c+i*dimc, 0, sizeof(double)*m);
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                for (j = 0; j < m; j++, ij++) {
                                        c[i*dimc+j] *= beta;
                                }
                        }
                }

#pragma omp parallel default(none) \
        shared(a, b, c, stride, stride_b, nthread, nblk) \
        private(i, ij, j, di, cpriv)
{
                nthread = omp_get_num_threads();
                nblk = MAX((k+nthread-1) / nthread, 1);
                cpriv = malloc(sizeof(double) * m * n);
                size_t astride = stride * nblk;
                size_t bstride = stride_b * nblk;
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, k-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &m, &n, &di,
                                       &alpha, a+astride*i, &lda,
                                       b+bstride*i, &ldb,
                                       &D0, cpriv, &m);
                        }
                }
#pragma omp critical
                if (di > 0) {
                        for (ij = 0, i = 0; i < n; i++) {
                                for (j = 0; j < m; j++, ij++) {
                                        c[i*dimc+j] += cpriv[ij];
                                }
                        }
                }
                free(cpriv);
}

        } else if (m > n+4) { // parallelize m

                if (trans_a == 'N') {
                        stride = 1;
                } else {
                        stride = lda;
                }

#pragma omp parallel default(none) \
        shared(a, b, c, stride, nthread, nblk) \
        private(i, di)
{
                nthread = omp_get_num_threads();
                nblk = MAX((m+nthread-1) / nthread, 1);
                nthread = (m+nblk-1) / nblk;
                size_t astride = stride * nblk;
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, m-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &di, &n, &k,
                                       &alpha, a+astride*i, &lda, b, &ldb,
                                       &beta, c+i*nblk, &ldc);
                        }
                }
}

        } else { // parallelize n

                if (trans_b == 'N') {
                        stride = ldb;
                } else {
                        stride = 1;
                }

#pragma omp parallel default(none) \
        shared(a, b, c, stride, nthread, nblk) \
        private(i, di)
{
                nthread = omp_get_num_threads();
                nblk = MAX((n+nthread-1) / nthread, 1);
                nthread = (n+nblk-1) / nblk;
                size_t bstride = stride * nblk;
                size_t cstride = dimc * nblk;
#pragma omp for
                for (i = 0; i < nthread; i++) {
                        di = MIN(nblk, n-i*nblk);
                        if (di > 0) {
                                dgemm_(&trans_a, &trans_b, &m, &di, &k,
                                       &alpha, a, &lda, b+bstride*i, &ldb,
                                       &beta, c+cstride*i, &ldc);
                        }
                }
}
        }
}
