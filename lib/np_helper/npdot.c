#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))

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

        int stride, nthread, nblk;
        int i, di;

        if ((k/m) > 3 && (k/n) > 3) { // parallelize k

                const double D0 = 0;
                double *cpriv, *pc;
                int ij, j, stride_b;
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
                                memset(c+i*ldc, 0, sizeof(double)*m);
                        }
                } else {
                        for (i = 0; i < n; i++) {
                                pc = c + i * ldc;
                                for (j = 0; j < m; j++, ij++) {
                                        pc[j] *= beta;
                                }
                        }
                }

#pragma omp parallel default(none) \
        shared(a, b, c, stride, stride_b, nthread, nblk) \
        private(i, ij, j, di, cpriv, pc)
                {
                        nthread = omp_get_num_threads();
                        nblk = (int)(k/nthread) + 1;
                        cpriv = malloc(sizeof(double) * m * n);

#pragma omp for nowait schedule(static)
                        for (i = 0; i < nthread; i++) {
                                di = MIN(nblk, k-i*nblk);
                                dgemm_(&trans_a, &trans_b, &m, &n, &di,
                                       &alpha, a+stride*i*nblk, &lda,
                                       b+stride_b*i*nblk, &ldb,
                                       &D0, cpriv, &m);
                        }
#pragma omp critical
                {
                        for (ij = 0, i = 0; i < n; i++) {
                                pc = c + i * ldc;
                                for (j = 0; j < m; j++, ij++) {
                                        pc[j] += cpriv[ij];
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
                        nblk = (int)(m/nthread) + 1;

#pragma omp for nowait schedule(static)
                        for (i = 0; i < nthread; i++) {
                                di = MIN(nblk, m-i*nblk);
                                dgemm_(&trans_a, &trans_b, &di, &n, &k,
                                       &alpha, a+stride*i*nblk, &lda, b, &ldb,
                                       &beta, c+i*nblk, &ldc);
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
                        nblk = (int)(n/nthread) + 1;

#pragma omp for nowait schedule(static)
                        for (i = 0; i < nthread; i++) {
                                di = MIN(nblk, n-i*nblk);
                                dgemm_(&trans_a, &trans_b, &m, &di, &k,
                                       &alpha, a, &lda, b+stride*i*nblk, &ldb,
                                       &beta, c+ldc*i*nblk, &ldc);
                        }
                }
        }
}
