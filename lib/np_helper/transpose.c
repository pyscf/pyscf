/*
 *
 */

#include <stddef.h>
#include <complex.h>

/*
 * matrix a[n,m]
 */
void NPdtranspose(int n, int m, double *a, double *at, int blk)
{
        int ic, jc;
        size_t i, j;
        double *po, *pi;

        for (jc = 0; jc < m-blk; jc+=blk) {
                for (ic = 0; ic < n-blk; ic+=blk) {
                        for (j = jc; j < jc+blk; j++) {
                                po = at + j * n;
                                pi = a + j;
                                for (i = ic; i < ic+blk; i++) {
                                        po[i] = pi[i*m];
                                }
                        }
                }
                for (j = jc; j < jc+blk; j++) {
                        po = at + j * n;
                        pi = a + j;
                        for (i = ic; i < n; i++) {
                                po[i] = pi[i*m];
                        }
                }
        }
        for (j = jc; j < m; j++) {
                po = at + j * n;
                pi = a + j;
                for (i = 0; i < n; i++) {
                        po[i] = pi[i*m];
                }
        }
}

void NPztranspose(int n, int m, double complex *a, double complex *at, int blk)
{
        int ic, jc;
        size_t i, j;
        double complex *po, *pi;

        for (jc = 0; jc < m-blk; jc+=blk) {
                for (ic = 0; ic < n-blk; ic+=blk) {
                        for (j = jc; j < jc+blk; j++) {
                                po = at + j * n;
                                pi = a + j;
                                for (i = ic; i < ic+blk; i++) {
                                        po[i] = pi[i*m];
                                }
                        }
                }
                for (j = jc; j < jc+blk; j++) {
                        po = at + j * n;
                        pi = a + j;
                        for (i = ic; i < n; i++) {
                                po[i] = pi[i*m];
                        }
                }
        }
        for (j = jc; j < m; j++) {
                po = at + j * n;
                pi = a + j;
                for (i = 0; i < n; i++) {
                        po[i] = pi[i*m];
                }
        }
}


void NPdtranspose_021(int count, int n, int m, double *a, double *at, int blk)
{
#pragma omp parallel default(none) \
        shared(count, n, m, a, at, blk)
{
        int ic;
        size_t nm = n * m;
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                NPdtranspose(n, m, a+ic*nm, at+ic*nm, blk);
        }
}
}

void NPztranspose_021(int count, int n, int m,
                      double complex *a, double complex *at, int blk)
{
#pragma omp parallel default(none) \
        shared(count, n, m, a, at, blk)
{
        int ic;
        size_t nm = n * m;
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                NPztranspose(n, m, a+ic*nm, at+ic*nm, blk);
        }
}
}

