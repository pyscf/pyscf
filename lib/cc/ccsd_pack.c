/*
 *
 */

#include <string.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

void CCunpack_tril(int count, int n, double *tril, double *mat)
{
#pragma omp parallel default(none) \
        shared(count, n, tril, mat)
{
        int ic, i, j, ij;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
        double *pmat, *ptril;
#pragma omp for
        for (ic = 0; ic < count; ic++) {
                ptril = tril + n2 * ic;
                pmat = mat + nn * ic;
                for (ij = 0, i = 0; i < n; i++) {
                        for (j = 0; j <= i; j++, ij++) {
                                pmat[i*n+j] = ptril[ij];
                                pmat[j*n+i] = ptril[ij];
                        }
                }
        }
}
}

void CCpack_tril(int count, int n, double *tril, double *mat)
{
#pragma omp parallel default(none) \
        shared(count, n, tril, mat)
{
        int ic, i, j, ij;
        size_t nn = n * n;
        size_t n2 = n*(n+1)/2;
        double *pmat, *ptril;
#pragma omp for
        for (ic = 0; ic < count; ic++) {
                ptril = tril + n2 * ic;
                pmat = mat + nn * ic;
                for (ij = 0, i = 0; i < n; i++) {
                        for (j = 0; j <= i; j++, ij++) {
                                ptril[ij] = pmat[i*n+j];
                        }
                }
        }
}
}

/*
 * g2[p,q,r,s] = a * v1 + b * v2.transpose(0,1,3,2)
 */
void CCmake_g0132(double *g2, double *v1, double *v2, int *shape,
                  double a, double b)
{
        int i, j, k, l, kl;
        size_t d2 = shape[2] * shape[3];
        for (i = 0; i < shape[0]; i++) {
        for (j = 0; j < shape[1]; j++) {
                for (kl = 0, k = 0; k < shape[2]; k++) {
                for (l = 0; l < shape[3]; l++, kl++) {
                        g2[kl] = v1[kl] * a + v2[l*shape[2]+k] * b;
                } }
                v1 += d2;
                v2 += d2;
                g2 += d2;
        } }
}

