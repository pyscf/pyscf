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
 * g2[p,q,r,s] = a * v1 + b * v2.transpose(0,2,1,3)
 */
void CCmake_g0213(double *g2, double *v1, double *v2, int *shape,
                  double a, double b)
{
        int i, j, k, l;
        size_t d1 = shape[1] * shape[2] * shape[3];
        size_t d2 = shape[2] * shape[3];
        size_t dv2 = shape[1] * shape[3];
        size_t d3 = shape[3];
        double *pv1, *pv2, *pg2;
        for (i = 0; i < shape[0]; i++) {
                for (j = 0; j < shape[1]; j++) {
                        pg2 = g2 + d2 * j;
                        pv1 = v1 + d2 * j;
                        pv2 = v2 + d3 * j;
                        for (k = 0; k < shape[2]; k++) {
                                for (l = 0; l < shape[3]; l++) {
                                        pg2[l] = pv1[l] * a + pv2[l] * b;
                                }
                                pg2 += d3;
                                pv1 += d3;
                                pv2 += dv2;
                        }
                }
                v1 += d1;
                v2 += d1;
                g2 += d1;
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

/*
 * tau[p,q,r,s] += t1a[p,s] * t1b[q,r]
 */
void CCset_tau(double *tau, double *t1a, int *shapea, double *t1b, int *shapeb)
{
        const int INC1 = 1;
        const double D1 = 1;
        int i;
        int sizeb = shapeb[0] * shapeb[1];
        size_t d1 = sizeb * shapea[1];
        for (i = 0; i < shapea[0]; i++) {
                dger_(shapea+1, &sizeb, &D1, t1a+i*shapea[1], &INC1,
                      t1b, &INC1, tau+i*d1, shapea+1);
        }
}

