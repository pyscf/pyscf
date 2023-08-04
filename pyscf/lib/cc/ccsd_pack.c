/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

/*
 * a * v1 + b * v2.transpose(0,2,1,3)
 */
void CCmake_0213(double *out, double *v1, double *v2, int count, int m,
                 double a, double b)
{
#pragma omp parallel default(none) \
        shared(count, m, out, v1, v2, a, b)
{
        int i, j, k, l, n;
        size_t d2 = m * m;
        size_t d1 = m * m * m;
        double *pv1, *pv2, *pout;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                for (n = 0, j = 0; j < m; j++) {
                for (k = 0; k < m; k++) {
                        pout = out + d1*i + d2*j + m*k;
                        pv1  = v1  + d1*i + d2*j + m*k;
                        pv2  = v2  + d1*i + d2*k + m*j;
                        for (l = 0; l < m; l++, n++) {
                                pout[l] = pv1[l] * a + pv2[l] * b;
                        }
        } } }
}
}

/*
 * out = v1 + v2.transpose(0,2,1)
 */
void CCsum021(double *out, double *v1, double *v2, int count, int m)
{
#pragma omp parallel default(none) \
        shared(count, m, out, v1, v2)
{
        int i, j, k, n;
        size_t mm = m * m;
        double *pout, *pv1, *pv2;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                pout = out + mm * i;
                pv1  = v1  + mm * i;
                pv2  = v2  + mm * i;
                for (n = 0, j = 0; j < m; j++) {
                for (k = 0; k < m; k++, n++) {
                        pout[n] = pv1[n] + pv2[k*m+j];
                } }
        }
}
}

/*
 * g2 = a * v1 + b * v2.transpose(0,2,1)
 */
void CCmake_021(double *out, double *v1, double *v2, int count, int m,
                double a, double b)
{
        if (a == 1 && b == 1) {
                return CCsum021(out, v1, v2, count, m);
        }

#pragma omp parallel default(none) \
        shared(count, m, out, v1, v2, a, b)
{
        int i, j, k, n;
        size_t mm = m * m;
        double *pout, *pv1, *pv2;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                pout = out + mm * i;
                pv1  = v1  + mm * i;
                pv2  = v2  + mm * i;
                for (n = 0, j = 0; j < m; j++) {
                for (k = 0; k < m; k++, n++) {
                        pout[n] = pv1[n] * a + pv2[k*m+j] * b;
                } }
        }
}
}

/*
 * if matrix B is symmetric for the contraction A_ij B_ij,
 * Tr(AB) ~ A_ii B_ii + (A_ij + A_ji) B_ij where i > j
 * This function extract the A_ii and the lower triangluar part of A_ij + A_ji
 */
void CCprecontract(double *out, double *in, int count, int m, double diagfac)
{
#pragma omp parallel default(none) \
        shared(count, m, in, out, diagfac)
{
        int i, j, k, n;
        size_t mm = m * m;
        size_t m2 = m * (m+1) / 2;
        double *pout, *pin;
#pragma omp for schedule (static)
        for (i = 0; i < count; i++) {
                pout = out + m2 * i;
                pin  = in  + mm * i;
                for (n = 0, j = 0; j < m; j++) {
                        for (k = 0; k < j; k++, n++) {
                                pout[n] = pin[j*m+k] + pin[k*m+j];
                        }
                        pout[n] = pin[j*m+j] * diagfac;
                        n++;
                }
        }
}
}

/*
 * if i1 == j1:
 *     eri = unpack_tril(eri, axis=0)
 * unpack_tril(eri).reshape(i1-i0,j1-j0,nao,nao).transpose(0,2,1,3)
 */
void CCload_eri(double *out, double *eri, int *orbs_slice, int nao)
{
        int i0 = orbs_slice[0];
        int i1 = orbs_slice[1];
        int j0 = orbs_slice[2];
        int j1 = orbs_slice[3];
        size_t ni = i1 - i0;
        size_t nj = j1 - j0;
        size_t nn = nj * nao;
        size_t nao_pair = nao * (nao + 1) / 2;

#pragma omp parallel default(none) \
        shared(out, eri, i1, j1, ni, nj, nn, nao, nao_pair)
{
        int i, j, k, l, ij;
        double *pout;
        double *buf = malloc(sizeof(double) * nao*nao);
#pragma omp for schedule (static)
        for (ij = 0; ij < ni*nj; ij++) {
                i = ij / nj;
                j = ij % nj;
                NPdunpack_tril(nao, eri+ij*nao_pair, buf, 1);
                pout = out + (i*nn+j)*nao;
                for (k = 0; k < nao; k++) {
                for (l = 0; l < nao; l++) {
                        pout[k*nn+l] = buf[k*nao+l];
                } }
        }
        free(buf);
}
}

/*
 * eri put virtual orbital first
 * [ v         ]
 * [ v .       ]
 * [ v . .     ]
 * [ o . . .   ]
 * [ o . . . . ]
 */
void CCsd_sort_inplace(double *eri, int nocc, int nvir, int count)
{
#pragma omp parallel default(none) \
        shared(eri, nocc, nvir, count)
{
        int ic, i, j, ij;
        size_t nmo = nocc + nvir;
        size_t nmo_pair = nmo * (nmo+1) / 2;
        size_t nocc_pair = nocc * (nocc+1) /2;
        size_t nvir_pair = nvir * (nvir+1) /2;
        double *peri, *pout;
        double *buf = malloc(sizeof(double) * nocc*nvir);
#pragma omp for schedule (static)
        for (ic = 0; ic < count; ic++) {
                peri = eri + ic*nmo_pair + nvir_pair;
                for (i = 0; i < nocc; i++, peri+=nvir+i) {
                        for (j = 0; j < nvir; j++) {
                                buf[i*nvir+j] = peri[j];
                        }
                }
                pout = eri + ic*nmo_pair + nvir_pair;
                peri = eri + ic*nmo_pair + nvir_pair + nvir;
                for (ij = 0, i = 0; i < nocc; i++, peri+=nvir+i) {
                        for (j = 0; j <= i; j++, ij++) {
                                pout[ij] = peri[j];
                        }
                }
                pout = eri + ic*nmo_pair + nvir_pair + nocc_pair;
                NPdcopy(pout, buf, nocc*nvir);
        }
        free(buf);
}
}

