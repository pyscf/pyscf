/* Copyright 2021 The PySCF Developers. All Rights Reserved.

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
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"

#define GSIZE           104
#define BLKSIZE         18

// out = einsum('ig,jg->ijg', a, b)
void PBC_djoin_NN_s1(double *outR, double *aR, double *bR,
                     int na, int nb, int ng)
{
#pragma omp parallel
{
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig;
        size_t ij;
#pragma omp for schedule(static)
        for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++) {
                        ij = i * nb + j;
#pragma GCC ivdep
                        for (ig = ig0; ig < ig1; ig++) {
                                outR[ij*ng+ig] = aR[i*ng+ig] * bR[j*ng+ig];
                        }
                } }
        } } }
}
}

// outR = einsum('ig,jg->ijg', a.conj(), b).real
void PBC_zjoinR_CN_s1(double *outR, double *aR, double *aI, double *bR, double *bI,
                      int na, int nb, int ng)
{
#pragma omp parallel
{
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig;
        size_t ij;
#pragma omp for schedule(static)
        for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++) {
                        ij = i * nb + j;
#pragma GCC ivdep
                        for (ig = ig0; ig < ig1; ig++) {
                                outR[ij*ng+ig] = aR[i*ng+ig] * bR[j*ng+ig] + aI[i*ng+ig] * bI[j*ng+ig];
                        }
                } }
        } } }
}
}

// outI = einsum('ig,jg->ijg', a.conj(), b).imag
void PBC_zjoinI_CN_s1(double *outI, double *aR, double *aI, double *bR, double *bI,
                      int na, int nb, int ng)
{
#pragma omp parallel
{
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig;
        size_t ij;
#pragma omp for schedule(static)
        for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++) {
                        ij = i * nb + j;
#pragma GCC ivdep
                        for (ig = ig0; ig < ig1; ig++) {
                                outI[ij*ng+ig] = aR[i*ng+ig] * bI[j*ng+ig] - aI[i*ng+ig] * bR[j*ng+ig];
                        }
                } }
        } } }
}
}

// outR = einsum('ig,jg->ijg', a.conj(), b).real
// outI = einsum('ig,jg->ijg', a.conj(), b).imag
void PBC_zjoin_CN_s1(double *outR, double *outI,
                     double *aR, double *aI, double *bR, double *bI,
                     int na, int nb, int ng)
{
#pragma omp parallel
{
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig;
        size_t ij;
#pragma omp for schedule(static)
        for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++) {
                        ij = i * nb + j;
#pragma GCC ivdep
                        for (ig = ig0; ig < ig1; ig++) {
                                outR[ij*ng+ig] = aR[i*ng+ig] * bR[j*ng+ig] + aI[i*ng+ig] * bI[j*ng+ig];
                                outI[ij*ng+ig] = aR[i*ng+ig] * bI[j*ng+ig] - aI[i*ng+ig] * bR[j*ng+ig];
                        }
                } }
        } } }
}
}

// outR = einsum('g,ig,jg->ijg', phase, a.conj(), b).real
// outI = einsum('g,ig,jg->ijg', phase, a.conj(), b).imag
void PBC_zjoin_fCN_s1(double *outR, double *outI, double *phaseR, double *phaseI,
                      double *aR, double *aI, double *bR, double *bI,
                      int na, int nb, int ng)
{
#pragma omp parallel
{
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig;
        size_t ij;
        double fbR[GSIZE*BLKSIZE];
        double fbI[GSIZE*BLKSIZE];
        double *pfbR, *pfbI;
#pragma omp for schedule(static)
        for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                for (j = j0; j < j1; j++) {
                        pfbR = fbR + (j-j0) * GSIZE;
                        pfbI = fbI + (j-j0) * GSIZE;
#pragma GCC ivdep
                        for (ig = ig0; ig < ig1; ig++) {
                                pfbR[ig-ig0] = phaseR[ig] * bR[j*ng+ig] - phaseI[ig] * bI[j*ng+ig];
                                pfbI[ig-ig0] = phaseR[ig] * bI[j*ng+ig] + phaseI[ig] * bR[j*ng+ig];
                        }
                }
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++) {
                        ij = i * nb + j;
                        pfbR = fbR + (j-j0) * GSIZE;
                        pfbI = fbI + (j-j0) * GSIZE;
#pragma GCC ivdep
                        for (ig = ig0; ig < ig1; ig++) {
                                outR[ij*ng+ig] = aR[i*ng+ig] * pfbR[ig-ig0] + aI[i*ng+ig] * pfbI[ig-ig0];
                                outI[ij*ng+ig] = aR[i*ng+ig] * pfbI[ig-ig0] - aI[i*ng+ig] * pfbR[ig-ig0];
                        }
                } }
        } } }
}
}

// einsum('ig,jg,kg->ijk', a, b, c)
void PBC_ddot_CNC_s1(double *outR, double *aR, double *bR, double *cR,
                     int na, int nb, int nc, int ng)
{
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1;
        int gsize = GSIZE;
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig, da, dg, dab;
        size_t nbc = nb * nc;
        double *bufR = malloc(sizeof(double) * BLKSIZE * nb * GSIZE);
        double *pbufR;
#pragma omp for schedule(static)
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
                da = i1 - i0;
                dab = da * nb;
                NPdset0(outR+i0*nbc, da * nbc);
                for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
                        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                                for (i = i0; i < i1; i++) {
                                for (j = j0; j < j1; j++) {
                                        pbufR = bufR + ((i - i0) * nb + j) * GSIZE - ig0;
#pragma GCC ivdep
                                        for (ig = ig0; ig < ig1; ig++) {
                                                pbufR[ig] = aR[i*ng+ig] * bR[j*ng+ig];
                                        }
                                } }
                        }
                        dg = ig1 - ig0;
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufR, &gsize,
                               &D1, outR+i0*nbc, &nc);
                }
        }
        free(bufR);
}
}

// outR = einsum('ig,jg,kg->ijk', a.conj(), b, c.conj()).real
// outI = einsum('ig,jg,kg->ijk', a.conj(), b, c.conj()).imag
void PBC_zdot_CNC_s1(double *outR, double *outI, double *aR, double *aI,
                     double *bR, double *bI, double *cR, double *cI,
                     int na, int nb, int nc, int ng)
{
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1;
        double ND1 = -1;
        int gsize = GSIZE;
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig, da, dg, dab;
        size_t nbc = nb * nc;
        double *bufR = malloc(sizeof(double) * BLKSIZE * nb * GSIZE * 2);
        double *bufI = bufR + BLKSIZE * nb * GSIZE;
        double *poutR, *poutI, *pbufR, *pbufI;
#pragma omp for schedule(static)
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
                da = i1 - i0;
                dab = da * nb;
                poutR = outR + i0 * nbc;
                poutI = outI + i0 * nbc;
                NPdset0(poutR, da * nbc);
                NPdset0(poutI, da * nbc);
                for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
                        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                                for (i = i0; i < i1; i++) {
                                for (j = j0; j < j1; j++) {
                                        pbufR = bufR + ((i - i0) * nb + j) * GSIZE - ig0;
                                        pbufI = bufI + ((i - i0) * nb + j) * GSIZE - ig0;
#pragma GCC ivdep
                                        for (ig = ig0; ig < ig1; ig++) {
                                                pbufR[ig] = aR[i*ng+ig] * bR[j*ng+ig] + aI[i*ng+ig] * bI[j*ng+ig];
                                                pbufI[ig] = aR[i*ng+ig] * bI[j*ng+ig] - aI[i*ng+ig] * bR[j*ng+ig];
                                        }
                                } }
                        }
                        dg = ig1 - ig0;
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufR, &gsize, &D1, outR, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cI+ig0, &ng, bufI, &gsize, &D1, outR, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufI, &gsize, &D1, outI, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &ND1, cI+ig0, &ng, bufR, &gsize, &D1, outI, &nc);
                }
        }
        free(bufR);
}
}

// outR = einsum('nig,njg,kg->nijk', a.conj(), b, c.conj()).real
// outI = einsum('nig,njg,kg->nijk', a.conj(), b, c.conj()).imag
void PBC_kzdot_CNC_s1(double *outR, double *outI,
                      double *aoR_ks, double *aoI_ks, double *cR, double *cI,
                      int *kpt_ij_idx, int na, int nb, int nc, int ng,
                      int nkptij, int nkpts)
{
        int nblocks_a = (na + BLKSIZE - 1) / BLKSIZE;
        int ntasks = nblocks_a * nkptij;
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1;
        double ND1 = -1;
        size_t nbc = nb * nc;
        size_t nabc = na * nbc;
        int gsize = GSIZE;
        int i, j, ig, i0, i1, j0, j1, ig0, ig1;
        int it, kk_idx, ki, kj, da, dg, dab;
        double *aR, *aI, *bR, *bI;
        double *bufR = malloc(sizeof(double) * BLKSIZE * nb * GSIZE * 2);
        double *bufI = bufR + BLKSIZE * nb * GSIZE;
        double *poutR, *poutI, *pbufR, *pbufI;
#pragma omp for schedule(static)
        for (it = 0; it < ntasks; it++) {
                kk_idx = it / nblocks_a;
                ki = kpt_ij_idx[kk_idx] / nkpts;
                kj = kpt_ij_idx[kk_idx] % nkpts;
                aR = aoR_ks + ki * nb * ng;
                aI = aoI_ks + ki * nb * ng;
                bR = aoR_ks + kj * nb * ng;
                bI = aoI_ks + kj * nb * ng;
                i0 = it % nblocks_a * BLKSIZE;
                i1 = MIN(i0+BLKSIZE, na);
                da = i1 - i0;
                dab = da * nb;
                poutR = outR + kk_idx * nabc + i0 * nbc;
                poutI = outI + kk_idx * nabc + i0 * nbc;
                NPdset0(poutR, da * nbc);
                NPdset0(poutI, da * nbc);
                for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
                        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                                for (i = i0; i < i1; i++) {
                                for (j = j0; j < j1; j++) {
                                        pbufR = bufR + ((i - i0) * nb + j) * GSIZE - ig0;
                                        pbufI = bufI + ((i - i0) * nb + j) * GSIZE - ig0;
#pragma GCC ivdep
                                        for (ig = ig0; ig < ig1; ig++) {
                                                pbufR[ig] = aR[i*ng+ig] * bR[j*ng+ig] + aI[i*ng+ig] * bI[j*ng+ig];
                                                pbufI[ig] = aR[i*ng+ig] * bI[j*ng+ig] - aI[i*ng+ig] * bR[j*ng+ig];
                                        }
                                } }
                        }
                        dg = ig1 - ig0;
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufR, &gsize, &D1, poutR, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufI, &gsize, &D1, poutI, &nc);
                        if (cI != NULL) {
                                dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                                       &D1, cI+ig0, &ng, bufI, &gsize, &D1, poutR, &nc);
                                dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                                       &ND1, cI+ig0, &ng, bufR, &gsize, &D1, poutI, &nc);
                        }
                }
        }
        free(bufR);
}
}

// outR = einsum('ig,jg,kg->ijk', a.conj(), b, c).real
// outI = einsum('ig,jg,kg->ijk', a.conj(), b, c).imag
void PBC_zdot_CNN_s1(double *outR, double *outI, double *aR, double *aI,
                     double *bR, double *bI, double *cR, double *cI,
                     int na, int nb, int nc, int ng)
{
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1;
        double ND1 = -1;
        int gsize = GSIZE;
        int i0, i1, j0, j1, ig0, ig1;
        int i, j, ig, da, dg, dab;
        size_t nbc = nb * nc;
        double *bufR = malloc(sizeof(double) * BLKSIZE * nb * GSIZE * 2);
        double *bufI = bufR + BLKSIZE * nb * GSIZE;
        double *poutR, *poutI, *pbufR, *pbufI;
#pragma omp for schedule(static)
        for (i0 = 0; i0 < na; i0 += BLKSIZE) { i1 = MIN(i0+BLKSIZE, na);
                da = i1 - i0;
                dab = da * nb;
                poutR = outR + i0 * nbc;
                poutI = outI + i0 * nbc;
                NPdset0(poutR, da * nbc);
                NPdset0(poutI, da * nbc);
                for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
                        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                                for (i = i0; i < i1; i++) {
                                for (j = j0; j < j1; j++) {
                                        pbufR = bufR + ((i - i0) * nb + j) * GSIZE - ig0;
                                        pbufI = bufI + ((i - i0) * nb + j) * GSIZE - ig0;
#pragma GCC ivdep
                                        for (ig = ig0; ig < ig1; ig++) {
                                                pbufR[ig] = aR[i*ng+ig] * bR[j*ng+ig] + aI[i*ng+ig] * bI[j*ng+ig];
                                                pbufI[ig] = aR[i*ng+ig] * bI[j*ng+ig] - aI[i*ng+ig] * bR[j*ng+ig];
                                        }
                                } }
                        }
                        dg = ig1 - ig0;
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufR, &gsize, &D1, outR, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &ND1, cI+ig0, &ng, bufI, &gsize, &D1, outR, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufI, &gsize, &D1, outI, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cI+ig0, &ng, bufR, &gsize, &D1, outI, &nc);
                }
        }
        free(bufR);
}
}

// outR = einsum('nig,njg,kg->nijk', a.conj(), b, c).real
// outI = einsum('nig,njg,kg->nijk', a.conj(), b, c).imag
void PBC_kzdot_CNN_s1(double *outR, double *outI,
                      double *aoR_ks, double *aoI_ks, double *cR, double *cI,
                      int *kpt_ij_idx, int na, int nb, int nc, int ng,
                      int nkptij, int nkpts)
{
        int nblocks_a = (na + BLKSIZE - 1) / BLKSIZE;
        int ntasks = nblocks_a * nkptij;
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1;
        double ND1 = -1;
        size_t nbc = nb * nc;
        size_t nabc = na * nbc;
        int gsize = GSIZE;
        int i, j, ig, i0, i1, j0, j1, ig0, ig1;
        int it, kk_idx, ki, kj, da, dg, dab;
        double *aR, *aI, *bR, *bI;
        double *bufR = malloc(sizeof(double) * BLKSIZE * nb * GSIZE * 2);
        double *bufI = bufR + BLKSIZE * nb * GSIZE;
        double *poutR, *poutI, *pbufR, *pbufI;
#pragma omp for schedule(static)
        for (it = 0; it < ntasks; it++) {
                kk_idx = it / nblocks_a;
                ki = kpt_ij_idx[kk_idx] / nkpts;
                kj = kpt_ij_idx[kk_idx] % nkpts;
                aR = aoR_ks + ki * nb * ng;
                aI = aoI_ks + ki * nb * ng;
                bR = aoR_ks + kj * nb * ng;
                bI = aoI_ks + kj * nb * ng;
                i0 = it % nblocks_a * BLKSIZE;
                i1 = MIN(i0+BLKSIZE, na);
                da = i1 - i0;
                dab = da * nb;
                poutR = outR + kk_idx * nabc + i0 * nbc;
                poutI = outI + kk_idx * nabc + i0 * nbc;
                NPdset0(poutR, da * nbc);
                NPdset0(poutI, da * nbc);
                for (ig0 = 0; ig0 < ng; ig0 += GSIZE) { ig1 = MIN(ig0+GSIZE, ng);
                        for (j0 = 0; j0 < nb; j0 += BLKSIZE) { j1 = MIN(j0+BLKSIZE, nb);
                                for (i = i0; i < i1; i++) {
                                for (j = j0; j < j1; j++) {
                                        pbufR = bufR + ((i - i0) * nb + j) * GSIZE - ig0;
                                        pbufI = bufI + ((i - i0) * nb + j) * GSIZE - ig0;
#pragma GCC ivdep
                                        for (ig = ig0; ig < ig1; ig++) {
                                                pbufR[ig] = aR[i*ng+ig] * bR[j*ng+ig] + aI[i*ng+ig] * bI[j*ng+ig];
                                                pbufI[ig] = aR[i*ng+ig] * bI[j*ng+ig] - aI[i*ng+ig] * bR[j*ng+ig];
                                        }
                                } }
                        }
                        dg = ig1 - ig0;
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufR, &gsize, &D1, poutR, &nc);
                        dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                               &D1, cR+ig0, &ng, bufI, &gsize, &D1, poutI, &nc);
                        if (cI != NULL) {
                                dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                                       &ND1, cI+ig0, &ng, bufI, &gsize, &D1, poutR, &nc);
                                dgemm_(&TRANS_T, &TRANS_N, &nc, &dab, &dg,
                                       &D1, cI+ig0, &ng, bufR, &gsize, &D1, poutI, &nc);
                        }
                }
        }
        free(bufR);
}
}
