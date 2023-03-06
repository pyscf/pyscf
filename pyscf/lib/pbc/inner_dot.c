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
#include <stdint.h>
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

#define BLEN    24
// vk += np.einsum('ijg,jk,lkg,g->il', pqG, dm, pqG.conj(), coulG)
// vk += np.einsum('ijg,li,lkg,g->kj', pqG, dm, pqG.conj(), coulG)
void PBC_kcontract(double *vkR, double *vkI, double *dmR, double *dmI,
                   double *pqGR, double *pqGI, double *coulG,
                   int *ki_idx, int *kj_idx, int8_t *k_to_compute,
                   int swap_2e, int n_dm, int nao, int ngrids, int nkpts, int nkptj)
{
        size_t nao2 = nao * nao;
        size_t size_vk = n_dm * nkpts * nao2;
        double *vtmpR = calloc(sizeof(double), size_vk*2);
        double *vtmpI = vtmpR + size_vk;
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1.;
        double D0 = 0.;
        double N1 = -1.;
        size_t Naog = nao * ngrids;
        double *pLqR = malloc(sizeof(double) * BLEN * nao2 * 4);
        double *pLqI = pLqR + BLEN * nao2;
        double *bufR = pLqI + BLEN * nao2;
        double *bufI = bufR + BLEN * nao2;
        double *outR, *outI, *inR, *inI;
        double *dR, *dI, *vR, *vI;
        int k, i, j, ig, ki, kj, g0, mg, nm, i_dm;
        double c;
#pragma omp for schedule(dynamic)
        for (k = 0; k < nkptj; k++) {
                ki = ki_idx[k];
                kj = kj_idx[k];
                if (!(k_to_compute[ki] || (swap_2e && k_to_compute[kj]))) {
                        continue;
                }

                for (g0 = 0; g0 < ngrids; g0 += BLEN) {
                        mg = MIN(BLEN, ngrids-g0);
                        nm = mg * nao;
                        // pLqR[:] = pqGR[kj].transpose(0,2,1)
                        // pLqI[:] = pqGI[kj].transpose(0,2,1)
                        for (i = 0; i < nao; i++) {
                                outR = pLqR + i * nm;
                                outI = pLqI + i * nm;
                                inR = pqGR + (kj * nao + i) * Naog + g0;
                                inI = pqGI + (kj * nao + i) * Naog + g0;
                                for (j = 0; j < nao; j++) {
                                for (ig = 0; ig < mg; ig++) {
                                        outR[ig*nao+j] = inR[j*ngrids+ig];
                                        outI[ig*nao+j] = inI[j*ngrids+ig];
                                } }
                        }
                        for (i_dm = 0; i_dm < n_dm; i_dm++) {
                                dR = dmR + i_dm * nkpts * nao2;
                                dI = dmI + i_dm * nkpts * nao2;
                                vR = vkR + i_dm * nkpts * nao2;
                                vI = vkI + i_dm * nkpts * nao2;

                                if (k_to_compute[ki]) {
// vk += np.einsum('igj,jk,lgk,g->il', pqG, dm, pqG.conj(), coulG)
dgemm_(&TRANS_N, &TRANS_N, &nao, &nm, &nao, &D1, dR+kj*nao2, &nao, pLqR, &nao, &D0, bufR, &nao);
dgemm_(&TRANS_N, &TRANS_N, &nao, &nm, &nao, &N1, dI+kj*nao2, &nao, pLqI, &nao, &D1, bufR, &nao);
dgemm_(&TRANS_N, &TRANS_N, &nao, &nm, &nao, &D1, dI+kj*nao2, &nao, pLqR, &nao, &D0, bufI, &nao);
dgemm_(&TRANS_N, &TRANS_N, &nao, &nm, &nao, &D1, dR+kj*nao2, &nao, pLqI, &nao, &D1, bufI, &nao);
for (i = 0; i < nao; i++) {
        outR = bufR + i * nm;
        outI = bufI + i * nm;
        for (ig = 0; ig < mg; ig++) {
                c = coulG[g0+ig];
                for (j = 0; j < nao; j++) {
                        outR[ig*nao+j] *= c;
                        outI[ig*nao+j] *= c;
                }
        }
}
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nm, &D1, pLqR, &nm, bufR, &nm, &D1, vR+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nm, &D1, pLqI, &nm, bufI, &nm, &D1, vR+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nm, &N1, pLqI, &nm, bufR, &nm, &D1, vI+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nm, &D1, pLqR, &nm, bufI, &nm, &D1, vI+ki*nao2, &nao);
                        }

                                if (swap_2e && k_to_compute[kj]) {
                                        vR = vtmpR + i_dm * nkpts * nao2;
                                        vI = vtmpI + i_dm * nkpts * nao2;
// vk += np.einsum('igj,li,lgk,g->kj', pqG, dm, pqG.conj(), coulG)
dgemm_(&TRANS_N, &TRANS_N, &nm, &nao, &nao, &D1, pLqR, &nm, dR+ki*nao2, &nao, &D0, bufR, &nm);
dgemm_(&TRANS_N, &TRANS_N, &nm, &nao, &nao, &N1, pLqI, &nm, dI+ki*nao2, &nao, &D1, bufR, &nm);
dgemm_(&TRANS_N, &TRANS_N, &nm, &nao, &nao, &D1, pLqI, &nm, dR+ki*nao2, &nao, &D0, bufI, &nm);
dgemm_(&TRANS_N, &TRANS_N, &nm, &nao, &nao, &D1, pLqR, &nm, dI+ki*nao2, &nao, &D1, bufI, &nm);
for (i = 0; i < nao; i++) {
        outR = bufR + i * nm;
        outI = bufI + i * nm;
        for (ig = 0; ig < mg; ig++) {
                c = coulG[g0+ig];
                for (j = 0; j < nao; j++) {
                        outR[ig*nao+j] *= c;
                        outI[ig*nao+j] *= c;
                }
        }
}
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &nm, &D1, bufR, &nao, pLqR, &nao, &D1, vR+kj*nao2, &nao);
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &nm, &D1, bufI, &nao, pLqI, &nao, &D1, vR+kj*nao2, &nao);
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &nm, &D1, bufI, &nao, pLqR, &nao, &D1, vI+kj*nao2, &nao);
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &nm, &N1, bufR, &nao, pLqI, &nao, &D1, vI+kj*nao2, &nao);
                                }
                        }
                }
        }
        free(pLqR);
#pragma omp barrier
#pragma omp for schedule(static)
        for (i = 0; i < size_vk; i++) {
                vkR[i] += vtmpR[i];
                vkI[i] += vtmpI[i];
        }
}
        free(vtmpR);
}

#define BLEN    24
// vk += np.einsum('ijg,nj,nk,lkg,g->il', pqG, mo, mo.conj(), pqG.conj(), coulG)
// vk += np.einsum('ijg,nl,ni,lkg,g->kj', pqG, mo, mo.conj(), pqG.conj(), coulG)
void PBC_kcontract_dmf(double *vkR, double *vkI, double *moR, double *moI,
                       double *pqGR, double *pqGI, double *coulG,
                       int *ki_idx, int *kj_idx, int8_t *k_to_compute, int swap_2e,
                       int n_dm, int nao, int nocc, int ngrids, int nkpts, int nkptj)
{
        size_t nao2 = nao * nao;
        size_t naoo = nao * nocc;
        size_t size_vk = n_dm * nkpts * nao2;
        double *vtmpR = calloc(sizeof(double), size_vk*2);
        double *vtmpI = vtmpR + size_vk;
#pragma omp parallel
{
        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1.;
        double D0 = 0.;
        double N1 = -1.;
        size_t Naog = nao * ngrids;
        double *pLqR = malloc(sizeof(double) * BLEN * (nao2*2+naoo*4));
        double *pLqI = pLqR + BLEN * nao2;
        double *bufR = pLqI + BLEN * nao2;
        double *bufI = bufR + BLEN * naoo;
        double *buf1R = bufI + BLEN * naoo;
        double *buf1I = buf1R + BLEN * naoo;
        double *outR, *outI, *inR, *inI;
        double *mR, *mI, *vR, *vI;
        int k, i, j, ig, ki, kj, g0, mg, nm, go, i_dm, ptr;
        double c;
#pragma omp for schedule(dynamic)
        for (k = 0; k < nkptj; k++) {
                ki = ki_idx[k];
                kj = kj_idx[k];
                if (!(k_to_compute[ki] || (swap_2e && k_to_compute[kj]))) {
                        continue;
                }

                for (g0 = 0; g0 < ngrids; g0 += BLEN) {
                        mg = MIN(BLEN, ngrids-g0);
                        nm = mg * nao;
                        go = mg * nocc;
                        // pLqR[:] = pqGR[kj].transpose(0,2,1)
                        // pLqI[:] = pqGI[kj].transpose(0,2,1)
                        for (i = 0; i < nao; i++) {
                                outR = pLqR + i * nm;
                                outI = pLqI + i * nm;
                                inR = pqGR + (kj * nao + i) * Naog + g0;
                                inI = pqGI + (kj * nao + i) * Naog + g0;
                                for (j = 0; j < nao; j++) {
                                for (ig = 0; ig < mg; ig++) {
                                        outR[ig*nao+j] = inR[j*ngrids+ig];
                                        outI[ig*nao+j] = inI[j*ngrids+ig];
                                } }
                        }
                        for (i_dm = 0; i_dm < n_dm; i_dm++) {
                                mR = moR + i_dm * nkpts * naoo;
                                mI = moI + i_dm * nkpts * naoo;

                                if (k_to_compute[ki]) {
                                        vR = vkR + i_dm * nkpts * nao2;
                                        vI = vkI + i_dm * nkpts * nao2;
// vk += np.einsum('igj,jn,kn,lgk,g->il', pqG, mo, mo.conj(), pqG.conj(), coulG)
dgemm_(&TRANS_N, &TRANS_N, &nocc, &nm, &nao, &D1, mR+kj*naoo, &nocc, pLqR, &nao, &D0, bufR, &nocc);
dgemm_(&TRANS_N, &TRANS_N, &nocc, &nm, &nao, &N1, mI+kj*naoo, &nocc, pLqI, &nao, &D1, bufR, &nocc);
dgemm_(&TRANS_N, &TRANS_N, &nocc, &nm, &nao, &D1, mI+kj*naoo, &nocc, pLqR, &nao, &D0, bufI, &nocc);
dgemm_(&TRANS_N, &TRANS_N, &nocc, &nm, &nao, &D1, mR+kj*naoo, &nocc, pLqI, &nao, &D1, bufI, &nocc);
for (i = 0; i < nao; i++) {
for (ig = 0; ig < mg; ig++) {
        c = coulG[g0+ig];
        ptr = i * go + ig * nocc;
        for (j = 0; j < nocc; j++) {
                buf1R[ptr+j] = bufR[ptr+j] * c;
                buf1I[ptr+j] = bufI[ptr+j] * c;
        }
} }
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &go, &D1, bufR, &go, buf1R, &go, &D1, vR+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &go, &D1, bufI, &go, buf1I, &go, &D1, vR+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &go, &N1, bufI, &go, buf1R, &go, &D1, vI+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &go, &D1, bufR, &go, buf1I, &go, &D1, vI+ki*nao2, &nao);
                        }

                                if (swap_2e && k_to_compute[kj]) {
                                        vR = vtmpR + i_dm * nkpts * nao2;
                                        vI = vtmpI + i_dm * nkpts * nao2;
// vk += np.einsum('igj,ln,in,lgk,g->kj', pqG, mo, mo.conj(), pqG.conj(), coulG)
dgemm_(&TRANS_N, &TRANS_T, &nm, &nocc, &nao, &D1, pLqR, &nm, mR+ki*naoo, &nocc, &D0, bufR, &nm);
dgemm_(&TRANS_N, &TRANS_T, &nm, &nocc, &nao, &D1, pLqI, &nm, mI+ki*naoo, &nocc, &D1, bufR, &nm);
dgemm_(&TRANS_N, &TRANS_T, &nm, &nocc, &nao, &D1, pLqI, &nm, mR+ki*naoo, &nocc, &D0, bufI, &nm);
dgemm_(&TRANS_N, &TRANS_T, &nm, &nocc, &nao, &N1, pLqR, &nm, mI+ki*naoo, &nocc, &D1, bufI, &nm);
for (i = 0; i < nocc; i++) {
for (ig = 0; ig < mg; ig++) {
        c = coulG[g0+ig];
        ptr = i * nm + ig * nao;
        for (j = 0; j < nao; j++) {
                buf1R[ptr+j] = bufR[ptr+j] * c;
                buf1I[ptr+j] = bufI[ptr+j] * c;
        }
} }
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &go, &D1, buf1R, &nao, bufR, &nao, &D1, vR+kj*nao2, &nao);
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &go, &D1, buf1I, &nao, bufI, &nao, &D1, vR+kj*nao2, &nao);
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &go, &D1, buf1I, &nao, bufR, &nao, &D1, vI+kj*nao2, &nao);
dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &go, &N1, buf1R, &nao, bufI, &nao, &D1, vI+kj*nao2, &nao);
                                }
                        }
                }
        }
        free(pLqR);
#pragma omp barrier
#pragma omp for schedule(static)
        for (i = 0; i < size_vk; i++) {
                vkR[i] += vtmpR[i];
                vkI[i] += vtmpI[i];
        }
}
        free(vtmpR);
}
