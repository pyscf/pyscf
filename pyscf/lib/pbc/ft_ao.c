/* Copyright 2014-2018, 2021 The PySCF Developers. All Rights Reserved.

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
#include <complex.h>
#include <math.h>
#include <assert.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "gto/ft_ao.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "pbc/pbc.h"
#include "np_helper/np_helper.h"

#define OF_CMPLX        2
#define BLOCK_SIZE      104

typedef int (*FPtrSort)(double *out, double *in, int fill_zero,
                        int *shls_slice, int *ao_loc, int nkpts, int comp,
                        int nGv, int ish, int jsh, int grid0, int grid1);

typedef int (*FPtrFill)(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrSort fsort,
                        double *out, double *buf, int *cell0_shls,
                        CINTEnvVars *envs_cint, BVKEnvs *envs_bvk);

void PBCminimal_CINTEnvVars(CINTEnvVars *envs, int *atm, int natm, int *bas, int nbas, double *env,
                            CINTOpt *cintopt);

static int _assemble2c(FPtrIntor intor, FPtr_eval_gz eval_gz,
                       double *eriR, double *eriI, double *cache,
                       int grid0, int grid1, int ish_cell0, int jsh_bvk,
                       double complex fac, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int natm = envs_cint->natm;
        int nbas = envs_cint->nbas;
//        int ncomp = envs_cint->ncomp;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int iseg0 = seg_loc[ish_cell0];
        int iseg1 = seg_loc[ish_cell0+1];
        int jseg0 = seg_loc[jsh_bvk];
        int jseg1 = seg_loc[jsh_bvk+1];
        int empty = 1;
        if (iseg0 == iseg1 || jseg0 == jseg1) {
                return !empty;
        }

        int jsh0 = seg2sh[jseg0];
        int jsh1 = seg2sh[jseg1];
        int ngrids = envs_bvk->nGv;
        int dg = grid1 - grid0;
        int *atm = envs_cint->atm;
        int *bas = envs_cint->bas;
        double *env = envs_cint->env;
        double *Gv = envs_bvk->Gv;
        double *b = envs_bvk->b;
        int *gxyz = envs_bvk->gxyz;
        int *gs = envs_bvk->gs;
        int8_t *ovlp_mask = envs_bvk->ovlp_mask;
        int shls[2] = {0,};
        int ish, jsh, iseg;

        for (iseg = iseg0; iseg < iseg1; iseg++) {
                ish = seg2sh[iseg];
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        if (!ovlp_mask[iseg*nbas+jsh]) {
                                continue;
                        }
                        shls[0] = ish;
                        shls[1] = jsh;
                        if ((*intor)(eriR, eriI, shls, NULL, eval_gz,
                                     fac, Gv+grid0, b, gxyz+grid0, gs, ngrids, dg,
                                     atm, natm, bas, nbas, env, NULL)) {
                                empty = 0;
                        }
                }
        }
        return !empty;
}

/*
 * Multiple k-points for BvK cell
 */
void PBC_ft_bvk_ks1(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrSort fsort,
                    double *out, double *buf, int *cell0_shls,
                    CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_bvk->shls_slice;
        int *cell0_ao_loc = envs_bvk->ao_loc;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];

        int di = cell0_ao_loc[ish_cell0+1] - cell0_ao_loc[ish_cell0];
        int dj = cell0_ao_loc[jsh_cell0+1] - cell0_ao_loc[jsh_cell0];
        int dij = di * dj;
        char TRANS_N = 'N';
        char TRANS_T = 'T';
        double D0 = 0;
        double D1 = 1;
        double ND1 = -1;
        double complex Z1 = 1;

        int comp = envs_bvk->ncomp;
        int nGv = envs_bvk->nGv;
        int bvk_ncells = envs_bvk->ncells;
        int nkpts = envs_bvk->nkpts;
        int nbasp = envs_bvk->nbasp;
        double *expLkR = envs_bvk->expLkR;
        double *expLkI = envs_bvk->expLkI;
        double *bufkR = buf;
        double *bufkI = bufkR + ((size_t)dij) * BLOCK_SIZE * comp * nkpts;
        double *bufLR = bufkI + ((size_t)dij) * BLOCK_SIZE * comp * nkpts;
        double *bufLI = bufLR + ((size_t)dij) * BLOCK_SIZE * comp * bvk_ncells;
        double *cache = bufLI + ((size_t)dij) * BLOCK_SIZE * comp * bvk_ncells;
        double *pbufR, *pbufI;
        int grid0, grid1, dg, dijg, jL, jLmax, nLj, empty;

        // TODO: precompute opts??

        for (grid0 = 0; grid0 < nGv; grid0 += BLOCK_SIZE) {
                grid1 = MIN(grid0+BLOCK_SIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg * comp;

                jLmax = -1;
                for (jL = 0; jL < bvk_ncells; jL++) {
                        pbufR = bufLR + jL * dijg;
                        pbufI = bufLI + jL * dijg;
                        NPdset0(pbufR, dijg);
                        NPdset0(pbufI, dijg);
                        if (_assemble2c(intor, eval_gz, pbufR, pbufI, cache,
                                        grid0, grid1, ish_cell0, jL*nbasp+jsh_cell0,
                                        Z1, envs_cint, envs_bvk)) {
                                jLmax = jL;
                        }
                }

                empty = jLmax == -1;
                if (!empty) {
                        nLj = jLmax + 1;
                        dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                               &D1, bufLR, &dijg, expLkR, &nkpts, &D0, bufkR, &dijg);
                        dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                               &ND1, bufLI, &dijg, expLkI, &nkpts, &D1, bufkR, &dijg);
                        dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                               &D1, bufLR, &dijg, expLkI, &nkpts, &D0, bufkI, &dijg);
                        dgemm_(&TRANS_N, &TRANS_T, &dijg, &nkpts, &nLj,
                               &D1, bufLI, &dijg, expLkR, &nkpts, &D1, bufkI, &dijg);
                }

                (*fsort)(out, bufkR, empty, shls_slice, cell0_ao_loc,
                         nkpts, comp, nGv, ish_cell0, jsh_cell0, grid0, grid1);
        }
}

/*
 * Single k-point for BvK cell
 */
void PBC_ft_bvk_nk1s1(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrSort fsort,
                      double *out, double *buf, int *cell0_shls,
                      CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_bvk->shls_slice;
        int *cell0_ao_loc = envs_bvk->ao_loc;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];

        int di = cell0_ao_loc[ish_cell0+1] - cell0_ao_loc[ish_cell0];
        int dj = cell0_ao_loc[jsh_cell0+1] - cell0_ao_loc[jsh_cell0];
        int dij = di * dj;
        int comp = envs_bvk->ncomp;
        int nGv = envs_bvk->nGv;
        int bvk_ncells = envs_bvk->ncells;
        int nkpts = envs_bvk->nkpts;
        int nbasp = envs_bvk->nbasp;
        double *expLkR = envs_bvk->expLkR;
        double *expLkI = envs_bvk->expLkI;
        double *bufR = buf;
        double *bufI = bufR + dij * BLOCK_SIZE * comp;
        double *cache = bufI + dij * BLOCK_SIZE * comp;
        double complex fac;
        int grid0, grid1, dg, jL, dijg, empty;

        for (grid0 = 0; grid0 < nGv; grid0 += BLOCK_SIZE) {
                grid1 = MIN(grid0+BLOCK_SIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg * comp;
                NPdset0(bufR, dijg);
                NPdset0(bufI, dijg);

                empty = 1;
                for (jL = 0; jL < bvk_ncells; jL++) {
                        fac = expLkR[jL] + expLkI[jL] * _Complex_I;
                        if (_assemble2c(intor, eval_gz, bufR, bufI, cache,
                                        grid0, grid1, ish_cell0, jL*nbasp+jsh_cell0,
                                        fac, envs_cint, envs_bvk)) {
                                empty = 0;
                        }
                }

                (*fsort)(out, bufR, empty, shls_slice, cell0_ao_loc,
                         nkpts, comp, nGv, ish_cell0, jsh_cell0, grid0, grid1);
        }
}

static void _fill0(FPtrSort fsort, double *out, int *cell0_shls,
                   BVKEnvs *envs_bvk)
{
        int *shls_slice = envs_bvk->shls_slice;
        int *cell0_ao_loc = envs_bvk->ao_loc;
        int comp = envs_bvk->ncomp;
        int nGv = envs_bvk->nGv;
        int nkpts = envs_bvk->nkpts;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int fill_zero = 1;
        (*fsort)(out, NULL, fill_zero, shls_slice, cell0_ao_loc,
                 nkpts, comp, nGv, ish_cell0, jsh_cell0, 0, nGv);
}

void PBC_ft_dsort_s1(double *out, double *in, int fill_zero,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nijg = naoi * naoj * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int ip = ao_loc[ish] - ao_loc[ish0];
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        int dg = grid1 - grid0;
        int dij = di * dj;
        int dijg = dij * dg;
        double *outR = out + (ip * naoj + jp) * NGv + grid0;
        double *outI = outR + nijg * nkpts * comp;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int i, j, n, ic, kk;
        double *pinR, *pinI, *poutR, *poutI;

        if (fill_zero) {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                poutR = outR + (i*naoj+j) * NGv;
                                poutI = outI + (i*naoj+j) * NGv;
                                for (n = 0; n < dg; n++) {
                                        poutR[n] = 0;
                                        poutI[n] = 0;
                                }
                        } }
                        outR += nijg;
                        outI += nijg;
                } }
        } else {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                poutR = outR + (i*naoj+j) * NGv;
                                poutI = outI + (i*naoj+j) * NGv;
                                pinR  = inR + (j*di+i) * dg;
                                pinI  = inI + (j*di+i) * dg;
                                for (n = 0; n < dg; n++) {
                                        poutR[n] = pinR[n];
                                        poutI[n] = pinI[n];
                                }
                        } }
                        outR += nijg;
                        outI += nijg;
                        inR  += dijg;
                        inI  += dijg;
                } }
        }
}

void PBC_ft_dsort_s2(double *out, double *in, int fill_zero,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        size_t nij  = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        size_t nijg = nij * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int dg = grid1 - grid0;
        size_t dijg = dij * dg;
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        double *outR = out + (((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * NGv + grid0;
        double *outI = outR + nijg * nkpts * comp;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int ip1 = ao_loc[ish] + 1;
        int i, j, n, ic, kk;
        double *pinR, *pinI, *poutR, *poutI;

        if (ish > jsh) {
                if (fill_zero) {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                poutR = outR + (kk * comp + ic) * nijg;
                                poutI = outI + (kk * comp + ic) * nijg;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                        for (n = 0; n < dg; n++) {
                                                poutR[j*NGv+n] = 0;
                                                poutI[j*NGv+n] = 0;
                                        } }
                                        poutR += (ip1 + i) * NGv;
                                        poutI += (ip1 + i) * NGv;
                                }
                        } }
                } else {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                poutR = outR + (kk * comp + ic) * nijg;
                                poutI = outI + (kk * comp + ic) * nijg;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                                pinR = inR + (j*di+i) * dg;
                                                pinI = inI + (j*di+i) * dg;
                                                for (n = 0; n < dg; n++) {
                                                        poutR[j*NGv+n] = pinR[n];
                                                        poutI[j*NGv+n] = pinI[n];
                                                }
                                        }
                                        poutR += (ip1 + i) * NGv;
                                        poutI += (ip1 + i) * NGv;
                                }
                                inR += dijg;
                                inI += dijg;
                        } }
                }
        } else if (ish == jsh) {
                if (fill_zero) {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                poutR = outR + (kk * comp + ic) * nijg;
                                poutI = outI + (kk * comp + ic) * nijg;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j <= i; j++) {
                                        for (n = 0; n < dg; n++) {
                                                poutR[j*NGv+n] = 0;
                                                poutI[j*NGv+n] = 0;
                                        } }
                                        poutR += (ip1 + i) * NGv;
                                        poutI += (ip1 + i) * NGv;
                                }
                        } }
                } else {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                poutR = outR + (kk * comp + ic) * nijg;
                                poutI = outI + (kk * comp + ic) * nijg;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j <= i; j++) {
                                                pinR = inR + (j*di+i) * dg;
                                                pinI = inI + (j*di+i) * dg;
                                                for (n = 0; n < dg; n++) {
                                                        poutR[j*NGv+n] = pinR[n];
                                                        poutI[j*NGv+n] = pinI[n];
                                                }
                                        }
                                        poutR += (ip1 + i) * NGv;
                                        poutI += (ip1 + i) * NGv;
                                }
                                inR += dijg;
                                inI += dijg;
                        } }
                }
        }
}

void PBC_ft_zsort_s1(double *out, double *in, int fill_zero,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nijg = naoi * naoj * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int ip = ao_loc[ish] - ao_loc[ish0];
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        int dg = grid1 - grid0;
        int dij = di * dj;
        int dijg = dij * dg;
        out += ((ip * naoj + jp) * NGv + grid0) * OF_CMPLX;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int i, j, n, ic, kk;
        double *pinR, *pinI, *pout;

        if (fill_zero) {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                pout = out + (i*naoj+j) * NGv * OF_CMPLX;
                                for (n = 0; n < dg*OF_CMPLX; n++) {
                                        pout[n] = 0;
                                }
                        } }
                        out += nijg * OF_CMPLX;
                } }
        } else {
                for (kk = 0; kk < nkpts; kk++) {
                for (ic = 0; ic < comp; ic++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                pout = out + (i*naoj+j) * NGv * OF_CMPLX;
                                pinR = inR + (j*di+i) * dg;
                                pinI = inI + (j*di+i) * dg;
                                for (n = 0; n < dg; n++) {
                                        pout[n*OF_CMPLX  ] = pinR[n];
                                        pout[n*OF_CMPLX+1] = pinI[n];
                                }
                        } }
                        out += nijg * OF_CMPLX;
                        inR += dijg;
                        inI += dijg;
                } }
        }
}

void PBC_ft_zsort_s2(double *out, double *in, int fill_zero,
                     int *shls_slice, int *ao_loc, int nkpts, int comp,
                     int nGv, int ish, int jsh, int grid0, int grid1)
{
        size_t NGv = nGv;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        size_t nij  = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        size_t nijg = nij * NGv;

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int dg = grid1 - grid0;
        size_t dijg = dij * dg;
        int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += ((((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * NGv + grid0) * OF_CMPLX;
        double *inR = in;
        double *inI = inR + dij * BLOCK_SIZE * comp * nkpts;

        int ip1 = ao_loc[ish] + 1;
        int i, j, n, ic, kk;
        double *pinR, *pinI, *pout;

        if (ish > jsh) {
                if (fill_zero) {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + (kk * comp + ic) * nijg * OF_CMPLX;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                        for (n = 0; n < dg*OF_CMPLX; n++) {
                                                pout[j*NGv*OF_CMPLX+n] = 0;
                                        } }
                                        pout += (ip1 + i) * NGv * OF_CMPLX;
                                }
                        } }
                } else {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + (kk * comp + ic) * nijg * OF_CMPLX;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                                pinR = inR + (j*di+i) * dg;
                                                pinI = inI + (j*di+i) * dg;
                                                for (n = 0; n < dg; n++) {
                                                        pout[(j*NGv+n)*OF_CMPLX  ] = pinR[n];
                                                        pout[(j*NGv+n)*OF_CMPLX+1] = pinI[n];
                                                }
                                        }
                                        pout += (ip1 + i) * NGv * OF_CMPLX;
                                }
                                inR += dijg;
                                inI += dijg;
                        } }
                }
        } else if (ish == jsh) {
                if (fill_zero) {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + (kk * comp + ic) * nijg * OF_CMPLX;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j <= i; j++) {
                                        for (n = 0; n < dg*OF_CMPLX; n++) {
                                                pout[j*NGv*OF_CMPLX+n] = 0;
                                        } }
                                        pout += (ip1 + i) * NGv * OF_CMPLX;
                                }
                        } }
                } else {
                        for (kk = 0; kk < nkpts; kk++) {
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + (kk * comp + ic) * nijg * OF_CMPLX;
                                for (i = 0; i < di; i++) {
                                        for (j = 0; j <= i; j++) {
                                                pinR = inR + (j*di+i) * dg;
                                                pinI = inI + (j*di+i) * dg;
                                                for (n = 0; n < dg; n++) {
                                                        pout[(j*NGv+n)*OF_CMPLX  ] = pinR[n];
                                                        pout[(j*NGv+n)*OF_CMPLX+1] = pinI[n];
                                                }
                                        }
                                        pout += (ip1 + i) * NGv * OF_CMPLX;
                                }
                                inR += dijg;
                                inI += dijg;
                        } }
                }
        }
}

void PBC_ft_bvk_ks2(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrSort fsort,
                    double *out, double *buf, int *cell0_shls,
                    CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        if (ish_cell0 >= jsh_cell0) {
                PBC_ft_bvk_ks1(intor, eval_gz, fsort, out, buf, cell0_shls,
                               envs_cint, envs_bvk);
        }
}

void PBC_ft_bvk_nk1s2(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrSort fsort,
                      double *out, double *buf, int *cell0_shls,
                      CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        if (ish_cell0 >= jsh_cell0) {
                PBC_ft_bvk_nk1s1(intor, eval_gz, fsort, out, buf, cell0_shls,
                                 envs_cint, envs_bvk);
        }
}

void PBC_ft_bvk_nk1s1hermi(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrSort fsort,
                           double *out, double *buf, int *cell0_shls,
                           CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        PBC_ft_bvk_nk1s2(intor, eval_gz, fsort, out, buf, cell0_shls,
                         envs_cint, envs_bvk);
}

void PBC_ft_zsort_s1hermi(double *out, double *in, int fill_zero,
                          int *shls_slice, int *ao_loc, int nkpts, int comp,
                          int nGv, int ish, int jsh, int grid0, int grid1)
{
        PBC_ft_zsort_s1(out, in, fill_zero, shls_slice, ao_loc, nkpts, comp,
                        nGv, ish, jsh, grid0, grid1);
}

static size_t max_cache_size(FPtrIntor intor, FPtr_eval_gz eval_gz, int *shls_slice,
                             int *seg_loc, int *seg2sh,
                             double *Gv, double *b, int *gxyz, int *gs, int nGv,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish_cell0 = shls_slice[0];
        int ish_cell1 = shls_slice[1];
        int jsh_cell0 = shls_slice[2];
        int jsh_cell1 = shls_slice[3];
        int iseg0 = seg_loc[ish_cell0];
        int iseg1 = seg_loc[ish_cell1];
        int jseg0 = seg_loc[jsh_cell0];
        int jseg1 = seg_loc[jsh_cell1];
        int ish0 = seg2sh[iseg0];
        int ish1 = seg2sh[iseg1];
        int jsh0 = seg2sh[jseg0];
        int jsh1 = seg2sh[jseg1];
        int sh0 = MIN(ish0, jsh0);
        int sh1 = MAX(ish1, jsh1);
        int blksize = MIN(nGv, BLOCK_SIZE);
        double complex fac = 0.;
        int shls[2];
        int i, cache_size;
        size_t max_size = 0;
        for (i = sh0; i < sh1; i++) {
                shls[0] = i;
                shls[1] = i;
                cache_size = (*intor)(NULL, NULL, shls, NULL, eval_gz,
                                      fac, Gv, b, gxyz, gs, nGv, blksize,
                                      atm, natm, bas, nbas, env, NULL);
                max_size = MAX(max_size, cache_size);
        }
        return max_size * blksize;
}

void PBC_ft_bvk_drv(FPtrIntor intor, FPtr_eval_gz eval_gz, FPtrFill fill, FPtrSort fsort,
                    double *out, double *expLkR, double *expLkI,
                    int bvk_ncells, int nimgs, int nkpts, int nbasp, int comp,
                    int *seg_loc, int *seg2sh, int *cell0_ao_loc, int *shls_slice,
                    int8_t *ovlp_mask, int8_t *cell0_ovlp_mask,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int nish = ish1 - ish0;
        int njsh = jsh1 - jsh0;
        int di = GTOmax_shell_dim(cell0_ao_loc, shls_slice, 2);
        BVKEnvs envs_bvk = {bvk_ncells, nimgs,
                nkpts, nkpts, nbasp, comp, nGv, 0,
                seg_loc, seg2sh, cell0_ao_loc, shls_slice, NULL,
                expLkR, expLkI, ovlp_mask, NULL, 0, 0.f, Gv, b, gxyz, gs};
        size_t count = nkpts + bvk_ncells;
        size_t buf_size = di * di * BLOCK_SIZE * count * comp * OF_CMPLX;
        size_t cache_size = max_cache_size(intor, eval_gz, shls_slice,
                                           seg_loc, seg2sh, Gv, b, gxyz, gs, nGv,
                                           atm, natm, bas, nbas, env);

#pragma omp parallel
{
        CINTEnvVars envs_cint;
        PBCminimal_CINTEnvVars(&envs_cint, atm, natm, bas, nbas, env, NULL);
        int ish, jsh, ij;
        int cell0_shls[2];
        double *buf = malloc(sizeof(double) * (buf_size+cache_size));
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh + ish0;
                jsh = ij % njsh + jsh0;
                cell0_shls[0] = ish;
                cell0_shls[1] = jsh;
                if (!cell0_ovlp_mask[ish*nbasp+jsh]) {
                        _fill0(fsort, out, cell0_shls, &envs_bvk);
                        continue;
                }
                (*fill)(intor, eval_gz, fsort, out, buf, cell0_shls,
                        &envs_cint, &envs_bvk);
        }
        free(buf);
}
}

void PBC_ft_zfuse_dd_s1(double *outR, double *outI, double complex *pqG_dd,
                       int *ao_idx, int *grid_slice, int nao, int naod, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t Nao = nao;
        int ig0 = grid_slice[0];
        int ig1 = grid_slice[1];
        size_t ng = ig1 - ig0;
#pragma omp parallel
{
        size_t off_out, off_in;
        int i, j, ij, ip, jp, n;
#pragma omp for schedule(static)
        for (ij = 0; ij < naod*naod; ij++) {
                i = ij / naod;
                j = ij % naod;
                ip = ao_idx[i];
                jp = ao_idx[j];
                off_in = (i * naod + j) * Ngrids + ig0;
                off_out = (ip * Nao + jp) * ng;
                for (n = 0; n < ng; n++) {
                        outR[off_out+n] += creal(pqG_dd[off_in+n]);
                        outI[off_out+n] += cimag(pqG_dd[off_in+n]);
                }
        }
}
}

void PBC_ft_zfuse_dd_s2(double *outR, double *outI, double complex *pqG_dd,
                       int *ao_idx, int *grid_slice, int nao, int naod, int ngrids)
{
        size_t Ngrids = ngrids;
        int ig0 = grid_slice[0];
        int ig1 = grid_slice[1];
        size_t ng = ig1 - ig0;
#pragma omp parallel
{
        size_t off_out, off_in;
        int i, j, ij, ip, jp, n;
#pragma omp for schedule(static)
        for (ij = 0; ij < naod*(naod+1)/2; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - i*(i+1)/2;
                ip = ao_idx[i];
                jp = ao_idx[j];
                off_in = (i * naod + j) * Ngrids + ig0;
                off_out = (ip*(ip+1)/2 + jp) * ng;
                for (n = 0; n < ng; n++) {
                        outR[off_out+n] += creal(pqG_dd[off_in+n]);
                        outI[off_out+n] += cimag(pqG_dd[off_in+n]);
                }
        }
}
}

void PBC_ft_fuse_dd_s1(double *outR, double *outI, double *pqG_ddR, double *pqG_ddI,
                       int *ao_idx, int nao, int naod, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t Nao = nao;
#pragma omp parallel
{
        size_t off_out, off_in;
        int i, j, ij, ip, jp, n;
#pragma omp for schedule(static)
        for (ij = 0; ij < naod*naod; ij++) {
                i = ij / naod;
                j = ij % naod;
                ip = ao_idx[i];
                jp = ao_idx[j];
                off_in = (i * naod + j) * Ngrids;
                off_out = (ip * Nao + jp) * Ngrids;
                for (n = 0; n < Ngrids; n++) {
                        outR[off_out+n] += pqG_ddR[off_in+n];
                        outI[off_out+n] += pqG_ddI[off_in+n];
                }
        }
}
}

void PBC_ft_fuse_dd_s2(double *outR, double *outI, double *pqG_ddR, double *pqG_ddI,
                       int *ao_idx, int nao, int naod, int ngrids)
{
        size_t Ngrids = ngrids;
#pragma omp parallel
{
        size_t off_out, off_in;
        int i, j, ij, ip, jp, n;
#pragma omp for schedule(static)
        for (ij = 0; ij < naod*(naod+1)/2; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - i*(i+1)/2;
                ip = ao_idx[i];
                jp = ao_idx[j];
                off_in = (i * naod + j) * Ngrids;
                off_out = (ip*(ip+1)/2 + jp) * Ngrids;
                for (n = 0; n < Ngrids; n++) {
                        outR[off_out+n] += pqG_ddR[off_in+n];
                        outI[off_out+n] += pqG_ddI[off_in+n];
                }
        }
}
}

void PBCsupmol_ovlp_mask(int8_t *out, double cutoff,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        size_t Nbas = nbas;
        size_t Nbas1 = nbas + 1;
        int *exps_group_loc = malloc(sizeof(int) * Nbas1);
        double *exps = malloc(sizeof(double) * Nbas1 * 4);
        double *rx = exps + Nbas1;
        double *ry = rx + Nbas1;
        double *rz = ry + Nbas1;
        int ptr_coord, nprim, n;
        double log4 = log(4.) * .75;
        double log_cutoff = log(cutoff) - log4;
        double exp_min, exp_last;
        int ngroups = 0;
        exp_last = 0.;
        for (n = 0; n < nbas; n++) {
                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, n));
                rx[n] = env[ptr_coord+0];
                ry[n] = env[ptr_coord+1];
                rz[n] = env[ptr_coord+2];
                nprim = bas(NPRIM_OF, n);
                // the most diffused function
                exp_min = env[bas(PTR_EXP, n) + nprim - 1];

                if (exp_min != exp_last) {
                        // partition all exponents into groups
                        exps[ngroups] = exp_min;
                        exps_group_loc[ngroups] = n;
                        exp_last = exp_min;
                        ngroups++;
                }
        }
        exps_group_loc[ngroups] = nbas;

#pragma omp parallel
{
        int ijb, ib, jb, i0, j0, i1, j1, i, j, li, lj;
        double dx, dy, dz, ai, aj, aij, a1, fi, fj, rr, rij, dri, drj;
        double log_a1, rr_cutoff, li_a1, lj_a1;
#pragma omp for schedule(dynamic, 1)
        for (ijb = 0; ijb < ngroups*(ngroups+1)/2; ijb++) {
                ib = (int)(sqrt(2*ijb+.25) - .5 + 1e-7);
                jb = ijb - ib*(ib+1)/2;

                i0 = exps_group_loc[ib];
                i1 = exps_group_loc[ib+1];
                li = bas(ANG_OF, i0);
                ai = exps[ib];
                j0 = exps_group_loc[jb];
                j1 = exps_group_loc[jb+1];
                lj = bas(ANG_OF, j0);
                aj = exps[jb];
                aij = ai + aj;
                fi = ai / aij;
                fj = aj / aij;
                a1 = ai * aj / aij;
                log_a1 = .75 * log(a1 / aij);
                rr_cutoff = (log_a1 - log_cutoff) / a1;
                if (li > 0 && lj > 0 && a1 < 0.3) {
                        // the contribution of r^n should be considered for
                        // overlap of smooth basis functions
                        li_a1 = li / -a1;
                        lj_a1 = lj / -a1;
                        for (i = i0; i < i1; i++) {
#pragma GCC ivdep
                        for (j = j0; j < j1; j++) {
                                dx = rx[i] - rx[j];
                                dy = ry[i] - ry[j];
                                dz = rz[i] - rz[j];
                                rr = dx * dx + dy * dy + dz * dz;
                                rij = sqrt(rr);
                                dri = fj * rij + 1.;
                                drj = fi * rij + 1.;
                                out[i*Nbas+j] = rr + li_a1 * log(dri) + lj_a1 * log(drj) < rr_cutoff;
                        } }
                } else {
                        for (i = i0; i < i1; i++) {
#pragma GCC ivdep
                        for (j = j0; j < j1; j++) {
                                dx = rx[i] - rx[j];
                                dy = ry[i] - ry[j];
                                dz = rz[i] - rz[j];
                                rr = dx * dx + dy * dy + dz * dz;
                                out[i*Nbas+j] = rr < rr_cutoff;
                        } }
                }
                if (ib > jb) {
                        for (i = i0; i < i1; i++) {
#pragma GCC ivdep
                        for (j = j0; j < j1; j++) {
                                out[j*Nbas+i] = out[i*Nbas+j];
                        } }
                }
        }
}
        free(exps);
        free(exps_group_loc);
}
