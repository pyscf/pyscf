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
#include <string.h>
#include <complex.h>
#include <assert.h>
#include "config.h"
#include "cint.h"
#include "gto/ft_ao.h"
#include "vhf/fblas.h"

#define INTBUFMAX       16000
#define IMGBLK          80
#define OF_CMPLX        2

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

int PBCsizeof_env(int *shls_slice,
                  int *atm, int natm, int *bas, int nbas, double *env);

static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
        env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
        env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
        env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

/*
 * Multiple k-points
 */
static void _ft_fill_k(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                       void (*fsort)(), double complex *out, int nkpts,
                       int comp, int nimgs, int blksize, int ish, int jsh,
                       double complex *buf, double *env_loc, double *Ls,
                       double complex *expkL, int *shls_slice, int *ao_loc,
                       double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int jsh0 = shls_slice[2];
        ish += ish0;
        jsh += jsh0;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const char TRANS_N = 'N';
        const double complex Z1 = 1;

        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        int shls[2] = {ish, jsh};
        int dims[2] = {di, dj};
        double complex *bufk = buf;
        double complex *bufL = buf + dij*blksize * comp * nkpts;
        double complex *pbuf;
        int gs0, gs1, dg, dijg;
        int jL0, jLcount, jL;
        int i;

        for (gs0 = 0; gs0 < nGv; gs0 += blksize) {
                gs1 = MIN(gs0+blksize, nGv);
                dg = gs1 - gs0;
                dijg = dij * dg * comp;
                for (i = 0; i < dijg*nkpts; i++) {
                        bufk[i] = 0;
                }

                for (jL0 = 0; jL0 < nimgs; jL0 += IMGBLK) {
                        jLcount = MIN(IMGBLK, nimgs-jL0);
                        pbuf = bufL;
                        for (jL = jL0; jL < jL0+jLcount; jL++) {
                                shift_bas(env_loc, env, Ls, jptrxyz, jL);
                                if ((*intor)(pbuf, shls, dims, eval_aopair, eval_gz,
                                             Z1, sGv, b, sgxyz, gs, dg,
                                             atm, natm, bas, nbas, env_loc)) {
                                } else {
                                        for (i = 0; i < dijg; i++) {
                                                pbuf[i] = 0;
                                        }
                                }
                                pbuf += dijg;
                        }
                        zgemm_(&TRANS_N, &TRANS_N, &dijg, &nkpts, &jLcount,
                               &Z1, bufL, &dijg, expkL+jL0, &nimgs,
                               &Z1, bufk, &dijg);
                }

                (*fsort)(out, bufk, shls_slice, ao_loc,
                         nkpts, comp, nGv, ish, jsh, gs0, gs1);

                sGv += dg * 3;
                if (sgxyz != NULL) {
                        sgxyz += dg * 3;
                }
        }
}

/*
 * Single k-point
 */
static void _ft_fill_nk1(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                         void (*fsort)(), double complex *out, int nkpts,
                         int comp, int nimgs, int blksize, int ish, int jsh,
                         double complex *buf, double *env_loc, double *Ls,
                         double complex *expkL, int *shls_slice, int *ao_loc,
                         double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int jsh0 = shls_slice[2];
        ish += ish0;
        jsh += jsh0;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;

        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        int shls[2] = {ish, jsh};
        int dims[2] = {di, dj};
        double complex *bufk = buf;
        double complex *bufL = buf + dij*blksize * comp;
        int gs0, gs1, dg, jL, i;
        size_t dijg;

        for (gs0 = 0; gs0 < nGv; gs0 += blksize) {
                gs1 = MIN(gs0+blksize, nGv);
                dg = gs1 - gs0;
                dijg = dij * dg * comp;
                for (i = 0; i < dijg; i++) {
                        bufk[i] = 0;
                }

                for (jL = 0; jL < nimgs; jL++) {
                        shift_bas(env_loc, env, Ls, jptrxyz, jL);
                        if ((*intor)(bufL, shls, dims, eval_aopair, eval_gz,
                                     expkL[jL], sGv, b, sgxyz, gs, dg,
                                     atm, natm, bas, nbas, env_loc)) {
                                for (i = 0; i < dijg; i++) {
                                        bufk[i] += bufL[i];
                                }
                        }
                }

                (*fsort)(out, bufk, shls_slice, ao_loc,
                         nkpts, comp, nGv, ish, jsh, gs0, gs1);

                sGv += dg * 3;
                if (sgxyz != NULL) {
                        sgxyz += dg * 3;
                }
        }
}

static void sort_s1(double complex *out, double complex *in,
                    int *shls_slice, int *ao_loc, int nkpts, int comp,
                    int nGv, int ish, int jsh, int gs0, int gs1)
{
        const size_t NGv = nGv;
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nijg = naoi * naoj * NGv;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int dg = gs1 - gs0;
        const size_t dijg = di * dj * dg;
        out += (ip * naoj + jp) * NGv + gs0;

        int i, j, n, ic, kk;
        double complex *pin, *pout;

        for (kk = 0; kk < nkpts; kk++) {
        for (ic = 0; ic < comp; ic++) {
                for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        pout = out + (i*naoj+j) * NGv;
                        pin  = in + (j*di+i) * dg;
                        for (n = 0; n < dg; n++) {
                                pout[n] = pin[n];
                        }
                } }
                out += nijg;
                in  += dijg;
        } }
}

static void sort_s2_igtj(double complex *out, double complex *in,
                         int *shls_slice, int *ao_loc, int nkpts, int comp,
                         int nGv, int ish, int jsh, int gs0, int gs1)
{
        const size_t NGv = nGv;
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijg = nij * NGv;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int dg = gs1 - gs0;
        const size_t dijg = dij * dg;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (ao_loc[ish]*(ao_loc[ish]+1)/2-off0 + jp) * NGv + gs0;

        const int ip1 = ao_loc[ish] + 1;
        int i, j, n, ic, kk;
        double complex *pin, *pout;

        for (kk = 0; kk < nkpts; kk++) {
        for (ic = 0; ic < comp; ic++) {
                pout = out;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pin = in + (j*di+i) * dg;
                                for (n = 0; n < dg; n++) {
                                        pout[j*NGv+n] = pin[n];
                                }
                        }
                        pout += (ip1 + i) * NGv;
                }
                out += nijg;
                in  += dijg;
        } }
}

static void sort_s2_ieqj(double complex *out, double complex *in,
                         int *shls_slice, int *ao_loc, int nkpts, int comp,
                         int nGv, int ish, int jsh, int gs0, int gs1)
{
        const size_t NGv = nGv;
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijg = nij * NGv;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int dg = gs1 - gs0;
        const size_t dijg = dij * dg;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (ao_loc[ish]*(ao_loc[ish]+1)/2-off0 + jp) * NGv + gs0;

        const int ip1 = ao_loc[ish] + 1;
        int i, j, n, ic, kk;
        double complex *pin, *pout;

        for (kk = 0; kk < nkpts; kk++) {
        for (ic = 0; ic < comp; ic++) {
                pout = out;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                pin = in + (j*di+i) * dg;
                                for (n = 0; n < dg; n++) {
                                        pout[j*NGv+n] = pin[n];
                                }
                        }
                        pout += (ip1 + i) * NGv;
                }
                out += nijg;
                in  += dijg;
        } }
}

void PBC_ft_fill_ks1(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                     double complex *out, int nkpts, int comp, int nimgs,
                     int blksize, int ish, int jsh,
                     double complex *buf, double *env_loc, double *Ls,
                     double complex *expkL, int *shls_slice, int *ao_loc,
                     double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        _ft_fill_k(intor, eval_aopair, eval_gz, &sort_s1,
                   out, nkpts, comp, nimgs, blksize, ish, jsh,
                   buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                   sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
}

void PBC_ft_fill_ks2(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                     double complex *out, int nkpts, int comp, int nimgs,
                     int blksize, int ish, int jsh,
                     double complex *buf, double *env_loc, double *Ls,
                     double complex *expkL, int *shls_slice, int *ao_loc,
                     double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
                _ft_fill_k(intor, eval_aopair, eval_gz, &sort_s2_igtj,
                           out, nkpts, comp, nimgs, blksize, ish, jsh,
                           buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                           sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
        } else if (ip == jp) {
                _ft_fill_k(intor, eval_aopair, eval_gz, &sort_s2_ieqj,
                           out, nkpts, comp, nimgs, blksize, ish, jsh,
                           buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                           sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
        }
}

void PBC_ft_fill_nk1s1(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                       double complex *out, int nkpts, int comp, int nimgs,
                       int blksize, int ish, int jsh,
                       double complex *buf, double *env_loc, double *Ls,
                       double complex *expkL, int *shls_slice, int *ao_loc,
                       double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        _ft_fill_nk1(intor, eval_aopair, eval_gz, &sort_s1,
                     out, nkpts, comp, nimgs, blksize, ish, jsh,
                     buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                     sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
}

void PBC_ft_fill_nk1s1hermi(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                            double complex *out, int nkpts, int comp, int nimgs,
                            int blksize, int ish, int jsh,
                            double complex *buf, double *env_loc, double *Ls,
                            double complex *expkL, int *shls_slice, int *ao_loc,
                            double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip >= jp) {
                _ft_fill_nk1(intor, eval_aopair, eval_gz, &sort_s1,
                             out, nkpts, comp, nimgs, blksize, ish, jsh,
                             buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                             sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
        }
}

void PBC_ft_fill_nk1s2(int (*intor)(), int (*eval_aopair)(), void (*eval_gz)(),
                       double complex *out, int nkpts, int comp, int nimgs,
                       int blksize, int ish, int jsh,
                       double complex *buf, double *env_loc, double *Ls,
                       double complex *expkL, int *shls_slice, int *ao_loc,
                       double *sGv, double *b, int *sgxyz, int *gs, int nGv,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
                _ft_fill_nk1(intor, eval_aopair, eval_gz, &sort_s2_igtj,
                             out, nkpts, comp, nimgs, blksize, ish, jsh,
                             buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                             sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
        } else if (ip == jp) {
                _ft_fill_nk1(intor, eval_aopair, eval_gz, &sort_s2_ieqj,
                             out, nkpts, comp, nimgs, blksize, ish, jsh,
                             buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                             sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
        }
}

static int subgroupGv(double *sGv, int *sgxyz, double *Gv, int *gxyz,
                      int nGv, int bufsize, int *shls_slice, int *ao_loc,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int i;
        int dimax = 0;
        int djmax = 0;
        for (i = shls_slice[0]; i < shls_slice[1]; i++) {
                dimax = MAX(dimax, ao_loc[i+1]-ao_loc[i]);
        }
        for (i = shls_slice[2]; i < shls_slice[3]; i++) {
                djmax = MAX(djmax, ao_loc[i+1]-ao_loc[i]);
        }
        int dij = dimax * djmax;
        int gblksize = 0xfffffff8 & (bufsize / dij);

        int gs0, dg;
        for (gs0 = 0; gs0 < nGv; gs0 += gblksize) {
                dg = MIN(nGv-gs0, gblksize);
                for (i = 0; i < 3; i++) {
                        memcpy(sGv+dg*i, Gv+nGv*i+gs0, sizeof(double)*dg);
                }
                sGv += dg * 3;
                if (gxyz != NULL) {
                        for (i = 0; i < 3; i++) {
                                memcpy(sgxyz+dg*i, gxyz+nGv*i+gs0, sizeof(int)*dg);
                        }
                        sgxyz += dg * 3;
                }
        }
        return gblksize;
}

void PBC_ft_latsum_drv(int (*intor)(), void (*eval_gz)(), void (*fill)(),
                       double complex *out, int nkpts, int comp, int nimgs,
                       double *Ls, double complex *expkL,
                       int *shls_slice, int *ao_loc,
                       double *Gv, double *b, int *gxyz, int *gs, int nGv,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        double *sGv = malloc(sizeof(double) * nGv * 3);
        int *sgxyz = NULL;
        if (gxyz != NULL) {
                sgxyz = malloc(sizeof(int) * nGv * 3);
        }
        int blksize;

        if (fill == &PBC_ft_fill_nk1s1 || fill == &PBC_ft_fill_nk1s2 ||
            fill == &PBC_ft_fill_nk1s1hermi) {
                blksize = subgroupGv(sGv, sgxyz, Gv, gxyz, nGv, INTBUFMAX*IMGBLK/2,
                                     shls_slice, ao_loc, atm, natm, bas, nbas, env);
        } else {
                blksize = subgroupGv(sGv, sgxyz, Gv, gxyz, nGv, INTBUFMAX,
                                     shls_slice, ao_loc, atm, natm, bas, nbas, env);
        }
        int (*eval_aopair)() = NULL;
        if (intor != &GTO_ft_ovlp_cart && intor != &GTO_ft_ovlp_sph) {
                eval_aopair = &GTO_aopair_lazy_contract;
        }

#pragma omp parallel
{
        int i, j, ij;
        int nenv = PBCsizeof_env(shls_slice, atm, natm, bas, nbas, env);
        nenv = MAX(nenv, PBCsizeof_env(shls_slice+2, atm, natm, bas, nbas, env));
        double *env_loc = malloc(sizeof(double)*nenv);
        memcpy(env_loc, env, sizeof(double)*nenv);
        size_t count = nkpts + IMGBLK;
        double complex *buf = malloc(sizeof(double complex)*count*INTBUFMAX*comp);
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                (*fill)(intor, eval_aopair, eval_gz,
                        out, nkpts, comp, nimgs, blksize, i, j,
                        buf, env_loc, Ls, expkL, shls_slice, ao_loc,
                        sGv, b, sgxyz, gs, nGv, atm, natm, bas, nbas, env);
        }
        free(buf);
        free(env_loc);
}
        free(sGv);
        if (sgxyz != NULL) {
                free(sgxyz);
        }
}

