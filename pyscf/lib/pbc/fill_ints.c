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
#include "vhf/fblas.h"
#include "pbc/optimizer.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static int shloc_partition(int *kshloc, int *ao_loc, int ksh0, int ksh1, int dkmax)
{
        int ksh;
        int nloc = 0;
        int loclast = ao_loc[ksh0];
        kshloc[0] = ksh0;
        for (ksh = ksh0+1; ksh < ksh1; ksh++) {
                assert(ao_loc[ksh+1] - ao_loc[ksh] < dkmax);
                if (ao_loc[ksh+1] - loclast > dkmax) {
                        nloc += 1;
                        kshloc[nloc] = ksh;
                        loclast = ao_loc[ksh];
                }
        }
        nloc += 1;
        kshloc[nloc] = ksh1;
        return nloc;
}

static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
        env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
        env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
        env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

static void sort3c_kks1(double complex *out, double *bufr, double *bufi,
                        int *kptij_idx, int *shls_slice, int *ao_loc,
                        int nkpts, int nkpts_ij, int comp, int ish, int jsh,
                        int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t njk = naoj * naok;
        const size_t nijk = njk * naoi;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int dij = di * dj;
        const int dkmax = ao_loc[msh1] - ao_loc[msh0];
        const size_t dijmc = dij * dkmax * comp;
        out += (ip * naoj + jp) * naok;

        int i, j, k, kk, ik, jk, ksh, ic, dk, dijk;
        size_t off;
        double *pbr, *pbi;
        double complex *pout;

        for (kk = 0; kk < nkpts_ij; kk++) {
                ik = kptij_idx[kk] / nkpts;
                jk = kptij_idx[kk] % nkpts;
                off = (ik*nkpts+jk) * dijmc;

                for (ksh = msh0; ksh < msh1; ksh++) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijk = dij * dk;
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + nijk*ic + ao_loc[ksh]-ao_loc[ksh0];
                                pbr = bufr + off + dijk*ic;
                                pbi = bufi + off + dijk*ic;
        for (j = 0; j < dj; j++) {
                for (k = 0; k < dk; k++) {
                for (i = 0; i < di; i++) {
                        pout[i*njk+k] = pbr[k*dij+i] + pbi[k*dij+i]*_Complex_I;
                } }
                pout += naok;
                pbr += di;
                pbi += di;
        }
                        }
                        off += dijk * comp;
                }
                out += nijk * comp;
        }
}

static void _nr3c_fill_kk(int (*intor)(), void (*fsort)(),
                          double complex *out, int nkpts_ij,
                          int nkpts, int comp, int nimgs, int ish, int jsh,
                          double *buf, double *env_loc, double *Ls,
                          double *expkL_r, double *expkL_i, int *kptij_idx,
                          int *shls_slice, int *ao_loc,
                          CINTOpt *cintopt, PBCOpt *pbcopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const double ND1 = -1;

        jsh += jsh0;
        ish += ish0;
        int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        int dkmax = INTBUFMAX / dij;
        int kshloc[ksh1-ksh0+1];
        int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

        int i, m, msh0, msh1, dijm, dijmc, dijmk, empty;
        int ksh, dk, iL0, iL, jL, iLcount;
        int shls[3];
        double *bufkk_r, *bufkk_i, *bufkL_r, *bufkL_i, *bufL, *pbuf, *cache;
        int (*fprescreen)();
        if (pbcopt != NULL) {
                fprescreen = pbcopt->fprescreen;
        } else {
                fprescreen = PBCnoscreen;
        }

        shls[0] = ish;
        shls[1] = jsh;
        for (m = 0; m < nkshloc; m++) {
                msh0 = kshloc[m];
                msh1 = kshloc[m+1];
                dkmax = ao_loc[msh1] - ao_loc[msh0];
                dijm = dij * dkmax;
                dijmc = dijm * comp;
                dijmk = dijmc * nkpts;
                bufkk_r = buf;
                bufkk_i = bufkk_r + (size_t)nkpts * dijmk;
                bufkL_r = bufkk_i + (size_t)nkpts * dijmk;
                bufkL_i = bufkL_r + (size_t)MIN(nimgs,IMGBLK) * dijmk;
                bufL    = bufkL_i + (size_t)MIN(nimgs,IMGBLK) * dijmk;
                cache   = bufL    + (size_t)nimgs * dijmc;
                for (i = 0; i < nkpts*dijmk*OF_CMPLX; i++) {
                        bufkk_r[i] = 0;
                }

                for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
                        iLcount = MIN(IMGBLK, nimgs - iL0);
                        for (iL = iL0; iL < iL0+iLcount; iL++) {
                                shift_bas(env_loc, env, Ls, iptrxyz, iL);
                                pbuf = bufL;
        for (jL = 0; jL < nimgs; jL++) {
                shift_bas(env_loc, env, Ls, jptrxyz, jL);
                if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                        for (ksh = msh0; ksh < msh1; ksh++) {
                                shls[2] = ksh;
                                if ((*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                                             env_loc, cintopt, cache)) {
                                        empty = 0;
                                }
                                dk = ao_loc[ksh+1] - ao_loc[ksh];
                                pbuf += dij*dk * comp;
                        }
                } else {
                        for (i = 0; i < dijmc; i++) {
                                pbuf[i] = 0;
                        }
                        pbuf += dijmc;
                }
        }
        dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &nimgs,
               &D1, bufL, &dijmc, expkL_r, &nimgs,
               &D0, bufkL_r+(iL-iL0)*(size_t)dijmk, &dijmc);
        dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &nimgs,
               &D1, bufL, &dijmc, expkL_i, &nimgs,
               &D0, bufkL_i+(iL-iL0)*(size_t)dijmk, &dijmc);

                        } // iL in range(0, nimgs)
                        // conj(exp(1j*dot(h,k)))
                        dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                               &D1, bufkL_r, &dijmk, expkL_r+iL0, &nimgs,
                               &D1, bufkk_r, &dijmk);
                        dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                               &D1, bufkL_i, &dijmk, expkL_i+iL0, &nimgs,
                               &D1, bufkk_r, &dijmk);
                        dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                               &D1, bufkL_i, &dijmk, expkL_r+iL0, &nimgs,
                               &D1, bufkk_i, &dijmk);
                        dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount,
                               &ND1, bufkL_r, &dijmk, expkL_i+iL0, &nimgs,
                               &D1, bufkk_i, &dijmk);
                }
                (*fsort)(out, bufkk_r, bufkk_i, kptij_idx, shls_slice,
                         ao_loc, nkpts, nkpts_ij, comp, ish, jsh,
                         msh0, msh1);
        }
}

/* ('...LM,kL,lM->...kl', int3c, exp_kL, exp_kL) */
void PBCnr3c_fill_kks1(int (*intor)(), double complex *out, int nkpts_ij,
                       int nkpts, int comp, int nimgs, int ish, int jsh,
                       double *buf, double *env_loc, double *Ls,
                       double *expkL_r, double *expkL_i, int *kptij_idx,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, PBCOpt *pbcopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        _nr3c_fill_kk(intor, &sort3c_kks1, out,
                      nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                      buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                      shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
}

static void sort3c_kks2_igtj(double complex *out, double *bufr, double *bufi,
                             int *kptij_idx, int *shls_slice, int *ao_loc,
                             int nkpts, int nkpts_ij, int comp, int ish, int jsh,
                             int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        assert(naoi == naoj);
        const size_t njk = naoj * naok;
        const size_t nijk = njk * naoi;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int dij = di * dj;
        const int dkmax = ao_loc[msh1] - ao_loc[msh0];
        const size_t dijmc = dij * dkmax * comp;
        double complex *outij = out + (ip * naoj + jp) * naok;
        double complex *outji = out + (jp * naoj + ip) * naok;

        int i, j, k, kk, ik, jk, ksh, ic, dk, dijk;
        size_t offij, offji;
        double *pbij_r, *pbij_i, *pbji_r, *pbji_i;
        double complex *poutij, *poutji;

        for (kk = 0; kk < nkpts_ij; kk++) {
                ik = kptij_idx[kk] / nkpts;
                jk = kptij_idx[kk] % nkpts;
                offij = (ik*nkpts+jk) * dijmc;
                offji = (jk*nkpts+ik) * dijmc;

                for (ksh = msh0; ksh < msh1; ksh++) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijk = dij * dk;
                        for (ic = 0; ic < comp; ic++) {
                                poutij = outij + nijk*ic + ao_loc[ksh]-ao_loc[ksh0];
                                poutji = outji + nijk*ic + ao_loc[ksh]-ao_loc[ksh0];
                                pbij_r = bufr + offij + dijk*ic;
                                pbij_i = bufi + offij + dijk*ic;
                                pbji_r = bufr + offji + dijk*ic;
                                pbji_i = bufi + offji + dijk*ic;
        for (j = 0; j < dj; j++) {
                for (k = 0; k < dk; k++) {
                for (i = 0; i < di; i++) {
                        poutij[i*njk +k] = pbij_r[k*dij+i] + pbij_i[k*dij+i]*_Complex_I;
                        poutji[i*naok+k] = pbji_r[k*dij+i] - pbji_i[k*dij+i]*_Complex_I;
                } }
                poutij += naok;
                poutji += njk;
                pbij_r += di;
                pbij_i += di;
                pbji_r += di;
                pbji_i += di;
        }
                        }
                        offij += dijk * comp;
                        offji += dijk * comp;
                }
                outij += nijk * comp;
                outji += nijk * comp;
        }
}

/* ('...LM,kL,lM->...kl', int3c, exp_kL, exp_kL) */
void PBCnr3c_fill_kks2(int (*intor)(), double complex *out, int nkpts_ij,
                       int nkpts, int comp, int nimgs, int ish, int jsh,
                       double *buf, double *env_loc, double *Ls,
                       double *expkL_r, double *expkL_i, int *kptij_idx,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, PBCOpt *pbcopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
                _nr3c_fill_kk(intor, &sort3c_kks2_igtj, out,
                              nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                              buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                              shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        } else if (ip == jp) {
                _nr3c_fill_kk(intor, &sort3c_kks1, out,
                              nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                              buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                              shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        }
}

static void sort3c_ks1(double complex *out, double *bufr, double *bufi,
                       int *shls_slice, int *ao_loc, int nkpts, int comp,
                       int ish, int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t njk = naoj * naok;
        const size_t nijk = njk * naoi;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int dij = di * dj;
        const int dkmax = ao_loc[msh1] - ao_loc[msh0];
        const size_t dijmc = dij * dkmax * comp;
        out += (ip * naoj + jp) * naok;

        int i, j, k, kk, ksh, ic, dk, dijk;
        size_t off;
        double *pbr, *pbi;
        double complex *pout;

        for (kk = 0; kk < nkpts; kk++) {
                off = kk * dijmc;
                for (ksh = msh0; ksh < msh1; ksh++) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijk = dij * dk;
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + nijk*ic + ao_loc[ksh]-ao_loc[ksh0];
                                pbr = bufr + off + dijk*ic;
                                pbi = bufi + off + dijk*ic;
        for (j = 0; j < dj; j++) {
                for (k = 0; k < dk; k++) {
                for (i = 0; i < di; i++) {
                        pout[i*njk+k] = pbr[k*dij+i] + pbi[k*dij+i]*_Complex_I;
                } }
                pout += naok;
                pbr += di;
                pbi += di;
        }
                        }
                        off += dijk * comp;
                }
                out += nijk * comp;
        }
}

/* ('...LM,kL,kM->...k', int3c, exp_kL, exp_kL) */
static void _nr3c_fill_k(int (*intor)(), void (*fsort)(),
                         double complex *out, int nkpts_ij,
                         int nkpts, int comp, int nimgs, int ish, int jsh,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i, int *kptij_idx,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt, PBCOpt *pbcopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];

        const char TRANS_N = 'N';
        const double D1 = 1;

        jsh += jsh0;
        ish += ish0;
        int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        int dkmax = INTBUFMAX10 / dij;
        int kshloc[ksh1-ksh0+1];
        int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

        int i, m, msh0, msh1, dijmc, empty;
        size_t dijmk;
        int ksh, dk, iL, jL, jLcount;
        int shls[3];
        double *bufexp_r = buf;
        double *bufexp_i = bufexp_r + nimgs * nkpts;
        double *bufk_r = bufexp_i + nimgs * nkpts;
        double *bufk_i, *bufL, *pbuf, *cache;
        int (*fprescreen)();
        if (pbcopt != NULL) {
                fprescreen = pbcopt->fprescreen;
        } else {
                fprescreen = PBCnoscreen;
        }

        shls[0] = ish;
        shls[1] = jsh;
        for (m = 0; m < nkshloc; m++) {
                msh0 = kshloc[m];
                msh1 = kshloc[m+1];
                dkmax = ao_loc[msh1] - ao_loc[msh0];
                dijmc = dij * dkmax * comp;
                dijmk = dijmc * nkpts;
                bufk_i = bufk_r + dijmk;
                bufL   = bufk_i + dijmk;
                cache  = bufL   + nimgs * dijmc;
                for (i = 0; i < dijmk*OF_CMPLX; i++) {
                        bufk_r[i] = 0;
                }

                for (iL = 0; iL < nimgs; iL++) {
                        shift_bas(env_loc, env, Ls, iptrxyz, iL);
                        pbuf = bufL;
                        jLcount = 0;
                        for (jL = 0; jL < nimgs; jL++) {
                                shift_bas(env_loc, env, Ls, jptrxyz, jL);

        if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                for (ksh = msh0; ksh < msh1; ksh++) {
                        shls[2] = ksh;
                        if ((*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache)) {
                                empty = 0;
                        }
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        pbuf += dij*dk * comp;
                }
                // ('k,kL->kL', conj(expkL[iL]), expkL)
                for (i = 0; i < nkpts; i++) {
                        bufexp_r[i*nimgs+jLcount] = expkL_r[i*nimgs+jL] * expkL_r[i*nimgs+iL];
                        bufexp_r[i*nimgs+jLcount]+= expkL_i[i*nimgs+jL] * expkL_i[i*nimgs+iL];
                        bufexp_i[i*nimgs+jLcount] = expkL_i[i*nimgs+jL] * expkL_r[i*nimgs+iL];
                        bufexp_i[i*nimgs+jLcount]-= expkL_r[i*nimgs+jL] * expkL_i[i*nimgs+iL];
                }
                jLcount++;
        }
                        }
                        dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &jLcount,
                               &D1, bufL, &dijmc, bufexp_r, &nimgs, &D1, bufk_r, &dijmc);
                        dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &jLcount,
                               &D1, bufL, &dijmc, bufexp_i, &nimgs, &D1, bufk_i, &dijmc);

                } // iL in range(0, nimgs)
                (*fsort)(out, bufk_r, bufk_i, shls_slice, ao_loc,
                         nkpts, comp, ish, jsh, msh0, msh1);
        }
}
/* ('...LM,kL,kM->...k', int3c, exp_kL, exp_kL) */
void PBCnr3c_fill_ks1(int (*intor)(), double complex *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        _nr3c_fill_k(intor, sort3c_ks1, out,
                     nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                     buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                     shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
}

static void sort3c_ks2_igtj(double complex *out, double *bufr, double *bufi,
                            int *shls_slice, int *ao_loc, int nkpts, int comp,
                            int ish, int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int dkmax = ao_loc[msh1] - ao_loc[msh0];
        const size_t dijmc = dij * dkmax * comp;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (ao_loc[ish]*(ao_loc[ish]+1)/2-off0 + jp) * naok;

        int i, j, k, ij, kk, ksh, ic, dk, dijk;
        size_t off;
        double *pbr, *pbi;
        double complex *pout;

        for (kk = 0; kk < nkpts; kk++) {
                off = kk * dijmc;
                for (ksh = msh0; ksh < msh1; ksh++) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijk = dij * dk;
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + nijk*ic + ao_loc[ksh]-ao_loc[ksh0];
                                pbr = bufr + off + dijk*ic;
                                pbi = bufi + off + dijk*ic;
        for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                        ij = j * di + i;
                        for (k = 0; k < dk; k++) {
                                pout[j*naok+k] = pbr[k*dij+ij] + pbi[k*dij+ij]*_Complex_I;
                        }
                }
                pout += (i+ao_loc[ish]+1) * naok;
        }
                        }
                        off += dijk * comp;
                }
                out += nijk * comp;
        }
}
static void sort3c_ks2_ieqj(double complex *out, double *bufr, double *bufi,
                            int *shls_slice, int *ao_loc, int nkpts, int comp,
                            int ish, int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int dkmax = ao_loc[msh1] - ao_loc[msh0];
        const size_t dijmc = dij * dkmax * comp;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (ao_loc[ish]*(ao_loc[ish]+1)/2-off0 + jp) * naok;

        int i, j, k, ij, kk, ksh, ic, dk, dijk;
        size_t off;
        double *pbr, *pbi;
        double complex *pout;

        for (kk = 0; kk < nkpts; kk++) {
                off = kk * dijmc;
                for (ksh = msh0; ksh < msh1; ksh++) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijk = dij * dk;
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + nijk*ic + ao_loc[ksh]-ao_loc[ksh0];
                                pbr = bufr + off + dijk*ic;
                                pbi = bufi + off + dijk*ic;
        for (i = 0; i < di; i++) {
                for (j = 0; j <= i; j++) {
                        ij = j * di + i;
                        for (k = 0; k < dk; k++) {
                                pout[j*naok+k] = pbr[k*dij+ij] + pbi[k*dij+ij]*_Complex_I;
                        }
                }
                pout += (i+ao_loc[ish]+1) * naok;
        }
                        }
                        off += dijk * comp;
                }
                out += nijk * comp;
        }
}

/* ('...LM,kL,kM->...k', int3c, exp_kL, exp_kL) */
void PBCnr3c_fill_ks2(int (*intor)(), double complex *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
                _nr3c_fill_k(intor, &sort3c_ks2_igtj, out,
                             nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                             buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                             shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        } else if (ip == jp) {
                _nr3c_fill_k(intor, &sort3c_ks2_ieqj, out,
                             nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                             buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                             shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        }
}

static void sort3c_gs1(double *out, double *in, int *shls_slice, int *ao_loc,
                       int comp, int ish, int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t njk = naoj * naok;
        const size_t nijk = njk * naoi;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int dij = di * dj;
        const int dkmax = ao_loc[msh1] - ao_loc[msh0];
        out += (ip * naoj + jp) * naok;

        int i, j, k, ksh, ic, dk, dijk;
        double *pin, *pout;

        for (ksh = msh0; ksh < msh1; ksh++) {
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dijk = dij * dk;
                for (ic = 0; ic < comp; ic++) {
                        pout = out + nijk * ic + ao_loc[ksh]-ao_loc[ksh0];
                        pin = in + dijk * ic;
                        for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++) {
                                for (k = 0; k < dk; k++) {
                                        pout[i*njk+k] = pin[k*dij+i];
                                } }
                                pout += naok;
                                pin += di;
                        }
                }
                in += dijk * comp;
        }
}

static void _nr3c_fill_g(int (*intor)(), void (*fsort)(), double *out, int nkpts_ij,
                         int nkpts, int comp, int nimgs, int ish, int jsh,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i, int *kptij_idx,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt, PBCOpt *pbcopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];

        jsh += jsh0;
        ish += ish0;
        int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        int dkmax = INTBUFMAX10 / dij / 2 * MIN(IMGBLK,nimgs);
        int kshloc[ksh1-ksh0+1];
        int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

        int i, m, msh0, msh1, dijm;
        int ksh, dk, iL, jL, dijkc;
        int shls[3];

        int dijmc = dij * dkmax * comp;
        double *bufL = buf + dijmc;
        double *cache = bufL + dijmc;
        double *pbuf;
        int (*fprescreen)();
        if (pbcopt != NULL) {
                fprescreen = pbcopt->fprescreen;
        } else {
                fprescreen = PBCnoscreen;
        }

        shls[0] = ish;
        shls[1] = jsh;
        for (m = 0; m < nkshloc; m++) {
                msh0 = kshloc[m];
                msh1 = kshloc[m+1];
                dkmax = ao_loc[msh1] - ao_loc[msh0];
                dijm = dij * dkmax;
                dijmc = dijm * comp;
                for (i = 0; i < dijmc; i++) {
                        bufL[i] = 0;
                }

                for (iL = 0; iL < nimgs; iL++) {
                        shift_bas(env_loc, env, Ls, iptrxyz, iL);
                        for (jL = 0; jL < nimgs; jL++) {
                                shift_bas(env_loc, env, Ls, jptrxyz, jL);

        if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                pbuf = bufL;
                for (ksh = msh0; ksh < msh1; ksh++) {
                        shls[2] = ksh;
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijkc = dij*dk * comp;
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache)) {
                                for (i = 0; i < dijkc; i++) {
                                        pbuf[i] += buf[i];
                                }
                        }
                        pbuf += dijkc;
                }
        }
                        }
                } // iL in range(0, nimgs)
                (*fsort)(out, bufL, shls_slice, ao_loc, comp, ish, jsh, msh0, msh1);
        }
}
/* ('...LM->...', int3c) */
void PBCnr3c_fill_gs1(int (*intor)(), double *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
     _nr3c_fill_g(intor, &sort3c_gs1, out, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                  buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                  shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
}

static void sort3c_gs2_igtj(double *out, double *in, int *shls_slice, int *ao_loc,
                            int comp, int ish, int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (ao_loc[ish]*(ao_loc[ish]+1)/2-off0 + jp) * naok;

        int i, j, k, ij, ksh, ic, dk, dijk;
        double *pin, *pout;

        for (ksh = msh0; ksh < msh1; ksh++) {
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dijk = dij * dk;
                for (ic = 0; ic < comp; ic++) {
                        pout = out + nijk * ic + ao_loc[ksh]-ao_loc[ksh0];
                        pin = in + dijk * ic;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        ij = j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                pout[j*naok+k] = pin[k*dij+ij];
                                        }
                                }
                                pout += (i+ao_loc[ish]+1) * naok;
                        }
                }
                in += dijk * comp;
        }
}
static void sort3c_gs2_ieqj(double *out, double *in, int *shls_slice, int *ao_loc,
                            int comp, int ish, int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dij = di * di;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (ao_loc[ish]*(ao_loc[ish]+1)/2-off0 + jp) * naok;

        int i, j, k, ij, ksh, ic, dk, dijk;
        double *pin, *pout;

        for (ksh = msh0; ksh < msh1; ksh++) {
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dijk = dij * dk;
                for (ic = 0; ic < comp; ic++) {
                        pout = out + nijk * ic + ao_loc[ksh]-ao_loc[ksh0];
                        pin = in + dijk * ic;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        ij = j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                pout[j*naok+k] = pin[k*dij+ij];
                                        }
                                }
                                pout += (i+ao_loc[ish]+1) * naok;
                        }
                }
                in += dijk * comp;
        }
}

/* ('...LM->...', int3c) */
void PBCnr3c_fill_gs2(int (*intor)(), double *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
             _nr3c_fill_g(intor, &sort3c_gs2_igtj, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        } else if (ip == jp) {
             _nr3c_fill_g(intor, &sort3c_gs2_ieqj, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        }
}

int PBCsizeof_env(int *shls_slice,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        int ish, ia, np, nc;
        int nenv = 0;
        for (ish = ish0; ish < ish1; ish++) {
                ia = bas[ATOM_OF +ish*BAS_SLOTS];
                nenv = MAX(atm[PTR_COORD+ia*ATM_SLOTS]+3, nenv);
                np = bas[NPRIM_OF+ish*BAS_SLOTS];
                nc = bas[NCTR_OF +ish*BAS_SLOTS];
                nenv = MAX(bas[PTR_EXP  +ish*BAS_SLOTS]+np, nenv);
                nenv = MAX(bas[PTR_COEFF+ish*BAS_SLOTS]+np*nc, nenv);
        }
        return nenv;
}

void PBCnr3c_drv(int (*intor)(), void (*fill)(), double complex *eri,
                 int nkpts_ij, int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        double *expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
        double *expkL_i = expkL_r + nimgs*nkpts;
        int i;
        for (i = 0; i < nimgs*nkpts; i++) {
                expkL_r[i] = creal(expkL[i]);
                expkL_i[i] = cimag(expkL[i]);
        }

        size_t count;
        if (fill == &PBCnr3c_fill_kks1 || fill == &PBCnr3c_fill_kks2) {
                int dijk =(GTOmax_shell_dim(ao_loc, shls_slice+0, 1) *
                           GTOmax_shell_dim(ao_loc, shls_slice+2, 1) *
                           GTOmax_shell_dim(ao_loc, shls_slice+4, 1));
                count = nkpts*nkpts * OF_CMPLX +
                        nkpts*MIN(nimgs,IMGBLK) * OF_CMPLX + nimgs;
// MAX(INTBUFMAX, dijk) to ensure buffer is enough for at least one (i,j,k) shell
                count*= MAX(INTBUFMAX, dijk) * comp;
        } else {
                count = (nkpts * OF_CMPLX + nimgs) * INTBUFMAX10 * comp;
                count+= nimgs * nkpts * OF_CMPLX;
        }
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int ish, jsh, ij;
        double *env_loc = malloc(sizeof(double)*nenv);
        memcpy(env_loc, env, sizeof(double)*nenv);
        double *buf = malloc(sizeof(double)*(count+cache_size));
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                (*fill)(intor, eri, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                        buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                        shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        }
        free(buf);
        free(env_loc);
}
        free(expkL_r);
}


static void sort2c_ks1(double complex *out, double *bufr, double *bufi,
                       int *shls_slice, int *ao_loc, int nkpts, int comp,
                       int jsh, int msh0, int msh1)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nij = naoi * naoj;

        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int dimax = ao_loc[msh1] - ao_loc[msh0];
        const size_t dmjc = dimax * dj * comp;
        out += jp;

        int i, j, kk, ish, ic, di, dij;
        size_t off;
        double *pbr, *pbi;
        double complex *pout;

        for (kk = 0; kk < nkpts; kk++) {
                off = kk * dmjc;
                for (ish = msh0; ish < msh1; ish++) {
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dij = di * dj;
                        for (ic = 0; ic < comp; ic++) {
                                pout = out + nij*ic + naoj*(ao_loc[ish]-ao_loc[ish0]);
                                pbr = bufr + off + dij*ic;
                                pbi = bufi + off + dij*ic;
        for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        pout[i*naoj+j] = pbr[j*di+i] + pbi[j*di+i]*_Complex_I;
                }
        }
                        }
                        off += dij * comp;
                }
                out += nij * comp;
        }
}
static void _nr2c_fill(int (*intor)(), double complex *out,
                       int nkpts, int comp, int nimgs, int jsh, int ish0,
                       double *buf, double *env_loc, double *Ls,
                       double *expkL_r, double *expkL_i,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, PBCOpt *pbcopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];

        const char TRANS_N = 'N';
        const double D1 = 1;
        const double D0 = 0;

        ish0 += shls_slice[0];
        jsh += jsh0;
        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dimax = INTBUFMAX10 / dj;
        int ishloc[ish1-ish0+1];
        int nishloc = shloc_partition(ishloc, ao_loc, ish0, ish1, dimax);

        int m, msh0, msh1, dmjc, ish, di, empty;
        int jL;
        int shls[2];
        double *bufk_r = buf;
        double *bufk_i, *bufL, *pbuf, *cache;

        shls[1] = jsh;
        for (m = 0; m < nishloc; m++) {
                msh0 = ishloc[m];
                msh1 = ishloc[m+1];
                dimax = ao_loc[msh1] - ao_loc[msh0];
                dmjc = dj * dimax * comp;
                bufk_i = bufk_r + dmjc * nkpts;
                bufL   = bufk_i + dmjc * nkpts;
                cache  = bufL   + dmjc * nimgs;

                pbuf = bufL;
                for (jL = 0; jL < nimgs; jL++) {
                        shift_bas(env_loc, env, Ls, jptrxyz, jL);
                        for (ish = msh0; ish < msh1; ish++) {
                                shls[0] = ish;
                                di = ao_loc[ish+1] - ao_loc[ish];
                                if ((*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                                             env_loc, cintopt, cache)) {
                                        empty = 0;
                                }
                                pbuf += di * dj * comp;
                        }
                }
                dgemm_(&TRANS_N, &TRANS_N, &dmjc, &nkpts, &nimgs,
                       &D1, bufL, &dmjc, expkL_r, &nimgs, &D0, bufk_r, &dmjc);
                dgemm_(&TRANS_N, &TRANS_N, &dmjc, &nkpts, &nimgs,
                       &D1, bufL, &dmjc, expkL_i, &nimgs, &D0, bufk_i, &dmjc);

                sort2c_ks1(out, bufk_r, bufk_i, shls_slice, ao_loc,
                           nkpts, comp, jsh, msh0, msh1);
        }
}

/* ('...M,kL->...k', int3c, exp_kL, exp_kL) */
void PBCnr2c_fill_ks1(int (*intor)(), double complex *out,
                      int nkpts, int comp, int nimgs, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        _nr2c_fill(intor, out, nkpts, comp, nimgs, jsh, 0,
                   buf, env_loc, Ls, expkL_r, expkL_i, shls_slice, ao_loc,
                   cintopt, pbcopt, atm, natm, bas, nbas, env);
}

void PBCnr2c_fill_ks2(int (*intor)(), double complex *out,
                      int nkpts, int comp, int nimgs, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        _nr2c_fill(intor, out, nkpts, comp, nimgs, jsh, jsh,
                   buf, env_loc, Ls, expkL_r, expkL_i, shls_slice, ao_loc,
                   cintopt, pbcopt, atm, natm, bas, nbas, env);
}

void PBCnr2c_drv(int (*intor)(), void (*fill)(), double complex *out,
                 int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int njsh = jsh1 - jsh0;
        double *expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
        double *expkL_i = expkL_r + nimgs*nkpts;
        int i;
        for (i = 0; i < nimgs*nkpts; i++) {
                expkL_r[i] = creal(expkL[i]);
                expkL_i[i] = cimag(expkL[i]);
        }
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int jsh;
        double *env_loc = malloc(sizeof(double)*nenv);
        memcpy(env_loc, env, sizeof(double)*nenv);
        size_t count = nkpts * OF_CMPLX + nimgs;
        double *buf = malloc(sizeof(double)*(count*INTBUFMAX10*comp+cache_size));
#pragma omp for schedule(dynamic)
        for (jsh = 0; jsh < njsh; jsh++) {
                (*fill)(intor, out, nkpts, comp, nimgs, jsh,
                        buf, env_loc, Ls, expkL_r, expkL_i,
                        shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env);
        }
        free(buf);
        free(env_loc);
}
        free(expkL_r);
}

