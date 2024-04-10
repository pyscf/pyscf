/* Copyright 2021-2024 The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include <assert.h>
#include <string.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "pbc/optimizer.h"
#include "pbc/fill_ints.h"
#include "pbc/neighbor_list.h"
#include "np_helper/np_helper.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2
#define MAX_THREADS     256

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

static void _nr3c_screened_fill_g(int (*intor)(), void (*fsort)(), double *out, int nkpts_ij,
                         int nkpts, int comp, int nimgs, int ish, int jsh,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i, int *kptij_idx,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt, PBCOpt *pbcopt,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         NeighborList** neighbor_list)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
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
        int ksh, dk, iL, jL, dijkc, ksh_off, jsh_off;
        int shls[3];

        int nshi = ish1 - ish0;
        int nshj = jsh1 - jsh0;
        int nshij = nshi + nshj;
        int idx_i, idx_j;

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
        jsh_off = jsh - nshi;
        NeighborList *nl0 = *neighbor_list;
        NeighborPair *np0_ki, *np0_kj;
        for (m = 0; m < nkshloc; m++) {
                msh0 = kshloc[m];
                msh1 = kshloc[m+1];
                dkmax = ao_loc[msh1] - ao_loc[msh0];
                dijm = dij * dkmax;
                dijmc = dijm * comp;
                for (i = 0; i < dijmc; i++) {
                    bufL[i] = 0;
                }

                pbuf = bufL;
                for (ksh = msh0; ksh < msh1; ksh++){
                    shls[2] = ksh;
                    ksh_off = ksh - nshij;
                    dk = ao_loc[ksh+1] - ao_loc[ksh];
                    dijkc = dij*dk * comp;
                    np0_ki = (nl0->pairs)[ksh_off*nshi + ish];
                    np0_kj = (nl0->pairs)[ksh_off*nshj + jsh_off];
                    if (np0_ki->nimgs > 0 && np0_kj->nimgs > 0) { 
                        for (idx_i = 0; idx_i < np0_ki->nimgs; idx_i++){
                            iL = (np0_ki->Ls_list)[idx_i];
                            shift_bas(env_loc, env, Ls, iptrxyz, iL);
                            for (idx_j = 0; idx_j < np0_kj->nimgs; idx_j++){
                                jL = (np0_kj->Ls_list)[idx_j];
                                shift_bas(env_loc, env, Ls, jptrxyz, jL);

                                if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                                    if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                        env_loc, cintopt, cache)) {
                                        for (i = 0; i < dijkc; i++) {
                                            pbuf[i] += buf[i];
                                        }
                                    }
                                }
                            } 

                        }
                    }
                    pbuf += dijkc;
                }

                (*fsort)(out, bufL, shls_slice, ao_loc, comp, ish, jsh, msh0, msh1);
        }
}

static void _nr3c_screened_sum_auxbas_fill_g(int (*intor)(), void (*fsort)(), double *out, int nkpts_ij,
                         int nkpts, int comp, int nimgs, int ish, int jsh,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i, int *kptij_idx,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt, PBCOpt *pbcopt,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         NeighborList** neighbor_list)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
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
        //int kshloc[ksh1-ksh0+1];
        //int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

        int i, k, ic;
        int ksh, dk, dijk, iL, jL, ksh_off, jsh_off;
        int shls[3];

        int nshi = ish1 - ish0;
        int nshj = jsh1 - jsh0;
        int nshij = nshi + nshj;
        int idx_i, idx_j;

        int dijmc = dij * dkmax * comp;
        double *bufL = buf + dijmc;
        double *cache = bufL + dijmc;
        double *pbuf, *pbufL;
        int (*fprescreen)();
        if (pbcopt != NULL) {
                fprescreen = pbcopt->fprescreen;
        } else {
                fprescreen = PBCnoscreen;
        }

        shls[0] = ish;
        shls[1] = jsh;
        jsh_off = jsh - nshi;
        NeighborList *nl0 = *neighbor_list;
        NeighborPair *np0_ki, *np0_kj;

        int dijc = dij * comp;
        for (i = 0; i < dijc; i++) {
            bufL[i] = 0;
        }

        for (ksh = ksh0; ksh < ksh1; ksh++){
            dk = ao_loc[ksh+1] - ao_loc[ksh];
            assert(dk < dkmax);
            dijk = dij * dk;
            shls[2] = ksh;
            ksh_off = ksh - nshij;
            np0_ki = (nl0->pairs)[ksh_off*nshi + ish];
            np0_kj = (nl0->pairs)[ksh_off*nshj + jsh_off];
            if (np0_ki->nimgs > 0 && np0_kj->nimgs > 0) { 
                for (idx_i = 0; idx_i < np0_ki->nimgs; idx_i++){
                    iL = (np0_ki->Ls_list)[idx_i];
                    shift_bas(env_loc, env, Ls, iptrxyz, iL);
                    for (idx_j = 0; idx_j < np0_kj->nimgs; idx_j++){
                        jL = (np0_kj->Ls_list)[idx_j];
                        shift_bas(env_loc, env, Ls, jptrxyz, jL);

                        if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                            if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                env_loc, cintopt, cache)) {
                                for (ic = 0; ic < comp; ic++) {
                                    pbufL = bufL + ic * dij;
                                    pbuf = buf + ic * dijk;
                                    for (k = 0; k < dk; k++) {
                                        for (i = 0; i < dij; i++) {
                                            pbufL[i] += pbuf[i];
                                        }
                                        pbuf += dij;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        (*fsort)(out, bufL, shls_slice, ao_loc, comp, ish, jsh);
}

void PBCnr3c_screened_fill_gs1(int (*intor)(), double *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      NeighborList** neighbor_list)
{
     _nr3c_screened_fill_g(intor, &sort3c_gs1, out, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                  buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                  shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
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
        const size_t off0 = ((size_t)ao_loc[ish0]) * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ((size_t)ao_loc[ish1]) * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * naok;

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

void sort2c_gs2_igtj(double *out, double *in, int *shls_slice, int *ao_loc,
                     int comp, int ish, int jsh)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const size_t off0 = ((size_t)ao_loc[ish0]) * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ((size_t)ao_loc[ish1]) * (ao_loc[ish1] + 1) / 2 - off0;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += ((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp;

        int i, j, ic;
        double *pin, *pout;

        for (ic = 0; ic < comp; ic++) {
                pout = out + nij * ic;
                pin = in + dij * ic;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pout[j] = pin[j*di+i];
                        }
                        pout += (i+ao_loc[ish]+1);
                }
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
        const size_t off0 = ((size_t)ao_loc[ish0]) * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ((size_t)ao_loc[ish1]) * (ao_loc[ish1] + 1) / 2 - off0;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dij = di * di;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += (((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * naok;

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

void sort2c_gs2_ieqj(double *out, double *in, int *shls_slice, int *ao_loc,
                     int comp, int ish, int jsh)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const size_t off0 = ((size_t)ao_loc[ish0]) * (ao_loc[ish0] + 1) / 2;
        const size_t nij = ((size_t)ao_loc[ish1]) * (ao_loc[ish1] + 1) / 2 - off0;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dij = di * di;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += ((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp;

        int i, j, ic;
        double *pin, *pout;

        for (ic = 0; ic < comp; ic++) {
                pout = out + nij * ic;
                pin = in + dij * ic;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                pout[j] = pin[j*di+i];
                        }
                        pout += (i+ao_loc[ish]+1);
                }
        }
}

void sort2c_gs1(double *out, double *in, int *shls_slice, int *ao_loc,
                int comp, int ish, int jsh)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nij = naoi * naoj;
        out += ip * naoj + jp;

        int i, j, ic;
        double *pin, *pout;

        for (ic = 0; ic < comp; ic++) {
                pout = out + nij * ic;
                pin = in + dij * ic;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pout[j] = pin[j*di+i];
                        }
                        pout += naoj;
                }
        }
}

void PBCnr3c_screened_fill_gs2(int (*intor)(), double *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      NeighborList** neighbor_list)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
             _nr3c_screened_fill_g(intor, &sort3c_gs2_igtj, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
        } else if (ip == jp) {
             _nr3c_screened_fill_g(intor, &sort3c_gs2_ieqj, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
        }
}

void PBCnr3c_screened_sum_auxbas_fill_gs1(int (*intor)(), double *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      NeighborList** neighbor_list)
{
        _nr3c_screened_sum_auxbas_fill_g(intor, &sort2c_gs1, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
}

void PBCnr3c_screened_sum_auxbas_fill_gs2(int (*intor)(), double *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      NeighborList** neighbor_list)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
             _nr3c_screened_sum_auxbas_fill_g(intor, &sort2c_gs2_igtj, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
        } else if (ip == jp) {
             _nr3c_screened_sum_auxbas_fill_g(intor, &sort2c_gs2_ieqj, out,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
        }
}

static void contract_3c1e_ipik_dm_gs1(double *grad, double* dm, double *eri,
                                      int *shls, int *ao_loc, int *atm, int natm,
                                      int *bas, int nbas, int comp, int nao)
{
    const int ish = shls[0];
    const int jsh = shls[1];
    const int ksh = shls[2];

    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const size_t i0 = ao_loc[ish];
    const size_t j0 = ao_loc[jsh] - nao;

    const int ia = bas[ATOM_OF+ish*BAS_SLOTS];
    const int ka = bas[ATOM_OF+ksh*BAS_SLOTS] - 2*natm;

    int i, j, ic;
    double *ptr_eri, *ptr_dm;
    double *dm0 = dm + (i0 * nao + j0);
    double ipi_dm[comp];
    for (ic = 0; ic < comp; ic++) {
        ipi_dm[ic] = 0;
        ptr_dm = dm0;
        ptr_eri = eri + dij * ic;
        for (i = 0; i < di; i++) {
            for (j = 0; j < dj; j++) {
                ipi_dm[ic] += ptr_eri[j*di+i] * ptr_dm[j];
            }
            ptr_dm += nao;
        }
    }

    for (ic = 0; ic < comp; ic++) {
        grad[ia*comp+ic] += ipi_dm[ic];
        grad[ka*comp+ic] -= ipi_dm[ic];
    }
}

static void _nr3c1e_screened_nuc_grad_fill_g(int (*intor)(), void (*fcontract)(),
            double *grad, double *dm, int nkpts_ij, int nkpts,
            int comp, int nimgs, int ish, int jsh,
            double *buf, double *env_loc, double *Ls,
            double *expkL_r, double *expkL_i, int *kptij_idx,
            int *shls_slice, int *ao_loc,
            CINTOpt *cintopt, PBCOpt *pbcopt,
            int *atm, int natm, int *bas, int nbas, double *env, int nao,
            NeighborList** neighbor_list)
{
    const int ish0 = shls_slice[0];
    //const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    //const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];

    ish += ish0;
    jsh += jsh0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    int dkmax = INTBUFMAX10 / dij / 2 * MIN(IMGBLK,nimgs);
    //int kshloc[ksh1-ksh0+1];
    //int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, k, ic;
    int ksh, dk, dijk, iL, jL, ksh_off, jsh_off;
    int shls[3];

    int idx_i, idx_j;

    int dijc = dij * comp;
    int dijmc = dijc * dkmax;
    double *bufL = buf + dijmc;
    double *cache = bufL + dijc;
    double *pbuf, *pbufL;
    int (*fprescreen)();
    if (pbcopt != NULL) {
            fprescreen = pbcopt->fprescreen;
    } else {
            fprescreen = PBCnoscreen;
    }

    shls[0] = ish;
    shls[1] = jsh;
    jsh_off = jsh - nbas;
    NeighborList *nl0 = *neighbor_list;
    NeighborPair *np0_ki, *np0_kj;

    for (ksh = ksh0; ksh < ksh1; ksh++){
        dk = ao_loc[ksh+1] - ao_loc[ksh];
        assert(dk < dkmax);
        dijk = dij * dk;
        shls[2] = ksh;
        ksh_off = ksh - nbas*2;
        np0_ki = (nl0->pairs)[ksh_off*nbas + ish];
        np0_kj = (nl0->pairs)[ksh_off*nbas + jsh_off];
        if (np0_ki->nimgs > 0 && np0_kj->nimgs > 0) {
            for (i = 0; i < dijc; i++) {
                bufL[i] = 0;
            }
            for (idx_i = 0; idx_i < np0_ki->nimgs; idx_i++){
                iL = (np0_ki->Ls_list)[idx_i];
                shift_bas(env_loc, env, Ls, iptrxyz, iL);
                for (idx_j = 0; idx_j < np0_kj->nimgs; idx_j++){
                    jL = (np0_kj->Ls_list)[idx_j];
                    shift_bas(env_loc, env, Ls, jptrxyz, jL);

                    if ((*fprescreen)(shls, pbcopt, atm, bas, env_loc)) {
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache))
                        {
                            for (ic = 0; ic < comp; ic++) {
                                pbufL = bufL + ic * dij;
                                pbuf = buf + ic * dijk;
                                for (k = 0; k < dk; k++) {
                                    for (i = 0; i < dij; i++) {
                                        pbufL[i] += pbuf[i];
                                    }
                                    pbuf += dij;
                                }
                            }
                        }
                    }
                }
            }
            (*fcontract)(grad, dm, bufL, shls, ao_loc, atm, natm, bas, nbas, comp, nao);
        }
    }
}

void PBCnr3c1e_screened_nuc_grad_fill_gs1(int (*intor)(), double *out, double* dm,
                      int nkpts_ij, int nkpts, int comp, int nimgs, int ish, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, PBCOpt *pbcopt,
                      int *atm, int natm, int *bas, int nbas, double *env, int nao,
                      NeighborList** neighbor_list)
{
        _nr3c1e_screened_nuc_grad_fill_g(intor, &contract_3c1e_ipik_dm_gs1, out, dm,
                          nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                          buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                          shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, nao, neighbor_list);
}

void PBCnr3c_screened_drv(int (*intor)(), void (*fill)(), double complex *eri,
                 int nkpts_ij, int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv,
                 NeighborList** neighbor_list)
{
        assert(neighbor_list != NULL);
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
        count = (nkpts * OF_CMPLX + nimgs) * INTBUFMAX10 * comp;
        count+= nimgs * nkpts * OF_CMPLX;
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int ish, jsh, ij;
        double *env_loc = malloc(sizeof(double)*nenv);
        NPdcopy(env_loc, env, nenv);
        double *buf = malloc(sizeof(double)*(count+cache_size));
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                (*fill)(intor, eri, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                        buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                        shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
        }
        free(buf);
        free(env_loc);
}
        free(expkL_r);
}

void PBCnr3c_screened_sum_auxbas_drv(int (*intor)(), void (*fill)(), double complex *eri,
                 int nkpts_ij, int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv,
                 NeighborList** neighbor_list)
{
        assert(neighbor_list != NULL);
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        double *expkL_r=NULL, *expkL_i=NULL;
        //expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
        //expkL_i = expkL_r + nimgs*nkpts;
        //int i;
        //for (i = 0; i < nimgs*nkpts; i++) {
        //        expkL_r[i] = creal(expkL[i]);
        //        expkL_i[i] = cimag(expkL[i]);
        //}

        size_t count;
        count = (nkpts * OF_CMPLX + nimgs) * INTBUFMAX10 * comp;
        count+= nimgs * nkpts * OF_CMPLX;
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int ish, jsh, ij;
        double *env_loc = malloc(sizeof(double)*nenv);
        NPdcopy(env_loc, env, nenv);
        double *buf = malloc(sizeof(double)*(count+cache_size));
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                (*fill)(intor, eri, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                        buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                        shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, neighbor_list);
        }
        free(buf);
        free(env_loc);
}
        //free(expkL_r);
}

void PBCnr3c1e_screened_nuc_grad_drv(int (*intor)(), void (*fill)(), 
                 double* grad, double* dm,
                 int nkpts_ij, int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt, PBCOpt *pbcopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv, int nao,
                 NeighborList** neighbor_list)
{
        assert(neighbor_list != NULL);
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        double *expkL_r=NULL, *expkL_i=NULL;
        //double *expkL_r = malloc(sizeof(double) * nimgs*nkpts * OF_CMPLX);
        //double *expkL_i = expkL_r + nimgs*nkpts;
        //int i;
        //for (i = 0; i < nimgs*nkpts; i++) {
        //        expkL_r[i] = creal(expkL[i]);
        //        expkL_i[i] = cimag(expkL[i]);
        //}

        size_t count;
        count = (nkpts * OF_CMPLX + nimgs) * INTBUFMAX10 * comp;
        count+= nimgs * nkpts * OF_CMPLX;
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);

        double *gradbufs[MAX_THREADS];
#pragma omp parallel
{
        int ish, jsh, ij;
        double *env_loc = malloc(sizeof(double)*nenv);
        NPdcopy(env_loc, env, nenv);
        double *grad_loc;
        int thread_id = omp_get_thread_num();
        if (thread_id == 0) {
                grad_loc = grad;
        } else {
                grad_loc = calloc(natm*comp, sizeof(double));
        }
        gradbufs[thread_id] = grad_loc;

        double *buf = malloc(sizeof(double)*(count+cache_size));
        #pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                (*fill)(intor, grad_loc, dm, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                        buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                        shls_slice, ao_loc, cintopt, pbcopt, atm, natm, bas, nbas, env, nao, neighbor_list);
        }
        free(buf);
        free(env_loc);

        NPomp_dsum_reduce_inplace(gradbufs, natm*comp);
        if (thread_id != 0) {
                free(grad_loc);
        }
}
        //free(expkL_r);
}


static int _nr2c_screened_fill(
                int (*intor)(), double complex *out,
                int nkpts, int comp, int nimgs, int jsh, int ish0,
                double *buf, double *env_loc, double *Ls,
                double *expkL_r, double *expkL_i,
                int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                int *atm, int natm, int *bas, int nbas, double *env,
                NeighborList** neighbor_list)
{
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nshi = ish1 - shls_slice[0];
        const int nshj = jsh1 - jsh0;

        const double D1 = 1;
        const int I1 = 1;

        ish0 += shls_slice[0];
        jsh += jsh0;
        int jsh_off = jsh - nshi;
        int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dimax = INTBUFMAX10 / dj;
        int ishloc[ish1-ish0+1];
        int nishloc = shloc_partition(ishloc, ao_loc, ish0, ish1, dimax);

        int m, msh0, msh1, dijc, dmjc, ish, di, empty;
        int jL, idx_j;
        int shls[2];
        double *bufk_r = buf;
        double *bufk_i, *bufL, *pbufk_r, *pbufk_i, *cache;

        NeighborList *nl0 = *neighbor_list;
        NeighborPair *np0;

        shls[1] = jsh;
        for (m = 0; m < nishloc; m++) {
                msh0 = ishloc[m];
                msh1 = ishloc[m+1];
                dimax = ao_loc[msh1] - ao_loc[msh0];
                dmjc = dj * dimax * comp;
                bufk_i = bufk_r + dmjc * nkpts;
                bufL   = bufk_i + dmjc * nkpts;
                cache  = bufL   + dmjc;

                memset(bufk_r, 0, 2*dmjc*nkpts*sizeof(double));
                pbufk_r = bufk_r;
                pbufk_i = bufk_i;
                for (ish = msh0; ish < msh1; ish++) {
                        shls[0] = ish;
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dijc = di * dj * comp;
                        np0 = (nl0->pairs)[ish*nshj + jsh_off];
                        if (np0->nimgs > 0) {
                                for (idx_j = 0; idx_j < np0->nimgs; idx_j++){
                                        jL = (np0->Ls_list)[idx_j];
                                        shift_bas(env_loc, env, Ls, jptrxyz, jL);
                                        if ((*intor)(bufL, NULL, shls, atm, natm, bas, nbas,
                                                     env_loc, cintopt, cache)) {
                                                empty = 0;
                                                dger_(&dijc, &nkpts, &D1, bufL, &I1,
                                                      expkL_r+jL, &nimgs, pbufk_r, &dmjc);
                                                dger_(&dijc, &nkpts, &D1, bufL, &I1,
                                                      expkL_i+jL, &nimgs, pbufk_i, &dmjc);
                                        }
                                }
                        }
                        pbufk_r += dijc;
                        pbufk_i += dijc;
                }
                sort2c_ks1(out, bufk_r, bufk_i, shls_slice, ao_loc,
                           nkpts, comp, jsh, msh0, msh1);
        }
        return !empty;
}

void PBCnr2c_screened_fill_ks1(int (*intor)(), double complex *out,
                      int nkpts, int comp, int nimgs, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      NeighborList** neighbor_list)
{
        _nr2c_screened_fill(intor, out, nkpts, comp, nimgs, jsh, 0,
                   buf, env_loc, Ls, expkL_r, expkL_i, shls_slice, ao_loc,
                   cintopt, atm, natm, bas, nbas, env, neighbor_list);
}

void PBCnr2c_screened_fill_ks2(int (*intor)(), double complex *out,
                      int nkpts, int comp, int nimgs, int jsh,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      NeighborList** neighbor_list)
{
        _nr2c_screened_fill(intor, out, nkpts, comp, nimgs, jsh, jsh,
                   buf, env_loc, Ls, expkL_r, expkL_i, shls_slice, ao_loc,
                   cintopt, atm, natm, bas, nbas, env, neighbor_list);
}

void PBCnr2c_screened_drv(int (*intor)(), void (*fill)(), double complex *out,
                 int nkpts, int comp, int nimgs,
                 double *Ls, double complex *expkL,
                 int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                 int *atm, int natm, int *bas, int nbas, double *env, int nenv,
                 NeighborList** neighbor_list)
{
        assert(neighbor_list != NULL);
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
        NPdcopy(env_loc, env, nenv);
        size_t count = (nkpts+1) * OF_CMPLX;
        double *buf = malloc(sizeof(double)*(count*INTBUFMAX10*comp+cache_size));
#pragma omp for schedule(dynamic)
        for (jsh = 0; jsh < njsh; jsh++) {
                (*fill)(intor, out, nkpts, comp, nimgs, jsh,
                        buf, env_loc, Ls, expkL_r, expkL_i,
                        shls_slice, ao_loc, cintopt, atm, natm, bas, nbas, env,
                        neighbor_list);
        }
        free(buf);
        free(env_loc);
}
        free(expkL_r);
}
