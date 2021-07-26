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
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include <assert.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2

#define ABS(X)          ((X>0)?(X):(-X))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);
double get_dsqure(double *, double *);
void get_rc(double *, double *, double *, double, double);
size_t max_shlsize(int *, int);

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

/* The following functions should be used solely for pyscf.pbc.df.rsdf.RSDF
*/

// non-split basis implementation
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

static void _nr3c_g(int (*intor)(), void (*fsort)(), double *out,
                    int comp, int nimgs,
                    int ish, int jsh,
                    double *buf, double *env_loc, double *Ls,
                    int *shls_slice, int *ao_loc,
                    CINTOpt *cintopt,
                    int *refuniqshl_map, int *auxuniqshl_map,
                    int nbasauxuniq, double *uniqexp,
                    double *uniq_dcut2s, double dcut_binsize,
                    double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
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
    int kptrxyz;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkaomax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dkmax = INTBUFMAX10 / dij;
    int kshloc[ksh1-ksh0+1];
    int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, m, msh0, msh1, dijm;
    int ksh, dk, dijkc;
    int iL, jL;
    int shls[3];

    int dijmc = dij * dkmax * comp;
    double *bufL = buf + dij*dkaomax;
    double *cache = bufL + dijmc;
    double *pbuf;

    const double omega = ABS(env_loc[PTR_RANGE_OMEGA]);

    shls[0] = ish;
    shls[1] = jsh;
// >>>>>>>>
    int Ish, Jsh, IJsh, Ksh, idij;
    Ish = refuniqshl_map[ish];
    Jsh = refuniqshl_map[jsh-nbas];
    IJsh = (Ish>=Jsh)?(Ish*(Ish+1)/2+Jsh):(Jsh*(Jsh+1)/2+Ish);
    const double *uniq_Rcut2s_IJ, *uniq_Rcut2s_K;
    uniq_Rcut2s_IJ = uniq_Rcut2s + uniqshlpr_dij_loc[IJsh] * nbasauxuniq;
    double *ri, *rj, *rk, rc[3];
    double dij2, dij2_cut, inv_d0, Rijk2, Rcut2, ei, ej;
    inv_d0 = 1./dcut_binsize;
    dij2_cut = uniq_dcut2s[IJsh];
    ei = uniqexp[Ish];
    ej = uniqexp[Jsh];
// <<<<<<<<
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
            ri = env_loc + iptrxyz;
            for (jL = 0; jL < nimgs; jL++) {
                shift_bas(env_loc, env, Ls, jptrxyz, jL);
                rj = env_loc + jptrxyz;
// >>>>>>>>
                dij2 = get_dsqure(ri, rj);
                if(dij2 > dij2_cut) {
                    continue;
                }
                idij = (int)(sqrt(dij2)*inv_d0);
                uniq_Rcut2s_K = uniq_Rcut2s_IJ + idij * nbasauxuniq;
// <<<<<<<<
                get_rc(rc, ri, rj, ei, ej);

                pbuf = bufL;
                for (ksh = msh0; ksh < msh1; ksh++) {
                    shls[2] = ksh;
                    dk = ao_loc[ksh+1] - ao_loc[ksh];
                    dijkc = dij*dk * comp;
                    Ksh = auxuniqshl_map[ksh-2*nbas];
                    Rcut2 = uniq_Rcut2s_K[Ksh];
                    kptrxyz = atm[PTR_COORD+bas[ATOM_OF+ksh*BAS_SLOTS]
                                  *ATM_SLOTS];
                    rk = env_loc + kptrxyz;
                    Rijk2 = get_dsqure(rc, rk);
                    if(Rijk2 < Rcut2) {
                        env_loc[PTR_RANGE_OMEGA] = 0.;
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache)) {
                            for (i = 0; i < dijkc; i++) {
                                pbuf[i] += buf[i];
                            }
                        }
                        env_loc[PTR_RANGE_OMEGA] = omega;
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache)) {
                            for (i = 0; i < dijkc; i++) {
                                pbuf[i] -= buf[i];
                            }
                        }
                    } // if Rcut
                    pbuf += dijkc;
                }
            } // jL
        } // iL
        (*fsort)(out, bufL, shls_slice, ao_loc, comp, ish, jsh, msh0, msh1);
    }
}

void PBCnr3c_gs2(int (*intor)(), double *out,
                 int comp, int nimgs,
                 int ish, int jsh,
                 double *buf, double *env_loc, double *Ls,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt,
                 int *refuniqshl_map, int *auxuniqshl_map,
                 int nbasauxuniq, double *uniqexp,
                 double *uniq_dcut2s, double dcut_binsize,
                 double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int ip = ish + shls_slice[0];
        int jp = jsh + shls_slice[2] - nbas;
        if (ip > jp) {
             _nr3c_g(intor, &sort3c_gs2_igtj, out,
                     comp, nimgs, ish, jsh,
                     buf, env_loc, Ls,
                     shls_slice, ao_loc, cintopt,
                     refuniqshl_map, auxuniqshl_map,
                     nbasauxuniq, uniqexp,
                     uniq_dcut2s, dcut_binsize,
                     uniq_Rcut2s, uniqshlpr_dij_loc,
                     atm, natm, bas, nbas, env);
        } else if (ip == jp) {
             _nr3c_g(intor, &sort3c_gs2_ieqj, out,
                     comp, nimgs, ish, jsh,
                     buf, env_loc, Ls,
                     shls_slice, ao_loc, cintopt,
                     refuniqshl_map, auxuniqshl_map,
                     nbasauxuniq, uniqexp,
                     uniq_dcut2s, dcut_binsize,
                     uniq_Rcut2s, uniqshlpr_dij_loc,
                     atm, natm, bas, nbas, env);
        }
}

void PBCnr3c_g_drv(int (*intor)(), void (*fill)(), double *out,
                       int comp, int nimgs,
                       double *Ls,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, char *shlpr_mask,
                       int *refuniqshl_map, int *auxuniqshl_map,
                       int nbasauxuniq, double *uniqexp,
                       double *uniq_dcut2s, double dcut_binsize,
                       double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                       int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;

        int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
        int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
        int dk = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
        int dijk = di*dj*dk;
        size_t count = (MAX(INTBUFMAX10, dijk) + dijk) * comp;
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
            if (!shlpr_mask[ij]) {
                continue;
            }
            (*fill)(intor, out, comp, nimgs,
                    ish, jsh,
                    buf, env_loc, Ls,
                    shls_slice, ao_loc, cintopt,
                    refuniqshl_map, auxuniqshl_map,
                    nbasauxuniq, uniqexp,
                    uniq_dcut2s, dcut_binsize,
                    uniq_Rcut2s, uniqshlpr_dij_loc,
                    atm, natm, bas, nbas, env);
        }
        free(buf);
        free(env_loc);
}
}

/* splitbas implementation starts here */
static void sort3c_gs2_spltbas_igtj(double *out, double *in, int *shls_slice,
                                    int *ao_loc, int comp, int ish, int jsh,
                                    int msh0, int msh1)
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
                        pout[j*naok+k] += pin[k*dij+ij];
                    }
                }
                pout += (i+ao_loc[ish]+1) * naok;
            }
        }
        in += dijk * comp;
    }
}

static void sort3c_gs2_spltbas_ieqj(double *out, double *in, int *shls_slice,
                                    int *ao_loc, int comp, int ish, int jsh,
                                    int msh0, int msh1)
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
                        pout[j*naok+k] += pin[k*dij+ij];
                    }
                }
                pout += (i+ao_loc[ish]+1) * naok;
            }
        }
        in += dijk * comp;
    }
}

static void _nr3c_g_spltbas(int (*intor)(), void (*fsort)(), double *out,
                    int comp, int nimgs,
                    int ish, int jsh, int ish_orig, int jsh_orig,
                    double *buf, double *env_loc, double *Ls,
                    int *shls_slice, int *ao_loc,
                    int *shls_slice_orig, int *ao_loc_orig,
                    CINTOpt *cintopt,
                    int *refuniqshl_map, int *auxuniqshl_map,
                    int nbasauxuniq, double *uniqexp,
                    double *uniq_dcut2s, double dcut_binsize,
                    double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];

    jsh_orig += shls_slice_orig[2];
    ish_orig += shls_slice_orig[0];
    const int msh_shift = shls_slice_orig[4] - ksh0;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int kptrxyz;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkaomax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dkmax = INTBUFMAX10 / dij;
    int kshloc[ksh1-ksh0+1];
    int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, m, msh0, msh1, dijm;
    int msh0_orig, msh1_orig;
    int ksh, dk, dijkc;
    int iL, jL;
    int shls[3];

    int dijmc = dij * dkmax * comp;
    double *bufL = buf + dij*dkaomax;
    double *cache = bufL + dijmc;
    double *pbuf;

    const double omega = ABS(env_loc[PTR_RANGE_OMEGA]);

    shls[0] = ish;
    shls[1] = jsh;
// >>>>>>>>
    int Ish, Jsh, IJsh, Ksh, idij;
    Ish = refuniqshl_map[ish];
    Jsh = refuniqshl_map[jsh-nbas];
    IJsh = (Ish>=Jsh)?(Ish*(Ish+1)/2+Jsh):(Jsh*(Jsh+1)/2+Ish);
    const double *uniq_Rcut2s_IJ, *uniq_Rcut2s_K;
    uniq_Rcut2s_IJ = uniq_Rcut2s + uniqshlpr_dij_loc[IJsh] * nbasauxuniq;
    double *ri, *rj, *rk, rc[3];
    double dij2, dij2_cut, inv_d0, Rijk2, Rcut2, ei, ej;
    inv_d0 = 1./dcut_binsize;
    dij2_cut = uniq_dcut2s[IJsh];
    ei = uniqexp[Ish];
    ej = uniqexp[Jsh];
// <<<<<<<<
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
            ri = env_loc + iptrxyz;
            for (jL = 0; jL < nimgs; jL++) {
                shift_bas(env_loc, env, Ls, jptrxyz, jL);
                rj = env_loc + jptrxyz;
// >>>>>>>>
                dij2 = get_dsqure(ri, rj);
                if(dij2 > dij2_cut) {
                    continue;
                }
                idij = (int)(sqrt(dij2)*inv_d0);
                uniq_Rcut2s_K = uniq_Rcut2s_IJ + idij * nbasauxuniq;
// <<<<<<<<
                get_rc(rc, ri, rj, ei, ej);

                pbuf = bufL;
                for (ksh = msh0; ksh < msh1; ksh++) {
                    shls[2] = ksh;
                    dk = ao_loc[ksh+1] - ao_loc[ksh];
                    dijkc = dij*dk * comp;
                    Ksh = auxuniqshl_map[ksh-2*nbas];
                    Rcut2 = uniq_Rcut2s_K[Ksh];
                    kptrxyz = atm[PTR_COORD+bas[ATOM_OF+ksh*BAS_SLOTS]
                                  *ATM_SLOTS];
                    rk = env_loc + kptrxyz;
                    Rijk2 = get_dsqure(rc, rk);
                    if(Rijk2 < Rcut2) {
                        env_loc[PTR_RANGE_OMEGA] = 0.;
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache)) {
                            for (i = 0; i < dijkc; i++) {
                                pbuf[i] += buf[i];
                            }
                        }
                        env_loc[PTR_RANGE_OMEGA] = omega;
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache)) {
                            for (i = 0; i < dijkc; i++) {
                                pbuf[i] -= buf[i];
                            }
                        }
                    } // if Rcut
                    pbuf += dijkc;
                }
            } // jL
        } // iL
        msh0_orig = msh0 + msh_shift;
        msh1_orig = msh1 + msh_shift;
        (*fsort)(out, bufL, shls_slice_orig, ao_loc_orig, comp,
                 ish_orig, jsh_orig, msh0_orig, msh1_orig);
    }
}

void PBCnr3c_gs2_spltbas(int (*intor)(), double *out,
                         int comp, int nimgs,
                         int ish, int jsh, int ish_orig, int jsh_orig,
                         double *buf, double *env_loc, double *Ls,
                         int *shls_slice, int *ao_loc,
                         int *shls_slice_orig, int *ao_loc_orig,
                         CINTOpt *cintopt,
                         int *refuniqshl_map, int *auxuniqshl_map,
                         int nbasauxuniq, double *uniqexp,
                         double *uniq_dcut2s, double dcut_binsize,
                         double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         int nbas_orig)
{
    int ip = ish_orig + shls_slice_orig[0];
    int jp = jsh_orig + shls_slice_orig[2] - nbas_orig;
    if (ip > jp) {
         _nr3c_g_spltbas(intor, &sort3c_gs2_spltbas_igtj, out,
                         comp, nimgs, ish, jsh, ish_orig, jsh_orig,
                         buf, env_loc, Ls,
                         shls_slice, ao_loc, shls_slice_orig, ao_loc_orig,
                         cintopt,
                         refuniqshl_map, auxuniqshl_map,
                         nbasauxuniq, uniqexp,
                         uniq_dcut2s, dcut_binsize,
                         uniq_Rcut2s, uniqshlpr_dij_loc,
                         atm, natm, bas, nbas, env);
    } else if (ip == jp) {
         _nr3c_g_spltbas(intor, &sort3c_gs2_spltbas_ieqj, out,
                         comp, nimgs, ish, jsh, ish_orig, jsh_orig,
                         buf, env_loc, Ls,
                         shls_slice, ao_loc, shls_slice_orig, ao_loc_orig,
                         cintopt,
                         refuniqshl_map, auxuniqshl_map,
                         nbasauxuniq, uniqexp,
                         uniq_dcut2s, dcut_binsize,
                         uniq_Rcut2s, uniqshlpr_dij_loc,
                         atm, natm, bas, nbas, env);
    }
}

void PBCnr3c_g_spltbas_drv(int (*intor)(), void (*fill)(), double *out,
                           int comp, int nimgs,
                           double *Ls,
                           int *shls_slice, int *ao_loc,
                           int *shls_slice_orig, int *ao_loc_orig,
                           int *shl_idx_orig, int *shl_loc_orig,
                           CINTOpt *cintopt, char *shlpr_mask,
                           int *refuniqshl_map, int *auxuniqshl_map,
                           int nbasauxuniq, double *uniqexp,
                           double *uniq_dcut2s, double dcut_binsize,
                           double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                           int *atm, int natm, int *bas, int nbas,
                           double *env, int nenv,
                           int *bas_orig, int nbas_orig, double *env_orig)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const int ish0_orig = shls_slice_orig[0];
    const int ish1_orig = shls_slice_orig[1];
    const int jsh0_orig = shls_slice_orig[2];
    const int jsh1_orig = shls_slice_orig[3];
    const int nish_orig = ish1_orig - ish0_orig;
    const int njsh_orig = jsh1_orig - jsh0_orig;

    int di = GTOmax_shell_dim(ao_loc, shls_slice_orig+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice_orig+2, 1);
    int dk = GTOmax_shell_dim(ao_loc, shls_slice_orig+4, 1);
    int dijk = di*dj*dk;
    size_t count = (MAX(INTBUFMAX10, dijk) + dijk) * comp;
    const int cache_size = GTOmax_cache_size(intor, shls_slice_orig, 3,
                                             atm, natm, bas_orig,
                                             nbas_orig, env_orig);

#pragma omp parallel
{
    int ish, jsh, ij;
    int ish_orig, jsh_orig, ij_orig, ish_loc, jsh_loc;
    double *env_loc = malloc(sizeof(double)*nenv);
    memcpy(env_loc, env, sizeof(double)*nenv);
    double *buf = malloc(sizeof(double)*(count+cache_size));
#pragma omp for schedule(dynamic)
    for (ij_orig = 0; ij_orig < nish_orig*njsh_orig; ij_orig++) {
        ish_orig = ij_orig / njsh_orig;
        jsh_orig = ij_orig % njsh_orig;
        for (ish_loc = shl_loc_orig[ish_orig];
             ish_loc < shl_loc_orig[ish_orig+1]; ish_loc++) {
            ish = shl_idx_orig[ish_loc];
            for (jsh_loc = shl_loc_orig[jsh_orig];
                 jsh_loc < shl_loc_orig[jsh_orig+1]; jsh_loc++) {
                jsh = shl_idx_orig[jsh_loc];
                ij = ish * njsh + jsh;
                if (!shlpr_mask[ij]) {
                        continue;
                }
                (*fill)(intor, out, comp, nimgs,
                        ish, jsh, ish_orig, jsh_orig,
                        buf, env_loc, Ls,
                        shls_slice, ao_loc,
                        shls_slice_orig, ao_loc_orig,
                        cintopt,
                        refuniqshl_map, auxuniqshl_map,
                        nbasauxuniq, uniqexp,
                        uniq_dcut2s, dcut_binsize,
                        uniq_Rcut2s, uniqshlpr_dij_loc,
                        atm, natm, bas, nbas, env,
                        nbas_orig);
            } // jsh_loc
        } // ish_loc
    } // ij_orig
    free(buf);
    free(env_loc);
}
}
