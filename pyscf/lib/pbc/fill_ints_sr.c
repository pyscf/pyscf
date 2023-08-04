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
#include <stdint.h>
#include <complex.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static double get_dsqure(double *ri, double *rj)
{
    double dx = ri[0]-rj[0];
    double dy = ri[1]-rj[1];
    double dz = ri[2]-rj[2];
    return dx*dx+dy*dy+dz*dz;
}
static void get_rc(double *rc, double *ri, double *rj, double ei, double ej) {
    double eij = ei+ej;
    rc[0] = (ri[0]*ei + rj[0]*ej) / eij;
    rc[1] = (ri[1]*ei + rj[1]*ej) / eij;
    rc[2] = (ri[2]*ei + rj[2]*ej) / eij;
}

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

// non-split basis implementation of j2c
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
static void _nr2c_k_fill(int (*intor)(), double complex *out,
                         int nkpts, int comp, int nimgs, int jsh, int ish0,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt,
                         const int *refuniqshl_map, const double *uniq_Rcut2s,
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

    int m, msh0, msh1, dmjc, ish, di;
    int jL;
    int shls[2];
    double *bufk_r = buf;
    double *bufk_i, *bufL, *pbuf, *pbuf2, *cache;
    int iptrxyz, dijc, ISH, JSH, IJSH, i;
    JSH = refuniqshl_map[jsh-jsh0];
    double *ri, *rj, Rij2, Rij2_cut;
    const double omega = fabs(env_loc[PTR_RANGE_OMEGA]);

    shls[1] = jsh;
    for (m = 0; m < nishloc; m++) {
        msh0 = ishloc[m];
        msh1 = ishloc[m+1];
        dimax = ao_loc[msh1] - ao_loc[msh0];
        dmjc = dj * dimax * comp;
        bufk_i = bufk_r + dmjc * nkpts;
        bufL   = bufk_i + dmjc * nkpts;
        pbuf2  = bufL   + dmjc * nimgs;
        cache  = pbuf2  + dmjc;

        pbuf = bufL;
        for (jL = 0; jL < nimgs; jL++) {
            shift_bas(env_loc, env, Ls, jptrxyz, jL);
            rj = env_loc + jptrxyz;
            for (ish = msh0; ish < msh1; ish++) {
                shls[0] = ish;
                di = ao_loc[ish+1] - ao_loc[ish];
                dijc = di * dj * comp;
                iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
                ri = env_loc + iptrxyz;
                Rij2 = get_dsqure(ri,rj);
                ISH = refuniqshl_map[ish];
                IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
                Rij2_cut = uniq_Rcut2s[IJSH];
                if (Rij2 < Rij2_cut) {
                    env_loc[PTR_RANGE_OMEGA] = 0.;
                    (*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                             env_loc, cintopt, cache);
                    env_loc[PTR_RANGE_OMEGA] = omega;
                    if ((*intor)(pbuf2, NULL, shls, atm, natm, bas, nbas,
                                 env_loc, cintopt, cache)) {
                        for (i = 0; i < dijc; ++i) {
                            pbuf[i] -= pbuf2[i];
                        }
                    }
                }
                else {
                    for (i = 0; i < dijc; ++i) {
                        pbuf[i] = 0.;
                    }
                } // if Rij2
                pbuf += dijc;
            } // ish
        } // jL
        dgemm_(&TRANS_N, &TRANS_N, &dmjc, &nkpts, &nimgs,
               &D1, bufL, &dmjc, expkL_r, &nimgs, &D0, bufk_r, &dmjc);
        dgemm_(&TRANS_N, &TRANS_N, &dmjc, &nkpts, &nimgs,
               &D1, bufL, &dmjc, expkL_i, &nimgs, &D0, bufk_i, &dmjc);

        sort2c_ks1(out, bufk_r, bufk_i, shls_slice, ao_loc,
                   nkpts, comp, jsh, msh0, msh1);
    }
}

void PBCsr2c_fill_ks1(int (*intor)(), double complex *out,
                         int nkpts, int comp, int nimgs, int jsh,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt,
                         const int *refuniqshl_map, const double *uniq_Rcut2s,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr2c_k_fill(intor, out, nkpts, comp, nimgs, jsh, 0,
                 buf, env_loc, Ls, expkL_r, expkL_i, shls_slice, ao_loc,
                 cintopt,
                 refuniqshl_map, uniq_Rcut2s,
                 atm, natm, bas, nbas, env);
}

void PBCsr2c_fill_ks2(int (*intor)(), double complex *out,
                         int nkpts, int comp, int nimgs, int jsh,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i,
                         int *shls_slice, int *ao_loc,
                         CINTOpt *cintopt,
                         const int *refuniqshl_map, const double *uniq_Rcut2s,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr2c_k_fill(intor, out, nkpts, comp, nimgs, jsh, jsh,
                 buf, env_loc, Ls, expkL_r, expkL_i, shls_slice, ao_loc,
                 cintopt,
                 refuniqshl_map, uniq_Rcut2s,
                 atm, natm, bas, nbas, env);
}

void PBCsr2c_k_drv(int (*intor)(), void (*fill)(), double complex *out,
                      int nkpts, int comp, int nimgs,
                      double *Ls, double complex *expkL,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt,
                      const int *refuniqshl_map, const double *uniq_Rcut2s,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      int nenv)
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
    NPdcopy(env_loc, env, nenv);
    size_t count = nkpts * OF_CMPLX + nimgs + 1;
    double *buf = malloc(sizeof(double)*(count*INTBUFMAX10*comp+cache_size));
#pragma omp for schedule(dynamic)
    for (jsh = 0; jsh < njsh; jsh++) {
        (*fill)(intor, out, nkpts, comp, nimgs, jsh,
                buf, env_loc, Ls, expkL_r, expkL_i,
                shls_slice, ao_loc, cintopt, refuniqshl_map, uniq_Rcut2s,
                atm, natm, bas, nbas, env);
    }
    free(buf);
    free(env_loc);
}
    free(expkL_r);
}

// non-split basis implementation of j3c
// Gamma point
static void sort3c_gs2_igtj(double *out, double *in, int *shls_slice,
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
                        pout[j*naok+k] = pin[k*dij+ij];
                    }
                }
                pout += (i+ao_loc[ish]+1) * naok;
            }
        }
        in += dijk * comp;
    }
}
static void sort3c_gs2_ieqj(double *out, double *in, int *shls_slice,
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
    int dkmax = MAX(INTBUFMAX10 / dij, dkaomax); // buf can hold at least 1 ksh
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

    const double omega = fabs(env_loc[PTR_RANGE_OMEGA]);

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
void PBCsr3c_gs2(int (*intor)(), double *out,
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
void PBCsr3c_g_drv(int (*intor)(), void (*fill)(), double *out,
                       int comp, int nimgs,
                       double *Ls,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, int8_t *shlpr_mask,
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

// single k-point, bvk
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
                        }
                    }
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
    const size_t off0 = ((size_t)ao_loc[ish0]) * (ao_loc[ish0] + 1) / 2;
    const size_t nij = ((size_t)ao_loc[ish1]) * (ao_loc[ish1] + 1) / 2 - off0;
    const size_t nijk = nij * naok;

    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkmax = ao_loc[msh1] - ao_loc[msh0];
    const size_t dijmc = dij * dkmax * comp;
    const int jp = ao_loc[jsh] - ao_loc[jsh0];
    out += (((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * naok;

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
    const size_t off0 = ((size_t)ao_loc[ish0]) * (ao_loc[ish0] + 1) / 2;
    const size_t nij = ((size_t)ao_loc[ish1]) * (ao_loc[ish1] + 1) / 2 - off0;
    const size_t nijk = nij * naok;

    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkmax = ao_loc[msh1] - ao_loc[msh0];
    const size_t dijmc = dij * dkmax * comp;
    const int jp = ao_loc[jsh] - ao_loc[jsh0];
    out += (((size_t)ao_loc[ish])*(ao_loc[ish]+1)/2-off0 + jp) * naok;

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
static void _nr3c_bvk_k(int (*intor)(), void (*fsort)(),
                        double complex *out, int nkpts_ij,
                        int nkpts, int comp, int nimgs, int bvk_nimgs,
                        int ish, int jsh, int *cell_loc_bvk,
                        double *buf, double *env_loc, double *Ls,
                        double *expkL_r, double *expkL_i, int *kptij_idx,
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

    const char TRANS_N = 'N';
    const double D1 = 1;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int kptrxyz;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkaomax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dkmax = MAX(INTBUFMAX10 / dij, dkaomax); // buf can hold at least 1 ksh
    int kshloc[ksh1-ksh0+1];
    int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, m, msh0, msh1, dijmc;
    size_t dijmk, dijkc;
    int ksh, dk;
    int iL_bvk, iL0, iL1, iL, jL_bvk, jL0, jL1, jL;
    int shls[3];
    double *bufexp_r = buf;
    double *bufexp_i = bufexp_r + bvk_nimgs * nkpts;
    double *bufk_r = bufexp_i + bvk_nimgs * nkpts;
    double *bufk_i, *bufL, *pbufL, *pbuf, *pbuf1, *pbuf2, *cache;

    shls[0] = ish;
    shls[1] = jsh;
// >>>>>>>>>>
    const double omega = fabs(env_loc[PTR_RANGE_OMEGA]);
    int Ish, Jsh, IJsh, Ksh, idij, kiLj, kiLi;
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
// <<<<<<<<<<
    for (m = 0; m < nkshloc; m++) {
        msh0 = kshloc[m];
        msh1 = kshloc[m+1];
        dkmax = ao_loc[msh1] - ao_loc[msh0];
        dijmc = dij * dkmax * comp;
        dijmk = dijmc * nkpts;
        bufk_i = bufk_r + dijmk;
        bufL   = bufk_i + dijmk;
        pbuf   = bufL   + ((size_t)bvk_nimgs) * dijmc;
        pbuf2  = pbuf   + dijmc;
        cache  = pbuf2  + dijmc;
        for (i = 0; i < dijmk*OF_CMPLX; i++) {
            bufk_r[i] = 0;
        }

        for (iL_bvk = 0; iL_bvk < bvk_nimgs; iL_bvk++) {
            for (i = 0; i < dijmc*bvk_nimgs; ++i) {
                bufL[i] = 0;
            }
            iL0 = cell_loc_bvk[iL_bvk];
            iL1 = cell_loc_bvk[iL_bvk+1];
            for (jL_bvk = 0; jL_bvk < bvk_nimgs; jL_bvk++) {
                pbufL = bufL + jL_bvk * dijmc;
                jL0 = cell_loc_bvk[jL_bvk];
                jL1 = cell_loc_bvk[jL_bvk+1];
                for (iL = iL0; iL < iL1; iL++) {
                    shift_bas(env_loc, env, Ls, iptrxyz, iL);
                    ri = env_loc + iptrxyz;

                    for (jL = jL0; jL < jL1; jL++) {
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

                        pbuf1 = pbuf;
                        for (ksh = msh0; ksh < msh1; ksh++) {
                            shls[2] = ksh;
                            dk = ao_loc[ksh+1] - ao_loc[ksh];
                            dijkc = dij * dk * comp;
                            Ksh = auxuniqshl_map[ksh-2*nbas];
                            Rcut2 = uniq_Rcut2s_K[Ksh];
                            kptrxyz = atm[PTR_COORD+bas[ATOM_OF+
                                                ksh*BAS_SLOTS]*ATM_SLOTS];
                            rk = env_loc + kptrxyz;
                            Rijk2 = get_dsqure(rc, rk);
                            if (Rijk2 < Rcut2) {
                                env_loc[PTR_RANGE_OMEGA] = 0.;
                                (*intor)(pbuf1, NULL, shls, atm, natm, bas, nbas,
                                         env_loc, cintopt, cache);
                                env_loc[PTR_RANGE_OMEGA] = omega;
                                if ((*intor)(pbuf2, NULL, shls, atm, natm,
                                             bas, nbas, env_loc, cintopt,
                                             cache)) {
                                    for (i = 0; i < dijkc; i++) {
                                        pbuf1[i] -= pbuf2[i];
                                    }
                                }
                            } else {
                                for (i = 0; i < dijkc; i++) {
                                    pbuf1[i] = 0;
                                }
                            } // if Rijk2
                            pbuf1 += dijkc;
                        } // ksh
                        for (i = 0; i < dijmc; i++) {
                            pbufL[i] += pbuf[i];
                        }
                    } // jL
                } // iL

                // ('k,kL->kL', conj(expkL[iL]), expkL)
                for (i = 0; i < nkpts; i++) {
                    kiLj = i*bvk_nimgs+jL_bvk;
                    kiLi = i*bvk_nimgs+iL_bvk;
                    bufexp_r[kiLj] = expkL_r[kiLj] * expkL_r[kiLi];
                    bufexp_r[kiLj]+= expkL_i[kiLj] * expkL_i[kiLi];
                    bufexp_i[kiLj] = expkL_i[kiLj] * expkL_r[kiLi];
                    bufexp_i[kiLj]-= expkL_r[kiLj] * expkL_i[kiLi];
                }

            } // jL_bvk

            dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &bvk_nimgs,
                   &D1, bufL, &dijmc, bufexp_r, &bvk_nimgs, &D1, bufk_r, &dijmc);
            dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &bvk_nimgs,
                   &D1, bufL, &dijmc, bufexp_i, &bvk_nimgs, &D1, bufk_i, &dijmc);

        } // iL_bvk
        (*fsort)(out, bufk_r, bufk_i, shls_slice, ao_loc, nkpts, comp,
                 ish, jsh, msh0, msh1);
    }
}
void PBCsr3c_bvk_ks1(int (*intor)(), double complex *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int bvk_nimgs,
                      int ish, int jsh, int *cell_loc_bvk,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt,
                      int *refuniqshl_map, int *auxuniqshl_map,
                      int nbasauxuniq, double *uniqexp,
                      double *uniq_dcut2s, double dcut_binsize,
                      double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr3c_bvk_k(intor, sort3c_ks1, out,
                nkpts_ij, nkpts, comp, nimgs, bvk_nimgs, ish, jsh, cell_loc_bvk,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                shls_slice, ao_loc, cintopt,
                refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                atm, natm, bas, nbas, env);
}
void PBCsr3c_bvk_ks2(int (*intor)(), double complex *out, int nkpts_ij,
                     int nkpts, int comp, int nimgs, int bvk_nimgs,
                     int ish, int jsh, int *cell_loc_bvk,
                     double *buf, double *env_loc, double *Ls,
                     double *expkL_r, double *expkL_i, int *kptij_idx,
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
        _nr3c_bvk_k(intor, &sort3c_ks2_igtj, out,
                    nkpts_ij, nkpts, comp, nimgs,
                    bvk_nimgs, ish, jsh, cell_loc_bvk,
                    buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                    shls_slice, ao_loc, cintopt,
                    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                    uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                    atm, natm, bas, nbas, env);
    } else if (ip == jp) {
        _nr3c_bvk_k(intor, &sort3c_ks2_ieqj, out,
                    nkpts_ij, nkpts, comp, nimgs,
                    bvk_nimgs, ish, jsh, cell_loc_bvk,
                    buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                    shls_slice, ao_loc, cintopt,
                    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                    uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                    atm, natm, bas, nbas, env);
    }
}
void PBCsr3c_bvk_k_drv(int (*intor)(), void (*fill)(), double *out,
                       int nkpts_ij, int nkpts,
                       int comp, int nimgs, int bvk_nimgs,
                       double *Ls,
                       double complex *expkL,
                       int *kptij_idx,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt,
                       int *cell_loc_bvk, int8_t *shlpr_mask,
                       int *refuniqshl_map, int *auxuniqshl_map,
                       int nbasauxuniq, double *uniqexp,
                       double *uniq_dcut2s, double dcut_binsize,
                       double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       int nenv)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;

    double *expkL_r = malloc(sizeof(double) * bvk_nimgs*nkpts * OF_CMPLX);
    double *expkL_i = expkL_r + bvk_nimgs*nkpts;
    int i;
    for (i = 0; i < bvk_nimgs*nkpts; i++) {
        expkL_r[i] = creal(expkL[i]);
        expkL_i[i] = cimag(expkL[i]);
    }

    int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    int dk = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dijk = di*dj*dk;
    int dijmk = MAX(INTBUFMAX10, dijk);
    size_t count = (nkpts*OF_CMPLX + bvk_nimgs + 2) * dijmk * comp +
                   nkpts*bvk_nimgs*OF_CMPLX;
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
        (*fill)(intor, out, nkpts_ij, nkpts, comp, nimgs, bvk_nimgs,
                ish, jsh, cell_loc_bvk,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
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
    free(expkL_r);
}

// single k-point, no bvk
static void _nr3c_k(int (*intor)(), void (*fsort)(),
                    double complex *out, int nkpts_ij,
                    int nkpts, int comp, int nimgs, int ish, int jsh,
                    double *buf, double *env_loc, double *Ls,
                    double *expkL_r, double *expkL_i, int *kptij_idx,
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

    const char TRANS_N = 'N';
    const double D1 = 1;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int kptrxyz;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkaomax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dkmax = MAX(INTBUFMAX10 / dij, dkaomax); // buf can hold at least 1 ksh
    int kshloc[ksh1-ksh0+1];
    int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, m, msh0, msh1, dijmc; 
    size_t dijmk, dijkc;
    int ksh, dk;
    int iL, jL, jLcount;
    int shls[3];
    double *bufexp_r = buf;
    double *bufexp_i = bufexp_r + nimgs * nkpts;
    double *bufk_r = bufexp_i + nimgs * nkpts;
    double *bufk_i, *bufL, *pbuf, *pbuf2, *cache;

    shls[0] = ish;
    shls[1] = jsh;
// >>>>>>>>>>
    const double omega = fabs(env_loc[PTR_RANGE_OMEGA]);
    int Ish, Jsh, IJsh, Ksh, idij, kiLj, kiLjc, kiLi;
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
// <<<<<<<<<<
    for (m = 0; m < nkshloc; m++) {
        msh0 = kshloc[m];
        msh1 = kshloc[m+1];
        dkmax = ao_loc[msh1] - ao_loc[msh0];
        dijmc = dij * dkmax * comp;
        dijmk = dijmc * nkpts;
        bufk_i = bufk_r + dijmk;
        bufL   = bufk_i + dijmk;
        pbuf2  = bufL   + ((size_t)nimgs) * dijmc;
        cache  = pbuf2  + dijmc;
        for (i = 0; i < dijmk*OF_CMPLX; i++) {
            bufk_r[i] = 0;
        }

        for (iL = 0; iL < nimgs; iL++) {
            shift_bas(env_loc, env, Ls, iptrxyz, iL);
            ri = env_loc + iptrxyz;
            pbuf = bufL;
            jLcount = 0;
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

                for (ksh = msh0; ksh < msh1; ksh++) {
                    shls[2] = ksh;
                    dk = ao_loc[ksh+1] - ao_loc[ksh];
                    dijkc = dij * dk * comp;
                    Ksh = auxuniqshl_map[ksh-2*nbas];
                    Rcut2 = uniq_Rcut2s_K[Ksh];
                    kptrxyz = atm[PTR_COORD+bas[ATOM_OF+
                                        ksh*BAS_SLOTS]*ATM_SLOTS];
                    rk = env_loc + kptrxyz;
                    Rijk2 = get_dsqure(rc, rk);
                    if (Rijk2 < Rcut2) {
                        env_loc[PTR_RANGE_OMEGA] = 0.;
                        (*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                                 env_loc, cintopt, cache);
                        env_loc[PTR_RANGE_OMEGA] = omega;
                        if ((*intor)(pbuf2, NULL, shls, atm, natm,
                                     bas, nbas, env_loc, cintopt,
                                     cache)) {
                            for (i = 0; i < dijkc; i++) {
                                pbuf[i] -= pbuf2[i];
                            }
                        }
                    } else {
                        for (i = 0; i < dijkc; i++) {
                            pbuf[i] = 0;
                        }
                    } // if Rijk2
                    pbuf += dijkc;
                } // ksh

                // ('k,kL->kL', conj(expkL[iL]), expkL)
                for (i = 0; i < nkpts; i++) {
                    kiLjc = i*nimgs+jLcount;
                    kiLj  = i*nimgs+jL;
                    kiLi  = i*nimgs+iL;
                    bufexp_r[kiLjc] = expkL_r[kiLj] * expkL_r[kiLi];
                    bufexp_r[kiLjc]+= expkL_i[kiLj] * expkL_i[kiLi];
                    bufexp_i[kiLjc] = expkL_i[kiLj] * expkL_r[kiLi];
                    bufexp_i[kiLjc]-= expkL_r[kiLj] * expkL_i[kiLi];
                }
                ++jLcount;
            } // jL

            dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &jLcount,
                   &D1, bufL, &dijmc, bufexp_r, &nimgs, &D1, bufk_r, &dijmc);
            dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &jLcount,
                   &D1, bufL, &dijmc, bufexp_i, &nimgs, &D1, bufk_i, &dijmc);
        } // iL

        (*fsort)(out, bufk_r, bufk_i, shls_slice, ao_loc, nkpts, comp,
                 ish, jsh, msh0, msh1);
    }
}
void PBCsr3c_ks1(int (*intor)(), double complex *out, int nkpts_ij,
                 int nkpts, int comp, int nimgs, int ish, int jsh,
                 double *buf, double *env_loc, double *Ls,
                 double *expkL_r, double *expkL_i, int *kptij_idx,
                 int *shls_slice, int *ao_loc,
                 CINTOpt *cintopt,
                 int *refuniqshl_map, int *auxuniqshl_map,
                 int nbasauxuniq, double *uniqexp,
                 double *uniq_dcut2s, double dcut_binsize,
                 double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr3c_k(intor, sort3c_ks1, out, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
            buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
            shls_slice, ao_loc, cintopt,
            refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
            uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
            atm, natm, bas, nbas, env);
}
void PBCsr3c_ks2(int (*intor)(), double complex *out, int nkpts_ij,
                 int nkpts, int comp, int nimgs, int ish, int jsh,
                 double *buf, double *env_loc, double *Ls,
                 double *expkL_r, double *expkL_i, int *kptij_idx,
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
        _nr3c_k(intor, &sort3c_ks2_igtj, out,
                nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                shls_slice, ao_loc, cintopt,
                refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                atm, natm, bas, nbas, env);
    } else if (ip == jp) {
        _nr3c_k(intor, &sort3c_ks2_ieqj, out,
                nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                shls_slice, ao_loc, cintopt,
                refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                atm, natm, bas, nbas, env);
    }
}
void PBCsr3c_k_drv(int (*intor)(), void (*fill)(), double *out,
                   int nkpts_ij, int nkpts,
                   int comp, int nimgs,
                   double *Ls,
                   double complex *expkL,
                   int *kptij_idx,
                   int *shls_slice, int *ao_loc,
                   CINTOpt *cintopt,
                   int8_t *shlpr_mask,
                   int *refuniqshl_map, int *auxuniqshl_map,
                   int nbasauxuniq, double *uniqexp,
                   double *uniq_dcut2s, double dcut_binsize,
                   double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int nenv)
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

    int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    int dk = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dijk = di*dj*dk;
    int dijmk = MAX(INTBUFMAX10, dijk);
    size_t count = (nkpts*OF_CMPLX + nimgs + 2) * dijmk * comp +
                   nkpts*nimgs*OF_CMPLX;
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
        (*fill)(intor, out, nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
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
    free(expkL_r);
}

// k-point pairs, bvk
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
                            pout[i*njk+k] = pbr[k*dij+i] +
                                            pbi[k*dij+i]*_Complex_I;
                        }
                    }
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
                        }
                    }
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
static void _nr3c_bvk_kk(int (*intor)(), void (*fsort)(),
                         double complex *out, int nkpts_ij,
                         int nkpts, int comp, int nimgs, int bvk_nimgs,
                         int ish, int jsh, int *cell_loc_bvk,
                         double *buf, double *env_loc, double *Ls,
                         double *expkL_r, double *expkL_i, int *kptij_idx,
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

    const char TRANS_N = 'N';
    const double D0 = 0;
    const double D1 = 1;
    const double ND1 = -1;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int kptrxyz;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkaomax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dkmax = MAX(INTBUFMAX / dij, dkaomax);
    int kshloc[ksh1-ksh0+1];
    int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, m, msh0, msh1, dijm, dijmc, dijmk, dijkc;
    int ksh, dk;
    int iL_bvk, iL0_bvk, iLcount_bvk, iL0, iL1, iL, jL_bvk, jL0, jL1, jL;
    int shls[3];
    double *bufkk_r, *bufkk_i, *bufkL_r, *bufkL_i, *bufL, *pbuf, *cache;
    double *buf_rs, *buf_rs0, *pbuf_rs;

    const double omega = fabs(env_loc[PTR_RANGE_OMEGA]);

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
        dijmk = dijmc * nkpts;
        bufkk_r = buf;
        bufkk_i = bufkk_r + (size_t)nkpts * dijmk;
        bufkL_r = bufkk_i + (size_t)nkpts * dijmk;
        bufkL_i = bufkL_r + (size_t)MIN(bvk_nimgs,IMGBLK) * dijmk;
        bufL    = bufkL_i + (size_t)MIN(bvk_nimgs,IMGBLK) * dijmk;
        buf_rs0 = bufL    + (size_t)bvk_nimgs * dijmc;
        pbuf_rs = buf_rs0 + (size_t)dijmc;
        cache   = pbuf_rs + (size_t)dijmc;
        for (i = 0; i < nkpts*dijmk*OF_CMPLX; i++) {
            bufkk_r[i] = 0;
        }

        for (iL0_bvk = 0; iL0_bvk < bvk_nimgs; iL0_bvk+=IMGBLK) {
            iLcount_bvk = MIN(IMGBLK, bvk_nimgs - iL0_bvk);
            for (iL_bvk = iL0_bvk; iL_bvk < iL0_bvk+iLcount_bvk; iL_bvk++) {
                for (i = 0; i < dijmc*bvk_nimgs; i++) {
                    bufL[i] = 0;
                }
                iL0 = cell_loc_bvk[iL_bvk];
                iL1 = cell_loc_bvk[iL_bvk+1];
                for (jL_bvk = 0; jL_bvk < bvk_nimgs; jL_bvk++) {
                    pbuf = bufL + dijmc * jL_bvk;
                    jL0 = cell_loc_bvk[jL_bvk];
                    jL1 = cell_loc_bvk[jL_bvk+1];
                    for (iL = iL0; iL < iL1; iL++) {
                        shift_bas(env_loc, env, Ls, iptrxyz, iL);
                        ri = env_loc + iptrxyz;

                        for (jL = jL0; jL < jL1; jL++) {
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
                            buf_rs = buf_rs0;
                            for (ksh = msh0; ksh < msh1; ksh++) {
                                shls[2] = ksh;
                                dk = ao_loc[ksh+1] - ao_loc[ksh];
                                dijkc = dij * dk * comp;
                                Ksh = auxuniqshl_map[ksh-2*nbas];
                                Rcut2 = uniq_Rcut2s_K[Ksh];
                                kptrxyz = atm[PTR_COORD+bas[ATOM_OF+
                                                    ksh*BAS_SLOTS]*ATM_SLOTS];
                                rk = env_loc + kptrxyz;
                                Rijk2 = get_dsqure(rc, rk);
                                if (Rijk2 < Rcut2) {
                                    env_loc[PTR_RANGE_OMEGA] = 0.;
                                    (*intor)(buf_rs, NULL, shls, atm, natm, bas, nbas,
                                             env_loc, cintopt, cache);
                                    env_loc[PTR_RANGE_OMEGA] = omega;
                                    if ((*intor)(pbuf_rs, NULL, shls, atm, natm,
                                                 bas, nbas, env_loc, cintopt,
                                                 cache)) {
                                        for (i = 0; i < dijkc; i++) {
                                            buf_rs[i] -= pbuf_rs[i];
                                        }
                                    }
                                } else {
                                    for (i = 0; i < dijkc; i++) {
                                        buf_rs[i] = 0;
                                    }
                                } // if Rijk2
                                buf_rs += dijkc;
                            } // ksh
                            for (i = 0; i < dijmc; i++) {
                                pbuf[i] += buf_rs0[i];
                            }
                        } // jL
                    }   // iL
                } // jL_bvk
                dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &bvk_nimgs,
                       &D1, bufL, &dijmc, expkL_r, &bvk_nimgs,
                       &D0, bufkL_r+(iL_bvk-iL0_bvk)*(size_t)dijmk, &dijmc);
                dgemm_(&TRANS_N, &TRANS_N, &dijmc, &nkpts, &bvk_nimgs,
                       &D1, bufL, &dijmc, expkL_i, &bvk_nimgs,
                       &D0, bufkL_i+(iL_bvk-iL0_bvk)*(size_t)dijmk, &dijmc);
            } // iL_bvk
            // conj(exp(1j*dot(h,k)))
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount_bvk,
                   &D1, bufkL_r, &dijmk, expkL_r+(size_t)iL0_bvk, &bvk_nimgs,
                   &D1, bufkk_r, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount_bvk,
                   &D1, bufkL_i, &dijmk, expkL_i+(size_t)iL0_bvk, &bvk_nimgs,
                   &D1, bufkk_r, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount_bvk,
                   &D1, bufkL_i, &dijmk, expkL_r+(size_t)iL0_bvk, &bvk_nimgs,
                   &D1, bufkk_i, &dijmk);
            dgemm_(&TRANS_N, &TRANS_N, &dijmk, &nkpts, &iLcount_bvk,
                   &ND1, bufkL_r, &dijmk, expkL_i+(size_t)iL0_bvk, &bvk_nimgs,
                   &D1, bufkk_i, &dijmk);
        } // iL0_bvk
        (*fsort)(out, bufkk_r, bufkk_i, kptij_idx, shls_slice, ao_loc,
                 nkpts, nkpts_ij, comp, ish, jsh, msh0, msh1);
    }   // m
}
void PBCsr3c_bvk_kks2(int (*intor)(), double complex *out, int nkpts_ij,
                      int nkpts, int comp, int nimgs, int bvk_nimgs,
                      int ish, int jsh, int *cell_loc_bvk,
                      double *buf, double *env_loc, double *Ls,
                      double *expkL_r, double *expkL_i, int *kptij_idx,
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
        _nr3c_bvk_kk(intor, &sort3c_kks2_igtj, out,
                     nkpts_ij, nkpts, comp, nimgs, bvk_nimgs,
                     ish, jsh, cell_loc_bvk,
                     buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                     shls_slice, ao_loc, cintopt,
                     refuniqshl_map, auxuniqshl_map,
                     nbasauxuniq, uniqexp,
                     uniq_dcut2s, dcut_binsize,
                     uniq_Rcut2s, uniqshlpr_dij_loc,
                     atm, natm, bas, nbas, env);
    } else if (ip == jp) {
        _nr3c_bvk_kk(intor, &sort3c_kks1, out,
                     nkpts_ij, nkpts, comp, nimgs, bvk_nimgs,
                     ish, jsh, cell_loc_bvk,
                     buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                     shls_slice, ao_loc, cintopt,
                     refuniqshl_map, auxuniqshl_map,
                     nbasauxuniq, uniqexp,
                     uniq_dcut2s, dcut_binsize,
                     uniq_Rcut2s, uniqshlpr_dij_loc,
                     atm, natm, bas, nbas, env);
    }
}
void PBCsr3c_bvk_kk_drv(int (*intor)(), void (*fill)(), double *out,
                        int nkpts_ij, int nkpts,
                        int comp, int nimgs, int bvk_nimgs,
                        double *Ls,
                        double complex *expkL,
                        int *kptij_idx,
                        int *shls_slice, int *ao_loc,
                        CINTOpt *cintopt,
                        int *cell_loc_bvk, int8_t *shlpr_mask,
                        int *refuniqshl_map, int *auxuniqshl_map,
                        int nbasauxuniq, double *uniqexp,
                        double *uniq_dcut2s, double dcut_binsize,
                        double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        int nenv)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;

    double *expkL_r = malloc(sizeof(double) * bvk_nimgs*nkpts * OF_CMPLX);
    double *expkL_i = expkL_r + bvk_nimgs*nkpts;
    int i;
    for (i = 0; i < bvk_nimgs*nkpts; i++) {
        expkL_r[i] = creal(expkL[i]);
        expkL_i[i] = cimag(expkL[i]);
    }

    int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    int dk = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dijk = di*dj*dk;
    int dijmk = MAX(INTBUFMAX, dijk);
    size_t count = ((nkpts + MIN(bvk_nimgs,IMGBLK))*nkpts * OF_CMPLX +
                     bvk_nimgs + 2) * dijmk * comp;
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
        (*fill)(intor, out, nkpts_ij, nkpts, comp, nimgs, bvk_nimgs,
                ish, jsh, cell_loc_bvk,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
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
    free(expkL_r);
}

// k-point pairs, no bvk
static void _nr3c_kk(int (*intor)(), void (*fsort)(),
                     double complex *out, int nkpts_ij,
                     int nkpts, int comp, int nimgs, int ish, int jsh,
                     double *buf, double *env_loc, double *Ls,
                     double *expkL_r, double *expkL_i, int *kptij_idx,
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

    const char TRANS_N = 'N';
    const double D0 = 0;
    const double D1 = 1;
    const double ND1 = -1;

    jsh += jsh0;
    ish += ish0;
    int iptrxyz = atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    int jptrxyz = atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    int kptrxyz;
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int dkaomax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dkmax = MAX(INTBUFMAX / dij, dkaomax); // buf can hold at least 1 ksh
    int kshloc[ksh1-ksh0+1];
    int nkshloc = shloc_partition(kshloc, ao_loc, ksh0, ksh1, dkmax);

    int i, m, msh0, msh1, dijm, dijmc, dijmk, dijkc;
    int ksh, dk, iL0, iL, jL, iLcount;
    int shls[3];
    double *bufkk_r, *bufkk_i, *bufkL_r, *bufkL_i, *bufL, *pbuf, *pbuf2, *cache;

    shls[0] = ish;
    shls[1] = jsh;

    const double omega = fabs(env_loc[PTR_RANGE_OMEGA]);
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
// <<<<<<<<<<
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
        pbuf2   = bufL    + (size_t)nimgs * dijmc;
        cache   = pbuf2   + dijmc;
        for (i = 0; i < nkpts*dijmk*OF_CMPLX; i++) {
            bufkk_r[i] = 0;
        }

        for (iL0 = 0; iL0 < nimgs; iL0+=IMGBLK) {
            iLcount = MIN(IMGBLK, nimgs - iL0);
            for (iL = iL0; iL < iL0+iLcount; iL++) {
                shift_bas(env_loc, env, Ls, iptrxyz, iL);
                ri = env_loc + iptrxyz;
                pbuf = bufL;
                for (jL = 0; jL < nimgs; jL++) {
                    shift_bas(env_loc, env, Ls, jptrxyz, jL);
                    rj = env_loc + jptrxyz;
// >>>>>>>>
                    dij2 = get_dsqure(ri, rj);
                    if(dij2 > dij2_cut) {
                        for (i = 0; i < dijmc; ++i) {
                            pbuf[i] = 0;
                        }
                        pbuf += dijmc;
                        continue;
                    }
                    idij = (int)(sqrt(dij2)*inv_d0);
                    uniq_Rcut2s_K = uniq_Rcut2s_IJ + idij * nbasauxuniq;
// <<<<<<<<
                    get_rc(rc, ri, rj, ei, ej);

                    for (ksh = msh0; ksh < msh1; ksh++) {
                        shls[2] = ksh;
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        dijkc = dij * dk * comp;
                        Ksh = auxuniqshl_map[ksh-2*nbas];
                        Rcut2 = uniq_Rcut2s_K[Ksh];
                        kptrxyz = atm[PTR_COORD+bas[ATOM_OF+
                                            ksh*BAS_SLOTS]*ATM_SLOTS];
                        rk = env_loc + kptrxyz;
                        Rijk2 = get_dsqure(rc, rk);
                        if (Rijk2 < Rcut2) {
                            env_loc[PTR_RANGE_OMEGA] = 0.;
                            (*intor)(pbuf, NULL, shls, atm, natm, bas, nbas,
                                     env_loc, cintopt, cache);
                            env_loc[PTR_RANGE_OMEGA] = omega;
                            if ((*intor)(pbuf2, NULL, shls, atm, natm,
                                         bas, nbas, env_loc, cintopt,
                                         cache)) {
                                for (i = 0; i < dijkc; i++) {
                                    pbuf[i] -= pbuf2[i];
                                }
                            }
                        } else {
                            for (i = 0; i < dijkc; i++) {
                                pbuf[i] = 0;
                            }
                        } // if Rijk2
                        pbuf += dijkc;
                    } // ksh
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
        } // iL0
        (*fsort)(out, bufkk_r, bufkk_i, kptij_idx, shls_slice,
                 ao_loc, nkpts, nkpts_ij, comp, ish, jsh,
                 msh0, msh1);
    } // m
}
void PBCsr3c_kks1(int (*intor)(), double complex *out, int nkpts_ij,
                  int nkpts, int comp, int nimgs, int ish, int jsh,
                  double *buf, double *env_loc, double *Ls,
                  double *expkL_r, double *expkL_i, int *kptij_idx,
                  int *shls_slice, int *ao_loc,
                  CINTOpt *cintopt,
                  int *refuniqshl_map, int *auxuniqshl_map,
                  int nbasauxuniq, double *uniqexp,
                  double *uniq_dcut2s, double dcut_binsize,
                  double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
    _nr3c_kk(intor, &sort3c_kks1, out,
             nkpts_ij, nkpts, comp, nimgs, ish, jsh,
             buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
             shls_slice, ao_loc, cintopt,
             refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
             uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
             atm, natm, bas, nbas, env);
}
void PBCsr3c_kks2(int (*intor)(), double complex *out, int nkpts_ij,
                  int nkpts, int comp, int nimgs, int ish, int jsh,
                  double *buf, double *env_loc, double *Ls,
                  double *expkL_r, double *expkL_i, int *kptij_idx,
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
        _nr3c_kk(intor, &sort3c_kks2_igtj, out,
                 nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                 buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                 shls_slice, ao_loc, cintopt,
                 refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                 uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                 atm, natm, bas, nbas, env);
    } else if (ip == jp) {
        _nr3c_kk(intor, &sort3c_kks1, out,
                 nkpts_ij, nkpts, comp, nimgs, ish, jsh,
                 buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
                 shls_slice, ao_loc, cintopt,
                 refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                 uniq_dcut2s, dcut_binsize, uniq_Rcut2s, uniqshlpr_dij_loc,
                 atm, natm, bas, nbas, env);
    }
}
void PBCsr3c_kk_drv(int (*intor)(), void (*fill)(), double *out,
                    int nkpts_ij, int nkpts,
                    int comp, int nimgs,
                    double *Ls,
                    double complex *expkL,
                    int *kptij_idx,
                    int *shls_slice, int *ao_loc,
                    CINTOpt *cintopt,
                    int8_t *shlpr_mask,
                    int *refuniqshl_map, int *auxuniqshl_map,
                    int nbasauxuniq, double *uniqexp,
                    double *uniq_dcut2s, double dcut_binsize,
                    double *uniq_Rcut2s, int *uniqshlpr_dij_loc,
                    int *atm, int natm, int *bas, int nbas, double *env,
                    int nenv)
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

    int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    int dk = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int dijk = di*dj*dk;
    int dijmk = MAX(INTBUFMAX, dijk);
    size_t count = ((nkpts + MIN(nimgs,IMGBLK))*nkpts * OF_CMPLX +
                     nimgs + 2) * dijmk * comp;
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
        (*fill)(intor, out, nkpts_ij, nkpts, comp, nimgs,
                ish, jsh,
                buf, env_loc, Ls, expkL_r, expkL_i, kptij_idx,
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
    free(expkL_r);
}
