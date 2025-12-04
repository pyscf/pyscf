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
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "pbc/pbc.h"

#ifdef QCINT_VERSION
int compiled_with_qcint = 1;
#else
int compiled_with_qcint = 0;
#endif

void CINTinit_int2e_EnvVars(CINTEnvVars *envs, int *ng, int *shls,
                            int *atm, int natm, int *bas, int nbas, double *env);
extern void CINTgout2e();
#ifdef QCINT_VERSION
extern void CINTgout2e_simd1();
#endif
int CINT2e_loop_nopt(double *gctr, CINTEnvVars *envs, double *cache, int *empty);
int CINT2e_loop(double *gctr, CINTEnvVars *envs, double *cache, int *empty);
int CINT2e_1111_loop(double *gctr, CINTEnvVars *envs, double *cache, int *empty);
void c2s_cart_2e1(double *out, double *gctr, int *dims, CINTEnvVars *envs, double *cache);
void c2s_sph_2e1(double *out, double *gctr, int *dims, CINTEnvVars *envs, double *cache);


void PBCinit_int2e_EnvVars(CINTEnvVars *envs, int *ng, int *cell0_shls, BVKEnvs *envs_bvk)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int ish0 = seg2sh[seg_loc[ish_cell0]];
        int jsh0 = seg2sh[seg_loc[jsh_cell0]];
        int ksh0 = seg2sh[seg_loc[ksh_cell0]];
        int lsh0 = seg2sh[seg_loc[lsh_cell0]];
        int shls[4] = {ish0, jsh0, ksh0, lsh0};
        CINTinit_int2e_EnvVars(envs, ng, shls, atm, natm, bas, nbas, env);
}

static void update_int2e_envs(CINTEnvVars *envs, int *shls)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int i_sh = shls[0];
        int j_sh = shls[1];
        int k_sh = shls[2];
        int l_sh = shls[3];
        envs->shls = shls;
        envs->ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        envs->rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        envs->rk = env + atm(PTR_COORD, bas(ATOM_OF, k_sh));
        envs->rl = env + atm(PTR_COORD, bas(ATOM_OF, l_sh));

        int ibase = envs->li_ceil > envs->lj_ceil;
        int kbase = envs->lk_ceil > envs->ll_ceil;
#ifdef CINT_SOVERSION
#if CINT_SOVERSION < 6
        if (envs->nrys_roots <= 2) {
                ibase = 0;
                kbase = 0;
        }
#endif
#else
        if (envs->nrys_roots <= 2) {
                ibase = 0;
                kbase = 0;
        }
#endif
        if (kbase) {
                envs->rx_in_rklrx = envs->rk;
                envs->rkrl[0] = envs->rk[0] - envs->rl[0];
                envs->rkrl[1] = envs->rk[1] - envs->rl[1];
                envs->rkrl[2] = envs->rk[2] - envs->rl[2];
        } else {
                envs->rx_in_rklrx = envs->rl;
                envs->rkrl[0] = envs->rl[0] - envs->rk[0];
                envs->rkrl[1] = envs->rl[1] - envs->rk[1];
                envs->rkrl[2] = envs->rl[2] - envs->rk[2];
        }

        if (ibase) {
                envs->rx_in_rijrx = envs->ri;
                envs->rirj[0] = envs->ri[0] - envs->rj[0];
                envs->rirj[1] = envs->ri[1] - envs->rj[1];
                envs->rirj[2] = envs->ri[2] - envs->rj[2];
        } else {
                envs->rx_in_rijrx = envs->rj;
                envs->rirj[0] = envs->rj[0] - envs->ri[0];
                envs->rirj[1] = envs->rj[1] - envs->ri[1];
                envs->rirj[2] = envs->rj[2] - envs->ri[2];
        }
}

int PBCint2e_loop(double *gctr, int *cell0_shls, int *bvk_cells, int cutoff,
                  float *rij_cond, float *rkl_cond,
                  CINTEnvVars *envs_cint, BVKEnvs *envs_bvk, double *cache)
{
        size_t Nbas = envs_cint->nbas;
        int nbasp = envs_bvk->nbasp;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int ish_bvk = ish_cell0;
        int jsh_bvk = jsh_cell0 + cell_j * nbasp;
        int ksh_bvk = ksh_cell0 + cell_k * nbasp;
        int lsh_bvk = lsh_cell0 + cell_l * nbasp;
        int iseg0 = seg_loc[ish_bvk];
        int jseg0 = seg_loc[jsh_bvk];
        int kseg0 = seg_loc[ksh_bvk];
        int lseg0 = seg_loc[lsh_bvk];
        int iseg1 = seg_loc[ish_bvk+1];
        int jseg1 = seg_loc[jsh_bvk+1];
        int kseg1 = seg_loc[ksh_bvk+1];
        int lseg1 = seg_loc[lsh_bvk+1];

        // basis in remote bvk cell may be skipped
        if (jseg0 == jseg1 || kseg0 == kseg1 || lseg0 == lseg1) {
                return 0;
        }

        int nksh = seg2sh[kseg1] - seg2sh[kseg0];
        int nlsh = seg2sh[lseg1] - seg2sh[lseg0];
        int nkl = nksh * nlsh;
        int rkl_off = seg2sh[kseg0] * nlsh + seg2sh[lseg0];
        int rs_cell_nbas = seg_loc[nbasp];

        int *x_ctr = envs_cint->x_ctr;
        int ncomp = envs_cint->ncomp_e1 * envs_cint->ncomp_e2 * envs_cint->ncomp_tensor;
        size_t nc = x_ctr[0] * x_ctr[1] * x_ctr[2] * x_ctr[3];
        size_t dijkl = (size_t)envs_cint->nf * nc * ncomp;
        size_t Nbas2 = Nbas * Nbas;
        int empty = 1;
        NPdset0(gctr, dijkl);

        int (*intor_loop)(double *, CINTEnvVars *, double *, int *);
        if (envs_cint->opt == NULL) {
                intor_loop = &CINT2e_loop_nopt;
        } else if (x_ctr[0] == 1 && x_ctr[1] == 1 && x_ctr[2] == 1 && x_ctr[3] == 1) {
#ifdef QCINT_VERSION
                // In qcint, CINT2e_1111_loop is problematic can cause seg fault.
                intor_loop = &CINT2e_loop;
#else
                intor_loop = &CINT2e_1111_loop;
#endif
        } else {
                intor_loop = &CINT2e_loop;
        }

        int shls[4];
        int ish, jsh, ksh, lsh;
        int iseg, jseg, kseg, lseg;
        int jsh0, ksh0, lsh0;
        int jsh1, ksh1, lsh1;
        int16_t *qidx_ijij = envs_bvk->qindex;
        int16_t *qidx_iijj = qidx_ijij + Nbas2;
        int16_t *sindex = qidx_iijj + Nbas2;
        float *xij_cond = rij_cond;
        float *yij_cond = rij_cond + rs_cell_nbas * Nbas;
        float *zij_cond = rij_cond + rs_cell_nbas * Nbas * 2;
        float *xkl_cond = rkl_cond;
        float *ykl_cond = rkl_cond + nkl;
        float *zkl_cond = rkl_cond + nkl * 2;
        int16_t *qidx_ij, *qidx_kl, *qidx_ik, *qidx_jk, *skl_idx;
        int kl_cutoff, jl_cutoff, il_cutoff, sij;
        float skl_cutoff;
        float xij, yij, zij, dx, dy, dz, r2;

        int *bas = envs_cint->bas;
        double *env = envs_cint->env;
        // the most diffused function in each shell
        double omega = env[PTR_RANGE_OMEGA];
        assert(omega != 0);
        float omega2 = omega * omega;
        float ai, aj, ak, al, aij, akl;
        float theta, theta_ij, theta_r2;

        for (iseg = iseg0; iseg < iseg1; iseg++) {
                ish = seg2sh[iseg];
                shls[0] = ish;
                ai = env[bas(PTR_EXP,ish) + bas(NPRIM_OF,ish)-1];
                for (jseg = jseg0; jseg < jseg1; jseg++) {
                        jsh0 = seg2sh[jseg];
                        jsh1 = seg2sh[jseg+1];
                        aj = env[bas(PTR_EXP,jsh0) + bas(NPRIM_OF,jsh0)-1];
                        aij = ai + aj;
                        theta_ij = omega2*aij / (omega2 + aij);
                        for (kseg = kseg0; kseg < kseg1; kseg++) {
                                ksh0 = seg2sh[kseg];
                                ksh1 = seg2sh[kseg+1];
                                ak = env[bas(PTR_EXP,ksh0) + bas(NPRIM_OF,ksh0)-1];
                                for (lseg = lseg0; lseg < lseg1; lseg++) {
                                        lsh0 = seg2sh[lseg];
                                        lsh1 = seg2sh[lseg+1];
                                        al = env[bas(PTR_EXP,lsh0) + bas(NPRIM_OF,lsh0)-1];
                                        akl = ak + al;
                                        // theta = 1/(1/aij+1/akl+1/omega2);
                                        theta = theta_ij*akl / (theta_ij + akl);
                                        qidx_ij = qidx_ijij + ish * Nbas;
                                        qidx_ik = qidx_iijj + ish * Nbas;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        if (qidx_ij[jsh] < cutoff) {
                continue;
        }
        shls[1] = jsh;
        kl_cutoff = cutoff - qidx_ij[jsh];
        qidx_jk = qidx_iijj + jsh * Nbas;
        sij = sindex[ish * Nbas + jsh];
        xij = xij_cond[iseg * Nbas + jsh];
        yij = yij_cond[iseg * Nbas + jsh];
        zij = zij_cond[iseg * Nbas + jsh];
        skl_cutoff = cutoff - sij;
        for (ksh = ksh0; ksh < ksh1; ksh++) {
                if (qidx_ik[ksh] < cutoff ||
                    qidx_jk[ksh] < cutoff) {
                        continue;
                }
                shls[2] = ksh;
                qidx_kl = qidx_ijij + ksh * Nbas;
                skl_idx = sindex + ksh * Nbas;
                jl_cutoff = cutoff - qidx_ik[ksh];
                il_cutoff = cutoff - qidx_jk[ksh];
                for (lsh = lsh0; lsh < lsh1; lsh++) {
                        if (qidx_kl[lsh] < kl_cutoff ||
                            qidx_jk[lsh] < jl_cutoff ||
                            qidx_ik[lsh] < il_cutoff) {
                                continue;
                        }
                        // the last level screening is theta*(Rij-Rkl)^2.
                        // For contracted basis, this estimation requires the
                        // values of Rij and theta for all primitive
                        // combinations in |i> and |j>. Given theta_kl and Rkl,
                        // for gaussians in |i> and |j>, sij + theta*(Rij-Rkl)^2
                        // is the same to the exponents of three gaussian overlap:
                        //    ai(r-Ri)^2, aj(r-Rj)^2, theta_kl(r-Rkl)^2
                        // This overlap cannot be larger than the overlap among
                        // the most diffused functions in |i> and |j>.
                        // Here theta_kl, sij, skl, xij, xkl are all constructed
                        // with the most diffused functions, which give a bound
                        // for the exponent part of SR-int2e (ij|kl).
                        dx = xij - xkl_cond[ksh * nlsh + lsh - rkl_off];
                        dy = yij - ykl_cond[ksh * nlsh + lsh - rkl_off];
                        dz = zij - zkl_cond[ksh * nlsh + lsh - rkl_off];
                        r2 = dx * dx + dy * dy + dz * dz;
                        theta_r2 = theta * r2 + logf(r2 + 1e-30f);
                        if (theta_r2*LOG_ADJUST + skl_cutoff < skl_idx[lsh]) {
                                shls[3] = lsh;
                                update_int2e_envs(envs_cint, shls);
                                (*intor_loop)(gctr, envs_cint, cache, &empty);
                        }
                }
        }
}
                                }
                        }
                }
        }
        return !empty;
}

// envs_cint are updated in this function. It needs to be allocated
// omp-privately
int PBCint2e_cart(double *eri_buf, int *cell0_shls, int *bvk_cells, int cutoff,
                  float *rij_cond, float *rkl_cond,
                  CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 1, 1};
        PBCinit_int2e_EnvVars(envs_cint, ng, cell0_shls, envs_bvk);
        envs_cint->f_gout = &CINTgout2e;
#ifdef QCINT_VERSION
        envs_cint->f_gout_simd1 = &CINTgout2e_simd1;
#endif

        int *x_ctr = envs_cint->x_ctr;
        int ncomp = 1;
        int di = envs_cint->nfi * x_ctr[0];
        int dj = envs_cint->nfj * x_ctr[1];
        int dk = envs_cint->nfk * x_ctr[2];
        int dl = envs_cint->nfl * x_ctr[3];
        size_t dijkl = (size_t)di * dj * dk * dl * ncomp;
        double *gctr = eri_buf + dijkl;
        double *cache = gctr + dijkl;
        int has_value = PBCint2e_loop(gctr, cell0_shls, bvk_cells, cutoff,
                                      rij_cond, rkl_cond, envs_cint, envs_bvk, cache);
        if (has_value) {
                int dims[4] = {di, dj, dk, dl};
                c2s_cart_2e1(eri_buf, gctr, dims, envs_cint, cache);
        } else {
                NPdset0(eri_buf, dijkl);
        }
        return has_value;
}

int PBCint2e_sph(double *eri_buf, int *cell0_shls, int *bvk_cells, int cutoff,
                 float *rij_cond, float *rkl_cond,
                 CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 1, 1};
        PBCinit_int2e_EnvVars(envs_cint, ng, cell0_shls, envs_bvk);
        envs_cint->f_gout = &CINTgout2e;
#ifdef QCINT_VERSION
        envs_cint->f_gout_simd1 = &CINTgout2e_simd1;
#endif

        int *x_ctr = envs_cint->x_ctr;
        int ncomp = 1;
        int di = (envs_cint->i_l * 2 + 1) * x_ctr[0];
        int dj = (envs_cint->j_l * 2 + 1) * x_ctr[1];
        int dk = (envs_cint->k_l * 2 + 1) * x_ctr[2];
        int dl = (envs_cint->l_l * 2 + 1) * x_ctr[3];
        size_t dijkl = (size_t)di * dj * dk * dl * ncomp;
        size_t nc = x_ctr[0] * x_ctr[1] * x_ctr[2] * x_ctr[3];
        double *gctr = eri_buf + dijkl;
        double *cache = gctr + (size_t)envs_cint->nf * nc * ncomp;
        int has_value = PBCint2e_loop(gctr, cell0_shls, bvk_cells, cutoff,
                                      rij_cond, rkl_cond, envs_cint, envs_bvk, cache);
        if (has_value) {
                int dims[4] = {di, dj, dk, dl};
                c2s_sph_2e1(eri_buf, gctr, dims, envs_cint, cache);
        } else {
                NPdset0(eri_buf, dijkl);
        }
        return has_value;
}
