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
        int *sh_loc = envs_bvk->sh_loc;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int ish0 = sh_loc[ish_cell0];
        int jsh0 = sh_loc[jsh_cell0];
        int ksh0 = sh_loc[ksh_cell0];
        int lsh0 = sh_loc[lsh_cell0];
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
        if (envs->nrys_roots <= 2) {
                ibase = 0;
                kbase = 0;
        }

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

int PBCint2e_loop(double *gctr, int *cell0_shls, int *bvk_cells, double cutoff,
                  CINTEnvVars *envs_cint, BVKEnvs *envs_bvk, double *cache)
{
        size_t Nbas = envs_cint->nbas;
        int nbasp = envs_bvk->nbasp;
        int *sh_loc = envs_bvk->sh_loc;
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
        int ish0 = sh_loc[ish_bvk];
        int jsh0 = sh_loc[jsh_bvk];
        int ksh0 = sh_loc[ksh_bvk];
        int lsh0 = sh_loc[lsh_bvk];
        int ish1 = sh_loc[ish_bvk+1];
        int jsh1 = sh_loc[jsh_bvk+1];
        int ksh1 = sh_loc[ksh_bvk+1];
        int lsh1 = sh_loc[lsh_bvk+1];

        if (ish0 == ish1 || jsh0 == jsh1 || ksh0 == ksh1 || lsh0 == lsh1) {
                return 0;
        }

        int *x_ctr = envs_cint->x_ctr;
        int ncomp = envs_cint->ncomp_e1 * envs_cint->ncomp_e2 * envs_cint->ncomp_tensor;
        size_t nc = x_ctr[0] * x_ctr[1] * x_ctr[2] * x_ctr[3];
        size_t dijkl = (size_t)envs_cint->nf * nc * ncomp;
        int empty = 1;
        NPdset0(gctr, dijkl);

        int (*intor_loop)(double *, CINTEnvVars *, double *, int *);
        if (envs_cint->opt == NULL) {
                intor_loop = &CINT2e_loop_nopt;
        } else if (x_ctr[0] == 1 && x_ctr[1] == 1 && x_ctr[2] == 1 && x_ctr[3] == 1) {
                intor_loop = &CINT2e_1111_loop;
        } else {
                intor_loop = &CINT2e_loop;
        }

        int *bas_map = envs_bvk->bas_map;
        int nimgs = envs_bvk->nimgs;
        double *qcond_ijij = envs_bvk->q_cond;
        double *qcond_iijj = envs_bvk->q_cond + Nbas * Nbas;
        double *qcond_ij, *qcond_kl, *qcond_ik, *qcond_jk;
        int shls[4];
        int ish, jsh, ksh, lsh;
        double kl_cutoff, jl_cutoff, il_cutoff;

        if (qcond_ijij != NULL) {
                for (ish = ish0; ish < ish1; ish++) {
                        // Must be the primitive cell
                        if (bas_map[ish] % nimgs != 0) {
                                continue;
                        }
                        shls[0] = ish;
                        qcond_ij = qcond_ijij + ish * Nbas;
                        qcond_ik = qcond_iijj + ish * Nbas;
                        for (jsh = jsh0; jsh < jsh1; jsh++) {
                                if (qcond_ij[jsh] < cutoff) {
                                        continue;
                                }
                                shls[1] = jsh;
                                kl_cutoff = cutoff / qcond_ij[jsh];
                                qcond_jk = qcond_iijj + jsh * Nbas;
                                for (ksh = ksh0; ksh < ksh1; ksh++) {
                                        if (qcond_ik[ksh] < cutoff ||
                                            qcond_jk[ksh] < cutoff) {
                                                continue;
                                        }
                                        shls[2] = ksh;
                                        qcond_kl = qcond_ijij + ksh * Nbas;
                                        jl_cutoff = cutoff / qcond_ik[ksh];
                                        il_cutoff = cutoff / qcond_jk[ksh];
                                        for (lsh = lsh0; lsh < lsh1; lsh++) {
                                                if (qcond_kl[lsh] < kl_cutoff ||
                                                    qcond_jk[lsh] < jl_cutoff ||
                                                    qcond_ik[lsh] < il_cutoff) {
                                                        continue;
                                                }
                                                shls[3] = lsh;
                                                update_int2e_envs(envs_cint, shls);
                                                (*intor_loop)(gctr, envs_cint, cache, &empty);
                                        }
                                }
                        }
                }
        } else {
                for (ish = ish0; ish < ish1; ish++) {
                        if (bas_map[ish] % nimgs != 0) {
                                continue;
                        }
                        shls[0] = ish;
                        for (jsh = jsh0; jsh < jsh1; jsh++) {
                                shls[1] = jsh;
                                for (ksh = ksh0; ksh < ksh1; ksh++) {
                                        shls[2] = ksh;
                                        for (lsh = lsh0; lsh < lsh1; lsh++) {
                                                shls[3] = lsh;
                                                update_int2e_envs(envs_cint, shls);
                                                (*intor_loop)(gctr, envs_cint, cache, &empty);
                                        }
                                }
                        }
                }
        }
        return !empty;
}

// envs_cint are updated in this function. It needs to be allocated
// omp-privately
int PBCint2e_cart(double *eri_buf, int *cell0_shls, int *bvk_cells, double cutoff,
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
                                      envs_cint, envs_bvk, cache);
        if (has_value) {
                int dims[4] = {di, dj, dk, dl};
                c2s_cart_2e1(eri_buf, gctr, dims, envs_cint, cache);
        } else {
                NPdset0(eri_buf, dijkl);
        }
        return has_value;
}

int PBCint2e_sph(double *eri_buf, int *cell0_shls, int *bvk_cells, double cutoff,
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
                                      envs_cint, envs_bvk, cache);
        if (has_value) {
                int dims[4] = {di, dj, dk, dl};
                c2s_sph_2e1(eri_buf, gctr, dims, envs_cint, cache);
        } else {
                NPdset0(eri_buf, dijkl);
        }
        return has_value;
}
