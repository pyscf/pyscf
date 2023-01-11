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
#include <stdint.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "pbc/pbc.h"

void CINTinit_int3c2e_EnvVars(CINTEnvVars *envs, int *ng, int *shls,
                              int *atm, int natm, int *bas, int nbas, double *env);
extern void CINTgout2e();
#ifdef QCINT_VERSION
extern void CINTgout2e_simd1();
#endif
int CINT3c2e_loop_nopt(double *gctr, CINTEnvVars *envs, double *cache, int *empty);
int CINT3c2e_loop(double *gctr, CINTEnvVars *envs, double *cache, int *empty);
int CINT3c2e_111_loop(double *gctr, CINTEnvVars *envs, double *cache, int *empty);
void c2s_cart_3c2e1(double *out, double *gctr, int *dims, CINTEnvVars *envs, double *cache);
void c2s_sph_3c2e1(double *out, double *gctr, int *dims, CINTEnvVars *envs, double *cache);


void PBCinit_int3c2e_EnvVars(CINTEnvVars *envs, int *ng, int *cell0_shls, BVKEnvs *envs_bvk)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        // the cell0_shls points to the basis of the primitive cell, the
        // integral environment envs_cint stores supmol._bas.
        int *sh_loc = envs_bvk->sh_loc;
        int nbasp = envs_bvk->nbasp;
        int nbas_bvk = nbasp * envs_bvk->ncells;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2] - nbasp + nbas_bvk;
        int ish0 = sh_loc[ish_cell0];
        int jsh0 = sh_loc[jsh_cell0];
        int ksh0 = sh_loc[ksh_cell0];
        int shls[3] = {ish0, jsh0, ksh0};
        CINTinit_int3c2e_EnvVars(envs, ng, shls, atm, natm, bas, nbas, env);
}

static void update_int3c2e_envs(CINTEnvVars *envs, int *shls)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int i_sh = shls[0];
        int j_sh = shls[1];
        int k_sh = shls[2];
        envs->shls = shls;
        envs->ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        envs->rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        envs->rk = env + atm(PTR_COORD, bas(ATOM_OF, k_sh));

        int ibase = envs->li_ceil > envs->lj_ceil;
        if (envs->nrys_roots <= 2) {
                ibase = 0;
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

        // see g3c2e.c in libcint
        envs->rkl[0] = envs->rk[0];
        envs->rkl[1] = envs->rk[1];
        envs->rkl[2] = envs->rk[2];
        envs->rx_in_rklrx = envs->rk;
        envs->rkrl[0] = envs->rk[0];
        envs->rkrl[1] = envs->rk[1];
        envs->rkrl[2] = envs->rk[2];
}

int PBCint3c2e_loop(double *gctr, int *cell0_shls, int *bvk_cells, double cutoff,
                    CINTEnvVars *envs_cint, BVKEnvs *envs_bvk, double *cache)
{
        size_t Nbas = envs_cint->nbas;
        int nbasp = envs_bvk->nbasp;
        int nbas_bvk = nbasp * envs_bvk->ncells;
        int *sh_loc = envs_bvk->sh_loc;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int cell_i = bvk_cells[0];
        int cell_j = bvk_cells[1];
        int ish_bvk = ish_cell0 + cell_i * nbasp;
        int jsh_bvk = jsh_cell0 + cell_j * nbasp;
        int ksh_bvk = ksh_cell0 - nbasp + nbas_bvk;
        int ish0 = sh_loc[ish_bvk];
        int jsh0 = sh_loc[jsh_bvk];
        int ksh0 = sh_loc[ksh_bvk];
        int ish1 = sh_loc[ish_bvk+1];
        int jsh1 = sh_loc[jsh_bvk+1];
        int ksh1 = sh_loc[ksh_bvk+1];

        if (ish0 == ish1 || jsh0 == jsh1 || ksh0 == ksh1) {
                return 0;
        }

        int *x_ctr = envs_cint->x_ctr;
        int n_comp = envs_cint->ncomp_e1 * envs_cint->ncomp_e2 * envs_cint->ncomp_tensor;
        size_t nc = x_ctr[0] * x_ctr[1] * x_ctr[2];
        size_t dijk = (size_t)envs_cint->nf * nc * n_comp;
        int empty = 1;
        NPdset0(gctr, dijk);

        int (*intor_loop)(double *, CINTEnvVars *, double *, int *);
        if (envs_cint->opt == NULL) {
                intor_loop = &CINT3c2e_loop_nopt;
//        } else if (x_ctr[0] == 1 && x_ctr[1] == 1 && x_ctr[2] == 1 && x_ctr[3] == 1) {
//                intor_loop = &CINT3c2e_111_loop;
        } else {
                intor_loop = &CINT3c2e_loop;
        }

        int8_t *ovlp_mask = envs_bvk->ovlp_mask;
        // To skip smooth auxiliary basis if needed
        int *aux_mask = envs_bvk->bas_map;
        double *qcond = envs_bvk->q_cond;
        double *qcond_k;
        double kj_cutoff;
        int shls[3];
        int ish, jsh, ksh;

        if (qcond != NULL) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                        // (ksh - nbas) because ksh the basis in auxcell was appended to
                        // the end of supmol basis
                        if (!aux_mask[ksh]) {
                                continue;
                        }
                        shls[2] = ksh;
                        qcond_k = qcond + (ksh - Nbas) * Nbas;
                        for (ish = ish0; ish < ish1; ish++) {
                                kj_cutoff = cutoff / qcond_k[ish];
                                shls[0] = ish;
                                for (jsh = jsh0; jsh < jsh1; jsh++) {
                                        if (!ovlp_mask[ish*Nbas+jsh] ||
                                            qcond_k[jsh] < kj_cutoff) {
                                                continue;
                                        }
                                        shls[1] = jsh;
                                        update_int3c2e_envs(envs_cint, shls);
                                        (*intor_loop)(gctr, envs_cint, cache, &empty);
                                }
                        }
                }
        } else {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                        // (ksh - nbas) because ksh the basis in auxcell was appended to
                        // the end of supmol basis
                        if (!aux_mask[ksh]) {
                                continue;
                        }
                        shls[2] = ksh;
                        for (ish = ish0; ish < ish1; ish++) {
                                shls[0] = ish;
                                for (jsh = jsh0; jsh < jsh1; jsh++) {
                                        if (!ovlp_mask[ish*Nbas+jsh]) {
                                                continue;
                                        }
                                        shls[1] = jsh;
                                        update_int3c2e_envs(envs_cint, shls);
                                        (*intor_loop)(gctr, envs_cint, cache, &empty);
                                }
                        }
                }
        }
        return !empty;
}

// envs_cint are updated in this function. It needs to be allocated
// omp-privately
int PBCint3c2e_cart(double *eri_buf, int *cell0_shls, int *bvk_cells, double cutoff,
                    CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 1, 1};
        PBCinit_int3c2e_EnvVars(envs_cint, ng, cell0_shls, envs_bvk);
        envs_cint->f_gout = &CINTgout2e;
#ifdef QCINT_VERSION
        envs_cint->f_gout_simd1 = &CINTgout2e_simd1;
#endif

        int *x_ctr = envs_cint->x_ctr;
        int ncomp = 1;
        int di = envs_cint->nfi * x_ctr[0];
        int dj = envs_cint->nfj * x_ctr[1];
        int dk = envs_cint->nfk * x_ctr[2];
        size_t dijk = (size_t)di * dj * dk * ncomp;
        double *gctr = eri_buf + dijk;
        double *cache = gctr + dijk;
        int has_value = PBCint3c2e_loop(gctr, cell0_shls, bvk_cells, cutoff,
                                        envs_cint, envs_bvk, cache);
        if (has_value) {
                int dims[3] = {di, dj, dk};
                c2s_cart_3c2e1(eri_buf, gctr, dims, envs_cint, cache);
        } else {
                NPdset0(eri_buf, dijk);
        }
        return has_value;
}

int PBCint3c2e_sph(double *eri_buf, int *cell0_shls, int *bvk_cells, double cutoff,
                   CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 1, 1};
        PBCinit_int3c2e_EnvVars(envs_cint, ng, cell0_shls, envs_bvk);
        envs_cint->f_gout = &CINTgout2e;
#ifdef QCINT_VERSION
        envs_cint->f_gout_simd1 = &CINTgout2e_simd1;
#endif

        int *x_ctr = envs_cint->x_ctr;
        int ncomp = 1;
        int di = (envs_cint->i_l * 2 + 1) * x_ctr[0];
        int dj = (envs_cint->j_l * 2 + 1) * x_ctr[1];
        int dk = (envs_cint->k_l * 2 + 1) * x_ctr[2];
        size_t dijk = (size_t)di * dj * dk * ncomp;
        size_t nc = x_ctr[0] * x_ctr[1] * x_ctr[2];
        double *gctr = eri_buf + dijk;
        double *cache = gctr + (size_t)envs_cint->nf * nc * ncomp;
        int has_value = PBCint3c2e_loop(gctr, cell0_shls, bvk_cells, cutoff,
                                        envs_cint, envs_bvk, cache);
        if (has_value) {
                int dims[3] = {di, dj, dk};
                c2s_sph_3c2e1(eri_buf, gctr, dims, envs_cint, cache);
        } else {
                NPdset0(eri_buf, dijk);
        }
        return has_value;
}
