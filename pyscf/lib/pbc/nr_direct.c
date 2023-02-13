/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
  
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
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "gto/gto.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "pbc/pbc.h"

#define PI3HALF 5.568327996831708

void PBCminimal_CINTEnvVars(CINTEnvVars *envs, int *atm, int natm, int *bas, int nbas, double *env,
                            CINTOpt *cintopt);

void PBCVHF_contract_k_s1(int (*intor)(), double *vk, double *dms, double *buf,
                          int *cell0_shls, int *bvk_cells,
                          int *dm_translation, int n_dm, float *dm_cond,
                          float *rij_cond, float *rkl_cond,
                          CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbands = envs_bvk->nbands;
        int nbasp = envs_bvk->nbasp;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int dm_jk_off = dm_translation[cell_j * bvk_ncells + cell_k];
        size_t Nbasp = nbasp;
        size_t nn0 = nbasp * nbasp;
        float log_cutoff = envs_bvk->cutoff;
        float dm_jk_cond = dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        if (dm_jk_cond < log_cutoff) {
                return;
        }
        log_cutoff -= dm_jk_cond;
        if (!(*intor)(buf, cell0_shls, bvk_cells, log_cutoff,
                      rij_cond, rkl_cond, envs_cint, envs_bvk)) {
                return;
        }

        int *cell0_ao_loc = envs_bvk->ao_loc;
        size_t naop = cell0_ao_loc[nbasp];
        size_t nn = naop * naop;
        size_t bn = naop * nbands;
        size_t knn = nn * bvk_ncells;
        size_t bnn = bn * naop;
        int i0 = cell0_ao_loc[ish_cell0];
        int j0 = cell0_ao_loc[jsh_cell0];
        int k0 = cell0_ao_loc[ksh_cell0];
        int l0 = cell0_ao_loc[lsh_cell0];
        int i1 = cell0_ao_loc[ish_cell0+1];
        int j1 = cell0_ao_loc[jsh_cell0+1];
        int k1 = cell0_ao_loc[ksh_cell0+1];
        int l1 = cell0_ao_loc[lsh_cell0+1];
        int idm, i, j, k, l, n;
        double sjk, qijkl;
        double *dm_jk;
        vk += cell_l * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                n = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        sjk = dm_jk[j*naop+k];
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vk[i*bn+l] += qijkl * sjk;
                        } } }
                }
                vk += bnn;
        }
}

static void contract_k_s2_kgtl(int (*intor)(), double *vk, double *dms, double *buf,
                               int *cell0_shls, int *bvk_cells,
                               int *dm_translation, int n_dm, float *dm_cond,
                               float *rij_cond, float *rkl_cond,
                               CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbands = envs_bvk->nbands;
        int nbasp = envs_bvk->nbasp;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int dm_jk_off = dm_translation[cell_j*bvk_ncells+cell_k];
        int dm_jl_off = dm_translation[cell_j*bvk_ncells+cell_l];
        size_t Nbasp = nbasp;
        size_t nn0 = Nbasp * Nbasp;
        float log_cutoff = envs_bvk->cutoff;
        float dm_jk_cond = dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        float dm_jl_cond = dm_cond[dm_jl_off*nn0 + jsh_cell0*Nbasp+lsh_cell0];
        float dm_cond_max = MAX(dm_jk_cond, dm_jl_cond);
        if (dm_cond_max < log_cutoff) {
                return;
        }
        log_cutoff -= dm_cond_max;
        if (!(*intor)(buf, cell0_shls, bvk_cells, log_cutoff,
                      rij_cond, rkl_cond, envs_cint, envs_bvk)) {
                return;
        }

        int *cell0_ao_loc = envs_bvk->ao_loc;
        size_t naop = cell0_ao_loc[nbasp];
        size_t nn = naop * naop;
        size_t bn = naop * nbands;
        size_t knn = nn * bvk_ncells;
        size_t bnn = bn * naop;
        int i0 = cell0_ao_loc[ish_cell0];
        int j0 = cell0_ao_loc[jsh_cell0];
        int k0 = cell0_ao_loc[ksh_cell0];
        int l0 = cell0_ao_loc[lsh_cell0];
        int i1 = cell0_ao_loc[ish_cell0+1];
        int j1 = cell0_ao_loc[jsh_cell0+1];
        int k1 = cell0_ao_loc[ksh_cell0+1];
        int l1 = cell0_ao_loc[lsh_cell0+1];
        int idm, i, j, k, l, n;
        double sjk, sjl, qijkl;
        double *dm_jk, *dm_jl;
        double *vk_ik = vk + cell_k * naop;
        double *vk_il = vk + cell_l * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                dm_jl = dms + dm_jl_off * nn + idm * knn;
                n = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        sjk = dm_jk[j*naop+k];
                        sjl = dm_jl[j*naop+l];
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vk_il[i*bn+l] += qijkl * sjk;
                                vk_ik[i*bn+k] += qijkl * sjl;
                        } } }
                }
                vk_ik += bnn;
                vk_il += bnn;
        }
}

void PBCVHF_contract_k_s2kl(int (*intor)(), double *vk, double *dms, double *buf,
                            int *cell0_shls, int *bvk_cells,
                            int *dm_translation, int n_dm, float *dm_cond,
                            float *rij_cond, float *rkl_cond,
                            CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int ksh = cell0_shls[2] + bvk_cells[2] * nbasp;
        int lsh = cell0_shls[3] + bvk_cells[3] * nbasp;
        if (ksh > lsh) {
                contract_k_s2_kgtl(intor, vk, dms, buf, cell0_shls, bvk_cells,
                                   dm_translation, n_dm, dm_cond,
                                   rij_cond, rkl_cond, envs_cint, envs_bvk);
        } else if (ksh == lsh) {
                PBCVHF_contract_k_s1(intor, vk, dms, buf, cell0_shls, bvk_cells,
                                     dm_translation, n_dm, dm_cond,
                                     rij_cond, rkl_cond, envs_cint, envs_bvk);
        }
}

//dist[lsh-lsh0] += xij[lsh0] * xkl[lsh0] + yij[lsh0] * ykl[lsh0] + zij[lsh0] * zkl[lsh0];
//
//fac = (theta_ij+theta_kl)*dist[lsh] + log(prec) - log(s_ij[jsh]) - log(s_kl[kl])
//
//if (log(dm_ij) > fac || log(dm_kl) > fac ||
//    (log(dm_ik) > fac || log(dm_il) > fac || log(dm_jk) > fac || log(dm_jl) > fac):
//        continue
void PBCVHF_contract_j_s1(int (*intor)(), double *vj, double *dms, double *buf,
                          int *cell0_shls, int *bvk_cells,
                          int *dm_translation, int n_dm, float *dm_cond,
                          float *rij_cond, float *rkl_cond,
                          CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbands = envs_bvk->nbands;
        int nbasp = envs_bvk->nbasp;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int dm_lk_off = dm_translation[cell_l * bvk_ncells + cell_k];
        size_t Nbasp = nbasp;
        size_t nn0 = Nbasp * Nbasp;
        float log_cutoff = envs_bvk->cutoff;
        float dm_lk_cond = dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        if (dm_lk_cond < log_cutoff) {
                return;
        }
        log_cutoff -= dm_lk_cond;
        if (!(*intor)(buf, cell0_shls, bvk_cells, log_cutoff,
                      rij_cond, rkl_cond, envs_cint, envs_bvk)) {
                return;
        }

        int *cell0_ao_loc = envs_bvk->ao_loc;
        size_t naop = cell0_ao_loc[nbasp];
        size_t nn = naop * naop;
        size_t bn = naop * nbands;
        size_t knn = nn * bvk_ncells;
        size_t bnn = bn * naop;
        int i0 = cell0_ao_loc[ish_cell0];
        int j0 = cell0_ao_loc[jsh_cell0];
        int k0 = cell0_ao_loc[ksh_cell0];
        int l0 = cell0_ao_loc[lsh_cell0];
        int i1 = cell0_ao_loc[ish_cell0+1];
        int j1 = cell0_ao_loc[jsh_cell0+1];
        int k1 = cell0_ao_loc[ksh_cell0+1];
        int l1 = cell0_ao_loc[lsh_cell0+1];
        int idm, i, j, k, l, n;
        double slk, qijkl;
        double *dm_lk;
        vj += cell_j * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                n = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        slk = dm_lk[l*naop+k];
                        for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vj[i*bn+j] += qijkl * slk;
                        } }
                } }
                vj += bnn;
        }
}

static void contract_j_s2_kgtl(int (*intor)(), double *vj, double *dms, double *buf,
                               int *cell0_shls, int *bvk_cells,
                               int *dm_translation, int n_dm, float *dm_cond,
                               float *rij_cond, float *rkl_cond,
                               CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbands = envs_bvk->nbands;
        int nbasp = envs_bvk->nbasp;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int dm_lk_off = dm_translation[cell_l * bvk_ncells + cell_k];
        int dm_kl_off = dm_translation[cell_k * bvk_ncells + cell_l];
        size_t Nbasp = nbasp;
        size_t nn0 = Nbasp * Nbasp;
        float log_cutoff = envs_bvk->cutoff;
        float dm_lk_cond = dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        float dm_kl_cond = dm_cond[dm_kl_off*nn0 + ksh_cell0*Nbasp+lsh_cell0];
        float dm_cond_max = MAX(dm_lk_cond, dm_kl_cond);
        if (dm_cond_max < log_cutoff) {
                return;
        }
        log_cutoff -= dm_cond_max;
        if (!(*intor)(buf, cell0_shls, bvk_cells, log_cutoff,
                      rij_cond, rkl_cond, envs_cint, envs_bvk)) {
                return;
        }

        int *cell0_ao_loc = envs_bvk->ao_loc;
        size_t naop = cell0_ao_loc[nbasp];
        size_t nn = naop * naop;
        size_t bn = naop * nbands;
        size_t knn = nn * bvk_ncells;
        size_t bnn = bn * naop;
        int i0 = cell0_ao_loc[ish_cell0];
        int j0 = cell0_ao_loc[jsh_cell0];
        int k0 = cell0_ao_loc[ksh_cell0];
        int l0 = cell0_ao_loc[lsh_cell0];
        int i1 = cell0_ao_loc[ish_cell0+1];
        int j1 = cell0_ao_loc[jsh_cell0+1];
        int k1 = cell0_ao_loc[ksh_cell0+1];
        int l1 = cell0_ao_loc[lsh_cell0+1];
        int idm, i, j, k, l, n;
        double slk, qijkl;
        double *dm_lk, *dm_kl;
        vj += cell_j * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                dm_kl = dms + dm_kl_off * nn + idm * knn;
                n = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        slk = dm_lk[l*naop+k] + dm_kl[k*naop+l];
                        for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vj[i*bn+j] += qijkl * slk;
                        } }
                } }
                vj += bnn;
        }
}

void PBCVHF_contract_j_s2kl(int (*intor)(), double *vj, double *dms, double *buf,
                            int *cell0_shls, int *bvk_cells,
                            int *dm_translation, int n_dm, float *dm_cond,
                            float *rij_cond, float *rkl_cond,
                            CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int ksh = cell0_shls[2] + bvk_cells[2] * nbasp;
        int lsh = cell0_shls[3] + bvk_cells[3] * nbasp;
        if (ksh > lsh) {
                contract_j_s2_kgtl(intor, vj, dms, buf, cell0_shls, bvk_cells,
                                   dm_translation, n_dm, dm_cond,
                                   rij_cond, rkl_cond, envs_cint, envs_bvk);
        } else if (ksh == lsh) {
                PBCVHF_contract_j_s1(intor, vj, dms, buf, cell0_shls, bvk_cells,
                                     dm_translation, n_dm, dm_cond,
                                     rij_cond, rkl_cond, envs_cint, envs_bvk);
        }
}

void PBCVHF_contract_jk_s1(int (*intor)(), double *jk, double *dms, double *buf,
                           int *cell0_shls, int *bvk_cells,
                           int *dm_translation, int n_dm, float *dm_cond,
                           float *rij_cond, float *rkl_cond,
                           CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbands = envs_bvk->nbands;
        int nbasp = envs_bvk->nbasp;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int dm_lk_off = dm_translation[cell_l * bvk_ncells + cell_k];
        int dm_jk_off = dm_translation[cell_j * bvk_ncells + cell_k];
        size_t Nbasp = nbasp;
        size_t nn0 = Nbasp * Nbasp;
        float log_cutoff = envs_bvk->cutoff;
        float dm_lk_cond = dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        float dm_jk_cond = dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        float dm_cond_max = MAX(dm_lk_cond, dm_jk_cond);
        if (dm_cond_max < log_cutoff) {
                return;
        }
        log_cutoff -= dm_cond_max;
        if (!(*intor)(buf, cell0_shls, bvk_cells, log_cutoff,
                      rij_cond, rkl_cond, envs_cint, envs_bvk)) {
                return;
        }

        int *cell0_ao_loc = envs_bvk->ao_loc;
        size_t naop = cell0_ao_loc[nbasp];
        size_t nn = naop * naop;
        size_t bn = naop * nbands;
        size_t knn = nn * bvk_ncells;
        size_t bnn = bn * naop;
        int i0 = cell0_ao_loc[ish_cell0];
        int j0 = cell0_ao_loc[jsh_cell0];
        int k0 = cell0_ao_loc[ksh_cell0];
        int l0 = cell0_ao_loc[lsh_cell0];
        int i1 = cell0_ao_loc[ish_cell0+1];
        int j1 = cell0_ao_loc[jsh_cell0+1];
        int k1 = cell0_ao_loc[ksh_cell0+1];
        int l1 = cell0_ao_loc[lsh_cell0+1];
        double *vj = jk + cell_j * naop;
        double *vk = jk + n_dm * bnn + cell_l * naop;
        int idm, i, j, k, l, n;
        double slk, sjk, qijkl;
        double *dm_lk, *dm_jk;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                n = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        slk = dm_lk[l*naop+k];
                        for (j = j0; j < j1; j++) {
                                sjk = dm_jk[j*naop+k];
                                for (i = i0; i < i1; i++, n++) {
                                        qijkl = buf[n];
                                        vj[i*bn+j] += qijkl * slk;
                                        vk[i*bn+l] += qijkl * sjk;
                                }
                        }
                } }
                vj += bnn;
                vk += bnn;
        }
}

static void contract_jk_s2_kgtl(int (*intor)(), double *jk, double *dms, double *buf,
                                int *cell0_shls, int *bvk_cells,
                                int *dm_translation, int n_dm, float *dm_cond,
                                float *rij_cond, float *rkl_cond,
                                CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbands = envs_bvk->nbands;
        int nbasp = envs_bvk->nbasp;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int ksh_cell0 = cell0_shls[2];
        int lsh_cell0 = cell0_shls[3];
        int cell_j = bvk_cells[1];
        int cell_k = bvk_cells[2];
        int cell_l = bvk_cells[3];
        int dm_jk_off = dm_translation[cell_j*bvk_ncells+cell_k];
        int dm_jl_off = dm_translation[cell_j*bvk_ncells+cell_l];
        int dm_lk_off = dm_translation[cell_l*bvk_ncells+cell_k];
        int dm_kl_off = dm_translation[cell_k*bvk_ncells+cell_l];
        size_t Nbasp = nbasp;
        size_t nn0 = Nbasp * Nbasp;
        float log_cutoff = envs_bvk->cutoff;
        float dm_jk_cond = dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        float dm_jl_cond = dm_cond[dm_jl_off*nn0 + jsh_cell0*Nbasp+lsh_cell0];
        float dm_lk_cond = dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        float dm_kl_cond = dm_cond[dm_kl_off*nn0 + ksh_cell0*Nbasp+lsh_cell0];
        float dm_cond_max = MAX(dm_jk_cond, dm_jl_cond);
        dm_cond_max = MAX(dm_cond_max, dm_lk_cond);
        dm_cond_max = MAX(dm_cond_max, dm_kl_cond);
        if (dm_cond_max < log_cutoff) {
                return;
        }
        log_cutoff -= dm_cond_max;
        if (!(*intor)(buf, cell0_shls, bvk_cells, log_cutoff,
                      rij_cond, rkl_cond, envs_cint, envs_bvk)) {
                return;
        }

        int *cell0_ao_loc = envs_bvk->ao_loc;
        size_t naop = cell0_ao_loc[nbasp];
        size_t nn = naop * naop;
        size_t bn = naop * nbands;
        size_t knn = nn * bvk_ncells;
        size_t bnn = bn * naop;
        int i0 = cell0_ao_loc[ish_cell0];
        int j0 = cell0_ao_loc[jsh_cell0];
        int k0 = cell0_ao_loc[ksh_cell0];
        int l0 = cell0_ao_loc[lsh_cell0];
        int i1 = cell0_ao_loc[ish_cell0+1];
        int j1 = cell0_ao_loc[jsh_cell0+1];
        int k1 = cell0_ao_loc[ksh_cell0+1];
        int l1 = cell0_ao_loc[lsh_cell0+1];
        double *vj = jk + cell_j * naop;
        double *vk_ik = jk + n_dm * bnn + cell_k * naop;
        double *vk_il = jk + n_dm * bnn + cell_l * naop;
        int idm, i, j, k, l, n;
        double sjk, sjl, slk, qijkl;
        double *dm_jk, *dm_jl, *dm_lk, *dm_kl;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                dm_kl = dms + dm_kl_off * nn + idm * knn;
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                dm_jl = dms + dm_jl_off * nn + idm * knn;
                n = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        slk = dm_lk[l*naop+k] + dm_kl[k*naop+l];
                        for (j = j0; j < j1; j++) {
                                sjk = dm_jk[j*naop+k];
                                sjl = dm_jl[j*naop+l];
                                for (i = i0; i < i1; i++, n++) {
                                        qijkl = buf[n];
                                        vj[i*bn+j] += qijkl * slk;
                                        vk_il[i*bn+l] += qijkl * sjk;
                                        vk_ik[i*bn+k] += qijkl * sjl;
                                } }
                        }
                }
                vj += bnn;
                vk_ik += bnn;
                vk_il += bnn;
        }
}

void PBCVHF_contract_jk_s2kl(int (*intor)(), double *jk, double *dms, double *buf,
                             int *cell0_shls, int *bvk_cells,
                             int *dm_translation, int n_dm, float *dm_cond,
                             float *rij_cond, float *rkl_cond,
                             CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int ksh = cell0_shls[2] + bvk_cells[2] * nbasp;
        int lsh = cell0_shls[3] + bvk_cells[3] * nbasp;
        if (ksh > lsh) {
                contract_jk_s2_kgtl(intor, jk, dms, buf, cell0_shls, bvk_cells,
                                    dm_translation, n_dm, dm_cond,
                                    rij_cond, rkl_cond, envs_cint, envs_bvk);
        } else if (ksh == lsh) {
                PBCVHF_contract_jk_s1(intor, jk, dms, buf, cell0_shls, bvk_cells,
                                      dm_translation, n_dm, dm_cond,
                                      rij_cond, rkl_cond, envs_cint, envs_bvk);
        }
}

static void approx_bvk_rcond0(float *rcond, int ish0, int ish1, BVKEnvs *envs_bvk,
                              int *atm, int natm, int *bas, int nbas, double *env)
{
        int nbasp = envs_bvk->nbasp;
        int nbas_bvk = nbasp * envs_bvk->ncells;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int iseg0 = seg_loc[ish0];
        int iseg1 = seg_loc[ish1];
        int jseg0 = 0;
        int jseg1 = seg_loc[nbas_bvk];
        int rs_cell_nbas = seg_loc[nbasp];
        float *xcond = rcond;
        float *ycond = rcond + rs_cell_nbas * nbas;
        float *zcond = rcond + rs_cell_nbas * nbas * 2;
        float *cache = malloc(sizeof(float) * nbas*3);
        float *xj = cache;
        float *yj = cache + nbas;
        float *zj = cache + nbas * 2;
        int ish, jsh, iseg, jseg, jsh0, jsh1;
        int ptr_coord, n;
        float ai, aj, aij, ci, cj, xi, yi, zi, xci, yci, zci;

        for (jsh = 0; jsh < nbas; jsh++) {
                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, jsh));
                xj[jsh] = env[ptr_coord+0];
                yj[jsh] = env[ptr_coord+1];
                zj[jsh] = env[ptr_coord+2];
        }

        for (iseg = iseg0; iseg < iseg1; iseg++) {
                ish = seg2sh[iseg];
                ai = env[bas(PTR_EXP, ish) + bas(NPRIM_OF, ish) - 1];
                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, ish));
                xi = env[ptr_coord+0];
                yi = env[ptr_coord+1];
                zi = env[ptr_coord+2];
                for (jseg = jseg0; jseg < jseg1; jseg++) {
                        jsh0 = seg2sh[jseg];
                        jsh1 = seg2sh[jseg+1];
                        aj = env[bas(PTR_EXP, jsh0) + bas(NPRIM_OF, jsh0) - 1];
                        aij = ai + aj;
                        ci = ai / aij;
                        cj = aj / aij;
                        xci = ci * xi;
                        yci = ci * yi;
                        zci = ci * zi;
#pragma GCC ivdep
                        for (jsh = jsh0; jsh < jsh1; jsh++) {
                                n = iseg * nbas + jsh;
                                xcond[n] = xci + cj * xj[jsh];
                                ycond[n] = yci + cj * yj[jsh];
                                zcond[n] = zci + cj * zj[jsh];
                        }
                }
        }
        free(cache);
}

void PBCapprox_bvk_rcond(float *rcond, int ish_bvk, int jsh_bvk, BVKEnvs *envs_bvk,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         float *cache)
{
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int iseg0 = seg_loc[ish_bvk];
        int jseg0 = seg_loc[jsh_bvk];
        int iseg1 = seg_loc[ish_bvk+1];
        int jseg1 = seg_loc[jsh_bvk+1];
        int nish = seg2sh[iseg1] - seg2sh[iseg0];
        int njsh = seg2sh[jseg1] - seg2sh[jseg0];
        int nij = nish * njsh;
        int ioff = seg2sh[iseg0];
        int joff = seg2sh[jseg0];
        int rij_off = ioff * njsh + joff;
        float *xcond = rcond;
        float *ycond = rcond + nij;
        float *zcond = rcond + nij * 2;
        float ai, aj, aij, ci, cj;
        float xci, yci, zci;
        float *xj = cache;
        float *yj = cache + njsh;
        float *zj = cache + njsh * 2;
        int iseg, jseg, ish, jsh;
        int ptr_coord, n;

        int ish0 = seg2sh[iseg0];
        int jsh0 = seg2sh[jseg0];
        int ish1 = seg2sh[iseg1];
        int jsh1 = seg2sh[jseg1];

        jsh0 = seg2sh[jseg0];
        jsh1 = seg2sh[jseg1];
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, jsh));
                xj[jsh-jsh0] = env[ptr_coord+0];
                yj[jsh-jsh0] = env[ptr_coord+1];
                zj[jsh-jsh0] = env[ptr_coord+2];
        }

        for (iseg = iseg0; iseg < iseg1; iseg++) {
                ish0 = seg2sh[iseg];
                ish1 = seg2sh[iseg+1];
                ai = env[bas(PTR_EXP, ish0) + bas(NPRIM_OF, ish0) - 1];
                for (jseg = jseg0; jseg < jseg1; jseg++) {
                        jsh0 = seg2sh[jseg];
                        jsh1 = seg2sh[jseg+1];
                        aj = env[bas(PTR_EXP, jsh0) + bas(NPRIM_OF, jsh0) - 1];
                        aij = ai + aj;
                        ci = ai / aij;
                        cj = aj / aij;
                        for (ish = ish0; ish < ish1; ish++) {
                                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, ish));
                                xci = ci * env[ptr_coord+0];
                                yci = ci * env[ptr_coord+1];
                                zci = ci * env[ptr_coord+2];
#pragma GCC ivdep
                                for (jsh = jsh0; jsh < jsh1; jsh++) {
                                        n = ish * njsh + jsh - rij_off;
                                        xcond[n] = xci + cj * xj[jsh-joff];
                                        ycond[n] = yci + cj * yj[jsh-joff];
                                        zcond[n] = zci + cj * zj[jsh-joff];
                                }
                        }
                }
        }
}

static float _max_q_cond(float *qcond, size_t Nbas, int ish0,
                        int ish1, int jsh0, int jsh1)
{
        float q_max = -500.f;
        int ish, jsh;
        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                q_max = MAX(q_max, qcond[ish*Nbas+jsh]);
        } }
        return q_max;
}

static float _max_dm_cond(float *dm_cond, BVKEnvs *envs_bvk)
{
        int bvk_ncells = envs_bvk->ncells;
        int nbasp = envs_bvk->nbasp;
        int i;
        float dm_max = -500.f;
        for (i = 0; i < nbasp*nbasp*bvk_ncells; i++) {
                dm_max = MAX(dm_max, dm_cond[i]);
        }
        return dm_max;
}

static void qcond_abstract(float *cond_abs, float *qcond, size_t Nbas,
                           BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int nbas_bvk = nbasp * envs_bvk->ncells;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        float q_max;
        int ish_bvk, jsh_bvk;
        int ish0, ish1, jsh0, jsh1;
        for (ish_bvk = 0; ish_bvk < nbas_bvk; ish_bvk++) {
        for (jsh_bvk = 0; jsh_bvk <= ish_bvk; jsh_bvk++) {
                ish0 = seg2sh[seg_loc[ish_bvk]];
                jsh0 = seg2sh[seg_loc[jsh_bvk]];
                ish1 = seg2sh[seg_loc[ish_bvk+1]];
                jsh1 = seg2sh[seg_loc[jsh_bvk+1]];
                q_max = _max_q_cond(qcond, Nbas, ish0, ish1, jsh0, jsh1);
                cond_abs[ish_bvk*nbas_bvk+jsh_bvk] = q_max;
                cond_abs[jsh_bvk*nbas_bvk+ish_bvk] = q_max;
        } }
}

/*
 * shls_slice refers to the shells in the bvk supcell
 * bvk_ao_loc are ao_locs of bvk-cell basis appeared in supmol (some basis are removed)
 * nbasp is the number of basis in primitive cell
 * dm_translation utilizes the translation symmetry for density matrices (wrt the full bvk-cell)
 * DM[M,N] = DM[N-M] by mapping the 2D subscripts to 1D subscripts
 */
void PBCVHF_direct_drv(void (*fdot)(), int (*intor)(),
                       double *out, double *dms, int size_v, int n_dm,
                       int bvk_ncells, int nimgs,
                       int nkpts, int nbands, int nbasp, int comp,
                       int *seg_loc, int *seg2sh, int *cell0_ao_loc,
                       int *cell0_bastype, int *shls_slice, int *dm_translation,
                       float *q_cond, float *dm_cond, float log_cutoff,
                       float *qcell0_ijij, float *qcell0_iijj, int *ish_idx, int *jsh_idx,
                       CINTOpt *cintopt, int cache_size,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        size_t ish0 = shls_slice[0];
        size_t ish1 = shls_slice[1];
        size_t jsh0 = shls_slice[2];
        size_t jsh1 = shls_slice[3];
        size_t ksh0 = shls_slice[4];
        size_t ksh1 = shls_slice[5];
        size_t lsh0 = shls_slice[6];
        size_t lsh1 = shls_slice[7];
        size_t nish = ish1 - ish0;
        size_t njsh = jsh1 - jsh0;
        size_t nksh = ksh1 - ksh0;
        size_t nlsh = lsh1 - lsh0;
        size_t nkl = nksh * nlsh;
        BVKEnvs envs_bvk = {bvk_ncells, nimgs,
                nkpts, nbands, nbasp, comp, 0, 0,
                seg_loc, seg2sh, cell0_ao_loc, shls_slice, NULL, NULL, NULL,
                NULL, q_cond, log_cutoff};

        int rs_cell_nbas = seg_loc[nbasp];
        float *rij_cond = malloc(sizeof(float) * rs_cell_nbas*nbas*3);
        approx_bvk_rcond0(rij_cond, ish0, ish1, &envs_bvk,
                          atm, natm, bas, nbas, env);
        int dsh_max = nimgs * 3;
        assert(env[PTR_RANGE_OMEGA] != 0);

        size_t Nbas = nbas;
        size_t nbas_bvk = nbasp * nbands;
        float *qcond_iijj = malloc(sizeof(float) * nbas_bvk * nbas_bvk);
        qcond_abstract(qcond_iijj, q_cond+Nbas*Nbas, Nbas, &envs_bvk);

        float dm_cond_max = _max_dm_cond(dm_cond, &envs_bvk);
        log_cutoff -= MIN(dm_cond_max, 0);

#pragma omp parallel
{
        size_t kl, n;
        int ij, i, j, k, l;
        int bvk_cells[4] = {0};
        int cell0_shls[4];
        CINTEnvVars envs_cint;
        PBCminimal_CINTEnvVars(&envs_cint, atm, natm, bas, nbas, env, cintopt);

        double *v_priv = calloc(size_v, sizeof(double));
        double *buf = malloc(sizeof(double) * MAX(cache_size, dsh_max*3));
        float *rkl_cond = malloc(sizeof(float) * dsh_max*dsh_max*3);
        float *qk, *ql, *qcell0k, *qcell0l;
        float kl_cutoff, qklkl_max;

#pragma omp for schedule(dynamic, 1)
        for (kl = 0; kl < nkl; kl++) {
                k = kl / nlsh + ksh0;
                l = kl % nlsh + lsh0;
                qklkl_max = _max_q_cond(q_cond, Nbas,
                                        seg2sh[seg_loc[k]], seg2sh[seg_loc[k+1]],
                                        seg2sh[seg_loc[l]], seg2sh[seg_loc[l+1]]);

                if (qklkl_max < log_cutoff ||
                    seg_loc[k] == seg_loc[k+1] || seg_loc[l] == seg_loc[l+1]) {
                        continue;
                }
                kl_cutoff = log_cutoff - qklkl_max;
                qk = qcond_iijj + k * nbas_bvk;
                ql = qcond_iijj + l * nbas_bvk;
                qcell0k = qcell0_iijj + k * nbasp;
                qcell0l = qcell0_iijj + l * nbasp;

                cell0_shls[2] = k % nbasp;
                cell0_shls[3] = l % nbasp;
                bvk_cells[2] = k / nbasp;
                bvk_cells[3] = l / nbasp;
                PBCapprox_bvk_rcond(rkl_cond, k, l, &envs_bvk,
                                    atm, natm, bas, nbas, env, (float *)buf);

                for (ij = 0; ij < nish * njsh; ij++) {
                        i = ish_idx[ij];
                        j = jsh_idx[ij];
                        if (qcell0_ijij[j*nbasp+i] < kl_cutoff) {
                                break;
                        }
                        if (qcell0k[i] + ql[j] < log_cutoff ||
                            qcell0l[i] + qk[j] < log_cutoff) {
                                continue;
                        }
                        cell0_shls[0] = i;
                        cell0_shls[1] = j % nbasp;
                        bvk_cells[1] = j / nbasp;
                        (*fdot)(intor, v_priv, dms, buf, cell0_shls, bvk_cells,
                                dm_translation, n_dm, dm_cond,
                                rij_cond, rkl_cond, &envs_cint, &envs_bvk);
                }
        }
#pragma omp critical
        {
                for (n = 0; n < size_v; n++) {
                        out[n] += v_priv[n];
                }
        }
        free(buf);
        free(v_priv);
        free(rkl_cond);
}
        free(rij_cond);
        free(qcond_iijj);
}

#define DIFFUSED        2
void PBCVHF_direct_drv_nodddd(
                       void (*fdot)(), int (*intor)(),
                       double *out, double *dms, int size_v, int n_dm,
                       int bvk_ncells, int nimgs,
                       int nkpts, int nbands, int nbasp, int comp,
                       int *seg_loc, int *seg2sh, int *cell0_ao_loc,
                       int *cell0_bastype, int *shls_slice, int *dm_translation,
                       float *q_cond, float *dm_cond, float log_cutoff,
                       float *qcell0_ijij, float *qcell0_iijj, int *ish_idx, int *jsh_idx,
                       CINTOpt *cintopt, int cache_size,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        size_t ish0 = shls_slice[0];
        size_t ish1 = shls_slice[1];
        size_t jsh0 = shls_slice[2];
        size_t jsh1 = shls_slice[3];
        size_t ksh0 = shls_slice[4];
        size_t ksh1 = shls_slice[5];
        size_t lsh0 = shls_slice[6];
        size_t lsh1 = shls_slice[7];
        size_t nish = ish1 - ish0;
        size_t njsh = jsh1 - jsh0;
        size_t nksh = ksh1 - ksh0;
        size_t nlsh = lsh1 - lsh0;
        size_t nkl = nksh * nlsh;
        BVKEnvs envs_bvk = {bvk_ncells, nimgs,
                nkpts, nbands, nbasp, comp, 0, 0,
                seg_loc, seg2sh, cell0_ao_loc, shls_slice, NULL, NULL, NULL,
                NULL, q_cond, log_cutoff};

        int rs_cell_nbas = seg_loc[nbasp];
        float *rij_cond = malloc(sizeof(float) * rs_cell_nbas*nbas*3);
        approx_bvk_rcond0(rij_cond, ish0, ish1, &envs_bvk,
                          atm, natm, bas, nbas, env);
        int dsh_max = nimgs * 3;
        assert(env[PTR_RANGE_OMEGA] != 0);

        size_t Nbas = nbas;
        size_t nbas_bvk = nbasp * nbands;
        float *qcond_iijj = malloc(sizeof(float) * nbas_bvk * nbas_bvk);
        qcond_abstract(qcond_iijj, q_cond+Nbas*Nbas, Nbas, &envs_bvk);

        float dm_cond_max = _max_dm_cond(dm_cond, &envs_bvk);
        log_cutoff -= MIN(dm_cond_max, 0);

        int *i_c_idx = malloc(sizeof(int) * nish * njsh * 2);
        int *j_c_idx = i_c_idx + nish * njsh;
        int nij = 0;
        int nij_c = 0;
        for (nij = 0; nij < nish * njsh; nij++) {
                int ip = ish_idx[nij];
                int jp = jsh_idx[nij];
                if (qcell0_ijij[jp*nbasp+ip] < log_cutoff) {
                        nij++;
                        break;
                }
                // exclude dd-block
                if (cell0_bastype[ip] != DIFFUSED ||
                    cell0_bastype[jp%nbasp] != DIFFUSED) {
                        i_c_idx[nij_c] = ip;
                        j_c_idx[nij_c] = jp;
                        nij_c++;
                }
        }

#pragma omp parallel
{
        size_t kl, n;
        int ij, i, j, k, l;
        int kshp, lshp;
        int bvk_cells[4] = {0};
        int cell0_shls[4];
        CINTEnvVars envs_cint;
        PBCminimal_CINTEnvVars(&envs_cint, atm, natm, bas, nbas, env, cintopt);

        double *v_priv = calloc(size_v, sizeof(double));
        double *buf = malloc(sizeof(double) * MAX(cache_size, dsh_max*3));
        float *rkl_cond = malloc(sizeof(float) * dsh_max*dsh_max*3);
        float *qk, *ql, *qcell0k, *qcell0l;
        float kl_cutoff, qklkl_max;

#pragma omp for schedule(dynamic, 1)
        for (kl = 0; kl < nkl; kl++) {
                k = kl / nlsh + ksh0;
                l = kl % nlsh + lsh0;
                qklkl_max = _max_q_cond(q_cond, Nbas,
                                        seg2sh[seg_loc[k]], seg2sh[seg_loc[k+1]],
                                        seg2sh[seg_loc[l]], seg2sh[seg_loc[l+1]]);

                if (qklkl_max < log_cutoff ||
                    seg_loc[k] == seg_loc[k+1] || seg_loc[l] == seg_loc[l+1]) {
                        continue;
                }
                kl_cutoff = log_cutoff - qklkl_max;
                qk = qcond_iijj + k * nbas_bvk;
                ql = qcond_iijj + l * nbas_bvk;
                qcell0k = qcell0_iijj + k * nbasp;
                qcell0l = qcell0_iijj + l * nbasp;

                kshp = k % nbasp;
                lshp = l % nbasp;
                cell0_shls[2] = kshp;
                cell0_shls[3] = lshp;
                bvk_cells[2] = k / nbasp;
                bvk_cells[3] = l / nbasp;
                PBCapprox_bvk_rcond(rkl_cond, k, l, &envs_bvk,
                                    atm, natm, bas, nbas, env, (float *)buf);

                if ((cell0_bastype[kshp] == DIFFUSED) &&
                    (cell0_bastype[lshp] == DIFFUSED)) {
                        // kl is in dd-block, exclude dd-block from ij
                        for (ij = 0; ij < nij_c; ij++) {
                                i = i_c_idx[ij];
                                j = j_c_idx[ij];
                                if (qcell0_ijij[j*nbasp+i] < kl_cutoff) {
                                        break;
                                }
                                if (qcell0k[i] + ql[j] < log_cutoff ||
                                    qcell0l[i] + qk[j] < log_cutoff) {
                                        continue;
                                }
                                cell0_shls[0] = i;
                                cell0_shls[1] = j % nbasp;
                                bvk_cells[1] = j / nbasp;
                                (*fdot)(intor, v_priv, dms, buf, cell0_shls, bvk_cells,
                                        dm_translation, n_dm, dm_cond,
                                        rij_cond, rkl_cond, &envs_cint, &envs_bvk);
                        }
                } else {
                        // kl is not in dd-block, ij are all basis products
                        for (ij = 0; ij < nij; ij++) {
                                i = ish_idx[ij];
                                j = jsh_idx[ij];
                                if (qcell0_ijij[j*nbasp+i] < kl_cutoff) {
                                        break;
                                }
                                if (qcell0k[i] + ql[j] < log_cutoff ||
                                    qcell0l[i] + qk[j] < log_cutoff) {
                                        continue;
                                }
                                cell0_shls[0] = i;
                                cell0_shls[1] = j % nbasp;
                                bvk_cells[1] = j / nbasp;
                                (*fdot)(intor, v_priv, dms, buf, cell0_shls, bvk_cells,
                                        dm_translation, n_dm, dm_cond,
                                        rij_cond, rkl_cond, &envs_cint, &envs_bvk);
                        }
                }
        }
#pragma omp critical
        {
                for (n = 0; n < size_v; n++) {
                        out[n] += v_priv[n];
                }
        }
        free(buf);
        free(v_priv);
        free(rkl_cond);
}
        free(i_c_idx);
        free(rij_cond);
        free(qcond_iijj);
}

static int _int2e_swap_jk(double *buf, int *dims, int *shls,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          CINTOpt *cintopt, double *cache)
{
        int shls_swap_jk[4] = {shls[0], shls[2], shls[1], shls[3]};
        return int2e_sph(buf, dims, shls_swap_jk, atm, natm, bas, nbas, env, cintopt, cache);
}

void PBCVHFsetnr_direct_scf(int (*intor)(), CINTOpt *cintopt, float *qcond,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        double *buf = (double *)malloc(sizeof(double) * Nbas2);
        float *qcond_ijij = qcond;
        float *qcond_iijj = qcond + Nbas2;
        size_t i;

        CVHFset_int2e_q_cond(intor, cintopt, buf, ao_loc,
                             atm, natm, bas, nbas, env);
        for (i = 0; i < Nbas2; i++) {
                qcond_ijij[i] = logf(buf[i]);
        }

        CVHFset_int2e_q_cond(_int2e_swap_jk, cintopt, buf, ao_loc,
                             atm, natm, bas, nbas, env);
        for (i = 0; i < Nbas2; i++) {
                qcond_iijj[i] = logf(buf[i]);
        }

        free(buf);
}

void PBCVHFsetnr_direct_scf1(int (*intor)(), CINTOpt *cintopt, uint8_t *qindex,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        uint8_t *qijij = qindex;
        uint8_t *qiijj = qindex + Nbas2;

        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        double qtmp, tmp;
        size_t ij, i, j, di, dj, dij, di2, dj2, ish, jsh;
        size_t Nbas = nbas;
        int itmp;
        int shls[4];
        double *cache = malloc(sizeof(double) * cache_size);
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *buf = malloc(sizeof(double) * di*di*di*di);
#pragma omp for schedule(dynamic)
        for (ish = 0; ish < Nbas; ish++) {
                for (jsh = 0; jsh <= ish; jsh++) {
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        dij = di * dj;
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ish;
                        shls[3] = jsh;
                        qtmp = 0;
                        if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                          cintopt, cache)) {
                                for (ij = 0; ij < dij; ij++) {
                                        tmp = fabs(buf[ij+dij*ij]);
                                        qtmp = MAX(qtmp, tmp);
                                }
                                // 2 * log(sqrt(qtmp))
                                qtmp = CUT_OFFSET + log(qtmp);
                        }
                        itmp = (int)qtmp;
                        itmp = MAX(0, itmp);
                        itmp = MIN(itmp, 127);
                        qijij[ish*nbas+jsh] = itmp;
                        qijij[jsh*nbas+ish] = itmp;

                        shls[0] = ish;
                        shls[1] = ish;
                        shls[2] = jsh;
                        shls[3] = jsh;
                        di2 = di * di;
                        dj2 = dj * dj;
                        qtmp = 0;
                        if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                          cintopt, cache)) {
                                for (j = 0; j < dj2; j+=dj+1) {
                                for (i = 0; i < di2; i+=di+1) {
                                        // buf[i,i,j,j]
                                        tmp = fabs(buf[i+di2*j]);
                                        qtmp = MAX(qtmp, tmp);
                                } }
                                qtmp = CUT_OFFSET + log(qtmp);
                        }
                        itmp = (int)qtmp;
                        itmp = MAX(0, itmp);
                        itmp = MIN(itmp, 127);
                        qiijj[ish*nbas+jsh] = itmp;
                        qiijj[jsh*nbas+ish] = itmp;
                }
        }
        free(buf);
        free(cache);
}
}

void PBCVHFsetnr_scond(float *scond, int *atm, int natm,
                       int *bas, int nbas, double *env)
{
        size_t Nbas = nbas;
        size_t Nbas1 = nbas + 1;
        int *exps_group_loc = malloc(sizeof(int) * Nbas1);
        float *exps = malloc(sizeof(double) * Nbas1 * 5);
        float *cs = exps + Nbas1;
        float *rx = cs + Nbas1;
        float *ry = rx + Nbas1;
        float *rz = ry + Nbas1;
        int ptr_coord, nprim, nctr, n, m, l;
        double exp_min, c_max, c1;
        int ngroups = 0;
        double exp_last = 0.;
        int l_last = -1;
        for (n = 0; n < nbas; n++) {
                ptr_coord = atm(PTR_COORD, bas(ATOM_OF, n));
                rx[n] = env[ptr_coord+0];
                ry[n] = env[ptr_coord+1];
                rz[n] = env[ptr_coord+2];
                nprim = bas(NPRIM_OF, n);
                // the most diffused function
                exp_min = env[bas(PTR_EXP, n) + nprim - 1];
                l = bas(ANG_OF, n);

                if (exp_min != exp_last || l_last != l) {
                        // partition all exponents into groups
                        exps[ngroups] = exp_min;
                        nctr = bas(NCTR_OF, n);
                        c_max = fabs(env[bas(PTR_COEFF, n) + nprim - 1]);
                        for (m = 1; m < nctr; m++) {
                                c1 = fabs(env[bas(PTR_COEFF, n) + (m+1)*nprim - 1]);
                                c_max = MAX(c_max, c1);
                        }
                        cs[ngroups] = c_max;
                        exps_group_loc[ngroups] = n;
                        exp_last = exp_min;
                        l_last = l;
                        ngroups++;
                }
        }
        exps_group_loc[ngroups] = nbas;

        double omega = env[PTR_RANGE_OMEGA];
        // log(sqrt(fl/sqrt(pi*theta)))
        // ~= log(sqrt(2/sqrt(pi*omega^2)))
        float omega2;
        if (omega == 0.) {
                omega2 = 0.3f;
        } else {
                omega2 = omega * omega;
        }

#pragma omp parallel
{
        float fac_guess = .6f - logf(omega2)/4;
        int ijb, ib, jb, i0, j0, i1, j1, i, j, li, lj;
        float dx, dy, dz, ai, aj, ci, cj, aij, a1, fi, fj, rr, rij, dri, drj;
        float log_fac, theta, theta_r, r_guess;
#pragma omp for schedule(dynamic, 1)
        for (ijb = 0; ijb < ngroups*(ngroups+1)/2; ijb++) {
                ib = (int)(sqrt(2*ijb+.25) - .5 + 1e-7);
                jb = ijb - ib*(ib+1)/2;

                i0 = exps_group_loc[ib];
                i1 = exps_group_loc[ib+1];
                li = bas(ANG_OF, i0);
                ai = exps[ib];
                ci = cs[ib];
                j0 = exps_group_loc[jb];
                j1 = exps_group_loc[jb+1];
                lj = bas(ANG_OF, j0);
                aj = exps[jb];
                cj = cs[jb];

                aij = ai + aj;
                fi = ai / aij;
                fj = aj / aij;
                a1 = ai * aj / aij;
                theta = omega2/(omega2+aij);
                r_guess = sqrtf(-logf(1e-9f) / (aij * theta));
                theta_r = theta * r_guess;
                // log(ci*cj * ((2*li+1)*(2*lj+1))**.5/(4*pi) * (pi/aij)**1.5)
                log_fac = logf(ci*cj * sqrtf((2*li+1.f)*(2*lj+1.f))/(4*M_PI))
                        + 1.5f*logf(M_PI/aij) + fac_guess;
                for (i = i0; i < i1; i++) {
#pragma GCC ivdep
                for (j = j0; j < j1; j++) {
                        dx = rx[i] - rx[j];
                        dy = ry[i] - ry[j];
                        dz = rz[i] - rz[j];
                        rr = dx * dx + dy * dy + dz * dz;
                        rij = sqrtf(rr);
                        dri = fj * rij + theta_r;
                        drj = fi * rij + theta_r;
                        scond[i*Nbas+j] = li*logf(dri) + lj*logf(drj)
                                - a1*rr + log_fac;
                } }
                if (ib > jb) {
                        for (i = i0; i < i1; i++) {
#pragma GCC ivdep
                        for (j = j0; j < j1; j++) {
                                scond[j*Nbas+i] = scond[i*Nbas+j];
                        } }
                }
        }
}
        free(exps);
        free(exps_group_loc);
}
