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
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "pbc/pbc.h"

void PBCminimal_CINTEnvVars(CINTEnvVars *envs, int *atm, int natm, int *bas, int nbas, double *env,
                            CINTOpt *cintopt);

void PBCVHF_contract_k_s1(int (*intor)(), double *vk, double *dms, double *buf,
                          int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                          int *dm_translation, int n_dm,
                          CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
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
        double direct_scf_cutoff = envs_bvk->cutoff;
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        if (dm_jk_cond < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_jk_cond;
        }
        if (!(*intor)(buf, cell0_shls, bvk_cells, direct_scf_cutoff, envs_cint, envs_bvk)) {
                return;
        }

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
                               int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                               int *dm_translation, int n_dm,
                               CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
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
        double direct_scf_cutoff = envs_bvk->cutoff;
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        double dm_jl_cond = vhfopt->dm_cond[dm_jl_off*nn0 + jsh_cell0*Nbasp+lsh_cell0];
        double dm_cond_max = MAX(dm_jk_cond, dm_jl_cond);
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!(*intor)(buf, cell0_shls, bvk_cells, direct_scf_cutoff, envs_cint, envs_bvk)) {
                return;
        }

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
                            int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                            int *dm_translation, int n_dm,
                            CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int ksh = cell0_shls[2] + bvk_cells[2] * nbasp;
        int lsh = cell0_shls[3] + bvk_cells[3] * nbasp;
        if (ksh > lsh) {
                contract_k_s2_kgtl(intor, vk, dms, buf, cell0_shls, bvk_cells,
                                   cell0_ao_loc, dm_translation, n_dm,
                                   vhfopt, envs_cint, envs_bvk);
        } else if (ksh == lsh) {
                PBCVHF_contract_k_s1(intor, vk, dms, buf, cell0_shls, bvk_cells,
                                     cell0_ao_loc, dm_translation, n_dm,
                                     vhfopt, envs_cint, envs_bvk);
        }
}

void PBCVHF_contract_j_s1(int (*intor)(), double *vj, double *dms, double *buf,
                          int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                          int *dm_translation, int n_dm,
                          CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
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
        double direct_scf_cutoff = envs_bvk->cutoff;
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        if (dm_lk_cond < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_lk_cond;
        }
        if (!(*intor)(buf, cell0_shls, bvk_cells, direct_scf_cutoff, envs_cint, envs_bvk)) {
                return;
        }

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
                               int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                               int *dm_translation, int n_dm,
                               CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
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
        double direct_scf_cutoff = envs_bvk->cutoff;
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        double dm_kl_cond = vhfopt->dm_cond[dm_kl_off*nn0 + ksh_cell0*Nbasp+lsh_cell0];
        double dm_cond_max = dm_lk_cond + dm_kl_cond;
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!(*intor)(buf, cell0_shls, bvk_cells, direct_scf_cutoff, envs_cint, envs_bvk)) {
                return;
        }

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
                            int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                            int *dm_translation, int n_dm,
                            CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int ksh = cell0_shls[2] + bvk_cells[2] * nbasp;
        int lsh = cell0_shls[3] + bvk_cells[3] * nbasp;
        if (ksh > lsh) {
                contract_j_s2_kgtl(intor, vj, dms, buf, cell0_shls, bvk_cells,
                                   cell0_ao_loc, dm_translation, n_dm,
                                   vhfopt, envs_cint, envs_bvk);
        } else if (ksh == lsh) {
                PBCVHF_contract_j_s1(intor, vj, dms, buf, cell0_shls, bvk_cells,
                                     cell0_ao_loc, dm_translation, n_dm,
                                     vhfopt, envs_cint, envs_bvk);
        }
}

void PBCVHF_contract_jk_s1(int (*intor)(), double *jk, double *dms, double *buf,
                           int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                           int *dm_translation, int n_dm,
                           CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
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
        double direct_scf_cutoff = envs_bvk->cutoff;
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        double dm_cond_max = MAX(dm_lk_cond, dm_jk_cond);
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!(*intor)(buf, cell0_shls, bvk_cells, direct_scf_cutoff, envs_cint, envs_bvk)) {
                return;
        }

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
                                int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                                int *dm_translation, int n_dm,
                                CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
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
        double direct_scf_cutoff = envs_bvk->cutoff;
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jsh_cell0*Nbasp+ksh_cell0];
        double dm_jl_cond = vhfopt->dm_cond[dm_jl_off*nn0 + jsh_cell0*Nbasp+lsh_cell0];
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lsh_cell0*Nbasp+ksh_cell0];
        double dm_kl_cond = vhfopt->dm_cond[dm_kl_off*nn0 + ksh_cell0*Nbasp+lsh_cell0];
        double dm_cond_max = MAX(dm_jk_cond, dm_jl_cond);
        dm_cond_max = MAX(dm_cond_max, dm_lk_cond + dm_kl_cond);
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!(*intor)(buf, cell0_shls, bvk_cells, direct_scf_cutoff, envs_cint, envs_bvk)) {
                return;
        }

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
                             int *cell0_shls, int *bvk_cells, int *cell0_ao_loc,
                             int *dm_translation, int n_dm,
                             CVHFOpt *vhfopt, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int nbasp = envs_bvk->nbasp;
        int ksh = cell0_shls[2] + bvk_cells[2] * nbasp;
        int lsh = cell0_shls[3] + bvk_cells[3] * nbasp;
        if (ksh > lsh) {
                contract_jk_s2_kgtl(intor, jk, dms, buf, cell0_shls, bvk_cells,
                                    cell0_ao_loc, dm_translation, n_dm,
                                    vhfopt, envs_cint, envs_bvk);
        } else if (ksh == lsh) {
                PBCVHF_contract_jk_s1(intor, jk, dms, buf, cell0_shls, bvk_cells,
                                      cell0_ao_loc, dm_translation, n_dm,
                                      vhfopt, envs_cint, envs_bvk);
        }
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
                       int *sh_loc, int *cell0_ao_loc, int *shls_slice,
                       int *dm_translation, int8_t *ovlp_mask, int *bas_map,
                       CINTOpt *cintopt, CVHFOpt *vhfopt, int cache_size,
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
        //size_t nksh = ksh1 - ksh0;
        size_t nlsh = lsh1 - lsh0;
        size_t nij = nish * njsh;
        BVKEnvs envs_bvk = {bvk_ncells, nimgs,
                nkpts, nbands, nbasp, comp, 0, 0,
                sh_loc, cell0_ao_loc, bas_map, shls_slice, NULL, NULL, NULL,
                ovlp_mask, vhfopt->q_cond, vhfopt->direct_scf_cutoff};

#pragma omp parallel
{
        size_t ij, n;
        int i, j, k, l;
        int bvk_cells[4] = {0};
        int cell0_shls[4];
        CINTEnvVars envs_cint;
        PBCminimal_CINTEnvVars(&envs_cint, atm, natm, bas, nbas, env, cintopt);

        double *v_priv = calloc(size_v, sizeof(double));
        double *buf = malloc(sizeof(double) * cache_size);

#pragma omp for schedule(dynamic, 1)
        for (ij = 0; ij < nij; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                if (!ovlp_mask[i*njsh+j]) {
                        continue;
                }
                i += ish0;
                j += jsh0;
                cell0_shls[0] = i;
                cell0_shls[1] = j % nbasp;
                bvk_cells[1] = j / nbasp;

                for (k = ksh0; k < ksh1; k++) {
                for (l = ksh0; l < lsh1; l++) {
                        if (!ovlp_mask[(k-ksh0)*nlsh+l-lsh0]) {
                                continue;
                        }
                        cell0_shls[2] = k % nbasp;
                        cell0_shls[3] = l % nbasp;
                        bvk_cells[2] = k / nbasp;
                        bvk_cells[3] = l / nbasp;
                        (*fdot)(intor, v_priv, dms, buf, cell0_shls, bvk_cells,
                                cell0_ao_loc, dm_translation, n_dm,
                                vhfopt, &envs_cint, &envs_bvk);
                } }
        }
#pragma omp critical
        {
                for (n = 0; n < size_v; n++) {
                        out[n] += v_priv[n];
                }
        }
        free(buf);
        free(v_priv);
}
}

/************************************************/
void CVHFset_int2e_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                          int *ao_loc, int *atm, int natm,
                          int *bas, int nbas, double *env);

static int _int2e_swap_jk(double *buf, int *dims, int *shls,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          CINTOpt *cintopt, double *cache)
{
        int shls_swap_jk[4] = {shls[0], shls[2], shls[1], shls[3]};
        return int2e_sph(buf, dims, shls_swap_jk, atm, natm, bas, nbas, env, cintopt, cache);
}

void PBCVHFsetnr_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        /* This memory is released in void CVHFdel_optimizer, Don't know
         * why valgrind raises memory leak here */
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        // nbas in the input arguments may different to opt->nbas.
        // Use opt->nbas because it is used in the prescreen function
        nbas = opt->nbas;
        size_t Nbas = nbas;
        opt->q_cond = (double *)malloc(sizeof(double) * Nbas * Nbas * 2);
        double *qcond_ijij = opt->q_cond;
        double *qcond_iijj = qcond_ijij + Nbas * Nbas;
        CVHFset_int2e_q_cond(intor, cintopt, qcond_ijij, ao_loc,
                             atm, natm, bas, nbas, env);
        CVHFset_int2e_q_cond(_int2e_swap_jk, cintopt, qcond_iijj, ao_loc,
                             atm, natm, bas, nbas, env);
}
