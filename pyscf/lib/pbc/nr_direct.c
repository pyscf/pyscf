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
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "vhf/nr_direct.h"

#define MIN(I,J)        ((I) < (J) ? (I) : (J))
#define MAX(I,J)        ((I) > (J) ? (I) : (J))

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter);

static int _max_cache_size(int (*intor)(), int *shls_slice, int *images_loc,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, n;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        int shls[4];
        int cache_size = 0;
        for (i = i0; i < i1; i++) {
                shls[0] = images_loc[i];
                shls[1] = images_loc[i];
                shls[2] = images_loc[i];
                shls[3] = images_loc[i];
                n = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

static int _assemble_eris(double *eri_buf, int *images_loc,
                          int ishell, int jshell, int kshell, int lshell,
                          double cutoff, CVHFOpt *vhfopt, IntorEnvs *envs)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        CINTOpt *cintopt = envs->cintopt;
        const size_t Nbas = nbas;
        const int *ao_loc = envs->ao_loc;
        const int ish0 = images_loc[ishell];
        const int jsh0 = images_loc[jshell];
        const int ksh0 = images_loc[kshell];
        const int lsh0 = images_loc[lshell];
        const int jsh1 = images_loc[jshell+1];
        const int ksh1 = images_loc[kshell+1];
        const int lsh1 = images_loc[lshell+1];
        const int i0 = ao_loc[ishell];
        const int j0 = ao_loc[jshell];
        const int k0 = ao_loc[kshell];
        const int l0 = ao_loc[lshell];
        const int i1 = ao_loc[ishell+1];
        const int j1 = ao_loc[jshell+1];
        const int k1 = ao_loc[kshell+1];
        const int l1 = ao_loc[lshell+1];
        const int di = i1 - i0;
        const int dj = j1 - j0;
        const int dk = k1 - k0;
        const int dl = l1 - l0;
        const int dijkl = di * dj * dk * dl;
        double *q_cond_ijij = vhfopt->q_cond;
        double *q_cond_iijj = vhfopt->q_cond + Nbas*Nbas;
        double *q_cond_ij, *q_cond_kl, *q_cond_ik, *q_cond_jk;
        double *eri = eri_buf;
        double *bufL = eri_buf + dijkl;
        double *cache = bufL + dijkl;
        int shls[4] = {ish0};
        int n, jsh, ksh, lsh;
        double kl_cutoff, jl_cutoff, il_cutoff;

        int empty = 1;
        for (n = 0; n < dijkl; n++) {
                eri[n] = 0;
        }

        q_cond_ij = q_cond_ijij + ish0 * Nbas;
        q_cond_ik = q_cond_iijj + ish0 * Nbas;
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                if (q_cond_ij[jsh] < cutoff) {
                        continue;
                }
                kl_cutoff = cutoff / q_cond_ij[jsh];
                q_cond_jk = q_cond_iijj + jsh * Nbas;
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                        if (q_cond_ik[ksh] < cutoff ||
                            q_cond_jk[ksh] < cutoff) {
                                continue;
                        }
                        q_cond_kl = q_cond_ijij + ksh * Nbas;
                        jl_cutoff = cutoff / q_cond_ik[ksh];
                        il_cutoff = cutoff / q_cond_jk[ksh];
                        for (lsh = lsh0; lsh < lsh1; lsh++) {
                                if (q_cond_kl[lsh] < kl_cutoff ||
                                    q_cond_jk[lsh] < jl_cutoff ||
                                    q_cond_ik[lsh] < il_cutoff) {
                                        continue;
                                }
                                shls[1] = jsh;
                                shls[2] = ksh;
                                shls[3] = lsh;
                                if (int2e_sph(bufL, NULL, shls, atm, natm,
                                              bas, nbas, env, cintopt, cache)) {
                                        for (n = 0; n < dijkl; n++) {
                                                eri[n] += bufL[n];
                                        }
                                        empty = 0;
                                }
                        }
                }

        }
        return !empty;
}

void PBCVHF_contract_k_s1(double *vk, double *dms, double *buf,
                          int n_dm, int nkpts, int nbands, int nbasp,
                          int ish, int jsh, int ksh, int lsh,
                          int *bvk_cell_id, int *cell0_shl_id,
                          int *images_loc, int *dm_translation,
                          CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int cell_j = bvk_cell_id[jsh];
        const int cell_k = bvk_cell_id[ksh];
        const int cell_l = bvk_cell_id[lsh];
        const int jshp = cell0_shl_id[jsh];
        const int kshp = cell0_shl_id[ksh];
        const int lshp = cell0_shl_id[lsh];
        const int dm_jk_off = dm_translation[cell_j * nkpts + cell_k];
        const size_t Nbasp = nbasp;
        const size_t nn0 = Nbasp * Nbasp;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jshp*Nbasp+kshp];
        if (dm_jk_cond < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_jk_cond;
        }
        if (!_assemble_eris(buf, images_loc, ish, jsh, ksh, lsh,
                            direct_scf_cutoff, vhfopt, envs)) {
                return;
        }

        const int *ao_loc = envs->ao_loc;
        const size_t naop = ao_loc[nbasp];
        const size_t nn = naop * naop;
        const size_t bn = naop * nbands;
        const size_t knn = nn * nkpts;
        const size_t bnn = bn * naop;
        const int i0  = ao_loc[ish];
        const int jp0 = ao_loc[jshp];
        const int kp0 = ao_loc[kshp];
        const int lp0 = ao_loc[lshp];
        const int i1  = ao_loc[ish+1];
        const int jp1 = ao_loc[jshp+1];
        const int kp1 = ao_loc[kshp+1];
        const int lp1 = ao_loc[lshp+1];
        int idm, i, jp, kp, lp, n;
        double sjk, qijkl;
        double *dm_jk;
        vk += cell_l * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                n = 0;
                for (lp = lp0; lp < lp1; lp++) {
                for (kp = kp0; kp < kp1; kp++) {
                for (jp = jp0; jp < jp1; jp++) {
                        sjk = dm_jk[jp*naop+kp];
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vk[i*bn+lp] += qijkl * sjk;
                        } } }
                }
                vk += bnn;
        }
}

static void contract_k_s2_kgtl(double *vk, double *dms, double *buf,
                          int n_dm, int nkpts, int nbands, int nbasp,
                          int ish, int jsh, int ksh, int lsh,
                          int *bvk_cell_id, int *cell0_shl_id,
                          int *images_loc, int *dm_translation,
                          CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int cell_j = bvk_cell_id[jsh];
        const int cell_k = bvk_cell_id[ksh];
        const int cell_l = bvk_cell_id[lsh];
        const int jshp = cell0_shl_id[jsh];
        const int kshp = cell0_shl_id[ksh];
        const int lshp = cell0_shl_id[lsh];
        const int dm_jk_off = dm_translation[cell_j*nkpts+cell_k];
        const int dm_jl_off = dm_translation[cell_j*nkpts+cell_l];
        const size_t Nbasp = nbasp;
        const size_t nn0 = Nbasp * Nbasp;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jshp*Nbasp+kshp];
        double dm_jl_cond = vhfopt->dm_cond[dm_jl_off*nn0 + jshp*Nbasp+lshp];
        double dm_cond_max = MAX(dm_jk_cond, dm_jl_cond);
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!_assemble_eris(buf, images_loc, ish, jsh, ksh, lsh,
                            direct_scf_cutoff, vhfopt, envs)) {
                return;
        }

        const int *ao_loc = envs->ao_loc;
        const size_t naop = ao_loc[nbasp];
        const size_t nn = naop * naop;
        const size_t bn = naop * nbands;
        const size_t knn = nn * nkpts;
        const size_t bnn = bn * naop;
        const int i0  = ao_loc[ish];
        const int jp0 = ao_loc[jshp];
        const int kp0 = ao_loc[kshp];
        const int lp0 = ao_loc[lshp];
        const int i1  = ao_loc[ish+1];
        const int jp1 = ao_loc[jshp+1];
        const int kp1 = ao_loc[kshp+1];
        const int lp1 = ao_loc[lshp+1];
        int idm, i, jp, kp, lp, n;
        double sjk, sjl, qijkl;
        double *dm_jk, *dm_jl;
        double *vk_ik = vk + cell_k * naop;
        double *vk_il = vk + cell_l * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                dm_jl = dms + dm_jl_off * nn + idm * knn;
                n = 0;
                for (lp = lp0; lp < lp1; lp++) {
                for (kp = kp0; kp < kp1; kp++) {
                for (jp = jp0; jp < jp1; jp++) {
                        sjk = dm_jk[jp*naop+kp];
                        sjl = dm_jl[jp*naop+lp];
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vk_il[i*bn+lp] += qijkl * sjk;
                                vk_ik[i*bn+kp] += qijkl * sjl;
                        } } }
                }
                vk_ik += bnn;
                vk_il += bnn;
        }
}

void PBCVHF_contract_k_s2kl(double *vk, double *dms, double *buf,
                            int n_dm, int nkpts, int nbands, int nbasp,
                            int ish, int jsh, int ksh, int lsh,
                            int *bvk_cell_id, int *cell0_shl_id,
                            int *images_loc, int *dm_translation,
                            CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ksh > lsh) {
                contract_k_s2_kgtl(vk, dms, buf, n_dm, nkpts, nbands, nbasp,
                                   ish, jsh, ksh, lsh, bvk_cell_id,
                                   cell0_shl_id, images_loc,
                                   dm_translation, vhfopt, envs);
        } else if (ksh == lsh) {
                PBCVHF_contract_k_s1(vk, dms, buf, n_dm, nkpts, nbands, nbasp,
                                     ish, jsh, ksh, lsh, bvk_cell_id,
                                     cell0_shl_id, images_loc,
                                     dm_translation, vhfopt, envs);
        }
}

void PBCVHF_contract_j_s1(double *vj, double *dms, double *buf,
                          int n_dm, int nkpts, int nbands, int nbasp,
                          int ish, int jsh, int ksh, int lsh,
                          int *bvk_cell_id, int *cell0_shl_id,
                          int *images_loc, int *dm_translation,
                          CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int cell_j = bvk_cell_id[jsh];
        const int cell_k = bvk_cell_id[ksh];
        const int cell_l = bvk_cell_id[lsh];
        const int jshp = cell0_shl_id[jsh];
        const int kshp = cell0_shl_id[ksh];
        const int lshp = cell0_shl_id[lsh];
        const int dm_lk_off = dm_translation[cell_l * nkpts + cell_k];
        const size_t Nbasp = nbasp;
        const size_t nn0 = Nbasp * Nbasp;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lshp*Nbasp+kshp];
        if (dm_lk_cond < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_lk_cond;
        }
        if (!_assemble_eris(buf, images_loc, ish, jsh, ksh, lsh,
                            direct_scf_cutoff, vhfopt, envs)) {
                return;
        }

        const int *ao_loc = envs->ao_loc;
        const size_t naop = ao_loc[nbasp];
        const size_t nn = naop * naop;
        const size_t bn = naop * nbands;
        const size_t knn = nn * nkpts;
        const size_t bnn = bn * naop;
        const int i0  = ao_loc[ish];
        const int jp0 = ao_loc[jshp];
        const int kp0 = ao_loc[kshp];
        const int lp0 = ao_loc[lshp];
        const int i1  = ao_loc[ish+1];
        const int jp1 = ao_loc[jshp+1];
        const int kp1 = ao_loc[kshp+1];
        const int lp1 = ao_loc[lshp+1];
        int idm, i, jp, kp, lp, n;
        double slk, qijkl;
        double *dm_lk;
        vj += cell_j * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                n = 0;
                for (lp = lp0; lp < lp1; lp++) {
                for (kp = kp0; kp < kp1; kp++) {
                        slk = dm_lk[lp*naop+kp];
                        for (jp = jp0; jp < jp1; jp++) {
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vj[i*bn+jp] += qijkl * slk;
                        } }
                } }
                vj += bnn;
        }
}

static void contract_j_s2_kgtl(double *vj, double *dms, double *buf,
                          int n_dm, int nkpts, int nbands, int nbasp,
                          int ish, int jsh, int ksh, int lsh,
                          int *bvk_cell_id, int *cell0_shl_id,
                          int *images_loc, int *dm_translation,
                          CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int cell_j = bvk_cell_id[jsh];
        const int cell_k = bvk_cell_id[ksh];
        const int cell_l = bvk_cell_id[lsh];
        const int jshp = cell0_shl_id[jsh];
        const int kshp = cell0_shl_id[ksh];
        const int lshp = cell0_shl_id[lsh];
        const int dm_lk_off = dm_translation[cell_l * nkpts + cell_k];
        const int dm_kl_off = dm_translation[cell_k * nkpts + cell_l];
        const size_t Nbasp = nbasp;
        const size_t nn0 = Nbasp * Nbasp;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lshp*Nbasp+kshp];
        double dm_kl_cond = vhfopt->dm_cond[dm_kl_off*nn0 + kshp*Nbasp+lshp];
        double dm_cond_max = dm_lk_cond + dm_kl_cond;
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!_assemble_eris(buf, images_loc, ish, jsh, ksh, lsh,
                            direct_scf_cutoff, vhfopt, envs)) {
                return;
        }

        const int *ao_loc = envs->ao_loc;
        const size_t naop = ao_loc[nbasp];
        const size_t nn = naop * naop;
        const size_t bn = naop * nbands;
        const size_t knn = nn * nkpts;
        const size_t bnn = bn * naop;
        const int i0  = ao_loc[ish];
        const int jp0 = ao_loc[jshp];
        const int kp0 = ao_loc[kshp];
        const int lp0 = ao_loc[lshp];
        const int i1  = ao_loc[ish+1];
        const int jp1 = ao_loc[jshp+1];
        const int kp1 = ao_loc[kshp+1];
        const int lp1 = ao_loc[lshp+1];
        int idm, i, jp, kp, lp, n;
        double slk, qijkl;
        double *dm_lk, *dm_kl;
        vj += cell_j * naop;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                dm_kl = dms + dm_kl_off * nn + idm * knn;
                n = 0;
                for (lp = lp0; lp < lp1; lp++) {
                for (kp = kp0; kp < kp1; kp++) {
                        slk = dm_lk[lp*naop+kp] + dm_kl[kp*naop+lp];
                        for (jp = jp0; jp < jp1; jp++) {
                        for (i = i0; i < i1; i++, n++) {
                                qijkl = buf[n];
                                vj[i*bn+jp] += qijkl * slk;
                        } }
                } }
                vj += bnn;
        }
}

void PBCVHF_contract_j_s2kl(double *vj, double *dms, double *buf,
                            int n_dm, int nkpts, int nbands, int nbasp,
                            int ish, int jsh, int ksh, int lsh,
                            int *bvk_cell_id, int *cell0_shl_id,
                            int *images_loc, int *dm_translation,
                            CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ksh > lsh) {
                contract_j_s2_kgtl(vj, dms, buf, n_dm, nkpts, nbands, nbasp,
                                   ish, jsh, ksh, lsh, bvk_cell_id,
                                   cell0_shl_id, images_loc,
                                   dm_translation, vhfopt, envs);
        } else if (ksh == lsh) {
                PBCVHF_contract_j_s1(vj, dms, buf, n_dm, nkpts, nbands, nbasp,
                                     ish, jsh, ksh, lsh, bvk_cell_id,
                                     cell0_shl_id, images_loc,
                                     dm_translation, vhfopt, envs);
        }
}

void PBCVHF_contract_jk_s1(double *jk, double *dms, double *buf,
                           int n_dm, int nkpts, int nbands, int nbasp,
                           int ish, int jsh, int ksh, int lsh,
                           int *bvk_cell_id, int *cell0_shl_id,
                           int *images_loc, int *dm_translation,
                           CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int cell_j = bvk_cell_id[jsh];
        const int cell_k = bvk_cell_id[ksh];
        const int cell_l = bvk_cell_id[lsh];
        const int jshp = cell0_shl_id[jsh];
        const int kshp = cell0_shl_id[ksh];
        const int lshp = cell0_shl_id[lsh];
        const int dm_lk_off = dm_translation[cell_l * nkpts + cell_k];
        const int dm_jk_off = dm_translation[cell_j * nkpts + cell_k];
        const size_t Nbasp = nbasp;
        const size_t nn0 = Nbasp * Nbasp;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lshp*Nbasp+kshp];
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jshp*Nbasp+kshp];
        double dm_cond_max = MAX(dm_lk_cond, dm_jk_cond);
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!_assemble_eris(buf, images_loc, ish, jsh, ksh, lsh,
                            direct_scf_cutoff, vhfopt, envs)) {
                return;
        }

        const int *ao_loc = envs->ao_loc;
        const size_t naop = ao_loc[nbasp];
        const size_t nn = naop * naop;
        const size_t bn = naop * nbands;
        const size_t knn = nn * nkpts;
        const size_t bnn = bn * naop;
        const int i0  = ao_loc[ish];
        const int jp0 = ao_loc[jshp];
        const int kp0 = ao_loc[kshp];
        const int lp0 = ao_loc[lshp];
        const int i1  = ao_loc[ish+1];
        const int jp1 = ao_loc[jshp+1];
        const int kp1 = ao_loc[kshp+1];
        const int lp1 = ao_loc[lshp+1];
        double *vj = jk + cell_j * naop;
        double *vk = jk + n_dm * bnn + cell_l * naop;
        int idm, i, jp, kp, lp, n;
        double slk, sjk, qijkl;
        double *dm_lk, *dm_jk;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                n = 0;
                for (lp = lp0; lp < lp1; lp++) {
                for (kp = kp0; kp < kp1; kp++) {
                        slk = dm_lk[lp*naop+kp];
                        for (jp = jp0; jp < jp1; jp++) {
                                sjk = dm_jk[jp*naop+kp];
                                for (i = i0; i < i1; i++, n++) {
                                        qijkl = buf[n];
                                        vj[i*bn+jp] += qijkl * slk;
                                        vk[i*bn+lp] += qijkl * sjk;
                                }
                        }
                } }
                vj += bnn;
                vk += bnn;
        }
}

static void contract_jk_s2_kgtl(double *jk, double *dms, double *buf,
                                int n_dm, int nkpts, int nbands, int nbasp,
                                int ish, int jsh, int ksh, int lsh,
                                int *bvk_cell_id, int *cell0_shl_id,
                                int *images_loc, int *dm_translation,
                                CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int cell_j = bvk_cell_id[jsh];
        const int cell_k = bvk_cell_id[ksh];
        const int cell_l = bvk_cell_id[lsh];
        const int jshp = cell0_shl_id[jsh];
        const int kshp = cell0_shl_id[ksh];
        const int lshp = cell0_shl_id[lsh];
        const int dm_jk_off = dm_translation[cell_j*nkpts+cell_k];
        const int dm_jl_off = dm_translation[cell_j*nkpts+cell_l];
        const int dm_lk_off = dm_translation[cell_l*nkpts+cell_k];
        const int dm_kl_off = dm_translation[cell_k*nkpts+cell_l];
        const size_t Nbasp = nbasp;
        const size_t nn0 = Nbasp * Nbasp;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double dm_jk_cond = vhfopt->dm_cond[dm_jk_off*nn0 + jshp*Nbasp+kshp];
        double dm_jl_cond = vhfopt->dm_cond[dm_jl_off*nn0 + jshp*Nbasp+lshp];
        double dm_lk_cond = vhfopt->dm_cond[dm_lk_off*nn0 + lshp*Nbasp+kshp];
        double dm_kl_cond = vhfopt->dm_cond[dm_kl_off*nn0 + kshp*Nbasp+lshp];
        double dm_cond_max = MAX(dm_jk_cond, dm_jl_cond);
        dm_cond_max = MAX(dm_cond_max, dm_lk_cond + dm_kl_cond);
        if (dm_cond_max < direct_scf_cutoff) {
                return;
        } else {
                direct_scf_cutoff /= dm_cond_max;
        }
        if (!_assemble_eris(buf, images_loc, ish, jsh, ksh, lsh,
                            direct_scf_cutoff, vhfopt, envs)) {
                return;
        }

        const int *ao_loc = envs->ao_loc;
        const size_t naop = ao_loc[nbasp];
        const size_t nn = naop * naop;
        const size_t bn = naop * nbands;
        const size_t knn = nn * nkpts;
        const size_t bnn = bn * naop;
        const int i0  = ao_loc[ish];
        const int jp0 = ao_loc[jshp];
        const int kp0 = ao_loc[kshp];
        const int lp0 = ao_loc[lshp];
        const int i1  = ao_loc[ish+1];
        const int jp1 = ao_loc[jshp+1];
        const int kp1 = ao_loc[kshp+1];
        const int lp1 = ao_loc[lshp+1];
        double *vj = jk + cell_j * naop;
        double *vk_ik = jk + n_dm * bnn + cell_k * naop;
        double *vk_il = jk + n_dm * bnn + cell_l * naop;
        int idm, i, jp, kp, lp, n;
        double sjk, sjl, slk, qijkl;
        double *dm_jk, *dm_jl, *dm_lk, *dm_kl;

        for (idm = 0; idm < n_dm; idm++) {
                dm_lk = dms + dm_lk_off * nn + idm * knn;
                dm_kl = dms + dm_kl_off * nn + idm * knn;
                dm_jk = dms + dm_jk_off * nn + idm * knn;
                dm_jl = dms + dm_jl_off * nn + idm * knn;
                n = 0;
                for (lp = lp0; lp < lp1; lp++) {
                for (kp = kp0; kp < kp1; kp++) {
                        slk = dm_lk[lp*naop+kp] + dm_kl[kp*naop+lp];
                        for (jp = jp0; jp < jp1; jp++) {
                                sjk = dm_jk[jp*naop+kp];
                                sjl = dm_jl[jp*naop+lp];
                                for (i = i0; i < i1; i++, n++) {
                                        qijkl = buf[n];
                                        vj[i*bn+jp] += qijkl * slk;
                                        vk_il[i*bn+lp] += qijkl * sjk;
                                        vk_ik[i*bn+kp] += qijkl * sjl;
                                } }
                        }
                }
                vj += bnn;
                vk_ik += bnn;
                vk_il += bnn;
        }
}

void PBCVHF_contract_jk_s2kl(double *jk, double *dms, double *buf,
                             int n_dm, int nkpts, int nbands, int nbasp,
                             int ish, int jsh, int ksh, int lsh,
                             int *bvk_cell_id, int *cell0_shl_id,
                             int *images_loc, int *dm_translation,
                             CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ksh > lsh) {
                contract_jk_s2_kgtl(jk, dms, buf, n_dm, nkpts, nbands, nbasp,
                                   ish, jsh, ksh, lsh, bvk_cell_id,
                                   cell0_shl_id, images_loc,
                                   dm_translation, vhfopt, envs);
        } else if (ksh == lsh) {
                PBCVHF_contract_jk_s1(jk, dms, buf, n_dm, nkpts, nbands, nbasp,
                                      ish, jsh, ksh, lsh, bvk_cell_id,
                                      cell0_shl_id, images_loc,
                                      dm_translation, vhfopt, envs);
        }
}

/*
 * shls_slice refers to the shells of entire sup-mol.
 * bvk_ao_loc are ao_locs of bvk-cell basis appeared in supmol (some basis are removed)
 * nbasp is the number of basis in primitive cell
 * dm_translation utilizes the translation symmetry for density matrices (wrt the full bvk-cell)
 * DM[M,N] = DM[N-M] by mapping the 2D subscripts to 1D subscripts
 */
void PBCVHF_direct_drv(void (*fdot)(), double *out, double *dms,
                       int n_dm, int nkpts, int nbands, int nbasp,
                       int8_t *ovlp_mask, int *bvk_cell_id,
                       int *cell0_shl_id, int *images_loc,
                       int *shls_slice, int *bvk_ao_loc,
                       int *dm_translation, CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, bvk_ao_loc,
                NULL, cintopt, 1};

        const size_t ish0 = shls_slice[0];
        const size_t ish1 = shls_slice[1];
        const size_t jsh0 = shls_slice[2];
        const size_t jsh1 = shls_slice[3];
        const size_t ksh0 = shls_slice[4];
        const size_t ksh1 = shls_slice[5];
        const size_t lsh0 = shls_slice[6];
        const size_t lsh1 = shls_slice[7];
        const size_t nish = ish1 - ish0;
        const size_t njsh = jsh1 - jsh0;
        const size_t nksh = ksh1 - ksh0;
        const size_t nlsh = lsh1 - lsh0;
        const int di = GTOmax_shell_dim(bvk_ao_loc, shls_slice, 1);
        const int cache_size = _max_cache_size(int2e_sph, shls_slice, images_loc,
                                               atm, natm, bas, nbas, env);
        const size_t nij = nish * njsh;
        const size_t naop = bvk_ao_loc[nbasp];

#pragma omp parallel
{
        size_t ij, n;
        int i, j, k, l;
        size_t size = n_dm * naop * naop * nbands;
        if (fdot == &PBCVHF_contract_jk_s2kl || fdot == &PBCVHF_contract_jk_s1) {
                size *= 2;  // vj and vk
        }
        double *v_priv = calloc(size, sizeof(double));
        double *buf = malloc(sizeof(double) * (di*di*di*di*2 + cache_size));

#pragma omp for schedule(dynamic, 1)
        for (ij = 0; ij < nij; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                if (!ovlp_mask[i*njsh+j]) {
                        continue;
                }

                for (k = 0; k < nksh; k++) {
                for (l = 0; l < nlsh; l++) {
                        if (!ovlp_mask[k*nlsh+l]) {
                                continue;
                        }
                        (*fdot)(v_priv, dms, buf, n_dm, nkpts, nbands, nbasp,
                                i, j, k, l, bvk_cell_id, cell0_shl_id, images_loc,
                                dm_translation, vhfopt, &envs);
                } }
        }
#pragma omp critical
        {
                for (n = 0; n < size; n++) {
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
