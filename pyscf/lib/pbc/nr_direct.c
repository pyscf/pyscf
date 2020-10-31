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
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "vhf/nr_direct.h"

#define SKIP_SMOOTH_BLOCK 2

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter);
int GTOmax_cache_size(CINTIntegralFunction intor, int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);
int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc);

#define MIN(I,J)        ((I) < (J) ? (I) : (J))
#define DECLARE_ALL \
        int *atm = envs->atm; \
        int *bas = envs->bas; \
        double *env = envs->env; \
        int natm = envs->natm; \
        int nbas = envs->nbas; \
        CINTOpt *cintopt = envs->cintopt; \
        const int *ao_loc = envs->ao_loc; \
        const int *shls_slice = envs->shls_slice; \
        const size_t Nbas = nbas; \
        const size_t njsh = shls_slice[3] - shls_slice[2]; \
        const size_t nlsh = shls_slice[7] - shls_slice[6]; \
        const size_t knbas = nbas0 * nkpts; \
        const size_t kn = nao * nkpts; \
        const size_t bn = nao * nbands; \
        const size_t knn = kn * nao; \
        const size_t bnn = bn * nao; \
        const size_t kknn = knn * nkpts; \
        const size_t ish0 = ishls[0]; \
        const size_t ish1 = ishls[1]; \
        const size_t jsh0 = jshls[0]; \
        const size_t jsh1 = jshls[1]; \
        const size_t ksh0 = kshls[0]; \
        const size_t ksh1 = kshls[1]; \
        const size_t lsh0 = lshls[0]; \
        const size_t lsh1 = lshls[1]; \
        int shls[4]; \
        size_t i, j, k, l, ijkl; \
        size_t i0, j0, k0, l0; \
        size_t i1, j1, k1, l1; \
        size_t ish, jsh, ksh, lsh, idm; \
        size_t ishp, jshp, kshp, lshp; \
        char mask_ij, mask_kl; \
        double *q_cond_ijij = vhfopt->q_cond; \
        double *q_cond_iijj = vhfopt->q_cond + Nbas*Nbas; \
        double *dm_cond = vhfopt->dm_cond; \
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff; \
        double qijkl, q1, q2, q3;

#define BEGIN_SHELLS_LOOP \
        for (ish = ish0; ish < ish1; ish++) { \
                ishp = bvkcell_shl_id[ish]; \
                i0 = ao_loc[ishp]; \
                i1 = ao_loc[ishp+1]; \
                for (jsh = jsh0; jsh < jsh1; jsh++) { \
                        mask_ij = ovlp_mask[ish * njsh + jsh]; \
                        if (!mask_ij) { \
                                continue; \
                        } \
                        jshp = bvkcell_shl_id[jsh]; \
                        j0 = ao_loc[jshp]; \
                        j1 = ao_loc[jshp+1]; \
                        for (ksh = ksh0; ksh < ksh1; ksh++) { \
                                kshp = bvkcell_shl_id[ksh]; \
                                k0 = ao_loc[kshp]; \
                                k1 = ao_loc[kshp+1]; \
                                for (lsh = lsh0; lsh < lsh1; lsh++) { \
                                        mask_kl = ovlp_mask[ksh * nlsh + lsh]; \
                                        if (!mask_kl) { \
                                                continue; \
                                        } \
                                        if (mask_ij & mask_kl & SKIP_SMOOTH_BLOCK) { \
                                                continue; \
                                        } \
                                        q1 = q_cond_ijij[ish*Nbas+jsh] * q_cond_ijij[ksh*Nbas+lsh]; \
                                        if (q1 < direct_scf_cutoff) { \
                                                continue; \
                                        } \
                                        q2 = q_cond_iijj[ish*Nbas+ksh] * q_cond_iijj[jsh*Nbas+lsh]; \
                                        if (q2 < direct_scf_cutoff) { \
                                                continue; \
                                        } \
                                        q3 = q_cond_iijj[ish*Nbas+lsh] * q_cond_iijj[jsh*Nbas+ksh]; \
                                        if (q3 < direct_scf_cutoff) { \
                                                continue; \
                                        } \
                                        qijkl = MIN(q1, q2); \
                                        qijkl = MIN(qijkl, q3);
#define END_SHELLS_LOOP \
                                } \
                        } \
                } \
        }

void PBCVHF_contract_k_s1(double *vk, double *dms, double *buf, double *cache,
                          int n_dm, int nao, int nkpts, int nbands, int nbas0,
                          int *ishls, int *jshls, int *kshls, int *lshls,
                          int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                          CVHFOpt *vhfopt, IntorEnvs *envs)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        CINTOpt *cintopt = envs->cintopt;
        const int *ao_loc = envs->ao_loc;
        const int *shls_slice = envs->shls_slice;
        const size_t Nbas = nbas;
        const size_t njsh = shls_slice[3] - shls_slice[2];
        const size_t nlsh = shls_slice[7] - shls_slice[6];
        const size_t knbas = nbas0 * nkpts;
        const size_t kn = nao * nkpts;
        const size_t bn = nao * nbands;
        const size_t knn = kn * nao;
        const size_t bnn = bn * nao;
        const size_t kknn = knn * nkpts;
        const size_t ish0 = ishls[0];
        const size_t ish1 = ishls[1];
        const size_t jsh0 = jshls[0];
        const size_t jsh1 = jshls[1];
        const size_t ksh0 = kshls[0];
        const size_t ksh1 = kshls[1];
        const size_t lsh0 = lshls[0];
        const size_t lsh1 = lshls[1];
        int shls[4];
        size_t i, j, k, l, ijkl;
        size_t i0, j0, k0, l0;
        size_t i1, j1, k1, l1;
        size_t ish, jsh, ksh, lsh, idm;
        size_t ishp, jshp, kshp;
        char mask_ij, mask_kl;

        double *q_cond_ijij = vhfopt->q_cond;
        double *q_cond_iijj = vhfopt->q_cond + Nbas*Nbas;
        double *dm_cond = vhfopt->dm_cond;
        double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
        double qijkl, q1, q2, q3, sjk;

        for (ish = ish0; ish < ish1; ish++) {
                ishp = bvkcell_shl_id[ish];
                i0 = ao_loc[ishp];
                i1 = ao_loc[ishp+1];
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        mask_ij = ovlp_mask[ish * njsh + jsh];
                        if (!mask_ij) {
                                continue;
                        }
                        jshp = bvkcell_shl_id[jsh];
                        j0 = ao_loc[jshp];
                        j1 = ao_loc[jshp+1];
                        for (ksh = ksh0; ksh < ksh1; ksh++) {
                                kshp = bvkcell_shl_id[ksh];
                                k0 = ao_loc[kshp];
                                k1 = ao_loc[kshp+1];
                                for (lsh = lsh0; lsh < lsh1; lsh++) {
                                        mask_kl = ovlp_mask[ksh * nlsh + lsh];
                                        if (!mask_kl) {
                                                continue;
                                        }
                                        // skip if all the four shells are smooth functions
                                        if (mask_ij & mask_kl & SKIP_SMOOTH_BLOCK) {
                                                continue;
                                        }

                                        q1 = q_cond_ijij[ish*Nbas+jsh] * q_cond_ijij[ksh*Nbas+lsh];
                                        if (q1 < direct_scf_cutoff) {
                                                continue;
                                        }
                                        q2 = q_cond_iijj[ish*Nbas+ksh] * q_cond_iijj[jsh*Nbas+lsh];
                                        if (q2 < direct_scf_cutoff) {
                                                continue;
                                        }
                                        q3 = q_cond_iijj[ish*Nbas+lsh] * q_cond_iijj[jsh*Nbas+ksh];
                                        if (q3 < direct_scf_cutoff) {
                                                continue;
                                        }
                                        qijkl = MIN(q1, q2);
                                        qijkl = MIN(qijkl, q3);
                                        if (dm_cond[jshp*knbas+kshp]*qijkl < direct_scf_cutoff) {
                                                continue;
                                        }

                                        shls[0] = ish;
                                        shls[1] = jsh;
                                        shls[2] = ksh;
                                        shls[3] = lsh;
                                        if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env,
                                                      cintopt, cache)) {
                                                l0 = bands_ao_loc[lsh];
                                                l1 = l0 + ao_loc[lsh+1] - ao_loc[lsh];
        if (n_dm == 1) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        sjk = dms[j*kn+k];
                        for (i = i0; i < i1; i++, ijkl++) {
                                qijkl = buf[ijkl];
                                vk[i*bn+l] += qijkl * sjk;
                        }
                } } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ijkl++) {
                        qijkl = buf[ijkl];
                        for (idm = 0; idm < n_dm; idm++) {
                                vk[idm*bnn + i*bn+l] += qijkl * dms[idm*kknn + j*kn+k];
                        }
                } } } }
        }
                                        }  // endif int2e_sph
                                }
                        }
                }
        }
}

static void contract_k_s2_kgtl(double *vk, double *dms, double *buf, double *cache,
                               int n_dm, int nao, int nkpts, int nbands, int nbas0,
                               int *ishls, int *jshls, int *kshls, int *lshls,
                               int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                               CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        double sjk, sjl;
        int koff, loff;

        BEGIN_SHELLS_LOOP {
                lshp = bvkcell_shl_id[lsh];
                if ((dm_cond[jshp*knbas+kshp]*qijkl < direct_scf_cutoff) &&
                    (dm_cond[jshp*knbas+lshp]*qijkl < direct_scf_cutoff)) {
                        continue;
                }

                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env,
                              cintopt, cache)) {
                        l0 = ao_loc[lshp];
                        l1 = ao_loc[lshp+1];
                        koff = bands_ao_loc[ksh] - k0;
                        loff = bands_ao_loc[lsh] - l0;
                        if (n_dm == 1) {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        sjk = dms[j*kn+k];
                                        sjl = dms[j*kn+l];
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                qijkl = buf[ijkl];
                                                vk[i*bn+l+loff] += qijkl * sjk;
                                                vk[i*bn+k+koff] += qijkl * sjl;
                                        }
                                } } }
                        } else {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        qijkl = buf[ijkl];
                                        for (idm = 0; idm < n_dm; idm++) {
                                                vk[idm*bnn + i*bn+l+loff] += qijkl * dms[idm*kknn + j*kn+k];
                                                vk[idm*bnn + i*bn+k+koff] += qijkl * dms[idm*kknn + j*kn+l];
                                        }
                                } } } }
                        }
                }  // endif int2e_sph
        } END_SHELLS_LOOP
}

void PBCVHF_contract_k_s2kl(double *vk, double *dms, double *buf, double *cache,
                            int n_dm, int nao, int nkpts, int nbands, int nbas0,
                            int *ishls, int *jshls, int *kshls, int *lshls,
                            int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                            CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (kshls[0] > lshls[0]) {
                contract_k_s2_kgtl(vk, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                                   ishls, jshls, kshls, lshls,
                                   bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {
                PBCVHF_contract_k_s1(vk, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                                     ishls, jshls, kshls, lshls,
                                     bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, envs);
        }
}

void PBCVHF_contract_j_s1(double *vj, double *dms, double *buf, double *cache,
                          int n_dm, int nao, int nkpts, int nbands, int nbas0,
                          int *ishls, int *jshls, int *kshls, int *lshls,
                          int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                          CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        double dm_sum[n_dm];

        BEGIN_SHELLS_LOOP {
                lshp = bvkcell_shl_id[lsh];
                if (dm_cond[lshp*knbas+kshp]*qijkl < direct_scf_cutoff) {
                        continue;
                }


                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env,
                              cintopt, cache)) {
                        j0 = bands_ao_loc[jsh];
                        j1 = j0 + ao_loc[jsh+1] - ao_loc[jsh];
                        l0 = ao_loc[lshp];
                        l1 = ao_loc[lshp+1];
                        if (n_dm == 1) {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        dm_sum[0] = dms[l*kn+k];
                                        for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                vj[i*bn+j] += buf[ijkl] * dm_sum[0];
                                        } }
                                } }
                        } else {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        for (idm = 0; idm < n_dm; idm++) {
                                                dm_sum[idm] = dms[idm*kknn + l*kn+k];
                                        }
                                        for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                qijkl = buf[ijkl];
                                                for (idm = 0; idm < n_dm; idm++) {
                                                        vj[idm*bnn + i*bn+j] += qijkl * dm_sum[idm];
                                                }
                                        } }
                                } }
                        }
                }  // endif int2e_sph
        } END_SHELLS_LOOP
}

static void contract_j_s2_kgtl(double *vj, double *dms, double *buf, double *cache,
                               int n_dm, int nao, int nkpts, int nbands, int nbas0,
                               int *ishls, int *jshls, int *kshls, int *lshls,
                               int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                               CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        double dm_sum[n_dm];

        BEGIN_SHELLS_LOOP {
                lshp = bvkcell_shl_id[lsh];
                if ((dm_cond[kshp*knbas+lshp]+dm_cond[lshp*knbas+kshp])*qijkl < direct_scf_cutoff) {
                        continue;
                }

                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env,
                              cintopt, cache)) {
                        j0 = bands_ao_loc[jsh];
                        j1 = j0 + ao_loc[jsh+1] - ao_loc[jsh];
                        l0 = ao_loc[lshp];
                        l1 = ao_loc[lshp+1];
                        if (n_dm == 1) {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        dm_sum[0] = dms[k*kn+l] + dms[l*kn+k];
                                        for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                vj[i*bn+j] += buf[ijkl] * dm_sum[0];
                                        } }
                                } }
                        } else {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        for (idm = 0; idm < n_dm; idm++) {
                                                dm_sum[idm] = dms[idm*kknn + l*kn+k]
                                                        + dms[idm*kknn + k*kn+l];
                                        }
                                        for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                qijkl = buf[ijkl];
                                                for (idm = 0; idm < n_dm; idm++) {
                                                        vj[idm*bnn + i*bn+j] += qijkl * dm_sum[idm];
                                                }
                                        } }
                                } }
                        }
                }  // endif int2e_sph
        } END_SHELLS_LOOP
}

void PBCVHF_contract_j_s2kl(double *vj, double *dms, double *buf, double *cache,
                            int n_dm, int nao, int nkpts, int nbands, int nbas0,
                            int *ishls, int *jshls, int *kshls, int *lshls,
                            int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                            CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (kshls[0] > lshls[0]) {
                contract_j_s2_kgtl(vj, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                                   ishls, jshls, kshls, lshls,
                                   bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {
                PBCVHF_contract_j_s1(vj, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                                     ishls, jshls, kshls, lshls,
                                     bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, envs);
        }
}

void PBCVHF_contract_jk_s1(double *jk, double *dms, double *buf, double *cache,
                           int n_dm, int nao, int nkpts, int nbands, int nbas0,
                           int *ishls, int *jshls, int *kshls, int *lshls,
                           int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                           CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        double sjk;
        double dm_sum[n_dm];
        double *vj = jk;
        double *vk = jk + n_dm * bnn;
        int joff, loff;

        BEGIN_SHELLS_LOOP {
                lshp = bvkcell_shl_id[lsh];
                if ((dm_cond[jshp*knbas+kshp]*qijkl < direct_scf_cutoff) &&
                    (dm_cond[jshp*knbas+lshp]*qijkl < direct_scf_cutoff) &&
                    (dm_cond[kshp*knbas+lshp]+dm_cond[lshp*knbas+kshp])*qijkl < direct_scf_cutoff) {
                        continue;
                }

                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env,
                              cintopt, cache)) {
                        l0 = ao_loc[lshp];
                        l1 = ao_loc[lshp+1];
                        joff = bands_ao_loc[jsh] - j0;
                        loff = bands_ao_loc[lsh] - l0;
                        if (n_dm == 1) {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        dm_sum[0] = dms[l*kn+k];
                                        for (j = j0; j < j1; j++) {
                                                sjk = dms[j*kn+k];
                                                for (i = i0; i < i1; i++, ijkl++) {
                                                        qijkl = buf[ijkl];
                                                        vj[i*bn+j+joff] += qijkl * dm_sum[0];
                                                        vk[i*bn+l+loff] += qijkl * sjk;
                                                }
                                        }
                                } }
                        } else {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        for (idm = 0; idm < n_dm; idm++) {
                                                dm_sum[idm] = dms[idm*kknn + l*kn+k];
                                        }
                                        for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                qijkl = buf[ijkl];
                                                for (idm = 0; idm < n_dm; idm++) {
                                                        vj[idm*bnn + i*bn+j+joff] += qijkl * dm_sum[idm];
                                                        vk[idm*bnn + i*bn+l+loff] += qijkl * dms[idm*kknn + j*kn+k];
                                                }
                                        } }
                                } }
                        }
                }  // endif int2e_sph
        } END_SHELLS_LOOP
}

static void contract_jk_s2_kgtl(double *jk, double *dms, double *buf, double *cache,
                                int n_dm, int nao, int nkpts, int nbands, int nbas0,
                                int *ishls, int *jshls, int *kshls, int *lshls,
                                int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                                CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        double sjk, sjl;
        double dm_sum[n_dm];
        double *vj = jk;
        double *vk = jk + n_dm * bnn;
        int joff, koff, loff;

        BEGIN_SHELLS_LOOP {
                lshp = bvkcell_shl_id[lsh];
                if ((dm_cond[jshp*knbas+kshp]*qijkl < direct_scf_cutoff) &&
                    (dm_cond[jshp*knbas+lshp]*qijkl < direct_scf_cutoff) &&
                    (dm_cond[kshp*knbas+lshp]+dm_cond[lshp*knbas+kshp])*qijkl < direct_scf_cutoff) {
                        continue;
                }

                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (int2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env,
                              cintopt, cache)) {
                        l0 = ao_loc[lshp];
                        l1 = ao_loc[lshp+1];
                        joff = bands_ao_loc[jsh] - j0;
                        koff = bands_ao_loc[ksh] - k0;
                        loff = bands_ao_loc[lsh] - l0;
                        if (n_dm == 1) {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        dm_sum[0] = dms[k*kn+l] + dms[l*kn+k];
                                        for (j = j0; j < j1; j++) {
                                                sjk = dms[j*kn+k];
                                                sjl = dms[j*kn+l];
                                                for (i = i0; i < i1; i++, ijkl++) {
                                                        qijkl = buf[ijkl];
                                                        vj[i*bn+j+joff] += qijkl * dm_sum[0];
                                                        vk[i*bn+l+loff] += qijkl * sjk;
                                                        vk[i*bn+k+koff] += qijkl * sjl;
                                                }
                                        }
                                } }
                        } else {
                                ijkl = 0;
                                for (l = l0; l < l1; l++) {
                                for (k = k0; k < k1; k++) {
                                        for (idm = 0; idm < n_dm; idm++) {
                                                dm_sum[idm] = dms[idm*kknn + l*kn+k]
                                                        + dms[idm*kknn + k*kn+l];
                                        }
                                        for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++, ijkl++) {
                                                qijkl = buf[ijkl];
                                                for (idm = 0; idm < n_dm; idm++) {
                                                        vj[idm*bnn + i*bn+j+joff] += qijkl * dm_sum[idm];
                                                        vk[idm*bnn + i*bn+l+loff] += qijkl * dms[idm*kknn + j*kn+k];
                                                        vk[idm*bnn + i*bn+k+koff] += qijkl * dms[idm*kknn + j*kn+l];
                                                }
                                        } }
                                } }
                        }
                }  // endif int2e_sph
        } END_SHELLS_LOOP
}

void PBCVHF_contract_jk_s2kl(double *jk, double *dms, double *buf, double *cache,
                             int n_dm, int nao, int nkpts, int nbands, int nbas0,
                             int *ishls, int *jshls, int *kshls, int *lshls,
                             int *bvkcell_shl_id, int *bands_ao_loc, char *ovlp_mask,
                             CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (kshls[0] > lshls[0]) {
                contract_jk_s2_kgtl(jk, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                                    ishls, jshls, kshls, lshls,
                                    bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {
                PBCVHF_contract_jk_s1(jk, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                                      ishls, jshls, kshls, lshls,
                                      bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, envs);
        }
}

// nbas0 is the number of basis in primitive cell
void PBCVHF_direct_drv(void (*fdot)(), double *out, double *dms,
                       int n_dm, int nao, int nkpts, int nbands, int nbas0,
                       int *shls_slice, int *ao_loc,
                       int *bvkcell_shl_id, int *bands_ao_loc,
                       char *ovlp_mask, CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc,
                NULL, cintopt, 1};

        // Note: shls_slice refers to the shells of entire sup-mol. ao_loc has
        // only the information of a subset of the sup-mol. Since bases of
        // supmol are extended from primitive cell. di can be generated in terms
        // of the primitive cell (cell0)
        int cell0_shls_slice[2] = {0, nbas0};
        const int di = GTOmax_shell_dim(ao_loc, cell0_shls_slice, 1);

        const int cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int lsh0 = shls_slice[6];
        const int lsh1 = shls_slice[7];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int nksh = ksh1 - ksh0;
        const int nlsh = lsh1 - lsh0;
        int *block_iloc = malloc(sizeof(int) * (nish + njsh + nksh + nlsh + 4));
        int *block_jloc = block_iloc + nish + 1;
        int *block_kloc = block_jloc + njsh + 1;
        int *block_lloc = block_kloc + nksh + 1;
        const size_t nblock_i = CVHFshls_block_partition(block_iloc, shls_slice+0, ao_loc);
        const size_t nblock_j = CVHFshls_block_partition(block_jloc, shls_slice+2, ao_loc);
        const size_t nblock_k = CVHFshls_block_partition(block_kloc, shls_slice+4, ao_loc);
        const size_t nblock_l = CVHFshls_block_partition(block_lloc, shls_slice+6, ao_loc);
        const size_t nblock_kl = nblock_k * nblock_l;
        const size_t nblock_jkl = nblock_j * nblock_kl;
        const size_t nblock_ijkl = nblock_i * nblock_jkl;

        char *block_mask = calloc(nblock_kl, sizeof(char));
        char has_type2;
        int kb, lb, ks0, ks1, ls0, ls1, ks, ls;
        for (kb = 0; kb < nblock_k; kb++) {
        for (lb = 0; lb < nblock_l; lb++) {
                ks0 = block_kloc[kb];
                ls0 = block_lloc[lb];
                ks1 = block_lloc[kb+1];
                ls1 = block_lloc[lb+1];
                has_type2 = 0;
                for (ks = ks0; ks < ks1; ks++) {
                for (ls = ls0; ls < ls1; ls++) {
                        if (ovlp_mask[ks * nbas + ls] == 1) {
                                // any ovlp_mask is type 1
                                block_mask[kb * nblock_l + lb] = 1;
                                goto next_block;
                        } else {
                                // any ovlp_mask is type 2 and none is type 1
                                has_type2 |= ovlp_mask[ks * nbas + ls] == 2;
                        }
                } }
                if (has_type2) {
                        block_mask[kb * nblock_l + lb] = 2;
                }
next_block:;
        } }

#pragma omp parallel
{
        size_t i, j, k, l, r, n, blk_id;
        size_t size = n_dm * nao * nao * nbands;
        if (fdot == &PBCVHF_contract_jk_s2kl || fdot == &PBCVHF_contract_jk_s1) {
                size *= 2;
        }
        double *v_priv = calloc(size, sizeof(double));
        double *buf = malloc(sizeof(double) * (di*di*di*di + cache_size));
        double *cache = buf + di*di*di*di;
        char mask_ij, mask_kl;
#pragma omp for nowait schedule(dynamic, 8)
        for (blk_id = 0; blk_id < nblock_ijkl; blk_id++) {
                // dispatch blk_id to sub-block indices (i, j, k, l)
                r = blk_id;
                i = r / nblock_jkl; r = r - i * nblock_jkl;
                j = r / nblock_kl ; r = r - j * nblock_kl;
                mask_ij = block_mask[i * nblock_j + j];
                if (mask_ij == 0) {
                        continue;
                }

                k = r / nblock_l  ; r = r - k * nblock_l;
                l = r;
                mask_kl = block_mask[k * nblock_l + l];
                if (mask_kl == 0) {
                        continue;
                }

                if (mask_ij & mask_kl & SKIP_SMOOTH_BLOCK) {
                        continue;
                }

                (*fdot)(v_priv, dms, buf, cache, n_dm, nao, nkpts, nbands, nbas0,
                        block_iloc+i, block_jloc+j, block_kloc+k, block_lloc+l,
                        bvkcell_shl_id, bands_ao_loc, ovlp_mask, vhfopt, &envs);
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
        free(block_iloc);
        free(block_mask);
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
        if (opt->q_cond) {
                free(opt->q_cond);
        }
        // nbas in the input arguments may different to opt->nbas.
        // Use opt->nbas because it is used in the prescreen function
        nbas = opt->nbas;
        opt->q_cond = (double *)malloc(sizeof(double) * nbas * nbas * 2);
        double *qcond_ijij = opt->q_cond;
        double *qcond_iijj = qcond_ijij + nbas * nbas;
        CVHFset_int2e_q_cond(intor, cintopt, qcond_ijij, ao_loc,
                             atm, natm, bas, nbas, env);
        CVHFset_int2e_q_cond(_int2e_swap_jk, cintopt, qcond_iijj, ao_loc,
                             atm, natm, bas, nbas, env);
}
