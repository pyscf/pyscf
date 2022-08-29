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
#include <math.h>
#include <assert.h>
#include "cint.h"
#include "cvhf.h"
#include "fblas.h"
#include "optimizer.h"
#include "nr_direct.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

#include <stdio.h>

int int2e_sph();

void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        CVHFOpt *opt0 = (CVHFOpt *)malloc(sizeof(CVHFOpt));
        opt0->nbas = nbas;
        opt0->ngrids = 0;
        opt0->direct_scf_cutoff = 1e-14;
        opt0->q_cond = NULL;
        opt0->dm_cond = NULL;
        opt0->fprescreen = &CVHFnoscreen;
        opt0->r_vkscreen = &CVHFr_vknoscreen;
        *opt = opt0;
}

void CVHFdel_optimizer(CVHFOpt **opt)
{
        CVHFOpt *opt0 = *opt;
        if (opt0 == NULL) {
                return;
        }

        if (opt0->q_cond != NULL) {
                free(opt0->q_cond);
        }
        if (opt0->dm_cond != NULL) {
                free(opt0->dm_cond);
        }

        free(opt0);
        *opt = NULL;
}

int CVHFnoscreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env)
{
        return 1;
}

int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        assert(opt->q_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        return qijkl > opt->direct_scf_cutoff;
}

int CVHFnrs8_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        double *q_cond = opt->q_cond;
        double *dm_cond = opt->dm_cond;
        assert(q_cond);
        assert(dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = q_cond[i*n+j] * q_cond[k*n+l];
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        return qijkl > direct_scf_cutoff
            &&((4*dm_cond[j*n+i]*qijkl > direct_scf_cutoff)
            || (4*dm_cond[l*n+k]*qijkl > direct_scf_cutoff)
            || (  dm_cond[j*n+k]*qijkl > direct_scf_cutoff)
            || (  dm_cond[j*n+l]*qijkl > direct_scf_cutoff)
            || (  dm_cond[i*n+k]*qijkl > direct_scf_cutoff)
            || (  dm_cond[i*n+l]*qijkl > direct_scf_cutoff));
}

int CVHFnrs8_vj_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        return qijkl > direct_scf_cutoff
            &&((4*qijkl*opt->dm_cond[j*n+i] > direct_scf_cutoff)
            || (4*qijkl*opt->dm_cond[l*n+k] > direct_scf_cutoff));
}

int CVHFnrs8_vk_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        size_t n = opt->nbas;
        double *q_cond = opt->q_cond;
        double *dm_cond = opt->dm_cond;
        assert(q_cond);
        assert(dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = q_cond[i*n+j] * q_cond[k*n+l];
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        return qijkl > direct_scf_cutoff
            &&((  dm_cond[j*n+k]*qijkl > direct_scf_cutoff)
            || (  dm_cond[j*n+l]*qijkl > direct_scf_cutoff)
            || (  dm_cond[i*n+k]*qijkl > direct_scf_cutoff)
            || (  dm_cond[i*n+l]*qijkl > direct_scf_cutoff));
}

int CVHFnrs8_vj_prescreen_block(CVHFOpt *opt, int *ishls, int *jshls, int *kshls, int *lshls)
{
        int i0 = ishls[0];
        int j0 = jshls[0];
        int k0 = kshls[0];
        int l0 = lshls[0];
        int i1 = ishls[1];
        int j1 = jshls[1];
        int k1 = kshls[1];
        int l1 = lshls[1];
        int i, j, k, l;
        size_t n = opt->nbas;
        double *q_cond = opt->q_cond;
        double *dm_cond = opt->dm_cond;
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        double rho, cutoff;
        rho = 0;
        for (j = j0; j < j1; j++) {
#pragma GCC ivdep
        for (i = i0; i < i1; i++) {
                rho += dm_cond[j*n+i] * q_cond[j*n+i];
        } }

        if (rho != 0) {
                cutoff = 4 * direct_scf_cutoff / fabs(rho);
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        if (q_cond[l*n+k] > cutoff) {
                                return 1;
                        }
                } }
        }

        rho = 0;
        for (l = l0; l < l1; l++) {
#pragma GCC ivdep
        for (k = k0; k < k1; k++) {
                rho += dm_cond[l*n+k] * q_cond[l*n+k];
        } }

        if (rho != 0) {
                cutoff = 4 * direct_scf_cutoff / fabs(rho);
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        if (q_cond[j*n+i] > cutoff) {
                                return 1;
                        }
                } }
        }
        return 0;
}

int CVHFnrs8_vk_prescreen_block(CVHFOpt *opt, int *ishls, int *jshls, int *kshls, int *lshls)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        int i0 = ishls[0];
        int j0 = jshls[0];
        int k0 = kshls[0];
        int l0 = lshls[0];
        int i1 = ishls[1];
        int j1 = jshls[1];
        int k1 = kshls[1];
        int l1 = lshls[1];
        int di = i1 - i0;
        int dj = j1 - j0;
        int dk = k1 - k0;
        int dl = l1 - l0;
        int i;
        int nbas = opt->nbas;
        size_t n = opt->nbas;
        double *q_cond = opt->q_cond;
        double *dm_cond = opt->dm_cond;
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        assert(di < 128);
        assert(dj < 128);
        assert(dk < 128);
        assert(dl < 128);
        double buf1[128*128];
        double buf2[128*128];

        dgemm_(&TRANS_N, &TRANS_T, &di, &dk, &dj, &D1, q_cond+j0*n+i0, &nbas, dm_cond+j0*n+k0, &nbas, &D0, buf1, &di);
        dgemm_(&TRANS_N, &TRANS_T, &dl, &di, &dk, &D1, q_cond+k0*n+l0, &nbas, buf1, &di, &D0, buf2, &dl);
        for (i = 0; i < di * dl; i++) {
                if (buf2[i] > direct_scf_cutoff) {
                        return 1;
                }
        }

        dgemm_(&TRANS_N, &TRANS_T, &di, &dl, &dj, &D1, q_cond+j0*n+i0, &nbas, dm_cond+j0*n+l0, &nbas, &D0, buf1, &di);
        dgemm_(&TRANS_N, &TRANS_T, &dk, &di, &dl, &D1, q_cond+l0*n+k0, &nbas, buf1, &di, &D0, buf2, &dk);
        for (i = 0; i < di * dk; i++) {
                if (buf2[i] > direct_scf_cutoff) {
                        return 1;
                }
        }

        dgemm_(&TRANS_N, &TRANS_T, &dj, &dk, &di, &D1, q_cond+i0*n+j0, &nbas, dm_cond+i0*n+k0, &nbas, &D0, buf1, &dj);
        dgemm_(&TRANS_N, &TRANS_T, &dl, &dj, &dk, &D1, q_cond+k0*n+l0, &nbas, buf1, &dj, &D0, buf2, &dl);
        for (i = 0; i < dj * dl; i++) {
                if (buf2[i] > direct_scf_cutoff) {
                        return 1;
                }
        }

        dgemm_(&TRANS_N, &TRANS_T, &dj, &dl, &di, &D1, q_cond+i0*n+j0, &nbas, dm_cond+i0*n+l0, &nbas, &D0, buf1, &dj);
        dgemm_(&TRANS_N, &TRANS_T, &dk, &dj, &dl, &D1, q_cond+l0*n+k0, &nbas, buf1, &dj, &D0, buf2, &dk);
        for (i = 0; i < dj * dk; i++) {
                if (buf2[i] > direct_scf_cutoff) {
                        return 1;
                }
        }
        return 0;
}

int CVHFnrs8_prescreen_block(CVHFOpt *opt, int *ishls, int *jshls, int *kshls, int *lshls)
{
        return (CVHFnrs8_vj_prescreen_block(opt, ishls, jshls, kshls, lshls) ||
                CVHFnrs8_vk_prescreen_block(opt, ishls, jshls, kshls, lshls));
}

// return flag to decide whether transpose01324
int CVHFr_vknoscreen(int *shls, CVHFOpt *opt,
                     double **dms_cond, int n_dm, double *dm_atleast,
                     int *atm, int *bas, double *env)
{
        int idm;
        for (idm = 0; idm < n_dm; idm++) {
                dms_cond[idm] = NULL;
        }
        *dm_atleast = 0;
        return 1;
}

int CVHFnr3c2e_vj_pass1_prescreen(int *shls, CVHFOpt *opt,
                               int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        size_t n = opt->nbas;
        int i = shls[0];
        int j = shls[1];
        // Be careful with the range of basis k, which is between nbas and
        // nbas+nauxbas. See shls_slice in df_jk.get_j function.
        int k = shls[2] - n;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[n*n+k];
        return qijkl > direct_scf_cutoff
            && (4*qijkl*opt->dm_cond[j*n+i] > direct_scf_cutoff);
}

int CVHFnr3c2e_vj_pass2_prescreen(int *shls, CVHFOpt *opt,
                               int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        size_t n = opt->nbas;
        int i = shls[0];
        int j = shls[1];
        // Be careful with the range of basis k, which is between nbas and
        // nbas+nauxbas. See shls_slice in df_jk.get_j function.
        int k = shls[2] - n;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        double direct_scf_cutoff = opt->direct_scf_cutoff;
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[n*n+k];
        return qijkl > direct_scf_cutoff
            && (4*qijkl*opt->dm_cond[k] > direct_scf_cutoff);
}

int CVHFnr3c2e_schwarz_cond(int *shls, CVHFOpt *opt,
                            int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        size_t n = opt->nbas;
        int i = shls[0];
        int j = shls[1];
        // Be careful with the range of basis k, which is between nbas and
        // nbas+nauxbas. See shls_slice in df_jk.get_j function.
        int k = shls[2] - n;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[n*n+k];
        return qijkl > opt->direct_scf_cutoff;
}


void CVHFset_direct_scf_cutoff(CVHFOpt *opt, double cutoff)
{
        opt->direct_scf_cutoff = cutoff;
}
double CVHFget_direct_scf_cutoff(CVHFOpt *opt)
{
        return opt->direct_scf_cutoff;
}


void CVHFsetnr_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
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
        opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        CVHFset_int2e_q_cond(intor, cintopt, opt->q_cond, ao_loc,
                             atm, natm, bas, nbas, env);
}

/*
 * Non-relativistic 2-electron integrals
 */
void CVHFset_int2e_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                          int *ao_loc, int *atm, int natm,
                          int *bas, int nbas, double *env)
{
        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        double qtmp, tmp;
        size_t ij, i, j, di, dj, ish, jsh;
        size_t Nbas = nbas;
        int shls[4];
        double *cache = malloc(sizeof(double) * cache_size);
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *buf = malloc(sizeof(double) * di*di*di*di);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < Nbas*(Nbas+1)/2; ij++) {
                ish = (size_t)(sqrt(2*ij+.25) - .5 + 1e-7);
                jsh = ij - ish*(ish+1)/2;
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ish;
                shls[3] = jsh;
                qtmp = 1e-100;
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                  cintopt, cache)) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                tmp = fabs(buf[i+di*j+di*dj*i+di*dj*di*j]);
                                qtmp = MAX(qtmp, tmp);
                        } }
                        qtmp = sqrt(qtmp);
                }
                q_cond[ish*nbas+jsh] = qtmp;
                q_cond[jsh*nbas+ish] = qtmp;
        }
        free(buf);
        free(cache);
}
}

void CVHFset_q_cond(CVHFOpt *opt, double *q_cond, int len)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        opt->q_cond = (double *)malloc(sizeof(double) * len);
        NPdcopy(opt->q_cond, q_cond, len);
}

void CVHFsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->dm_cond != NULL) { // NOT reuse opt->dm_cond because nset may be diff in different call
                free(opt->dm_cond);
        }
        // nbas in the input arguments may different to opt->nbas.
        // Use opt->nbas because it is used in the prescreen function
        nbas = opt->nbas;
        opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        NPdset0(opt->dm_cond, ((size_t)nbas)*nbas);

        const size_t nao = ao_loc[nbas];
        double dmax, tmp;
        size_t i, j, ish, jsh, iset;
        double *pdm;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        pdm = dm + nao*nao*iset;
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
// symmetrize dm_cond because nrs8_prescreen only tests the lower (or upper)
// triangular part of dm_cond. Without the symmetrization, some integrals may be
// incorrectly skipped.
                                tmp = fabs(pdm[i*nao+j]) + fabs(pdm[j*nao+i]);
                                dmax = MAX(dmax, tmp);
                        } }
                }
                opt->dm_cond[ish*nbas+jsh] = .5 * dmax;
                opt->dm_cond[jsh*nbas+ish] = .5 * dmax;
        } }
}

void CVHFset_dm_cond(CVHFOpt *opt, double *dm_cond, int len)
{
        if (opt->dm_cond != NULL) {
                free(opt->dm_cond);
        }
        opt->dm_cond = (double *)malloc(sizeof(double) * len);
        NPdcopy(opt->dm_cond, dm_cond, len);
}



/*
 *************************************************
 */
void CVHFnr_optimizer(CVHFOpt **vhfopt, int (*intor)(), CINTOpt *cintopt,
                      int *ao_loc, int *atm, int natm,
                      int *bas, int nbas, double *env)
{
        CVHFinit_optimizer(vhfopt, atm, natm, bas, nbas, env);
        (*vhfopt)->fprescreen = &CVHFnrs8_prescreen;
        CVHFsetnr_direct_scf(*vhfopt, intor, cintopt, ao_loc,
                             atm, natm, bas, nbas, env);
}
