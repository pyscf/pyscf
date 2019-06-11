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
#include <math.h>
#include <assert.h>
#include "cint.h"
#include "cvhf.h"
#include "optimizer.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

int int2e_sph();
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        CVHFOpt *opt0 = (CVHFOpt *)malloc(sizeof(CVHFOpt));
        opt0->nbas = nbas;
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
        if (!opt0) {
                return;
        }

        if (!opt0->q_cond) {
                free(opt0->q_cond);
        }
        if (!opt0->dm_cond) {
                free(opt0->dm_cond);
        }

        free(opt0);
        *opt = NULL;
}

int CVHFnoscreen(int *shls, CVHFOpt *opt,
                 int *atm, int *bas, double *env)
{
        return 1;
}

int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt,
                        int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        return qijkl > opt->direct_scf_cutoff;
}

int CVHFnrs8_prescreen(int *shls, CVHFOpt *opt,
                       int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
        double *q_cond = opt->q_cond;
        double *dm_cond = opt->dm_cond;
        assert(q_cond);
        assert(dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = q_cond[i*n+j] * q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((4*dm_cond[j*n+i] > dmin)
            || (4*dm_cond[l*n+k] > dmin)
            || (  dm_cond[j*n+k] > dmin)
            || (  dm_cond[j*n+l] > dmin)
            || (  dm_cond[i*n+k] > dmin)
            || (  dm_cond[i*n+l] > dmin));
}

int CVHFnrs8_vj_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
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

int CVHFnrs8_vk_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
        double *q_cond = opt->q_cond;
        double *dm_cond = opt->dm_cond;
        assert(q_cond);
        assert(dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = q_cond[i*n+j] * q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((  dm_cond[j*n+k] > dmin)
            || (  dm_cond[j*n+l] > dmin)
            || (  dm_cond[i*n+k] > dmin)
            || (  dm_cond[i*n+l] > dmin));
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
        if (opt->q_cond) {
                free(opt->q_cond);
        }
        opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        double qtmp, tmp;
        int ij, i, j, di, dj, ish, jsh;
        int shls[4];
        double *cache = malloc(sizeof(double) * cache_size);
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *buf = malloc(sizeof(double) * di*di*di*di);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
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
                opt->q_cond[ish*nbas+jsh] = qtmp;
                opt->q_cond[jsh*nbas+ish] = qtmp;
        }
        free(buf);
        free(cache);
}
}

void CVHFsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->dm_cond) { // NOT reuse opt->dm_cond because nset may be diff in different call
                free(opt->dm_cond);
        }
        opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        memset(opt->dm_cond, 0, sizeof(double)*nbas*nbas);

        const size_t nao = ao_loc[nbas];
        double dmax, tmp;
        int i, j, ish, jsh;
        int iset;
        double *pdm;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        pdm = dm + nao*nao*iset;
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
// symmetrize dm_cond because nrs8_prescreen only tests the lower (or upper)
// triangular part of dm_cond. If density matrix is not hermitian, some
// integrals may be skipped incorrectly.
                                tmp = .5 * (fabs(pdm[i*nao+j]) + fabs(pdm[j*nao+i]));
                                dmax = MAX(dmax, tmp);
                        } }
                }
                opt->dm_cond[ish*nbas+jsh] = dmax;
                opt->dm_cond[jsh*nbas+ish] = dmax;
        } }
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
