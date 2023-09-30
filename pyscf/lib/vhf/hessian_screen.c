/* Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
  
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include "cint.h"
#include "cvhf.h"
#include "optimizer.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

int int2e_sph();
int int2e_cart();
int int2e_ipvip1_cart();
int int2e_spsp1spsp2_cart();
int int2e_spsp1spsp2_sph();

/*
 * Gradients screening for grad/rhf.py
 */

// ijkl,lk->ij
// ijkl,jk->il
// ijkl,kl->ij
// ijkl,jl->ik
int CVHFgrad_jk_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
        double *q_cond_kl = opt->q_cond + n * n;
        double qijkl = opt->q_cond[i*n+j] * q_cond_kl[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((2*opt->dm_cond[l*n+k] > dmin)
            || (  opt->dm_cond[j*n+k] > dmin)
            || (  opt->dm_cond[j*n+l] > dmin));
}

void CVHFnr_int2e_pp_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        int nbas2 = nbas * nbas;
        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel \
        shared(intor, cintopt, ao_loc, atm, natm, bas, nbas, env)
{
        double qtmp;
        int i, j, iijj, di, dj, ish, jsh;
        size_t ij;
        int shls[4];
        double *cache = malloc(sizeof(double) * cache_size);
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *buf = malloc(sizeof(double) * 9 * di*di*di*di);
        double *bufx = buf;
        double *bufy, *bufz;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas2; ij++) {
                ish = ij / nbas;
                jsh = ij - ish * nbas;
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ish;
                shls[3] = jsh;
                qtmp = 1e-100;
                bufy = buf + 4*(di*dj*di*dj);
                bufz = buf + 8*(di*dj*di*dj);
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                  cintopt, cache)) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                iijj = i+di*j+di*dj*i+di*dj*di*j;
                                qtmp = MAX(qtmp, fabs(bufx[iijj]));
                                qtmp = MAX(qtmp, fabs(bufy[iijj]));
                                qtmp = MAX(qtmp, fabs(bufz[iijj]));
                        } }
                        qtmp = sqrt(qtmp);
                }
                q_cond[ish*nbas+jsh] = qtmp;
        }
        free(buf);
        free(cache);
}
}

void CVHFgrad_jk_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        nbas = opt->nbas;
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        // First n*n elements for derivatives, the next n*n elements for regular ERIs
        opt->q_cond = (double *)malloc(sizeof(double) * Nbas2*2);

        if (ao_loc[nbas] == CINTtot_cgto_spheric(bas, nbas)) {
                CVHFnr_int2e_q_cond(int2e_sph, NULL, opt->q_cond+Nbas2, ao_loc,
                                    atm, natm, bas, nbas, env);
        } else {
                CVHFnr_int2e_q_cond(int2e_cart, NULL, opt->q_cond+Nbas2, ao_loc,
                                    atm, natm, bas, nbas, env);
        }
        CVHFnr_int2e_pp_q_cond(intor, cintopt, opt->q_cond, ao_loc,
                               atm, natm, bas, nbas, env);
}

void CVHFgrad_jk_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                               int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->dm_cond != NULL) {
                free(opt->dm_cond);
        }
        nbas = opt->nbas;
        opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        CVHFnr_dm_cond1(opt->dm_cond, dm, nset, ao_loc, atm, natm, bas, nbas, env);
}


/*
 * Hessian screening for hessian/rhf.py
 */

// ijkl,ji->kl
// ijkl,li->kj
// ijkl,lj->ki
int CVHFip1ip2_prescreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((opt->dm_cond[j*n+i] > dmin)
            || (opt->dm_cond[l*n+i] > dmin)
            || (opt->dm_cond[l*n+j] > dmin));
}


void CVHFip1ip2_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                           int *ao_loc, int *atm, int natm,
                           int *bas, int nbas, double *env)
{
        CVHFgrad_jk_direct_scf(opt, intor, cintopt, ao_loc, atm, natm, bas, nbas, env);
}

void CVHFip1ip2_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                              int *atm, int natm, int *bas, int nbas, double *env)
{
        CVHFgrad_jk_direct_scf_dm(opt, dm, nset, ao_loc, atm, natm, bas, nbas, env);
}


// ijkl,lk->ij
// ijkl,jk->il
// ijkl,kl->ij
// ijkl,jl->ik
int CVHFipip1_prescreen(int *shls, CVHFOpt *opt,
                        int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
        double *q_cond_kl = opt->q_cond + n * n;
        double qijkl = opt->q_cond[i*n+j] * q_cond_kl[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((2*opt->dm_cond[l*n+k] > dmin)
            || (  opt->dm_cond[j*n+k] > dmin)
            || (  opt->dm_cond[j*n+l] > dmin));
}

void CVHFnr_int2e_pppp_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                              int *ao_loc, int *atm, int natm,
                              int *bas, int nbas, double *env)
{
        int nbas2 = nbas * nbas;
        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel \
        shared(intor, cintopt, ao_loc, atm, natm, bas, nbas, env)
{
        double qtmp;
        int i, j, iijj, di, dj, ish, jsh;
        size_t ij;
        int shls[4];
        double *cache = malloc(sizeof(double) * cache_size);
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *buf = malloc(sizeof(double) * 256 * di*di*di*di);
        double *bufxx = buf;
        double *bufxy, *bufxz, *bufyx, *bufyy, *bufyz, *bufzx, *bufzy, *bufzz;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas2; ij++) {
                ish = ij / nbas;
                jsh = ij - ish * nbas;
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ish;
                shls[3] = jsh;
                qtmp = 1e-100;
                iijj = di * dj * di * dj;
                bufxy = buf + ( 1*16+ 1)*iijj;
                bufxz = buf + ( 2*16+ 2)*iijj;
                bufyx = buf + ( 4*16+ 4)*iijj;
                bufyy = buf + ( 5*16+ 5)*iijj;
                bufyz = buf + ( 6*16+ 6)*iijj;
                bufzx = buf + ( 8*16+ 8)*iijj;
                bufzy = buf + ( 9*16+ 9)*iijj;
                bufzz = buf + (10*16+10)*iijj;
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                  cintopt, cache)) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                iijj = i+di*j+di*dj*i+di*dj*di*j;
                                qtmp = MAX(qtmp, fabs(bufxx[iijj]));
                                qtmp = MAX(qtmp, fabs(bufxy[iijj]));
                                qtmp = MAX(qtmp, fabs(bufxz[iijj]));
                                qtmp = MAX(qtmp, fabs(bufyx[iijj]));
                                qtmp = MAX(qtmp, fabs(bufyy[iijj]));
                                qtmp = MAX(qtmp, fabs(bufyz[iijj]));
                                qtmp = MAX(qtmp, fabs(bufzx[iijj]));
                                qtmp = MAX(qtmp, fabs(bufzy[iijj]));
                                qtmp = MAX(qtmp, fabs(bufzz[iijj]));
                        } }
                        qtmp = sqrt(qtmp);
                }
                q_cond[ish*nbas+jsh] = qtmp;
        }
        free(buf);
        free(cache);
}
}

void CVHFipip1_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                          int *ao_loc, int *atm, int natm,
                          int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        nbas = opt->nbas;
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        // First n*n elements for derivatives, the next n*n elements for regular ERIs
        opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas*2);

        if (ao_loc[nbas] == CINTtot_cgto_spheric(bas, nbas)) {
                CVHFnr_int2e_q_cond(int2e_sph, NULL, opt->q_cond+Nbas2, ao_loc,
                                     atm, natm, bas, nbas, env);
        } else {
                CVHFnr_int2e_q_cond(int2e_cart, NULL, opt->q_cond+Nbas2, ao_loc,
                                     atm, natm, bas, nbas, env);
        }
        CVHFnr_int2e_pppp_q_cond(intor, cintopt, opt->q_cond, ao_loc,
                                 atm, natm, bas, nbas, env);
}

void CVHFipip1_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        CVHFgrad_jk_direct_scf_dm(opt, dm, nset, ao_loc, atm, natm, bas, nbas, env);
}


// ijkl,lk->ij
// ijkl,li->kj
// ijkl,kl->ij
// ijkl,ki->lj
int CVHFipvip1_prescreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
        double *q_cond_kl = opt->q_cond + n * n;
        double qijkl = opt->q_cond[i*n+j] * q_cond_kl[k*n+l];
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((2*opt->dm_cond[l*n+k] > dmin)
            || (  opt->dm_cond[l*n+i] > dmin)
            || (  opt->dm_cond[k*n+i] > dmin));
}


void CVHFipvip1_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                           int *ao_loc, int *atm, int natm,
                           int *bas, int nbas, double *env)
{
        CVHFipip1_direct_scf(opt, intor, cintopt, ao_loc, atm, natm, bas, nbas, env);
}

void CVHFipvip1_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                              int *atm, int natm, int *bas, int nbas, double *env)
{
        CVHFgrad_jk_direct_scf_dm(opt, dm, nset, ao_loc, atm, natm, bas, nbas, env);
}

