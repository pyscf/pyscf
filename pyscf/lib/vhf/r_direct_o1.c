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
#include <assert.h>
#include <math.h>
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "nr_direct.h"
#include "time_rev.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))

#define DECLARE_ALL \
        const int *atm = envs->atm; \
        const int *bas = envs->bas; \
        const double *env = envs->env; \
        const int natm = envs->natm; \
        const int nbas = envs->nbas; \
        const int *ao_loc = envs->ao_loc; \
        const int *shls_slice = envs->shls_slice; \
        const int *tao = envs->tao; \
        const CINTOpt *cintopt = envs->cintopt; \
        const int nao = ao_loc[nbas]; \
        const int di = ao_loc[ish+1] - ao_loc[ish]; \
        const int dj = ao_loc[jsh+1] - ao_loc[jsh]; \
        const int dim = GTOmax_shell_dim(ao_loc, shls_slice+4, 2); \
        double *cache = (double *)(buf + di * dj * dim * dim * ncomp); \
        int (*fprescreen)(); \
        int (*r_vkscreen)(); \
        if (vhfopt != NULL) { \
                fprescreen = vhfopt->fprescreen; \
                r_vkscreen = vhfopt->r_vkscreen; \
        } else { \
                fprescreen = CVHFnoscreen; \
                r_vkscreen = CVHFr_vknoscreen; \
        }

static void transpose01324(double complex * __restrict__ a,
                           double complex * __restrict__ at,
                           int di, int dj, int dk, int dl, int ncomp)
{
        int i, j, k, l, m, ic;
        int dij = di * dj;
        int dijk = dij * dk;
        double complex *pa;

        m = 0;
        for (ic = 0; ic < ncomp; ic++) {
                for (l = 0; l < dl; l++) {
                        for (j = 0; j < dj; j++) {
                                pa = a + j*di;
                                for (k = 0; k < dk; k++) {
                                        for (i = 0; i < di; i++) {
                                                at[m] = pa[i];
                                                m++;
                                        }
                                        pa += dij;
                                }
                        }
                        a += dijk;
                }
        }
}
/*
 * for given ksh, lsh, loop all ish, jsh
 */
void CVHFdot_rs1(int (*intor)(), void (**fjk)(),
                 double complex **dms, double complex *vjk, double complex *buf,
                 int n_dm, int ncomp, int ish, int jsh,
                 CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        const size_t nao2 = nao * nao;
        int idm, ksh, lsh, dk, dl, dijkl;
        int shls[4];
        double complex *pv;
        double *dms_cond[n_dm+1];
        double dm_atleast;
        void (*pf)();

// to make fjk compatible to C-contiguous dm array, put ksh, lsh inner loop
        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < nbas; ksh++) {
        for (lsh = 0; lsh < nbas; lsh++) {
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)) {
// append buf.transpose(0,2,1,3) to eris, to reduce the cost of r_direct_dot
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                     cintopt, cache)) {
                                dijkl = di * dj * dk * dl;
                                if ((*r_vkscreen)(shls, vhfopt, dms_cond, n_dm,
                                                  &dm_atleast, atm, bas, env)) {
                                        transpose01324(buf, buf+dijkl*ncomp,
                                                       di, dj, dk, dl, ncomp);
                                }
                                pv = vjk;
                                for (idm = 0; idm < n_dm; idm++) {
                                        pf = fjk[idm];
                                        (*pf)(buf, dms[idm], pv, nao, ncomp,
                                              shls, ao_loc, tao,
                                              dms_cond[idm], nbas, dm_atleast);
                                        pv += nao2 * ncomp;
                                }
                        }
                }
        } }
}

/*
 * for given ish, jsh, loop all ksh > lsh
 */
static void dot_rs2sub(int (*intor)(), void (**fjk)(),
                       double complex **dms, double complex *vjk, double complex *buf,
                       int n_dm, int ncomp, int ish, int jsh, int ksh_count,
                       CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        const size_t nao2 = nao * nao;
        int idm, ksh, lsh, dk, dl, dijkl;
        int shls[4];
        double complex *pv;
        double *dms_cond[n_dm+1];
        double dm_atleast;
        void (*pf)();

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < ksh_count; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)) {
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                     cintopt, cache)) {
                                dijkl = di * dj * dk * dl;
                                if ((*r_vkscreen)(shls, vhfopt, dms_cond, n_dm,
                                                  &dm_atleast, atm, bas, env)) {
                                        transpose01324(buf, buf+dijkl*ncomp,
                                                       di, dj, dk, dl, ncomp);
                                }
                                pv = vjk;
                                for (idm = 0; idm < n_dm; idm++) {
                                        pf = fjk[idm];
                                        (*pf)(buf, dms[idm], pv, nao, ncomp,
                                              shls, ao_loc, tao,
                                              dms_cond[idm], nbas, dm_atleast);
                                        pv += nao2 * ncomp;
                                }
                        }
                }
        } }
}

void CVHFdot_rs2ij(int (*intor)(), void (**fjk)(),
                   double complex **dms, double complex *vjk, double complex *buf,
                   int n_dm, int ncomp, int ish, int jsh,
                   CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ish >= jsh) {
                CVHFdot_rs1(intor, fjk, dms, vjk, buf, n_dm, ncomp,
                            ish, jsh, vhfopt, envs);
        }
}

void CVHFdot_rs2kl(int (*intor)(), void (**fjk)(),
                   double complex **dms, double complex *vjk, double complex *buf,
                   int n_dm, int ncomp, int ish, int jsh,
                   CVHFOpt *vhfopt, IntorEnvs *envs)
{
        dot_rs2sub(intor, fjk, dms, vjk, buf, n_dm, ncomp,
                   ish, jsh, envs->nbas, vhfopt, envs);
}

void CVHFdot_rs4(int (*intor)(), void (**fjk)(),
                 double complex **dms, double complex *vjk, double complex *buf,
                 int n_dm, int ncomp, int ish, int jsh,
                 CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ish >= jsh) {
                dot_rs2sub(intor, fjk, dms, vjk, buf, n_dm, ncomp,
                           ish, jsh, envs->nbas, vhfopt, envs);
        }
}

void CVHFdot_rs8(int (*intor)(), void (**fjk)(),
                 double complex **dms, double complex *vjk, double complex *buf,
                 int n_dm, int ncomp, int ish, int jsh,
                 CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ish < jsh) {
                return;
        }
        DECLARE_ALL;
        const size_t nao2 = nao * nao;
        int idm, ksh, lsh, dk, dl, dijkl;
        int shls[4];
        double complex *pv;
        double *dms_cond[n_dm+1];
        double dm_atleast;
        void (*pf)();

// to make fjk compatible to C-contiguous dm array, put ksh, lsh inner loop
        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh <= ish; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
/* when ksh==ish, (lsh<jsh) misses some integrals (eg k<i&&l>j).
 * These integrals are calculated in the next (ish,jsh) pair. To show
 * that, we just need to prove that every elements in shell^4 appeared
 * only once in fjk_s8.  */
                if ((ksh == ish) && (lsh > jsh)) {
                        break;
                }
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)) {
                        if ((*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                     cintopt, cache)) {
                                dijkl = di * dj * dk * dl;
                                if ((*r_vkscreen)(shls, vhfopt, dms_cond, n_dm,
                                                  &dm_atleast, atm, bas, env)) {
                                        transpose01324(buf, buf+dijkl*ncomp,
                                                       di, dj, dk, dl, ncomp);
                                }
                                pv = vjk;
                                for (idm = 0; idm < n_dm; idm++) {
                                        pf = fjk[idm];
                                        (*pf)(buf, dms[idm], pv, nao, ncomp,
                                              shls, ao_loc, tao,
                                              dms_cond[idm], nbas, dm_atleast);
                                        pv += nao2 * ncomp;
                                }
                        }
                }
        } }
}

/*
 * drv loop over ij, generate eris of kl for given ij, call fjk to
 * calculate vj, vk.
 * 
 * n_dm is the number of dms for one [array(ij|kl)],
 * ncomp is the number of components that produced by intor
 */
void CVHFr_direct_drv(int (*intor)(), void (*fdot)(), void (**fjk)(),
                      double complex **dms, double complex *vjk,
                      int n_dm, int ncomp, int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, CVHFOpt *vhfopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        int *tao = malloc(sizeof(int)*nao);
        CVHFtimerev_map(tao, bas, nbas);
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, tao,
                cintopt, ncomp};

        const size_t nbas2 = ((size_t)nbas) * nbas;
        const size_t jk_size = nao * nao * n_dm * ncomp;
        NPzset0(vjk, jk_size);

        const size_t di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        size_t i, j, ij;
        double complex *v_priv = malloc(sizeof(double complex) * jk_size);
        NPzset0(v_priv, jk_size);
        size_t bufsize = di*di*di*di*ncomp;
        bufsize = bufsize + di*di*8 + MAX(bufsize, (cache_size+1)/2);  // /2 for double complex
        double complex *buf = malloc(sizeof(double complex) * bufsize);
#pragma omp for nowait schedule(dynamic)
        for (ij = 0; ij < nbas2; ij++) {
                i = ij / nbas;
                j = ij - i * nbas;
                (*fdot)(intor, fjk, dms, v_priv, buf, n_dm, ncomp, i, j,
                        vhfopt, &envs);
        }
#pragma omp critical
        {
                for (i = 0; i < jk_size; i++) {
                        vjk[i] += v_priv[i];
                }
        }
        free(v_priv);
        free(buf);
}
        free(tao);
}

