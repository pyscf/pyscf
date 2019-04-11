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
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "nr_direct.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

typedef struct {
        int ncomp;
        int v_dims[3];
        double *data;
} SGXJKArray;

typedef struct {
        SGXJKArray *(*allocate)(int *shls_slice, int *ao_loc, int ncomp);
        void (*contract)(double *eri, double *dm, SGXJKArray *vjk,
                         int i0, int i1, int j0, int j1, int k0);
        void (*set0)(SGXJKArray *, int);
        void (*send)(SGXJKArray *, int, double *);
        void (*finalize)(SGXJKArray *, double *);
        void (*sanity_check)(int *shls_slice);
} SGXJKOperator;

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

#define DECLARE_ALL \
        const int *atm = envs->atm; \
        const int *bas = envs->bas; \
        const double *env = envs->env; \
        const int natm = envs->natm; \
        const int nbas = envs->nbas; \
        const int *ao_loc = envs->ao_loc; \
        const int *shls_slice = envs->shls_slice; \
        const CINTOpt *cintopt = envs->cintopt; \
        const int ish0 = shls_slice[0]; \
        const int ish1 = shls_slice[1]; \
        const int jsh0 = shls_slice[2]; \
        const int jsh1 = shls_slice[3]; \
        const int ksh0 = shls_slice[4]; \
        const int ioff = ao_loc[ish0]; \
        const int joff = ao_loc[jsh0]; \
        int i0, j0, i1, j1, ish, jsh, idm; \
        int shls[3]; \
        int (*fprescreen)(); \
        if (vhfopt) { \
                fprescreen = vhfopt->fprescreen; \
        } else { \
                fprescreen = CVHFnoscreen; \
        } \

/*
 * for given ksh, lsh, loop all ish, jsh
 */
void SGXdot_nrs1(int (*intor)(), SGXJKOperator **jkop, SGXJKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm, int ksh,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;

        shls[2] = ksh0 + ksh;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)
                    && (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                cintopt, cache)) {
                        i0 = ao_loc[ish  ] - ioff;
                        j0 = ao_loc[jsh  ] - joff;
                        i1 = ao_loc[ish+1] - ioff;
                        j1 = ao_loc[jsh+1] - joff;
                        for (idm = 0; idm < n_dm; idm++) {
                                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                                    i0, i1, j0, j1, ksh);
                        }
                }
        } }
}

/*
 * ish >= jsh
 */
void SGXdot_nrs2(int (*intor)(), SGXJKOperator **jkop, SGXJKArray **vjk,
                 double **dms, double *buf, double *cache, int n_dm, int ksh,
                 CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;

        shls[2] = ksh0 + ksh;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh <= ish; jsh++) {
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)
                    && (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                cintopt, cache)) {
                        i0 = ao_loc[ish  ] - ioff;
                        j0 = ao_loc[jsh  ] - joff;
                        i1 = ao_loc[ish+1] - ioff;
                        j1 = ao_loc[jsh+1] - joff;
                        for (idm = 0; idm < n_dm; idm++) {
                                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                                    i0, i1, j0, j1, ksh);
                        }
                }
        } }
}


void SGXnr_direct_drv(int (*intor)(), void (*fdot)(), SGXJKOperator **jkop,
                      double **dms, double **vjk, int n_dm, int ncomp,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, CVHFOpt *vhfopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};

        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int nksh = ksh1 - ksh0;
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel default(none) \
        shared(intor, fdot, jkop, ao_loc, shls_slice, \
               dms, vjk, n_dm, ncomp, nbas, vhfopt, envs)
{
        int i, ksh;
        SGXJKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = jkop[i]->allocate(shls_slice, ao_loc, ncomp);
        }
        double *buf = malloc(sizeof(double) * di*di*ncomp);
        double *cache = malloc(sizeof(double) * cache_size);
#pragma omp for nowait schedule(dynamic, 1)
        for (ksh = 0; ksh < nksh; ksh++) {
                for (i = 0; i < n_dm; i++) {
                        jkop[i]->set0(v_priv[i], ksh);
                }
                (*fdot)(intor, jkop, v_priv, dms, buf, cache, n_dm, ksh,
                        vhfopt, &envs);
                for (i = 0; i < n_dm; i++) {
                        jkop[i]->send(v_priv[i], ksh, vjk[i]);
                }
        }
#pragma omp critical
{
        for (i = 0; i < n_dm; i++) {
                jkop[i]->finalize(v_priv[i], vjk[i]);
        }
}
        free(buf);
        free(cache);
}
}


void SGXsetnr_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                         int *ao_loc, int *atm, int natm,
                         int *bas, int nbas, double *env)
{
        if (opt->q_cond) {
                free(opt->q_cond);
        }
        nbas = opt->nbas;
        double *q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        opt->q_cond = q_cond;

        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel default(none) \
        shared(intor, q_cond, ao_loc, atm, natm, bas, nbas, env)
{
        double qtmp, tmp;
        int ij, i, j, di, dj, ish, jsh;
        int shls[2];
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *cache = malloc(sizeof(double) * (di*di + cache_size));
        double *buf = cache + cache_size;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                jsh = ij - ish*(ish+1)/2;
                if (bas(ATOM_OF,ish) == bas(ATOM_OF,jsh)) {
                        // If two shells are on the same center, their
                        // overlap integrals may be zero due to symmetry.
                        // But their contributions to sgX integrals should
                        // be recognized.
                        q_cond[ish*nbas+jsh] = 1;
                        q_cond[jsh*nbas+ish] = 1;
                        continue;
                }

                shls[0] = ish;
                shls[1] = jsh;
                qtmp = 1e-100;
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                  NULL, cache)) {
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                tmp = fabs(buf[i+di*j]);
                                qtmp = MAX(qtmp, tmp);
                        } }
                }
                q_cond[ish*nbas+jsh] = qtmp;
                q_cond[jsh*nbas+ish] = qtmp;
        }
        free(cache);
}
}
int SGXnr_ovlp_prescreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(i < n);
        assert(j < n);
        return opt->q_cond[i*n+j] > opt->direct_scf_cutoff;
}


#define JTYPE1  1
#define JTYPE2  2
#define KTYPE1  3

#define ALLOCATE(label, task) \
        static SGXJKArray *SGXJKOperator_allocate_##label(int *shls_slice, int *ao_loc, int ncomp) \
{ \
        SGXJKArray *jkarray = malloc(sizeof(SGXJKArray)); \
        jkarray->v_dims[0]  = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]; \
        jkarray->v_dims[1]  = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]; \
        jkarray->v_dims[2]  = ao_loc[shls_slice[5]] - ao_loc[shls_slice[4]]; \
        if (task == JTYPE1) { \
                jkarray->data = malloc(sizeof(double) * ncomp); \
        } else if (task == JTYPE2) { \
                jkarray->data = calloc(ncomp * jkarray->v_dims[0] * jkarray->v_dims[1], sizeof(double)); \
        } else { \
                jkarray->data = malloc(sizeof(double) * ncomp * jkarray->v_dims[0]); \
        } \
        jkarray->ncomp = ncomp; \
        return jkarray; \
} \
static void SGXJKOperator_set0_##label(SGXJKArray *jkarray, int k) \
{ \
        int ncomp = jkarray->ncomp; \
        int i; \
        double *data = jkarray->data; \
        if (task == JTYPE1) { \
                for (i = 0; i < ncomp; i++) { \
                        data[i] = 0; \
                } \
        } else if (task == KTYPE1) { \
                for (i = 0; i < ncomp * jkarray->v_dims[0]; i++) { \
                        data[i] = 0; \
                } \
        } \
} \
static void SGXJKOperator_send_##label(SGXJKArray *jkarray, int k, double *out) \
{ \
        int ncomp = jkarray->ncomp; \
        int i, icomp; \
        double *data = jkarray->data; \
        int ni = jkarray->v_dims[0]; \
        int nk = jkarray->v_dims[2]; \
        if (task == JTYPE1) { \
                for (i = 0; i < ncomp; i++) { \
                        out[i*nk+k] = data[i]; \
                } \
        } else if (task == KTYPE1) { \
                for (icomp = 0; icomp < ncomp; icomp++) { \
                        for (i = 0; i < ni; i++) { \
                                out[k*ni+i] = data[i]; \
                        } \
                        out += nk * ni; \
                        data += ni; \
                } \
        } \
} \
static void SGXJKOperator_final_##label(SGXJKArray *jkarray, double *out) \
{ \
        int i; \
        double *data = jkarray->data; \
        if (task == JTYPE2) { \
                for (i = 0; i < jkarray->ncomp * jkarray->v_dims[0] * jkarray->v_dims[1]; i++) { \
                        out[i] += data[i]; \
                } \
        } \
        SGXJKOperator_deallocate(jkarray); \
}

#define ADD_OP(fname, task, type) \
        ALLOCATE(fname, task) \
SGXJKOperator SGX##fname = {SGXJKOperator_allocate_##fname, fname, \
        SGXJKOperator_set0_##fname, SGXJKOperator_send_##fname, \
        SGXJKOperator_final_##fname, \
        SGXJKOperator_sanity_check_##type}

static void SGXJKOperator_deallocate(SGXJKArray *jkarray)
{
        free(jkarray->data);
        free(jkarray);
}

static void SGXJKOperator_sanity_check_s1(int *shls_slice)
{
}
static void SGXJKOperator_sanity_check_s2(int *shls_slice)
{
        if (!((shls_slice[0] == shls_slice[2]) &&
              (shls_slice[1] == shls_slice[3]))) {
                fprintf(stderr, "Fail at s2\n");
                exit(1);
        };
}

static void nrs1_ijg_ji_g(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1, int k0)
{
        const int ncol = out->v_dims[0];
        int i, j, icomp;
        double g;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                g = 0;
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        g += eri[ij] * dm[j*ncol+i];
                } }
                data[icomp] += g;
        }
}
ADD_OP(nrs1_ijg_ji_g, JTYPE1, s1);

static void nrs2_ijg_ji_g(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1, int k0)
{
        if (i0 == j0) {
                return nrs1_ijg_ji_g(eri, dm, out, i0, i1, j0, j1, k0);
        }

        const int ncol = out->v_dims[0];
        int i, j, icomp;
        double g;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                g = 0;
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        g += eri[ij] * (dm[j*ncol+i] + dm[i*ncol+j]);
                } }
                data[icomp] += g;
        }
}
ADD_OP(nrs2_ijg_ji_g, JTYPE1, s2);

static void nrs1_ijg_g_ij(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1, int k0)
{
        int ni = out->v_dims[0];
        int nj = out->v_dims[1];
        int i, j, icomp;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        data[i*nj+j] += eri[ij] * dm[k0];
                } }
                data += ni * nj;
        }
}
ADD_OP(nrs1_ijg_g_ij, JTYPE2, s1);

SGXJKOperator SGXnrs2_ijg_g_ij = {SGXJKOperator_allocate_nrs1_ijg_g_ij,
        nrs1_ijg_g_ij, SGXJKOperator_set0_nrs1_ijg_g_ij,
        SGXJKOperator_send_nrs1_ijg_g_ij, SGXJKOperator_final_nrs1_ijg_g_ij,
        SGXJKOperator_sanity_check_s2};

static void nrs1_ijg_gj_gi(double *eri, double *dm, SGXJKArray *out,
                           int i0, int i1, int j0, int j1, int k0)
{
        const int ncol = out->v_dims[1];
        double *data = out->data;
        int i, j, icomp;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        data[i] += eri[ij] * dm[k0*ncol+j];
                } }
                data += out->v_dims[0];
        }
}
ADD_OP(nrs1_ijg_gj_gi, KTYPE1, s1);

static void nrs2_ijg_gj_gi(double *eri, double *dm, SGXJKArray *out,
                           int i0, int i1, int j0, int j1, int k0)
{
        if (i0 == j0) {
                return nrs1_ijg_gj_gi(eri, dm, out, i0, i1, j0, j1, k0);
        }

        const int ncol = out->v_dims[0];
        double *data = out->data;
        int i, j, icomp;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        data[i] += eri[ij] * dm[k0*ncol+j];
                        data[j] += eri[ij] * dm[k0*ncol+i];
                } }
                data += out->v_dims[0];
        }
}
ADD_OP(nrs2_ijg_gj_gi, KTYPE1, s2);
