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
#include "gto/gto.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))

typedef struct {
        int ncomp;
        int v_dims[3];
        double *data;
} SGXJKArray;

typedef struct {
        SGXJKArray *(*allocate)(int *shls_slice, int *ao_loc, int ncomp, int ngrids);
        //void (*contract)(double *eri, double *dm, SGXJKArray *vjk,
        //                 int i0, int i1, int j0, int j1);
        void (*contract)(double *eri, double *dm, SGXJKArray *vjk,
                             int i0, int i1, int j0, int j1,
                             int* inds, int ngrids);
        void (*set0)(SGXJKArray *, int);
        void (*send)(SGXJKArray *, int, double *);
        void (*finalize)(SGXJKArray *, double *);
        void (*sanity_check)(int *shls_slice);
} SGXJKOperator;

#define BLKSIZE         312

// for grids integrals only
size_t _max_cache_size_sgx(int (*intor)(), int *shls_slice, int ncenter,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int i;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        size_t (*f)() = (size_t (*)())intor;
        size_t cache_size = 0;
        size_t n;
        int shls[4];
        for (i = i0; i < i1; i++) {
                shls[0] = i;
                shls[1] = i;
                shls[2] = 0;
                shls[3] = BLKSIZE;
                n = (*f)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

#define DECLARE_ALL \
        int *atm = envs->atm; \
        int *bas = envs->bas; \
        double *env = envs->env; \
        int natm = envs->natm; \
        int nbas = envs->nbas; \
        int *ao_loc = envs->ao_loc; \
        int *shls_slice = envs->shls_slice; \
        CINTOpt *cintopt = envs->cintopt; \
        int ioff = ao_loc[shls_slice[0]]; \
        int joff = ao_loc[shls_slice[2]]; \
        int i0, j0, i1, j1, ish, jsh, idm; \
        ish = shls[0]; \
        jsh = shls[1];

int SGXnr_pj_prescreen(int *shls, CVHFOpt *opt,
                       int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int n = opt->nbas;
        int nk = opt->ngrids;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < nk);
        return opt->q_cond[i*n+j]
               * MAX(fabs(opt->dm_cond[j*nk+k]), fabs(opt->dm_cond[i*nk+k]))
               > opt->direct_scf_cutoff;
}

void SGXdot_nrk(int (*intor)(), SGXJKOperator **jkop, SGXJKArray **vjk,
                double **dms, double *buf, double *cache, int n_dm, int* shls,
                CVHFOpt *vhfopt, IntorEnvs *envs,
                double* all_grids, int tot_grids)
{
        DECLARE_ALL;

        i0 = ao_loc[ish  ] - ioff;
        j0 = ao_loc[jsh  ] - joff;
        i1 = ao_loc[ish+1] - ioff;
        j1 = ao_loc[jsh+1] - joff;

        int tmp_ngrids = 0;
        int k;
        int* inds = (int*) malloc(tot_grids*sizeof(int));

        double *grids = env + (size_t) env[PTR_GRIDS];
        
        if (vhfopt != NULL && vhfopt->dm_cond != NULL) {
                for (k = 0; k < tot_grids; k++) {
                        shls[2] = k;
                        if (SGXnr_pj_prescreen(shls, vhfopt, atm, bas, env)) {
                                grids[3*tmp_ngrids+0] = all_grids[3*k+0];
                                grids[3*tmp_ngrids+1] = all_grids[3*k+1];
                                grids[3*tmp_ngrids+2] = all_grids[3*k+2];
                                inds[tmp_ngrids] = k;
                                tmp_ngrids++;
                        }
                }
                env[NGRIDS] = tmp_ngrids;
        } else {
                for (k = 0; k < tot_grids; k++) {
                        shls[2] = k;
                        grids[3*tmp_ngrids+0] = all_grids[3*k+0];
                        grids[3*tmp_ngrids+1] = all_grids[3*k+1];
                        grids[3*tmp_ngrids+2] = all_grids[3*k+2];
                        inds[tmp_ngrids] = k;
                        tmp_ngrids++;
                }
                env[NGRIDS] = tmp_ngrids;
        }

        int grid0, grid1;
        const int dims[] = {ao_loc[ish+1]-ao_loc[ish], ao_loc[jsh+1]-ao_loc[jsh], tmp_ngrids};
        for (grid0 = 0; grid0 < tmp_ngrids; grid0 += BLKSIZE) {
                grid1 = MIN(grid0 + BLKSIZE, tmp_ngrids);
                shls[2] = grid0;
                shls[3] = grid1;
                (*intor)(buf+grid0, dims, shls, atm, natm, bas, nbas, env, cintopt, cache);
        }
        //(*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache);

        for (idm = 0; idm < n_dm; idm++) {
                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                    i0, i1, j0, j1, inds, tmp_ngrids);
        }
        
        free(inds);
}


void SGXnr_direct_drv(int (*intor)(), void (*fdot)(), SGXJKOperator **jkop,
                        double **dms, double **vjk, int n_dm, int ncomp,
                        int *shls_slice, int *ao_loc,
                        CINTOpt *cintopt, CVHFOpt *vhfopt,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        int env_size, int aosym)
{

        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        int nish = ish1 - ish0;
        int di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        int cache_size = _max_cache_size_sgx(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);
        int npair;
        if (aosym == 2) {
            npair = nish * (nish+1) / 2;
        } else {
            npair = nish * nish;
        }

        int (*fprescreen)();
        if (vhfopt != NULL) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }
        int ngrids = (int) env[NGRIDS];
        double* all_grids = env+(size_t)env[PTR_GRIDS];

#pragma omp parallel default(none) firstprivate(ish0, jsh0) \
        shared(intor, fdot, jkop, ao_loc, shls_slice, \
               dms, vjk, n_dm, ncomp, nbas, vhfopt, \
               atm, bas, env, natm, \
               nish, di, cache_size, fprescreen, \
               aosym, npair, cintopt, env_size, \
               ngrids, all_grids)
{
        int i, ij, ish, jsh;
        int shls[4];
        double* tmp_env = (double*) malloc(env_size * sizeof(double));
        for (i = 0; i < env_size; i++) {
            tmp_env[i] = env[i];
        }
        IntorEnvs envs = {natm, nbas, atm, bas, tmp_env, shls_slice, ao_loc, NULL,
                          cintopt, ncomp};
        SGXJKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = jkop[i]->allocate(shls_slice, ao_loc, ncomp, ngrids);
        }
        double *buf = malloc(sizeof(double) * ngrids*di*di*ncomp);
        double *cache = malloc(sizeof(double) * cache_size);
#pragma omp for nowait schedule(dynamic, 1)
        for (ij = 0; ij < npair; ij++) {
                if (aosym == 2) {
                    ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                    jsh = ij - ish*(ish+1)/2;
                } else {
                    ish = ij / nish;
                    jsh = ij % nish;
                }
                shls[0] = ish + ish0;
                shls[1] = jsh + jsh0;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env))
                {
                    (*fdot)(intor, jkop, v_priv, dms, buf, cache, n_dm, shls,
                            vhfopt, &envs, all_grids, ngrids);
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
        free(tmp_env);
}
}

void SGXnr_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                  int *ao_loc, int *atm, int natm,
                  int *bas, int nbas, double *env)
{
        int shls_slice[] = {0, nbas};
        int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                           atm, natm, bas, nbas, env);
#pragma omp parallel default(none) \
        shared(intor, q_cond, ao_loc, atm, natm, bas, nbas, env, cache_size)
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

void SGXsetnr_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                         int *ao_loc, int *atm, int natm,
                         int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        nbas = opt->nbas;
        double *q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        opt->q_cond = q_cond;
        SGXnr_q_cond(intor, cintopt, q_cond, ao_loc, atm, natm, bas, nbas, env);
}

void SGXnr_dm_cond(double *dm_cond, double *dm, int nset, int *ao_loc,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int ngrids)
{
        size_t nao = ao_loc[nbas] - ao_loc[0];
        double dmax;
        size_t i, j, jsh, iset;
        double *pdm;
        for (i = 0; i < ngrids; i++) {
        for (jsh = 0; jsh < nbas; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        pdm = dm + nao*ngrids*iset;
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dmax = MAX(dmax, fabs(pdm[i*nao+j]));
                        }
                }
                dm_cond[jsh*ngrids+i] = dmax;
        } }
}

void SGXsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                            int *atm, int natm, int *bas, int nbas, double *env,
                            int ngrids)
{
        nbas = opt->nbas;
        if (opt->dm_cond != NULL) {
                free(opt->dm_cond);
        }
        opt->dm_cond = (double *)malloc(sizeof(double) * nbas*ngrids);
        if (opt->dm_cond == NULL) {
                fprintf(stderr, "malloc(%zu) failed in SGXsetnr_direct_scf_dm\n",
                        sizeof(double) * nbas*ngrids);
                exit(1);
        }
        // nbas in the input arguments may different to opt->nbas.
        // Use opt->nbas because it is used in the prescreen function
        memset(opt->dm_cond, 0, sizeof(double)*nbas*ngrids);
        opt->ngrids = ngrids;

        SGXnr_dm_cond(opt->dm_cond, dm, nset, ao_loc,
                      atm, natm, bas, nbas, env, ngrids);
}

int SGXnr_ovlp_prescreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
static SGXJKArray *SGXJKOperator_allocate_##label(int *shls_slice, int *ao_loc, \
                                                  int ncomp, int ngrids) \
{ \
        SGXJKArray *jkarray = malloc(sizeof(SGXJKArray)); \
        jkarray->v_dims[0]  = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]; \
        jkarray->v_dims[1]  = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]; \
        jkarray->v_dims[2]  = ngrids; \
        if (task == JTYPE1) { \
                jkarray->data = calloc(ncomp * jkarray->v_dims[2], sizeof(double)); \
        } else if (task == JTYPE2) { \
                jkarray->data = calloc(ncomp * jkarray->v_dims[0] \
                                       * jkarray->v_dims[1], sizeof(double)); \
        } else { \
                jkarray->data = calloc(ncomp * jkarray->v_dims[0] \
                                       * jkarray->v_dims[2], sizeof(double)); \
        } \
        jkarray->ncomp = ncomp; \
        return jkarray; \
} \
static void SGXJKOperator_set0_##label(SGXJKArray *jkarray, int k) \
{ } \
static void SGXJKOperator_send_##label(SGXJKArray *jkarray, int k, double *out) \
{ } \
static void SGXJKOperator_final_##label(SGXJKArray *jkarray, double *out) \
{ \
        int i, k, icomp; \
        int ni = jkarray->v_dims[0]; \
        double *data = jkarray->data; \
        int ngrids = jkarray->v_dims[2]; \
        if (task == JTYPE1) { \
                for (i = 0; i < jkarray->ncomp; i++) { \
                for (k = 0; k < ngrids; k++) { \
                        out[i*ngrids+k] += data[i*ngrids+k]; \
                } } \
        } else if (task == JTYPE2) { \
                for (i = 0; i < jkarray->ncomp * jkarray->v_dims[0] * jkarray->v_dims[1]; i++) { \
                        out[i] += data[i]; \
                } \
        } else { \
                for (icomp = 0; icomp < jkarray->ncomp; icomp++) { \
                        for (i = 0; i < ni; i++) { \
                        for (k = 0; k < ngrids; k++) { \
                                out[i*ngrids+k] += data[i*ngrids+k]; \
                        } } \
                        out += ngrids * ni; \
                        data += ngrids * ni; \
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
                          int i0, int i1, int j0, int j1,
                           int* inds, int pngrids)
{
        const int ncol = out->v_dims[0];
        int i, j, k, icomp;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < pngrids; k++) {
                        data[inds[k]] += eri[ij*pngrids+k] * dm[j*ncol+i];
                } } }
                data += out->v_dims[2];
        }
}
ADD_OP(nrs1_ijg_ji_g, JTYPE1, s1);

static void nrs2_ijg_ji_g(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1,
                           int* inds, int pngrids)
{
        if (i0 == j0) {
                return nrs1_ijg_ji_g(eri, dm, out, i0, i1, j0, j1, inds, pngrids);
        }

        const int ncol = out->v_dims[0];
        int i, j, k, icomp;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < pngrids; k++) {
                        data[inds[k]] += eri[ij*pngrids+k] * (dm[j*ncol+i] + dm[i*ncol+j]);
                } } }
                data += out->v_dims[2];
        }
}
ADD_OP(nrs2_ijg_ji_g, JTYPE1, s2);

static void nrs1_ijg_g_ij(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1,
                           int* inds, int pngrids)
{
        int ni = out->v_dims[0];
        int nj = out->v_dims[1];
        int i, j, k, icomp;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < pngrids; k++) {
                        data[i*nj+j] += eri[ij*pngrids+k] * dm[inds[k]];
                } } }
                data += ni * nj;
        }
}
ADD_OP(nrs1_ijg_g_ij, JTYPE2, s1);

SGXJKOperator SGXnrs2_ijg_g_ij = {SGXJKOperator_allocate_nrs1_ijg_g_ij,
        nrs1_ijg_g_ij, SGXJKOperator_set0_nrs1_ijg_g_ij,
        SGXJKOperator_send_nrs1_ijg_g_ij, SGXJKOperator_final_nrs1_ijg_g_ij,
        SGXJKOperator_sanity_check_s2};

static void nrs1_ijg_gj_gi(double *eri, double *dm, SGXJKArray *out,
                           int i0, int i1, int j0, int j1,
                           int* inds, int pngrids)
{
        double *data = out->data;
        int i, j, k, icomp;
        const int ngrids = out->v_dims[2];

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < pngrids; k++) {
                        data[i*ngrids+inds[k]] += eri[ij*pngrids+k] * dm[j*ngrids+inds[k]];
                } } }
                data += out->v_dims[0] * out->v_dims[2];
        }
}
ADD_OP(nrs1_ijg_gj_gi, KTYPE1, s1);

static void nrs2_ijg_gj_gi(double *eri, double *dm, SGXJKArray *out,
                           int i0, int i1, int j0, int j1,
                           int* inds, int pngrids)
{
        if (i0 == j0) {
                return nrs1_ijg_gj_gi(eri, dm, out, i0, i1, j0, j1, inds, pngrids);
        }

        double *data = out->data;
        const int ngrids = out->v_dims[2];
        int i, j, k, icomp;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < pngrids; k++) {
                        data[i*ngrids+inds[k]] += eri[ij*pngrids+k] * dm[j*ngrids+inds[k]];
                }
                for (k = 0; k < pngrids; k++) {
                        data[j*ngrids+inds[k]] += eri[ij*pngrids+k] * dm[i*ngrids+inds[k]];
                }
                } }
                data += out->v_dims[0] * out->v_dims[2];
        }
}
ADD_OP(nrs2_ijg_gj_gi, KTYPE1, s2);
