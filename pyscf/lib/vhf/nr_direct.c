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
#include "optimizer.h"
#include "nr_direct.h"

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
        const int ioff = ao_loc[shls_slice[0]]; \
        const int joff = ao_loc[shls_slice[2]]; \
        const int koff = ao_loc[shls_slice[4]]; \
        const int loff = ao_loc[shls_slice[6]]; \
        const int i0 = ao_loc[ish] - ioff; \
        const int j0 = ao_loc[jsh] - joff; \
        const int i1 = ao_loc[ish+1] - ioff; \
        const int j1 = ao_loc[jsh+1] - joff; \
        const int di = i1 - i0; \
        const int dj = j1 - j0; \
        const int ncomp = envs->ncomp; \
        const int dk = GTOmax_shell_dim(ao_loc, shls_slice+4, 2); \
        double *cache = buf + di * dj * dk * dk * ncomp; \
        int shls[4]; \
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls, \
                   int i0, int i1, int j0, int j1, \
                   int k0, int k1, int l0, int l1); \
        int (*fprescreen)(); \
        if (vhfopt) { \
                fprescreen = vhfopt->fprescreen; \
        } else { \
                fprescreen = CVHFnoscreen; \
        } \
        int ksh, lsh, k0, k1, l0, l1, idm;

#define INTOR_AND_CONTRACT \
        if ((*fprescreen)(shls, vhfopt, atm, bas, env) \
            && (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, \
                        cintopt, cache)) { \
                k0 = ao_loc[ksh] - koff; \
                l0 = ao_loc[lsh] - loff; \
                k1 = ao_loc[ksh+1] - koff; \
                l1 = ao_loc[lsh+1] - loff; \
                for (idm = 0; idm < n_dm; idm++) { \
                        pf = jkop[idm]->contract; \
                        (*pf)(buf, dms[idm], vjk[idm], shls, \
                              i0, i1, j0, j1, k0, k1, l0, l1); \
                } \
        }

/*
 * for given ksh, lsh, loop all ish, jsh
 */
void CVHFdot_nrs1(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, int n_dm, int ish, int jsh,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int lsh0 = shls_slice[6];
        const int lsh1 = shls_slice[7];

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                INTOR_AND_CONTRACT;
        } }
}

/*
 * for given ish, jsh, loop all ksh > lsh
 */
static void dot_nrs2sub(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                        double **dms, double *buf, int n_dm, int ish, int jsh,
                        CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int lsh0 = shls_slice[6];

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh <= ksh; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                INTOR_AND_CONTRACT;
        } }
}

void CVHFdot_nrs2ij(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                    double **dms, double *buf, int n_dm, int ish, int jsh,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ish >= jsh) {
                CVHFdot_nrs1(intor, jkop, vjk, dms, buf, n_dm, ish, jsh, vhfopt, envs);
        }
}

void CVHFdot_nrs2kl(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                    double **dms, double *buf, int n_dm, int ish, int jsh,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        dot_nrs2sub(intor, jkop, vjk, dms, buf, n_dm, ish, jsh, vhfopt, envs);
}

void CVHFdot_nrs4(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, int n_dm, int ish, int jsh,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ish >= jsh) {
                dot_nrs2sub(intor, jkop, vjk, dms, buf, n_dm, ish, jsh, vhfopt, envs);
        }
}

void CVHFdot_nrs8(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, int n_dm, int ish, int jsh,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ish < jsh) {
                return;
        }
        DECLARE_ALL;
        const int ksh0 = shls_slice[4];
        const int lsh0 = shls_slice[6];

// to make fjk compatible to C-contiguous dm array, put ksh, lsh inner loop
        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh <= ish; ksh++) {
        for (lsh = lsh0; lsh <= ksh; lsh++) {
/* when ksh==ish, (lsh<jsh) misses some integrals (eg k<i&&l>j).
 * These integrals are calculated in the next (ish,jsh) pair. To show
 * that, we just need to prove that every elements in shell^4 appeared
 * only once in fjk_s8.  */
                if ((ksh == ish) && (lsh > jsh)) {
                        break;
                }
                shls[2] = ksh;
                shls[3] = lsh;
                INTOR_AND_CONTRACT;
        } }
}

static void assemble_v(double *vjk, JKArray *jkarray, int *ao_loc)
{
        int ish0 = jkarray->v_bra_sh0;
        int ish1 = jkarray->v_bra_sh1;
        int jsh0 = jkarray->v_ket_sh0;
        int jsh1 = jkarray->v_ket_sh1;
        int njsh = jsh1 - jsh0;
        size_t vrow = jkarray->v_dims[0];
        size_t vcol = jkarray->v_dims[1];
        int ncomp = jkarray->ncomp;
        int voffset = ao_loc[ish0] * vcol + ao_loc[jsh0];
        int i, j, ish, jsh;
        int di, dj, icomp;
        int optr;
        double *data, *pv;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                optr = jkarray->outptr[ish*njsh+jsh-jkarray->offset0_outptr];
                if (optr != NOVALUE) {
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        data = jkarray->data + optr;
                        pv = vjk + ao_loc[ish]*vcol+ao_loc[jsh] - voffset;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        pv[i*vcol+j] += data[i*dj+j];
                                } }
                                pv += vrow * vcol;
                                data += di * dj;
                        }
                }
        } }
}


/*
 * drv loop over ij, generate eris of kl for given ij, call fjk to
 * calculate vj, vk.
 * 
 * n_dm is the number of dms for one [array(ij|kl)], it is also the size of dms and vjk
 * ncomp is the number of components that produced by intor
 * shls_slice = [ishstart, ishend, jshstart, jshend, kshstart, kshend, lshstart, lshend]
 *
 * ao_loc[i+1] = ao_loc[i] + CINTcgto_spheric(i, bas)  for i = 0..nbas
 *
 * Return [(ptr[ncomp,nao,nao] in C-contiguous) for ptr in vjk]
 */
void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                       double **dms, double **vjk, int n_dm, int ncomp,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};
        int idm;
        size_t size;
        for (idm = 0; idm < n_dm; idm++) {
                size = jkop[idm]->data_size(shls_slice, ao_loc) * ncomp;
                memset(vjk[idm], 0, sizeof(double)*size);
        }

        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int i, j, ij, ij1;
        JKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = jkop[i]->allocate(shls_slice, ao_loc, ncomp);
        }
        double *buf = malloc(sizeof(double) * (di*di*di*di*ncomp + cache_size));
#pragma omp for nowait schedule(dynamic, 1)
        for (ij = 0; ij < nish*njsh; ij++) {
                ij1 = nish*njsh-1 - ij;

//                        if (ij % 2) {
///* interlace the iteration to balance memory usage
// * map [0,1,2...,N] to [0,N,1,N-1,...] */
//                                ij1 = nish*njsh-1 - ij/2;
//                        } else {
//                                ij1 = ij / 2;
//                        }

                i = ij1 / njsh + ish0;
                j = ij1 % njsh + jsh0;
                (*fdot)(intor, jkop, v_priv, dms, buf, n_dm, i, j,
                        vhfopt, &envs);
        }
#pragma omp critical
        {
                for (i = 0; i < n_dm; i++) {
                        assemble_v(vjk[i], v_priv[i], ao_loc);
                        jkop[i]->deallocate(v_priv[i]);
                }
        }
        free(buf);
}
}

