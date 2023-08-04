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
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cvhf.h"
#include "nr_direct.h"
#include "optimizer.h"
#include "gto/gto.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

void int2e_optimizer(CINTOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
/*
 * 8-fold symmetry, k>=l, k>=i>=j, 
 */
static void fillnr_s8(int (*intor)(), int (*fprescreen)(), double *eri,
                      int ish, int jsh, CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int *atm = envs->atm;
        const int *bas = envs->bas;
        const double *env = envs->env;
        const int natm = envs->natm;
        const int nbas = envs->nbas;
        const int *ao_loc = envs->ao_loc;
        const CINTOpt *cintopt = envs->cintopt;
        const int nao = ao_loc[nbas];
        const size_t nao2 = nao * nao;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        double *cache = eri + di * dj * nao2;
        int dims[4] = {nao, nao, dj, di};
        int ksh, lsh, ij, k, l;
        int shls[4];
        double *peri;

        shls[2] = jsh;
        shls[3] = ish;

        for (ksh = 0; ksh <= ish; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                shls[0] = lsh;
                shls[1] = ksh;
                peri = eri + ao_loc[ksh] * nao + ao_loc[lsh];
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)) {
                        (*intor)(peri, dims, shls, atm, natm, bas, nbas, env,
                                 cintopt, cache);
                } else {
                        for (ij = 0; ij < di*dj; ij++) {
                                for (k = 0; k < ao_loc[ksh+1]-ao_loc[ksh]; k++) {
                                for (l = 0; l < ao_loc[lsh+1]-ao_loc[lsh]; l++) {
                                        peri[k*nao+l] = 0;
                                } }
                                peri += nao2;
                        }
                }
        } }
}

static void store_ij(int (*intor)(), double *eri, double *buf, int ish, int jsh,
                     CVHFOpt *vhfopt, IntorEnvs *envs)
{
        const int nbas = envs->nbas;
        const int *ao_loc = envs->ao_loc;
        const int nao = ao_loc[nbas];
        const size_t nao2 = nao * nao;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int i, j, k, l, i0, j0, kl;
        size_t ij0;
        double *peri, *pbuf;

        fillnr_s8(intor, vhfopt->fprescreen, buf, ish, jsh, vhfopt, envs);
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++) {
        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                if (i0 >= j0) {
                        ij0 = i0*(i0+1)/2 + j0;
                        peri = eri + ij0*(ij0+1)/2;
                        pbuf = buf + nao2 * (i*dj+j);
                        for (kl = 0, k = 0; k < i0; k++) {
                        for (l = 0; l <= k; l++, kl++) {
                                peri[kl] = pbuf[k*nao+l];
                        } }
                        // k == i0
                        for (l = 0; l <= j0; l++, kl++) {
                                peri[kl] = pbuf[k*nao+l];
                        }
                }
        } }
}

void GTO2e_cart_or_sph(int (*intor)(), CINTOpt *cintopt, double *eri, int *ao_loc,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        IntorEnvs envs = {natm, nbas, atm, bas, env, NULL, ao_loc, NULL,
                cintopt, 1};
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, intor, cintopt, ao_loc, atm, natm, bas, nbas, env);
        vhfopt->fprescreen = CVHFnr_schwarz_cond;
        int shls_slice[] = {0, nbas};
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 1);
        const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                    atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int i, j, ij;
        double *buf = malloc(sizeof(double) * (di*di*nao*nao + cache_size));
#pragma omp for nowait schedule(dynamic, 2)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - (i*(i+1)/2);
                store_ij(intor, eri, buf, i, j, vhfopt, &envs);
        }
        free(buf);
}
        CVHFdel_optimizer(&vhfopt);
}

