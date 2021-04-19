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
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define SYMMETRIC       3

// from cint.h
#define NGRIDS          11

/*
 * mat(ngrids,naoi,naoj,comp) in F-order
 */
void GTOgrids_int2c(int (*intor)(), double *mat, int comp, int hermi,
                    int *shls_slice, int *ao_loc, CINTOpt *opt,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t ngrids = env[NGRIDS];
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                                 atm, natm, bas, nbas, env);
        const int dims[] = {ngrids, naoi, naoj};
#pragma omp parallel
{
        size_t ij;
        int ish, jsh, i0, j0;
        int shls[2];
        double *cache = malloc(sizeof(double) * cache_size);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        // fill up only upper triangle of F-array
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                (*intor)(mat+ngrids*(j0*naoi+i0), dims, shls,
                         atm, natm, bas, nbas, env, opt, cache);
        }
        free(cache);

        if (hermi != PLAIN) { // lower triangle of F-array
                size_t ic, i, j, ig;
                size_t nao2 = naoi * naoj;
                double *mat_ij, *mat_ji;
#pragma omp for schedule(dynamic, 4)
                for (ij = 0; ij < nao2*comp; ij++) {
                        ic = ij / nao2;
                        ig = ij % nao2;
                        i = ig / naoj;
                        j = ig % naoj;
                        if (i > j) {
                                continue;
                        }
                        // Note the F-order array mat are filled in the upper
                        // triangular part
                        mat_ij = mat + ngrids * (ic * nao2 + j * naoi + i);
                        mat_ji = mat + ngrids * (ic * nao2 + i * naoi + j);
                        if (hermi == HERMITIAN || hermi == SYMMETRIC) {
                                for (ig = 0; ig < ngrids; ig++) {
                                        mat_ji[ig] = mat_ij[ig];
                                }
                        } else {
                                for (ig = 0; ig < ngrids; ig++) {
                                        mat_ji[ig] = -mat_ij[ig];
                                }
                        }
                }
        }
}
}

void GTOgrids_int2c_spinor(int (*intor)(), double complex *mat, int comp, int hermi,
                           int *shls_slice, int *ao_loc, CINTOpt *opt,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t ngrids = env[NGRIDS];
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                                 atm, natm, bas, nbas, env);
        int dims[] = {ngrids, naoi, naoj};
#pragma omp parallel
{
        size_t ij;
        int ish, jsh, i0, j0;
        int shls[2];
        double *cache = malloc(sizeof(double) * cache_size);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                (*intor)(mat+ngrids*(j0*naoi+i0), dims, shls,
                         atm, natm, bas, nbas, env, opt, cache);
        }
        free(cache);

        if (hermi != PLAIN) { // lower triangle of F-array
                size_t ic, i, j, ig;
                size_t nao2 = naoi * naoj;
                double complex *mat_ij, *mat_ji;
#pragma omp for schedule(dynamic, 4)
                for (ij = 0; ij < nao2*comp; ij++) {
                        ic = ij / nao2;
                        ig = ij % nao2;
                        i = ig / naoj;
                        j = ig % naoj;
                        if (i > j) {
                                continue;
                        }
                        // Note the F-order array mat are filled in the upper
                        // triangular part
                        mat_ij = mat + ngrids * (ic * nao2 + j * naoi + i);
                        mat_ji = mat + ngrids * (ic * nao2 + i * naoi + j);
                        if (hermi == HERMITIAN) {
                                for (ig = 0; ig < ngrids; ig++) {
                                        mat_ji[ig] = conj(mat_ij[ig]);
                                }
                        } else if (hermi == SYMMETRIC) {
                                for (ig = 0; ig < ngrids; ig++) {
                                        mat_ji[ig] = mat_ij[ig];
                                }
                        } else {
                                for (ig = 0; ig < ngrids; ig++) {
                                        mat_ji[ig] = -conj(mat_ij[ig]);
                                }
                        }
                }
        }
}
}
