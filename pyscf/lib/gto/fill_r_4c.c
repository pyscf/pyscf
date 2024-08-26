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
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"

/*
 * out[naoi,naoj,naok,comp] in F-order
 */
void GTOr4c_fill_s1(int (*intor)(), double complex *out, double *buf,
                    int comp, int ish, int jsh,
                    int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int lsh0 = shls_slice[6];
        const int lsh1 = shls_slice[7];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t naol = ao_loc[lsh1] - ao_loc[lsh0];
        const size_t nij = naoi * naoj;
        const int dims[] = {naoi, naoj, naok, naol};

        ish += ish0;
        jsh += jsh0;
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += jp * naoi + ip;

        int ksh, lsh, k0, l0;
        int shls[4];

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                (*intor)(out+(l0*naok+k0)*nij, dims, shls,
                         atm, natm, bas, nbas, env, cintopt, buf);
        } }
}


void GTOr4c_drv(int (*intor)(), void (*fill)(), int (*prescreen)(),
                double complex *eri, int comp,
                int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        int ish, jsh, ij;
        double *buf = malloc(sizeof(double) * cache_size);
        if (buf == NULL) {
                fprintf(stderr, "malloc(%zu) failed in GTOr4c_drv\n",
                        sizeof(double) * cache_size);
                exit(1);
        }
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                (*fill)(intor, eri, buf, comp, ish, jsh, shls_slice, ao_loc,
                        cintopt, atm, natm, bas, nbas, env);
        }
        free(buf);
}
}
