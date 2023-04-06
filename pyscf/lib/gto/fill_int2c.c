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
#include <math.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define SYMMETRIC       3

/*
 * mat(naoi,naoj,comp) in F-order
 */
void GTOint2c(int (*intor)(), double *mat, int comp, int hermi,
              int *shls_slice, int *ao_loc, CINTOpt *opt,
              int *atm, int natm, int *bas, int nbas, double *env)
{
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
#pragma omp parallel
{
        int dims[] = {naoi, naoj};
        int ish, jsh, ij, i0, j0;
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
                (*intor)(mat+j0*naoi+i0, dims, shls,
                         atm, natm, bas, nbas, env, opt, cache);
        }
        free(cache);
}
        if (hermi != PLAIN) { // lower triangle of F-array
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPdsymm_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}

void GTOint2c_spinor(int (*intor)(), double complex *mat, int comp, int hermi,
                     int *shls_slice, int *ao_loc, CINTOpt *opt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
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

#pragma omp parallel
{
        int dims[] = {naoi, naoj};
        int ish, jsh, ij, i0, j0;
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
                (*intor)(mat+j0*naoi+i0, dims, shls,
                         atm, natm, bas, nbas, env, opt, cache);
        }
        free(cache);
}
        if (hermi != PLAIN) {
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPzhermi_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}

// Similar to CINTOpt_log_max_pgto_coeff and CINTOpt_set_log_maxc
static void _log_max_pgto_coeff(double *log_maxc, double *coeff, int nprim, int nctr)
{
        int i, ip;
        double maxc;
        for (ip = 0; ip < nprim; ip++) {
                maxc = 0;
                for (i = 0; i < nctr; i++) {
                        maxc = MAX(maxc, fabs(coeff[i*nprim+ip]));
                }
                log_maxc[ip] = log(maxc);
        }
}

static void _set_log_max_coeff(double **log_max_coeff,
                               int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, iprim, ictr;
        double *ci;
        size_t tot_prim = 0;
        for (i = 0; i < nbas; i++) {
                tot_prim += bas(NPRIM_OF, i);
        }

        double *plog_maxc = malloc(sizeof(double) * (tot_prim+1));
        log_max_coeff[0] = plog_maxc;
        for (i = 0; i < nbas; i++) {
                iprim = bas(NPRIM_OF, i);
                ictr = bas(NCTR_OF, i);
                ci = env + bas(PTR_COEFF, i);
                log_max_coeff[i] = plog_maxc;
                _log_max_pgto_coeff(plog_maxc, ci, iprim, ictr);
                plog_maxc += iprim;
        }
}

void GTOoverlap_cond(double *cond, int *shls_slice,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int njsh = jsh1 - jsh0;
        double **log_max_coeff = malloc(sizeof(double *) * (nbas + 1));
        _set_log_max_coeff(log_max_coeff, atm, natm, bas, nbas, env);
#pragma omp parallel
{
        int ish, jsh, ip, jp, li, lj, iprim, jprim;
        double aij, eij, cceij, min_cceij, log_rr_ij, dx, dy, dz, rr_ij;
        double *ai, *aj, *ri, *rj, *log_maxci, *log_maxcj;
#pragma omp for nowait schedule(static)
        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                iprim = bas(NPRIM_OF, ish);
                jprim = bas(NPRIM_OF, jsh);
                li = bas(ANG_OF, ish);
                lj = bas(ANG_OF, jsh);
                ai = env + bas(PTR_EXP, ish);
                aj = env + bas(PTR_EXP, jsh);
                ri = env + atm(PTR_COORD, bas(ATOM_OF, ish));
                rj = env + atm(PTR_COORD, bas(ATOM_OF, jsh));
                dx = ri[0] - rj[0];
                dy = ri[1] - rj[1];
                dz = ri[2] - rj[2];
                rr_ij = dx * dx + dy * dy + dz * dz;

// This estimation is based on the assumption that the two gaussian charge
// distributions are separated in space.
                log_rr_ij = (li+lj+1) * log(rr_ij+1) / 2;

                log_maxci = log_max_coeff[ish];
                log_maxcj = log_max_coeff[jsh];
                min_cceij = 1e9;
                for (jp = 0; jp < jprim; jp++) {
                for (ip = 0; ip < iprim; ip++) {
                        aij = ai[ip] + aj[jp];
                        eij = rr_ij * ai[ip] * aj[jp] / aij;
                        cceij = eij - log_rr_ij - log_maxci[ip] - log_maxcj[jp];
                        min_cceij = MIN(min_cceij, cceij);
                } }
                cond[(ish-ish0)*njsh+(jsh-jsh0)] = min_cceij;
        } }
}
        free(log_max_coeff[0]);
        free(log_max_coeff);
}
