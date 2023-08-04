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
void GTOr3c_fill_s1(int (*intor)(), double complex *out, double complex *buf,
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
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t nij = naoi * naoj;
        const int dims[] = {naoi, naoj, naok};

        ish += ish0;
        jsh += jsh0;
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += jp * naoi + ip;

        int ksh, k0;
        int shls[3];

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
                shls[2] = ksh;
                k0 = ao_loc[ksh  ] - ao_loc[ksh0];
                (*intor)(out+k0*nij, dims, shls, atm, natm, bas, nbas, env, cintopt, buf);
        }
}


static void zcopy_s2_igtj(double complex *out, double complex *in, int comp,
                          int ip, int nij, int nijk, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double complex *pout, *pin;
        for (ic = 0; ic < comp; ic++) {
                for (k = 0; k < dk; k++) {
                        pout = out + k * nij;
                        pin  = in  + k * dij;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        pout[j] = pin[j*di+i];
                                }
                                pout += ip1 + i;
                        }
                }
                out += nijk;
                in  += dij * dk;
        }
}
static void zcopy_s2_ieqj(double complex *out, double complex *in, int comp,
                          int ip, int nij, int nijk, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double complex *pout, *pin;
        for (ic = 0; ic < comp; ic++) {
                for (k = 0; k < dk; k++) {
                        pout = out + k * nij;
                        pin  = in  + k * dij;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        pout[j] = pin[j*di+i];
                                }
                                pout += ip1 + i;
                        }
                }
                out += nijk;
                in  += dij * dk;
        }
}
/*
 * out[comp,naok,nij] in C-order
 * nij = i1*(i1+1)/2 - i0*(i0+1)/2
 *     [  \    ]
 *     [****   ]
 *     [*****  ]
 *     [*****. ]  <= . may not be filled, if jsh-upper-bound < ish-upper-bound
 *     [      \]
 */
void GTOr3c_fill_s2ij(int (*intor)(), double complex *out, double complex *buf,
                      int comp, int ish, int jsh,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        ish += ish0;
        jsh += jsh0;
        const int ip = ao_loc[ish];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        if (ip < jp) {
                return;
        }

        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int i0 = ao_loc[ish0];
        const int i1 = ao_loc[ish1];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off = i0 * (i0 + 1) / 2;
        const size_t nij = i1 * (i1 + 1) / 2 - off;
        const size_t nijk = nij * naok;

        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        out += ip * (ip + 1) / 2 - off + jp;

        int ksh, dk, k0;
        int shls[3];
        dk = GTOmax_shell_dim(ao_loc, shls_slice, 3);
        double *cache = (double *)(buf + di * dj * dk * comp);

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
                shls[2] = ksh;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                k0 = ao_loc[ksh  ] - ao_loc[ksh0];
                (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache);
                if (ip != jp) {
                        zcopy_s2_igtj(out+k0*nij, buf, comp, ip, nij, nijk, di, dj, dk);
                } else {
                        zcopy_s2_ieqj(out+k0*nij, buf, comp, ip, nij, nijk, di, dj, dk);
                }
        }
}

void GTOr3c_fill_s2jk(int (*intor)(), double complex *out, double complex *buf,
                      int comp, int ish, int jsh,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        fprintf(stderr, "GTOr3c_fill_s2jk not implemented\n");
        exit(1);
}

void GTOr3c_drv(int (*intor)(), void (*fill)(), double complex *eri, int comp,
                int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 3);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        int ish, jsh, ij;
        double complex *buf = malloc(sizeof(double complex) *
                                     (di*di*di*comp + cache_size/2));
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
