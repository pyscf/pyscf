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
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

#define BLKSIZE 8

/*
 * out[naoi,naoj,naok,comp] in F-order
 */
void GTOnr3c_fill_s1(int (*intor)(), double *out, double *buf,
                     int comp, int jobid,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int nksh = ksh1 - ksh0;

        const int ksh = jobid % nksh + ksh0;
        const int jstart = jobid / nksh * BLKSIZE + jsh0;
        const int jend = MIN(jstart + BLKSIZE, jsh1);
        if (jstart >= jend) {
                return;
        }

        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const int dims[] = {naoi, naoj, naok};

        const int k0 = ao_loc[ksh] - ao_loc[ksh0];
        out += naoi * naoj * k0;

        int ish, jsh, i0, j0;
        int shls[3] = {0, 0, ksh};

        for (jsh = jstart; jsh < jend; jsh++) {
        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                (*intor)(out+j0*naoi+i0, dims, shls, atm, natm, bas, nbas, env,
                         cintopt, buf);
        } }
}


static void dcopy_s2_igtj(double *out, double *in, int comp,
                          int ip, int nij, int nijk, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double *pout, *pin;
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
static void dcopy_s2_ieqj(double *out, double *in, int comp,
                          int ip, int nij, int nijk, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        const size_t ip1 = ip + 1;
        int i, j, k, ic;
        double *pout, *pin;
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
void GTOnr3c_fill_s2ij(int (*intor)(), double *out, double *buf,
                       int comp, int jobid,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int nksh = ksh1 - ksh0;

        const int ksh = jobid % nksh + ksh0;
        const int istart = jobid / nksh * BLKSIZE + ish0;
        const int iend = MIN(istart + BLKSIZE, ish1);
        if (istart >= iend) {
                return;
        }

        const int i0 = ao_loc[ish0];
        const int i1 = ao_loc[ish1];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t off = i0 * (i0 + 1) / 2;
        const size_t nij = i1 * (i1 + 1) / 2 - off;
        const size_t nijk = nij * naok;

        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int k0 = ao_loc[ksh] - ao_loc[ksh0];
        out += nij * k0;

        int ish, jsh, ip, jp, di, dj;
        int shls[3] = {0, 0, ksh};
        di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        double *cache = buf + di * di * dk * comp;
        double *pout;

        for (ish = istart; ish < iend; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                ip = ao_loc[ish];
                jp = ao_loc[jsh] - ao_loc[jsh0];
                if (ip < jp) {
                        continue;
                }
                shls[0] = ish;
                shls[1] = jsh;
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];

                (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache);

                pout = out + ip * (ip + 1) / 2 - off + jp;
                if (ip != jp) {
                        dcopy_s2_igtj(pout, buf, comp, ip, nij, nijk, di, dj, dk);
                } else {
                        dcopy_s2_ieqj(pout, buf, comp, ip, nij, nijk, di, dj, dk);
                }
        } }
}

void GTOnr3c_fill_s2jk(int (*intor)(), double *out, double *buf,
                       int comp, int jobid,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        fprintf(stderr, "GTOnr3c_fill_s2jk not implemented\n");
        exit(1);
}

void GTOnr3c_drv(int (*intor)(), void (*fill)(), double *eri, int comp,
                 int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int nksh = ksh1 - ksh0;
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 3);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                                 atm, natm, bas, nbas, env);
        const int njobs = (MAX(nish,njsh) / BLKSIZE + 1) * nksh;

#pragma omp parallel
{
        int jobid;
        double *buf = malloc(sizeof(double) * (di*di*di*comp + cache_size));
#pragma omp for nowait schedule(dynamic)
        for (jobid = 0; jobid < njobs; jobid++) {
                (*fill)(intor, eri, buf, comp, jobid, shls_slice, ao_loc,
                        cintopt, atm, natm, bas, nbas, env);
        }
        free(buf);
}
}

