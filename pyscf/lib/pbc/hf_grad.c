/* Copyright 2021-2024 The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdlib.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "pbc/neighbor_list.h"

#define MAX_THREADS 256

void contract_vhf_dm(double* out, double* vhf, double* dm,
                     NeighborList** neighbor_list,
                     int* shls_slice, int* ao_loc, int* shls_atm,
                     int comp, int natm, int nbas)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const size_t nijsh = (size_t)nish * njsh;
    const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
    const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];

    const int I1 = 1;
    double *out_bufs[MAX_THREADS];

#pragma omp parallel
{
    size_t ij, ish, jsh, p0, q0;
    int ni, nj, i, ic, iatm, nimgs=1;
    NeighborList *nl0=NULL;
    if (neighbor_list != NULL) {
        nl0 = *neighbor_list;
    }
    double *pvhf, *pdm;

    int thread_id = omp_get_thread_num();
    double *buf;
    if (thread_id == 0) {
        buf = out;
    } else {
        buf = calloc(comp*natm, sizeof(double));
    }
    out_bufs[thread_id] = buf;

    #pragma omp for schedule(dynamic) 
    for (ij = 0; ij < nijsh; ij++) {
        ish = ij / njsh + ish0;
        jsh = ij % njsh + jsh0;

        if (nl0 != NULL) {
            nimgs = ((nl0->pairs)[ish*nbas + jsh])->nimgs;
        }
        if (nimgs > 0) { // this shell pair has contribution
            p0 = ao_loc[ish] - ao_loc[ish0];
            q0 = ao_loc[jsh] - ao_loc[jsh0];
            ni = ao_loc[ish+1] - ao_loc[ish];
            nj = ao_loc[jsh+1] - ao_loc[jsh];

            iatm = shls_atm[ish];
            pvhf = vhf + (p0 * naoj + q0);
            pdm = dm + (p0 * naoj + q0);
            for (ic = 0; ic < comp; ic++) {
                for (i = 0; i < ni; i++) {
                    buf[iatm*3+ic] += ddot_(&nj, pvhf+i*naoj, &I1, pdm+i*naoj, &I1);
                }
                pvhf += naoi * naoj;
            }
        }
    }

    NPomp_dsum_reduce_inplace(out_bufs, comp*natm);
    if (thread_id != 0) {
        free(buf);
    }
}
}
