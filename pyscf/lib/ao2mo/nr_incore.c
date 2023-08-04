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
#include <assert.h>
//#include <omp.h>
#include "config.h"

#include "np_helper/np_helper.h"
#include "nr_ao2mo.h"
#define OUTPUTIJ        1
#define INPUT_IJ        2


void AO2MOtranse1_incore_s4(int (*fmmm)(), int row_id,
                            double *vout, double *eri_ao, double *buf,
                            struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t npair = (*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        double *peri = eri_ao + npair * (row_id+envs->klsh_start);

        NPdunpack_tril(nao, peri, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
}

void AO2MOtranse1_incore_s8(int (*fmmm)(), int row_id,
                            double *vout, double *eri_ao, double *buf,
                            struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t npair = (*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        double *buf0 = malloc(sizeof(double) * npair);

// Note AO2MOnr_e1incore_drv stores ij_start in envs.klsh_start
        NPdunpack_row(npair, row_id+envs->klsh_start, eri_ao, buf0);
        NPdunpack_tril(nao, buf0, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
        free(buf0);
}

// ij_start and ij_count for the ij-AO-pair in eri_ao
void AO2MOnr_e1incore_drv(void (*ftranse2_like)(), int (*fmmm)(),
                          double *vout, double *eri_ao, double *mo_coeff,
                          int ij_start, int ij_count, int nao,
                          int i_start, int i_count, int j_start, int j_count)
{
        struct _AO2MOEnvs envs;
        envs.bra_start = i_start;
        envs.bra_count = i_count;
        envs.ket_start = j_start;
        envs.ket_count = j_count;
        envs.nao = nao;
        envs.mo_coeff = mo_coeff;

        envs.klsh_start = ij_start;

        int i;
#pragma omp parallel default(none) \
        shared(ftranse2_like, fmmm, vout, eri_ao, ij_count, envs, \
               nao, i_count, j_count) \
        private(i)
{
        double *buf = malloc(sizeof(double) * (nao+i_count) * (nao+j_count));
#pragma omp for schedule(dynamic)
        for (i = 0; i < ij_count; i++) {
                (*ftranse2_like)(fmmm, i, vout, eri_ao, buf, &envs);
        }
        free(buf);
}
}

