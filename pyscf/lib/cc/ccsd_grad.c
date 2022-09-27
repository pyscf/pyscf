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
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "ao2mo/nr_ao2mo.h"
#define OUTPUTIJ        1
#define INPUT_IJ        2

/*
 * a = reduce(numpy.dot, (mo_coeff, vin, mo_coeff.T))
 * numpy.tril(a + a.T)
 */
int CCmmm_transpose_sum(double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->nao * (envs->nao + 1) / 2;
                case INPUT_IJ: return envs->bra_count * envs->ket_count;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i, j, ij;
        double *mo_coeff = envs->mo_coeff; // in Fortran order
        double *buf1 = buf + nao*j_count;

        dgemm_(&TRANS_N, &TRANS_T, &j_count, &nao, &i_count,
               &D1, vin, &j_count, mo_coeff+i_start*nao, &nao,
               &D0, buf, &j_count);
        dgemm_(&TRANS_N, &TRANS_N, &nao, &nao, &j_count,
               &D1, mo_coeff+j_start*nao, &nao, buf, &j_count,
               &D0, buf1, &nao);

        for (ij = 0, i = 0; i < nao; i++) {
        for (j = 0; j <= i; j++, ij++) {
                vout[ij] = buf1[i*nao+j] + buf1[j*nao+i];
        } }
        return 0;
}

/*
 * for (ij|kl) == (ij|lk), in lower triangle kl
 * (ij|kl),lk->ij
 * (ij|kl),jk->il
 */
void CVHFics2kl_kl_s1ij(double *eri, double *dm, double *vj,
                        int nao, int ic, int jc);
void CVHFics2kl_jk_s1il(double *eri, double *dm, double *vk,
                        int nao, int ic, int jc);
void CCvhfs2kl(double *eri, double *dm, double *vj, double *vk, int ni, int nj)
{
        const size_t npair = nj*(nj+1)/2;
        int i, j;
        size_t ij, off;
        for (i = 0; i < ni * nj; i++) {
                vj[i] = 0;
                vk[i] = 0;
        }

#pragma omp parallel private(ij, i, j, off)
        {
                double *vj_priv = malloc(sizeof(double)*ni*nj);
                double *vk_priv = malloc(sizeof(double)*ni*nj);
                for (i = 0; i < ni * nj; i++) {
                        vj_priv[i] = 0;
                        vk_priv[i] = 0;
                }
#pragma omp for nowait schedule(dynamic, 4)
                for (ij = 0; ij < ni*nj; ij++) {
                        i = ij / nj;
                        j = ij - i * nj;
                        off = ij * npair;
                        CVHFics2kl_kl_s1ij(eri+off, dm, vj_priv, nj, i, j);
                        CVHFics2kl_jk_s1il(eri+off, dm, vk_priv, nj, i, j);
                }
#pragma omp critical
                {
                        for (i = 0; i < ni*nj; i++) {
                                vj[i] += vj_priv[i];
                                vk[i] += vk_priv[i];
                        }
                }
                free(vj_priv);
                free(vk_priv);
        }
}

