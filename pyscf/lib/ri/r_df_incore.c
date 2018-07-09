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
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "ao2mo/r_ao2mo.h"

void zhemm_(const char*, const char*,
            const int*, const int*,
            const double complex*, const double complex*, const int*,
            const double complex*, const int*,
            const double complex*, double complex*, const int*);

/*
 * transform bra (without doing conj(mo)), v_{iq} = C_{pi} v_{pq}
 * s1 to label AO symmetry
 */
int RIhalfmmm_r_s1_bra_noconj(double complex *vout, double complex *vin,
                              struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->nao;
                case 2: return envs->nao * envs->nao;
        }
        const double complex Z0 = 0;
        const double complex Z1 = 1;
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        double complex *mo_coeff = envs->mo_coeff;

        zgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &Z1, vin, &n2c, mo_coeff+i_start*n2c, &n2c,
               &Z0, vout, &n2c);
        return 0;
}

/*
 * transform ket, s1 to label AO symmetry
 */
int RIhalfmmm_r_s1_ket(double complex *vout, double complex *vin,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double complex Z0 = 0;
        const double complex Z1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double complex *mo_coeff = envs->mo_coeff;

        zgemm_(&TRANS_T, &TRANS_N, &j_count, &n2c, &n2c,
               &Z1, mo_coeff+j_start*n2c, &n2c, vin, &n2c,
               &Z0, vout, &j_count);
        return 0;
}

/*
 * transform bra (without doing conj(mo)), v_{iq} = C_{pi} v_{pq}
 * s2 to label AO symmetry
 */
int RIhalfmmm_r_s2_bra_noconj(double complex *vout, double complex *vin,
                              struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->nao;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double complex Z0 = 0;
        const double complex Z1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        double complex *mo_coeff = envs->mo_coeff;

        zhemm_(&SIDE_L, &UPLO_U, &n2c, &i_count,
               &Z1, vin, &n2c, mo_coeff+i_start*n2c, &n2c,
               &Z0, vout, &n2c);
        return 0;
}

/*
 * transform ket, s2 to label AO symmetry
 */
int RIhalfmmm_r_s2_ket(double complex *vout, double complex *vin,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double complex Z0 = 0;
        const double complex Z1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int n2c = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double complex *mo_coeff = envs->mo_coeff;
        double complex *buf = malloc(sizeof(double complex)*n2c*j_count);
        int i, j;

        zhemm_(&SIDE_L, &UPLO_U, &n2c, &j_count,
               &Z1, vin, &n2c, mo_coeff+j_start*n2c, &n2c,
               &Z0, buf, &n2c);
        for (j = 0; j < n2c; j++) {
                for (i = 0; i < j_count; i++) {
                        vout[i] = buf[i*n2c+j];
                }
                vout += j_count;
        }
        free(buf);
        return 0;
}

/*
 * unpack the AO integrals and copy to vout, s2 to label AO symmetry
 */
int RImmm_r_s2_copy(double complex *vout, double complex *vin,
                    struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->nao;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        int n2c = envs->nao;
        int i, j;
        for (i = 0; i < n2c; i++) {
                for (j = 0; j < i; j++) {
                        vout[i*n2c+j] = vin[j];
                        vout[j*n2c+i] = conj(vin[j]);
                }
                vout[i*n2c+i] = vin[i];
                vin += n2c;
        }
        return 0;
}

/*
 * transpose (no conj) the AO integrals and copy to vout, s2 to label AO symmetry
 */
int RImmm_r_s2_transpose(double complex *vout, double complex *vin,
                         struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->nao;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        int n2c = envs->nao;
        int i, j;
        for (i = 0; i < n2c; i++) {
                for (j = 0; j < i; j++) {
                        vout[j*n2c+i] = vin[j];
                        vout[i*n2c+j] = conj(vin[j]);
                }
                vout[i*n2c+i] = vin[i];
                vin += n2c;
        }
        return 0;
}


/*
 * ************************************************
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
void RItranse2_r_s1(int (*fmmm)(),
                    double complex *vout, double complex *vin, int row_id,
                    struct _AO2MOEnvs *envs)
{
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = envs->nao * envs->nao;
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, envs, 0);
}

void RItranse2_r_s2(int (*fmmm)(),
                    double complex *vout, double complex *vin, int row_id,
                    struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = nao*(nao+1)/2;
        double complex *buf = malloc(sizeof(double complex) * nao*nao);
        NPzunpack_tril(nao, vin+nao2*row_id, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
}

