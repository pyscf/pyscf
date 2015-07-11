/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "ao2mo/nr_ao2mo.h"

/*
 * transform ket
 * s2 to label AO symmetry
 * copy from RIhalfmmm_nr_s2_ket
 */
int MCSCFhalfmmm_nr_s2_ket(double *vout, double *vin, struct _AO2MOEnvs *envs,
                           int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*j_count);
        int i, j;

        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, vin, &nao, mo_coeff+j_start*nao, &nao,
               &D0, buf, &nao);
        for (j = 0; j < nao; j++) {
                for (i = 0; i < j_count; i++) {
                        vout[i] = buf[i*nao+j];
                }
                vout += j_count;
        }
        free(buf);
        return 0;
}

