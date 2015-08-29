/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdlib.h>
#include <assert.h>
#include "vhf/fblas.h"
#include "ao2mo/nr_ao2mo.h"
#define OUTPUTIJ        1
#define INPUT_IJ        2

/*
 * a = reduce(numpy.dot, (mo_coeff, vin, mo_coeff.T))
 * numpy.tril(a + a.T)
 */
int CCmmm_transpose_sum(double *vout, double *vin, struct _AO2MOEnvs *envs,
                        int seekdim)
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
        double *buf = malloc(sizeof(double)*nao*nao*2);
        double *buf1 = buf + nao*nao;

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
        free(buf);
        return 0;
}

