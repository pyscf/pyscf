/*
 * File: nr_incore.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "nr_ao2mo_o3.h"
#include "nr_incore.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))

#define BLOCKDIM        56

// 8 fold symmetry of eri_ao
void AO2MOnr_incore8f_acc(double *vout, double *eri, double *mo_coeff,
                          int row_start, int row_count, int nao,
                          int i_start, int i_count, int j_start, int j_count,
                          void (*ftrans)())
{
        const int npair = nao*(nao+1)/2;
        const int nij = AO2MOcount_ij(i_start, i_count, j_start, j_count);
        double *buf1 = malloc(sizeof(double) * npair);
        double *buf2 = malloc(sizeof(double) * nij*row_count);
        int i, j, k, row_id;

        for (row_id = row_start, k = 0; k < row_count; row_id++, k++) {
                NPdunpack_row(npair, row_id, eri, buf1);
                (*ftrans)(buf2+nij*k, buf1, mo_coeff, nao,
                          i_start, i_count, j_start, j_count);
        }
        for (i = 0; i < nij; i++) {
                for (j = 0, k = row_start; j < row_count; k++, j++) {
                        vout[k] = buf2[j*nij+i];
                }
                vout += npair;
        }

        free(buf1);
        free(buf2);
}

// 4 fold symmetry of eri_ao
void AO2MOnr_incore4f_acc(double *vout, double *eri, double *mo_coeff,
                          int row_start, int row_count, int nao,
                          int i_start, int i_count, int j_start, int j_count,
                          void (*ftrans)())
{
        const long npair = nao*(nao+1)/2;
        const int nij = AO2MOcount_ij(i_start, i_count, j_start, j_count);
        double *buf2 = malloc(sizeof(double) * nij*row_count);
        int i, j, k, row_id;

        for (row_id = row_start, k = 0; k < row_count; row_id++, k++) {
                (*ftrans)(buf2+nij*k, eri+npair*row_id, mo_coeff, nao,
                          i_start, i_count, j_start, j_count);
        }
        for (i = 0; i < nij; i++) {
                for (j = 0, k = row_start; j < row_count; k++, j++) {
                        vout[k] = buf2[j*nij+i];
                }
                vout += npair;
        }

        free(buf2);
}


void AO2MOnr_e1incore_drv(double *eri_mo, double *eri_ao, double *mo_coeff,
                          void (*facc)(), void (*ftrans)(), int nao,
                          int i_start, int i_count, int j_start, int j_count)
{
        assert(j_start <= i_start);
        assert(j_start+j_count <= i_start+i_count);

        const int npair = nao*(nao+1)/2;
        int ij = 0;
#pragma omp parallel default(none) \
        shared(eri_mo, eri_ao, mo_coeff, facc, ftrans, \
               nao, i_start, i_count, j_start, j_count) \
        private(ij)
#pragma omp for nowait schedule(dynamic)
        for (ij = 0; ij < npair-BLOCKDIM+1; ij+=BLOCKDIM) {
                (*facc)(eri_mo, eri_ao, mo_coeff, ij, BLOCKDIM,
                        nao, i_start, i_count, j_start, j_count,
                        ftrans);
        }
        ij = npair - npair % BLOCKDIM;
        if (ij < npair) {
                (*facc)(eri_mo, eri_ao, mo_coeff, ij, npair-ij,
                        nao, i_start, i_count, j_start, j_count,
                        ftrans);
        }
}

