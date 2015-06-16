/*
 * File: nr_incore.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdlib.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"

#include "np_helper/np_helper.h"
#include "nr_ao2mo.h"


void AO2MOtranse1_incore_s4(int (*fmmm)(),
                            double *vout, double *eri_ao, int row_id,
                            struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t npair = nao * (nao+1) / 2;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        double *buf = malloc(sizeof(double) * nao*nao);
        double *buf1 = eri_ao + npair * (row_id+envs->klsh_start);

        NPdunpack_tril(nao, buf1, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
}

void AO2MOtranse1_incore_s8(int (*fmmm)(),
                            double *vout, double *eri_ao, int row_id,
                            struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int npair = nao * (nao+1) / 2;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        double *buf = malloc(sizeof(double) * nao*nao*2);
        double *buf1 = buf + nao*nao;

// Note AO2MOnr_e1incore_drv stores ij_start in envs.klsh_start
        NPdunpack_row(npair, row_id+envs->klsh_start, eri_ao, buf1);
        NPdunpack_tril(nao, buf1, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
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
        shared(ftranse2_like, fmmm, vout, eri_ao, ij_count, envs) \
        private(i)
#pragma omp for nowait schedule(static)
        for (i = 0; i < ij_count; i++) {
                (*ftranse2_like)(fmmm, vout, eri_ao, i, &envs);
        }
}

