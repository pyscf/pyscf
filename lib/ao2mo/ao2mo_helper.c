#include <stdlib.h>
#include <string.h>
#include <omp.h>

#if defined SCIPY_MKL_H
typedef long FINT;
#else
typedef int FINT;
#endif

#include "vhf/fblas.h"
#include "vhf/misc.h"

static void ao2mo_unpack(int n, double *vec, double *mat)
{
        int i, j;
        for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, vec++) {
                        mat[i*n+j] = *vec;
                        mat[j*n+i] = *vec;
                }
        }
}


/* eri uses 4-fold symmetry: i>=j,k>=l */
void ao2mo_half_trans_o2(int nao, int nmo, double *eri, double *c,
                         double *mat)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const FINT lao = nao;
        const FINT lmo = nmo;
        int i, j;
        double *tmp1 = malloc(sizeof(double)*nao*nao);
        double *tmp2 = malloc(sizeof(double)*nao*nmo);
        ao2mo_unpack(nao, eri, tmp1);
        dgemm_(&TRANS_N, &TRANS_N, &lao, &lmo, &lao,
               &D1, tmp1, &lao, c, &lao, &D0, tmp2, &lao);
        dgemm_(&TRANS_T, &TRANS_N, &lmo, &lmo, &lao,
               &D1, c, &lao, tmp2, &lao, &D0, tmp1, &lmo);

        for (i = 0; i < nmo; i++) {
                for (j = 0; j <= i; j++, mat++) {
                        *mat = tmp1[i*nmo+j];
                }
        }
        free(tmp1);
        free(tmp2);
}


/* eri uses 8-fold symmetry: i>=j,k>=l,ij>=kl */
void ao2mo_half_trans_o3(int nao, int nmo, int pair_id,
                         double *eri, double *c, double *mat)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const FINT lao = nao;
        const FINT lmo = nmo;
        int i, j;
        int nao_pair = nao * (nao+1) / 2;
        double *row = malloc(sizeof(double)*nao_pair);
        double *tmp1 = malloc(sizeof(double)*nao*nao);
        double *tmp2 = malloc(sizeof(double)*nao*nmo);

        extract_row_from_tri(row, pair_id, nao_pair, eri);
        ao2mo_unpack(nao, row, tmp1);
        dgemm_(&TRANS_N, &TRANS_N, &lao, &lmo, &lao,
               &D1, tmp1, &lao, c, &lao, &D0, tmp2, &lao);
        dgemm_(&TRANS_T, &TRANS_N, &lmo, &lmo, &lao,
               &D1, c, &lao, tmp2, &lao, &D0, tmp1, &lmo);

        for (i = 0; i < nmo; i++) {
                for (j = 0; j <= i; j++, mat++) {
                        *mat = tmp1[i*nmo+j];
                }
        }
        free(row);
        free(tmp1);
        free(tmp2);
}

