#include "fft.h"
#include <omp.h>
#include "vhf/fblas.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

int get_omp_threads();
int omp_get_thread_num();

void _process_dm(
    const double *dm,
    const int nao,
    const double cutoff,
    int *nElmtRow, // record the number of elements in each row, size of which should be larger than nao + 1
    int *nNonZeroElmt)
{
    *nNonZeroElmt = 0;
    int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int i = 0;

        int NonZeroFound = 0;

#pragma omp for schedule(dynamic)
        for (i = 0; i < nao; i++)
        {
            int nNonZero = 0;
            for (int j = 0; j < nao; j++)
            {
                if (fabs(dm[i * nao + j]) > cutoff)
                {
                    nNonZero++;
                }
            }
            nElmtRow[i] = nNonZero;
            NonZeroFound += nNonZero;
        }

#pragma omp critical
        {
            *nNonZeroElmt += NonZeroFound;
        }
    }
}

void _compress_dm(
    const double *dm,
    const int nao,
    const double cutoff,
    const int *nElmtRow,
    int *RowLoc,
    int *ColIndx,
    double *dm_sparse)
{
    *RowLoc = 0;
    for (int i = 0; i < nao; i++)
    {
        RowLoc[i + 1] = RowLoc[i] + nElmtRow[i];
    }

    int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int i = 0;

        double *dm_ptr;
        int *indx_ptr;

#pragma omp for schedule(dynamic)
        for (i = 0; i < nao; i++)
        {
            dm_ptr = dm_sparse + RowLoc[i];
            indx_ptr = ColIndx + RowLoc[i];
            for (int j = 0; j < nao; j++)
            {
                if (fabs(dm[i * nao + j]) > cutoff)
                {
                    *dm_ptr++ = dm[i * nao + j];
                    *indx_ptr++ = j;
                }
            }
        }
    }
}

void _dm_aoR_spMM(
    const double *dm_sparse,
    const int *RowLoc,
    const int *ColIndx,
    const double *aoR,
    const int nao,
    const int ngrids,
    double *out)
{
    static const int ONE = 1.0;

    // parallel over each row of dm_sparse

    int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int i = 0;

        double *out_ptr;
        const double *aoR_ptr;
        const double *dm_ptr;
        const int *indx_ptr;

#pragma omp for schedule(dynamic)
        for (i = 0; i < nao; i++)
        {
            out_ptr = out + i * ngrids;
            dm_ptr = dm_sparse + RowLoc[i];
            indx_ptr = ColIndx + RowLoc[i];
            memset(out_ptr, 0, sizeof(double) * ngrids);
            for (int j = 0; j < RowLoc[i + 1] - RowLoc[i]; j++)
            {
                aoR_ptr = aoR + indx_ptr[j] * ngrids;
                daxpy_(&ngrids, dm_ptr + j, aoR_ptr, &ONE, out_ptr, &ONE);
            }
        }
    }
}

void NPdcwisemul(double *out, double *a, double *b, size_t n);

void _cwise_product_check_Sparsity(
    const double *V,
    const double *dmRgR,
    double *out,
    const int naux,
    const int ngrids,
    const double cutoff,
    double *buf,
    int *UseSparsity,
    int *IsSparsity)
{
    /// choose seed based on the current time

    static const double COMPRESS_CRITERION = 0.15;

    srand(time(NULL));

    *UseSparsity = 1;
    int nThread = get_omp_threads();

    int nNonZeroElmt = 0;

    NPdcwisemul(out, V, dmRgR, naux * ngrids);

#pragma omp parallel num_threads(nThread)
    {
        int i = 0;
        int nNonZero = 0;

#pragma omp for schedule(static) nowait
        for (i = 0; i < naux * ngrids; i++)
        {
            if (fabs(out[i]) > cutoff)
            {
                nNonZero++;
            }
            else
            {
                out[i] = 0.0;
            }
        }

#pragma omp critical
        {
            nNonZeroElmt += nNonZero;
        }
    }

    double sparsity = (double)nNonZeroElmt / (naux * ngrids);
    printf("sparsity: %8.2f percentage \n", sparsity * 100);

    if (sparsity < COMPRESS_CRITERION)
    {
        *UseSparsity = 1;
    }
    else
    {
        *UseSparsity = 0;
    }

    if (*UseSparsity == 1)
    {
        const int nMaxElmt = ngrids * COMPRESS_CRITERION * 2;

        int nDense = 0;

#pragma omp parallel num_threads(nThread)
        {
            int32_t *nElmt_ptr, *indx_ptr;
            double *Elmt_ptr, *out_ptr;

            int thread_id = omp_get_thread_num();
            double *buf_thread = buf + thread_id * ngrids;

#pragma omp for schedule(static) nowait
            for (int i = 0; i < naux; i++)
            {
                nElmt_ptr = (int32_t *)buf_thread;
                indx_ptr = (int32_t *)((char *)buf_thread + sizeof(int32_t));
                Elmt_ptr = buf_thread + ngrids - 1;
                out_ptr = out + i * ngrids;

                *nElmt_ptr = 0;
                for (int j = 0; j < ngrids; j++)
                {
                    if (fabs(out_ptr[j]) > cutoff)
                    {
                        *Elmt_ptr-- = out_ptr[j];
                        *indx_ptr++ = j;
                        *nElmt_ptr += 1;
                    }
                }
                if (*nElmt_ptr > nMaxElmt)
                {
                    IsSparsity[i] = 0;

#pragma omp atomic
                    nDense++;
                }
                else
                {
                    IsSparsity[i] = 1;
                    memcpy(out_ptr, buf_thread, sizeof(double) * ngrids);
                }
            }
        }
        printf("nDense: %d \n", nDense);
    }
}

void _V_Dm_product_SpMM(
    const double *V_Dm_Product,
    const int *IsSparsity,
    const double *aoR,
    const int nao,
    const int naux,
    const int ngrids,
    double *out)
{
    static const int ONE = 1;

    int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int i = 0, j = 0, k = 0;

        double *out_ptr;

        int32_t *nElmt_ptr, *indx_ptr;
        const double *Elmt_ptr;
        const double *aoR_ptr;

        int32_t nElmt;

#pragma omp for schedule(static) nowait
        for (i = 0; i < naux; i++)
        {
            if (IsSparsity[i] == 0)
            {
                out_ptr = out + i * nao;
                memset(out_ptr, 0, sizeof(double) * nao);

                for (j = 0; j < nao; j++)
                {
                    out_ptr[j] = ddot_(&ngrids, aoR + j * ngrids, &ONE, V_Dm_Product + i * ngrids, &ONE);
                }
            }
            else
            {
                // # note extremely slow
                nElmt_ptr = (int32_t *)(V_Dm_Product + i * ngrids);
                indx_ptr = (int32_t *)(nElmt_ptr + 1);
                Elmt_ptr = V_Dm_Product + (i + 1) * ngrids - 1;
                nElmt = *nElmt_ptr;

                out_ptr = out + i * nao;
                memset(out_ptr, 0, sizeof(double) * nao);

                if (nElmt == 0)
                {
                    continue;
                }

                for (j = 0; j < nao; j++)
                {
                    aoR_ptr = aoR + j * ngrids;
                    for (k = 0; k < nElmt; k++)
                    {
                        out_ptr[j] += aoR_ptr[indx_ptr[k]] * Elmt_ptr[-k];
                    }
                }
            }
        }
    }
}

void _V_Dm_product_SpMM2(
    const double *V_Dm_Product,
    const int *IsSparsity,
    const double *aoRT,
    const int nao,
    const int naux,
    const int ngrids,
    double *out)
{
    static const int ONE = 1;

    int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int i = 0, j = 0, k = 0;

        double *out_ptr;

        int32_t *nElmt_ptr, *indx_ptr;
        const double *Elmt_ptr;
        const double *aoR_ptr;

        int32_t nElmt;

#pragma omp for schedule(static) nowait
        for (i = 0; i < naux; i++)
        {
            if (IsSparsity[i] == 0)
            {
                out_ptr = out + i * nao;
                memset(out_ptr, 0, sizeof(double) * nao);

                // summation over grids

                for (j = 0; j < ngrids; j++)
                {
                    daxpy_(&nao, V_Dm_Product + i * ngrids + j, aoRT + j * nao, &ONE, out_ptr, &ONE);
                }
            }
            else
            {
                // # note extremely slow
                nElmt_ptr = (int32_t *)(V_Dm_Product + i * ngrids);
                indx_ptr = (int32_t *)(nElmt_ptr + 1);
                Elmt_ptr = V_Dm_Product + (i + 1) * ngrids - 1;
                nElmt = *nElmt_ptr;

                out_ptr = out + i * nao;
                memset(out_ptr, 0, sizeof(double) * nao);

                if (nElmt == 0)
                {
                    continue;
                }

                for (j = 0; j < nElmt; j++)
                {
                    aoR_ptr = aoRT + indx_ptr[j] * nao;
                    daxpy_(&nao, Elmt_ptr - j, aoR_ptr, &ONE, out_ptr, &ONE);
                }
            }
        }
    }
}

//////// BASIC OPERATION USED IN GET_JK for k_ISDF ////////

void dcwisemul_dense_sparse_kernel(
    double *out,
    const double *global,
    const int nrow_global,
    const int ncol_global,
    const int row_shift,
    const int col_shift,
    const double *local,
    const int *ao_invovled,
    const int row_local,
    const int col_local,
    const int row_begin,
    const int row_end,
    const int col_begin,
    const int col_end)
{
    const int nrow = row_end - row_begin;
    const int ncol = col_end - col_begin;
    memset(out, 0, sizeof(double) * nrow * ncol);

    int nthread = get_omp_threads();

    const double *global_head = global + row_shift * ncol_global + col_shift;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (int i = 0; i < row_local; i++)
    {
        const int irow = ao_invovled[i];
        if (irow < row_begin || irow >= row_end)
        {
            continue;
        }
        double *p_global = global_head + (irow - row_begin) * ncol_global;
        double *p_local = local + i * col_local;
        double *p_out = out + i * ncol;
        for (int j = 0; j < ncol; j++)
        {
            p_out[j] = p_global[j] * p_local[j];
        }
    }
}