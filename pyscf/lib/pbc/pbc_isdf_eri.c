#include "fft.h"
#include <omp.h>
#include <string.h>
#include <complex.h>
#include "vhf/fblas.h"
#include <math.h>
#include "np_helper/np_helper.h"
#include <stdbool.h>

int get_omp_threads();
int omp_get_thread_num();

void _pack_aoR_to_aoPairR_diff(
    double *aoR_i,
    double *aoR_j,
    double *aoPairR,
    int nao_i,
    int nao_j,
    int ngrid)
{
    int nPair = nao_i * nao_j;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nPair; i++)
    {
        int i1 = i / nao_j;
        int j1 = i % nao_j;
        for (int k = 0; k < ngrid; k++)
        {
            aoPairR[i * ngrid + k] = aoR_i[i1 * ngrid + k] * aoR_j[j1 * ngrid + k];
        }
    }
}

void _pack_aoR_to_aoPairR_same(
    double *aoR,
    double *aoPairR,
    int nao,
    int ngrid)
{
    // int nPair = nao * (nao + 1) / 2;

#pragma omp parallel for schedule(static)
    for (int i1 = 0; i1 < nao; ++i1)
    {
        for (int j1 = 0; j1 <= i1; ++j1)
        {
            int i = i1 * (i1 + 1) / 2 + j1;
            for (int k = 0; k < ngrid; ++k)
            {
                aoPairR[i * ngrid + k] = aoR[i1 * ngrid + k] * aoR[j1 * ngrid + k];
            }
        }
    }
}

#define COMBINE2(i, j) ((i) < (j) ? (j) * (j + 1) / 2 + i : i * (i + 1) / 2 + j)

void _unpack_suberi_to_eri(
    double *eri,
    const int nao,
    double *suberi,
    const int nao_bra,
    const int *ao_loc_bra,
    const int nao_ket,
    const int *ao_loc_ket,
    const int add_transpose)
{
    int nPair = nao * (nao + 1) / 2;

    int nPair_ket = nao_ket * (nao_ket + 1) / 2;
    // int nPair_bra = nao_bra * (nao_bra + 1) / 2;

#pragma omp parallel for schedule(static)
    for (int i1 = 0; i1 < nao_bra; ++i1)
    {
        for (int j1 = 0; j1 <= i1; ++j1)
        {
            int i = ao_loc_bra[i1];
            int j = ao_loc_bra[j1];
            int ij = COMBINE2(i, j);
            int i1j1 = COMBINE2(i1, j1);
            // printf("i1: %d, j1: %d, i: %d, j: %d, ij: %d, i1j1: %d\n", i1, j1, i, j, ij, i1j1);
            for (int k1 = 0; k1 < nao_ket; ++k1)
            {
                for (int l1 = 0; l1 <= k1; ++l1)
                {
                    int k = ao_loc_ket[k1];
                    int l = ao_loc_ket[l1];
                    int kl = COMBINE2(k, l);
                    int k1l1 = COMBINE2(k1, l1);
                    eri[ij * nPair + kl] += suberi[i1j1 * nPair_ket + k1l1];
                }
            }
        }
    }

    if (add_transpose)
    {
#pragma omp parallel for schedule(static)
        for (int i1 = 0; i1 < nao_bra; ++i1)
        {
            for (int j1 = 0; j1 <= i1; ++j1)
            {
                int i = ao_loc_bra[i1];
                int j = ao_loc_bra[j1];
                int ij = COMBINE2(i, j);
                int i1j1 = COMBINE2(i1, j1);
                for (int k1 = 0; k1 < nao_ket; ++k1)
                {
                    for (int l1 = 0; l1 <= k1; ++l1)
                    {
                        int k = ao_loc_ket[k1];
                        int l = ao_loc_ket[l1];
                        int kl = COMBINE2(k, l);
                        int k1l1 = COMBINE2(k1, l1);
                        eri[kl * nPair + ij] += suberi[i1j1 * nPair_ket + k1l1];
                    }
                }
            }
        }
    }
}

void _unpack_suberi_to_eri_ovov(
    double *eri,
    double *suberi,
    const int nPair,
    const int add_transpose)
{
    static const double ALPHA = 1.0;
    static const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < nPair; i++)
    {
        daxpy_(&nPair, &ALPHA, suberi + i * nPair, &INCX, eri + i * nPair, &INCX);
    }

    if (add_transpose)
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nPair; i++)
        {
            daxpy_(&nPair, &ALPHA, suberi + i * nPair, &INCX, eri + i, &nPair);
        }
    }
}

#undef COMBINE2

/// sliced operation ///

void fn_slice_2_0(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int slice_0_0,
    const int slice_0_1)
{
    int dim0 = slice_0_1 - slice_0_0;

#pragma omp parallel for
    for (size_t i = slice_0_0; i < slice_0_1; i++)
    {
        memcpy(tensor_B + (i - slice_0_0) * n1, tensor_A + i * n1, sizeof(double) * n1);
    }
}

void fn_slice_2_1(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int slice_1_0,
    const int slice_1_1)
{
    int dim1 = slice_1_1 - slice_1_0;
#pragma omp parallel for
    for (size_t i = 0; i < n0; i++)
    {
        memcpy(tensor_B + i * dim1, tensor_A + i * n1 + slice_1_0, sizeof(double) * dim1);
    }
}

void fn_slice_3_2(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int slice_2_0,
    const int slice_2_1)
{
    int dim2 = slice_2_1 - slice_2_0;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        int i = ij / n1;
        int j = ij % n1;
        memcpy(tensor_B + ij * dim2, tensor_A + i * n1 * n2 + j * n2 + slice_2_0, sizeof(double) * dim2);
    }
}

void fn_slice_3_0_2(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int slice_0_0,
    const int slice_0_1,
    const int slice_2_0,
    const int slice_2_1)
{
    int dim0 = slice_0_1 - slice_0_0;
    int dim2 = slice_2_1 - slice_2_0;

#pragma omp parallel for
    for (size_t i = slice_0_0; i < slice_0_1; i++)
    {
        for (size_t j = 0; j < n1; j++)
        {
            memcpy(tensor_B + (i - slice_0_0) * n1 * dim2 + j * dim2,
                   tensor_A + i * n1 * n2 + j * n2 + slice_2_0, sizeof(double) * dim2);
        }
    }
}

void fn_slice_4_0_1_2(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int slice_0_0,
    const int slice_0_1,
    const int slice_1_0,
    const int slice_1_1,
    const int slice_2_0,
    const int slice_2_1)
{
    int dim1 = slice_1_1 - slice_1_0;
    int dim2 = slice_2_1 - slice_2_0;

#pragma omp parallel for
    for (size_t i = slice_0_0; i < slice_0_1; i++)
    {
        for (size_t j = slice_1_0; j < slice_1_1; j++)
        {
            memcpy(tensor_B + (i - slice_0_0) * dim1 * dim2 * n3 + (j - slice_1_0) * dim2 * n3,
                   tensor_A + i * n1 * n2 * n3 + j * n2 * n3 + slice_2_0 * n3, sizeof(double) * dim2 * n3);
        }
    }
}

void fn_slice_3_1_2(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int slice_1_0,
    const int slice_1_1,
    const int slice_2_0,
    const int slice_2_1)
{
    int dim1 = slice_1_1 - slice_1_0;
    int dim2 = slice_2_1 - slice_2_0;

#pragma omp parallel for
    for (size_t i = 0; i < n0; i++)
    {
        for (size_t j = slice_1_0; j < slice_1_1; j++)
        {
            memcpy(tensor_B + i * dim1 * dim2 + (j - slice_1_0) * dim2,
                   tensor_A + i * n1 * n2 + j * n2 + slice_2_0, sizeof(double) * dim2);
        }
    }
}

void fn_slice_4_1_2(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int slice_1_0,
    const int slice_1_1,
    const int slice_2_0,
    const int slice_2_1)
{
    int dim1 = slice_1_1 - slice_1_0;
    int dim2 = slice_2_1 - slice_2_0;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        for (size_t j = slice_1_0; j < slice_1_1; j++)
        {
            memcpy(tensor_B + i * dim1 * dim2 * n3 + (j - slice_1_0) * dim2 * n3,
                   tensor_A + i * n1 * n2 * n3 + j * n2 * n3 + slice_2_0 * n3, sizeof(double) * dim2 * n3);
        }
    }
}

void fn_slice_3_0_1(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int slice_0_0,
    const int slice_0_1,
    const int slice_1_0,
    const int slice_1_1)
{
    int dim0 = slice_0_1 - slice_0_0;
    int dim1 = slice_1_1 - slice_1_0;

#pragma omp parallel for schedule(static)
    for (size_t i = slice_0_0; i < slice_0_1; i++)
    {
        for (size_t j = slice_1_0; j < slice_1_1; j++)
        {
            memcpy(tensor_B + (i - slice_0_0) * dim1 * n2 + (j - slice_1_0) * n2,
                   tensor_A + i * n1 * n2 + j * n2, sizeof(double) * n2);
        }
    }
}

/// packadd ///

void fn_packadd_3_1_2(
    double *tensor_A,
    const double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int slice_1_0,
    const int slice_1_1,
    const int slice_2_0,
    const int slice_2_1)
{
    int dim1 = slice_1_1 - slice_1_0;
    int dim2 = slice_2_1 - slice_2_0;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        for (size_t j = slice_1_0; j < slice_1_1; j++)
        {
            for (size_t k = slice_2_0; k < slice_2_1; k++)
            {
                tensor_A[i * n1 * n2 + j * n2 + k] += tensor_B[i * dim1 * dim2 + (j - slice_1_0) * dim2 + (k - slice_2_0)];
                // printf("tensor_A[%d,%d,%d] = %f\n", i, j, k, tensor_A[i * n1 * n2 + j * n2 + k]);
            }
        }
    }
}

void fn_packadd_3_1(
    double *tensor_A,
    const double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int slice_1_0,
    const int slice_1_1)
{
    int dim1 = slice_1_1 - slice_1_0;

    static const int INCX = 1;
    static const double ALPHA = 1;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        for (size_t j = slice_1_0; j < slice_1_1; j++)
        {
            daxpy_(&n2, &ALPHA, tensor_B + i * dim1 * n2 + (j - slice_1_0) * n2, &INCX, tensor_A + i * n1 * n2 + j * n2, &INCX);
        }
    }
}

void fn_copy(
    const double *tensor_A,
    double *tensor_B,
    const int size)
{
    if (tensor_A != tensor_B)
    {
        memcpy(tensor_B, tensor_A, sizeof(double) * size);
    }
}

void fn_add(
    const double *tensor_A,
    double *tensor_B,
    const int size)
{
    static const int INCX = 1;
    static const double ALPHA = 1;

    const int nthread = get_omp_threads();
    const int bunch_size = size / nthread + 1;

    if (size < 1024)
    {
        daxpy_(&size, &ALPHA, tensor_A, &INCX, tensor_B, &INCX);
        return;
    }

#pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        int start = ithread * bunch_size;
        int end = start + bunch_size;
        start = start > size ? size : start;
        end = end > size ? size : end;
        const int n = end - start;

        if (n > 0)
        {
            daxpy_(&n, &ALPHA, tensor_A + start, &INCX, tensor_B + start, &INCX);
        }
    }
}

void fn_clean(
    double *tensor_A,
    const int size)
{
    memset(tensor_A, 0, sizeof(double) * size);
}