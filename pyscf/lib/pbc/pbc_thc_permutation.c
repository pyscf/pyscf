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

///// various permutation functions /////

/// NOTE: for permutation A can be equal to B

void fn_permutation_01_10(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    double *buffer)
{
    static const int INCX = 1;
    int nthread = get_omp_threads();
#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        size_t ind_A = i * n1;
        dcopy_(&n1, tensor_A + ind_A, &INCX, buffer + i, &n0);
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1);
}

void fn_permutation_0123_1230(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0, n1 * n2 * n3, buffer);
}

void fn_permutation_012_120(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0, n1 * n2, buffer);
}

void fn_permutation_012_201(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0 * n1, n2, buffer);
}

void fn_permutation_0123_2301(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0 * n1, n2 * n3, buffer);
}

void fn_permutation_01234_23401(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0 * n1, n2 * n3 * n4, buffer);
}

void fn_permutation_01234_02134(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_0213(tensor_A, tensor_B, n0, n1, n2, n3 * n4, buffer);
}

void fn_permutation_01234_14230(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_1320(tensor_A, tensor_B, n0, n1, n2 * n3, n4, buffer);
}

void fn_permutation_01234_34012(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0 * n1 * n2, n3 * n4, buffer);
}

void fn_permutation_012_210(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    int nthread = get_omp_threads();

    int nkj = n2 * n1;

    if (nthread > nkj)
    {
        nthread = nkj;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t kj = 0; kj < nkj; kj++)
    {
        size_t k = kj / n1;
        size_t j = kj % n1;

        for (size_t i = 0; i < n0; i++)
        {
            buffer[k * n1 * n0 + j * n0 + i] = tensor_A[i * n1 * n2 + j * n2 + k];
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_permutation_0123_0321(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    for (int i = 0; i < n0; ++i)
    {
        fn_permutation_012_210(tensor_A + i * n1 * n2 * n3, tensor_B + i * n1 * n2 * n3, n1, n2, n3, buffer);
    }
}

void fn_permutation_012_021(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    int nthread = get_omp_threads();

    if (nthread > n0)
    {
        nthread = n0;
    }

#pragma omp parallel num_threads(nthread)
    {
        int thread_id = omp_get_thread_num();

        double *local_buffer = buffer + thread_id * n1 * n2;

#pragma omp for schedule(static)
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_A = i * n1 * n2;
            for (size_t j = 0; j < n1; j++)
            {
                for (size_t k = 0; k < n2; k++, ind_A++)
                {
                    local_buffer[k * n1 + j] = tensor_A[ind_A];
                }
            }
            memcpy(tensor_B + i * n1 * n2, local_buffer, sizeof(double) * n1 * n2);
        }
    }
}

void fn_permutation_012_021_wob(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2)
{
    static const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        size_t ind_A = i * n1 * n2;
        size_t ind_B = i * n1 * n2;
        for (size_t j = 0; j < n1; j++, ind_B++, ind_A += n2)
        {
            dcopy_(&n2, tensor_A + ind_A, &INCX, tensor_B + ind_B, &n1);
        }
    }
}

void fn_permutation_0123_0312(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0, n1 * n2, n3, buffer);
}

void fn_permutation_0123_0132(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0 * n1, n2, n3, buffer);
}

fn_permutation_01234_01342(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0 * n1, n2, n3 * n4, buffer);
}

void fn_permutation_01234_01243(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0 * n1 * n2, n3, n4, buffer);
}

void fn_permutation_01234_01423(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0 * n1, n2 * n3, n4, buffer);
}

void fn_permutation_012_102(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_B = (j * n0 + i) * n2;
        memcpy(buffer + ind_B, tensor_A + ij * n2, sizeof(double) * n2);
    }
    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_permutation_012_102_wob(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_B = (j * n0 + i) * n2;
        memcpy(tensor_B + ind_B, tensor_A + ij * n2, sizeof(double) * n2);
    }
}

void fn_permutation_01234_10234(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_102(tensor_A, tensor_B, n0, n1, n2 * n3 * n4, buffer);
}

void fn_permutation_0123_1023(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_012_102(tensor_A, tensor_B, n0, n1, n2 * n3, buffer);
}

void fn_permutation_01234_20134(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_102(tensor_A, tensor_B, n0 * n1, n2, n3 * n4, buffer);
}

void fn_permutation_01234_23014(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_102(tensor_A, tensor_B, n0 * n1, n2 * n3, n4, buffer);
}

void fn_permutation_0123_0231(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0, n1, n2 * n3, buffer);
}

void fn_permutation_01234_02341(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_021(tensor_A, tensor_B, n0, n1, n2 * n3 * n4, buffer);
}

void fn_permutation_01234_12340(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_01_10(tensor_A, tensor_B, n0, n1 * n2 * n3 * n4, buffer);
}

void fn_permutation_0123_1203(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_012_102(tensor_A, tensor_B, n0, n1 * n2, n3, buffer);
}

void fn_permutation_01234_34120(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_210(tensor_A, tensor_B, n0, n1 * n2, n3 * n4, buffer);
}

void fn_permutation_0123_1032(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                buffer[j * n0 * n3 * n2 + i * n3 * n2 + l * n2 + k] = tensor_A[ij * n2 * n3 + k * n3 + l];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_01234_02143(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    for (int i = 0; i < n0; ++i)
    {
        fn_permutation_0123_1032(tensor_A + i * n1 * n2 * n3 * n4, tensor_B + i * n1 * n2 * n3 * n4, n1, n2, n3, n4, buffer);
    }
}

void fn_permutation_01234_12043(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_1032(tensor_A, tensor_B, n0, n1 * n2, n3, n4, buffer);
}

void fn_permutation_0123_1320(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        size_t ind_A = ij * n2 * n3;
        size_t ind_B = j * n3 * n2 * n0 + i;
        for (size_t l = 0; l < n3; l++, ind_B += n2 * n0, ind_A++)
        {
            dcopy_(&n2, tensor_A + ind_A, &n3, buffer + ind_B, &n0);
            // for (size_t l = 0; l < n3; l++, ind_A++)
            // {
            //     buffer[ind_B + (l * n2 + k) * n0] = tensor_A[ind_A];
            // }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_0123_1320_wob(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        size_t ind_A = ij * n2 * n3;
        size_t ind_B = j * n3 * n2 * n0 + i;
        for (size_t l = 0; l < n3; l++, ind_B += n2 * n0, ind_A++)
        {
            dcopy_(&n2, tensor_A + ind_A, &n3, tensor_B + ind_B, &n0);
        }
    }
}

void fn_permutation_0123_1302(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                buffer[j * n0 * n3 * n2 + l * n0 * n2 + i * n2 + k] = tensor_A[ij * n2 * n3 + k * n3 + l];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_0123_3102(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                buffer[l * n0 * n1 * n2 + j * n0 * n2 + i * n2 + k] = tensor_A[ij * n2 * n3 + k * n3 + l];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_01234_34102(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_3102(tensor_A, tensor_B, n0, n1, n2, n3 * n4, buffer);
}

void fn_permutation_01234_14023(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_1302(tensor_A, tensor_B, n0, n1, n2 * n3, n4, buffer);
}

void fn_permutation_0123_3210(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                buffer[l * n0 * n1 * n2 + k * n0 * n1 + j * n0 + i] = tensor_A[ij * n2 * n3 + k * n3 + l];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_0123_3021(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_A++)
            {
                buffer[l * n0 * n1 * n2 + i * n1 * n2 + k * n1 + j] = tensor_A[ind_A];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_01234_34021(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_3021(tensor_A, tensor_B, n0, n1, n2, n3 * n4, buffer);
}

void fn_permutation_0123_2013(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;
        size_t ind_B_tmp = ij * n3;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_A++)
            {
                buffer[ind_B_tmp + l + k * n0 * n1 * n3] = tensor_A[ind_A];
            }
        }
    }
    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_0123_0213(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{

    int nthread = get_omp_threads();
    if (nthread > n0)
    {
        nthread = n0;
    }

#pragma omp parallel num_threads(nthread)
    {
        int thread_id = omp_get_thread_num();

        double *local_buffer = buffer + thread_id * n1 * n2 * n3;

#pragma omp for schedule(static)
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_A = i * n1 * n2 * n3;
            for (size_t j = 0; j < n1; j++)
            {
                for (size_t k = 0; k < n2; k++)
                {
                    size_t ind_B = (k * n1 + j) * n3;
                    for (size_t l = 0; l < n3; l++, ind_A++, ind_B++)
                    {
                        local_buffer[ind_B] = tensor_A[ind_A];
                    }
                }
            }
            memcpy(tensor_B + i * n1 * n2 * n3, local_buffer, sizeof(double) * n1 * n2 * n3);
        }
    }
}

void fn_permutation_01234_02314(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_0213(tensor_A, tensor_B, n0, n1, n2 * n3, n4, buffer);
}

void fn_permutation_01234_31402(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3 * n4;
        size_t ind_B_tmp = i * n2 + j * n0 * n2 * n4;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                for (size_t m = 0; m < n4; m++, ind_A++)
                {
                    buffer[ind_B_tmp + k + l * n1 * n4 * n0 * n2 + m * n0 * n2] = tensor_A[ind_A];
                }
            }
        }
    }
    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_permutation_0123_2130(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;
        size_t ind_B_tmp = j * n0 * n3 + i;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_A++)
            {
                buffer[ind_B_tmp + (k * n1 * n3 + l) * n0] = tensor_A[ind_A];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_01234_43201(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_3210(tensor_A, tensor_B, n0 * n1, n2, n3, n4, buffer);
}

void fn_permutation_01234_30124(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_2013(tensor_A, tensor_B, n0, n1 * n2, n3, n4, buffer);
}

void fn_permutation_0123_3012(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_permutation_012_201(tensor_A, tensor_B, n0, n1 * n2, n3, buffer);
}

void fn_permutation_01234_23140(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_2130(tensor_A, tensor_B, n0, n1, n2 * n3, n4, buffer);
}

void fn_permutation_01234_32401(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_2130(tensor_A, tensor_B, n0 * n1, n2, n3, n4, buffer);
}

void fn_permutation_01234_04312(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_0321(tensor_A, tensor_B, n0, n1 * n2, n3, n4, buffer);
}

void fn_permutation_01234_03412(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_021(
        tensor_A, tensor_B, n0, n1 * n2, n3 * n4, buffer);
}

void fn_permutation_01234_40231(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_3021(tensor_A, tensor_B, n0, n1, n2 * n3, n4, buffer);
}

void fn_permutation_01234_23410(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_012_210(tensor_A, tensor_B, n0, n1, n2 * n3 * n4, buffer);
}

void fn_permutation_0123_2031(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;
        size_t ind_B_tmp = i * n3 * n1 + j;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_A++)
            {
                buffer[ind_B_tmp + (k * n0 * n3 + l) * n1] = tensor_A[ind_A];
            }
        }
    }
    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_01234_20341(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_2031(tensor_A, tensor_B, n0, n1, n2, n3 * n4, buffer);
}

void fn_permutation_01234_01324(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_0213(tensor_A, tensor_B, n0 * n1, n2, n3, n4, buffer);
}

void fn_permutation_0123_2310(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;
        size_t ind_B_tmp = j * n0 + i;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_A++)
            {
                buffer[ind_B_tmp + (k * n3 + l) * n0 * n1] = tensor_A[ind_A];
            }
        }
    }
    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_permutation_01234_03421(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_0321(tensor_A, tensor_B, n0, n1, n2, n3 * n4, buffer);
}

void fn_permutation_01234_03124(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_permutation_0123_0213(tensor_A, tensor_B, n0, n1 * n2, n3, n4, buffer);
}

void fn_permutation_01234_24130(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3 * n4;
        size_t ind_B_tmp = j * n0 * n3 + i;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                for (size_t m = 0; m < n4; m++, ind_A++)
                {
                    buffer[ind_B_tmp + k * n1 * n3 * n0 * n4 + l * n0 + m * n0 * n3 * n1] = tensor_A[ind_A];
                }
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_permutation_01234_24031(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < n0 * n1; ij++)
    {
        int i = ij / n1;
        int j = ij % n1;
        size_t ind_A = ij * n2 * n3 * n4;
        for (int k = 0; k < n2; k++)
        {
            for (int l = 0; l < n3; l++)
            {
                for (int m = 0; m < n4; m++, ind_A++)
                {
                    int ind_B = k * n4 * n0 * n1 * n3 + m * n0 * n1 * n3 + i * n1 * n3 + l * n1 + j;
                    buffer[ind_B] = tensor_A[ind_A];
                }
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_permutation_01234_03241(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < n0 * n1; ij++)
    {
        int i = ij / n1;
        int j = ij % n1;
        size_t ind_A = ij * n2 * n3 * n4;
        size_t ind_B = i * n1 * n2 * n3 * n4 + j;
        for (int k = 0; k < n2; k++)
        {
            for (int l = 0; l < n3; l++)
            {
                for (int m = 0; m < n4; m++, ind_A++)
                {
                    int ind_B2 = ind_B + ((l * n2 + k) * n4 + m) * n1;
                    buffer[ind_B2] = tensor_A[ind_A];
                }
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}