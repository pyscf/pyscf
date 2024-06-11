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

void fn_permutation_01_10(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    double *buffer);
void fn_permutation_012_210(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer);
void fn_permutation_012_021(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    const int n2,
    double *buffer);

///////////// the following linear algebra functions are used in THC-posthf /////////////

void fn_contraction_0_0_0(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        tensor_C[i] = tensor_A[i] * tensor_B[i];
    }
}

void fn_contraction_012_012_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    fn_contraction_0_0_0(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, buffer);
}

void fn_contraction_012_012_012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
    fn_contraction_012_012_012(tensor_A, tensor_B, tensor_C, n0, n1, n2, NULL);
}

void fn_contraction_0123_0123_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_0_0_0(tensor_A, tensor_B, tensor_C, n0 * n1 * n2 * n3, buffer);
}

void fn_contraction_01_21_021(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    size_t size_A = n0 * n1;
    size_t size_B = n1 * n2;
    size_t size_C = n0 * n1 * n2;

    bool overwrite_A = ((tensor_A >= tensor_C) && (tensor_A < (tensor_C + size_C))) || ((tensor_C >= tensor_A) && (tensor_C < (tensor_A + size_A)));
    bool overwrite_B = ((tensor_B >= tensor_C) && (tensor_B < (tensor_C + size_C))) || ((tensor_C >= tensor_B) && (tensor_C < (tensor_B + size_B)));

    // 01 -> 10

    fn_permutation_01_10(tensor_A, (double *)tensor_A, n0, n1, buffer);

    // 21 -> 12

    if (tensor_A != tensor_B)
    {
        fn_permutation_01_10(tensor_B, (double *)tensor_B, n2, n1, buffer);
    }

    // 10 * 12 -> 102

    fn_contraction_01_02_012(tensor_A, tensor_B, tensor_C, n1, n0, n2, buffer);

    // 102 -> 021

    fn_permutation_01_10(tensor_C, (double *)tensor_C, n1, n0 * n2, buffer);

    if (tensor_A == tensor_B)
    {
        // assert(overwrite_A == overwrite_B);
        if (!overwrite_A)
        {
            fn_permutation_01_10(tensor_A, (double *)tensor_A, n1, n0, buffer);
        }
    }
    else
    {
        if (!overwrite_A)
        {
            fn_permutation_01_10(tensor_A, (double *)tensor_A, n1, n0, buffer);
        }

        if (!overwrite_B)
        {
            fn_permutation_01_10(tensor_B, (double *)tensor_B, n1, n2, buffer);
        }
    }
}

void fn_contraction_01_20_120_slow(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t jk = 0; jk < n1 * n2; jk++)
    {
        size_t j = jk / n2;
        size_t k = jk % n2;
        for (size_t i = 0; i < n0; i++)
        {
            buffer[jk * n0 + i] = tensor_A[i * n1 + j] * tensor_B[jk];
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01_20_120(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    if (tensor_A == tensor_B)
    {
        fn_contraction_01_20_120_slow(tensor_A, tensor_B, tensor_C, n0, n1, n2, buffer);
        return;
    }

    size_t size_B = n0 * n2;
    size_t size_C = n0 * n1 * n2;

    bool overwrite_B = ((tensor_B >= tensor_C) && (tensor_B < (tensor_C + size_C))) || ((tensor_C >= tensor_B) && (tensor_C < (tensor_B + size_B)));

    // 20 -> 02

    fn_permutation_01_10(tensor_B, (double *)tensor_B, n2, n0, buffer);

    // 01 * 02 -> 012

    fn_contraction_01_02_012(tensor_A, tensor_B, tensor_C, n0, n1, n2, buffer);

    // 012 -> 120

    fn_permutation_01_10(tensor_C, (double *)tensor_C, n0, n1 * n2, buffer);

    if (!overwrite_B)
    {
        fn_permutation_01_10(tensor_B, (double *)tensor_B, n0, n2, buffer);
    }
}

void fn_contraction_01_20_120_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
#pragma omp parallel for schedule(static)
    for (size_t jk = 0; jk < n1 * n2; jk++)
    {
        size_t j = jk / n2;
        size_t k = jk % n2;
        size_t ind_A = j;
        size_t ind_B = k * n0;
        size_t ind_C = jk * n0;
        for (size_t i = 0; i < n0; i++, ind_C++, ind_A += n1, ind_B++)
        {
            tensor_C[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
        }
    }
}

void fn_contraction_01_02341_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_201(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_012_2304_13402(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    static const int INCX = 1;

    int nik = n0 * n2;

    memset(buffer, 0, sizeof(double) * n1 * n3 * n4 * n0 * n2);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        for (size_t k = 0; k < n2; ++k)
        {
            double A_tmp = tensor_A[ij * n2 + k];
            for (size_t l = 0; l < n3; l++)
            {
                size_t ind_B = ((k * n3 + l) * n0 + i) * n4;
                size_t ind_C = ((j * n3 + l) * n4 * n0 + i) * n2 + k;
                daxpy_(&n4, &A_tmp, tensor_B + ind_B, &INCX, buffer + ind_C, &nik);
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_01_02_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    const double ALPHA = 1.0;
    const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        dger_(&n2, &n1, &ALPHA, tensor_B + i * n2, &INCX, tensor_A + i * n1, &INCX, buffer + i * n1 * n2, &n2);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01_02_012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
    memset(tensor_C, 0, sizeof(double) * n0 * n1 * n2);

    const double ALPHA = 1.0;
    const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        dger_(&n2, &n1, &ALPHA, tensor_B + i * n2, &INCX, tensor_A + i * n1, &INCX, tensor_C + i * n1 * n2, &n2);
    }
}

void fn_contraction_01_02_120(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    fn_contraction_01_02_012(tensor_A, tensor_B, tensor_C, n0, n1, n2, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0, n1 * n2, buffer);
}

void fn_contraction_01_02_120_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
#pragma omp parallel for schedule(static)
    for (size_t jk = 0; jk < n1 * n2; jk++)
    {
        size_t j = jk / n2;
        size_t k = jk % n2;
        size_t ind_A = j;
        size_t ind_B = k;
        size_t ind_C = jk * n0;
        for (size_t i = 0; i < n0; i++, ind_A += n1, ind_B += n2, ind_C++)
        {
            tensor_C[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
        }
    }
}

void fn_contraction_01_023_1230(const double *tensor_A,
                                const double *tensor_B,
                                double *tensor_C,
                                const int n0,
                                const int n1,
                                const int n2,
                                const int n3,
                                double *buffer)
{
    fn_contraction_01_02_120(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_023_1230_wob(const double *tensor_A,
                                    const double *tensor_B,
                                    double *tensor_C,
                                    const int n0,
                                    const int n1,
                                    const int n2,
                                    const int n3)
{
    fn_contraction_01_02_120_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3);
}

void fn_contraction_01_12_120(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    // memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (int jk = 0; jk < n1 * n2; jk++)
    {
        int j = jk / n2;
        int k = jk % n2;
        for (int i = 0; i < n0; i++)
        {
            buffer[jk * n0 + i] = tensor_A[i * n1 + j] * tensor_B[jk];
        }
    }
}

void fn_contraction_01_12_021(const double *tensor_A,
                              const double *tensor_B,
                              double *tensor_C,
                              const int n0,
                              const int n1,
                              const int n2,
                              double *buffer)
{

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (int jk = 0; jk < n1 * n2; jk++)
    {
        int j = jk / n2;
        int k = jk % n2;
        for (int i = 0; i < n0; i++)
        {
            buffer[i * n1 * n2 + k * n1 + j] = tensor_A[i * n1 + j] * tensor_B[jk];
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01_12_021_wob(const double *tensor_A,
                                  const double *tensor_B,
                                  double *tensor_C,
                                  const int n0,
                                  const int n1,
                                  const int n2)
{

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (int jk = 0; jk < n1 * n2; jk++)
    {
        int j = jk / n2;
        int k = jk % n2;
        for (int i = 0; i < n0; i++)
        {
            tensor_C[i * n1 * n2 + k * n1 + j] = tensor_A[i * n1 + j] * tensor_B[jk];
        }
    }
}

void fn_contraction_01_123_0231(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_12_021(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_123_0231_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    fn_contraction_01_12_021_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3);
}

void fn_contraction_01_0234_12340(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_02_120(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_01_0234_12340_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4)
{
    fn_contraction_01_02_120_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4);
}

void _fn_contraction_01_2031_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    static const int INCX = 1;

    memset(tensor_C, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        size_t ind_B = i * n3 * n1 + j;
        size_t ind_C = ij * n2 * n3;

        for (size_t k = 0; k < n2; k++, ind_C += n3, ind_B += n0 * n1 * n3)
        {
            daxpy_(&n3, tensor_A + ij, tensor_B + ind_B, &n1, tensor_C + ind_C, &INCX);
        }
    }
    // memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void _fn_permutation_01_10(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1)
{
    static const int INCX = 1;
    int nthread = get_omp_threads();
#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        size_t ind_A = i * n1;
        dcopy_(&n1, tensor_A + ind_A, &INCX, tensor_B + i, &n0);
    }
}

void fn_contraction_01_2031_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t kl = 0; kl < n2 * n3; kl++)
    {
        size_t k = kl / n3;
        size_t l = kl % n3;
        size_t ind_B = ((k * n0 * n3) + l) * n1;
        size_t ind_C = kl * n0 * n1;
        size_t ind_A = 0;
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_B2 = ind_B + i * n3 * n1;
            for (size_t j = 0; j < n1; j++, ind_A++, ind_C++, ind_B2++)
            {
                buffer[ind_C] = tensor_A[ind_A] * tensor_B[ind_B2];
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01_2031_2301_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
#pragma omp parallel for schedule(static)
    for (size_t kl = 0; kl < n2 * n3; kl++)
    {
        size_t k = kl / n3;
        size_t l = kl % n3;
        size_t ind_B = ((k * n0 * n3) + l) * n1;
        size_t ind_C = kl * n0 * n1;
        size_t ind_A = 0;
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_B2 = ind_B + i * n3 * n1;
            for (size_t j = 0; j < n1; j++, ind_A++, ind_C++, ind_B2++)
            {
                tensor_C[ind_C] = tensor_A[ind_A] * tensor_B[ind_B2];
            }
        }
    }
}

void fn_contraction_01_230_1230(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_20_120(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_230_1230_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    fn_contraction_01_20_120_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3);
}

void fn_contraction_01_01_0(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    double *buffer)
{
    memset(buffer, 0, sizeof(double) * n0);

    int nthread = get_omp_threads();

    const int INCX = 1;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        buffer[i] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + i * n1, &INCX);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0);
}

void fn_contraction_01_01_0_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1)
{
    int nthread = get_omp_threads();

    const int INCX = 1;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        tensor_C[i] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + i * n1, &INCX);
    }
}

void fn_contraction_01_01_0_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    double *buffer)
{
    memset(buffer, 0, sizeof(double) * n0);

    int nthread = get_omp_threads();

    const int INCX = 1;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        tensor_C[i] += ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + i * n1, &INCX);
    }

    // memcpy(tensor_C, buffer, sizeof(double) * n0);
}

void fn_contraction_01_2134_02341(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_213_0231(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_01234_01234_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_01_0(tensor_A, tensor_B, tensor_C, n0 * n1 * n2 * n3, n4, buffer);
}

void fn_contraction_01234_01234_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_01_0(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, n3 * n4, buffer);
}

void fn_contraction_012_032_013(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

    static const char NOTRANS = 'N';
    static const double ALPHA = 1.0;
    static const double BETA = 0.0;
    static const char TRANS = 'T';

    memset(buffer, 0, sizeof(double) * n0 * n1 * n3);

    if (n0 > nthread * 2)
    {
#pragma omp parallel num_threads(nthread)
        {
            int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < n0; i++)
            {
                dgemm_(&TRANS,
                       &NOTRANS,
                       &n3, &n1, &n2,
                       &ALPHA,
                       tensor_B + i * n3 * n2, &n2,
                       tensor_A + i * n1 * n2, &n2,
                       &BETA, buffer + i * n1 * n3, &n3);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < n0; i++)
        {
            /// call parallel dgemm

            NPdgemm(TRANS,
                    NOTRANS,
                    n3, n1, n2,
                    n2, n2, n3,
                    0, 0, 0,
                    tensor_B + i * n2 * n3,
                    tensor_A + i * n1 * n2,
                    buffer + i * n1 * n3,
                    ALPHA, BETA);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n3);
}

void fn_contraction_0123_0453_01245(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    const int n5,
    double *buffer)
{
    fn_contraction_012_032_013(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3, n4 * n5, buffer);
}

void fn_contraction_01_20341_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_2031_2301(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_01_021_021(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    if ((double *)tensor_B == tensor_C)
    {

#pragma omp parallel for schedule(static)
        for (size_t ik = 0; ik < n0 * n2; ik++)
        {
            size_t i = ik / n2;
            for (size_t j = 0; j < n1; j++)
            {
                tensor_C[ik * n1 + j] = tensor_A[i * n1 + j] * tensor_B[ik * n1 + j];
            }
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (size_t ik = 0; ik < n0 * n2; ik++)
        {
            size_t i = ik / n2;
            for (size_t j = 0; j < n1; j++)
            {
                buffer[ik * n1 + j] = tensor_A[i * n1 + j] * tensor_B[ik * n1 + j];
            }
        }
        memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
    }
}

void fn_contraction_012_01342_01342(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_021(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_01_021_02(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        for (size_t j = 0; j < n1; j++)
        {
            buffer[ik] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + ik * n1, &INCX);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n2);
}

void fn_contraction_01_021_02_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
    static const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        for (size_t j = 0; j < n1; j++)
        {
            tensor_C[ik] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + ik * n1, &INCX);
        }
    }
}

void fn_contraction_01_021_02_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    int nthread = get_omp_threads();
    int benchsize = n0 * n2 / nthread + 1;

#pragma omp parallel num_threads(nthread)
    {
#pragma omp for schedule(static)
        for (size_t ik = 0; ik < n0 * n2; ik++)
        {
            size_t i = ik / n2;
            for (size_t j = 0; j < n1; j++)
            {
                buffer[ik] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + ik * n1, &INCX);
            }
        }

        int thread_id = omp_get_thread_num();
        int start = thread_id * benchsize;
        int end = (thread_id + 1) * benchsize;

        start = start < n0 * n2 ? start : n0 * n2;
        end = end < n0 * n2 ? end : n0 * n2;

        int bunchsize_tmp = end - start;

        daxpy_(&bunchsize_tmp, &ALPHA, buffer + start, &INCX, tensor_C + start, &INCX);
    }
}

void fn_contraction_01_021_02_plus_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    int nthread = get_omp_threads();
    int benchsize = n0 * n2 / nthread + 1;

#pragma omp parallel num_threads(nthread)
    {
#pragma omp for schedule(static)
        for (size_t ik = 0; ik < n0 * n2; ik++)
        {
            size_t i = ik / n2;
            // for (size_t j = 0; j < n1; j++)
            // {
            tensor_C[ik] += ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + ik * n1, &INCX);
            // }
        }
    }
}

void fn_contraction_012_01342_0134(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_02(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_01342_0134_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4)
{
    fn_contraction_01_021_02_wob(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4);
}

void fn_contraction_012_01342_0134_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_02_plus(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_02_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

    if ((double *)tensor_A == tensor_C)
    {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (size_t ij = 0; ij < nij; ij++)
        {
            size_t i = ij / n1;

            for (size_t k = 0; k < n2; k++)
            {
                tensor_C[ij * n2 + k] = tensor_A[ij * n2 + k] * tensor_B[i * n2 + k];
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (size_t ij = 0; ij < nij; ij++)
        {
            size_t i = ij / n1;

            for (size_t k = 0; k < n2; k++)
            {
                buffer[ij * n2 + k] = tensor_A[ij * n2 + k] * tensor_B[i * n2 + k];
            }
        }
        memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
    }
}

void fn_contraction_01234_0134_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_02_012(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_021_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

    if ((double *)tensor_A == tensor_C)
    {

#pragma omp parallel for num_threads(nthread) schedule(static)
        for (size_t ij = 0; ij < nij; ij++)
        {
            size_t i = ij / n1;
            size_t j = ij % n1;

            for (size_t k = 0; k < n2; k++)
            {
                tensor_C[ij * n2 + k] = tensor_A[ij * n2 + k] * tensor_B[i * n2 * n1 + k * n1 + j];
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (size_t ij = 0; ij < nij; ij++)
        {
            size_t i = ij / n1;
            size_t j = ij % n1;

            for (size_t k = 0; k < n2; k++)
            {
                buffer[ij * n2 + k] = tensor_A[ij * n2 + k] * tensor_B[i * n2 * n1 + k * n1 + j];
            }
        }
        memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
    }
}

void fn_contraction_01234_01423_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_021_012(tensor_A, tensor_B, tensor_C, n0 * n1, n2 * n3, n4, buffer);
}

void fn_contraction_01_012_02(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    size_t nik = n0 * n2;

    int nthread = get_omp_threads();

    if (nthread > nik)
    {
        nthread = nik;
    }

    static const int INCX = 1;
    const int INCY = n2;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ik = 0; ik < nik; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        for (size_t j = 0; j < n1; j++)
        {
            buffer[ik] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + i * n1 * n2 + k, &INCY);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n2);
}

void fn_contraction_01_012_02_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
    size_t nik = n0 * n2;

    int nthread = get_omp_threads();

    if (nthread > nik)
    {
        nthread = nik;
    }

    static const int INCX = 1;
    const int INCY = n2;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ik = 0; ik < nik; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        for (size_t j = 0; j < n1; j++)
        {
            tensor_C[ik] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + i * n1 * n2 + k, &INCY);
        }
    }
}

void fn_contraction_01234_0154_01235(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    const int n5,
    double *buffer)
{
    fn_contraction_012_032_013(tensor_A, tensor_B, tensor_C, n0 * n1, n2 * n3, n4, n5, buffer);
}

void fn_contraction_0123_01423_014(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_02(tensor_A, tensor_B, tensor_C, n0 * n1, n2 * n3, n4, buffer);
}

void fn_contraction_01_0231_023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_021_02(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_0231_023_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    fn_contraction_01_021_02_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3);
}

void fn_contraction_012_02_01(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

    static const int INCX = 1;

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        // size_t j = ij % n1;

        buffer[ij] = ddot_(&n2, tensor_A + ij * n2, &INCX, tensor_B + i * n2, &INCX);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1);
}

void fn_contraction_01234_0134_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_02_01(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_0342_0134(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_032_013(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_031_023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nthread = get_omp_threads();

    static const char TRANS = 'T';
    static const double ALPHA = 1.0;
    static const double BETA = 0.0;

    memset(buffer, 0, sizeof(double) * n0 * n2 * n3);

    if (n0 > nthread * 2)
    {
#pragma omp parallel num_threads(nthread)
        {
            int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < n0; i++)
            {
                dgemm_(&TRANS,
                       &TRANS,
                       &n3, &n2, &n1,
                       &ALPHA,
                       tensor_B + i * n3 * n1, &n1,
                       tensor_A + i * n1 * n2, &n2,
                       &BETA, buffer + i * n2 * n3, &n3);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < n0; i++)
        {
            /// call parallel dgemm

            NPdgemm(
                TRANS,
                TRANS,
                n3, n2, n1,
                n1, n2, n3,
                0, 0, 0,
                tensor_B + i * n1 * n3,
                tensor_A + i * n1 * n2,
                buffer + i * n2 * n3,
                ALPHA, BETA);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n2 * n3);
}

void fn_contraction_012_0341_0234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_031_023(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_0123_2031_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

    if ((double *)tensor_A == tensor_C)
    {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (size_t ij = 0; ij < nij; ij++)
        {
            size_t i = ij / n1;
            size_t j = ij % n1;

            for (size_t k = 0; k < n2; k++)
            {
                for (size_t l = 0; l < n3; l++)
                {
                    tensor_C[ij * n2 * n3 + k * n3 + l] = tensor_A[ij * n2 * n3 + k * n3 + l] * tensor_B[k * n0 * n3 * n1 + i * n3 * n1 + l * n1 + j];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (size_t ij = 0; ij < nij; ij++)
        {
            size_t i = ij / n1;
            size_t j = ij % n1;

            for (size_t k = 0; k < n2; k++)
            {
                for (size_t l = 0; l < n3; l++)
                {
                    buffer[ij * n2 * n3 + k * n3 + l] = tensor_A[ij * n2 * n3 + k * n3 + l] * tensor_B[k * n0 * n3 * n1 + i * n3 * n1 + l * n1 + j];
                }
            }
        }
        memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
    }
}

void fn_contraction_01234_30412_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_2031_0123(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3, n4, buffer);
}

void fn_contraction_012_0132_3012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            double tmp = tensor_A[ij * n2 + k];
            for (size_t l = 0; l < n3; l++)
            {
                buffer[l * n0 * n1 * n2 + i * n1 * n2 + j * n2 + k] =
                    tmp * tensor_B[i * n1 * n2 * n3 + j * n3 * n2 + l * n2 + k];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_203_1023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                buffer[j * n0 * n2 * n3 + i * n2 * n3 + k * n3 + l] =
                    tensor_A[ij * n2 * n3 + k * n3 + l] * tensor_B[k * n0 * n3 + i * n3 + l];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01_21_021_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
    double *buffer = (double *)malloc(sizeof(double) * n0 * n1 * n2);
    fn_contraction_01_21_021(tensor_A, tensor_B, tensor_C, n0, n1, n2, buffer);
    free(buffer);
}

void fn_contraction_01_231_0231(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_21_021(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_02341_0234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_02(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_0123_142_03412(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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

        size_t indx_C = i * n1 * n2 * n3 * n4 + j * n2;

        for (size_t l = 0; l < n3; ++l)
        {
            for (size_t m = 0; m < n4; ++m)
            {
                size_t indx_A = ij * n2 * n3 + l;
                size_t indx_B = j * n2 * n4 + m * n2;
                size_t indx_C2 = indx_C + (l * n4 + m) * n1 * n2;
                for (size_t k = 0; k < n2; ++k, indx_A += n3, indx_B++, indx_C2++)
                {
                    buffer[indx_C2] = tensor_A[indx_A] * tensor_B[indx_B];
                }
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_0123_0123_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_01_0(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, n3, buffer);
}

void fn_contraction_0123_0123_012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    fn_contraction_01_01_0_wob(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, n3);
}

void fn_contraction_01_20314_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t indx_klm = 0;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                for (size_t m = 0; m < n4; m++)
                {
                    buffer[indx_klm * nij + ij] =
                        tensor_A[ij] * tensor_B[k * n0 * n3 * n1 * n4 + i * n3 * n1 * n4 + l * n1 * n4 + j * n4 + m];

                    indx_klm += 1;
                }
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_012_1032_3012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            double tmp = tensor_A[ij * n2 + k];
            for (size_t l = 0; l < n3; l++)
            {
                buffer[l * n0 * n1 * n2 + i * n1 * n2 + j * n2 + k] =
                    tmp * tensor_B[j * n0 * n2 * n3 + i * n2 * n3 + l * n2 + k];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_013_2013(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

    if (nthread > nij)
    {
        nthread = nij;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t l = 0; l < n3; l++)
        {
            double tmp = tensor_B[ij * n3 + l];
            for (size_t k = 0; k < n2; k++)
            {

                buffer[k * n0 * n1 * n3 + i * n1 * n3 + j * n3 + l] =
                    tensor_A[ij * n2 * n3 + k * n3 + l] * tmp;
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01234_1042_30124(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    const int nijk = n0 * n1 * n2;

    int nthread = get_omp_threads();

    if (nthread > nijk)
    {
        nthread = nijk;
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ijk = 0; ijk < nijk; ijk++)
    {
        size_t i = ijk / (n1 * n2);
        size_t j = (ijk % (n1 * n2)) / n2;
        size_t k = ijk % n2;

        for (size_t l = 0; l < n3; l++)
        {
            for (size_t m = 0; m < n4; m++)
            {
                buffer[l * n0 * n1 * n2 * n4 + ijk * n4 + m] =
                    tensor_A[ijk * n3 * n4 + l * n4 + m] * tensor_B[j * n0 * n4 * n2 + i * n2 * n4 + m * n2 + k];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_0123_01243_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_201(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, n3, n4, buffer);
}

void fn_contraction_012_0213_3012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            double tmp = tensor_A[ij * n2 + k];
            size_t ind_tmp = i * n1 * n2 * n3 + k * n1 * n3 + j * n3;
            size_t ind_tmp2 = i * n1 * n2 + j * n2 + k;
            for (size_t l = 0; l < n3; l++)
            {
                buffer[l * n0 * n1 * n2 + ind_tmp2] = tmp * tensor_B[ind_tmp + l];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_0213_3012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            double tmp = tensor_A[ij * n2 + k];
            size_t ind_tmp = i * n1 * n2 * n3 + k * n1 * n3 + j * n3;
            size_t ind_tmp2 = i * n1 * n2 + j * n2 + k;
            for (size_t l = 0; l < n3; l++)
            {
                tensor_C[l * n0 * n1 * n2 + ind_tmp2] = tmp * tensor_B[ind_tmp + l];
            }
        }
    }
}

void fn_contraction_012_02134_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_0213_3012(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_02134_34012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4)
{
    fn_contraction_012_0213_3012_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4);
}

void fn_contraction_0123_02143_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_tmp = ij * n2 * n3;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_tmp++)
            {
                double tmp = tensor_A[ij * n2 * n3 + k * n3 + l];
                size_t ind_tmp2 = i * n1 * n2 * n3 * n4 + k * n1 * n4 * n3 + j * n4 * n3 + l;
                for (size_t m = 0; m < n4; m++)
                {
                    buffer[m * n0 * n1 * n2 * n3 + ind_tmp] = tmp * tensor_B[ind_tmp2 + m * n3];
                }
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_0123_02143_40123_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4)
{
    size_t nij = n0 * n1;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_tmp = ij * n2 * n3;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_tmp++)
            {
                double tmp = tensor_A[ij * n2 * n3 + k * n3 + l];
                size_t ind_tmp2 = i * n1 * n2 * n3 * n4 + k * n1 * n4 * n3 + j * n4 * n3 + l;
                for (size_t m = 0; m < n4; m++)
                {
                    tensor_C[m * n0 * n1 * n2 * n3 + ind_tmp] = tmp * tensor_B[ind_tmp2 + m * n3];
                }
            }
        }
    }
}

void fn_contraction_0123_021_3012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

    static const int INCX = 1;
    static const double ALPHA = 1.0;

    int INC_C = n0 * n1 * n2;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A_tmp = ij * n2 * n3;
        size_t ind_C_tmp = ij * n2;
        double *B_TMP = tensor_B + i * n1 * n2 + j;

        for (size_t k = 0; k < n2; k++, ind_A_tmp += n3, ind_C_tmp++, B_TMP += n1)
        {
            daxpy_(&n3, B_TMP, tensor_A + ind_A_tmp, &INCX, buffer + ind_C_tmp, &INC_C);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_021_3012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{

    memset(tensor_C, 0, sizeof(double) * n0 * n1 * n2 * n3);

    static const int INCX = 1;
    static const double ALPHA = 1.0;

    int INC_C = n0 * n1 * n2;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A_tmp = ij * n2 * n3;
        size_t ind_C_tmp = ij * n2;
        double *B_TMP = tensor_B + i * n1 * n2 + j;

        for (size_t k = 0; k < n2; k++, ind_A_tmp += n3, ind_C_tmp++, B_TMP += n1)
        {
            daxpy_(&n3, B_TMP, tensor_A + ind_A_tmp, &INCX, tensor_C + ind_C_tmp, &INC_C);
        }
    }
}

void fn_contraction_01_203_1230(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{

    static const int INCX = 1;
    static const double ALPHA = 1.0;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t kl = 0; kl < n2 * n3; kl++)
    {
        size_t k = kl / n3;
        size_t l = kl % n3;

        size_t ind_B_tmp = k * n0 * n3 + l;

        for (size_t j = 0; j < n1; ++j)
        {
            size_t ind_C_tmp = k * n3 * n0 + l * n0 + j * n2 * n3 * n0;
            size_t ind_A_tmp = j;
            for (size_t i = 0; i < n0; i++, ind_C_tmp++, ind_A_tmp += n1)
            {
                buffer[ind_C_tmp] = tensor_A[ind_A_tmp] * tensor_B[ind_B_tmp + i * n3];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01_0213_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    size_t nkl = n2 * n3;

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t kl = 0; kl < nkl; kl++)
    {
        size_t k = kl / n3;
        size_t l = kl % n3;

        size_t ind_A = 0;
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_C_tmp = kl * n0 * n1 + i * n1;
            size_t ind_B_tmp = i * n1 * n2 * n3 + k * n1 * n3 + l;
            for (size_t j = 0; j < n1; j++, ind_C_tmp++, ind_B_tmp += n3, ind_A++)
            {
                buffer[ind_C_tmp] = tensor_A[ind_A] * tensor_B[ind_B_tmp];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01_02314_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_0213_2301(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, n4, buffer);
}

void fn_contraction_01234_0312_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_021_3012(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3, n4, buffer);
}

void fn_contraction_012_034_12340(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_02_120(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3 * n4, buffer);
}

void fn_contraction_01_2034_12340(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_203_1230(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_0123_103_0132(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

    static const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;
        size_t ind_B = j * n0 * n3 + i * n3;
        size_t ind_C = ij * n3 * n2;

        for (size_t l = 0; l < n3; l++, ind_B++, ind_A++, ind_C += n2)
        {
            daxpy_(&n2, tensor_B + ind_B, tensor_A + ind_A, &n3, buffer + ind_C, &INCX);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_103_2013(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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
        size_t ind_B = (j * n0 + i) * n3;
        size_t ind_C = ij * n3;

        for (size_t k = 0; k < n2; k++, ind_C += n0 * n1 * n3)
        {
            size_t ind_B2 = ind_B;
            size_t ind_C2 = ind_C;
            for (size_t l = 0; l < n3; l++, ind_A++, ind_B2++, ind_C2++)
            {
                buffer[ind_C2] = tensor_A[ind_A] * tensor_B[ind_B2];
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_103_2013_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_A = ij * n2 * n3;
        size_t ind_B = (j * n0 + i) * n3;
        size_t ind_C = ij * n3;

        for (size_t k = 0; k < n2; k++, ind_C += n0 * n1 * n3)
        {
            size_t ind_B2 = ind_B;
            size_t ind_C2 = ind_C;
            for (size_t l = 0; l < n3; l++, ind_A++, ind_B2++, ind_C2++)
            {
                tensor_C[ind_C2] = tensor_A[ind_A] * tensor_B[ind_B2];
            }
        }
    }
}

void fn_contraction_012_102_012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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

        size_t B_ind_tmp = j * n0 * n2 + i * n2;
        size_t AC_ind_tmp = ij * n2;

        for (size_t k = 0; k < n2; k++, B_ind_tmp++, AC_ind_tmp++)
        {
            buffer[AC_ind_tmp] = tensor_A[AC_ind_tmp] * tensor_B[B_ind_tmp];
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01234_10234_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_102_012(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_0123_01453_01245(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    const int n5,
    double *buffer)
{
    fn_contraction_012_032_013(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3, n4 * n5, buffer);
}

void fn_contraction_0123_10342_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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

        size_t ind_B = j * n0 * n2 * n3 * n4 + i * n3 * n4 * n2;
        size_t ind_C = ij * n2 * n3;

        for (size_t m = 0; m < n4; ++m)
        {
            size_t indA = ij * n2 * n3;
            size_t ind_C2 = ind_C + m * n0 * n1 * n2 * n3;
            for (size_t k = 0; k < n2; k++)
            {
                for (size_t l = 0; l < n3; l++, indA++, ind_C2++)
                {
                    buffer[ind_C2] = tensor_A[indA] * tensor_B[ind_B + l * n4 * n2 + m * n2 + k];
                }
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_01234_01234_0123_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_01_0_plus(tensor_A, tensor_B, tensor_C, n0 * n1 * n2 * n3, n4, buffer);
}

void fn_contraction_01_213_0231(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        double tmpA = tensor_A[ij];

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = i * n1 * n2 * n3 + k * n3 * n1 + j;
            size_t ind_B = j * n3 + k * n3 * n1;
            daxpy_(&n3, &tmpA, tensor_B + ind_B, &INCX, buffer + ind_C, &n1);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_032_1302(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (size_t jl = 0; jl < n1 * n3; jl++)
    {
        size_t j = jl / n3;
        size_t l = jl % n3;

        size_t ind_B = l * n2;
        size_t ind_A = j * n2;
        size_t ind_C = jl * n0 * n2;

        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_A_now = ind_A + i * n1 * n2;
            size_t ind_B_now = ind_B + i * n3 * n2;
            for (size_t k = 0; k < n2; k++)
            {

                buffer[ind_C++] = tensor_A[ind_A_now++] * tensor_B[ind_B_now++];
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_043_12403(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_032_1302(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3, n4, buffer);
}

void fn_contraction_012_023_0213(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        dger_(&n3, &n1, &ALPHA, tensor_B + ik * n3, &INCX, tensor_A + i * n1 * n2 + k, &n2, buffer + ik * n1 * n3, &n3);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_0234_13402(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_023_0213(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n2, n1 * n3 * n4, buffer);
}

void fn_contraction_01_1234_02341(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_12_021(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_012_13_0231(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        for (size_t j = 0; j < n1; j++)
        {
            size_t ind_C = i * n1 * n2 * n3 + k * n1 * n3 + j;
            size_t ind_A = i * n1 * n2 + j * n2 + k;
            size_t ind_B = j * n3;

            daxpy_(&n3, tensor_A + ind_A, tensor_B + ind_B, &INCX, buffer + ind_C, &n1);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_134_02341(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_13_0231(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_01_0_10(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        daxpy_(&n1, tensor_B + i, tensor_A + i * n1, &INCX, buffer + i, &n0);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1);
}

void fn_contraction_012_230_0213(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nik = n0 * n2;

    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        size_t ind_A = i * n1 * n2 + k;
        size_t ind_B = k * n0 * n3 + i;

        dger_(&n3, &n1, &ALPHA, tensor_B + ind_B, &n0, tensor_A + ind_A, &n2, buffer + ik * n1 * n3, &n3);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_2340_13402(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_230_0213(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n2, n1 * n3 * n4, buffer);
}

void fn_contraction_012_03412_034(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_02(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3 * n4, buffer);
}

void fn_contraction_01234_012_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_0_10(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, n3 * n4, buffer);
}

void fn_contraction_012_132_1203(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int njk = n1 * n2;

    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t jk = 0; jk < njk; jk++)
    {
        size_t j = jk / n2;
        size_t k = jk % n2;

        size_t ind_A = j * n2 + k;
        size_t ind_B = j * n3 * n2 + k;

        dger_(&n3, &n0, &ALPHA, tensor_B + ind_B, &n2, tensor_A + ind_A, &njk, buffer + jk * n0 * n3, &n3);
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_1342_03412(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_132_1203(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n1 * n2, n0 * n3 * n4, buffer);
}

void fn_contraction_01_012_021(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = i * n1 * n2 + j;
            daxpy_(&n2, tensor_A + ij, tensor_B + ij * n2, &INCX, buffer + ind_C, &n1);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01_012_021_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = i * n1 * n2 + j;
            daxpy_(&n2, tensor_A + ij, tensor_B + ij * n2, &INCX, tensor_C + ind_C, &n1);
        }
    }
}

void fn_contraction_01_01234_02341(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_012_021(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_01_01234_02341_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_012_021_plus(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_01_102_201(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nij = n0 * n1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_C = i * n1 + j;
        size_t ind_A = i * n1 + j;
        size_t ind_B = (j * n0 + i) * n2;

        daxpy_(&n2, tensor_A + ij, tensor_B + ind_B, &INCX, buffer + ind_C, &nij);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01_10234_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_102_201(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_0_01_10(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    double *buffer)
{
    static const double ALPHA = 1.0;
    static const int INCX = 1;
    memset(buffer, 0, sizeof(double) * n0 * n1);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        daxpy_(&n1, tensor_A + i, tensor_B + i * n1, &INCX, buffer + i, &n0);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1);
}

void fn_contraction_01_01234_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0_01_10(tensor_A, tensor_B, tensor_C, n0 * n1, n2 * n3 * n4, buffer);
}

void fn_contraction_01_012_02_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        size_t ind_A = i * n1;
        size_t ind_B = i * n1 * n2 + k;
        tensor_C[i * n2 + k] += ddot_(&n1, tensor_A + ind_A, &INCX, tensor_B + ind_B, &n2);
    }
}

void fn_contraction_01_01234_0234_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_012_02_plus(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_01_01234_0234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_012_02(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_01_1023_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_102_201(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_0123_0231(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_012_021(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_0123_023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_012_02(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_021_201(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nij = n0 * n1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ++ij)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        size_t ind_C = ij;
        size_t ind_B = i * n1 * n2 + j;
        daxpy_(&n2, tensor_A + ij, tensor_B + ind_B, &n1, buffer + ind_C, &nij);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_01234_034_12034(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_201(tensor_B, tensor_A, tensor_C, n0, n3 * n4, n1 * n2, buffer);
}

/// fn_dot ///

void fn_dot(
    const double *tensor_A,
    const double *tensor_B,
    const int size,
    double *result)
{
    static const int INC = 1;

    int nthreads = get_omp_threads();

    int bunch = size / nthreads + 1;

    *result = 0.0;

#pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();

        double tmp = 0.0;

        size_t start = thread_id * bunch;
        size_t end = (thread_id + 1) * bunch;
        start = start > size ? size : start;
        end = end > size ? size : end;
        int size_tmp = end - start;

        tmp = ddot_(&size_tmp, tensor_A + start, &INC, tensor_B + start, &INC);

#pragma omp critical
        {
            *result += tmp;
        }
    }
}

void fn_dot_plus(
    const double *tensor_A,
    const double *tensor_B,
    const int size,
    double *result)
{
    static const int INC = 1;

    int nthreads = get_omp_threads();

    int bunch = size / nthreads + 1;

#pragma omp parallel num_threads(nthreads)
    {
        int thread_id = omp_get_thread_num();

        double tmp = 0.0;

        size_t start = thread_id * bunch;
        size_t end = (thread_id + 1) * bunch;
        start = start > size ? size : start;
        end = end > size ? size : end;
        int size_tmp = end - start;

        tmp = ddot_(&size_tmp, tensor_A + start, &INC, tensor_B + start, &INC);

#pragma omp critical
        {
            *result += tmp;
        }
    }
}

void fn_contraction_012_0342_13402(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_032_1302(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_01_0123_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_0_01_10(tensor_A, tensor_B, tensor_C, n0 * n1, n2 * n3, buffer);
}

void fn_contraction_0123_04123_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_201(tensor_A, tensor_B, tensor_C, n0, n1 * n2 * n3, n4, buffer);
}

void fn_contraction_01_02134_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_0213_2301(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_23104_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    int nij = n0 * n1;
    // int nijk = n0 * n1 * n2;

    static const int INCX = 1;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3 * n4);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = (ij * n2 + k) * n3 * n4;

            size_t ind_A = ij * n2 + k;
            size_t ind_B = (((k * n3 * n1) + j) * n0 + i) * n4;

            for (size_t l = 0; l < n3; l++)
            {
                size_t ind_C2 = ind_C + l * n4;
                size_t ind_B2 = ind_B + l * n1 * n0 * n4;

                daxpy_(&n4, tensor_A + ind_A, tensor_B + ind_B2, &INCX, buffer + ind_C2, &INCX);
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_012_23104_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_23104_01234(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3, n4, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n2 * n1, n3 * n4, buffer);
}

void fn_contraction_012_0312_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

    int nij = n0 * n1;
    int njk = n1 * n2;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_C = ij * n2 * n3;
        size_t ind_A = ij * n2;
        size_t ind_B = (i * n3 * n1 + j) * n2;

        for (size_t k = 0; k < n2; k++)
        {
            daxpy_(&n3, tensor_A + ind_A + k, tensor_B + ind_B + k, &njk, buffer + ind_C + k * n3, &INCX);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_012_0312_3012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_012_0312_0123(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n1 * n2, n3, buffer);
}

void fn_contraction_012_0312_3012_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
#pragma omp parallel for schedule(static)
    for (size_t il = 0; il < n0 * n3; il++)
    {
        size_t i = il / n3;
        size_t l = il % n3;

        size_t ind_A = i * n1 * n2;
        size_t ind_B = (i * n3 + l) * n1 * n2;
        size_t ind_C = (l * n0 + i) * n1 * n2;

        for (size_t jk = 0; jk < n1 * n2; jk++)
        {
            tensor_C[ind_C++] = tensor_A[ind_A++] * tensor_B[ind_B++];
        }
    }
}

void fn_contraction_012_03412_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_0312_3012(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_01234_0124_12403(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    static const int INCX = 1;

    int njk = n1 * n2;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3 * n4);

#pragma omp parallel for schedule(static)
    for (size_t jk = 0; jk < n1 * n2; jk++)
    {
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_A = (i * n1 * n2 + jk) * n3 * n4;
            size_t ind_B = (jk + i * n1 * n2) * n4;
            size_t ind_C = (jk * n4 * n0 + i) * n3;
            for (size_t m = 0; m < n4; m++)
            {
                daxpy_(&n3, tensor_B + ind_B + m, tensor_A + ind_A + m, &n4, buffer + ind_C + m * n0 * n3, &INCX);
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_01234_0124_30124(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01234_0124_12403(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3, n4, buffer);
    fn_permutation_012_210(tensor_C, tensor_C, n1 * n2 * n4, n0, n3, buffer);
}

void fn_contraction_01_0231_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_021_201(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01234_014_23014(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_013_2013(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, n4, buffer);
}

void fn_contraction_01_02341_0234_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_02_plus(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3 * n4, buffer);
}

void fn_contraction_012_01342_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_021_201(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_01_1203_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

    int nij = n0 * n1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_C = ij * n2 * n3;
        size_t ind_A = ij;
        size_t ind_B = ((j * n2 * n0) + i) * n3;

        for (size_t k = 0; k < n2; k++)
        {
            daxpy_(&n3, tensor_A + ind_A, tensor_B + ind_B + k * n0 * n3, &INCX, buffer + ind_C + k * n3, &INCX);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01_1203_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_1203_0123(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n1, n2 * n3, buffer);
}

void fn_contraction_01234_021_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_021_3012(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_01234_04123_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_021_012(tensor_A, tensor_B, tensor_C, n0, n1 * n2 * n3, n4, buffer);
}

void fn_contraction_0123_032_1023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{

    int nij = n0 * n1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = ((((j * n0) + i) * n2) + k) * n3;
            size_t ind_A = ij * n2 * n3 + k * n3;
            size_t ind_B = i * n3 * n2 + k;

            for (size_t l = 0; l < n3; l++)
            {
                buffer[ind_C + l] = tensor_A[ind_A + l] * tensor_B[ind_B + l * n2];
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01234_0423_10234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_032_1023(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, n4, buffer);
}

void fn_contraction_01234_0123_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_0_10(tensor_A, tensor_B, tensor_C, n0 * n1 * n2 * n3, n4, buffer);
}

fn_contraction_012_031_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;
    static const double ALPHA = 1.0;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ++ij)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_C = ij * n2 * n3;
        size_t ind_A = ij * n2;
        size_t ind_B = i * n3 * n1 + j;

        dger_(&n3, &n2, &ALPHA, tensor_B + ind_B, &n1, tensor_A + ind_A, &INCX, buffer + ind_C, &n3);
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_041_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_031_0123(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, n4, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n1, n2 * n3 * n4, buffer);
}

void fn_contraction_012_23140_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    int nij = n0 * n1;
    int nijk = n0 * n1 * n2;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3 * n4);

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = (ij * n2 + k);
            size_t ind_A = ij * n2 + k;
            size_t ind_B = (k * n3 * n1 + j) * n4 * n0 + i;

            for (size_t l = 0; l < n3; l++)
            {
                size_t ind_C2 = ind_C + l * n4 * n0 * n1 * n2;
                size_t ind_B2 = ind_B + l * n1 * n4 * n0;

                daxpy_(&n4, tensor_A + ind_A, tensor_B + ind_B2, &n0, buffer + ind_C2, &nijk);
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_0123_01423_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_0132_3012(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, n4, buffer);
}

void fn_contraction_01234_0134_20134(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_013_2013(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_012_01234_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0_01_10(tensor_A, tensor_B, tensor_C, n0 * n1 * n2, n3 * n4, buffer);
}

void fn_contraction_012_01_02_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    double *buffer)
{
    static const int INCX = 1;

#pragma omp parallel for schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        size_t ind_A = i * n1 * n2 + k;
        size_t ind_B = i * n1;

        tensor_C[ik] += ddot_(&n1, tensor_A + ind_A, &n2, tensor_B + ind_B, &INCX);
    }
}

void fn_contraction_01234_012_0134_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_012_01_02_plus(
        tensor_A,
        tensor_B,
        tensor_C,
        n0 * n1,
        n2,
        n3 * n4,
        buffer);
}

void fn_contraction_01_12034_23401(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_1203_2301(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3 * n4, buffer);
}

void fn_contraction_0123_03421_40123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3 * n4);

    int nijkl = n0 * n1 * n2 * n3;
    int njk = n1 * n2;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t indA = ij * n2 * n3;

        for (size_t k = 0; k < n2; k++)
        {
            size_t ind_C = (ij * n2 + k) * n3;
            size_t ind_B = i * n1 * n2 * n3 * n4 + j + k * n1;

            for (size_t l = 0; l < n3; l++, indA++, ind_C++)
            {
                size_t ind_B2 = ind_B + l * n4 * n2 * n1;

                daxpy_(&n4, tensor_A + indA, tensor_B + ind_B2, &njk, buffer + ind_C, &nijkl);
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_0123_0213_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

    int nij = n0 * n1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t ind_C = ij * n2 * n3;
        size_t ind_B = i * n3 * n1 * n2 + j * n3;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, ind_C++)
            {
                buffer[ind_C] = tensor_A[ind_C] * tensor_B[ind_B + l + k * n1 * n3];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01234_01324_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_0213_0123(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3, n4, buffer);
}

void fn_contraction_01234_031_24013(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (int il = 0; il < n0 * n3; il++)
    {
        int i = il / n3;
        int l = il % n3;
        for (int j = 0; j < n1; j++)
        {
            for (int k = 0; k < n2; k++)
            {
                for (int m = 0; m < n4; m++)
                {
                    int ind_A = i * n1 * n2 * n3 * n4 + j * n2 * n3 * n4 + k * n3 * n4 + l * n4 + m;
                    int ind_B = i * n3 * n1 + l * n1 + j;
                    int ind_C = k * n4 * n0 * n1 * n3 + m * n0 * n1 * n3 + i * n1 * n3 + j * n3 + l;
                    buffer[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
                }
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_012_02314_34012(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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
        for (int k = 0; k < n2; k++)
        {
            int ind_A = i * n1 * n2 + j * n2 + k;
            for (int l = 0; l < n3; l++)
            {
                for (int m = 0; m < n4; m++)
                {
                    int ind_B = i * n2 * n3 * n1 * n4 + k * n3 * n1 * n4 + l * n1 * n4 + j * n4 + m;
                    int ind_C = l * n4 * n0 * n1 * n2 + m * n0 * n1 * n2 + i * n1 * n2 + j * n2 + k;
                    buffer[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
                }
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_01234_012_0134(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_01_012_02(tensor_B, tensor_A, tensor_C, n0 * n1, n2, n3 * n4, buffer);
}

void fn_contraction_0123_0321_0123(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    int nij = n0 * n1;

#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        size_t indA = ij * n2 * n3;
        size_t indB = i * n1 * n2 * n3 + j;

        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++, indA++)
            {
                buffer[indA] = tensor_A[indA] * tensor_B[indB + (l * n2 + k) * n1];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01234_01432_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_0321_0123(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3, n4, buffer);
}

void fn_contraction_01234_032_10234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    static const int INCX = 1;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3 * n4);

#pragma omp parallel for schedule(static)
    for (int il = 0; il < n0 * n3; il++)
    {
        int i = il / n3;
        int l = il % n3;
        size_t ind_B = i * n3 * n2 + l * n2;
        for (int k = 0; k < n2; k++, ind_B++)
        {
            int ind_A = ((i * n1 * n2 + k) * n3 + l) * n4;
            int ind_C = ((i * n2 + k) * n3 + l) * n4;
            for (int j = 0; j < n1; ++j, ind_A += n2 * n3 * n4, ind_C += n0 * n2 * n3 * n4)
            {
                daxpy_(&n4, tensor_B + ind_B, tensor_A + ind_A, &INCX, buffer + ind_C, &INCX);
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_01234_032_14023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{

    fn_contraction_01234_032_10234(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3, n4, buffer);
    fn_permutation_012_021(tensor_C, tensor_C, n1, n0 * n2 * n3, n4, buffer);
}

void fn_contraction_0123_031_0132(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    static const int INCX = 1;

    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

#pragma omp parallel for schedule(static)
    for (int il = 0; il < n0 * n3; il++)
    {
        int i = il / n3;
        int l = il % n3;
        size_t indB = i * n3 * n1 + l * n1;
        size_t indA = i * n1 * n2 * n3 + l;
        size_t indC = i * n1 * n3 * n2 + l * n2;
        for (int j = 0; j < n1; j++, indB++, indA += n2 * n3, indC += n2 * n3)
        {
            daxpy_(&n2, tensor_B + indB, tensor_A + indA, &n3, buffer + indC, &INCX);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_0123_031_2013(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{

#pragma omp parallel for schedule(static)
    for (int ik = 0; ik < n0 * n2; ik++)
    {
        int i = ik / n2;
        int k = ik % n2;
        for (int j = 0; j < n1; j++)
        {
            size_t ind_A = (((i * n1) + j) * n2 + k) * n3;
            size_t ind_B = (i * n3 * n1) + j;
            size_t ind_C = (((k * n0) + i) * n1 + j) * n3;
            for (int l = 0; l < n3; l++, ind_A++, ind_B += n1, ind_C++)
            {
                buffer[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
}

void fn_contraction_01234_0412_30124(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
    fn_contraction_0123_031_2013(tensor_A, tensor_B, tensor_C, n0, n1 * n2, n3, n4, buffer);
}

void fn_contraction_01234_04132_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    const int n4,
    double *buffer)
{
#pragma omp parallel for schedule(static)
    for (int im = 0; im < n0 * n4; im++)
    {
        int i = im / n4;
        int m = im % n4;

        for (int j = 0; j < n1; j++)
        {
            for (int k = 0; k < n2; k++)
            {
                for (int l = 0; l < n3; l++)
                {
                    int ind_A = i * n1 * n2 * n3 * n4 + j * n2 * n3 * n4 + k * n3 * n4 + l * n4 + m;
                    int ind_B = i * n4 * n1 * n2 * n3 + m * n1 * n3 * n2 + j * n2 * n3 + l * n2 + k;
                    buffer[ind_A] = tensor_A[ind_A] * tensor_B[ind_B];
                }
            }
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_01234_04321_01234(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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
        size_t indA = ij * n2 * n3 * n4;
        size_t indB = i * n1 * n2 * n3 * n4 + j;
        for (int k = 0; k < n2; k++)
        {
            for (int l = 0; l < n3; l++)
            {
                for (int m = 0; m < n4; m++, indA++)
                {
                    buffer[indA] = tensor_A[indA] * tensor_B[indB + ((m * n3 + l) * n2 + k) * n1];
                }
            }
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3 * n4);
}

void fn_contraction_012_0132_013(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_021_02(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3, buffer);
}

void fn_contraction_012_0132_013_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    fn_contraction_01_021_02_wob(tensor_A, tensor_B, tensor_C, n0 * n1, n2, n3);
}

void fn_contraction_012_02_102(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
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
        size_t ind_A = ij * n2;
        size_t ind_B = i * n2;
        size_t ind_C = (j * n0 + i) * n2;
        for (size_t k = 0; k < n2; k++, ind_A++, ind_B++, ind_C++)
        {
            buffer[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
        }
    }
    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
}

void fn_contraction_012_02_102_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2)
{
#pragma omp parallel for schedule(static)
    for (size_t ij = 0; ij < n0 * n1; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;
        size_t ind_A = ij * n2;
        size_t ind_B = i * n2;
        size_t ind_C = (j * n0 + i) * n2;
        for (size_t k = 0; k < n2; k++, ind_A++, ind_B++, ind_C++)
        {
            tensor_C[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
        }
    }
}

void fn_contraction_0123_023_1023(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_012_02_102(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_0123_023_1023_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_012_02_102_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3);
}

void fn_contraction_01_0231_023_plus(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_01_021_02_plus(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3, buffer);
}

void fn_contraction_01_0231_023_plus_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
    fn_contraction_01_021_02_plus_wob(tensor_A, tensor_B, tensor_C, n0, n1, n2 * n3);
}

void fn_contraction_012_031_2301(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3,
    double *buffer)
{
    fn_contraction_012_031_0123(tensor_A, tensor_B, tensor_C, n0, n1, n2, n3, buffer);
    fn_permutation_01_10(tensor_C, tensor_C, n0 * n1, n2 * n3, buffer);
}

void fn_contraction_012_031_2301_wob(
    const double *tensor_A,
    const double *tensor_B,
    double *tensor_C,
    const int n0,
    const int n1,
    const int n2,
    const int n3)
{
#pragma omp parallel for schedule(static)
    for (size_t kl = 0; kl < n2 * n3; kl++)
    {
        size_t k = kl / n3;
        size_t l = kl % n3;
        size_t ind_C = kl * n0 * n1;
        size_t ind_A = k;
        for (size_t i = 0; i < n0; i++)
        {
            size_t ind_B = i * n1 * n3 + l * n1;
            for (size_t j = 0; j < n1; j++, ind_A += n2, ind_B++, ind_C++)
            {
                tensor_C[ind_C] = tensor_A[ind_A] * tensor_B[ind_B];
            }
        }
    }
}