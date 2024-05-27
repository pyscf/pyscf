#include "fft.h"
#include <omp.h>
#include <string.h>
#include <complex.h>
#include "vhf/fblas.h"
#include <math.h>
#include "np_helper/np_helper.h"

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
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ik = 0; ik < n0 * n2; ik++)
    {
        size_t i = ik / n2;
        size_t k = ik % n2;

        for (size_t j = 0; j < n1; j++)
        {
            buffer[i * n1 * n2 + k * n1 + j] = tensor_A[i * n1 + j] * tensor_B[k * n1 + j];
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
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t jk = 0; jk < n1 * n2; jk++)
    {
        size_t j = jk / n2;
        size_t k = jk % n2;

        for (size_t i = 0; i < n0; i++)
        {
            buffer[jk * n0 + i] = tensor_A[i * n1 + j] * tensor_B[k * n0 + i];
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
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
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2);

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t jk = 0; jk < n1 * n2; jk++)
    {
        size_t j = jk / n2;
        size_t k = jk % n2;

        for (size_t i = 0; i < n0; i++)
        {
            buffer[jk * n0 + i] = tensor_A[i * n1 + j] * tensor_B[i * n2 + k];
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
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
    memset(buffer, 0, sizeof(double) * n0 * n1 * n2 * n3);

    int nthread = get_omp_threads();

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t kl = 0; kl < n2 * n3; kl++)
    {
        size_t k = kl / n3;
        size_t l = kl % n3;
        for (size_t ij = 0; ij < n0 * n1; ij++)
        {
            size_t i = ij / n1;
            size_t j = ij % n1;
            size_t idx_B = k * n0 * n3 * n1 + i * n3 * n1 + l * n1 + j;
            buffer[kl * n0 * n1 + ij] = tensor_A[ij] * tensor_B[idx_B];
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2 * n3);
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

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        double tmp = 0.0;
        for (size_t j = 0; j < n1; j++)
        {
            tmp += tensor_A[i * n1 + j] * tensor_B[i * n1 + j];
        }
        // tensor_C[i] = tmp;
        buffer[i] = tmp;
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0);
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
        // size_t k = ik % n2;
        for (size_t j = 0; j < n1; j++)
        {
            buffer[ik] = ddot_(&n1, tensor_A + i * n1, &INCX, tensor_B + ik * n1, &INCX);
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n2);
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

void fn_contraction_01_021_201(
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

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t ij = 0; ij < nij; ij++)
    {
        size_t i = ij / n1;
        size_t j = ij % n1;

        for (size_t k = 0; k < n2; k++)
        {
            buffer[k * n0 * n1 + ij] = tensor_A[ij] * tensor_B[i * n2 * n1 + k * n1 + j];
        }
    }

    memcpy(tensor_C, buffer, sizeof(double) * n0 * n1 * n2);
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

///// various permutation functions /////

/// NOTE: for permutation A can be equal to B

void fn_permutation_01_10(
    const double *tensor_A,
    double *tensor_B,
    const int n0,
    const int n1,
    double *buffer)
{
    int nthread = get_omp_threads();
#pragma omp parallel for num_threads(nthread) schedule(static)
    for (size_t i = 0; i < n0; i++)
    {
        for (size_t j = 0; j < n1; j++)
        {
            buffer[j * n0 + i] = tensor_A[i * n1 + j];
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1);
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
            for (size_t j = 0; j < n1; j++)
            {
                for (size_t k = 0; k < n2; k++)
                {
                    local_buffer[k * n1 + j] = tensor_A[i * n1 * n2 + j * n2 + k];
                }
            }
            memcpy(tensor_B + i * n1 * n2, local_buffer, sizeof(double) * n1 * n2);
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

void fn_permutation_012_102(
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
        nthread = n2;
    }

#pragma omp parallel num_threads(nthread)
    {
        int thread_id = omp_get_thread_num();

        double *local_buffer = buffer + thread_id * n0 * n1;

#pragma omp for schedule(static)
        for (size_t k = 0; k < n2; k++)
        {
            for (size_t i = 0; i < n0; i++)
            {
                for (size_t j = 0; j < n1; j++)
                {
                    local_buffer[j * n0 + i] = tensor_A[i * n1 * n2 + j * n2 + k];
                }
            }

            size_t idx = 0;
            for (size_t j = 0; j < n1; j++)
            {
                for (size_t i = 0; i < n0; i++)
                {
                    tensor_B[idx * n2 + k] = local_buffer[idx];
                    idx += 1;
                }
            }
        }
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
        for (size_t k = 0; k < n2; k++)
        {
            for (size_t l = 0; l < n3; l++)
            {
                buffer[j * n0 * n3 * n2 + l * n2 * n0 + k * n0 + i] = tensor_A[ij * n2 * n3 + k * n3 + l];
            }
        }
    }

    memcpy(tensor_B, buffer, sizeof(double) * n0 * n1 * n2 * n3);
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
