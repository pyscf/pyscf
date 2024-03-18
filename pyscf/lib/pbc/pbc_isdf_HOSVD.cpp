
#include "vhf/fblas.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "fft.h"

//// HOSVD holder, should not be called in py

struct HOSVD_4D
{
    int shape[4];
    int Bshape[4];
    double *B = NULL;
    double *U0 = NULL;
    double *U1 = NULL;
    double *U2 = NULL;
    double *U3 = NULL;
    double *S0 = NULL;
    double *S1 = NULL;
    double *S2 = NULL;
    double *S3 = NULL;
};

struct HOSVD_4D_complex
{
    int shape[4];
    int Bshape[4];
    double *B = NULL;
    double __complex__ *U0 = NULL;
    double __complex__ *U1 = NULL;
    double __complex__ *U2 = NULL;
    double __complex__ *U3 = NULL;
    double *S0 = NULL;
    double *S1 = NULL;
    double *S2 = NULL;
    double *S3 = NULL;
};

HOSVD_4D HOSVD_4D_init(int *shape, int *Bshape, double *B,
                       double *U0, double *U1, double *U2, double *U3,
                       double *S0, double *S1, double *S2, double *S3)
{
    HOSVD_4D hosvd;

    hosvd.shape[0] = shape[0];
    hosvd.shape[1] = shape[1];
    hosvd.shape[2] = shape[2];
    hosvd.shape[3] = shape[3];
    hosvd.Bshape[0] = Bshape[0];
    hosvd.Bshape[1] = Bshape[1];
    hosvd.Bshape[2] = Bshape[2];
    hosvd.Bshape[3] = Bshape[3];
    hosvd.B = B;
    hosvd.U0 = U0;
    hosvd.U1 = U1;
    hosvd.U2 = U2;
    hosvd.U3 = U3;
    hosvd.S0 = S0;
    hosvd.S1 = S1;
    hosvd.S2 = S2;
    hosvd.S3 = S3;

    return hosvd;
}

void permute4(int (*result)[4])
{
    result[0][0] = 0;
    result[0][1] = 1;
    result[0][2] = 2;
    result[0][3] = 3;
    result[1][0] = 0;
    result[1][1] = 1;
    result[1][2] = 3;
    result[1][3] = 2;
    result[2][0] = 0;
    result[2][1] = 2;
    result[2][2] = 1;
    result[2][3] = 3;
    result[3][0] = 0;
    result[3][1] = 2;
    result[3][2] = 3;
    result[3][3] = 1;
    result[4][0] = 0;
    result[4][1] = 3;
    result[4][2] = 1;
    result[4][3] = 2;
    result[5][0] = 0;
    result[5][1] = 3;
    result[5][2] = 2;
    result[5][3] = 1;
    result[6][0] = 1;
    result[6][1] = 0;
    result[6][2] = 2;
    result[6][3] = 3;
    result[7][0] = 1;
    result[7][1] = 0;
    result[7][2] = 3;
    result[7][3] = 2;
    result[8][0] = 1;
    result[8][1] = 2;
    result[8][2] = 0;
    result[8][3] = 3;
    result[9][0] = 1;
    result[9][1] = 2;
    result[9][2] = 3;
    result[9][3] = 0;
    result[10][0] = 1;
    result[10][1] = 3;
    result[10][2] = 0;
    result[10][3] = 2;
    result[11][0] = 1;
    result[11][1] = 3;
    result[11][2] = 2;
    result[11][3] = 0;
    result[12][0] = 2;
    result[12][1] = 0;
    result[12][2] = 1;
    result[12][3] = 3;
    result[13][0] = 2;
    result[13][1] = 0;
    result[13][2] = 3;
    result[13][3] = 1;
    result[14][0] = 2;
    result[14][1] = 1;
    result[14][2] = 0;
    result[14][3] = 3;
    result[15][0] = 2;
    result[15][1] = 1;
    result[15][2] = 3;
    result[15][3] = 0;
    result[16][0] = 2;
    result[16][1] = 3;
    result[16][2] = 0;
    result[16][3] = 1;
    result[17][0] = 2;
    result[17][1] = 3;
    result[17][2] = 1;
    result[17][3] = 0;
    result[18][0] = 3;
    result[18][1] = 0;
    result[18][2] = 1;
    result[18][3] = 2;
    result[19][0] = 3;
    result[19][1] = 0;
    result[19][2] = 2;
    result[19][3] = 1;
    result[20][0] = 3;
    result[20][1] = 1;
    result[20][2] = 0;
    result[20][3] = 2;
    result[21][0] = 3;
    result[21][1] = 1;
    result[21][2] = 2;
    result[21][3] = 0;
    result[22][0] = 3;
    result[22][1] = 2;
    result[22][2] = 0;
    result[22][3] = 1;
    result[23][0] = 3;
    result[23][1] = 2;
    result[23][2] = 1;
    result[23][3] = 0;
}

#define MAX_PATH 24

void analysis_GetFullMat_cost(HOSVD_4D *hosvd, int *min_cost, int *min_path, int *min_path_storage)
{
    int perms[MAX_PATH][4];
    int count = 24;

    permute4(perms);

    *min_cost = -1;
    *min_path_storage = 0;

    for (int i = 0; i < MAX_PATH; i++)
    {
        int shape_now[4];
        memcpy(shape_now, hosvd->Bshape, 4 * sizeof(int));
        int cost = 0;
        int storage = 0;

        for (int j = 0; j < 4; j++)
        {
            int idx = perms[i][j];
            shape_now[idx] = hosvd->shape[idx];
            cost += shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3] * hosvd->Bshape[idx];
            storage = fmax(storage, shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }

        if (*min_cost == -1 || cost < *min_cost)
        {
            *min_cost = cost;
            memcpy(min_path, perms[i], 4 * sizeof(int));
            *min_path_storage = storage;
        }
    }
}

void analysis_GetFullMat_cost_complex(HOSVD_4D_complex *hosvd, int *min_cost, int *min_path, int *min_path_storage)
{
    int perms[MAX_PATH][4];
    int count = 24;

    permute4(perms);

    *min_cost = -1;
    *min_path_storage = 0;

    for (int i = 0; i < MAX_PATH; i++)
    {
        int shape_now[4];
        memcpy(shape_now, hosvd->Bshape, 4 * sizeof(int));
        int cost = 0;
        int storage = 0;

        for (int j = 0; j < 4; j++)
        {
            int idx = perms[i][j];
            shape_now[idx] = hosvd->shape[idx];
            cost += shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3] * hosvd->Bshape[idx];
            storage = fmax(storage, shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }

        if (*min_cost == -1 || cost < *min_cost)
        {
            *min_cost = cost;
            memcpy(min_path, perms[i], 4 * sizeof(int));
            *min_path_storage = storage;
        }
    }
}

void HOSVD_4D_GetFullMat(HOSVD_4D *input, double *output, double *buf)
{
    // Get the best path
    int min_cost, min_path[4], min_path_storage;
    analysis_GetFullMat_cost(input, &min_cost, min_path, &min_path_storage);

    // Copy B to buf
    // memcpy(buf, input->B, sizeof(double) * input->Bshape[0] * input->Bshape[1] * input->Bshape[2] * input->Bshape[3]);

    int shape_now[4];

    shape_now[0] = input->Bshape[0];
    shape_now[1] = input->Bshape[1];
    shape_now[2] = input->Bshape[2];
    shape_now[3] = input->Bshape[3];

    for (int i = 0; i < 4; i++)
    {
        int idx = min_path[i];

        if (idx == 0)
        {
            // Transpose A from 'ijkl' to 'jkli', treat jkl as a single dimension
            //

            size_t stride_jkl = shape_now[1] * shape_now[2] * shape_now[3];
            size_t stride_i = shape_now[0];

            for (int p = 0; p < shape_now[0]; p++)
            {
                for (size_t q = 0; q < stride_jkl; ++q)
                {
                    buf[q * stride_i + p] = output[p * stride_jkl + q];
                }
            }

            // Perform dgemm
            char transA = 'N';
            char transB = 'T';
            int m = shape_now[1] * shape_now[2] * shape_now[3];
            int n = shape_now[0];
            int k = input->shape[0];
            double *A = buf;
            double *B = input->U0;
            double *C = output;
            double alpha = 1.0;
            double beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // jkli, ai -> jkla
            dgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            // Transpose A from 'jkla' to 'ajkl', treat jkl as a single dimension

            for (int p = 0; p < input->shape[0]; p++)
            {
                for (size_t q = 0; q < stride_jkl; ++q)
                {
                    buf[p * stride_jkl + q] = output[q * stride_i + p];
                }
            }

            shape_now[0] = input->shape[0];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else if (idx == 1)
        {
            // Transpose A from 'ijkl' to 'iklj',
            // loop over i, treat kl as a single dimension

            size_t stride_kl = shape_now[2] * shape_now[3];

            for (int p = 0; p < shape_now[0]; p++)
            {
                size_t shift = p * shape_now[1] * stride_kl;
                for (size_t r = 0; r < stride_kl; ++r)
                {
                    for (int q = 0; q < shape_now[1]; q++)
                    {
                        buf[shift + r * shape_now[1] + q] = output[shift + q * stride_kl + r];
                    }
                }
            }

            // Perform dgemm
            char transA = 'N';
            char transB = 'T';
            int m = shape_now[0] * shape_now[2] * shape_now[3];
            int n = shape_now[1];
            int k = input->shape[1];
            double *A = buf;
            double *B = input->U1;
            double *C = output;
            double alpha = 1.0;
            double beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // iklj, bj -> iklb
            dgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            // Transpose A from 'iklb' to 'ibkl'
            // loop over i, treat kl as a single dimension

            for (int p = 0; p < shape_now[0]; p++)
            {
                size_t shift = p * input->shape[1] * stride_kl;
                for (size_t r = 0; r < stride_kl; ++r)
                {
                    for (int q = 0; q < input->shape[1]; q++)
                    {
                        buf[shift + q * stride_kl + r] = output[shift + r * input->shape[1] + q];
                    }
                }
            }

            shape_now[1] = input->shape[1];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else if (idx == 2)
        {
            // Transpose A from 'ijkl' to 'ijlk', treat ij as a single dimension
            // loop over ij

            size_t nij = shape_now[0] * shape_now[1];
            size_t stride_kl = shape_now[2] * shape_now[3];

            for (int p = 0; p < nij; p++)
            {
                size_t shift = p * stride_kl;
                for (size_t r = 0; r < shape_now[2]; ++r)
                {
                    for (int s = 0; s < shape_now[3]; s++)
                    {
                        buf[shift + s * shape_now[2] + r] = output[shift + r * shape_now[3] + s];
                    }
                }
            }

            // Perform dgemm
            char transA = 'N';
            char transB = 'T';
            int m = shape_now[0] * shape_now[1] * shape_now[3];
            int n = shape_now[2];
            int k = input->shape[2];
            double *A = buf;
            double *B = input->U2;
            double *C = output;
            double alpha = 1.0;
            double beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // ijlk, ck -> ijlc
            dgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            // Transpose A from 'ijlc' to 'ijcl', treat ij as a single dimension

            for (int p = 0; p < nij; p++)
            {
                size_t shift = p * input->shape[2] * shape_now[3];
                for (size_t s = 0; s < shape_now[3]; ++s)
                {
                    for (int r = 0; r < input->shape[2]; r++)
                    {
                        buf[shift + r * shape_now[3] + s] = output[shift + s * input->shape[2] + r];
                    }
                }
            }

            shape_now[2] = input->shape[2];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else if (idx == 3)
        {
            /// no transopose

            // Perform dgemm

            char transA = 'N';
            char transB = 'T';
            int m = shape_now[0] * shape_now[1] * shape_now[2];
            int n = shape_now[3];
            int k = input->shape[3];
            double *A = output;
            double *B = input->U3;
            double *C = buf;
            double alpha = 1.0;
            double beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // ijkl, dl -> ijkd
            dgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            shape_now[3] = input->shape[3];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else
        {
            printf("Error: idx out of range\n");
            exit(1);
        }
    }
}

void HOSVD_4D_complex_GetFull(HOSVD_4D_complex *input, double __complex__ *output, double __complex__ *buf)
{
    // Get the best path
    int min_cost, min_path[4], min_path_storage;

    analysis_GetFullMat_cost_complex(input, &min_cost, min_path, &min_path_storage);

    // Copy B to buf
    // memcpy(buf, input->B, sizeof(double) * input->Bshape[0] * input->Bshape[1] * input->Bshape[2] * input->Bshape[3]);

    for (int i = 0; i < input->Bshape[0] * input->Bshape[1] * input->Bshape[2] * input->Bshape[3]; i++)
    {
        output[i] = input->B[i];
    }

    int shape_now[4];

    shape_now[0] = input->Bshape[0];
    shape_now[1] = input->Bshape[1];
    shape_now[2] = input->Bshape[2];
    shape_now[3] = input->Bshape[3];

    for (int i = 0; i < 4; i++)
    {
        int idx = min_path[i];

        if (idx == 0)
        {
            // Transpose A from 'ijkl' to 'jkli', treat jkl as a single dimension
            //

            size_t stride_jkl = shape_now[1] * shape_now[2] * shape_now[3];
            size_t stride_i = shape_now[0];

            for (int p = 0; p < shape_now[0]; p++)
            {
                for (size_t q = 0; q < stride_jkl; ++q)
                {
                    buf[q * stride_i + p] = output[p * stride_jkl + q];
                }
            }

            // Perform dgemm
            char transA = 'N';
            char transB = 'T';
            int m = shape_now[1] * shape_now[2] * shape_now[3];
            int n = shape_now[0];
            int k = input->shape[0];
            double __complex__ *A = buf;
            double __complex__ *B = input->U0;
            double __complex__ *C = output;
            double __complex__ alpha = 1.0;
            double __complex__ beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // jkli, ai -> jkla
            zgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            // Transpose A from 'jkla' to 'ajkl', treat jkl as a single dimension

            for (int p = 0; p < input->shape[0]; p++)
            {
                for (size_t q = 0; q < stride_jkl; ++q)
                {
                    buf[p * stride_jkl + q] = output[q * stride_i + p];
                }
            }

            shape_now[0] = input->shape[0];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double __complex__) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else if (idx == 1)
        {
            // Transpose A from 'ijkl' to 'iklj',
            // loop over i, treat kl as a single dimension

            size_t stride_kl = shape_now[2] * shape_now[3];

            for (int p = 0; p < shape_now[0]; p++)
            {
                size_t shift = p * shape_now[1] * stride_kl;
                for (size_t r = 0; r < stride_kl; ++r)
                {
                    for (int q = 0; q < shape_now[1]; q++)
                    {
                        buf[shift + r * shape_now[1] + q] = output[shift + q * stride_kl + r];
                    }
                }
            }

            // Perform dgemm
            char transA = 'N';
            char transB = 'T';
            int m = shape_now[0] * shape_now[2] * shape_now[3];
            int n = shape_now[1];
            int k = input->shape[1];
            double __complex__ *A = buf;
            double __complex__ *B = input->U1;
            double __complex__ *C = output;
            double __complex__ alpha = 1.0;
            double __complex__ beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // iklj, bj -> iklb
            zgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            // Transpose A from 'iklb' to 'ibkl'
            // loop over i, treat kl as a single dimension

            for (int p = 0; p < shape_now[0]; p++)
            {
                size_t shift = p * input->shape[1] * stride_kl;
                for (size_t r = 0; r < stride_kl; ++r)
                {
                    for (int q = 0; q < input->shape[1]; q++)
                    {
                        buf[shift + q * stride_kl + r] = output[shift + r * input->shape[1] + q];
                    }
                }
            }

            shape_now[1] = input->shape[1];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double __complex__) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else if (idx == 2)
        {
            // Transpose A from 'ijkl' to 'ijlk', treat ij as a single dimension
            // loop over ij

            size_t nij = shape_now[0] * shape_now[1];
            size_t stride_kl = shape_now[2] * shape_now[3];

            for (int p = 0; p < nij; p++)
            {
                size_t shift = p * stride_kl;
                for (size_t r = 0; r < shape_now[2]; ++r)
                {
                    for (int s = 0; s < shape_now[3]; s++)
                    {
                        buf[shift + s * shape_now[2] + r] = output[shift + r * shape_now[3] + s];
                    }
                }
            }

            // Perform dgemm
            char transA = 'N';
            char transB = 'T';
            int m = shape_now[0] * shape_now[1] * shape_now[3];
            int n = shape_now[2];
            int k = input->shape[2];
            double __complex__ *A = buf;
            double __complex__ *B = input->U2;
            double __complex__ *C = output;
            double __complex__ alpha = 1.0;
            double __complex__ beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // ijlk, ck -> ijlc
            zgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            // Transpose A from 'ijlc' to 'ijcl', treat ij as a single dimension

            for (int p = 0; p < nij; p++)
            {
                size_t shift = p * input->shape[2] * shape_now[3];
                for (size_t s = 0; s < shape_now[3]; ++s)
                {
                    for (int r = 0; r < input->shape[2]; r++)
                    {
                        buf[shift + r * shape_now[3] + s] = output[shift + s * input->shape[2] + r];
                    }
                }
            }

            shape_now[2] = input->shape[2];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double __complex__) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else if (idx == 3)
        {
            /// no transopose

            // Perform dgemm

            char transA = 'N';
            char transB = 'T';
            int m = shape_now[0] * shape_now[1] * shape_now[2];
            int n = shape_now[3];
            int k = input->shape[3];
            double __complex__ *A = output;
            double __complex__ *B = input->U3;
            double __complex__ *C = buf;
            double __complex__ alpha = 1.0;
            double __complex__ beta = 0.0;
            int lda = m;
            int ldb = k;
            int ldc = m;
            // ijkl, dl -> ijkd
            zgemm_(&transA, &transB, &m, &k, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

            shape_now[3] = input->shape[3];

            // Copy the result back to output

            memcpy(output, buf, sizeof(double __complex__) * shape_now[0] * shape_now[1] * shape_now[2] * shape_now[3]);
        }
        else
        {
            printf("Error: idx out of range\n");
            exit(1);
        }
    }
}