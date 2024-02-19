#include "vhf/fblas.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "fft.h"

int get_omp_threads();
int omp_get_thread_num();

void Complex_Cholesky(double __complex__ *A, int n)
{    
    // A will be overwritten with the lower triangular Cholesky factor
    lapack_int info;
    info = LAPACKE_zpotrf(LAPACK_ROW_MAJOR, 'U', n, A, n);
    if (info != 0)
    {
        fprintf(stderr, "Cholesky decomposition failed: %d\n", info);
        exit(1);
    }
}

void _FFT_Matrix_Col_InPlace(double *matrix, // the size of matrix should be (nRow, nCol* *mesh)
                             int nRow, int nCol, int *mesh,
                             double *buf)
{
    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};
    int nComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    int nReal = mesh[0] * mesh[1] * mesh[2];
    const int nThread = get_omp_threads();

    // printf("nThread: %d\n", nThread);
    // printf("nRow: %d\n", nRow);
    // printf("nCol: %d\n", nCol);
    // printf("mesh: %d %d %d\n", mesh[0], mesh[1], mesh[2]);
    // printf("nComplex: %d\n", nComplex);

    const int m = nRow;
    const int n = nCol * mesh[0] * mesh[1] * mesh[2];
    const int n_complex = nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const int nMesh = mesh[0] * mesh[1] * mesh[2];
    const int nMeshComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];

    // printf("m: %d\n", m);
    // printf("n: %d\n", n);
    // printf("nMesh: %d\n", nMesh);
    // printf("nMeshComplex: %d\n", nMeshComplex);

    // (1) transform (Row, Block, Col) -> (Row, Col, Block)

#pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < m; i++)
    {
        int iCol = 0;

        for (int iBlock = 0; iBlock < nMesh; iBlock++)
        {
            for (int j = 0; j < nCol; j++, iCol++)
            {
                buf[i * n + j * nMesh + iBlock] = matrix[i * n + iCol];
            }
        }
    }

    // printf("finish (1) \n");

    // (2) perform FFT on the last dimension

    int nFFT = nRow * nCol;

    double __complex__ *mat_complex = (double __complex__ *)buf;
    double __complex__ *buf_complex = (double __complex__ *)matrix;

    // create plan

    const int BunchSize = nFFT / nThread + 1;

#pragma omp parallel num_threads(nThread)
    {
        int tid = omp_get_thread_num();
        int start = tid * BunchSize;
        int end = (tid + 1) * BunchSize;
        if (end > nFFT)
        {
            end = nFFT;
        }

        fftw_plan plan = fftw_plan_many_dft_r2c(3, mesh, end - start, buf + start * nReal, mesh, 1, nReal, (fftw_complex *)buf_complex + start * nComplex, mesh_complex, 1, nComplex, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    // printf("finish (2) \n");

    // (3) transform (Row, Col, Block) -> (Row, Block, Col)

    mat_complex = (double __complex__ *)matrix;
    buf_complex = (double __complex__ *)buf;

#pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < m; i++)
    {
        int iCol = 0;

        for (int j = 0; j < nCol; j++)
        {
            for (int iBlock = 0; iBlock < nMeshComplex; iBlock++, iCol++)
            {
                buf_complex[i * n_complex + iBlock * nCol + j] = mat_complex[i * n_complex + iCol];
            }
        }
    }

    // printf("finish (3) \n");

    memcpy(matrix, buf, sizeof(double __complex__) * m * nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2]);

    // printf("finish memcpy \n");
}

void _iFFT_Matrix_Col_InPlace(double __complex__ *matrix, // the size of matrix should be (nRow, nCol* *mesh)
                              int nRow, int nCol, int *mesh,
                              double __complex__ *buf)
{
    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};
    int nComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    int nReal = mesh[0] * mesh[1] * mesh[2];
    const int nThread = get_omp_threads();

    const int m = nRow;
    const int n = nCol * mesh[0] * mesh[1] * mesh[2];
    const int n_Complex = nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const int nMesh = mesh[0] * mesh[1] * mesh[2];
    const int nMeshComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const double factor = 1.0 / (double)(nMesh);

    printf("m: %d\n", m);
    printf("n: %d\n", n);
    printf("n_Complex: %d\n", n_Complex);
    printf("nMesh: %d\n", nMesh);
    printf("nMeshComplex: %d\n", nMeshComplex);
    printf("nThread: %d\n", nThread);
    printf("nRow: %d\n", nRow);
    printf("nCol: %d\n", nCol);
    printf("mesh: %d %d %d\n", mesh[0], mesh[1], mesh[2]);
    printf("nComplex: %d\n", nComplex);
    printf("nReal: %d\n", nReal);

    // (1) transform (Row, Block, Col) -> (Row, Col, Block)

#pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < m; i++)
    {
        int iCol = 0;

        for (int iBlock = 0; iBlock < nMeshComplex; iBlock++)
        {
            for (int j = 0; j < nCol; j++, iCol++)
            {
                buf[i * n_Complex + j * nMeshComplex + iBlock] = matrix[i * n_Complex + iCol];
            }
        }
    }

    // (2) perform iFFT on the last dimension

    int nFFT = nRow * nCol;

    double *mat_real = (double *)buf;
    double *buf_real = (double *)matrix;

    // create plan

    const int BunchSize = nFFT / nThread + 1;

#pragma omp parallel num_threads(nThread)
    {
        int tid = omp_get_thread_num();
        int start = tid * BunchSize;
        int end = (tid + 1) * BunchSize;
        if (end > nFFT)
        {
            end = nFFT;
        }

        fftw_plan plan = fftw_plan_many_dft_c2r(3, mesh, end - start, (fftw_complex *)buf + start * nComplex, mesh_complex, 1, nComplex, buf_real + start * nReal, mesh, 1, nReal, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    // (3) transform (Row, Col, Block) -> (Row, Block, Col)

    mat_real = (double *)matrix;
    buf_real = (double *)buf;

#pragma omp parallel for num_threads(nThread)
    for (int i = 0; i < m; i++)
    {
        int iCol = 0;

        for (int j = 0; j < nCol; j++)
        {
            for (int iBlock = 0; iBlock < nMesh; iBlock++, iCol++)
            {
                // printf("i: %d, j: %d, iBlock: %d, iCol: %d %15.8f\n", i, j, iBlock, iCol, mat_real[i * n + iCol]);
                buf_real[i * n + iBlock * nCol + j] = mat_real[i * n + iCol] * factor;
            }
        }
    }

    memcpy(mat_real, buf_real, sizeof(double) * m * nCol * mesh[0] * mesh[1] * mesh[2]);
}

void Solve_LLTEqualB_Complex_Parallel(
    const int n,
    const double __complex__ *a, // call cholesky first!
    double __complex__ *b,
    const int nrhs,
    const int BunchSize)
{
    int nThread = get_omp_threads();

    int nBunch = (nrhs / BunchSize);
    int nLeft = nrhs - nBunch * BunchSize;

    printf("nThread  : %d\n", nThread);
    printf("nBunch   : %d\n", nBunch);
    printf("nLeft    : %d\n", nLeft);
    printf("BunchSize: %d\n", BunchSize);
    printf("n        : %d\n", n);
    printf("nrhs     : %d\n", nrhs);

#pragma omp parallel num_threads(nThread)
    {
        double __complex__ *ptr_b;
        lapack_int info;

#pragma omp for schedule(static, 1) nowait
        for (int i = 0; i < nBunch; i++)
        {
            ptr_b = b + BunchSize * i;

            // forward transform

            info = LAPACKE_zpotrs(LAPACK_ROW_MAJOR, 'U', n, BunchSize, a, n, ptr_b, nrhs);

            if (info != 0)
            {
                fprintf(stderr, "Solving system failed: %d\n", info);
                exit(1);
            }
        }

#pragma omp single
        {
            if (nLeft > 0)
            {
                // int thread_id = omp_get_thread_num();

                double __complex__ *ptr_b = b + BunchSize * nBunch;

                lapack_int info;

                // forward transform

                info = LAPACKE_zpotrs(LAPACK_ROW_MAJOR, 'U', n, nLeft, a, n, ptr_b, nrhs);

                if (info != 0)
                {
                    fprintf(stderr, "Solving system failed: %d\n", info);
                    exit(1);
                }
            }
        }
    }
}