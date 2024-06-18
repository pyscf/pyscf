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
    int64_t nComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    int64_t nReal = mesh[0] * mesh[1] * mesh[2];
    const int nThread = get_omp_threads();

    // printf("nThread: %d\n", nThread);
    // printf("nRow: %d\n", nRow);
    // printf("nCol: %d\n", nCol);
    // printf("mesh: %d %d %d\n", mesh[0], mesh[1], mesh[2]);
    // printf("nComplex: %d\n", nComplex);

    const int64_t m = nRow;
    const int64_t n = nCol * mesh[0] * mesh[1] * mesh[2];
    const int64_t n_complex = nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const int64_t nMesh = mesh[0] * mesh[1] * mesh[2];
    const int64_t nMeshComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];

    // printf("m: %d\n", m);
    // printf("n: %d\n", n);
    // printf("nMesh: %d\n", nMesh);
    // printf("nMeshComplex: %d\n", nMeshComplex);

    // (1) transform (Row, Block, Col) -> (Row, Col, Block)

#pragma omp parallel for num_threads(nThread)
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t iBlock = 0; iBlock < nMesh; iBlock++)
        {
            for (int64_t j = 0; j < nCol; j++, iCol++)
            {
                buf[i * n + j * nMesh + iBlock] = matrix[i * n + iCol];
            }
        }
    }

    // printf("finish (1) \n");

    // (2) perform FFT on the last dimension

    int64_t nFFT = nRow * nCol;

    double __complex__ *mat_complex = (double __complex__ *)buf;
    double __complex__ *buf_complex = (double __complex__ *)matrix;

    // create plan

    const int BunchSize = nFFT / nThread + 1;

#pragma omp parallel num_threads(nThread)
    {
        int tid = omp_get_thread_num();
        int64_t start = tid * BunchSize;
        int64_t end = (tid + 1) * BunchSize;
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
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t j = 0; j < nCol; j++)
        {
            for (int64_t iBlock = 0; iBlock < nMeshComplex; iBlock++, iCol++)
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
    int64_t nComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    int64_t nReal = mesh[0] * mesh[1] * mesh[2];
    const int64_t nThread = get_omp_threads();

    const int64_t m            = nRow;
    const int64_t n            = nCol * mesh[0] * mesh[1] * mesh[2];
    const int64_t n_Complex    = nCol * mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const int64_t nMesh        = mesh[0] * mesh[1] * mesh[2];
    const int64_t nMeshComplex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const double factor = 1.0 / (double)(nMesh);

    // printf("m: %d\n", m);
    // printf("n: %d\n", n);
    // printf("n_Complex: %d\n", n_Complex);
    // printf("nMesh: %d\n", nMesh);
    // printf("nMeshComplex: %d\n", nMeshComplex);
    // printf("nThread: %d\n", nThread);
    // printf("nRow: %d\n", nRow);
    // printf("nCol: %d\n", nCol);
    // printf("mesh: %d %d %d\n", mesh[0], mesh[1], mesh[2]);
    // printf("nComplex: %d\n", nComplex);
    // printf("nReal: %d\n", nReal);

    // (1) transform (Row, Block, Col) -> (Row, Col, Block)

#pragma omp parallel for num_threads(nThread)
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t iBlock = 0; iBlock < nMeshComplex; iBlock++)
        {
            for (int64_t j = 0; j < nCol; j++, iCol++)
            {
                buf[i * n_Complex + j * nMeshComplex + iBlock] = matrix[i * n_Complex + iCol];
            }
        }
    }

    // (2) perform iFFT on the last dimension

    int64_t nFFT = nRow * nCol;

    double *mat_real = (double *)buf;
    double *buf_real = (double *)matrix;

    // create plan

    const int64_t BunchSize = nFFT / nThread + 1;

#pragma omp parallel num_threads(nThread)
    {
        int64_t tid = omp_get_thread_num();
        int64_t start = tid * BunchSize;
        int64_t end = (tid + 1) * BunchSize;
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
    for (int64_t i = 0; i < m; i++)
    {
        int64_t iCol = 0;

        for (int64_t j = 0; j < nCol; j++)
        {
            for (int64_t iBlock = 0; iBlock < nMesh; iBlock++, iCol++)
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

    int64_t nBunch = (nrhs / BunchSize);
    int64_t nLeft = nrhs - nBunch * BunchSize;

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
        for (int64_t i = 0; i < nBunch; i++)
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

void _FinalFFT(
    double __complex__ *a,
    const double __complex__ *freq,
    int m, int n, int *mesh,
    double __complex__ *buf)
{
    const int nThread = get_omp_threads();

    if (n != mesh[0] * mesh[1] * mesh[2])
    {
        fprintf(stderr, "The size of a is not compatible with mesh\n");
        exit(1);
    }

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();

        double __complex__ *buf_thread = buf + thread_id * n;

        fftw_plan plan = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2], (fftw_complex *)buf_thread, (fftw_complex *)a, FFTW_FORWARD, FFTW_ESTIMATE);

#pragma omp for schedule(static, 1) nowait
        for (size_t i = 0; i < m; i++)
        {
            double __complex__ *in = a + i * n;
            for (int j = 0; j < n; j++)
            {
                buf_thread[j] = in[j] * freq[j];
            }
            fftw_execute_dft(plan, (fftw_complex *)buf_thread, (fftw_complex *)in);
        }

        fftw_destroy_plan(plan);
    }
}

void _FinaliFFT(
    double __complex__ *a,
    const double __complex__ *freq,
    int m, int n, int *mesh,
    double __complex__ *buf)
{
    const int nThread = get_omp_threads();

    double factor = 1.0 / (double)n;

    if (n != mesh[0] * mesh[1] * mesh[2])
    {
        printf("n: %d\n", n);
        printf("mesh: %d %d %d\n", mesh[0], mesh[1], mesh[2]);
        fprintf(stderr, "The size of a is not compatible with mesh\n");
        exit(1);
    }

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();

        double __complex__ *buf_thread = buf + thread_id * n;

        fftw_plan plan = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2], (fftw_complex *)buf_thread, (fftw_complex *)a, FFTW_BACKWARD, FFTW_ESTIMATE);

#pragma omp for schedule(static, 1) nowait
        for (size_t i = 0; i < m; i++)
        {
            double __complex__ *in = a + i * n;
            fftw_execute_dft(plan, (fftw_complex *)in, (fftw_complex *)buf_thread);
            for (int j = 0; j < n; j++)
            {
                // buf_thread[j] = in[j] * conj(freq[j]) * factor;
                in[j] = buf_thread[j] * conj(freq[j]) * factor;
            }
        }

        fftw_destroy_plan(plan);
    }
}

void _PermutationConj(
    double __complex__ *a,
    int m, int n, int *permutation,
    double __complex__ *buf)
{
    const int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();

        double __complex__ *buf_thread = buf + thread_id * n;

#pragma omp for schedule(static, 1) nowait
        for (size_t i = 0; i < m; i++)
        {
            double __complex__ *in = a + i * n;
            for (int j = 0; j < n; j++)
            {
                buf_thread[j] = conj(in[permutation[j]]);
                // buf_thread[permutation[j]] = conj(in[j]);
            }
            memcpy(in, buf_thread, sizeof(double __complex__) * n);
        }
    }
}

#define PI 3.14159265358979323846

void meshgrid(int *range1, int size1, int *range2, int size2, int *range3, int size3, int *output)
{
#pragma omp parallel for collapse(3)
    for (int i = 0; i < size1; i++)
    {
        for (int j = 0; j < size2; j++)
        {
            for (int k = 0; k < size3; k++)
            {
                output[(i * size2 * size3 + j * size3 + k) * 3 + 0] = range1[i];
                output[(i * size2 * size3 + j * size3 + k) * 3 + 1] = range2[j];
                output[(i * size2 * size3 + j * size3 + k) * 3 + 2] = range3[k];
            }
        }
    }
}

void _FREQ(
    double __complex__ *FREQ,
    const int *meshPrim,
    const int *Ls)
{
    int *freq1_q = (int *)malloc(meshPrim[0] * sizeof(int));
    int *freq2_q = (int *)malloc(meshPrim[1] * sizeof(int));
    int *freq3_q = (int *)malloc(meshPrim[2] * sizeof(int));

    for (int i = 0; i < meshPrim[0]; i++)
    {
        freq1_q[i] = i;
    }
    for (int i = 0; i < meshPrim[1]; i++)
    {
        freq2_q[i] = i;
    }
    for (int i = 0; i < meshPrim[2]; i++)
    {
        freq3_q[i] = i;
    }

    int *freq_q = (int *)malloc(meshPrim[0] * meshPrim[1] * meshPrim[2] * 3 * sizeof(int));
    meshgrid(freq1_q, meshPrim[0], freq2_q, meshPrim[1], freq3_q, meshPrim[2], freq_q);

    int *freq1_Q = (int *)malloc(Ls[0] * sizeof(int));
    int *freq2_Q = (int *)malloc(Ls[1] * sizeof(int));
    int *freq3_Q = (int *)malloc((Ls[2] / 2 + 1) * sizeof(int));

    for (int i = 0; i < Ls[0]; i++)
    {
        freq1_Q[i] = i;
    }
    for (int i = 0; i < Ls[1]; i++)
    {
        freq2_Q[i] = i;
    }
    for (int i = 0; i < Ls[2] / 2 + 1; i++)
    {
        freq3_Q[i] = i;
    }

    int *freq_Q = (int *)malloc(Ls[0] * Ls[1] * (Ls[2] / 2 + 1) * 3 * sizeof(int));
    meshgrid(freq1_Q, Ls[0], freq2_Q, Ls[1], freq3_Q, Ls[2] / 2 + 1, freq_Q);

#pragma omp parallel for collapse(6)
    for (int i = 0; i < Ls[0]; i++)
    {
        for (int j = 0; j < Ls[1]; j++)
        {
            for (int k = 0; k < Ls[2] / 2 + 1; k++)
            {
                for (int p = 0; p < meshPrim[0]; p++)
                {
                    for (int q = 0; q < meshPrim[1]; q++)
                    {
                        for (int s = 0; s < meshPrim[2]; s++)
                        {
                            FREQ[(i * Ls[1] * (Ls[2] / 2 + 1) * meshPrim[0] * meshPrim[1] * meshPrim[2] +
                                  j * (Ls[2] / 2 + 1) * meshPrim[0] * meshPrim[1] * meshPrim[2] +
                                  k * meshPrim[0] * meshPrim[1] * meshPrim[2] +
                                  p * meshPrim[1] * meshPrim[2] +
                                  q * meshPrim[2] +
                                  s)] = freq_Q[(i * Ls[1] * (Ls[2] / 2 + 1) + j * (Ls[2] / 2 + 1) + k) * 3 + 0] * freq_q[(p * meshPrim[1] * meshPrim[2] + q * meshPrim[2] + s) * 3 + 0] / (double)(Ls[0] * meshPrim[0]) +
                                        freq_Q[(i * Ls[1] * (Ls[2] / 2 + 1) + j * (Ls[2] / 2 + 1) + k) * 3 + 1] * freq_q[(p * meshPrim[1] * meshPrim[2] + q * meshPrim[2] + s) * 3 + 1] / (double)(Ls[1] * meshPrim[1]) +
                                        freq_Q[(i * Ls[1] * (Ls[2] / 2 + 1) + j * (Ls[2] / 2 + 1) + k) * 3 + 2] * freq_q[(p * meshPrim[1] * meshPrim[2] + q * meshPrim[2] + s) * 3 + 2] / (double)(Ls[2] * meshPrim[2]);
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < Ls[0] * Ls[1] * (Ls[2] / 2 + 1) * meshPrim[0] * meshPrim[1] * meshPrim[2]; i++)
    {
        FREQ[i] = cexp(-2.0 * PI * I * FREQ[i]);
    }

    free(freq1_q);
    free(freq2_q);
    free(freq3_q);
    free(freq_q);
    free(freq1_Q);
    free(freq2_Q);
    free(freq3_Q);
    free(freq_Q);
}

#undef PI

void _permutation(int nx, int ny, int nz, int shift_x, int shift_y, int shift_z, int *res)
{

#pragma omp parallel for collapse(3)
    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                int ix2 = (nx - ix - shift_x) % nx;
                int iy2 = (ny - iy - shift_y) % ny;
                int iz2 = (nz - iz - shift_z) % nz;
                int loc = ix2 * ny * nz + iy2 * nz + iz2;
                int loc_now = ix * ny * nz + iy * nz + iz;
                res[loc] = loc_now;
            }
        }
    }
}

void _get_permutation(
    const int *meshPrim,
    int *res)
{
    int nGridPrim = meshPrim[0] * meshPrim[1] * meshPrim[2];

#pragma omp parallel sections
    {
#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 0, 0, &res[0 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 0, 1, &res[1 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 1, 0, &res[2 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 0, 1, 1, &res[3 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 0, 0, &res[4 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 0, 1, &res[5 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 1, 0, &res[6 * nGridPrim]);

#pragma omp section
        _permutation(meshPrim[0], meshPrim[1], meshPrim[2], 1, 1, 1, &res[7 * nGridPrim]);
    }
}

int _get_loc(
    const int freq,
    const int mesh)
{
    int max_freq = mesh / 2;
    int min_freq = -mesh / 2;

    if (mesh % 2 == 0)
    {
        max_freq = mesh / 2 - 1;
        min_freq = -mesh / 2;
    }

    if (freq > max_freq || freq < min_freq)
    {
        return -1;
    }

    if (freq >= 0)
    {
        return freq;
    }
    else
    {
        int shift = mesh / 2;
        if (mesh % 2 == 1)
        {
            shift += 1;
        }
        return (freq - min_freq) + shift;
    }
}

int _get_loc2(
    const int freq,
    const int mesh) // for real signal, the freq and loc must always be non-negative !
{
    int max_freq = (mesh / 2) + 1;

    if (freq >= 0 && freq < max_freq)
    {
        return freq;
    }
    else
    {
        return -1;
    }
}

int _get_freq(
    const int loc,
    const int mesh)
{
    int mid_loc = mesh / 2;
    if (mesh % 2 == 1)
    {
        mid_loc += 1;
    }

    if ((loc < 0) || (loc >= mesh))
    {
        printf("loc: %d, mesh: %d\n", loc, mesh);
        exit(1);
    }

    if (loc < mid_loc)
    {
        return loc;
    }
    else
    {
        return loc - mesh;
    }
}

int _get_freq2(
    const int loc,
    const int mesh)
{
    int loc_max = mesh / 2 + 1;

    if (loc >= 0 && loc < loc_max)
    {
        return loc;
    }
    else
    {
        return -1;
    }
}

void map_fftfreq(int *mesh_source, int *mesh_target, int *res)
{
    int nGrid = mesh_source[0] * mesh_source[1] * mesh_source[2];

#pragma omp parallel for
    for (int i = 0; i < nGrid; i++)
    {
        int ix_loc = i / (mesh_source[1] * mesh_source[2]);
        int iy_loc = (i % (mesh_source[1] * mesh_source[2])) / mesh_source[2];
        int iz_loc = i % mesh_source[2];

        int ix_freq = _get_freq(ix_loc, mesh_source[0]);
        int iy_freq = _get_freq(iy_loc, mesh_source[1]);
        int iz_freq = _get_freq(iz_loc, mesh_source[2]);

        int ix_target = _get_loc(ix_freq, mesh_target[0]);
        int iy_target = _get_loc(iy_freq, mesh_target[1]);
        int iz_target = _get_loc(iz_freq, mesh_target[2]);

        if (ix_target == -1 || iy_target == -1 || iz_target == -1)
        {
            res[i] = -1;
        }
        else
        {
            res[i] = ix_target * mesh_target[1] * mesh_target[2] + iy_target * mesh_target[2] + iz_target;
        }

        res[i] = ix_target * mesh_target[1] * mesh_target[2] + iy_target * mesh_target[2] + iz_target;
    }
}

void map_rfftfreq(int *mesh_source, int *mesh_target, int *res)
{
    int nGrid = mesh_source[0] * mesh_source[1] * (mesh_source[2] / 2 + 1);

#pragma omp parallel for
    for (int i = 0; i < nGrid; i++)
    {
        int ix_loc = i / (mesh_source[1] * (mesh_source[2] / 2 + 1));
        int iy_loc = (i % (mesh_source[1] * (mesh_source[2] / 2 + 1))) / (mesh_source[2] / 2 + 1);
        int iz_loc = i % (mesh_source[2] / 2 + 1);

        int ix_freq = _get_freq(ix_loc, mesh_source[0]);
        int iy_freq = _get_freq(iy_loc, mesh_source[1]);
        int iz_freq = _get_freq2(iz_loc, mesh_source[2]);

        if (iz_freq == -1)
        {
            printf("iz_loc: %d, mesh_source[2]: %d\n", iz_loc, mesh_source[2]);
            exit(1);
        }

        int ix_target = _get_loc(ix_freq, mesh_target[0]);
        int iy_target = _get_loc(iy_freq, mesh_target[1]);
        int iz_target = _get_loc2(iz_freq, mesh_target[2]);

        if (ix_target == -1 || iy_target == -1 || iz_target == -1)
        {
            res[i] = -1;
        }
        else
        {
            res[i] = ix_target * mesh_target[1] * (mesh_target[2] / 2 + 1) + iy_target * (mesh_target[2] / 2 + 1) + iz_target;
        }
    }
}