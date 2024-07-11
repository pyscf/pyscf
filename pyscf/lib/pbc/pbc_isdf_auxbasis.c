#include "vhf/fblas.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

int get_omp_threads();
int omp_get_thread_num();

//

void Cholesky(double *A, int n)
{
    // A will be overwritten with the lower triangular Cholesky factor
    lapack_int info;
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', n, A, n);
    if (info != 0)
    {
        fprintf(stderr, "Cholesky decomposition failed: %d\n", info);
        exit(1);
    }
}

void Solve_LLTEqualB_Parallel(
    const int n,
    const double *a, // call cholesky first!
    double *b,
    const int nrhs,
    const int BunchSize)
{
    int nThread = get_omp_threads();

    int nBunch = (nrhs / BunchSize);
    int nLeft = nrhs - nBunch * BunchSize;

    // printf("nThread  : %d\n", nThread);
    // printf("nBunch   : %d\n", nBunch);
    // printf("nLeft    : %d\n", nLeft);
    // printf("BunchSize: %d\n", BunchSize);
    // printf("n        : %d\n", n);
    // printf("nrhs     : %d\n", nrhs);

#pragma omp parallel num_threads(nThread)
    {
        double *ptr_b;
        lapack_int info;

#pragma omp for schedule(static, 1) nowait
        for (int i = 0; i < nBunch; i++)
        {
            ptr_b = b + BunchSize * i;

            // forward transform

            info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', n, BunchSize, a, n, ptr_b, nrhs);

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

                double *ptr_b = b + BunchSize * nBunch;

                lapack_int info;

                // forward transform

                info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', n, nLeft, a, n, ptr_b, nrhs);

                if (info != 0)
                {
                    fprintf(stderr, "Solving system failed: %d\n", info);
                    exit(1);
                }
            }
        }
    }
}

void ColPivotQRRelaCut(
    double *aoPaironGrid, // (nPair, nGrid)
    const int nPair,
    const int nGrid,
    const int max_rank,
    const double cutoff, // abs_cutoff
    const double relacutoff,
    int *pivot,
    double *R,
    int *npt_find,
    double *thread_buffer, // (nThread, nGrid)
    double *global_buffer) // nGrid
{
    static const int INC = 1;

    // printf("nPair: %d\n", nPair);
    // printf("nGrid: %d\n", nGrid);
    // printf("max_rank: %d\n", max_rank);
    // printf("cutoff: %f\n", cutoff);

    double *Q = aoPaironGrid;

    for (int i = 0; i < nGrid; ++i)
    {
        pivot[i] = i;
    }

    int nThread = get_omp_threads();
    *npt_find = 0;

    int *reduce_indx_buffer = (int *)(thread_buffer + nThread * nGrid);

    int i;

    int argmaxnorm = 0;
    double maxnorm = 0.0;

    for (i = 0; i < max_rank; i++)
    {
        // printf("i: %d\n", i);

#pragma omp parallel num_threads(nThread)
        {

            int thread_id = omp_get_thread_num();
            double *buf = thread_buffer + thread_id * nGrid;
            memset(buf, 0, sizeof(double) * nGrid);

            int j, k;

            double *dptr;

            //// 1. determine the arg of maxinaml norm

#pragma omp for schedule(static)
            for (j = 0; j < nPair; j++)
            {
                dptr = Q + j * nGrid;
                for (k = i; k < nGrid; k++)
                {
                    buf[k] += dptr[k] * dptr[k];
                }
            }

            int bunchsize = (nGrid - i) / nThread + 1;
            int begin_id = i + thread_id * bunchsize;
            int end_id = i + (thread_id + 1) * bunchsize;
            if (thread_id == nThread - 1)
            {
                end_id = nGrid;
            }

            if (begin_id >= nGrid)
            {
                begin_id = nGrid;
            }

            if (end_id > nGrid)
            {
                end_id = nGrid;
            }

            memcpy(global_buffer + begin_id, thread_buffer + begin_id, sizeof(double) * (end_id - begin_id));

            for (j = 1; j < nThread; j++)
            {
                dptr = thread_buffer + j * nGrid;
                for (k = begin_id; k < end_id; ++k)
                {
                    global_buffer[k] += dptr[k];
                }
            }

            // get the local max

            if (begin_id < end_id)
            {
                double max_norm2 = global_buffer[begin_id];
                reduce_indx_buffer[thread_id] = begin_id;
                for (j = begin_id + 1; j < end_id; j++)
                {
                    if (global_buffer[j] > max_norm2)
                    {
                        max_norm2 = global_buffer[j];
                        reduce_indx_buffer[thread_id] = j;
                    }
                }
            }
            else
            {
                reduce_indx_buffer[thread_id] = begin_id - 1;
            }

            // printf("max_norm2: %.3e\n", max_norm2);

#pragma omp barrier

#pragma omp single
            {
                // printf("--------------------------------\n");
                maxnorm = global_buffer[reduce_indx_buffer[0]];
                argmaxnorm = reduce_indx_buffer[0];
                // printf("maxnorm: %.3e\n", maxnorm);
                // printf("argmaxnorm: %d\n", argmaxnorm);
                for (j = 1; j < nThread; j++)
                {
                    if (global_buffer[reduce_indx_buffer[j]] > maxnorm)
                    {
                        // printf("j = %d\n", j);
                        // printf("global_buffer[reduce_indx_buffer[j]]: %.3e\n", global_buffer[reduce_indx_buffer[j]]);

                        maxnorm = global_buffer[reduce_indx_buffer[j]];
                        argmaxnorm = reduce_indx_buffer[j];

                        // printf("maxnorm: %.3e\n", maxnorm);
                        // printf("argmaxnorm: %d\n", argmaxnorm);
                    }
                }

                // printf("i = %d\n", i);
                // printf("argmaxnorm = %d\n", argmaxnorm);

                int tmp;
                tmp = pivot[i];
                pivot[i] = pivot[argmaxnorm];
                pivot[argmaxnorm] = tmp;

                // printf("argmaxnorm: %d\n", argmaxnorm);
                // printf("tmp = %d\n", tmp);
                // printf("pivot[i] = %d\n", pivot[i]);
                // printf("pivot[argmaxnorm] = %d\n", pivot[argmaxnorm]);
                // printf("--------------------------------\n");

                maxnorm = sqrt(maxnorm);
                R[i * nGrid + i] = maxnorm;
                // printf("R[%3d,%3d] = maxnorm = %10.3e\n", i, i, maxnorm);
            }

#pragma omp barrier

            //// 2. switch

            ///// Q

#pragma omp for schedule(static) nowait
            for (j = 0; j < nPair; ++j)
            {
                dptr = Q + j * nGrid;
                double tmp;
                tmp = dptr[i];
                dptr[i] = dptr[argmaxnorm];
                dptr[argmaxnorm] = tmp;
                dptr[i] /= maxnorm;
            }

            ///// R

#pragma omp for schedule(static)
            for (j = 0; j < i; ++j)
            {
                dptr = R + i * nGrid;
                double tmp;
                tmp = dptr[i];
                dptr[i] = dptr[argmaxnorm];
                dptr[argmaxnorm] = tmp;
            }

            //// 3. perform Schimidt decomposition

            ///// calculate the inner product

            memset(buf, 0, sizeof(double) * nGrid);

            int nleft = nGrid - i - 1;

#pragma omp for schedule(static)
            for (j = 0; j < nPair; ++j)
            {
                dptr = Q + j * nGrid;
                daxpy_(&nleft, dptr + i, dptr + i + 1, &INC, buf + i + 1, &INC);
            }

            bunchsize = nleft / nThread;
            begin_id = i + 1 + thread_id * bunchsize;
            end_id = i + 1 + (thread_id + 1) * bunchsize;
            if (thread_id == nThread - 1)
            {
                end_id = nGrid;
            }

            memcpy(global_buffer + begin_id, thread_buffer + begin_id, sizeof(double) * (end_id - begin_id));

            for (j = 1; j < nThread; j++)
            {
                dptr = thread_buffer + j * nGrid;
                for (k = begin_id; k < end_id; ++k)
                {
                    global_buffer[k] += dptr[k];
                }
            }

#pragma omp barrier

            // project out

            double *inner_prod = global_buffer + i + 1;

#pragma omp for schedule(static) nowait
            for (j = 0; j < nPair; ++j)
            {
                dptr = Q + j * nGrid;
                double alpha = -dptr[i];
                daxpy_(&nleft, &alpha, inner_prod, &INC, dptr + i + 1, &INC);
            }

            // update R

#pragma omp single
            {
                memcpy(R + i * nGrid + i + 1, inner_prod, sizeof(double) * nleft);
            }
        }

        if ((maxnorm < cutoff) || (maxnorm < R[0] * relacutoff))
        {
            break;
        }
        else
        {
            (*npt_find)++;
        }
    }
}

void ColPivotQR(
    double *aoPaironGrid, // (nPair, nGrid)
    const int nPair,
    const int nGrid,
    const int max_rank,
    const double cutoff,
    int *pivot,
    double *R,
    int *npt_find,
    double *thread_buffer, // (nThread, nGrid)
    double *global_buffer) // nGrid
{
    ColPivotQRRelaCut(
        aoPaironGrid, nPair, nGrid, max_rank, cutoff, 0.0, pivot, R, npt_find, thread_buffer, global_buffer);
}

void NP_d_ik_jk_ijk(
    const double *A,
    const double *B,
    double *out,
    const int nA,
    const int nB,
    const int nC)
{
    // printf("nA: %d\n", nA);
    // printf("nB: %d\n", nB);
    // printf("nC: %d\n", nC);

    int i, j;
#pragma omp parallel for private(i, j)
    for (i = 0; i < nA * nB; ++i)
    {
        int i1 = i / nB;
        int i2 = i % nB;
        for (j = 0; j < nC; ++j)
        {
            out[i * nC + j] = A[i1 * nC + j] * B[i2 * nC + j];
        }
    }
}

void NPdsliceFirstCol(double *out, const double *a, size_t ncol_left, size_t nrow, size_t ncol)
{
#pragma omp parallel
    {
        size_t i;
#pragma omp for schedule(static)
        for (i = 0; i < nrow; i++)
        {
            memcpy(out + i * ncol_left, a + i * ncol, sizeof(double) * ncol_left);
        }
    }
}

void CalculateNormRemained(
    const double *InnerProd, // (nIP, nPntPotential)
    const int nIP,
    const int nPntPotential,
    const double *aoPaironGrid, // (nPair, nPntPotential)
    const int nPair,
    double *thread_buffer,
    double *global_buffer)
{
    int nThread = get_omp_threads();

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();
        double *buf = thread_buffer + thread_id * nPntPotential;
        memset(buf, 0, sizeof(double) * nPntPotential);

        int i, j;

        double *dptr;
        const double *cdptr;

#pragma omp for schedule(static)
        for (i = 0; i < nPair; i++)
        {
            cdptr = aoPaironGrid + i * nPntPotential;
            for (j = 0; j < nPntPotential; j++)
            {
                buf[j] += cdptr[j] * cdptr[j];
            }
        }

        int bunchsize = nPntPotential / nThread;
        int begin_id = thread_id * bunchsize;
        int end_id = (thread_id + 1) * bunchsize;
        if (thread_id == nThread - 1)
        {
            end_id = nPntPotential;
        }

        memcpy(global_buffer + begin_id, thread_buffer + begin_id, sizeof(double) * (end_id - begin_id));

        for (i = 1; i < nThread; i++)
        {
            dptr = thread_buffer + i * nPntPotential;
            for (j = begin_id; j < end_id; j++)
            {
                global_buffer[j] += dptr[j];
            }
        }

        // if (begin_id == 0)
        // {
        //     printf("global_buffer[0]: %f\n", sqrt(global_buffer[0]));
        // }

        for (i = 0; i < nIP; i++)
        {
            const double *dptr = InnerProd + i * nPntPotential;
            for (j = begin_id; j < end_id; j++)
            {
                global_buffer[j] -= dptr[j] * dptr[j];
            }
        }

        for (j = begin_id; j < end_id; j++)
        {
            global_buffer[j] = sqrt(global_buffer[j]);
        }
    }
}

void PackAFirstCol(
    const double *A, //
    double *out,     //
    const int nRow,
    const int nACol,
    const int nFirst)
{
}

void PackABwithSlice(
    const double *A, //
    const double *B, //
    double *out,     //
    const int nRow,
    const int nACol,
    const int nBCol,
    const int *SliceB,
    const int nSliceB,
    double *Packbuf)
{
    int i, j;
    int nThread = get_omp_threads();

    const int nOutCol = nACol + nSliceB;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (i = 0; i < nRow; ++i)
    {
        memcpy(Packbuf + i * nOutCol, A + i * nACol, sizeof(double) * nACol);
        for (j = 0; j < nSliceB; ++j)
        {
            Packbuf[i * nOutCol + nACol + j] = B[i * nBCol + SliceB[j]];
        }
    }

    memcpy(out, Packbuf, sizeof(double) * nRow * nOutCol);
}

void PackABwithABSlice(
    const double *A, //
    const double *B, //
    double *out,     //
    const int nRow,
    const int nACol,
    const int nBCol,
    const int *Slice,
    const int nSlice,
    double *Packbuf,
    double *thread_buffer)
{
    int i, j;
    int nThread = get_omp_threads();

    const int nOutCol = nSlice;

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();
        double *buf = thread_buffer + thread_id * (nACol + nBCol);

#pragma omp for schedule(static)
        for (i = 0; i < nRow; ++i)
        {
            memcpy(buf, A + i * nACol, sizeof(double) * nACol);
            memcpy(buf + nACol, B + i * nBCol, sizeof(double) * nBCol);
            for (j = 0; j < nSlice; ++j)
            {
                Packbuf[i * nSlice + j] = buf[Slice[j]];
            }
        }
    }
    memcpy(out, Packbuf, sizeof(double) * nRow * nOutCol);
}

void PackAB(
    const double *A, //
    const double *B, //
    double *out,     //
    const int nRow,
    const int nACol,
    const int nBCol)
{
    int i, j;
    int nThread = get_omp_threads();

    const int nOutCol = nACol + nBCol;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (i = 0; i < nRow; ++i)
    {
        memcpy(out + i * nOutCol, A + i * nACol, sizeof(double) * nACol);
        memcpy(out + i * nOutCol + nACol, B + i * nBCol, sizeof(double) * nBCol);
    }
}
