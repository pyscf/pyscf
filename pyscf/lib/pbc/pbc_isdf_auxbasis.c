#include "vhf/fblas.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

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

    printf("nThread  : %d\n", nThread);
    printf("nBunch   : %d\n", nBunch);
    printf("nLeft    : %d\n", nLeft);
    printf("BunchSize: %d\n", BunchSize);
    printf("n        : %d\n", n);
    printf("nrhs     : %d\n", nrhs);

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
                int thread_id = omp_get_thread_num();

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