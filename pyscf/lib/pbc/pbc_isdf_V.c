#include "fft.h"
#include <omp.h>

int get_omp_threads();
int omp_get_thread_num();

void _construct_V(int *mesh,
                  int naux,
                  double *auxBasis,
                  double *CoulG,
                  double *V,
                  const int BunchSize,
                  double *buf,         // use the ptr of the ptr to ensure that the memory for each thread is aligned
                  const int buffersize // must be a multiple of 16 to ensure memory alignment
)
{
    // print all the input info 

    // printf("mesh       : %d %d %d\n", mesh[0], mesh[1], mesh[2]);
    // printf("naux       : %d\n", naux);
    // printf("BunchSize  : %d\n", BunchSize);
    // printf("buffersize : %d\n", buffersize);
    // printf("auxBasis   : %p\n", auxBasis);
    // printf("CoulG      : %p\n", CoulG);
    // printf("V          : %p\n", V);
    // printf("buf        : %p\n", buf);

    static const int SMALL_SIZE = 8;

    const int nThread = get_omp_threads();
    const int nBunch = ((naux / BunchSize) / nThread) * nThread; // dispatch evenly
    const int nLeft = naux - nBunch * BunchSize;

    // print the dispatch info

    // printf("nThread : %d\n", nThread);
    // printf("nBunch  : %d\n", nBunch);
    // printf("nLeft   : %d\n", nLeft);

    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};

    // printf("mesh_complex: %d %d %d\n", mesh_complex[0], mesh_complex[1], mesh_complex[2]);

    const int n_real = mesh[0] * mesh[1] * mesh[2];
    const int n_complex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const double fac = 1. / (double)n_real;

    // create plan for fft

    fftw_plan p_forward = fftw_plan_many_dft_r2c(
        3, mesh, BunchSize, auxBasis, mesh, 1, n_real, (fftw_complex *)buf, mesh_complex, 1, n_complex, FFTW_ESTIMATE);
    fftw_plan p_backward = fftw_plan_many_dft_c2r(
        3, mesh, BunchSize, (fftw_complex *)buf, mesh_complex, 1, n_complex, V, mesh, 1, n_real, FFTW_ESTIMATE);

    // execute parallelly sharing the same plan

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();
        double *buf_thread = buf + thread_id * (size_t)buffersize;
        size_t bunch_i, bunch_start, bunch_end, j, k;
        double *ptr;

        // printf("thread_id: %d\n", thread_id);

#pragma omp for schedule(static)
        for (bunch_i = 0; bunch_i < nBunch; ++bunch_i)
        // for (bunch_i = 0; bunch_i < 0; ++bunch_i)
        {
            bunch_start = bunch_i * BunchSize;
            bunch_end = bunch_start + BunchSize;

            // forward transform

            fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real, (fftw_complex *)buf_thread);

            // multiply CoulG

            ptr = buf_thread;

            for (j = bunch_start; j < bunch_end; ++j)
            {
                for (k = 0; k < n_complex; ++k)
                {
                    *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                    *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                }
            }

            // backward transform

            fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf_thread, V + (size_t)bunch_start * (size_t)n_real);

            // scale

            ptr = V + (size_t)bunch_start * (size_t)n_real;

            for (j = bunch_start; j < bunch_end; ++j)
            {
                for (k = 0; k < n_real; ++k)
                {
                    *ptr++ *= fac; /// TODO: use ISPC to accelerate
                }
            }
        }
    }

    // destroy plan

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);

    // printf("nLeft: %d\n", nLeft);

    // return;

    if (nLeft > 0)
    {
        if ((nLeft <= SMALL_SIZE) && (nLeft <= BunchSize))
        // if (1)
        {
            // use single thread to handle the left

            // printf("use single thread to handle the left\n");

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // printf("bunch_start : %d\n", bunch_start);
            // printf("bunch_end   : %d\n", bunch_end);   

            // create plan

            fftw_plan p_forward = fftw_plan_many_dft_r2c(
                3, mesh, nLeft, auxBasis + bunch_start * n_real, mesh, 1, n_real, (fftw_complex *)buf, mesh_complex, 1, n_complex, FFTW_ESTIMATE);
            fftw_plan p_backward = fftw_plan_many_dft_c2r(
                3, mesh, nLeft, (fftw_complex *)buf, mesh_complex, 1, n_complex, V + bunch_start * n_real, mesh, 1, n_real, FFTW_ESTIMATE);

            // forward transform

            // printf("forward transform\n");
            // printf("buf = %p\n", buf);
            // printf("auxBasis   = %p\n", auxBasis + (size_t)bunch_start * (size_t)n_real);
            // fflush(stdout);

            fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real, (fftw_complex *)buf);

            // printf("forward transform done\n");
            // fflush(stdout);

            // multiply CoulG

            double *ptr = buf;

            for (int j = bunch_start; j < bunch_end; ++j)
            {
                for (int k = 0; k < n_complex; ++k)
                {
                    *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                    *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                }
            }

            // backward transform

            // printf("backward transform\n");
            // printf("buf = %p\n", buf);
            // printf("V   = %p\n", V);
            // printf("stride = %lu\n", (size_t)n_real * bunch_start);
            // printf("V   = %p\n", V + (size_t)bunch_start * (size_t)n_real);
            // fflush(stdout);
            // printf("bunch_start: %d\n", bunch_start);
            // printf("n_real:      %d\n", n_real);
            // printf("buf required = %d\n", n_complex * 2 * nLeft);
            // printf("buf max      = %d\n", buffersize*nThread);

            fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf, V + (size_t)bunch_start * (size_t)n_real);

            // scale

            ptr = V + (size_t)bunch_start * (size_t)n_real;

            for (int j = bunch_start; j < bunch_end; ++j)
            {
                for (int k = 0; k < n_real; ++k)
                {
                    *ptr++ *= fac; /// TODO: use ISPC to accelerate
                }
            }

            // destroy plan

            fftw_destroy_plan(p_forward);
            fftw_destroy_plan(p_backward);
        }
        else
        {

            // printf("use parallel thread to handle the left\n");
            // printf("n_real    = %d\n", n_real);
            // printf("n_complex = %d\n", n_complex);

            // use parallel thread to handle the left, assume the nTransform is 1

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // printf("bunch_start: %d\n", bunch_start);   
            // printf("bunch_end: %d\n", bunch_end);

            // create plan

            fftw_plan p_forward = fftw_plan_dft_r2c(3, mesh, auxBasis + bunch_start * n_real, (fftw_complex *)buf, FFTW_ESTIMATE);
            fftw_plan p_backward = fftw_plan_dft_c2r(3, mesh, (fftw_complex *)buf, V + bunch_start * n_real, FFTW_ESTIMATE);

            size_t nbuf_per_thread = ((n_complex * 2 + 15)/16)*16; // make sure the memory is aligned

#pragma omp parallel num_threads(nThread)
            {
                int thread_id = omp_get_thread_num();
                // printf("thread_id: %d\n", thread_id);
                // printf("buffsize per thread: %d\n", nbuf_per_thread);
                double *buf_thread = buf + thread_id * (size_t)nbuf_per_thread;
                size_t k;
                double *ptr;

#pragma omp for schedule(static)
                for (size_t j = bunch_start; j < bunch_end; ++j)
                {
                
                    // forward transform

                    fftw_execute_dft_r2c(p_forward, auxBasis + j * (size_t)n_real, (fftw_complex *)buf_thread);

                    // multiply CoulG

                    ptr = buf_thread;

                    for (k = 0; k < n_complex; ++k)
                    {
                        *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                        *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                    }

                    // backward transform

                    fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf_thread, V + j * (size_t)n_real);

                    // scale

                    ptr = V + j * (size_t)n_real;

                    for (k = 0; k < n_real; ++k)
                    {
                        *ptr++ *= fac; /// TODO: use ISPC to accelerate
                    }
                }
            }

            // destroy plan

            fftw_destroy_plan(p_forward);
            fftw_destroy_plan(p_backward);
        }
    }
}