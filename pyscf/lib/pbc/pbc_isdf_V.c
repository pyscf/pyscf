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

    static const int SMALL_SIZE = 8;

    const int nThread = get_omp_threads();
    const int nBunch = ((naux / BunchSize) / nThread) * nThread; // dispatch evenly
    const int nLeft = naux - nBunch * BunchSize;

    // print the dispatch info

    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};

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

    if (nLeft > 0)
    {
        if ((nLeft <= SMALL_SIZE) && (nLeft <= BunchSize))
        // if (1)
        {
            // use single thread to handle the left

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // create plan

            fftw_plan p_forward = fftw_plan_many_dft_r2c(
                3, mesh, nLeft, auxBasis + bunch_start * n_real, mesh, 1, n_real, (fftw_complex *)buf, mesh_complex, 1, n_complex, FFTW_ESTIMATE);
            fftw_plan p_backward = fftw_plan_many_dft_c2r(
                3, mesh, nLeft, (fftw_complex *)buf, mesh_complex, 1, n_complex, V + bunch_start * n_real, mesh, 1, n_real, FFTW_ESTIMATE);

            // forward transform

            fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real, (fftw_complex *)buf);

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

            // use parallel thread to handle the left, assume the nTransform is 1

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // create plan

            fftw_plan p_forward = fftw_plan_dft_r2c(3, mesh, auxBasis + bunch_start * n_real, (fftw_complex *)buf, FFTW_ESTIMATE);
            fftw_plan p_backward = fftw_plan_dft_c2r(3, mesh, (fftw_complex *)buf, V + bunch_start * n_real, FFTW_ESTIMATE);

            size_t nbuf_per_thread = ((n_complex * 2 + 15) / 16) * 16; // make sure the memory is aligned

#pragma omp parallel num_threads(nThread)
            {
                int thread_id = omp_get_thread_num();
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

void _construct_V2(int *mesh,
                   int naux,
                   double *auxBasis,
                   double *CoulG,
                   double *V,
                   double *auxBasisFFT,
                   const int BunchSize,
                   double *buf,          // use the ptr of the ptr to ensure that the memory for each thread is aligned
                   const int buffersize, // must be a multiple of 16 to ensure memory alignment
                   const int CONSTRUCT_V)
{
    // printf("CONSTRUCT_V: %d\n", CONSTRUCT_V);

    // print all the input info

    static const int SMALL_SIZE = 8;

    const int nThread = get_omp_threads();
    const int nBunch = ((naux / BunchSize) / nThread) * nThread; // dispatch evenly
    const int nLeft = naux - nBunch * BunchSize;

    // print the dispatch info

    int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};

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

            // copy

            memcpy(auxBasisFFT + (size_t)bunch_start * (size_t)n_complex * 2, buf_thread, (size_t)BunchSize * (size_t)n_complex * sizeof(double) * 2);

            if (CONSTRUCT_V > 0)
            {
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
    }

    // destroy plan

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);

    if (nLeft > 0)
    {
        if ((nLeft <= SMALL_SIZE) && (nLeft <= BunchSize))
        // if (1)
        {
            // use single thread to handle the left

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // create plan

            fftw_plan p_forward = fftw_plan_many_dft_r2c(
                3, mesh, nLeft, auxBasis + bunch_start * n_real, mesh, 1, n_real, (fftw_complex *)buf, mesh_complex, 1, n_complex, FFTW_ESTIMATE);
            fftw_plan p_backward = fftw_plan_many_dft_c2r(
                3, mesh, nLeft, (fftw_complex *)buf, mesh_complex, 1, n_complex, V + bunch_start * n_real, mesh, 1, n_real, FFTW_ESTIMATE);

            // forward transform

            fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real, (fftw_complex *)buf);

            // multiply CoulG

            double *ptr = buf;

            // copy

            memcpy(auxBasisFFT + (size_t)bunch_start * (size_t)n_complex * 2, buf, (size_t)nLeft * (size_t)n_complex * sizeof(double) * 2);

            if (CONSTRUCT_V > 0)
            {
                for (int j = bunch_start; j < bunch_end; ++j)
                {
                    for (int k = 0; k < n_complex; ++k)
                    {
                        *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                        *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                    }
                }

                // backward transform

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
            }

            // destroy plan

            fftw_destroy_plan(p_forward);
            fftw_destroy_plan(p_backward);
        }
        else
        {

            // use parallel thread to handle the left, assume the nTransform is 1

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // create plan

            fftw_plan p_forward = fftw_plan_dft_r2c(3, mesh, auxBasis + bunch_start * n_real, (fftw_complex *)buf, FFTW_ESTIMATE);
            fftw_plan p_backward = fftw_plan_dft_c2r(3, mesh, (fftw_complex *)buf, V + bunch_start * n_real, FFTW_ESTIMATE);

            size_t nbuf_per_thread = ((n_complex * 2 + 15) / 16) * 16; // make sure the memory is aligned

#pragma omp parallel num_threads(nThread)
            {
                int thread_id = omp_get_thread_num();
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

                    // copy

                    memcpy(auxBasisFFT + j * (size_t)n_complex * 2, buf_thread, (size_t)n_complex * sizeof(double) * 2);

                    if (CONSTRUCT_V > 0)
                    {
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
            }

            // destroy plan

            fftw_destroy_plan(p_forward);
            fftw_destroy_plan(p_backward);
        }
    }
}

void _construct_W_multiG(
    int naux,
    int p0,
    int p1,
    double *auxBasisFFT,
    double *CoulG)
{
    int ngrid = p1 - p0;
    int nThread = get_omp_threads();

    int i;

    const double *ptr_G = CoulG + p0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (i = 0; i < naux; i++)
    {
        int j;
        double *ptr_basis = auxBasisFFT + i * ngrid * 2;
        for (j = 0; j < ngrid; j++)
        {
            ptr_basis[j * 2] *= ptr_G[j];
            ptr_basis[j * 2 + 1] *= ptr_G[j];
        }
    }
}