#include "fft.h"
#include <omp.h>
#include <string.h>
#include <complex.h>
#include "vhf/fblas.h"
#include <math.h>

int get_omp_threads();
int omp_get_thread_num();

void _construct_J(
    int *mesh,
    double *DensityR,
    double *CoulG,
    double *J)
{
    const int nThread = get_omp_threads();
    // int mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};
    const int n_real = mesh[0] * mesh[1] * mesh[2];
    // const int n_complex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const double fac = 1. / (double)n_real;

    fftw_complex *DensityR_complex = fftw_malloc(sizeof(double __complex__) * n_real);
    fftw_complex *buf = fftw_malloc(sizeof(double __complex__) * n_real);
    fftw_complex *J_complex = fftw_malloc(sizeof(double __complex__) * n_real);

    memset(buf, 0, sizeof(double __complex__) * n_real);
    memset(J_complex, 0, sizeof(double __complex__) * n_real);
    memset(DensityR_complex, 0, sizeof(double __complex__) * n_real);

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (int i = 0; i < n_real; ++i)
    {
        DensityR_complex[i][0] = DensityR[i];
    }

    fftw_plan p_forward = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2], DensityR_complex, (fftw_complex *)buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan p_backward = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2], (fftw_complex *)buf, J_complex, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p_forward);

    double *ptr = (double *)buf;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (int i = 0; i < n_real; i++)
    {
        ptr[i * 2] *= CoulG[i] * fac;
        ptr[i * 2 + 1] *= CoulG[i] * fac;
    }

    fftw_execute(p_backward);

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (int i = 0; i < n_real; i++)
    {
        J[i] = J_complex[i][0];
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);

    fftw_free(buf);
    fftw_free(DensityR_complex);
    fftw_free(J_complex);
}

void _fn_J_dmultiplysum(double *out,
                        const int nrow, const int ncol,
                        const double *a,
                        const int nrow_a, const int ncol_a,
                        const int row_a_shift,
                        const int col_a_shift,
                        const double *b,
                        const int nrow_b, const int ncol_b,
                        const int row_b_shift,
                        const int col_b_shift)
{
    static const int BUNCHSIZE = 512;

    const double *pa = a + row_a_shift * ncol_a + col_a_shift;
    const double *pb = b + row_b_shift * ncol_b + col_b_shift;

    memset(out, 0, sizeof(double) * ncol);

    const int nThread = get_omp_threads();
    const int nBunch = ncol / BUNCHSIZE + 1;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (int i = 0; i < nBunch; i++)
    {
        int bunch_start = i * BUNCHSIZE;
        int bunch_end = (i + 1) * BUNCHSIZE;
        if (bunch_end > ncol)
        {
            bunch_end = ncol;
        }
        if (bunch_end > ncol)
        {
            bunch_end = ncol;
        }

        for (int j = 0; j < nrow; j++)
        {
            const double *ppa = pa + j * ncol_a;
            const double *ppb = pb + j * ncol_b;
            for (int k = bunch_start; k < bunch_end; k++)
            {
                out[k] += ppa[k] * ppb[k];
            }
        }
    }
}

void _Pack_Matrix_SparseRow_DenseCol(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    int *RowLoc,
    const int ColBegin,
    const int ColEnd)
{
    if (ColEnd - ColBegin <= 0)
    {
        return;
    }

    if (ColEnd < (ColBegin + ncol_source))
    {
        printf("ColEnd<ColBegin+ncol_source\n");
        exit(1);
    }

    if (ColEnd > ncol_target)
    {
        printf("ColEnd>ncol_target\n");
        exit(1);
    }

    int i;

    for (i = 0; i < nrow_source; i++)
    {
        int row_loc = RowLoc[i];
        memcpy(target + row_loc * ncol_target + ColBegin, source + i * ncol_source, sizeof(double) * ncol_source);
    }
}

void _Reorder_Grid_to_Original_Grid(int ngrid, int *gridID, double *Density_or_J,
                                    double *out)
{
    int i;
    for (i = 0; i < ngrid; i++)
    {
        out[gridID[i]] = Density_or_J[i];
    }
}

void _Original_Grid_to_Reorder_Grid(
    int ngrid, int *gridID, double *Density_or_J, double *out)
{
    int i;
    for (i = 0; i < ngrid; i++)
    {
        out[i] = Density_or_J[gridID[i]];
    }
}

void _construct_V_local_bas(
    int *mesh,
    int nrow,
    int ncol,
    int *gridID,
    double *auxBasis,
    double *CoulG,
    int row_shift,
    double *V,
    int *grid_ordering,
    double *buf,         // use the ptr of the ptr to ensure that the memory for each thread is aligned
    const int buffersize // must be a multiple of 16 to ensure memory alignment
)
{
    // printf("nrow: %d, ncol: %d\n", nrow, ncol);
    // printf("row_shift: %d\n", row_shift);

    const int nThread = get_omp_threads();
    size_t mesh_complex[3] = {mesh[0], mesh[1], mesh[2] / 2 + 1};
    const size_t n_real = mesh[0] * mesh[1] * mesh[2];
    const size_t n_complex = mesh_complex[0] * mesh_complex[1] * mesh_complex[2];
    const double fac = 1. / (double)n_real;

    // create plan for fft

    fftw_plan p_forward = fftw_plan_dft_r2c(3, mesh, auxBasis, (fftw_complex *)buf, FFTW_ESTIMATE);
    fftw_plan p_backward = fftw_plan_dft_c2r(3, mesh, (fftw_complex *)buf, V, FFTW_ESTIMATE);

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();
        double *buf_thread = buf + thread_id * buffersize;
        fftw_complex *buf_fft = (fftw_complex *)(buf_thread + n_real);

#pragma omp for schedule(static)
        for (size_t i = 0; i < nrow; i++)
        {
            // pack

            memset(buf_thread, 0, sizeof(double) * n_real);

            for (size_t j = 0; j < ncol; j++)
            {
                buf_thread[gridID[j]] = auxBasis[i * ncol + j];
            }

            // forward transform

            fftw_execute_dft_r2c(p_forward, buf_thread, (fftw_complex *)buf_fft);

            // multiply CoulG

            double *ptr = (double *)buf_fft;

            for (size_t j = 0; j < n_complex; j++)
            {
                *ptr++ *= CoulG[j]; /// TODO: use ISPC to accelerate
                *ptr++ *= CoulG[j]; /// TODO: use ISPC to accelerate
            }

            // backward transform

            memset(buf_thread, 0, sizeof(double) * n_real);

            fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf_fft, buf_thread);

            // scale

            ptr = V + (i + row_shift) * n_real;

            for (size_t j = 0; j < n_real; j++)
            {
                // *ptr++ *= fac; /// TODO: use ISPC to accelerate
                // ptr[grid_ordering[j]] = buf_thread[j] * fac;
                ptr[j] = buf_thread[grid_ordering[j]] * fac;
            }
        }
    }

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);
}

void _construct_V_kernel(int *mesh_bra,
                         int *mesh_ket,
                         int *map_bra_2_ket,
                         int naux,
                         double *auxBasis,
                         double *CoulG, // bra
                         double *V,
                         const int BunchSize,
                         double *buf,         // use the ptr of the ptr to ensure that the memory for each thread is aligned
                         const int buffersize // must be a multiple of 16 to ensure memory alignment
)
{
    // printf("naux      = %d\n", naux);
    // printf("BunchSize = %d\n", BunchSize);

    // print all the input info

    static const int INC1 = 1;
    static const int SMALL_SIZE = 8;

    const int nThread = get_omp_threads();
    const int nBunch = ((naux / BunchSize) / nThread) * nThread; // dispatch evenly
    const int nLeft = naux - nBunch * BunchSize;

    // printf("nBunch = %d\n", nBunch);
    // printf("nLeft  = %d\n", nLeft);

    // print the dispatch info

    int mesh_bra_complex[3] = {mesh_bra[0], mesh_bra[1], mesh_bra[2] / 2 + 1};
    int mesh_ket_complex[3] = {mesh_ket[0], mesh_ket[1], mesh_ket[2] / 2 + 1};

    const int n_real_bra = mesh_bra[0] * mesh_bra[1] * mesh_bra[2];
    const int n_complex_bra = mesh_bra_complex[0] * mesh_bra_complex[1] * mesh_bra_complex[2];
    const int n_real_ket = mesh_ket[0] * mesh_ket[1] * mesh_ket[2];
    const int n_complex_ket = mesh_ket_complex[0] * mesh_ket_complex[1] * mesh_ket_complex[2];

    if (n_real_bra > n_real_ket)
    {
        printf("n_real_bra > n_real_ket\n");
        exit(1);
    }

    const double fac = 1. / sqrtl((double)n_real_bra * (double)n_real_ket);

    // create plan for fft

    fftw_plan p_forward = fftw_plan_many_dft_r2c(
        3, mesh_bra, BunchSize, auxBasis, mesh_bra, 1, n_real_bra, (fftw_complex *)buf, mesh_bra_complex, 1, n_complex_bra, FFTW_ESTIMATE);

    fftw_plan p_backward = fftw_plan_many_dft_c2r(
        3, mesh_ket, BunchSize, (fftw_complex *)buf, mesh_ket_complex, 1, n_complex_ket, V, mesh_ket, 1, n_real_ket, FFTW_ESTIMATE);

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

            fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real_bra, (fftw_complex *)buf_thread);

            // multiply CoulG

            ptr = buf_thread;

            for (j = bunch_start; j < bunch_end; ++j)
            {
                for (k = 0; k < n_complex_bra; ++k)
                {
                    *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                    *ptr++ *= CoulG[k]; /// TODO: use ISPC to accelerate
                }
            }

            if (map_bra_2_ket != NULL)
            {
                ptr = buf_thread + n_complex_bra * 2 * BunchSize;
                memset(ptr, 0, sizeof(double) * n_complex_ket * 2 * BunchSize);
                for (j = bunch_start; j < bunch_end; ++j)
                {
                    size_t shift = (j - bunch_start) * n_complex_bra * 2;
                    for (k = 0; k < n_complex_bra; ++k)
                    {
                        ptr[2 * map_bra_2_ket[k]] = buf_thread[shift + 2 * k];
                        ptr[2 * map_bra_2_ket[k] + 1] = buf_thread[shift + 2 * k + 1];
                    }
                    ptr += n_complex_ket * 2;
                }
                ptr = buf_thread + n_complex_bra * 2 * BunchSize;
            }
            else
            {
                ptr = buf_thread;
            }

            // backward transform

            // fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf_thread, V + (size_t)bunch_start * (size_t)n_real);
            fftw_execute_dft_c2r(p_backward, (fftw_complex *)ptr, V + (size_t)bunch_start * (size_t)n_real_ket);

            // scale

            ptr = V + (size_t)bunch_start * (size_t)n_real_ket;
            int _size_ = n_real_ket * BunchSize;
            dscal_(&_size_, &fac, ptr, &INC1);

            // for (j = bunch_start; j < bunch_end; ++j)
            // {
            //     for (k = 0; k < n_real_ket; ++k)
            //     {
            //         *ptr++ *= fac; /// TODO: use ISPC to accelerate
            //     }
            // }
        }
    }

    // destroy plan

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);

    // printf("finish bulk nLeft = %d\n", nLeft);
    // fflush(stdout);

    if (nLeft > 0)
    {
        if ((nLeft <= SMALL_SIZE) && (nLeft <= BunchSize))
        {
            // printf("nLeft <= SMALL_SIZE or nLeft <= BunchSize\n");
            // fflush(stdout);

            // use single thread to handle the left

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // create plan

            fftw_plan p_forward = fftw_plan_many_dft_r2c(
                // 3, mesh, nLeft, auxBasis + bunch_start * n_real, mesh, 1, n_real, (fftw_complex *)buf, mesh_complex, 1, n_complex, FFTW_ESTIMATE);
                3, mesh_bra, nLeft, auxBasis + bunch_start * n_real_bra, mesh_bra, 1, n_real_bra, (fftw_complex *)buf, mesh_bra_complex, 1, n_complex_bra, FFTW_ESTIMATE);

            fftw_plan p_backward = fftw_plan_many_dft_c2r(
                // 3, mesh, nLeft, (fftw_complex *)buf, mesh_complex, 1, n_complex, V + bunch_start * n_real, mesh, 1, n_real, FFTW_ESTIMATE);
                3, mesh_ket, nLeft, (fftw_complex *)buf, mesh_ket_complex, 1, n_complex_ket, V + bunch_start * n_real_ket, mesh_ket, 1, n_real_ket, FFTW_ESTIMATE);

            // forward transform

            // fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real, (fftw_complex *)buf);
            fftw_execute_dft_r2c(p_forward, auxBasis + (size_t)bunch_start * (size_t)n_real_bra, (fftw_complex *)buf);

            // multiply CoulG

            double *ptr = buf;

            for (int j = bunch_start; j < bunch_end; ++j)
            {
                // for (int k = 0; k < n_complex; ++k)
                for (int k = 0; k < n_complex_bra; ++k)
                {
                    *ptr++ *= CoulG[k]; ///
                    *ptr++ *= CoulG[k]; ///
                }
            }

            if (map_bra_2_ket != NULL)
            {
                ptr = buf + n_complex_bra * 2 * nLeft;
                memset(ptr, 0, sizeof(double) * n_complex_ket * 2 * nLeft);
                for (int j = bunch_start; j < bunch_end; ++j)
                {
                    size_t shift = (j - bunch_start) * n_complex_bra * 2;
                    for (int k = 0; k < n_complex_bra; ++k)
                    {
                        ptr[2 * map_bra_2_ket[k]] = buf[shift + 2 * k];
                        ptr[2 * map_bra_2_ket[k] + 1] = buf[shift + 2 * k + 1];
                    }
                    ptr += n_complex_ket * 2;
                }
                ptr = buf + n_complex_bra * 2 * nLeft;
            }
            else
            {
                ptr = buf;
            }

            // backward transform

            // fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf, V + (size_t)bunch_start * (size_t)n_real);
            fftw_execute_dft_c2r(p_backward, (fftw_complex *)ptr, V + (size_t)bunch_start * (size_t)n_real_ket);

            // scale

            // ptr = V + (size_t)bunch_start * (size_t)n_real;
            ptr = V + (size_t)bunch_start * (size_t)n_real_ket;
            int _size_ = n_real_ket * nLeft;
            dscal_(&_size_, &fac, ptr, &INC1);

            // for (int j = bunch_start; j < bunch_end; ++j)
            // {
            //     for (int k = 0; k < n_real; ++k)
            //     {
            //         *ptr++ *= fac; ///
            //     }
            // }

            // destroy plan

            fftw_destroy_plan(p_forward);
            fftw_destroy_plan(p_backward);
        }
        else
        {
            // printf("nLeft > SMALL_SIZE or nLeft > BunchSize\n");

            // use parallel thread to handle the left, assume the nTransform is 1

            int bunch_start = nBunch * BunchSize;
            int bunch_end = bunch_start + nLeft;

            // create plan

            // fftw_plan p_forward = fftw_plan_dft_r2c(3, mesh, auxBasis + bunch_start * n_real, (fftw_complex *)buf, FFTW_ESTIMATE);
            fftw_plan p_forward = fftw_plan_dft_r2c(3,
                                                    // mesh, auxBasis + bunch_start * n_real, (fftw_complex *)buf, FFTW_ESTIMATE);
                                                    mesh_bra, auxBasis + bunch_start * n_real_bra, (fftw_complex *)buf, FFTW_ESTIMATE);

            fftw_plan p_backward = fftw_plan_dft_c2r(3,
                                                     // mesh, (fftw_complex *)buf, V + bunch_start * n_real, FFTW_ESTIMATE);
                                                     mesh_ket, (fftw_complex *)buf, V + bunch_start * n_real_ket, FFTW_ESTIMATE);

            // size_t nbuf_per_thread = ((n_complex * 2 + 15) / 16) * 16; // make sure the memory is aligned
            size_t nbuf_per_thread = ((n_complex_bra * 2 + 15) / 16) * 16; // make sure the memory is aligned
            if (map_bra_2_ket != NULL)
            {
                nbuf_per_thread += ((n_complex_ket * 2 + 15) / 16) * 16;
            }

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

                    // fftw_execute_dft_r2c(p_forward, auxBasis + j * (size_t)n_real, (fftw_complex *)buf_thread);
                    fftw_execute_dft_r2c(p_forward, auxBasis + j * (size_t)n_real_bra, (fftw_complex *)buf_thread);

                    // multiply CoulG

                    ptr = buf_thread;

                    // for (k = 0; k < n_complex; ++k)
                    for (k = 0; k < n_complex_bra; ++k)
                    {
                        *ptr++ *= CoulG[k];
                        *ptr++ *= CoulG[k];
                    }

                    if (map_bra_2_ket != NULL)
                    {
                        ptr = buf_thread + n_complex_bra * 2;
                        memset(ptr, 0, sizeof(double) * n_complex_ket * 2);
                        for (k = 0; k < n_complex_bra; ++k)
                        {
                            ptr[2 * map_bra_2_ket[k]] = buf_thread[2 * k];
                            ptr[2 * map_bra_2_ket[k] + 1] = buf_thread[2 * k + 1];
                        }
                        ptr = buf_thread + n_complex_bra * 2;
                    }
                    else
                    {
                        ptr = buf_thread;
                    }

                    // backward transform

                    // fftw_execute_dft_c2r(p_backward, (fftw_complex *)buf_thread, V + j * (size_t)n_real);
                    fftw_execute_dft_c2r(p_backward, (fftw_complex *)ptr, V + j * (size_t)n_real_ket);

                    // scale

                    // ptr = V + j * (size_t)n_real;
                    ptr = V + j * (size_t)n_real_ket;
                    int _size_ = n_real_ket;
                    dscal_(&_size_, &fac, ptr, &INC1);

                    // for (k = 0; k < n_real; ++k)
                    // {
                    //     *ptr++ *= fac;
                    // }
                }
            }

            // destroy plan

            fftw_destroy_plan(p_forward);
            fftw_destroy_plan(p_backward);
        }
    }
}

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
    _construct_V_kernel(mesh, mesh, NULL, naux, auxBasis, CoulG, V, BunchSize, buf, buffersize);
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

    size_t i;

    const double *ptr_G = CoulG + p0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (i = 0; i < naux; i++)
    {
        size_t j;
        double *ptr_basis = auxBasisFFT + i * ngrid * 2;
        for (j = 0; j < ngrid; j++)
        {
            ptr_basis[j * 2] *= ptr_G[j];
            ptr_basis[j * 2 + 1] *= ptr_G[j];
        }
    }
}

///////////// get_jk linear scaling /////////////

void _extract_dm_involved_ao(
    double *dm,
    const int nao,
    double *res_buf,
    const int *ao_involved,
    const int nao_involved)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nao_involved; ++i)
    {
        for (size_t j = 0; j < nao_involved; ++j)
        {
            res_buf[i * nao_involved + j] = dm[ao_involved[i] * nao + ao_involved[j]];
        }
    }
}

void _extract_dm_involved_ao_RS(
    double *dm,
    const int nao,
    double *res_buf,
    const int *bra_ao_involved,
    const int bra_nao_involved,
    const int *ket_ao_involved,
    const int ket_nao_involved)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < bra_nao_involved; ++i)
    {
        for (size_t j = 0; j < ket_nao_involved; ++j)
        {
            res_buf[i * ket_nao_involved + j] = dm[bra_ao_involved[i] * nao + ket_ao_involved[j]];
        }
    }
}

void _packadd_local_dm(
    double *local_dm,
    const int nao_involved,
    const int *ao_involved,
    double *dm,
    const int nao)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nao_involved; ++i)
    {
        for (size_t j = 0; j < nao_involved; ++j)
        {
            dm[ao_involved[i] * nao + ao_involved[j]] += local_dm[i * nao_involved + j];
        }
    }
}

void _packadd_local_dm2_add_transpose(
    double *local_dm,
    const int bra_nao_involved,
    const int *bra_ao_involved,
    const int ket_nao_involved,
    const int *ket_ao_involved,
    double *dm,
    const int nao)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < bra_nao_involved; ++i)
    {
        for (size_t j = 0; j < ket_nao_involved; ++j)
        {
            dm[bra_ao_involved[i] * nao + ket_ao_involved[j]] += local_dm[i * ket_nao_involved + j];
            dm[ket_ao_involved[j] * nao + bra_ao_involved[i]] += local_dm[i * ket_nao_involved + j];
        }
    }
}

void _packadd_local_dm2(
    double *local_dm,
    const int bra_nao_involved,
    const int *bra_ao_involved,
    const int ket_nao_involved,
    const int *ket_ao_involved,
    double *dm,
    const int nao)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < bra_nao_involved; ++i)
    {
        for (size_t j = 0; j < ket_nao_involved; ++j)
        {
            dm[bra_ao_involved[i] * nao + ket_ao_involved[j]] += local_dm[i * ket_nao_involved + j];
        }
    }
}

void _packadd_local_RS(
    double *local_dm,
    const int bra_nao_involved,
    const int *bra_ao_involved,
    const int ket_nao_involved,
    const int *ket_ao_involved,
    double *dm,
    const int nao)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < bra_nao_involved; ++i)
    {
        for (size_t j = 0; j < ket_nao_involved; ++j)
        {
            dm[bra_ao_involved[i] * nao + ket_ao_involved[j]] += local_dm[i * ket_nao_involved + j];
            dm[ket_ao_involved[j] * nao + bra_ao_involved[i]] += local_dm[i * ket_nao_involved + j];
        }
    }
}

void _buildJ_k_packaddrow(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *rowloc,
    const int *colloc)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_source; ++i)
    {
        size_t row_loc = rowloc[i];
        for (size_t j = 0; j < ncol_source; ++j)
        {
            target[row_loc * ncol_target + colloc[j]] += source[i * ncol_source + j];
        }
    }
}

void _buildK_packaddrow(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *ao_involved)
{
    int nThread = get_omp_threads();

    static const int INC = 1;
    static const double ONE = 1.0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_source; ++i)
    {
        size_t row_loc = ao_involved[i];
        // memcpy(target + row_loc * ncol_target, source + i * ncol_source, sizeof(double) * ncol_source);
        daxpy_(&ncol_source, &ONE, source + i * ncol_source, &INC, target + row_loc * ncol_target, &INC);
    }
}

void _buildK_packaddrow_shift_col(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *ao_involved,
    const int kmesh,
    const int nao_prim,
    const int *box_permutation)
{
    int nThread = get_omp_threads();

    static const int INC = 1;
    static const double ONE = 1.0;

    if (ncol_target != (kmesh * nao_prim))
    {
        printf("Error: ncol_target!=(kmesh *nao_prim)\n");
        exit(1);
    }

    if (ncol_source != ncol_target)
    {
        printf("Error: ncol_source!=ncol_target\n");
        exit(1);
    }

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_source; ++i)
    {
        size_t row_loc = ao_involved[i];
        // memcpy(target + row_loc * ncol_target, source + i * ncol_source, sizeof(double) * ncol_source);
        // daxpy_(&ncol_source, &ONE, source + i * ncol_source, &INC, target + row_loc * ncol_target, &INC);
        for (size_t j = 0; j < kmesh; ++j)
        {
            daxpy_(&nao_prim, &ONE, source + i * ncol_source + j * nao_prim, &INC, target + row_loc * ncol_target + box_permutation[j] * nao_prim, &INC);
        }
    }
}

void _buildK_packaddcol(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *ao_involved)

{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_source; ++i)
    {
        for (size_t j = 0; j < ncol_source; ++j)
        {
            target[i * ncol_target + ao_involved[j]] += source[i * ncol_source + j];
        }
    }
}

void _buildK_packrow(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *ao_involved)
{
    int nThread = get_omp_threads();

    // static const int INC = 1;
    // static const double ONE = 1.0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_target; ++i)
    {
        size_t row_loc = ao_involved[i];
        memcpy(target + i * ncol_target, source + row_loc * ncol_source, sizeof(double) * ncol_source);
    }
}

void _buildK_packcol(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *ao_involved)
{
    int nThread = get_omp_threads();

    // static const int INC = 1;
    // static const double ONE = 1.0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_target; ++i)
    {
        for (size_t j = 0; j < ncol_target; ++j)
        {
            target[i * ncol_target + j] = source[i * ncol_source + ao_involved[j]];
        }
    }
}

void _buildK_unpackcol(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int *source_ind)
{
    int nThread = get_omp_threads();

    // static const int INC = 1;
    // static const double ONE = 1.0;

    memset(target, 0, sizeof(double) * nrow_target * ncol_target);

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_source; ++i)
    {
        for (size_t j = 0; j < ncol_source; ++j)
        {
            target[i * ncol_target + source_ind[j]] = source[i * ncol_source + j];
        }
    }
}

void _buildK_packcol2(
    double *target,
    const int nrow_target,
    const int ncol_target,
    double *source,
    const int nrow_source,
    const int ncol_source,
    const int col_indx_begin,
    const int col_indx_end)
{
    int nThread = get_omp_threads();

    // static const int INC = 1;
    // static const double ONE = 1.0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_target; ++i)
    {
        memcpy(target + i * ncol_target, source + i * ncol_source + col_indx_begin, sizeof(double) * (col_indx_end - col_indx_begin));
    }
}

void _buildK_packcol3(
    double *target,
    const int nrow_target,
    const int ncol_target,
    const int col_indx_begin,
    const int col_indx_end,
    double *source,
    const int nrow_source,
    const int ncol_source)
{
    int nThread = get_omp_threads();

    // static const int INC = 1;
    // static const double ONE = 1.0;

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < nrow_target; ++i)
    {
        memcpy(target + i * ncol_target + col_indx_begin, source + i * ncol_source, sizeof(double) * (col_indx_end - col_indx_begin));
    }
}

void _buildK_copy(double *target, double *source, const size_t size)
{
    memcpy(target, source, sizeof(double) * size);
}

////////// used in moR to density //////////

void moR_to_Density(
    const int ngrids,
    const int nMO,
    const double *moR,
    double *rhoR)
{
    int nThread = get_omp_threads();

    int ngrid_per_thread = (ngrids + nThread - 1) / nThread;

    memset(rhoR, 0, sizeof(double) * ngrids);

#pragma omp parallel num_threads(nThread)
    {
        int thread_id = omp_get_thread_num();
        int grid_start = thread_id * ngrid_per_thread;
        grid_start = grid_start < ngrids ? grid_start : ngrids;
        int grid_end = (thread_id + 1) * ngrid_per_thread;
        grid_end = grid_end < ngrids ? grid_end : ngrids;

        for (int i = 0; i < nMO; i++)
        {
            for (int j = grid_start; j < grid_end; j++)
            {
                rhoR[j] += moR[i * ngrids + j] * moR[i * ngrids + j];
            }
        }
    }
}

////////// transpose 012 -> 021 //////////

void transpose_012_to_021(
    double *target,
    double *source,
    const int n1,
    const int n2,
    const int n3)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < n1; i++)
    {
        size_t shift = i * n2 * n3;
        double *ptr_target = target + shift;
        double *ptr_source = source + shift;
        for (size_t j = 0; j < n2; j++)
        {
            for (size_t k = 0; k < n3; k++)
            {
                ptr_target[k * n2 + j] = ptr_source[j * n3 + k];
            }
        }
    }
}

void transpose_012_to_021_InPlace(
    double *target,
    const int n1,
    const int n2,
    const int n3,
    double *buf)
{
    int nThread = get_omp_threads();

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < n1; i++)
    {
        size_t shift = i * n2 * n3;
        double *ptr_buf = buf + shift;
        double *ptr_source = target + shift;
        for (size_t j = 0; j < n2; j++)
        {
            for (size_t k = 0; k < n3; k++)
            {
                ptr_buf[k * n2 + j] = ptr_source[j * n3 + k];
            }
        }
    }

    memcpy(target, buf, sizeof(double) * n1 * n2 * n3);
}

void contract_ipk_pk_to_ik(
    double *A,
    double *B,
    double *C,
    const int n1,
    const int n2,
    const int n3)
{
    int nThread = get_omp_threads();

    memset(C, 0, sizeof(double) * n1 * n3);

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < n1; i++)
    {
        double *ptr_A = A + i * n2 * n3;
        double *ptr_B = B;
        for (size_t j = 0; j < n2; j++)
        {
            double *ptr_res = C + i * n3;
            for (size_t k = 0; k < n3; k++)
            {
                *ptr_res++ += *ptr_A++ * *ptr_B++;
            }
        }
    }
}

////////// used in CCCC for LR part in RS-ISDF //////////

void _unpack_aoPairR(
    double *target,
    const int n1,
    const int n2,
    const int n3,
    double *source,
    const int m1,
    const int m2,
    const int m3,
    const int m2_begin,
    const int m2_end,
    const int *grid_involved)
{
    int nThread = get_omp_threads();

    int ntask = n1 * (m2_end - m2_begin);

    memset(target, 0, sizeof(double) * n1 * n2 * n3);

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < ntask; i++)
    {
        size_t i1 = i / (m2_end - m2_begin);
        size_t i2 = i % (m2_end - m2_begin);

        size_t shift_target = i1 * n2 * n3 + i2 * n3;
        size_t shift_source = i1 * m2 * m3 + (i2 + m2_begin) * m3;

        for (size_t j = 0; j < m3; ++j)
        {
            target[shift_target + grid_involved[j]] = source[shift_source + j];
        }
    }
}

void _pack_aoPairR_index1(
    double *target,
    const int n1,
    const int n2,
    const int n3,
    double *source,
    const int m1,
    const int m2,
    const int m3,
    const int m2_begin,
    const int m2_end)
{
    int nThread = get_omp_threads();

    // memset(target, 0, sizeof(double) * n1 * n2 * n3);

#pragma omp parallel for num_threads(nThread) schedule(static)
    for (size_t i = 0; i < n1; i++)
    {
        size_t shift_target = i * n2 * n3;
        size_t shift_source = i * m2 * m3 + m2_begin * m3;

        memcpy(target + shift_target, source + shift_source, sizeof(double) * n2 * m3);
    }
}

////////// in determing partition //////////

double _distance_translation(double *pa, double *pb, double *a)
{
    double dx, dx1, dx2;
    double dy, dy1, dy2;
    double dz, dz1, dz2;

    dx = pa[0] - pb[0];
    dx1 = dx - a[0];
    dx2 = dx + a[0];
    dx = fabs(dx);
    dx1 = fabs(dx1);
    dx2 = fabs(dx2);
    dx = fmin(fmin(dx, dx1), dx2);

    dy = pa[1] - pb[1];
    dy1 = dy - a[1];
    dy2 = dy + a[1];
    dy = fabs(dy);
    dy1 = fabs(dy1);
    dy2 = fabs(dy2);
    dy = fmin(fmin(dy, dy1), dy2);

    dz = pa[2] - pb[2];
    dz1 = dz - a[2];
    dz2 = dz + a[2];
    dz = fabs(dz);
    dz1 = fabs(dz1);
    dz2 = fabs(dz2);
    dz = fmin(fmin(dz, dz1), dz2);

    return sqrt(dx * dx + dy * dy + dz * dz);
}

void distance_between_point_atms(
    double *distance,
    double *pnt,
    double *atm_coords,
    const int natm,
    const double *lattice_vector)
{
    double a[3];
    a[0] = lattice_vector[0 * 3 + 0];
    a[1] = lattice_vector[1 * 3 + 1];
    a[2] = lattice_vector[2 * 3 + 2];

#pragma omp parallel for schedule(static) num_threads(get_omp_threads())
    for (int i = 0; i < natm; i++)
    {
        distance[i] = _distance_translation(pnt, atm_coords + i * 3, a);
    }
}

void distance_between_points_atms(
    double *distance,
    double *pnt,
    const int npnt,
    double *atm_coords,
    const int natm,
    const double *lattice_vector)
{
    double a[3];
    a[0] = lattice_vector[0 * 3 + 0];
    a[1] = lattice_vector[1 * 3 + 1];
    a[2] = lattice_vector[2 * 3 + 2];

#pragma omp parallel for schedule(static) num_threads(get_omp_threads())
    for (size_t i = 0; i < npnt; i++)
    {
        for (size_t j = 0; j < natm; j++)
        {
            distance[i * natm + j] = _distance_translation(pnt + i * 3, atm_coords + j * 3, a);
        }
    }
}