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

    if (ColEnd != (ColBegin + ncol_source))
    {
        printf("ColEnd!=ColBegin+ncol_source\n");
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