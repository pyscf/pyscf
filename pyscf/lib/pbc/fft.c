#include <complex.h>
#include <fft.h>

fftw_plan fft_create_r2c_plan(double* in, complex double* out, int rank, int* mesh)
{
    fftw_plan p;
    p = fftw_plan_dft_r2c(rank, mesh, in, out, FFTW_ESTIMATE);
    return p;
}

fftw_plan fft_create_c2r_plan(complex double* in, double* out, int rank, int* mesh)
{
    fftw_plan p;
    p = fftw_plan_dft_c2r(rank, mesh, in, out, FFTW_ESTIMATE);
    return p;
}

void fft_execute(fftw_plan p)
{
    fftw_execute(p);
}

void fft_destroy_plan(fftw_plan p)
{
    fftw_destroy_plan(p);
}
