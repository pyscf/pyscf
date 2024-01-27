#include <fftw3.h>

#define FFT_PLAN fftw_plan

FFT_PLAN fft_create_r2c_plan(double *in, fftw_complex *out, int rank, int *mesh);
FFT_PLAN fft_create_c2r_plan(fftw_complex *in, double *out, int rank, int *mesh);
void fft_execute(FFT_PLAN p);
void fft_destroy_plan(FFT_PLAN p);
