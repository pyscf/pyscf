#include <complex.h>
#include "config.h"
#include "vhf/fblas.h"
#if defined(HAVE_LIBXSMM)
#include "libxsmm.h"
#endif


void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc)
{
#if defined(HAVE_LIBXSMM)
    if (transa == 'N') {
        //libxsmm_dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        int prefetch = LIBXSMM_PREFETCH_AUTO;
        int flags = transb != 'T' ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B;
        libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc,
                                                         &alpha, &beta, &flags, &prefetch);
        if (kernel) {
            kernel(a,b,c,a,b,c);
            return;
        }
    }
#endif
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void get_gga_vrho_gs(double complex *out, double complex *vrho_gs, double complex *vsigma1_gs,
                     double *Gv, double weight, int ngrid)
{
    int i;
    int ngrid2 = 2 * ngrid;
    double complex fac = -2. * _Complex_I;
    #pragma omp parallel for simd schedule(static)
    for (i = 0; i < ngrid; i++) {
        out[i] = ( Gv[i*3]   * vsigma1_gs[i]
                  +Gv[i*3+1] * vsigma1_gs[i+ngrid]
                  +Gv[i*3+2] * vsigma1_gs[i+ngrid2]) * fac + vrho_gs[i];
        out[i] *= weight;
    }
}
