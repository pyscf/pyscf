#include <stdlib.h>
#include <complex.h>

void gradient_gs(double complex* out, double complex* f_gs, double* Gv,
                 int n, size_t ng)
{
    int i;
    double complex *outx, *outy, *outz;
    for (i = 0; i < n; i++) {
        outx = out;
        outy = outx + ng;
        outz = outy + ng;
        #pragma omp parallel
        {
            size_t igrid;
            double *pGv;
            #pragma omp for schedule(static)
            for (igrid = 0; igrid < ng; igrid++) {
                pGv = Gv + igrid * 3;
                outx[igrid] = pGv[0] * creal(f_gs[igrid]) * _Complex_I - pGv[0] * cimag(f_gs[igrid]);
                outy[igrid] = pGv[1] * creal(f_gs[igrid]) * _Complex_I - pGv[1] * cimag(f_gs[igrid]);
                outz[igrid] = pGv[2] * creal(f_gs[igrid]) * _Complex_I - pGv[2] * cimag(f_gs[igrid]);
            }
        }
        f_gs += ng;
        out += 3 * ng;
    }
}
