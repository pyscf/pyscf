#include <stdlib.h>
#include <math.h>
#include "config.h"

#define TWO_PI 6.283185307179586

void get_eps(double* eps, double* deps_intermediate, double* rho,
             double rho_min, double rho_max, double eps0, size_t ng)
{
    #pragma omp parallel
    {
        size_t i;
        double log_eps0 = log(eps0);
        double fac = log_eps0 / TWO_PI;
        double log_rho_max_over_rho_min = log(rho_max / rho_min);
        double tmp;
        #pragma omp for schedule(static)
        for (i = 0; i < ng; i++) {
            double rho_i = rho[i];
            if (rho_i > rho_max) {
                eps[i] = 1.;
                deps_intermediate[i] = 0;
            } else if (rho_i < rho_min) {
                eps[i] = eps0;
                deps_intermediate[i] = 0;
            } else {
                tmp = TWO_PI * log(rho_max / rho_i) / log_rho_max_over_rho_min;
                eps[i] = exp(fac * (tmp - sin(tmp)));
                deps_intermediate[i] = log_eps0 / log_rho_max_over_rho_min * (-1.+cos(tmp)) / rho_i;
            }
        }
    }
}

void rs_gradient_cd3(double* f, double* df, int* mesh, double* dr)
{
    const size_t nx = mesh[0];
    const size_t ny = mesh[1];
    const size_t nz = mesh[2];
    const size_t nyz = ny * nz;
    const size_t nxyz = nx * nyz;
    double h[3] = {dr[0]*2, dr[1]*2, dr[2]*2};
    double *dfdx = df;
    double *dfdy = dfdx + nxyz;
    double *dfdz = dfdy + nxyz;
    #pragma omp parallel
    {
        size_t x, y, z;
        size_t xm, xp, ym, yp, zm, zp;
        size_t xoff, yoff, ioff;
        size_t xmoff, xpoff, ymoff, ypoff;
        #pragma omp for schedule(static)
        for (x = 0; x < nx; x++) {
            xm = (x == 0) ? (nx-1) : (x - 1);
            xp = (x == nx-1) ? 0 : (x + 1);
            xoff = x * nyz;
            xmoff = xm * nyz;
            xpoff = xp * nyz;
            for (y = 0; y < ny; y++) {
                ym = (y == 0) ? (ny-1) : (y - 1);
                yp = (y == ny-1) ? 0 : (y + 1);
                yoff = y * nz;
                ymoff = ym * nz;
                ypoff = yp * nz;
                for (z = 0; z < nz; z++) {
                    zm = (z == 0) ? (nz-1) : (z - 1);
                    zp = (z == nz-1) ? 0 : (z + 1);
                    ioff = xoff + yoff + z;
                    dfdx[ioff] = (f[xpoff+yoff+z] - f[xmoff+yoff+z]) / h[0];
                    dfdy[ioff] = (f[xoff+ypoff+z] - f[xoff+ymoff+z]) / h[1];
                    dfdz[ioff] = (f[xoff+yoff+zp] - f[xoff+yoff+zm]) / h[2];
                }
            }
        }
    }
}

void rs_laplacian_cd3(double* f, double* lf, int* mesh, double* dr)
{
    const size_t nx = mesh[0];
    const size_t ny = mesh[1];
    const size_t nz = mesh[2];
    const size_t nyz = ny * nz;
    double h2[3] = {dr[0]*dr[0], dr[1]*dr[0], dr[2]*dr[0]};
    #pragma omp parallel
    {
        size_t x, y, z;
        size_t xm, xp, ym, yp, zm, zp;
        size_t xoff, yoff, ioff;
        size_t xmoff, xpoff, ymoff, ypoff;
        double fc;
        #pragma omp for schedule(static)
        for (x = 0; x < nx; x++) {
            xm = (x == 0) ? (nx-1) : (x - 1);
            xp = (x == nx-1) ? 0 : (x + 1);
            xoff = x * nyz;
            xmoff = xm * nyz;
            xpoff = xp * nyz;
            for (y = 0; y < ny; y++) {
                ym = (y == 0) ? (ny-1) : (y - 1);
                yp = (y == ny-1) ? 0 : (y + 1);
                yoff = y * nz;
                ymoff = ym * nz;
                ypoff = yp * nz;
                for (z = 0; z < nz; z++) {
                    zm = (z == 0) ? (nz-1) : (z - 1);
                    zp = (z == nz-1) ? 0 : (z + 1);
                    ioff = xoff + yoff + z;
                    fc = 2. * f[ioff];
                    lf[ioff]  = (f[xpoff+yoff+z] + f[xmoff+yoff+z] - fc) / h2[0];
                    lf[ioff] += (f[xoff+ypoff+z] + f[xoff+ymoff+z] - fc) / h2[1];
                    lf[ioff] += (f[xoff+yoff+zp] + f[xoff+yoff+zm] - fc) / h2[2];
                }
            }
        }
    }
}
