#include <stdlib.h>
#include <math.h>
#include "config.h"

#define TWO_PI 6.283185307179586

inline double _eps_interstitial(double rho, double rho_max,
                                double log_rho_max_over_rho_min,
                                double two_pi_divide_log_eps0)
{   
    double tmp = TWO_PI * log(rho_max / rho) / log_rho_max_over_rho_min;
    double eps = exp(two_pi_divide_log_eps0 * (tmp - sin(tmp)));
    return eps;
}

void get_eps(double* out, double* rho, double rho_min, double rho_max,
             double eps0, size_t ng)
{
    #pragma omp parallel
    {
        size_t i;
        double two_pi_divide_log_eps0 = log(eps0) / TWO_PI;
        double log_rho_max_over_rho_min = log(rho_max / rho_min);
        #pragma omp for schedule(static)
        for (i = 0; i < ng; i++) {
            double rho_i = rho[i];
            if (rho_i > rho_max) {
                out[i] = 1.;
            } else if (rho_i < rho_min) {
                out[i] = eps0;
            } else {
                out[i] = _eps_interstitial(rho_i, rho_max,
                                           log_rho_max_over_rho_min,
                                           two_pi_divide_log_eps0);
            }
        }
    }
}

inline double _eps1_intermediate_interstitial(double rho, double rho_max,
                                              double log_rho_max_over_rho_min,
                                              double log_eps0)
{
    double tmp = TWO_PI * log(rho_max / rho) / log_rho_max_over_rho_min;
    double out = log_eps0 / log_rho_max_over_rho_min * (-1. + cos(tmp)) / rho;
    return out;
}

void get_eps1_intermediate(double* out, double* rho,
                           double rho_min, double rho_max,
                           double eps0, size_t ng)
{
    #pragma omp parallel
    {
        size_t i;
        double log_eps0 = log(eps0);
        double log_rho_max_over_rho_min = log(rho_max / rho_min);
        #pragma omp for schedule(static)
        for (i = 0; i < ng; i++) {
            double rho_i = rho[i]; 
            if (rho_i > rho_max || rho_i < rho_min) {
                out[i] = 0;
            } else {
                out[i] = _eps1_intermediate_interstitial(rho_i, rho_max,
                                                         log_rho_max_over_rho_min,
                                                         log_eps0);
            } 
        }
    }
}

void fdiff_gradient(double* out, double* field,
                    double* rho, double rho_min, double rho_max,
                    size_t nx, size_t ny, size_t nz,
                    double hx, double hy, double hz)
{
    //#pragma omp parallel
    {
        size_t i = 0;
        size_t x, y, z, offset;
        size_t nxyz = nx * ny * nz;
        size_t nyz = ny * nz;
        for (x = 0; x < nx; x++) {
            for (y = 0; y < ny; y++) {
                for (z = 0; z < nz; z++, i++) {
                    if (rho[i] > rho_max || rho[i] < rho_min) {
                        out[i] = 0;
                        out[i + nxyz] = 0;
                        out[i + 2*nxyz] = 0;
                    }
                    offset = y*nz + z;
                    if (x == 0) {
                        out[x*nyz+offset] = (field[(x+1)*nyz+offset] - field[(nx-1)*nyz+offset]) / (2.*hx);
                    } else if (x == nx-1) {
                        out[x*nyz+offset] = (field[0*nyz+offset] - field[(x-1)*nyz+offset]) / (2.*hx);
                    } else {
                        out[x*nyz+offset] = (field[(x+1)*nyz+offset] - field[(x-1)*nyz+offset]) / (2.*hx);
                    }

                    offset = x*nyz + z;
                    if (y == 0) {
                        out[y*nz+offset+nxyz] = (field[(y+1)*nz+offset] - field[(ny-1)*nz+offset]) / (2.*hy);
                    } else if (y == ny-1) {
                        out[y*nz+offset+nxyz] = (field[0*nz+offset] - field[(y-1)*nz+offset]) / (2.*hy);
                    } else {
                        out[y*nz+offset+nxyz] = (field[(y+1)*nz+offset] - field[(y-1)*nz+offset]) / (2.*hy);
                    }

                    offset = x*nyz + y*nz;
                    if (z == 0) {
                        out[z+offset+2*nxyz] = (field[z+1+offset] - field[nz-1+offset]) / (2.*hz);
                    }
                    else if (z == nz-1) {
                        out[z+offset+2*nxyz] = (field[0+offset] - field[z-1+offset]) / (2.*hz);
                    }
                    else {
                        out[z+offset+2*nxyz] = (field[z+1+offset] - field[z-1+offset]) / (2.*hz);
                    }
                }
            }
        }
    }
}
