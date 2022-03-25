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
