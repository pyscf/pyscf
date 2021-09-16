#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "pbc/cell.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define RCUT_EPS 1e-3

double pgf_rcut(int l, double alpha, double coeff, 
                double precision, double r0, int max_cycle)
{
    double rcut;
    if (l == 0) {
        if (coeff <= precision) {
            rcut = 0;
        }
        else {
            rcut = sqrt(log(coeff / precision) / alpha);
        }
        return rcut;
    }

    double rmin = sqrt(.5 * l / alpha);
    double gmax = coeff * pow(rmin, l) * exp(-alpha * rmin * rmin);
    if (gmax < precision) {
        return 0;
    }

    int i;
    double eps = MIN(rmin/10, RCUT_EPS);
    double log_c_by_prec= log(coeff / precision);
    double rcut_old;
    rcut = MAX(r0, rmin+eps);
    for (i = 0; i < max_cycle; i++) {
        rcut_old = rcut;
        rcut = sqrt((l*log(rcut) + log_c_by_prec) / alpha);
        if (fabs(rcut - rcut_old) < eps) {
            break;
        }
    }
    if (i == max_cycle) {
        printf("pgf_rcut did not converge");
    }
    return rcut; 
}

void rcut_by_shells(double* shell_radius, double** ptr_pgf_rcut, 
                    int* bas, double* env, int nbas, 
                    double r0, double precision)
{
    int max_cycle = RCUT_MAX_CYCLE;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < nbas; ib ++) {
        int l = bas[ANG_OF+ib*BAS_SLOTS];
        int nprim = bas[NPRIM_OF+ib*BAS_SLOTS];
        int ptr_exp = bas[PTR_EXP+ib*BAS_SLOTS];
        int nctr = bas[NCTR_OF+ib*BAS_SLOTS];
        int ptr_c = bas[PTR_COEFF+ib*BAS_SLOTS];
        double rcut_max = 0, rcut;
        for (int p=0; p<nprim; p++) {
            double alpha = env[ptr_exp+p];
            double cmax = 0;
            for (int ic=0; ic<nctr; ic++) {
                cmax = MAX(fabs(env[ptr_c+ic*nprim+p]), cmax);
            }
            rcut = pgf_rcut(l, alpha, cmax, precision, r0, max_cycle);
            if (ptr_pgf_rcut) {
                ptr_pgf_rcut[ib][p] = rcut;
            }
            rcut_max = MAX(rcut, rcut_max);
        }
        shell_radius[ib] = rcut_max;
    }
}
