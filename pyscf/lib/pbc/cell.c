#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "cint.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void rcut_by_shells(double* out, int* bas, double* env, int nbas, 
                    double r0, double precision)
{
    double log_prec = log(precision);
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < nbas; ib ++) {
        int l = bas[ANG_OF+ib*BAS_SLOTS];
        int nprim = bas[NPRIM_OF+ib*BAS_SLOTS];
        int ptr_exp = bas[PTR_EXP+ib*BAS_SLOTS];
        int nctr = bas[NCTR_OF+ib*BAS_SLOTS];
        int ptr_c = bas[PTR_COEFF+ib*BAS_SLOTS];
        double rcut = 0, rcut_max=0;
        for (int p=0; p<nprim; p++) {
            double alpha = env[ptr_exp+p];
            double cmax = 0;
            for (int ic=0; ic<nctr; ic++) {
                cmax = MAX(fabs(env[ptr_c+ic*nprim+p]), cmax);
            }
            rcut = sqrt((l*log(r0) + log(cmax) - log_prec) / alpha);
            rcut = sqrt((l*log(rcut) + log(cmax) - log_prec) / alpha);
            rcut_max = MAX(rcut, rcut_max);
        }
        out[ib] = rcut_max;
    }
}
