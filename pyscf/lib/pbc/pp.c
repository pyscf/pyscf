#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "vhf/fblas.h"
#include "config.h"

#define HL_TABLE_SLOTS    6

//#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
static double maxval(double* a, int n)
{
    double amax = 0;
    for (size_t i=0; i<n; i++) {
        amax = MAX(amax, fabs(a[i]));
    }
    return amax;
}*/

void contract_ppnl(double* out,
                   double* ppnl_half0, 
                   double* ppnl_half1, 
                   double* ppnl_half2,
                   int* hl_table, double* hl_data,
                   int nhl, int nao)
{
    const int One = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D1 = 1.;
    const double D0 = 0.;

    size_t nao_pair = (size_t) nao*nao;
    for (size_t i = 0; i < nao_pair; i++) {
        out[i] = 0;
    }

#pragma omp parallel
{
    size_t ib, i, p;
    double *pout;
    double *buf = (double*) malloc(sizeof(double)*54*nao);
    #pragma omp for schedule(dynamic)
    for (p = 0; p < nao; p++){
        pout = out + (size_t)p*nao;
    for (ib = 0; ib < nhl; ib++) {
        int *table = hl_table + ib * HL_TABLE_SLOTS;
        int hl_dim = table[0];
        int ptr = table[1];
        int nd = table[2];
        int *offset = table + 3;
        double *hl = hl_data + ptr;
        double *ilp = buf;
        int lp_dim = nd * nao;
        //int ilp_dim = hl_dim * lp_dim;
        int il_dim = hl_dim * nd;
        for (i=0; i<hl_dim; i++) {
            int p0 = offset[i];
            if (i == 0) {
                dcopy_(&lp_dim, ppnl_half0+p0*nao, &One, ilp+i*lp_dim, &One);
            }
            else if (i == 1) {
                dcopy_(&lp_dim, ppnl_half1+p0*nao, &One, ilp+i*lp_dim, &One);
            }
            else if (i == 2) {
                dcopy_(&lp_dim, ppnl_half2+p0*nao, &One, ilp+i*lp_dim, &One);
            }
        }
        double *hilp = ilp + hl_dim*lp_dim;
        dgemm_(&TRANS_N, &TRANS_N, &lp_dim, &hl_dim, &hl_dim, 
               &D1, ilp, &lp_dim, hl, &hl_dim, &D0, hilp, &lp_dim);
        dgemm_(&TRANS_N, &TRANS_T, &nao, &One, &il_dim,
               &D1, hilp, &nao, ilp+p, &nao, &D1, pout, &nao);
    }}
    free(buf);
}
}


void pp_loc_part1_gs(double complex* out, double* coulG,
                     double* Gv, double* G2, int G0idx, int ngrid,
                     double* Z, double* coords, double* rloc,
                     int natm)
{
#pragma omp parallel
{
    int ig, ia;
    double vlocG, r0, RG;
    double *Gv_loc, *coords_local;
    #pragma omp for schedule(static)
    for (ig = 0; ig < ngrid; ig++){
        out[ig] = 0;
        Gv_loc = Gv + ig*3;
        for (ia = 0; ia < natm; ia++)
        {
            coords_local = coords + ia*3;
            RG = (coords_local[0] * Gv_loc[0]
                  + coords_local[1] * Gv_loc[1]
                  + coords_local[2] * Gv_loc[2]);

            r0 = rloc[ia];
            if (r0 > 0) {
                if (ig == G0idx) {
                    vlocG = -2. * M_PI * Z[ia] * r0*r0;
                }
                else {
                    vlocG = Z[ia] * coulG[ig] * exp(-0.5*r0*r0 * G2[ig]);
                }
            }
            else { // Z/r
                vlocG = Z[ia] * coulG[ig];
            }
            out[ig] -= (vlocG * cos(RG)) - (vlocG * sin(RG)) * _Complex_I;
        }
    }
}
}
