#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "pbc/cell.h"
#include "np_helper/np_helper.h"

#define SQUARE(r) (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
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
#pragma omp parallel
{
    int ib, ic, p;
    #pragma omp for schedule(static)
    for (ib = 0; ib < nbas; ib ++) {
        int l = bas[ANG_OF+ib*BAS_SLOTS];
        int nprim = bas[NPRIM_OF+ib*BAS_SLOTS];
        int ptr_exp = bas[PTR_EXP+ib*BAS_SLOTS];
        int nctr = bas[NCTR_OF+ib*BAS_SLOTS];
        int ptr_c = bas[PTR_COEFF+ib*BAS_SLOTS];
        double rcut_max = 0, rcut;
        for (p = 0; p < nprim; p++) {
            double alpha = env[ptr_exp+p];
            double cmax = 0;
            for (ic = 0; ic < nctr; ic++) {
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
}

void get_SI(complex double* out, double* coords, double* Gv, int natm, int ngrid)
{
#pragma omp parallel
{
    int i, ia;
    double RG;
    double *pcoords, *pGv;
    complex double *pout;
    #pragma omp for schedule(static)
    for (ia = 0; ia < natm; ia++) {
        pcoords = coords + ia * 3;
        pout = out + ((size_t)ia) * ngrid;
        for (i = 0; i < ngrid; i++) {
            pGv = Gv + i * 3;
            RG = pcoords[0] * pGv[0] + pcoords[1] * pGv[1] + pcoords[2] * pGv[2];
            pout[i] = cos(RG) - _Complex_I * sin(RG);
        }
    }
}
}


void get_SI_real_imag(double* out_real, double* out_imag, double* coords, double* Gv, int natm, int ngrid)
{
#pragma omp parallel
{
    int i, ia;
    double RG;
    double *pcoords, *pGv;
    double *pout_real, *pout_imag;
    #pragma omp for schedule(static)
    for (ia = 0; ia < natm; ia++) {
        pcoords = coords + ia * 3;
        pout_real = out_real + ((size_t)ia) * ngrid;
        pout_imag = out_imag + ((size_t)ia) * ngrid;
        for (i = 0; i < ngrid; i++) {
            pGv = Gv + i * 3;
            RG = pcoords[0] * pGv[0] + pcoords[1] * pGv[1] + pcoords[2] * pGv[2];
            pout_real[i] = cos(RG);
            pout_imag[i] = -sin(RG);
        }
    }
}
}


void get_Gv(double* Gv, double* rx, double* ry, double* rz, int* mesh, double* b)
{
#pragma omp parallel
{
    int x, y, z;
    double *pGv;
    #pragma omp for schedule(dynamic)
    for (x = 0; x < mesh[0]; x++) {
        pGv = Gv + x * (size_t)mesh[1] * mesh[2] * 3;
        for (y = 0; y < mesh[1]; y++) {
        for (z = 0; z < mesh[2]; z++) {
            pGv[0]  = rx[x] * b[0];
            pGv[0] += ry[y] * b[3];
            pGv[0] += rz[z] * b[6];
            pGv[1]  = rx[x] * b[1];
            pGv[1] += ry[y] * b[4];
            pGv[1] += rz[z] * b[7];
            pGv[2]  = rx[x] * b[2];
            pGv[2] += ry[y] * b[5];
            pGv[2] += rz[z] * b[8];
            pGv += 3;
        }}
    }
}
}


void ewald_overlap_nuc_grad(double* out, double* charges, double* coords, int natm, double* Ls, int nL, double ew_eta)
{
    double fac = 2. * ew_eta / sqrt(M_PI);
    double ew_eta2 = ew_eta * ew_eta;

#pragma omp parallel
{
    int i, j, iL;
    double charge_i, charge_j;
    double tmp, r, r2;
    double rij[3];
    double *pout, *ri, *rj, *pLs;
    #pragma omp for nowait schedule(static)
    for (i = 0; i < natm; i++) {
        charge_i = charges[i];
        pout = out + i*3;
        ri = coords + i*3;
        for (j = 0; j < natm; j++) {
            if (i == j) {
                continue;
            }
            charge_j = charges[j];
            rj = coords + j*3;
            rij[0] = ri[0] - rj[0];
            rij[1] = ri[1] - rj[1];
            rij[2] = ri[2] - rj[2];
            r2 = SQUARE(rij);
            r = sqrt(r2);
            tmp  = erfc(ew_eta * r) / (r2 * r) + fac * exp(-ew_eta2 * r2) / r2;
            tmp *= charge_i * charge_j;
            pout[0] -= tmp * rij[0];
            pout[1] -= tmp * rij[1];
            pout[2] -= tmp * rij[2];
        }
    }

    #pragma omp for schedule(static)
    for (i = 0; i < natm; i++) {
        charge_i = charges[i];
        pout = out + i*3;
        ri = coords + i*3;
        for (j = 0; j < natm; j++) {
            charge_j = charges[j];
            rj = coords + j*3;
            rij[0] = ri[0] - rj[0];
            rij[1] = ri[1] - rj[1];
            rij[2] = ri[2] - rj[2];

            for (iL = 0; iL < nL; iL++) {
                pLs = Ls + iL * 3;
                rij[0] = ri[0] - rj[0] + pLs[0];
                rij[1] = ri[1] - rj[1] + pLs[1];
                rij[2] = ri[2] - rj[2] + pLs[2];
                r2 = SQUARE(rij);
                r = sqrt(r2);
                tmp  = erfc(ew_eta * r) / (r2 * r) + fac * exp(-ew_eta2 * r2) / r2;
                tmp *= charge_i * charge_j;
                pout[0] -= tmp * rij[0];
                pout[1] -= tmp * rij[1];
                pout[2] -= tmp * rij[2];
            }
        }
    }
}
}


void ewald_gs_nuc_grad(double* out, double* Gv, double* charges, double* coords, double ew_eta, double weights, int natm, int ngrid)
{
    double *SI_real = (double*) malloc(natm*ngrid*sizeof(double));
    double *SI_imag = (double*) malloc(natm*ngrid*sizeof(double)); 
    get_SI_real_imag(SI_real, SI_imag, coords, Gv, natm, ngrid);

    double *ZSI_real = calloc(ngrid, sizeof(double));
    double *ZSI_imag = calloc(ngrid, sizeof(double));

    NPdgemm('N', 'N', ngrid, 1, natm,
            ngrid, natm, ngrid, 0, 0, 0,
            SI_real, charges, ZSI_real, 1., 0.);
    NPdgemm('N', 'N', ngrid, 1, natm,
            ngrid, natm, ngrid, 0, 0, 0,
            SI_imag, charges, ZSI_imag, 1., 0.);

#pragma omp parallel
{
    int ia, i;
    double charge_i;
    double G2, coulG, tmp;
    double *pout, *pGv;
    double *pSI_real, *pSI_imag;
    double fac = 4. * M_PI * weights;
    double fac1 = 4. * ew_eta * ew_eta;

    #pragma omp for schedule(static)
    for (ia = 0; ia < natm; ia++) {
        charge_i = charges[ia];
        pout = out + ia * 3;
        pSI_real = SI_real + ia * ngrid;
        pSI_imag = SI_imag + ia * ngrid;
        #pragma omp simd
        for (i = 0; i < ngrid; i++) {
            pGv = Gv + i*3;
            G2 = SQUARE(pGv);
            if (G2 < 1e-12) {continue;}
            coulG = fac / G2 * exp(-G2 / fac1);
            tmp  = coulG * charge_i;
            tmp *= (pSI_imag[i] * ZSI_real[i] - pSI_real[i] * ZSI_imag[i]);
            pout[0] += tmp * pGv[0];
            pout[1] += tmp * pGv[1];
            pout[2] += tmp * pGv[2];
        }
    }
}
    free(SI_real);
    free(SI_imag);
    free(ZSI_real);
    free(ZSI_imag);
}
