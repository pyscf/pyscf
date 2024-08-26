/* Copyright 2021- The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "pbc/cell.h"
#include "np_helper/np_helper.h"

#define SQUARE(r) (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])

double pgf_rcut(int l, double alpha, double coeff, double precision, double r0)
{
    l += 2;

    double rcut;
    double rmin = sqrt(.5 * l / alpha) * 2.;
    double gmax = coeff * pow(rmin, l) * exp(-alpha * rmin * rmin);
    if (gmax < precision) {
        return rmin;
    }

    double eps = MIN(rmin/10, RCUT_EPS);
    double c = log(coeff / precision);
    double rcut_last;
    rcut = MAX(r0, rmin+eps);

    int i;
    for (i = 0; i < RCUT_MAX_CYCLE; i++) {
        rcut_last = rcut;
        rcut = sqrt((l*log(rcut) + c) / alpha);
        if (fabs(rcut - rcut_last) < eps) {
            break;
        }
    }
    if (i == RCUT_MAX_CYCLE) {
        //printf("r0 = %.6e, l = %d, alpha = %.6e, coeff = %.6e, precision=%.6e\n", r0, l, alpha, coeff, precision);
        fprintf(stderr, "pgf_rcut did not converge in %d cycles: %.6f > %.6f.\n",
                RCUT_MAX_CYCLE, fabs(rcut - rcut_last), eps);
    }
    return rcut; 
}

void rcut_by_shells(double* shell_radius, double** ptr_pgf_rcut, 
                    int* bas, double* env, int nbas, 
                    double r0, double precision)
{
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
            rcut = pgf_rcut(l, alpha, cmax, precision, r0);
            if (ptr_pgf_rcut) {
                ptr_pgf_rcut[ib][p] = rcut;
            }
            rcut_max = MAX(rcut, rcut_max);
        }
        shell_radius[ib] = rcut_max;
    }
}
}


static void get_SI_real_imag(double* out_real, double* out_imag,
                             double* coords, double* Gv,
                             int natm, size_t ngrid)
{
#pragma omp parallel
{
    int ia;
    size_t i;
    double RG;
    double *pcoords, *pGv;
    double *pout_real, *pout_imag;
    #pragma omp for schedule(static)
    for (ia = 0; ia < natm; ia++) {
        pcoords = coords + ia * 3;
        pout_real = out_real + ia * ngrid;
        pout_imag = out_imag + ia * ngrid;
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


void ewald_gs_nuc_grad(double* out, double* Gv, double* charges, double* coords,
                       double ew_eta, double weights, int natm, size_t ngrid)
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
    int ia;
    size_t i;
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


void get_ewald_direct(double* ewovrl, double* chargs, double* coords, double* Ls,
                      double beta, double rcut, int natm, int nL)
{
    *ewovrl = 0.0;

    #pragma omp parallel
    {
        int i, j, l;
        double *ri, *rj, *rL;
        double rij[3];
        double r, qi, qj;
        double e_loc = 0.0;
        #pragma omp for schedule(static)
        for (i = 0; i < natm; i++) {
            ri = coords + i*3;
            qi = chargs[i];
            for (j = 0; j < natm; j++) {
                rj = coords + j*3;
                qj = chargs[j];
                for (l = 0; l < nL; l++) {
                    rL = Ls + l*3;
                    rij[0] = rj[0] + rL[0] - ri[0];
                    rij[1] = rj[1] + rL[1] - ri[1];
                    rij[2] = rj[2] + rL[2] - ri[2];
                    r = sqrt(SQUARE(rij));
                    if (r > 1e-10 && r < rcut) {
                        e_loc += qi * qj * erfc(beta * r) / r;
                    }
                }
            }
        }
        e_loc *= 0.5;

        #pragma omp critical
        *ewovrl += e_loc;
    }
}


void get_ewald_direct_nuc_grad(double* out, double* chargs, double* coords, double* Ls,
                               double beta, double rcut, int natm, int nL)
{
    double fac = 2. * beta / sqrt(M_PI);
    double beta2 = beta * beta;

    #pragma omp parallel
    {
        int i, j, l;
        double *ri, *rj, *rL, *pout;
        double rij[3];
        double r, r2, qi, qj, tmp;
        #pragma omp for schedule(static)
        for (i = 0; i < natm; i++) {
            pout = out + i*3;
            ri = coords + i*3;
            qi = chargs[i];
            for (j = 0; j < natm; j++) {
                rj = coords + j*3;
                qj = chargs[j];
                for (l = 0; l < nL; l++) {
                    rL = Ls + l*3;
                    rij[0] = ri[0] - rj[0] + rL[0];
                    rij[1] = ri[1] - rj[1] + rL[1];
                    rij[2] = ri[2] - rj[2] + rL[2];
                    r2 = SQUARE(rij);
                    r = sqrt(r2);
                    if (r > 1e-10 && r < rcut) {
                        tmp  = qi * qj * (erfc(beta * r) / (r2 * r) + fac * exp(-beta2 * r2) / r2);
                        pout[0] -= tmp * rij[0];
                        pout[1] -= tmp * rij[1];
                        pout[2] -= tmp * rij[2];
                    }
                }
            }
        }
    }
}
