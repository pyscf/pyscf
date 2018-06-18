/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "grid_ao_drv.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double exp_cephes(double x);
double CINTcommon_fac_sp(int l);

int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac)
{
        size_t i, j, k;
        double arr, maxc, eprim;
        double logcoeff[nprim];
        double rr[ngrids];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        int not0 = 0;

        // the maximum value of the coefficients for each pGTO
        for (j = 0; j < nprim; j++) {
                maxc = 0;
                for (i = 0; i < nctr; i++) {
                        maxc = MAX(maxc, fabs(coeff[i*nprim+j]));
                }
                logcoeff[j] = log(maxc);
        }

        for (i = 0; i < ngrids; i++) {
                rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
        }

        for (i = 0; i < nctr*BLKSIZE; i++) {
                ectr[i] = 0;
        }
        for (j = 0; j < nprim; j++) {
        for (i = 0; i < ngrids; i++) {
                arr = alpha[j] * rr[i];
                if (arr-logcoeff[j] < EXPCUTOFF) {
                        not0 = 1;
                        eprim = exp_cephes(-arr) * fac;
                        for (k = 0; k < nctr; k++) {
                                ectr[k*BLKSIZE+i] += eprim * coeff[k*nprim+j];
                        }
                }
        } }

        return not0;
}

/*
 * deriv 0: exp(-ar^2) x^n
 * deriv 1: exp(-ar^2)[nx^{n-1} - 2ax^{n+1}]
 * deriv 2: exp(-ar^2)[n(n-1)x^{n-2} - 2a(2n+1)x^n + 4a^2x^{n+2}]
 * deriv 3: exp(-ar^2)[n(n-1)(n-2)x^{n-3} - 2a3n^2x^{n-1} + 4a^2(3n+3)x^{n+1} - 8a^3x^{n+3}]
 * deriv 4: exp(-ar^2)[n(n-1)(n-2)(n-3)x^{n-4} - 2a(4n^3-6n^2+2)x^n{-2}
 *                     + 4a^2(6n^2+6n+3)x^n - 8a(4n+6)x^{n+2} + 16a^4x^{n+4}]
 */

// pre-contracted grid AO evaluator
// contracted factors = \sum c_{i} exp(-a_i*r_i**2)
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff,
                             double *env, int l, int np, int nc,
                             size_t nao, size_t ngrids, size_t blksize)
{
        int lx, ly, lz;
        size_t i, k;
        double ce[3];
        double xpows[8*blksize];
        double ypows[8*blksize];
        double zpows[8*blksize];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;

        switch (l) {
        case 0:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                gto[k*ngrids+i] = exps[k*BLKSIZE+i];
                        }
                }
                break;
        case 1:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                gto[         i] = gridx[i] * exps[k*BLKSIZE+i];
                                gto[1*ngrids+i] = gridy[i] * exps[k*BLKSIZE+i];
                                gto[2*ngrids+i] = gridz[i] * exps[k*BLKSIZE+i];
                        }
                        gto += ngrids * 3;
                }
                break;
        case 2:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        ce[0] = gridx[i] * exps[k*BLKSIZE+i];
                                        ce[1] = gridy[i] * exps[k*BLKSIZE+i];
                                        ce[2] = gridz[i] * exps[k*BLKSIZE+i];
                                        gto[         i] = ce[0] * gridx[i]; // xx
                                        gto[1*ngrids+i] = ce[0] * gridy[i]; // xy
                                        gto[2*ngrids+i] = ce[0] * gridz[i]; // xz
                                        gto[3*ngrids+i] = ce[1] * gridy[i]; // yy
                                        gto[4*ngrids+i] = ce[1] * gridz[i]; // yz
                                        gto[5*ngrids+i] = ce[2] * gridz[i]; // zz
                                } else {
                                        gto[         i] = 0;
                                        gto[1*ngrids+i] = 0;
                                        gto[2*ngrids+i] = 0;
                                        gto[3*ngrids+i] = 0;
                                        gto[4*ngrids+i] = 0;
                                        gto[5*ngrids+i] = 0;
                                }
                        }
                        gto += ngrids * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        ce[0] = gridx[i] * gridx[i] * exps[k*BLKSIZE+i];
                                        ce[1] = gridy[i] * gridy[i] * exps[k*BLKSIZE+i];
                                        ce[2] = gridz[i] * gridz[i] * exps[k*BLKSIZE+i];
                                        gto[         i] = ce[0] * gridx[i]; // xxx
                                        gto[1*ngrids+i] = ce[0] * gridy[i]; // xxy
                                        gto[2*ngrids+i] = ce[0] * gridz[i]; // xxz
                                        gto[3*ngrids+i] = gridx[i] * ce[1]; // xyy
                                        gto[4*ngrids+i] = gridx[i]*gridy[i]*gridz[i] * exps[k*BLKSIZE+i]; // xyz
                                        gto[5*ngrids+i] = gridx[i] * ce[2]; // xzz
                                        gto[6*ngrids+i] = ce[1] * gridy[i]; // yyy
                                        gto[7*ngrids+i] = ce[1] * gridz[i]; // yyz
                                        gto[8*ngrids+i] = gridy[i] * ce[2]; // yzz
                                        gto[9*ngrids+i] = gridz[i] * ce[2]; // zzz
                                } else {
                                        gto[         i] = 0;
                                        gto[1*ngrids+i] = 0;
                                        gto[2*ngrids+i] = 0;
                                        gto[3*ngrids+i] = 0;
                                        gto[4*ngrids+i] = 0;
                                        gto[5*ngrids+i] = 0;
                                        gto[6*ngrids+i] = 0;
                                        gto[7*ngrids+i] = 0;
                                        gto[8*ngrids+i] = 0;
                                        gto[9*ngrids+i] = 0;
                                }
                        }
                        gto += ngrids * 10;
                }
                break;
        default:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                xpows[i] = 1;
                                ypows[i] = 1;
                                zpows[i] = 1;
                        }
                        for (lx = 1; lx < l+1; lx++) {
                                for (i = 0; i < blksize; i++) {
                                        xpows[lx*blksize+i] = xpows[(lx-1)*blksize+i] * gridx[i];
                                        ypows[lx*blksize+i] = ypows[(lx-1)*blksize+i] * gridy[i];
                                        zpows[lx*blksize+i] = zpows[(lx-1)*blksize+i] * gridz[i];
                                }
                        }
                        for (lx = l; lx >= 0; lx--) {
                        for (ly = l - lx; ly >= 0; ly--) {
                                lz = l - lx - ly;
                                for (i = 0; i < blksize; i++) {
                                        gto[i] = xpows[lx*blksize+i]
                                               * ypows[ly*blksize+i]
                                               * zpows[lz*blksize+i]*exps[k*BLKSIZE+i];
                                }
                                gto += ngrids;
                        } }
                }
        }
}

int GTOcontract_exp1(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac)
{
        size_t i, j, k;
        double arr, maxc, eprim;
        double logcoeff[nprim];
        double rr[ngrids];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *ectr_2a = ectr + NPRIMAX*BLKSIZE;
        double coeff2a[nprim*nctr];
        int not0 = 0;

        // the maximum value of the coefficients for each pGTO
        for (j = 0; j < nprim; j++) {
                maxc = 0;
                for (i = 0; i < nctr; i++) {
                        maxc = MAX(maxc, fabs(coeff[i*nprim+j]));
                }
                logcoeff[j] = log(maxc);
        }

        for (i = 0; i < ngrids; i++) {
                rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
        }

        memset(ectr   , 0, sizeof(double)*nctr*BLKSIZE);
        memset(ectr_2a, 0, sizeof(double)*nctr*BLKSIZE);
        // -2 alpha_i C_ij exp(-alpha_i r_k^2)
        for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
                coeff2a[i*nprim+j] = -2.*alpha[j] * coeff[i*nprim+j];
        } }

        for (j = 0; j < nprim; j++) {
        for (i = 0; i < ngrids; i++) {
                arr = alpha[j] * rr[i];
                if (arr-logcoeff[j] < EXPCUTOFF) {
                        not0 = 1;
                        eprim = exp_cephes(-arr) * fac;
                        for (k = 0; k < nctr; k++) {
                                ectr   [k*BLKSIZE+i] += eprim * coeff  [k*nprim+j];
                                ectr_2a[k*BLKSIZE+i] += eprim * coeff2a[k*nprim+j];
                        }
                }
        } }

        return not0;
}

void GTOshell_eval_grid_ip_cart(double *gto, double *ri, double *exps,
                                double *coord, double *alpha, double *coeff,
                                double *env, int l, int np, int nc,
                                size_t nao, size_t ngrids, size_t blksize)
{
        const size_t degen = (l+1)*(l+2)/2;
        int lx, ly, lz;
        size_t i, k, n;
        double ax, ay, az, tmp;
        double ce[6];
        double rre[10];
        double xpows_1less_in_power[64];
        double ypows_1less_in_power[64];
        double zpows_1less_in_power[64];
        double *xpows = xpows_1less_in_power + 1;
        double *ypows = ypows_1less_in_power + 1;
        double *zpows = zpows_1less_in_power + 1;
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *gtox = gto;
        double *gtoy = gto + nao * ngrids;
        double *gtoz = gto + nao * ngrids * 2;
        double *exps_2a = exps + NPRIMAX*BLKSIZE;

        switch (l) {
        case 0:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                gtox[i] = exps_2a[k*BLKSIZE+i] * gridx[i];
                                gtoy[i] = exps_2a[k*BLKSIZE+i] * gridy[i];
                                gtoz[i] = exps_2a[k*BLKSIZE+i] * gridz[i];
                        }
                        gtox += ngrids;
                        gtoy += ngrids;
                        gtoz += ngrids;
                }
                break;
        case 1:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        ax = exps_2a[k*BLKSIZE+i] * gridx[i];
                                        ay = exps_2a[k*BLKSIZE+i] * gridy[i];
                                        az = exps_2a[k*BLKSIZE+i] * gridz[i];
                                        gtox[         i] = ax * gridx[i] + exps[k*BLKSIZE+i];
                                        gtox[1*ngrids+i] = ax * gridy[i];
                                        gtox[2*ngrids+i] = ax * gridz[i];
                                        gtoy[         i] = ay * gridx[i];
                                        gtoy[1*ngrids+i] = ay * gridy[i] + exps[k*BLKSIZE+i];
                                        gtoy[2*ngrids+i] = ay * gridz[i];
                                        gtoz[         i] = az * gridx[i];
                                        gtoz[1*ngrids+i] = az * gridy[i];
                                        gtoz[2*ngrids+i] = az * gridz[i] + exps[k*BLKSIZE+i];
                                } else {
                                        gtox[         i] = 0;
                                        gtox[1*ngrids+i] = 0;
                                        gtox[2*ngrids+i] = 0;
                                        gtoy[         i] = 0;
                                        gtoy[1*ngrids+i] = 0;
                                        gtoy[2*ngrids+i] = 0;
                                        gtoz[         i] = 0;
                                        gtoz[1*ngrids+i] = 0;
                                        gtoz[2*ngrids+i] = 0;
                                }
                        }
                        gtox += ngrids * 3;
                        gtoy += ngrids * 3;
                        gtoz += ngrids * 3;
                }
                break;
        case 2:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        tmp = exps_2a[k*BLKSIZE+i]/(exps[k*BLKSIZE+i]+1e-200);
                                        ax = tmp * gridx[i];
                                        ay = tmp * gridy[i];
                                        az = tmp * gridz[i];
                                        ce[0] = gridx[i] * exps[k*BLKSIZE+i];
                                        ce[1] = gridy[i] * exps[k*BLKSIZE+i];
                                        ce[2] = gridz[i] * exps[k*BLKSIZE+i];
                                        rre[0] = gridx[i] * ce[0]; // xx
                                        rre[1] = gridx[i] * ce[1]; // xy
                                        rre[2] = gridx[i] * ce[2]; // xz
                                        rre[3] = gridy[i] * ce[1]; // yy
                                        rre[4] = gridy[i] * ce[2]; // yz
                                        rre[5] = gridz[i] * ce[2]; // zz
                                        gtox[         i] = ax * rre[0] + 2 * ce[0];
                                        gtox[1*ngrids+i] = ax * rre[1] +     ce[1];
                                        gtox[2*ngrids+i] = ax * rre[2] +     ce[2];
                                        gtox[3*ngrids+i] = ax * rre[3];
                                        gtox[4*ngrids+i] = ax * rre[4];
                                        gtox[5*ngrids+i] = ax * rre[5];
                                        gtoy[         i] = ay * rre[0];
                                        gtoy[1*ngrids+i] = ay * rre[1] +     ce[0];
                                        gtoy[2*ngrids+i] = ay * rre[2];
                                        gtoy[3*ngrids+i] = ay * rre[3] + 2 * ce[1];
                                        gtoy[4*ngrids+i] = ay * rre[4] +     ce[2];
                                        gtoy[5*ngrids+i] = ay * rre[5];
                                        gtoz[         i] = az * rre[0];
                                        gtoz[1*ngrids+i] = az * rre[1];
                                        gtoz[2*ngrids+i] = az * rre[2] +     ce[0];
                                        gtoz[3*ngrids+i] = az * rre[3];
                                        gtoz[4*ngrids+i] = az * rre[4] +     ce[1];
                                        gtoz[5*ngrids+i] = az * rre[5] + 2 * ce[2];
                                } else {
                                        gtox[         i] = 0;
                                        gtox[1*ngrids+i] = 0;
                                        gtox[2*ngrids+i] = 0;
                                        gtox[3*ngrids+i] = 0;
                                        gtox[4*ngrids+i] = 0;
                                        gtox[5*ngrids+i] = 0;
                                        gtoy[         i] = 0;
                                        gtoy[1*ngrids+i] = 0;
                                        gtoy[2*ngrids+i] = 0;
                                        gtoy[3*ngrids+i] = 0;
                                        gtoy[4*ngrids+i] = 0;
                                        gtoy[5*ngrids+i] = 0;
                                        gtoz[         i] = 0;
                                        gtoz[1*ngrids+i] = 0;
                                        gtoz[2*ngrids+i] = 0;
                                        gtoz[3*ngrids+i] = 0;
                                        gtoz[4*ngrids+i] = 0;
                                        gtoz[5*ngrids+i] = 0;
                                }
                        }
                        gtox += ngrids * 6;
                        gtoy += ngrids * 6;
                        gtoz += ngrids * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        tmp = exps_2a[k*BLKSIZE+i]/(exps[k*BLKSIZE+i]+1e-200);
                                        ax = tmp * gridx[i];
                                        ay = tmp * gridy[i];
                                        az = tmp * gridz[i];
                                        ce[0] = gridx[i] * gridx[i] * exps[k*BLKSIZE+i];
                                        ce[1] = gridx[i] * gridy[i] * exps[k*BLKSIZE+i];
                                        ce[2] = gridx[i] * gridz[i] * exps[k*BLKSIZE+i];
                                        ce[3] = gridy[i] * gridy[i] * exps[k*BLKSIZE+i];
                                        ce[4] = gridy[i] * gridz[i] * exps[k*BLKSIZE+i];
                                        ce[5] = gridz[i] * gridz[i] * exps[k*BLKSIZE+i];
                                        rre[0] = gridx[i] * ce[0]; // xxx
                                        rre[1] = gridx[i] * ce[1]; // xxy
                                        rre[2] = gridx[i] * ce[2]; // xxz
                                        rre[3] = gridx[i] * ce[3]; // xyy
                                        rre[4] = gridx[i] * ce[4]; // xyz
                                        rre[5] = gridx[i] * ce[5]; // xzz
                                        rre[6] = gridy[i] * ce[3]; // yyy
                                        rre[7] = gridy[i] * ce[4]; // yyz
                                        rre[8] = gridy[i] * ce[5]; // yzz
                                        rre[9] = gridz[i] * ce[5]; // zzz
                                        gtox[         i] = ax * rre[0] + 3 * ce[0];
                                        gtox[1*ngrids+i] = ax * rre[1] + 2 * ce[1];
                                        gtox[2*ngrids+i] = ax * rre[2] + 2 * ce[2];
                                        gtox[3*ngrids+i] = ax * rre[3] +     ce[3];
                                        gtox[4*ngrids+i] = ax * rre[4] +     ce[4];
                                        gtox[5*ngrids+i] = ax * rre[5] +     ce[5];
                                        gtox[6*ngrids+i] = ax * rre[6];
                                        gtox[7*ngrids+i] = ax * rre[7];
                                        gtox[8*ngrids+i] = ax * rre[8];
                                        gtox[9*ngrids+i] = ax * rre[9];
                                        gtoy[         i] = ay * rre[0];
                                        gtoy[1*ngrids+i] = ay * rre[1] +     ce[0];
                                        gtoy[2*ngrids+i] = ay * rre[2];
                                        gtoy[3*ngrids+i] = ay * rre[3] + 2 * ce[1];
                                        gtoy[4*ngrids+i] = ay * rre[4] +     ce[2];
                                        gtoy[5*ngrids+i] = ay * rre[5];
                                        gtoy[6*ngrids+i] = ay * rre[6] + 3 * ce[3];
                                        gtoy[7*ngrids+i] = ay * rre[7] + 2 * ce[4];
                                        gtoy[8*ngrids+i] = ay * rre[8] +     ce[5];
                                        gtoy[9*ngrids+i] = ay * rre[9];
                                        gtoz[         i] = az * rre[0];
                                        gtoz[1*ngrids+i] = az * rre[1];
                                        gtoz[2*ngrids+i] = az * rre[2] +     ce[0];
                                        gtoz[3*ngrids+i] = az * rre[3];
                                        gtoz[4*ngrids+i] = az * rre[4] +     ce[1];
                                        gtoz[5*ngrids+i] = az * rre[5] + 2 * ce[2];
                                        gtoz[6*ngrids+i] = az * rre[6];
                                        gtoz[7*ngrids+i] = az * rre[7] +     ce[3];
                                        gtoz[8*ngrids+i] = az * rre[8] + 2 * ce[4];
                                        gtoz[9*ngrids+i] = az * rre[9] + 3 * ce[5];
                                } else {
                                        gtox[         i] = 0;
                                        gtox[1*ngrids+i] = 0;
                                        gtox[2*ngrids+i] = 0;
                                        gtox[3*ngrids+i] = 0;
                                        gtox[4*ngrids+i] = 0;
                                        gtox[5*ngrids+i] = 0;
                                        gtox[6*ngrids+i] = 0;
                                        gtox[7*ngrids+i] = 0;
                                        gtox[8*ngrids+i] = 0;
                                        gtox[9*ngrids+i] = 0;
                                        gtoy[         i] = 0;
                                        gtoy[1*ngrids+i] = 0;
                                        gtoy[2*ngrids+i] = 0;
                                        gtoy[3*ngrids+i] = 0;
                                        gtoy[4*ngrids+i] = 0;
                                        gtoy[5*ngrids+i] = 0;
                                        gtoy[6*ngrids+i] = 0;
                                        gtoy[7*ngrids+i] = 0;
                                        gtoy[8*ngrids+i] = 0;
                                        gtoy[9*ngrids+i] = 0;
                                        gtoz[         i] = 0;
                                        gtoz[1*ngrids+i] = 0;
                                        gtoz[2*ngrids+i] = 0;
                                        gtoz[3*ngrids+i] = 0;
                                        gtoz[4*ngrids+i] = 0;
                                        gtoz[5*ngrids+i] = 0;
                                        gtoz[6*ngrids+i] = 0;
                                        gtoz[7*ngrids+i] = 0;
                                        gtoz[8*ngrids+i] = 0;
                                        gtoz[9*ngrids+i] = 0;
                                }
                        }
                        gtox += ngrids * 10;
                        gtoy += ngrids * 10;
                        gtoz += ngrids * 10;
                }
                break;
        default:
                xpows_1less_in_power[0] = 0;
                ypows_1less_in_power[0] = 0;
                zpows_1less_in_power[0] = 0;
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        xpows[0] = 1;
                                        ypows[0] = 1;
                                        zpows[0] = 1;
                                        for (lx = 1; lx <= l; lx++) {
                                                xpows[lx] = xpows[lx-1] *gridx[i];
                                                ypows[lx] = ypows[lx-1] *gridy[i];
                                                zpows[lx] = zpows[lx-1] *gridz[i];
                                        }
                                        for (lx = l, n = 0; lx >= 0; lx--) {
                                        for (ly = l - lx; ly >= 0; ly--, n++) {
                                                lz = l - lx - ly;
                                                tmp = xpows[lx] * ypows[ly] * zpows[lz];
                                                gtox[n*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridx[i] * tmp;
                                                gtoy[n*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridy[i] * tmp;
                                                gtoz[n*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridz[i] * tmp;
                                                gtox[n*ngrids+i] += exps[k*BLKSIZE+i] * lx * xpows[lx-1] * ypows[ly] * zpows[lz];
                                                gtoy[n*ngrids+i] += exps[k*BLKSIZE+i] * ly * xpows[lx] * ypows[ly-1] * zpows[lz];
                                                gtoz[n*ngrids+i] += exps[k*BLKSIZE+i] * lz * xpows[lx] * ypows[ly] * zpows[lz-1];
                                        } }
                                } else {
                                        for (n = 0; n < degen; n++) {
                                                gtox[n*ngrids+i] = 0;
                                                gtoy[n*ngrids+i] = 0;
                                                gtoz[n*ngrids+i] = 0;
                                        }
                                }
                        }
                        gtox += ngrids * degen;
                        gtoy += ngrids * degen;
                        gtoz += ngrids * degen;
                }
        }
}

void GTOval_cart(int ngrids, int *shls_slice, int *ao_loc,
                 double *ao, double *coord, char *non0table,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_cart_drv(GTOshell_eval_grid_cart, GTOcontract_exp0, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph(int ngrids, int *shls_slice, int *ao_loc,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor(int ngrids, int *shls_slice, int *ao_loc,
                   double complex *ao, double *coord, char *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_spinor_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOval_ip_cart(int ngrids, int *shls_slice, int *ao_loc,
                    double *ao, double *coord, char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_cart_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_ip_sph(int ngrids, int *shls_slice, int *ao_loc,
                   double *ao, double *coord, char *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_sph_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_ip_spinor(int ngrids, int *shls_slice, int *ao_loc,
                      double complex *ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_spinor_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

