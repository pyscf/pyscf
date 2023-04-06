/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.

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
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include "grid_ao_drv.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double CINTcommon_fac_sp(int l);

int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac)
{
        size_t i, j, k;
        double arr, eprim;
        double rr[BLKSIZE];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;

#pragma GCC ivdep
        for (i = 0; i < ngrids; i++) {
                rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
        }

        for (i = 0; i < nctr*BLKSIZE; i++) {
                ectr[i] = 0;
        }
        for (j = 0; j < nprim; j++) {
        for (i = 0; i < ngrids; i++) {
                arr = alpha[j] * rr[i];
                eprim = exp(-arr) * fac;
                for (k = 0; k < nctr; k++) {
                        ectr[k*BLKSIZE+i] += eprim * coeff[k*nprim+j];
                }
        } }
        return 1;
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
        double buf[(LMAX+1)*3 * BLKSIZE + 8];
        double *xpows = ALIGN8_UP(buf);
        double *ypows = xpows + (LMAX+1) * BLKSIZE;
        double *zpows = ypows + (LMAX+1) * BLKSIZE;
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
#pragma GCC ivdep
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
#pragma GCC ivdep
                        for (i = 0; i < blksize; i++) {
                                gto[         i] = exps[k*BLKSIZE+i] * gridx[i] * gridx[i]; // xx
                                gto[1*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridy[i]; // xy
                                gto[2*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridz[i]; // xz
                                gto[3*ngrids+i] = exps[k*BLKSIZE+i] * gridy[i] * gridy[i]; // yy
                                gto[4*ngrids+i] = exps[k*BLKSIZE+i] * gridy[i] * gridz[i]; // yz
                                gto[5*ngrids+i] = exps[k*BLKSIZE+i] * gridz[i] * gridz[i]; // zz
                        }
                        gto += ngrids * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
#pragma GCC ivdep
                        for (i = 0; i < blksize; i++) {
                                gto[         i] = exps[k*BLKSIZE+i] * gridx[i] * gridx[i] * gridx[i]; // xxx
                                gto[1*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridx[i] * gridy[i]; // xxy
                                gto[2*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridx[i] * gridz[i]; // xxz
                                gto[3*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridy[i] * gridy[i]; // xyy
                                gto[4*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridy[i] * gridz[i]; // xyz
                                gto[5*ngrids+i] = exps[k*BLKSIZE+i] * gridx[i] * gridz[i] * gridz[i]; // xzz
                                gto[6*ngrids+i] = exps[k*BLKSIZE+i] * gridy[i] * gridy[i] * gridy[i]; // yyy
                                gto[7*ngrids+i] = exps[k*BLKSIZE+i] * gridy[i] * gridy[i] * gridz[i]; // yyz
                                gto[8*ngrids+i] = exps[k*BLKSIZE+i] * gridy[i] * gridz[i] * gridz[i]; // yzz
                                gto[9*ngrids+i] = exps[k*BLKSIZE+i] * gridz[i] * gridz[i] * gridz[i]; // zzz
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
                        for (lx = 1; lx <= l; lx++) {
#pragma GCC ivdep
                                for (i = 0; i < blksize; i++) {
                                        xpows[lx*BLKSIZE+i] = xpows[(lx-1)*BLKSIZE+i] * gridx[i];
                                        ypows[lx*BLKSIZE+i] = ypows[(lx-1)*BLKSIZE+i] * gridy[i];
                                        zpows[lx*BLKSIZE+i] = zpows[(lx-1)*BLKSIZE+i] * gridz[i];
                                }
                        }
                        for (lx = l; lx >= 0; lx--) {
                        for (ly = l - lx; ly >= 0; ly--) {
                                lz = l - lx - ly;
#pragma GCC ivdep
                                for (i = 0; i < blksize; i++) {
                                        gto[i] = xpows[lx*BLKSIZE+i]
                                               * ypows[ly*BLKSIZE+i]
                                               * zpows[lz*BLKSIZE+i]*exps[k*BLKSIZE+i];
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
        double arr, eprim;
        double rr[BLKSIZE];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *ectr_2a = ectr + NPRIMAX*BLKSIZE;
        double coeff2a[NPRIMAX*NPRIMAX];

#pragma GCC ivdep
        for (i = 0; i < ngrids; i++) {
                rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
        }

        for (i = 0; i < nctr * BLKSIZE; i++) {
                ectr   [i] = 0;
                ectr_2a[i] = 0;
        }
        // -2 alpha_i C_ij exp(-alpha_i r_k^2)
        for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
                coeff2a[i*nprim+j] = -2.*alpha[j] * coeff[i*nprim+j];
        } }

        for (j = 0; j < nprim; j++) {
        for (i = 0; i < ngrids; i++) {
                arr = alpha[j] * rr[i];
                eprim = exp(-arr) * fac;
                for (k = 0; k < nctr; k++) {
                        ectr   [k*BLKSIZE+i] += eprim * coeff  [k*nprim+j];
                        ectr_2a[k*BLKSIZE+i] += eprim * coeff2a[k*nprim+j];
                }
        } }
        return 1;
}

void GTOshell_eval_grid_ip_cart(double *gto, double *ri, double *exps,
                                double *coord, double *alpha, double *coeff,
                                double *env, int l, int np, int nc,
                                size_t nao, size_t ngrids, size_t blksize)
{
        const size_t degen = (l+1)*(l+2)/2;
        int lx, ly, lz;
        size_t i, k, n;
        double ax, ay, az, tmp, e2a, e, rx, ry, rz;
        double ce[6];
        double rr[10];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *gtox = gto;
        double *gtoy = gto + nao * ngrids;
        double *gtoz = gto + nao * ngrids * 2;
        double *exps_2a = exps + NPRIMAX*BLKSIZE;
        double buf[(LMAX+2)*3 * BLKSIZE + 8];
        double *xpows_1less_in_power = ALIGN8_UP(buf);
        double *ypows_1less_in_power = xpows_1less_in_power + (LMAX+2)* BLKSIZE;
        double *zpows_1less_in_power = ypows_1less_in_power + (LMAX+2)* BLKSIZE;
        double *xpows = xpows_1less_in_power + BLKSIZE;
        double *ypows = ypows_1less_in_power + BLKSIZE;
        double *zpows = zpows_1less_in_power + BLKSIZE;

        switch (l) {
        case 0:
                for (k = 0; k < nc; k++) {
#pragma GCC ivdep
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
#pragma GCC ivdep
                        for (i = 0; i < blksize; i++) {
                                gtox[         i] = exps_2a[k*BLKSIZE+i] * gridx[i] * gridx[i] + exps[k*BLKSIZE+i];
                                gtox[1*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridx[i] * gridy[i];
                                gtox[2*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridx[i] * gridz[i];
                                gtoy[         i] = exps_2a[k*BLKSIZE+i] * gridy[i] * gridx[i];
                                gtoy[1*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridy[i] * gridy[i] + exps[k*BLKSIZE+i];
                                gtoy[2*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridy[i] * gridz[i];
                                gtoz[         i] = exps_2a[k*BLKSIZE+i] * gridz[i] * gridx[i];
                                gtoz[1*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridz[i] * gridy[i];
                                gtoz[2*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridz[i] * gridz[i] + exps[k*BLKSIZE+i];
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
                                        e = exps[k*BLKSIZE+i];
                                        e2a = exps_2a[k*BLKSIZE+i];
                                        rx = gridx[i];
                                        ry = gridy[i];
                                        rz = gridz[i];
                                        ax = e2a * rx;
                                        ay = e2a * ry;
                                        az = e2a * rz;
                                        ce[0] = rx * e;
                                        ce[1] = ry * e;
                                        ce[2] = rz * e;
                                        rr[0] = rx * rx; // xx
                                        rr[1] = rx * ry; // xy
                                        rr[2] = rx * rz; // xz
                                        rr[3] = ry * ry; // yy
                                        rr[4] = ry * rz; // yz
                                        rr[5] = rz * rz; // zz
                                        gtox[         i] = ax * rr[0] + 2 * ce[0];
                                        gtox[1*ngrids+i] = ax * rr[1] +     ce[1];
                                        gtox[2*ngrids+i] = ax * rr[2] +     ce[2];
                                        gtox[3*ngrids+i] = ax * rr[3];
                                        gtox[4*ngrids+i] = ax * rr[4];
                                        gtox[5*ngrids+i] = ax * rr[5];
                                        gtoy[         i] = ay * rr[0];
                                        gtoy[1*ngrids+i] = ay * rr[1] +     ce[0];
                                        gtoy[2*ngrids+i] = ay * rr[2];
                                        gtoy[3*ngrids+i] = ay * rr[3] + 2 * ce[1];
                                        gtoy[4*ngrids+i] = ay * rr[4] +     ce[2];
                                        gtoy[5*ngrids+i] = ay * rr[5];
                                        gtoz[         i] = az * rr[0];
                                        gtoz[1*ngrids+i] = az * rr[1];
                                        gtoz[2*ngrids+i] = az * rr[2] +     ce[0];
                                        gtoz[3*ngrids+i] = az * rr[3];
                                        gtoz[4*ngrids+i] = az * rr[4] +     ce[1];
                                        gtoz[5*ngrids+i] = az * rr[5] + 2 * ce[2];
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
                                        e = exps[k*BLKSIZE+i];
                                        e2a = exps_2a[k*BLKSIZE+i];
                                        rx = gridx[i];
                                        ry = gridy[i];
                                        rz = gridz[i];
                                        ax = e2a * rx;
                                        ay = e2a * ry;
                                        az = e2a * rz;
                                        ce[0] = rx * rx * e;
                                        ce[1] = rx * ry * e;
                                        ce[2] = rx * rz * e;
                                        ce[3] = ry * ry * e;
                                        ce[4] = ry * rz * e;
                                        ce[5] = rz * rz * e;
                                        rr[0] = rx * rx * rx; // xxx
                                        rr[1] = rx * rx * ry; // xxy
                                        rr[2] = rx * rx * rz; // xxz
                                        rr[3] = rx * ry * ry; // xyy
                                        rr[4] = rx * ry * rz; // xyz
                                        rr[5] = rx * rz * rz; // xzz
                                        rr[6] = ry * ry * ry; // yyy
                                        rr[7] = ry * ry * rz; // yyz
                                        rr[8] = ry * rz * rz; // yzz
                                        rr[9] = rz * rz * rz; // zzz
                                        gtox[         i] = ax * rr[0] + 3 * ce[0];
                                        gtox[1*ngrids+i] = ax * rr[1] + 2 * ce[1];
                                        gtox[2*ngrids+i] = ax * rr[2] + 2 * ce[2];
                                        gtox[3*ngrids+i] = ax * rr[3] +     ce[3];
                                        gtox[4*ngrids+i] = ax * rr[4] +     ce[4];
                                        gtox[5*ngrids+i] = ax * rr[5] +     ce[5];
                                        gtox[6*ngrids+i] = ax * rr[6];
                                        gtox[7*ngrids+i] = ax * rr[7];
                                        gtox[8*ngrids+i] = ax * rr[8];
                                        gtox[9*ngrids+i] = ax * rr[9];
                                        gtoy[         i] = ay * rr[0];
                                        gtoy[1*ngrids+i] = ay * rr[1] +     ce[0];
                                        gtoy[2*ngrids+i] = ay * rr[2];
                                        gtoy[3*ngrids+i] = ay * rr[3] + 2 * ce[1];
                                        gtoy[4*ngrids+i] = ay * rr[4] +     ce[2];
                                        gtoy[5*ngrids+i] = ay * rr[5];
                                        gtoy[6*ngrids+i] = ay * rr[6] + 3 * ce[3];
                                        gtoy[7*ngrids+i] = ay * rr[7] + 2 * ce[4];
                                        gtoy[8*ngrids+i] = ay * rr[8] +     ce[5];
                                        gtoy[9*ngrids+i] = ay * rr[9];
                                        gtoz[         i] = az * rr[0];
                                        gtoz[1*ngrids+i] = az * rr[1];
                                        gtoz[2*ngrids+i] = az * rr[2] +     ce[0];
                                        gtoz[3*ngrids+i] = az * rr[3];
                                        gtoz[4*ngrids+i] = az * rr[4] +     ce[1];
                                        gtoz[5*ngrids+i] = az * rr[5] + 2 * ce[2];
                                        gtoz[6*ngrids+i] = az * rr[6];
                                        gtoz[7*ngrids+i] = az * rr[7] +     ce[3];
                                        gtoz[8*ngrids+i] = az * rr[8] + 2 * ce[4];
                                        gtoz[9*ngrids+i] = az * rr[9] + 3 * ce[5];
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
#pragma GCC ivdep
                for (i = 0; i < blksize; i++) {
                        xpows_1less_in_power[i] = 0;
                        ypows_1less_in_power[i] = 0;
                        zpows_1less_in_power[i] = 0;
                        xpows[i] = 1;
                        ypows[i] = 1;
                        zpows[i] = 1;
                }
                for (lx = 0; lx < l; lx++) {
#pragma GCC ivdep
                for (i = 0; i < blksize; i++) {
                        xpows[(lx+1)*BLKSIZE+i] = xpows[lx*BLKSIZE+i] * gridx[i];
                        ypows[(lx+1)*BLKSIZE+i] = ypows[lx*BLKSIZE+i] * gridy[i];
                        zpows[(lx+1)*BLKSIZE+i] = zpows[lx*BLKSIZE+i] * gridz[i];
                } }
                for (k = 0; k < nc; k++) {
                        for (lx = l, n = 0; lx >= 0; lx--) {
                        for (ly = l - lx; ly >= 0; ly--, n++) {
                                lz = l - lx - ly;
#pragma GCC ivdep
                                for (i = 0; i < blksize; i++) {
                                        e = exps[k*BLKSIZE+i];
                                        e2a = exps_2a[k*BLKSIZE+i];
                                        tmp = xpows[lx*BLKSIZE+i] * ypows[ly*BLKSIZE+i] * zpows[lz*BLKSIZE+i];
                                        gtox[n*ngrids+i] = e2a * gridx[i] * tmp;
                                        gtoy[n*ngrids+i] = e2a * gridy[i] * tmp;
                                        gtoz[n*ngrids+i] = e2a * gridz[i] * tmp;
                                        gtox[n*ngrids+i] += e * lx * xpows_1less_in_power[lx*BLKSIZE+i] * ypows[ly*BLKSIZE+i] * zpows[lz*BLKSIZE+i];
                                        gtoy[n*ngrids+i] += e * ly * xpows[lx*BLKSIZE+i] * ypows_1less_in_power[ly*BLKSIZE+i] * zpows[lz*BLKSIZE+i];
                                        gtoz[n*ngrids+i] += e * lz * xpows[lx*BLKSIZE+i] * ypows[ly*BLKSIZE+i] * zpows_1less_in_power[lz*BLKSIZE+i];
                                }
                        } }
                        gtox += ngrids * degen;
                        gtoy += ngrids * degen;
                        gtoz += ngrids * degen;
                }
        }
}

void GTOval_cart(int ngrids, int *shls_slice, int *ao_loc,
                 double *ao, double *coord, uint8_t *non0table,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_cart_drv(GTOshell_eval_grid_cart, GTOcontract_exp0, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph(int ngrids, int *shls_slice, int *ao_loc,
                double *ao, double *coord, uint8_t *non0table,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor(int ngrids, int *shls_slice, int *ao_loc,
                   double complex *ao, double *coord, uint8_t *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_spinor_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOval_ip_cart(int ngrids, int *shls_slice, int *ao_loc,
                    double *ao, double *coord, uint8_t *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_cart_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_ip_sph(int ngrids, int *shls_slice, int *ao_loc,
                   double *ao, double *coord, uint8_t *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_sph_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_ip_spinor(int ngrids, int *shls_slice, int *ao_loc,
                      double complex *ao, double *coord, uint8_t *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_spinor_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

