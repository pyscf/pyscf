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
#include <math.h>
#include <complex.h>
#include "gto/grid_ao_drv.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double CINTcommon_fac_sp(int l);

/*
 * deriv 0: exp(-ar^2) x^n
 * deriv 1: exp(-ar^2)[nx^{n-1} - 2ax^{n+1}]
 * deriv 2: exp(-ar^2)[n(n-1)x^{n-2} - 2a(2n+1)x^n + 4a^2x^{n+2}]
 * deriv 3: exp(-ar^2)[n(n-1)(n-2)x^{n-3} - 2a3n^2x^{n-1} + 4a^2(3n+3)x^{n+1} - 8a^3x^{n+3}]
 * deriv 4: exp(-ar^2)[n(n-1)(n-2)(n-3)x^{n-4} - 2a(4n^3-6n^2+2)x^n{-2}
 *                     + 4a^2(6n^2+6n+3)x^n - 8a(4n+6)x^{n+2} + 16a^4x^{n+4}]
 */

void GTOshell_eval_grid_cart_deriv2(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t bgrids)
{
        const size_t degen = (l+1)*(l+2)/2;
        const size_t bgrids0 = (bgrids >= SIMDD) ? (bgrids+1-SIMDD) : 0;
        int lx, ly, lz;
        size_t i, j, j1, k, l1, n;
        double fx0[SIMDD*16];
        double fy0[SIMDD*16];
        double fz0[SIMDD*16];
        double fx1[SIMDD*16];
        double fy1[SIMDD*16];
        double fz1[SIMDD*16];
        double fx2[SIMDD*16];
        double fy2[SIMDD*16];
        double fz2[SIMDD*16];
        double buf[SIMDD*10];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *gto   = cgto;
        double *gtox  = gto + nao*ngrids;
        double *gtoy  = gto + nao*ngrids * 2;
        double *gtoz  = gto + nao*ngrids * 3;
        double *gtoxx = gto + nao*ngrids * 4;
        double *gtoxy = gto + nao*ngrids * 5;
        double *gtoxz = gto + nao*ngrids * 6;
        double *gtoyy = gto + nao*ngrids * 7;
        double *gtoyz = gto + nao*ngrids * 8;
        double *gtozz = gto + nao*ngrids * 9;
        double *pgto;
        double e;

        for (j = 0; j < 10; j++) {
                pgto = cgto + j*nao*ngrids;
                for (n = 0; n < degen*nc; n++) {
                for (i = 0; i < bgrids; i++) {
                        pgto[n*ngrids+i] = 0;
                } }
        }

        for (i = 0; i < bgrids0; i+=SIMDD) {
                for (k = 0; k < np; k++) {
                        if (_nonzero_in(exps+k*BLKSIZE+i, SIMDD)) {
        for (n = 0; n < SIMDD; n++) {
                fx0[n] = 1;
                fy0[n] = 1;
                fz0[n] = 1;
        }
        for (lx = 1; lx <= l+2; lx++) {
        for (n = 0; n < SIMDD; n++) {
                fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
                fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
                fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
        } }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+1, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l  , alpha[k]);
        for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                        e = exps[k*BLKSIZE+i+n];
                        buf[0*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[1*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[2*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[3*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[4*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[5*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[6*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[7*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[8*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[9*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < SIMDD; n++) {
                        gto  [j1*ngrids+i+n] += buf[0*SIMDD+n] * coeff[j*np+k];
                        gtox [j1*ngrids+i+n] += buf[1*SIMDD+n] * coeff[j*np+k];
                        gtoy [j1*ngrids+i+n] += buf[2*SIMDD+n] * coeff[j*np+k];
                        gtoz [j1*ngrids+i+n] += buf[3*SIMDD+n] * coeff[j*np+k];
                        gtoxx[j1*ngrids+i+n] += buf[4*SIMDD+n] * coeff[j*np+k];
                        gtoxy[j1*ngrids+i+n] += buf[5*SIMDD+n] * coeff[j*np+k];
                        gtoxz[j1*ngrids+i+n] += buf[6*SIMDD+n] * coeff[j*np+k];
                        gtoyy[j1*ngrids+i+n] += buf[7*SIMDD+n] * coeff[j*np+k];
                        gtoyz[j1*ngrids+i+n] += buf[8*SIMDD+n] * coeff[j*np+k];
                        gtozz[j1*ngrids+i+n] += buf[9*SIMDD+n] * coeff[j*np+k];
                } }
        } }
                        }
                }
        }

        if (i < bgrids) {
                for (k = 0; k < np; k++) {
                        if (_nonzero_in(exps+k*BLKSIZE+i, bgrids-i)) {
        for (n = 0; n < SIMDD; n++) {
                fx0[n] = 1;
                fy0[n] = 1;
                fz0[n] = 1;
        }
        for (lx = 1; lx <= l+2; lx++) {
        for (n = 0; n < SIMDD; n++) {
                fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
                fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
                fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
        } }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+1, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l  , alpha[k]);
        for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                        e = exps[k*BLKSIZE+i+n];
                        buf[0*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[1*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[2*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[3*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[4*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[5*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[6*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[7*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[8*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[9*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < bgrids-i; n++) {
                        gto  [j1*ngrids+i+n] += buf[0*SIMDD+n] * coeff[j*np+k];
                        gtox [j1*ngrids+i+n] += buf[1*SIMDD+n] * coeff[j*np+k];
                        gtoy [j1*ngrids+i+n] += buf[2*SIMDD+n] * coeff[j*np+k];
                        gtoz [j1*ngrids+i+n] += buf[3*SIMDD+n] * coeff[j*np+k];
                        gtoxx[j1*ngrids+i+n] += buf[4*SIMDD+n] * coeff[j*np+k];
                        gtoxy[j1*ngrids+i+n] += buf[5*SIMDD+n] * coeff[j*np+k];
                        gtoxz[j1*ngrids+i+n] += buf[6*SIMDD+n] * coeff[j*np+k];
                        gtoyy[j1*ngrids+i+n] += buf[7*SIMDD+n] * coeff[j*np+k];
                        gtoyz[j1*ngrids+i+n] += buf[8*SIMDD+n] * coeff[j*np+k];
                        gtozz[j1*ngrids+i+n] += buf[9*SIMDD+n] * coeff[j*np+k];
                } }
        } }
                        }
                }
        }
}
void GTOval_cart_deriv2(int ngrids, int *shls_slice, int *ao_loc,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv2(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor_deriv2(int ngrids, int *shls_slice, int *ao_loc,
                          double complex *ao, double *coord, char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        GTOeval_spinor_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}


void GTOshell_eval_grid_cart_deriv3(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t bgrids)
{
        const size_t degen = (l+1)*(l+2)/2;
        const size_t bgrids0 = (bgrids >= SIMDD) ? (bgrids+1-SIMDD) : 0;
        int lx, ly, lz;
        size_t i, j, j1, k, l1, n;
        double fx0[SIMDD*16];
        double fy0[SIMDD*16];
        double fz0[SIMDD*16];
        double fx1[SIMDD*16];
        double fy1[SIMDD*16];
        double fz1[SIMDD*16];
        double fx2[SIMDD*16];
        double fy2[SIMDD*16];
        double fz2[SIMDD*16];
        double fx3[SIMDD*16];
        double fy3[SIMDD*16];
        double fz3[SIMDD*16];
        double buf[SIMDD*20];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *gto    = cgto;
        double *gtox   = gto + nao*ngrids;
        double *gtoy   = gto + nao*ngrids * 2;
        double *gtoz   = gto + nao*ngrids * 3;
        double *gtoxx  = gto + nao*ngrids * 4;
        double *gtoxy  = gto + nao*ngrids * 5;
        double *gtoxz  = gto + nao*ngrids * 6;
        double *gtoyy  = gto + nao*ngrids * 7;
        double *gtoyz  = gto + nao*ngrids * 8;
        double *gtozz  = gto + nao*ngrids * 9;
        double *gtoxxx = gto + nao*ngrids * 10;
        double *gtoxxy = gto + nao*ngrids * 11;
        double *gtoxxz = gto + nao*ngrids * 12;
        double *gtoxyy = gto + nao*ngrids * 13;
        double *gtoxyz = gto + nao*ngrids * 14;
        double *gtoxzz = gto + nao*ngrids * 15;
        double *gtoyyy = gto + nao*ngrids * 16;
        double *gtoyyz = gto + nao*ngrids * 17;
        double *gtoyzz = gto + nao*ngrids * 18;
        double *gtozzz = gto + nao*ngrids * 19;
        double *pgto;
        double e;

        for (j = 0; j < 20; j++) {
                pgto = cgto + j*nao*ngrids;
                for (n = 0; n < degen*nc; n++) {
                for (i = 0; i < bgrids; i++) {
                        pgto[n*ngrids+i] = 0;
                } }
        }

        for (i = 0; i < bgrids0; i+=SIMDD) {
                for (k = 0; k < np; k++) {
                        if (_nonzero_in(exps+k*BLKSIZE+i, SIMDD)) {
        for (n = 0; n < SIMDD; n++) {
                fx0[n] = 1;
                fy0[n] = 1;
                fz0[n] = 1;
        }
        for (lx = 1; lx <= l+3; lx++) {
        for (n = 0; n < SIMDD; n++) {
                fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
                fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
                fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
        } }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+2, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l+1, alpha[k]);
        GTOnabla1(fx3, fy3, fz3, fx2, fy2, fz2, l  , alpha[k]);
        for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                        e = exps[k*BLKSIZE+i+n];
                        buf[ 0*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 1*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 2*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 3*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 4*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 5*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 6*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 7*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 8*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 9*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[10*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[11*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[12*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[13*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[14*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[15*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[16*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[17*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[18*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[19*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < SIMDD; n++) {
                        gto   [j1*ngrids+i+n] += buf[ 0*SIMDD+n] * coeff[j*np+k];
                        gtox  [j1*ngrids+i+n] += buf[ 1*SIMDD+n] * coeff[j*np+k];
                        gtoy  [j1*ngrids+i+n] += buf[ 2*SIMDD+n] * coeff[j*np+k];
                        gtoz  [j1*ngrids+i+n] += buf[ 3*SIMDD+n] * coeff[j*np+k];
                        gtoxx [j1*ngrids+i+n] += buf[ 4*SIMDD+n] * coeff[j*np+k];
                        gtoxy [j1*ngrids+i+n] += buf[ 5*SIMDD+n] * coeff[j*np+k];
                        gtoxz [j1*ngrids+i+n] += buf[ 6*SIMDD+n] * coeff[j*np+k];
                        gtoyy [j1*ngrids+i+n] += buf[ 7*SIMDD+n] * coeff[j*np+k];
                        gtoyz [j1*ngrids+i+n] += buf[ 8*SIMDD+n] * coeff[j*np+k];
                        gtozz [j1*ngrids+i+n] += buf[ 9*SIMDD+n] * coeff[j*np+k];
                        gtoxxx[j1*ngrids+i+n] += buf[10*SIMDD+n] * coeff[j*np+k];
                        gtoxxy[j1*ngrids+i+n] += buf[11*SIMDD+n] * coeff[j*np+k];
                        gtoxxz[j1*ngrids+i+n] += buf[12*SIMDD+n] * coeff[j*np+k];
                        gtoxyy[j1*ngrids+i+n] += buf[13*SIMDD+n] * coeff[j*np+k];
                        gtoxyz[j1*ngrids+i+n] += buf[14*SIMDD+n] * coeff[j*np+k];
                        gtoxzz[j1*ngrids+i+n] += buf[15*SIMDD+n] * coeff[j*np+k];
                        gtoyyy[j1*ngrids+i+n] += buf[16*SIMDD+n] * coeff[j*np+k];
                        gtoyyz[j1*ngrids+i+n] += buf[17*SIMDD+n] * coeff[j*np+k];
                        gtoyzz[j1*ngrids+i+n] += buf[18*SIMDD+n] * coeff[j*np+k];
                        gtozzz[j1*ngrids+i+n] += buf[19*SIMDD+n] * coeff[j*np+k];
                } }
        } }
                        }
                }
        }
        if (i < bgrids) {
                for (k = 0; k < np; k++) {
                        if (_nonzero_in(exps+k*BLKSIZE+i, bgrids-i)) {
        for (n = 0; n < SIMDD; n++) {
                fx0[n] = 1;
                fy0[n] = 1;
                fz0[n] = 1;
        }
        for (lx = 1; lx <= l+3; lx++) {
        for (n = 0; n < SIMDD; n++) {
                fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
                fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
                fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
        } }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+2, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l+1, alpha[k]);
        GTOnabla1(fx3, fy3, fz3, fx2, fy2, fz2, l  , alpha[k]);
        for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                        e = exps[k*BLKSIZE+i+n];
                        buf[ 0*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 1*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 2*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 3*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 4*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 5*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 6*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 7*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 8*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 9*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[10*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[11*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[12*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[13*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[14*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[15*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[16*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[17*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[18*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[19*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < bgrids-i; n++) {
                        gto   [j1*ngrids+i+n] += buf[ 0*SIMDD+n] * coeff[j*np+k];
                        gtox  [j1*ngrids+i+n] += buf[ 1*SIMDD+n] * coeff[j*np+k];
                        gtoy  [j1*ngrids+i+n] += buf[ 2*SIMDD+n] * coeff[j*np+k];
                        gtoz  [j1*ngrids+i+n] += buf[ 3*SIMDD+n] * coeff[j*np+k];
                        gtoxx [j1*ngrids+i+n] += buf[ 4*SIMDD+n] * coeff[j*np+k];
                        gtoxy [j1*ngrids+i+n] += buf[ 5*SIMDD+n] * coeff[j*np+k];
                        gtoxz [j1*ngrids+i+n] += buf[ 6*SIMDD+n] * coeff[j*np+k];
                        gtoyy [j1*ngrids+i+n] += buf[ 7*SIMDD+n] * coeff[j*np+k];
                        gtoyz [j1*ngrids+i+n] += buf[ 8*SIMDD+n] * coeff[j*np+k];
                        gtozz [j1*ngrids+i+n] += buf[ 9*SIMDD+n] * coeff[j*np+k];
                        gtoxxx[j1*ngrids+i+n] += buf[10*SIMDD+n] * coeff[j*np+k];
                        gtoxxy[j1*ngrids+i+n] += buf[11*SIMDD+n] * coeff[j*np+k];
                        gtoxxz[j1*ngrids+i+n] += buf[12*SIMDD+n] * coeff[j*np+k];
                        gtoxyy[j1*ngrids+i+n] += buf[13*SIMDD+n] * coeff[j*np+k];
                        gtoxyz[j1*ngrids+i+n] += buf[14*SIMDD+n] * coeff[j*np+k];
                        gtoxzz[j1*ngrids+i+n] += buf[15*SIMDD+n] * coeff[j*np+k];
                        gtoyyy[j1*ngrids+i+n] += buf[16*SIMDD+n] * coeff[j*np+k];
                        gtoyyz[j1*ngrids+i+n] += buf[17*SIMDD+n] * coeff[j*np+k];
                        gtoyzz[j1*ngrids+i+n] += buf[18*SIMDD+n] * coeff[j*np+k];
                        gtozzz[j1*ngrids+i+n] += buf[19*SIMDD+n] * coeff[j*np+k];
                } }
        } }
                        }
                }
        }
}
void GTOval_cart_deriv3(int ngrids, int *shls_slice, int *ao_loc,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv3(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor_deriv3(int ngrids, int *shls_slice, int *ao_loc,
                          double complex *ao, double *coord, char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        GTOeval_spinor_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOshell_eval_grid_cart_deriv4(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t bgrids)
{
        const size_t degen = (l+1)*(l+2)/2;
        const size_t bgrids0 = (bgrids >= SIMDD) ? (bgrids+1-SIMDD) : 0;
        int lx, ly, lz;
        size_t i, j, j1, k, l1, n;
        double fx0[SIMDD*16];
        double fy0[SIMDD*16];
        double fz0[SIMDD*16];
        double fx1[SIMDD*16];
        double fy1[SIMDD*16];
        double fz1[SIMDD*16];
        double fx2[SIMDD*16];
        double fy2[SIMDD*16];
        double fz2[SIMDD*16];
        double fx3[SIMDD*16];
        double fy3[SIMDD*16];
        double fz3[SIMDD*16];
        double fx4[SIMDD*16];
        double fy4[SIMDD*16];
        double fz4[SIMDD*16];
        double buf[SIMDD*35];
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *gto     = cgto;
        double *gtox    = gto + nao*ngrids;
        double *gtoy    = gto + nao*ngrids * 2;
        double *gtoz    = gto + nao*ngrids * 3;
        double *gtoxx   = gto + nao*ngrids * 4;
        double *gtoxy   = gto + nao*ngrids * 5;
        double *gtoxz   = gto + nao*ngrids * 6;
        double *gtoyy   = gto + nao*ngrids * 7;
        double *gtoyz   = gto + nao*ngrids * 8;
        double *gtozz   = gto + nao*ngrids * 9;
        double *gtoxxx  = gto + nao*ngrids * 10;
        double *gtoxxy  = gto + nao*ngrids * 11;
        double *gtoxxz  = gto + nao*ngrids * 12;
        double *gtoxyy  = gto + nao*ngrids * 13;
        double *gtoxyz  = gto + nao*ngrids * 14;
        double *gtoxzz  = gto + nao*ngrids * 15;
        double *gtoyyy  = gto + nao*ngrids * 16;
        double *gtoyyz  = gto + nao*ngrids * 17;
        double *gtoyzz  = gto + nao*ngrids * 18;
        double *gtozzz  = gto + nao*ngrids * 19;
        double *gtoxxxx = gto + nao*ngrids * 20;
        double *gtoxxxy = gto + nao*ngrids * 21;
        double *gtoxxxz = gto + nao*ngrids * 22;
        double *gtoxxyy = gto + nao*ngrids * 23;
        double *gtoxxyz = gto + nao*ngrids * 24;
        double *gtoxxzz = gto + nao*ngrids * 25;
        double *gtoxyyy = gto + nao*ngrids * 26;
        double *gtoxyyz = gto + nao*ngrids * 27;
        double *gtoxyzz = gto + nao*ngrids * 28;
        double *gtoxzzz = gto + nao*ngrids * 29;
        double *gtoyyyy = gto + nao*ngrids * 30;
        double *gtoyyyz = gto + nao*ngrids * 31;
        double *gtoyyzz = gto + nao*ngrids * 32;
        double *gtoyzzz = gto + nao*ngrids * 33;
        double *gtozzzz = gto + nao*ngrids * 34;
        double *pgto;
        double e;

        for (j = 0; j < 35; j++) {
                pgto = cgto + j*nao*ngrids;
                for (n = 0; n < degen*nc; n++) {
                for (i = 0; i < bgrids; i++) {
                        pgto[n*ngrids+i] = 0;
                } }
        }

        for (i = 0; i < bgrids0; i+=SIMDD) {
                for (k = 0; k < np; k++) {
                        if (_nonzero_in(exps+k*BLKSIZE+i, SIMDD)) {
        for (n = 0; n < SIMDD; n++) {
                fx0[n] = 1;
                fy0[n] = 1;
                fz0[n] = 1;
        }
        for (lx = 1; lx <= l+4; lx++) {
        for (n = 0; n < SIMDD; n++) {
                fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
                fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
                fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
        } }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+3, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l+2, alpha[k]);
        GTOnabla1(fx3, fy3, fz3, fx2, fy2, fz2, l+1, alpha[k]);
        GTOnabla1(fx4, fy4, fz4, fx3, fy3, fz3, l  , alpha[k]);
        for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                        e = exps[k*BLKSIZE+i+n];
                        buf[ 0*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 1*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 2*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 3*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 4*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 5*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 6*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 7*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 8*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 9*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[10*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[11*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[12*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[13*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[14*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[15*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[16*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[17*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[18*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[19*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                        buf[20*SIMDD+n] = e * fx4[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[21*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[22*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[23*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[24*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[25*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[26*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[27*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[28*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[29*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                        buf[30*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy4[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[31*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[32*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[33*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                        buf[34*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz4[lz*SIMDD+n];
                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < SIMDD; n++) {
                        gto    [j1*ngrids+i+n] += buf[ 0*SIMDD+n] * coeff[j*np+k];
                        gtox   [j1*ngrids+i+n] += buf[ 1*SIMDD+n] * coeff[j*np+k];
                        gtoy   [j1*ngrids+i+n] += buf[ 2*SIMDD+n] * coeff[j*np+k];
                        gtoz   [j1*ngrids+i+n] += buf[ 3*SIMDD+n] * coeff[j*np+k];
                        gtoxx  [j1*ngrids+i+n] += buf[ 4*SIMDD+n] * coeff[j*np+k];
                        gtoxy  [j1*ngrids+i+n] += buf[ 5*SIMDD+n] * coeff[j*np+k];
                        gtoxz  [j1*ngrids+i+n] += buf[ 6*SIMDD+n] * coeff[j*np+k];
                        gtoyy  [j1*ngrids+i+n] += buf[ 7*SIMDD+n] * coeff[j*np+k];
                        gtoyz  [j1*ngrids+i+n] += buf[ 8*SIMDD+n] * coeff[j*np+k];
                        gtozz  [j1*ngrids+i+n] += buf[ 9*SIMDD+n] * coeff[j*np+k];
                        gtoxxx [j1*ngrids+i+n] += buf[10*SIMDD+n] * coeff[j*np+k];
                        gtoxxy [j1*ngrids+i+n] += buf[11*SIMDD+n] * coeff[j*np+k];
                        gtoxxz [j1*ngrids+i+n] += buf[12*SIMDD+n] * coeff[j*np+k];
                        gtoxyy [j1*ngrids+i+n] += buf[13*SIMDD+n] * coeff[j*np+k];
                        gtoxyz [j1*ngrids+i+n] += buf[14*SIMDD+n] * coeff[j*np+k];
                        gtoxzz [j1*ngrids+i+n] += buf[15*SIMDD+n] * coeff[j*np+k];
                        gtoyyy [j1*ngrids+i+n] += buf[16*SIMDD+n] * coeff[j*np+k];
                        gtoyyz [j1*ngrids+i+n] += buf[17*SIMDD+n] * coeff[j*np+k];
                        gtoyzz [j1*ngrids+i+n] += buf[18*SIMDD+n] * coeff[j*np+k];
                        gtozzz [j1*ngrids+i+n] += buf[19*SIMDD+n] * coeff[j*np+k];
                        gtoxxxx[j1*ngrids+i+n] += buf[20*SIMDD+n] * coeff[j*np+k];
                        gtoxxxy[j1*ngrids+i+n] += buf[21*SIMDD+n] * coeff[j*np+k];
                        gtoxxxz[j1*ngrids+i+n] += buf[22*SIMDD+n] * coeff[j*np+k];
                        gtoxxyy[j1*ngrids+i+n] += buf[23*SIMDD+n] * coeff[j*np+k];
                        gtoxxyz[j1*ngrids+i+n] += buf[24*SIMDD+n] * coeff[j*np+k];
                        gtoxxzz[j1*ngrids+i+n] += buf[25*SIMDD+n] * coeff[j*np+k];
                        gtoxyyy[j1*ngrids+i+n] += buf[26*SIMDD+n] * coeff[j*np+k];
                        gtoxyyz[j1*ngrids+i+n] += buf[27*SIMDD+n] * coeff[j*np+k];
                        gtoxyzz[j1*ngrids+i+n] += buf[28*SIMDD+n] * coeff[j*np+k];
                        gtoxzzz[j1*ngrids+i+n] += buf[29*SIMDD+n] * coeff[j*np+k];
                        gtoyyyy[j1*ngrids+i+n] += buf[30*SIMDD+n] * coeff[j*np+k];
                        gtoyyyz[j1*ngrids+i+n] += buf[31*SIMDD+n] * coeff[j*np+k];
                        gtoyyzz[j1*ngrids+i+n] += buf[32*SIMDD+n] * coeff[j*np+k];
                        gtoyzzz[j1*ngrids+i+n] += buf[33*SIMDD+n] * coeff[j*np+k];
                        gtozzzz[j1*ngrids+i+n] += buf[34*SIMDD+n] * coeff[j*np+k];
                } }
        } }
                        }
                }
        }

        if (i < bgrids) {
                for (k = 0; k < np; k++) {
                        if (_nonzero_in(exps+k*BLKSIZE+i, bgrids-i)) {
        for (n = 0; n < SIMDD; n++) {
                fx0[n] = 1;
                fy0[n] = 1;
                fz0[n] = 1;
        }
        for (lx = 1; lx <= l+4; lx++) {
        for (n = 0; n < SIMDD; n++) {
                fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
                fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
                fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
        } }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+3, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l+2, alpha[k]);
        GTOnabla1(fx3, fy3, fz3, fx2, fy2, fz2, l+1, alpha[k]);
        GTOnabla1(fx4, fy4, fz4, fx3, fy3, fz3, l  , alpha[k]);
        for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                        e = exps[k*BLKSIZE+i+n];
                        buf[ 0*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 1*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 2*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 3*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 4*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 5*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 6*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 7*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[ 8*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[ 9*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[10*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[11*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[12*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[13*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[14*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[15*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[16*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[17*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[18*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[19*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                        buf[20*SIMDD+n] = e * fx4[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[21*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[22*SIMDD+n] = e * fx3[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[23*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[24*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[25*SIMDD+n] = e * fx2[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[26*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[27*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[28*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[29*SIMDD+n] = e * fx1[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                        buf[30*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy4[ly*SIMDD+n] * fz0[lz*SIMDD+n];
                        buf[31*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy3[ly*SIMDD+n] * fz1[lz*SIMDD+n];
                        buf[32*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy2[ly*SIMDD+n] * fz2[lz*SIMDD+n];
                        buf[33*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy1[ly*SIMDD+n] * fz3[lz*SIMDD+n];
                        buf[34*SIMDD+n] = e * fx0[lx*SIMDD+n] * fy0[ly*SIMDD+n] * fz4[lz*SIMDD+n];
                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < bgrids-i; n++) {
                        gto    [j1*ngrids+i+n] += buf[ 0*SIMDD+n] * coeff[j*np+k];
                        gtox   [j1*ngrids+i+n] += buf[ 1*SIMDD+n] * coeff[j*np+k];
                        gtoy   [j1*ngrids+i+n] += buf[ 2*SIMDD+n] * coeff[j*np+k];
                        gtoz   [j1*ngrids+i+n] += buf[ 3*SIMDD+n] * coeff[j*np+k];
                        gtoxx  [j1*ngrids+i+n] += buf[ 4*SIMDD+n] * coeff[j*np+k];
                        gtoxy  [j1*ngrids+i+n] += buf[ 5*SIMDD+n] * coeff[j*np+k];
                        gtoxz  [j1*ngrids+i+n] += buf[ 6*SIMDD+n] * coeff[j*np+k];
                        gtoyy  [j1*ngrids+i+n] += buf[ 7*SIMDD+n] * coeff[j*np+k];
                        gtoyz  [j1*ngrids+i+n] += buf[ 8*SIMDD+n] * coeff[j*np+k];
                        gtozz  [j1*ngrids+i+n] += buf[ 9*SIMDD+n] * coeff[j*np+k];
                        gtoxxx [j1*ngrids+i+n] += buf[10*SIMDD+n] * coeff[j*np+k];
                        gtoxxy [j1*ngrids+i+n] += buf[11*SIMDD+n] * coeff[j*np+k];
                        gtoxxz [j1*ngrids+i+n] += buf[12*SIMDD+n] * coeff[j*np+k];
                        gtoxyy [j1*ngrids+i+n] += buf[13*SIMDD+n] * coeff[j*np+k];
                        gtoxyz [j1*ngrids+i+n] += buf[14*SIMDD+n] * coeff[j*np+k];
                        gtoxzz [j1*ngrids+i+n] += buf[15*SIMDD+n] * coeff[j*np+k];
                        gtoyyy [j1*ngrids+i+n] += buf[16*SIMDD+n] * coeff[j*np+k];
                        gtoyyz [j1*ngrids+i+n] += buf[17*SIMDD+n] * coeff[j*np+k];
                        gtoyzz [j1*ngrids+i+n] += buf[18*SIMDD+n] * coeff[j*np+k];
                        gtozzz [j1*ngrids+i+n] += buf[19*SIMDD+n] * coeff[j*np+k];
                        gtoxxxx[j1*ngrids+i+n] += buf[20*SIMDD+n] * coeff[j*np+k];
                        gtoxxxy[j1*ngrids+i+n] += buf[21*SIMDD+n] * coeff[j*np+k];
                        gtoxxxz[j1*ngrids+i+n] += buf[22*SIMDD+n] * coeff[j*np+k];
                        gtoxxyy[j1*ngrids+i+n] += buf[23*SIMDD+n] * coeff[j*np+k];
                        gtoxxyz[j1*ngrids+i+n] += buf[24*SIMDD+n] * coeff[j*np+k];
                        gtoxxzz[j1*ngrids+i+n] += buf[25*SIMDD+n] * coeff[j*np+k];
                        gtoxyyy[j1*ngrids+i+n] += buf[26*SIMDD+n] * coeff[j*np+k];
                        gtoxyyz[j1*ngrids+i+n] += buf[27*SIMDD+n] * coeff[j*np+k];
                        gtoxyzz[j1*ngrids+i+n] += buf[28*SIMDD+n] * coeff[j*np+k];
                        gtoxzzz[j1*ngrids+i+n] += buf[29*SIMDD+n] * coeff[j*np+k];
                        gtoyyyy[j1*ngrids+i+n] += buf[30*SIMDD+n] * coeff[j*np+k];
                        gtoyyyz[j1*ngrids+i+n] += buf[31*SIMDD+n] * coeff[j*np+k];
                        gtoyyzz[j1*ngrids+i+n] += buf[32*SIMDD+n] * coeff[j*np+k];
                        gtoyzzz[j1*ngrids+i+n] += buf[33*SIMDD+n] * coeff[j*np+k];
                        gtozzzz[j1*ngrids+i+n] += buf[34*SIMDD+n] * coeff[j*np+k];
                } }
        } }
                        }
                }
        }
}
void GTOval_cart_deriv4(int ngrids, int *shls_slice, int *ao_loc,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv4(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor_deriv4(int ngrids, int *shls_slice, int *ao_loc,
                          double complex *ao, double *coord, char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        GTOeval_spinor_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t bgrids)
{
        const size_t degen = (l+1)*(l+2)/2;
        int lx, ly, lz;
        size_t i, k, n;
        double ax, ay, az, tmp;
        double ce[6];
        double xpows_1less_in_power[64];
        double ypows_1less_in_power[64];
        double zpows_1less_in_power[64];
        double *xpows = xpows_1less_in_power + 1;
        double *ypows = ypows_1less_in_power + 1;
        double *zpows = zpows_1less_in_power + 1;
        double *gridx = coord;
        double *gridy = coord+BLKSIZE;
        double *gridz = coord+BLKSIZE*2;
        double *gtox = gto + nao * ngrids;
        double *gtoy = gto + nao * ngrids * 2;
        double *gtoz = gto + nao * ngrids * 3;
        double *exps_2a = exps + NPRIMAX*BLKSIZE;
        switch (l) {
        case 0:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < bgrids; i++) {
                                if (NOTZERO(exps[k*BLKSIZE+i])) {
                                        gto [i] = exps[k*BLKSIZE+i];
                                        gtox[i] = exps_2a[k*BLKSIZE+i] * gridx[i];
                                        gtoy[i] = exps_2a[k*BLKSIZE+i] * gridy[i];
                                        gtoz[i] = exps_2a[k*BLKSIZE+i] * gridz[i];
                                } else {
                                        gto [i] = 0;
                                        gtox[i] = 0;
                                        gtoy[i] = 0;
                                        gtoz[i] = 0;
                                }
                        }
                        gto  += ngrids;
                        gtox += ngrids;
                        gtoy += ngrids;
                        gtoz += ngrids;
                }
                break;
        case 1:
                for (k = 0; k < nc; k++) {
                for (i = 0; i < bgrids; i++) {
                        if (NOTZERO(exps[k*BLKSIZE+i])) {
                                ax = exps_2a[k*BLKSIZE+i] * gridx[i];
                                ay = exps_2a[k*BLKSIZE+i] * gridy[i];
                                az = exps_2a[k*BLKSIZE+i] * gridz[i];
                                gto [         i] = gridx[i] * exps[k*BLKSIZE+i];
                                gto [1*ngrids+i] = gridy[i] * exps[k*BLKSIZE+i];
                                gto [2*ngrids+i] = gridz[i] * exps[k*BLKSIZE+i];
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
                                gto [         i] = 0;
                                gto [1*ngrids+i] = 0;
                                gto [2*ngrids+i] = 0;
                                gtox[         i] = 0;
                                gtox[1*ngrids+i] = 0;
                                gtox[2*ngrids+i] = 0;
                                gtoy[         i] = 0;
                                gtoy[1*ngrids+i] = 0;
                                gtoy[2*ngrids+i] = 0;
                                gtoz[         i] = 0;
                                gtoz[1*ngrids+i] = 0;
                                gtoz[2*ngrids+i] = 0;
                        } }
                        gto  += ngrids * 3;
                        gtox += ngrids * 3;
                        gtoy += ngrids * 3;
                        gtoz += ngrids * 3;
                }
                break;
        case 2:
                for (k = 0; k < nc; k++) {
                for (i = 0; i < bgrids; i++) {
                        if (NOTZERO(exps[k*BLKSIZE+i])) {
                                tmp = exps_2a[k*BLKSIZE+i]/(exps[k*BLKSIZE+i]+1e-200);
                                ax = tmp * gridx[i];
                                ay = tmp * gridy[i];
                                az = tmp * gridz[i];
                                ce[0] = gridx[i] * exps[k*BLKSIZE+i];
                                ce[1] = gridy[i] * exps[k*BLKSIZE+i];
                                ce[2] = gridz[i] * exps[k*BLKSIZE+i];
                                gto [         i] = gridx[i] * ce[0]; // xx
                                gto [1*ngrids+i] = gridx[i] * ce[1]; // xy
                                gto [2*ngrids+i] = gridx[i] * ce[2]; // xz
                                gto [3*ngrids+i] = gridy[i] * ce[1]; // yy
                                gto [4*ngrids+i] = gridy[i] * ce[2]; // yz
                                gto [5*ngrids+i] = gridz[i] * ce[2]; // zz
                                gtox[         i] = ax * gto[         i] + 2 * ce[0];
                                gtox[1*ngrids+i] = ax * gto[1*ngrids+i] +     ce[1];
                                gtox[2*ngrids+i] = ax * gto[2*ngrids+i] +     ce[2];
                                gtox[3*ngrids+i] = ax * gto[3*ngrids+i];
                                gtox[4*ngrids+i] = ax * gto[4*ngrids+i];
                                gtox[5*ngrids+i] = ax * gto[5*ngrids+i];
                                gtoy[         i] = ay * gto[         i];
                                gtoy[1*ngrids+i] = ay * gto[1*ngrids+i] +     ce[0];
                                gtoy[2*ngrids+i] = ay * gto[2*ngrids+i];
                                gtoy[3*ngrids+i] = ay * gto[3*ngrids+i] + 2 * ce[1];
                                gtoy[4*ngrids+i] = ay * gto[4*ngrids+i] +     ce[2];
                                gtoy[5*ngrids+i] = ay * gto[5*ngrids+i];
                                gtoz[         i] = az * gto[         i];
                                gtoz[1*ngrids+i] = az * gto[1*ngrids+i];
                                gtoz[2*ngrids+i] = az * gto[2*ngrids+i] +     ce[0];
                                gtoz[3*ngrids+i] = az * gto[3*ngrids+i];
                                gtoz[4*ngrids+i] = az * gto[4*ngrids+i] +     ce[1];
                                gtoz[5*ngrids+i] = az * gto[5*ngrids+i] + 2 * ce[2];
                        } else {
                                gto [         i] = 0;
                                gto [1*ngrids+i] = 0;
                                gto [2*ngrids+i] = 0;
                                gto [3*ngrids+i] = 0;
                                gto [4*ngrids+i] = 0;
                                gto [5*ngrids+i] = 0;
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
                        } }
                        gto  += ngrids * 6;
                        gtox += ngrids * 6;
                        gtoy += ngrids * 6;
                        gtoz += ngrids * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
                for (i = 0; i < bgrids; i++) {
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
                                gto [         i] = gridx[i] * ce[0]; // xxx
                                gto [1*ngrids+i] = gridx[i] * ce[1]; // xxy
                                gto [2*ngrids+i] = gridx[i] * ce[2]; // xxz
                                gto [3*ngrids+i] = gridx[i] * ce[3]; // xyy
                                gto [4*ngrids+i] = gridx[i] * ce[4]; // xyz
                                gto [5*ngrids+i] = gridx[i] * ce[5]; // xzz
                                gto [6*ngrids+i] = gridy[i] * ce[3]; // yyy
                                gto [7*ngrids+i] = gridy[i] * ce[4]; // yyz
                                gto [8*ngrids+i] = gridy[i] * ce[5]; // yzz
                                gto [9*ngrids+i] = gridz[i] * ce[5]; // zzz
                                gtox[         i] = ax * gto[         i] + 3 * ce[0];
                                gtox[1*ngrids+i] = ax * gto[1*ngrids+i] + 2 * ce[1];
                                gtox[2*ngrids+i] = ax * gto[2*ngrids+i] + 2 * ce[2];
                                gtox[3*ngrids+i] = ax * gto[3*ngrids+i] +     ce[3];
                                gtox[4*ngrids+i] = ax * gto[4*ngrids+i] +     ce[4];
                                gtox[5*ngrids+i] = ax * gto[5*ngrids+i] +     ce[5];
                                gtox[6*ngrids+i] = ax * gto[6*ngrids+i];
                                gtox[7*ngrids+i] = ax * gto[7*ngrids+i];
                                gtox[8*ngrids+i] = ax * gto[8*ngrids+i];
                                gtox[9*ngrids+i] = ax * gto[9*ngrids+i];
                                gtoy[         i] = ay * gto[         i];
                                gtoy[1*ngrids+i] = ay * gto[1*ngrids+i] +     ce[0];
                                gtoy[2*ngrids+i] = ay * gto[2*ngrids+i];
                                gtoy[3*ngrids+i] = ay * gto[3*ngrids+i] + 2 * ce[1];
                                gtoy[4*ngrids+i] = ay * gto[4*ngrids+i] +     ce[2];
                                gtoy[5*ngrids+i] = ay * gto[5*ngrids+i];
                                gtoy[6*ngrids+i] = ay * gto[6*ngrids+i] + 3 * ce[3];
                                gtoy[7*ngrids+i] = ay * gto[7*ngrids+i] + 2 * ce[4];
                                gtoy[8*ngrids+i] = ay * gto[8*ngrids+i] +     ce[5];
                                gtoy[9*ngrids+i] = ay * gto[9*ngrids+i];
                                gtoz[         i] = az * gto[         i];
                                gtoz[1*ngrids+i] = az * gto[1*ngrids+i];
                                gtoz[2*ngrids+i] = az * gto[2*ngrids+i] +     ce[0];
                                gtoz[3*ngrids+i] = az * gto[3*ngrids+i];
                                gtoz[4*ngrids+i] = az * gto[4*ngrids+i] +     ce[1];
                                gtoz[5*ngrids+i] = az * gto[5*ngrids+i] + 2 * ce[2];
                                gtoz[6*ngrids+i] = az * gto[6*ngrids+i];
                                gtoz[7*ngrids+i] = az * gto[7*ngrids+i] +     ce[3];
                                gtoz[8*ngrids+i] = az * gto[8*ngrids+i] + 2 * ce[4];
                                gtoz[9*ngrids+i] = az * gto[9*ngrids+i] + 3 * ce[5];
                        } else {
                                gto [         i] = 0;
                                gto [1*ngrids+i] = 0;
                                gto [2*ngrids+i] = 0;
                                gto [3*ngrids+i] = 0;
                                gto [4*ngrids+i] = 0;
                                gto [5*ngrids+i] = 0;
                                gto [6*ngrids+i] = 0;
                                gto [7*ngrids+i] = 0;
                                gto [8*ngrids+i] = 0;
                                gto [9*ngrids+i] = 0;
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
                        } }
                        gto  += ngrids * 10;
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
                        for (i = 0; i < bgrids; i++) {
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
                                                gto [n*ngrids+i] = exps[k*BLKSIZE+i] * tmp;
                                                gtox[n*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridx[i] * tmp;
                                                gtoy[n*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridy[i] * tmp;
                                                gtoz[n*ngrids+i] = exps_2a[k*BLKSIZE+i] * gridz[i] * tmp;
                                                gtox[n*ngrids+i] += exps[k*BLKSIZE+i] * lx * xpows[lx-1] * ypows[ly] * zpows[lz];
                                                gtoy[n*ngrids+i] += exps[k*BLKSIZE+i] * ly * xpows[lx] * ypows[ly-1] * zpows[lz];
                                                gtoz[n*ngrids+i] += exps[k*BLKSIZE+i] * lz * xpows[lx] * ypows[ly] * zpows[lz-1];
                                        } }
                                } else {
                                        for (n = 0; n < degen; n++) {
                                                gto [n*ngrids+i] = 0;
                                                gtox[n*ngrids+i] = 0;
                                                gtoy[n*ngrids+i] = 0;
                                                gtoz[n*ngrids+i] = 0;
                                        }
                                }
                        }
                        gto     += ngrids * degen;
                        gtox    += ngrids * degen;
                        gtoy    += ngrids * degen;
                        gtoz    += ngrids * degen;
                }
        }
}

void GTOval_cart(int ngrids, int *shls_slice, int *ao_loc,
                 double *ao, double *coord, char *non0table,
                 int *atm, int natm, int *bas, int nbas, double *env);
void GTOval_sph(int ngrids, int *shls_slice, int *ao_loc,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env);
void GTOval_spinor(int ngrids, int *shls_slice, int *ao_loc,
                   double complex *ao, double *coord, char *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env);
void GTOval_cart_deriv0(int ngrids, int *shls_slice, int *ao_loc,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOval_cart(ngrids, shls_slice, ao_loc,
                    ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv0(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOval_sph(ngrids, shls_slice, ao_loc,
                   ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor_deriv0(int ngrids, int *shls_slice, int *ao_loc,
                          double complex *ao, double *coord, char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOval_spinor(ngrids, shls_slice, ao_loc,
                      ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_spinor_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                          double complex *ao, double *coord, char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        GTOeval_spinor_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

