/*
 * Copyright (C) 2016-  Qiming Sun <osirpt.sun@gmail.com>
 */

#include <math.h>
#include "gto/grid_ao_drv.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double CINTcommon_fac_sp(int l);

static int _len_cart[] = {
        1, 3, 6, 10, 15, 21, 28, 36
};

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
                                    int l, int np, int nc, int blksize)
{
        const int degen = (l+1)*(l+2)/2;
        const int mblksize = blksize * degen;
        const int gtosize = np * mblksize;
        int lx, ly, lz, i, k, n;
        double fx0[16];
        double fy0[16];
        double fz0[16];
        double fx1[16];
        double fy1[16];
        double fz1[16];
        double fx2[16];
        double fy2[16];
        double fz2[16];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double gtobuf[gtosize*10];
        double *gto = gtobuf;
        double *gtox = gto + gtosize;
        double *gtoy = gto + gtosize * 2;
        double *gtoz = gto + gtosize * 3;
        double *gtoxx = gto + gtosize * 4;
        double *gtoxy = gto + gtosize * 5;
        double *gtoxz = gto + gtosize * 6;
        double *gtoyy = gto + gtosize * 7;
        double *gtoyz = gto + gtosize * 8;
        double *gtozz = gto + gtosize * 9;

        for (k = 0; k < np; k++) {
                for (i = 0; i < blksize; i++) {
                        if (NOTZERO(exps[i])) {
        fx0[0] = 1;
        fy0[0] = 1;
        fz0[0] = 1;
        for (lx = 1; lx <= l+2; lx++) {
                fx0[lx] = fx0[lx-1] * gridx[i];
                fy0[lx] = fy0[lx-1] * gridy[i];
                fz0[lx] = fz0[lx-1] * gridz[i];
        }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+1, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l  , alpha[k]);
        for (lx = l, n = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, n++) {
                lz = l - lx - ly;
                gto  [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz0[lz];
                gtox [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz1[lz];
                gtoxx[n*blksize+i] = exps[i] * fx2[lx] * fy0[ly] * fz0[lz];
                gtoxy[n*blksize+i] = exps[i] * fx1[lx] * fy1[ly] * fz0[lz];
                gtoxz[n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz1[lz];
                gtoyy[n*blksize+i] = exps[i] * fx0[lx] * fy2[ly] * fz0[lz];
                gtoyz[n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz1[lz];
                gtozz[n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz2[lz];
        } }
                        } else {
                                for (n = 0; n < degen; n++) {
                                        gto  [n*blksize+i] = 0;
                                        gtox [n*blksize+i] = 0;
                                        gtoy [n*blksize+i] = 0;
                                        gtoz [n*blksize+i] = 0;
                                        gtoxx[n*blksize+i] = 0;
                                        gtoxy[n*blksize+i] = 0;
                                        gtoxz[n*blksize+i] = 0;
                                        gtoyy[n*blksize+i] = 0;
                                        gtoyz[n*blksize+i] = 0;
                                        gtozz[n*blksize+i] = 0;
                                }
                        }
                }
                exps  += blksize;
                gto   += mblksize;
                gtox  += mblksize;
                gtoy  += mblksize;
                gtoz  += mblksize;
                gtoxx += mblksize;
                gtoxy += mblksize;
                gtoxz += mblksize;
                gtoyy += mblksize;
                gtoyz += mblksize;
                gtozz += mblksize;
        }

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        for (k = 0; k < 10; k++) {
                dgemm_(&TRANS_N, &TRANS_N, &mblksize, &nc, &np,
                       &D1, gtobuf+gtosize*k, &mblksize, coeff, &np,
                       &D0, cgto+nc*mblksize*k, &mblksize);
        }
}
void GTOval_cart_deriv2(int *shls_slice, int *ao_loc, int ngrids,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                         param, shls_slice, ao_loc, ngrids,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv2(int *shls_slice, int *ao_loc, int ngrids,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                        param, shls_slice, ao_loc, ngrids,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}


void GTOshell_eval_grid_cart_deriv3(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize)
{
        const int degen = (l+1)*(l+2)/2;
        const int mblksize = blksize * degen;
        const int gtosize = np * mblksize;
        int lx, ly, lz, i, k, n;
        double fx0[16];
        double fy0[16];
        double fz0[16];
        double fx1[16];
        double fy1[16];
        double fz1[16];
        double fx2[16];
        double fy2[16];
        double fz2[16];
        double fx3[16];
        double fy3[16];
        double fz3[16];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double gtobuf[gtosize*20];
        double *gto = gtobuf;
        double *gtox = gto + gtosize;
        double *gtoy = gto + gtosize * 2;
        double *gtoz = gto + gtosize * 3;
        double *gtoxx = gto + gtosize * 4;
        double *gtoxy = gto + gtosize * 5;
        double *gtoxz = gto + gtosize * 6;
        double *gtoyy = gto + gtosize * 7;
        double *gtoyz = gto + gtosize * 8;
        double *gtozz = gto + gtosize * 9;
        double *gtoxxx = gto + gtosize * 10;
        double *gtoxxy = gto + gtosize * 11;
        double *gtoxxz = gto + gtosize * 12;
        double *gtoxyy = gto + gtosize * 13;
        double *gtoxyz = gto + gtosize * 14;
        double *gtoxzz = gto + gtosize * 15;
        double *gtoyyy = gto + gtosize * 16;
        double *gtoyyz = gto + gtosize * 17;
        double *gtoyzz = gto + gtosize * 18;
        double *gtozzz = gto + gtosize * 19;

        for (k = 0; k < np; k++) {
                for (i = 0; i < blksize; i++) {
                        if (NOTZERO(exps[i])) {
        fx0[0] = 1;
        fy0[0] = 1;
        fz0[0] = 1;
        for (lx = 1; lx <= l+3; lx++) {
                fx0[lx] = fx0[lx-1] * gridx[i];
                fy0[lx] = fy0[lx-1] * gridy[i];
                fz0[lx] = fz0[lx-1] * gridz[i];
        }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+2, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l+1, alpha[k]);
        GTOnabla1(fx3, fy3, fz3, fx2, fy2, fz2, l  , alpha[k]);
        for (lx = l, n = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, n++) {
                lz = l - lx - ly;
                gto   [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz0[lz];
                gtox  [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy  [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz  [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz1[lz];
                gtoxx [n*blksize+i] = exps[i] * fx2[lx] * fy0[ly] * fz0[lz];
                gtoxy [n*blksize+i] = exps[i] * fx1[lx] * fy1[ly] * fz0[lz];
                gtoxz [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz1[lz];
                gtoyy [n*blksize+i] = exps[i] * fx0[lx] * fy2[ly] * fz0[lz];
                gtoyz [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz1[lz];
                gtozz [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz2[lz];
                gtoxxx[n*blksize+i] = exps[i] * fx3[lx] * fy0[ly] * fz0[lz];
                gtoxxy[n*blksize+i] = exps[i] * fx2[lx] * fy1[ly] * fz0[lz];
                gtoxxz[n*blksize+i] = exps[i] * fx2[lx] * fy0[ly] * fz1[lz];
                gtoxyy[n*blksize+i] = exps[i] * fx1[lx] * fy2[ly] * fz0[lz];
                gtoxyz[n*blksize+i] = exps[i] * fx1[lx] * fy1[ly] * fz1[lz];
                gtoxzz[n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz2[lz];
                gtoyyy[n*blksize+i] = exps[i] * fx0[lx] * fy3[ly] * fz0[lz];
                gtoyyz[n*blksize+i] = exps[i] * fx0[lx] * fy2[ly] * fz1[lz];
                gtoyzz[n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz2[lz];
                gtozzz[n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz3[lz];
        } }
                        } else {
                                for (n = 0; n < degen; n++) {
                                        gto   [n*blksize+i] = 0;
                                        gtox  [n*blksize+i] = 0;
                                        gtoy  [n*blksize+i] = 0;
                                        gtoz  [n*blksize+i] = 0;
                                        gtoxx [n*blksize+i] = 0;
                                        gtoxy [n*blksize+i] = 0;
                                        gtoxz [n*blksize+i] = 0;
                                        gtoyy [n*blksize+i] = 0;
                                        gtoyz [n*blksize+i] = 0;
                                        gtozz [n*blksize+i] = 0;
                                        gtoxxx[n*blksize+i] = 0;
                                        gtoxxy[n*blksize+i] = 0;
                                        gtoxxz[n*blksize+i] = 0;
                                        gtoxyy[n*blksize+i] = 0;
                                        gtoxyz[n*blksize+i] = 0;
                                        gtoxzz[n*blksize+i] = 0;
                                        gtoyyy[n*blksize+i] = 0;
                                        gtoyyz[n*blksize+i] = 0;
                                        gtoyzz[n*blksize+i] = 0;
                                        gtozzz[n*blksize+i] = 0;
                                }
                        }
                }
                exps   += blksize;
                gto    += mblksize;
                gtox   += mblksize;
                gtoy   += mblksize;
                gtoz   += mblksize;
                gtoxx  += mblksize;
                gtoxy  += mblksize;
                gtoxz  += mblksize;
                gtoyy  += mblksize;
                gtoyz  += mblksize;
                gtozz  += mblksize;
                gtoxxx += mblksize;
                gtoxxy += mblksize;
                gtoxxz += mblksize;
                gtoxyy += mblksize;
                gtoxyz += mblksize;
                gtoxzz += mblksize;
                gtoyyy += mblksize;
                gtoyyz += mblksize;
                gtoyzz += mblksize;
                gtozzz += mblksize;
        }

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        for (k = 0; k < 20; k++) {
                dgemm_(&TRANS_N, &TRANS_N, &mblksize, &nc, &np,
                       &D1, gtobuf+gtosize*k, &mblksize, coeff, &np,
                       &D0, cgto+nc*mblksize*k, &mblksize);
        }
}
void GTOval_cart_deriv3(int *shls_slice, int *ao_loc, int ngrids,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                         param, shls_slice, ao_loc, ngrids,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv3(int *shls_slice, int *ao_loc, int ngrids,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                        param, shls_slice, ao_loc, ngrids,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOshell_eval_grid_cart_deriv4(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize)
{
        const int degen = (l+1)*(l+2)/2;
        const int mblksize = blksize * degen;
        const int gtosize = np * mblksize;
        int lx, ly, lz, i, k, n;
        double fx0[16];
        double fy0[16];
        double fz0[16];
        double fx1[16];
        double fy1[16];
        double fz1[16];
        double fx2[16];
        double fy2[16];
        double fz2[16];
        double fx3[16];
        double fy3[16];
        double fz3[16];
        double fx4[16];
        double fy4[16];
        double fz4[16];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double gtobuf[gtosize*35];
        double *gto = gtobuf;
        double *gtox = gto + gtosize;
        double *gtoy = gto + gtosize * 2;
        double *gtoz = gto + gtosize * 3;
        double *gtoxx = gto + gtosize * 4;
        double *gtoxy = gto + gtosize * 5;
        double *gtoxz = gto + gtosize * 6;
        double *gtoyy = gto + gtosize * 7;
        double *gtoyz = gto + gtosize * 8;
        double *gtozz = gto + gtosize * 9;
        double *gtoxxx = gto + gtosize * 10;
        double *gtoxxy = gto + gtosize * 11;
        double *gtoxxz = gto + gtosize * 12;
        double *gtoxyy = gto + gtosize * 13;
        double *gtoxyz = gto + gtosize * 14;
        double *gtoxzz = gto + gtosize * 15;
        double *gtoyyy = gto + gtosize * 16;
        double *gtoyyz = gto + gtosize * 17;
        double *gtoyzz = gto + gtosize * 18;
        double *gtozzz = gto + gtosize * 19;
        double *gtoxxxx = gto + gtosize * 20;
        double *gtoxxxy = gto + gtosize * 21;
        double *gtoxxxz = gto + gtosize * 22;
        double *gtoxxyy = gto + gtosize * 23;
        double *gtoxxyz = gto + gtosize * 24;
        double *gtoxxzz = gto + gtosize * 25;
        double *gtoxyyy = gto + gtosize * 26;
        double *gtoxyyz = gto + gtosize * 27;
        double *gtoxyzz = gto + gtosize * 28;
        double *gtoxzzz = gto + gtosize * 29;
        double *gtoyyyy = gto + gtosize * 30;
        double *gtoyyyz = gto + gtosize * 31;
        double *gtoyyzz = gto + gtosize * 32;
        double *gtoyzzz = gto + gtosize * 33;
        double *gtozzzz = gto + gtosize * 34;

        for (k = 0; k < np; k++) {
                for (i = 0; i < blksize; i++) {
                        if (NOTZERO(exps[i])) {
        fx0[0] = 1;
        fy0[0] = 1;
        fz0[0] = 1;
        for (lx = 1; lx <= l+4; lx++) {
                fx0[lx] = fx0[lx-1] * gridx[i];
                fy0[lx] = fy0[lx-1] * gridy[i];
                fz0[lx] = fz0[lx-1] * gridz[i];
        }
        GTOnabla1(fx1, fy1, fz1, fx0, fy0, fz0, l+3, alpha[k]);
        GTOnabla1(fx2, fy2, fz2, fx1, fy1, fz1, l+2, alpha[k]);
        GTOnabla1(fx3, fy3, fz3, fx2, fy2, fz2, l+1, alpha[k]);
        GTOnabla1(fx4, fy4, fz4, fx3, fy3, fz3, l  , alpha[k]);
        for (lx = l, n = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, n++) {
                lz = l - lx - ly;
                gto    [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz0[lz];
                gtox   [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy   [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz   [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz1[lz];
                gtoxx  [n*blksize+i] = exps[i] * fx2[lx] * fy0[ly] * fz0[lz];
                gtoxy  [n*blksize+i] = exps[i] * fx1[lx] * fy1[ly] * fz0[lz];
                gtoxz  [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz1[lz];
                gtoyy  [n*blksize+i] = exps[i] * fx0[lx] * fy2[ly] * fz0[lz];
                gtoyz  [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz1[lz];
                gtozz  [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz2[lz];
                gtoxxx [n*blksize+i] = exps[i] * fx3[lx] * fy0[ly] * fz0[lz];
                gtoxxy [n*blksize+i] = exps[i] * fx2[lx] * fy1[ly] * fz0[lz];
                gtoxxz [n*blksize+i] = exps[i] * fx2[lx] * fy0[ly] * fz1[lz];
                gtoxyy [n*blksize+i] = exps[i] * fx1[lx] * fy2[ly] * fz0[lz];
                gtoxyz [n*blksize+i] = exps[i] * fx1[lx] * fy1[ly] * fz1[lz];
                gtoxzz [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz2[lz];
                gtoyyy [n*blksize+i] = exps[i] * fx0[lx] * fy3[ly] * fz0[lz];
                gtoyyz [n*blksize+i] = exps[i] * fx0[lx] * fy2[ly] * fz1[lz];
                gtoyzz [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz2[lz];
                gtozzz [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz3[lz];
                gtoxxxx[n*blksize+i] = exps[i] * fx4[lx] * fy0[ly] * fz0[lz];
                gtoxxxy[n*blksize+i] = exps[i] * fx3[lx] * fy1[ly] * fz0[lz];
                gtoxxxz[n*blksize+i] = exps[i] * fx3[lx] * fy0[ly] * fz1[lz];
                gtoxxyy[n*blksize+i] = exps[i] * fx2[lx] * fy2[ly] * fz0[lz];
                gtoxxyz[n*blksize+i] = exps[i] * fx2[lx] * fy1[ly] * fz1[lz];
                gtoxxzz[n*blksize+i] = exps[i] * fx2[lx] * fy0[ly] * fz2[lz];
                gtoxyyy[n*blksize+i] = exps[i] * fx1[lx] * fy3[ly] * fz0[lz];
                gtoxyyz[n*blksize+i] = exps[i] * fx1[lx] * fy2[ly] * fz1[lz];
                gtoxyzz[n*blksize+i] = exps[i] * fx1[lx] * fy1[ly] * fz2[lz];
                gtoxzzz[n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz3[lz];
                gtoyyyy[n*blksize+i] = exps[i] * fx0[lx] * fy4[ly] * fz0[lz];
                gtoyyyz[n*blksize+i] = exps[i] * fx0[lx] * fy3[ly] * fz1[lz];
                gtoyyzz[n*blksize+i] = exps[i] * fx0[lx] * fy2[ly] * fz2[lz];
                gtoyzzz[n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz3[lz];
                gtozzzz[n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz4[lz];
        } }
                        } else {
                                for (n = 0; n < degen; n++) {
                                        gto    [n*blksize+i] = 0;
                                        gtox   [n*blksize+i] = 0;
                                        gtoy   [n*blksize+i] = 0;
                                        gtoz   [n*blksize+i] = 0;
                                        gtoxx  [n*blksize+i] = 0;
                                        gtoxy  [n*blksize+i] = 0;
                                        gtoxz  [n*blksize+i] = 0;
                                        gtoyy  [n*blksize+i] = 0;
                                        gtoyz  [n*blksize+i] = 0;
                                        gtozz  [n*blksize+i] = 0;
                                        gtoxxx [n*blksize+i] = 0;
                                        gtoxxy [n*blksize+i] = 0;
                                        gtoxxz [n*blksize+i] = 0;
                                        gtoxyy [n*blksize+i] = 0;
                                        gtoxyz [n*blksize+i] = 0;
                                        gtoxzz [n*blksize+i] = 0;
                                        gtoyyy [n*blksize+i] = 0;
                                        gtoyyz [n*blksize+i] = 0;
                                        gtoyzz [n*blksize+i] = 0;
                                        gtozzz [n*blksize+i] = 0;
                                        gtoxxxx[n*blksize+i] = 0;
                                        gtoxxxy[n*blksize+i] = 0;
                                        gtoxxxz[n*blksize+i] = 0;
                                        gtoxxyy[n*blksize+i] = 0;
                                        gtoxxyz[n*blksize+i] = 0;
                                        gtoxxzz[n*blksize+i] = 0;
                                        gtoxyyy[n*blksize+i] = 0;
                                        gtoxyyz[n*blksize+i] = 0;
                                        gtoxyzz[n*blksize+i] = 0;
                                        gtoxzzz[n*blksize+i] = 0;
                                        gtoyyyy[n*blksize+i] = 0;
                                        gtoyyyz[n*blksize+i] = 0;
                                        gtoyyzz[n*blksize+i] = 0;
                                        gtoyzzz[n*blksize+i] = 0;
                                        gtozzzz[n*blksize+i] = 0;
                                }
                        }
                }
                exps    += blksize;
                gto     += mblksize;
                gtox    += mblksize;
                gtoy    += mblksize;
                gtoz    += mblksize;
                gtoxx   += mblksize;
                gtoxy   += mblksize;
                gtoxz   += mblksize;
                gtoyy   += mblksize;
                gtoyz   += mblksize;
                gtozz   += mblksize;
                gtoxxx  += mblksize;
                gtoxxy  += mblksize;
                gtoxxz  += mblksize;
                gtoxyy  += mblksize;
                gtoxyz  += mblksize;
                gtoxzz  += mblksize;
                gtoyyy  += mblksize;
                gtoyyz  += mblksize;
                gtoyzz  += mblksize;
                gtozzz  += mblksize;
                gtoxxxx += mblksize;
                gtoxxxy += mblksize;
                gtoxxxz += mblksize;
                gtoxxyy += mblksize;
                gtoxxyz += mblksize;
                gtoxxzz += mblksize;
                gtoxyyy += mblksize;
                gtoxyyz += mblksize;
                gtoxyzz += mblksize;
                gtoxzzz += mblksize;
                gtoyyyy += mblksize;
                gtoyyyz += mblksize;
                gtoyyzz += mblksize;
                gtoyzzz += mblksize;
                gtozzzz += mblksize;
        }

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        for (k = 0; k < 35; k++) {
                dgemm_(&TRANS_N, &TRANS_N, &mblksize, &nc, &np,
                       &D1, gtobuf+gtosize*k, &mblksize, coeff, &np,
                       &D0, cgto+nc*mblksize*k, &mblksize);
        }
}
void GTOval_cart_deriv4(int *shls_slice, int *ao_loc, int ngrids,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                         param, shls_slice, ao_loc, ngrids,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv4(int *shls_slice, int *ao_loc, int ngrids,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                        param, shls_slice, ao_loc, ngrids,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}


/*
static int contract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
                         int l, int nprim, int nctr, int blksize, double fac)
{
        int i, j;
        double arr, maxc;
        double eprim[nprim*blksize];
        double logcoeff[nprim];
        double rr[blksize];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double *peprim = eprim;
        int not0 = 0;

        // the maximum value of the coefficients for each pGTO
        for (j = 0; j < nprim; j++) {
                maxc = 0;
                for (i = 0; i < nctr; i++) {
                        maxc = MAX(maxc, fabs(coeff[i*nprim+j]));
                }
                logcoeff[j] = log(maxc);
        }

        for (i = 0; i < blksize; i++) {
                rr[i] = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
        }

        for (i = 0; i < blksize; i++) {
                for (j = 0; j < nprim; j++) {
                        arr = alpha[j] * rr[i];
                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                peprim[j] = exp(-arr) * fac;
                                not0 = 1;
                        } else {
                                peprim[j] = 0;
                        }
                }
                peprim += nprim;
        }

        return not0;
}
*/

// pre-contracted grid AO evaluator
// contracted factors = \sum c_{i} exp(-a_i*r_i**2)
/*
static void grid_cart_gto0(double *gto, double *ri, double *exps,
                           double *coord, double *alpha, double *coeff,
                           int l, int np, int nc, int blksize)
{
        int lx, ly, lz, i, k;
        double ce[3];
        double xpows[8*blksize];
        double ypows[8*blksize];
        double zpows[8*blksize];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double        *gto1, *gto2, *gto3, *gto4,
               *gto5, *gto6, *gto7, *gto8, *gto9;

        switch (l) {
        case 0:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                gto[i] = exps[i];
                        }
                        exps += blksize;
                        gto += blksize;
                }
                break;
        case 1:
                for (k = 0; k < nc; k++) {
                        gto1 = gto + blksize*1;
                        gto2 = gto + blksize*2;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        gto [i] = gridx[i] * exps[i];
                                        gto1[i] = gridy[i] * exps[i];
                                        gto2[i] = gridz[i] * exps[i];
                                } else {
                                        gto [i] = 0;
                                        gto1[i] = 0;
                                        gto2[i] = 0;
                                }
                        }
                        exps += blksize;
                        gto += blksize * 3;
                }
                break;
        case 2:
                for (k = 0; k < nc; k++) {
                        gto1 = gto + blksize*1;
                        gto2 = gto + blksize*2;
                        gto3 = gto + blksize*3;
                        gto4 = gto + blksize*4;
                        gto5 = gto + blksize*5;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        ce[0] = gridx[i] * exps[i];
                                        ce[1] = gridy[i] * exps[i];
                                        ce[2] = gridz[i] * exps[i];
                                        gto [i] = ce[0] * gridx[i]; // xx
                                        gto1[i] = ce[0] * gridy[i]; // xy
                                        gto2[i] = ce[0] * gridz[i]; // xz
                                        gto3[i] = ce[1] * gridy[i]; // yy
                                        gto4[i] = ce[1] * gridz[i]; // yz
                                        gto5[i] = ce[2] * gridz[i]; // zz
                                } else {
                                        gto [i] = 0;
                                        gto1[i] = 0;
                                        gto2[i] = 0;
                                        gto3[i] = 0;
                                        gto4[i] = 0;
                                        gto5[i] = 0;
                                }
                        }
                        exps += blksize;
                        gto += blksize * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
                        gto1 = gto + blksize*1;
                        gto2 = gto + blksize*2;
                        gto3 = gto + blksize*3;
                        gto4 = gto + blksize*4;
                        gto5 = gto + blksize*5;
                        gto6 = gto + blksize*6;
                        gto7 = gto + blksize*7;
                        gto8 = gto + blksize*8;
                        gto9 = gto + blksize*9;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        ce[0] = gridx[i] * gridx[i] * exps[i];
                                        ce[1] = gridy[i] * gridy[i] * exps[i];
                                        ce[2] = gridz[i] * gridz[i] * exps[i];
                                        gto [i] = ce[0] * gridx[i]; // xxx
                                        gto1[i] = ce[0] * gridy[i]; // xxy
                                        gto2[i] = ce[0] * gridz[i]; // xxz
                                        gto3[i] = gridx[i] * ce[1]; // xyy
                                        gto4[i] = gridx[i]*gridy[i]*gridz[i] * exps[i]; // xyz
                                        gto5[i] = gridx[i] * ce[2]; // xzz
                                        gto6[i] = ce[1] * gridy[i]; // yyy
                                        gto7[i] = ce[1] * gridz[i]; // yyz
                                        gto8[i] = gridy[i] * ce[2]; // yzz
                                        gto9[i] = gridz[i] * ce[2]; // zzz
                                } else {
                                        gto [i] = 0;
                                        gto1[i] = 0;
                                        gto2[i] = 0;
                                        gto3[i] = 0;
                                        gto4[i] = 0;
                                        gto5[i] = 0;
                                        gto6[i] = 0;
                                        gto7[i] = 0;
                                        gto8[i] = 0;
                                        gto9[i] = 0;
                                }
                        }
                        exps += blksize;
                        gto += blksize * 10;
                }
                break;
        default:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        xpows[i*8+0] = 1;
                                        ypows[i*8+0] = 1;
                                        zpows[i*8+0] = 1;
                                        for (lx = 1; lx < l+1; lx++) {
                                                xpows[i*8+lx] = xpows[i*8+lx-1] * gridx[i];
                                                ypows[i*8+lx] = ypows[i*8+lx-1] * gridy[i];
                                                zpows[i*8+lx] = zpows[i*8+lx-1] * gridz[i];
                                        }
                                }
                        }
                        for (lx = l; lx >= 0; lx--) {
                        for (ly = l - lx; ly >= 0; ly--) {
                                lz = l - lx - ly;
                                for (i = 0; i < blksize; i++) {
                                        if (NOTZERO(exps[i])) {
                                                gto[i] = xpows[i*8+lx]
                                                       * ypows[i*8+ly]
                                                       * zpows[i*8+lz]*exps[i];
                                        } else {
                                                gto[i] = 0;
                                        }
                                }
                                gto += blksize;
                        } }
                        exps += blksize;
                }
        }
}
*/

void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
        const int gtosize = nc*degen*blksize;
        int lx, ly, lz, i, k, n;
        double ax, ay, az, tmp;
        double ce[6];
        double xpows_1less_in_power[64];
        double ypows_1less_in_power[64];
        double zpows_1less_in_power[64];
        double *xpows = xpows_1less_in_power + 1;
        double *ypows = ypows_1less_in_power + 1;
        double *zpows = zpows_1less_in_power + 1;
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double *gtox = gto + gtosize;
        double *gtoy = gto + gtosize * 2;
        double *gtoz = gto + gtosize * 3;
        double *exps_2a = exps + NPRIMAX*blksize;
        double        *gto1, *gto2, *gto3, *gto4,
               *gto5, *gto6, *gto7, *gto8, *gto9;
        double         *gtox1, *gtox2, *gtox3, *gtox4,
               *gtox5, *gtox6, *gtox7, *gtox8, *gtox9;
        double         *gtoy1, *gtoy2, *gtoy3, *gtoy4,
               *gtoy5, *gtoy6, *gtoy7, *gtoy8, *gtoy9;
        double         *gtoz1, *gtoz2, *gtoz3, *gtoz4,
               *gtoz5, *gtoz6, *gtoz7, *gtoz8, *gtoz9;
        switch (l) {
        case 0:
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        gto [i] = exps[i];
                                        gtox[i] = exps_2a[i] * gridx[i];
                                        gtoy[i] = exps_2a[i] * gridy[i];
                                        gtoz[i] = exps_2a[i] * gridz[i];
                                } else {
                                        gto [i] = 0;
                                        gtox[i] = 0;
                                        gtoy[i] = 0;
                                        gtoz[i] = 0;
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gto  += blksize;
                        gtox += blksize;
                        gtoy += blksize;
                        gtoz += blksize;
                }
                break;
        case 1:
                for (k = 0; k < nc; k++) {
                        gto1  = gto  + blksize*1;
                        gtox1 = gtox + blksize*1;
                        gtoy1 = gtoy + blksize*1;
                        gtoz1 = gtoz + blksize*1;
                        gto2  = gto  + blksize*2;
                        gtox2 = gtox + blksize*2;
                        gtoy2 = gtoy + blksize*2;
                        gtoz2 = gtoz + blksize*2;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        ax = exps_2a[i] * gridx[i];
                                        ay = exps_2a[i] * gridy[i];
                                        az = exps_2a[i] * gridz[i];
                                        gto  [i] = gridx[i] * exps[i];
                                        gto1 [i] = gridy[i] * exps[i];
                                        gto2 [i] = gridz[i] * exps[i];
                                        gtox [i] = ax * gridx[i] + exps[i];
                                        gtox1[i] = ax * gridy[i];
                                        gtox2[i] = ax * gridz[i];
                                        gtoy [i] = ay * gridx[i];
                                        gtoy1[i] = ay * gridy[i] + exps[i];
                                        gtoy2[i] = ay * gridz[i];
                                        gtoz [i] = az * gridx[i];
                                        gtoz1[i] = az * gridy[i];
                                        gtoz2[i] = az * gridz[i] + exps[i];
                                } else {
                                        gto  [i] = 0;
                                        gto1 [i] = 0;
                                        gto2 [i] = 0;
                                        gtox [i] = 0;
                                        gtox1[i] = 0;
                                        gtox2[i] = 0;
                                        gtoy [i] = 0;
                                        gtoy1[i] = 0;
                                        gtoy2[i] = 0;
                                        gtoz [i] = 0;
                                        gtoz1[i] = 0;
                                        gtoz2[i] = 0;
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gto  += blksize * 3;
                        gtox += blksize * 3;
                        gtoy += blksize * 3;
                        gtoz += blksize * 3;
                }
                break;
        case 2:
                for (k = 0; k < nc; k++) {
                        gto1  = gto  + blksize*1;
                        gtox1 = gtox + blksize*1;
                        gtoy1 = gtoy + blksize*1;
                        gtoz1 = gtoz + blksize*1;
                        gto2  = gto  + blksize*2;
                        gtox2 = gtox + blksize*2;
                        gtoy2 = gtoy + blksize*2;
                        gtoz2 = gtoz + blksize*2;
                        gto3  = gto  + blksize*3;
                        gtox3 = gtox + blksize*3;
                        gtoy3 = gtoy + blksize*3;
                        gtoz3 = gtoz + blksize*3;
                        gto4  = gto  + blksize*4;
                        gtox4 = gtox + blksize*4;
                        gtoy4 = gtoy + blksize*4;
                        gtoz4 = gtoz + blksize*4;
                        gto5  = gto  + blksize*5;
                        gtox5 = gtox + blksize*5;
                        gtoy5 = gtoy + blksize*5;
                        gtoz5 = gtoz + blksize*5;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        tmp = exps_2a[i]/(exps[i]+1e-200);
                                        ax = tmp * gridx[i];
                                        ay = tmp * gridy[i];
                                        az = tmp * gridz[i];
                                        ce[0] = gridx[i] * exps[i];
                                        ce[1] = gridy[i] * exps[i];
                                        ce[2] = gridz[i] * exps[i];
                                        gto  [i] = gridx[i] * ce[0]; // xx
                                        gto1 [i] = gridx[i] * ce[1]; // xy
                                        gto2 [i] = gridx[i] * ce[2]; // xz
                                        gto3 [i] = gridy[i] * ce[1]; // yy
                                        gto4 [i] = gridy[i] * ce[2]; // yz
                                        gto5 [i] = gridz[i] * ce[2]; // zz
                                        gtox [i] = ax * gto [i] + 2 * ce[0];
                                        gtox1[i] = ax * gto1[i] +     ce[1];
                                        gtox2[i] = ax * gto2[i] +     ce[2];
                                        gtox3[i] = ax * gto3[i];
                                        gtox4[i] = ax * gto4[i];
                                        gtox5[i] = ax * gto5[i];
                                        gtoy [i] = ay * gto [i];
                                        gtoy1[i] = ay * gto1[i] +     ce[0];
                                        gtoy2[i] = ay * gto2[i];
                                        gtoy3[i] = ay * gto3[i] + 2 * ce[1];
                                        gtoy4[i] = ay * gto4[i] +     ce[2];
                                        gtoy5[i] = ay * gto5[i];
                                        gtoz [i] = az * gto [i];
                                        gtoz1[i] = az * gto1[i];
                                        gtoz2[i] = az * gto2[i] +     ce[0];
                                        gtoz3[i] = az * gto3[i];
                                        gtoz4[i] = az * gto4[i] +     ce[1];
                                        gtoz5[i] = az * gto5[i] + 2 * ce[2];
                                } else {
                                        gto  [i] = 0;
                                        gto1 [i] = 0;
                                        gto2 [i] = 0;
                                        gto3 [i] = 0;
                                        gto4 [i] = 0;
                                        gto5 [i] = 0;
                                        gtox [i] = 0;
                                        gtox1[i] = 0;
                                        gtox2[i] = 0;
                                        gtox3[i] = 0;
                                        gtox4[i] = 0;
                                        gtox5[i] = 0;
                                        gtoy [i] = 0;
                                        gtoy1[i] = 0;
                                        gtoy2[i] = 0;
                                        gtoy3[i] = 0;
                                        gtoy4[i] = 0;
                                        gtoy5[i] = 0;
                                        gtoz [i] = 0;
                                        gtoz1[i] = 0;
                                        gtoz2[i] = 0;
                                        gtoz3[i] = 0;
                                        gtoz4[i] = 0;
                                        gtoz5[i] = 0;
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gto  += blksize * 6;
                        gtox += blksize * 6;
                        gtoy += blksize * 6;
                        gtoz += blksize * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
                        gto1  = gto  + blksize*1;
                        gtox1 = gtox + blksize*1;
                        gtoy1 = gtoy + blksize*1;
                        gtoz1 = gtoz + blksize*1;
                        gto2  = gto  + blksize*2;
                        gtox2 = gtox + blksize*2;
                        gtoy2 = gtoy + blksize*2;
                        gtoz2 = gtoz + blksize*2;
                        gto3  = gto  + blksize*3;
                        gtox3 = gtox + blksize*3;
                        gtoy3 = gtoy + blksize*3;
                        gtoz3 = gtoz + blksize*3;
                        gto4  = gto  + blksize*4;
                        gtox4 = gtox + blksize*4;
                        gtoy4 = gtoy + blksize*4;
                        gtoz4 = gtoz + blksize*4;
                        gto5  = gto  + blksize*5;
                        gtox5 = gtox + blksize*5;
                        gtoy5 = gtoy + blksize*5;
                        gtoz5 = gtoz + blksize*5;
                        gto6  = gto  + blksize*6;
                        gtox6 = gtox + blksize*6;
                        gtoy6 = gtoy + blksize*6;
                        gtoz6 = gtoz + blksize*6;
                        gto7  = gto  + blksize*7;
                        gtox7 = gtox + blksize*7;
                        gtoy7 = gtoy + blksize*7;
                        gtoz7 = gtoz + blksize*7;
                        gto8  = gto  + blksize*8;
                        gtox8 = gtox + blksize*8;
                        gtoy8 = gtoy + blksize*8;
                        gtoz8 = gtoz + blksize*8;
                        gto9  = gto  + blksize*9;
                        gtox9 = gtox + blksize*9;
                        gtoy9 = gtoy + blksize*9;
                        gtoz9 = gtoz + blksize*9;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        tmp = exps_2a[i]/(exps[i]+1e-200);
                                        ax = tmp * gridx[i];
                                        ay = tmp * gridy[i];
                                        az = tmp * gridz[i];
                                        ce[0] = gridx[i] * gridx[i] * exps[i];
                                        ce[1] = gridx[i] * gridy[i] * exps[i];
                                        ce[2] = gridx[i] * gridz[i] * exps[i];
                                        ce[3] = gridy[i] * gridy[i] * exps[i];
                                        ce[4] = gridy[i] * gridz[i] * exps[i];
                                        ce[5] = gridz[i] * gridz[i] * exps[i];
                                        gto  [i] = gridx[i] * ce[0]; // xxx
                                        gto1 [i] = gridx[i] * ce[1]; // xxy
                                        gto2 [i] = gridx[i] * ce[2]; // xxz
                                        gto3 [i] = gridx[i] * ce[3]; // xyy
                                        gto4 [i] = gridx[i] * ce[4]; // xyz
                                        gto5 [i] = gridx[i] * ce[5]; // xzz
                                        gto6 [i] = gridy[i] * ce[3]; // yyy
                                        gto7 [i] = gridy[i] * ce[4]; // yyz
                                        gto8 [i] = gridy[i] * ce[5]; // yzz
                                        gto9 [i] = gridz[i] * ce[5]; // zzz
                                        gtox [i] = ax * gto [i] + 3 * ce[0];
                                        gtox1[i] = ax * gto1[i] + 2 * ce[1];
                                        gtox2[i] = ax * gto2[i] + 2 * ce[2];
                                        gtox3[i] = ax * gto3[i] +     ce[3];
                                        gtox4[i] = ax * gto4[i] +     ce[4];
                                        gtox5[i] = ax * gto5[i] +     ce[5];
                                        gtox6[i] = ax * gto6[i];
                                        gtox7[i] = ax * gto7[i];
                                        gtox8[i] = ax * gto8[i];
                                        gtox9[i] = ax * gto9[i];
                                        gtoy [i] = ay * gto [i];
                                        gtoy1[i] = ay * gto1[i] +     ce[0];
                                        gtoy2[i] = ay * gto2[i];
                                        gtoy3[i] = ay * gto3[i] + 2 * ce[1];
                                        gtoy4[i] = ay * gto4[i] +     ce[2];
                                        gtoy5[i] = ay * gto5[i];
                                        gtoy6[i] = ay * gto6[i] + 3 * ce[3];
                                        gtoy7[i] = ay * gto7[i] + 2 * ce[4];
                                        gtoy8[i] = ay * gto8[i] +     ce[5];
                                        gtoy9[i] = ay * gto9[i];
                                        gtoz [i] = az * gto [i];
                                        gtoz1[i] = az * gto1[i];
                                        gtoz2[i] = az * gto2[i] +     ce[0];
                                        gtoz3[i] = az * gto3[i];
                                        gtoz4[i] = az * gto4[i] +     ce[1];
                                        gtoz5[i] = az * gto5[i] + 2 * ce[2];
                                        gtoz6[i] = az * gto6[i];
                                        gtoz7[i] = az * gto7[i] +     ce[3];
                                        gtoz8[i] = az * gto8[i] + 2 * ce[4];
                                        gtoz9[i] = az * gto9[i] + 3 * ce[5];
                                } else {
                                        gto  [i] = 0;
                                        gto1 [i] = 0;
                                        gto2 [i] = 0;
                                        gto3 [i] = 0;
                                        gto4 [i] = 0;
                                        gto5 [i] = 0;
                                        gto6 [i] = 0;
                                        gto7 [i] = 0;
                                        gto8 [i] = 0;
                                        gto9 [i] = 0;
                                        gtox [i] = 0;
                                        gtox1[i] = 0;
                                        gtox2[i] = 0;
                                        gtox3[i] = 0;
                                        gtox4[i] = 0;
                                        gtox5[i] = 0;
                                        gtox6[i] = 0;
                                        gtox7[i] = 0;
                                        gtox8[i] = 0;
                                        gtox9[i] = 0;
                                        gtoy [i] = 0;
                                        gtoy1[i] = 0;
                                        gtoy2[i] = 0;
                                        gtoy3[i] = 0;
                                        gtoy4[i] = 0;
                                        gtoy5[i] = 0;
                                        gtoy6[i] = 0;
                                        gtoy7[i] = 0;
                                        gtoy8[i] = 0;
                                        gtoy9[i] = 0;
                                        gtoz [i] = 0;
                                        gtoz1[i] = 0;
                                        gtoz2[i] = 0;
                                        gtoz3[i] = 0;
                                        gtoz4[i] = 0;
                                        gtoz5[i] = 0;
                                        gtoz6[i] = 0;
                                        gtoz7[i] = 0;
                                        gtoz8[i] = 0;
                                        gtoz9[i] = 0;
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gto  += blksize * 10;
                        gtox += blksize * 10;
                        gtoy += blksize * 10;
                        gtoz += blksize * 10;
                }
                break;
        default:
                xpows_1less_in_power[0] = 0;
                ypows_1less_in_power[0] = 0;
                zpows_1less_in_power[0] = 0;
                for (k = 0; k < nc; k++) {
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
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
                                                gto[n*blksize+i] = exps[i] * tmp;
                                                gtox[n*blksize+i] = exps_2a[i] * gridx[i] * tmp;
                                                gtoy[n*blksize+i] = exps_2a[i] * gridy[i] * tmp;
                                                gtoz[n*blksize+i] = exps_2a[i] * gridz[i] * tmp;
                                                gtox[n*blksize+i] += exps[i] * lx * xpows[lx-1] * ypows[ly] * zpows[lz];
                                                gtoy[n*blksize+i] += exps[i] * ly * xpows[lx] * ypows[ly-1] * zpows[lz];
                                                gtoz[n*blksize+i] += exps[i] * lz * xpows[lx] * ypows[ly] * zpows[lz-1];
                                        } }
                                } else {
                                        for (n = 0; n < degen; n++) {
                                                gto [n*blksize+i] = 0;
                                                gtox[n*blksize+i] = 0;
                                                gtoy[n*blksize+i] = 0;
                                                gtoz[n*blksize+i] = 0;
                                        }
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gto     += blksize * degen;
                        gtox    += blksize * degen;
                        gtoy    += blksize * degen;
                        gtoz    += blksize * degen;
                }
        }
}

int GTOcontract_exp1(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, int blksize, double fac);

/*
void GTOval_cart_deriv0(int *shls_slice, int *ao_loc, int ngrids,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_cart_drv(grid_cart_gto0, contract_exp0,
                         param, shls_slice, ao_loc, ngrids,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv0(int *shls_slice, int *ao_loc, int ngrids,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_sph_drv(grid_cart_gto0, contract_exp0,
                        param, shls_slice, ao_loc, ngrids,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}
*/
void GTOval_cart(int *shls_slice, int *ao_loc, int ngrids,
                 double *ao, double *coord, char *non0table,
                 int *atm, int natm, int *bas, int nbas, double *env);
void GTOval_sph(int *shls_slice, int *ao_loc, int ngrids,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env);
void GTOval_cart_deriv0(int *shls_slice, int *ao_loc, int ngrids,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOval_cart(shls_slice, ao_loc, ngrids,
                    ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv0(int *shls_slice, int *ao_loc, int ngrids,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        GTOval_sph(shls_slice, ao_loc, ngrids,
                   ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOval_cart_deriv1(int *shls_slice, int *ao_loc, int ngrids,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                         param, shls_slice, ao_loc, ngrids,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph_deriv1(int *shls_slice, int *ao_loc, int ngrids,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                        param, shls_slice, ao_loc, ngrids,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

