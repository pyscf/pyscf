/*
 * Copyright (C) 2016-  Qiming Sun <osirpt.sun@gmail.com>
 */

#include <string.h>
#include <math.h>
#include "grid_ao_drv.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double exp_cephes(double x);
double CINTcommon_fac_sp(int l);

static int _len_cart[] = {
        1, 3, 6, 10, 15, 21, 28, 36
};

int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
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
                                peprim[j] = exp_cephes(-arr) * fac;
                                not0 = 1;
                        } else {
                                peprim[j] = 0;
                        }
                }
                peprim += nprim;
        }

        if (not0) {
                const char TRANS_T = 'T';
                const char TRANS_N = 'N';
                const double D0 = 0;
                const double D1 = 1;
                dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                       &D1, eprim, &nprim, coeff, &nprim, &D0, ectr, &blksize);
        } else {
                memset(ectr, 0, sizeof(double)*nctr*blksize);
        }

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

int GTOcontract_exp1(double *ectr, double *coord, double *alpha, double *coeff,
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
                                peprim[j] = exp_cephes(-arr) * fac;
                                not0 = 1;
                        } else {
                                peprim[j] = 0;
                        }
                }
                peprim += nprim;
        }

        if (not0) {
                const char TRANS_T = 'T';
                const char TRANS_N = 'N';
                const double D0 = 0;
                const double D1 = 1;
                double d2 = -2;
                double *ectr_2a = ectr;
                double coeff_a[nprim*nctr];
                dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                       &D1, eprim, &nprim, coeff, &nprim, &D0, ectr, &blksize);

                // -2 alpha_i C_ij exp(-alpha_i r_k^2)
                for (i = 0; i < nctr; i++) {
                for (j = 0; j < nprim; j++) {
                        coeff_a[i*nprim+j] = coeff[i*nprim+j]*alpha[j];
                } }

                ectr_2a += NPRIMAX*blksize;
                dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                       &d2, eprim, &nprim, coeff_a, &nprim,
                       &D0, ectr_2a, &blksize);
        } else {
                memset(ectr, 0, sizeof(double)*nctr*blksize*2);
        }

        return not0;
}

void GTOshell_eval_grid_ip_cart(double *gto, double *ri, double *exps,
                                double *coord, double *alpha, double *coeff,
                                int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
        const int gtosize = nc*degen*blksize;
        int lx, ly, lz, i, j, k, n;
        char mask[blksize+16];
        int *imask = (int *)mask;
        double xinv, yinv, zinv;
        double ax, ay, az, tmp, tmp1;
        double ce[6];
        double rre[10];
        double xpows_1less_in_power[64];
        double ypows_1less_in_power[64];
        double zpows_1less_in_power[64];
        double *xpows = xpows_1less_in_power + 1;
        double *ypows = ypows_1less_in_power + 1;
        double *zpows = zpows_1less_in_power + 1;
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double *gtox = gto;
        double *gtoy = gto + gtosize;
        double *gtoz = gto + gtosize * 2;
        double *exps_2a = exps + NPRIMAX*blksize;
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
                                        gtox[i] = exps_2a[i] * gridx[i];
                                        gtoy[i] = exps_2a[i] * gridy[i];
                                        gtoz[i] = exps_2a[i] * gridz[i];
                                } else {
                                        gtox[i] = 0;
                                        gtoy[i] = 0;
                                        gtoz[i] = 0;
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gtox += blksize;
                        gtoy += blksize;
                        gtoz += blksize;
                }
                break;
        case 1:
                for (k = 0; k < nc; k++) {
                        gtox1 = gtox + blksize*1;
                        gtoy1 = gtoy + blksize*1;
                        gtoz1 = gtoz + blksize*1;
                        gtox2 = gtox + blksize*2;
                        gtoy2 = gtoy + blksize*2;
                        gtoz2 = gtoz + blksize*2;
                        for (i = 0; i < blksize; i++) {
                                if (NOTZERO(exps[i])) {
                                        ax = exps_2a[i] * gridx[i];
                                        ay = exps_2a[i] * gridy[i];
                                        az = exps_2a[i] * gridz[i];
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
                        gtox += blksize * 3;
                        gtoy += blksize * 3;
                        gtoz += blksize * 3;
                }
                break;
        case 2:
                for (k = 0; k < nc; k++) {
                        gtox1 = gtox + blksize*1;
                        gtoy1 = gtoy + blksize*1;
                        gtoz1 = gtoz + blksize*1;
                        gtox2 = gtox + blksize*2;
                        gtoy2 = gtoy + blksize*2;
                        gtoz2 = gtoz + blksize*2;
                        gtox3 = gtox + blksize*3;
                        gtoy3 = gtoy + blksize*3;
                        gtoz3 = gtoz + blksize*3;
                        gtox4 = gtox + blksize*4;
                        gtoy4 = gtoy + blksize*4;
                        gtoz4 = gtoz + blksize*4;
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
                                        rre[0] = gridx[i] * ce[0]; // xx
                                        rre[1] = gridx[i] * ce[1]; // xy
                                        rre[2] = gridx[i] * ce[2]; // xz
                                        rre[3] = gridy[i] * ce[1]; // yy
                                        rre[4] = gridy[i] * ce[2]; // yz
                                        rre[5] = gridz[i] * ce[2]; // zz
                                        gtox [i] = ax * rre[0] + 2 * ce[0];
                                        gtox1[i] = ax * rre[1] +     ce[1];
                                        gtox2[i] = ax * rre[2] +     ce[2];
                                        gtox3[i] = ax * rre[3];
                                        gtox4[i] = ax * rre[4];
                                        gtox5[i] = ax * rre[5];
                                        gtoy [i] = ay * rre[0];
                                        gtoy1[i] = ay * rre[1] +     ce[0];
                                        gtoy2[i] = ay * rre[2];
                                        gtoy3[i] = ay * rre[3] + 2 * ce[1];
                                        gtoy4[i] = ay * rre[4] +     ce[2];
                                        gtoy5[i] = ay * rre[5];
                                        gtoz [i] = az * rre[0];
                                        gtoz1[i] = az * rre[1];
                                        gtoz2[i] = az * rre[2] +     ce[0];
                                        gtoz3[i] = az * rre[3];
                                        gtoz4[i] = az * rre[4] +     ce[1];
                                        gtoz5[i] = az * rre[5] + 2 * ce[2];
                                } else {
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
                        gtox += blksize * 6;
                        gtoy += blksize * 6;
                        gtoz += blksize * 6;
                }
                break;
        case 3:
                for (k = 0; k < nc; k++) {
                        gtox1 = gtox + blksize*1;
                        gtoy1 = gtoy + blksize*1;
                        gtoz1 = gtoz + blksize*1;
                        gtox2 = gtox + blksize*2;
                        gtoy2 = gtoy + blksize*2;
                        gtoz2 = gtoz + blksize*2;
                        gtox3 = gtox + blksize*3;
                        gtoy3 = gtoy + blksize*3;
                        gtoz3 = gtoz + blksize*3;
                        gtox4 = gtox + blksize*4;
                        gtoy4 = gtoy + blksize*4;
                        gtoz4 = gtoz + blksize*4;
                        gtox5 = gtox + blksize*5;
                        gtoy5 = gtoy + blksize*5;
                        gtoz5 = gtoz + blksize*5;
                        gtox6 = gtox + blksize*6;
                        gtoy6 = gtoy + blksize*6;
                        gtoz6 = gtoz + blksize*6;
                        gtox7 = gtox + blksize*7;
                        gtoy7 = gtoy + blksize*7;
                        gtoz7 = gtoz + blksize*7;
                        gtox8 = gtox + blksize*8;
                        gtoy8 = gtoy + blksize*8;
                        gtoz8 = gtoz + blksize*8;
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
                                        gtox [i] = ax * rre[0] + 3 * ce[0];
                                        gtox1[i] = ax * rre[1] + 2 * ce[1];
                                        gtox2[i] = ax * rre[2] + 2 * ce[2];
                                        gtox3[i] = ax * rre[3] +     ce[3];
                                        gtox4[i] = ax * rre[4] +     ce[4];
                                        gtox5[i] = ax * rre[5] +     ce[5];
                                        gtox6[i] = ax * rre[6];
                                        gtox7[i] = ax * rre[7];
                                        gtox8[i] = ax * rre[8];
                                        gtox9[i] = ax * rre[9];
                                        gtoy [i] = ay * rre[0];
                                        gtoy1[i] = ay * rre[1] +     ce[0];
                                        gtoy2[i] = ay * rre[2];
                                        gtoy3[i] = ay * rre[3] + 2 * ce[1];
                                        gtoy4[i] = ay * rre[4] +     ce[2];
                                        gtoy5[i] = ay * rre[5];
                                        gtoy6[i] = ay * rre[6] + 3 * ce[3];
                                        gtoy7[i] = ay * rre[7] + 2 * ce[4];
                                        gtoy8[i] = ay * rre[8] +     ce[5];
                                        gtoy9[i] = ay * rre[9];
                                        gtoz [i] = az * rre[0];
                                        gtoz1[i] = az * rre[1];
                                        gtoz2[i] = az * rre[2] +     ce[0];
                                        gtoz3[i] = az * rre[3];
                                        gtoz4[i] = az * rre[4] +     ce[1];
                                        gtoz5[i] = az * rre[5] + 2 * ce[2];
                                        gtoz6[i] = az * rre[6];
                                        gtoz7[i] = az * rre[7] +     ce[3];
                                        gtoz8[i] = az * rre[8] + 2 * ce[4];
                                        gtoz9[i] = az * rre[9] + 3 * ce[5];
                                } else {
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
                                        mask[i] = 1;
                                } else {
                                        mask[i] = 0;
                                }
                        }
                        for (i = 0; i < blksize/4; i++) {
                                if (imask[i]) {
        for (j = 0; j < 4; j++) {
                xpows[j] = 1;
                ypows[j] = 1;
                zpows[j] = 1;
        }
        for (lx = 1; lx <= l; lx++) {
                for (j = 0; j < 4; j++) {
                        xpows[lx*4+j] = xpows[(lx-1)*4+j] * gridx[i*4+j];
                        ypows[lx*4+j] = ypows[(lx-1)*4+j] * gridy[i*4+j];
                        zpows[lx*4+j] = zpows[(lx-1)*4+j] * gridz[i*4+j];
                }
        }
        for (lx = l, n = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, n++) {
                lz = l - lx - ly;
                for (j = 0; j < 4; j++) {
                        xinv = lx/(gridx[i*4+j]+1e-200);
                        yinv = ly/(gridy[i*4+j]+1e-200);
                        zinv = lz/(gridz[i*4+j]+1e-200);
                        tmp = exps_2a[i*4+j]/(exps[i*4+j]+1e-200);
                        tmp1 = xpows[lx*4+j] * ypows[ly*4+j]
                             * zpows[lz*4+j] * exps[i*4+j];
                        gtox[n*blksize+i*4+j] = (xinv + tmp*gridx[i*4+j]) * tmp1;
                        gtoy[n*blksize+i*4+j] = (yinv + tmp*gridy[i*4+j]) * tmp1;
                        gtoz[n*blksize+i*4+j] = (zinv + tmp*gridz[i*4+j]) * tmp1;
                }
        } }
                                } else {
        for (n = 0; n < degen; n++) {
                for (j = 0; j < 4; j++) {
                        gtox[n*blksize+i*4+j] = 0;
                        gtoy[n*blksize+i*4+j] = 0;
                        gtoz[n*blksize+i*4+j] = 0;
                }
        }
                                }
                        }
                        for (i = i*4; i < blksize; i++) {
                                if (mask[i]) {
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
                                                gtox[n*blksize+i] = exps_2a[i] * gridx[i] * tmp;
                                                gtoy[n*blksize+i] = exps_2a[i] * gridy[i] * tmp;
                                                gtoz[n*blksize+i] = exps_2a[i] * gridz[i] * tmp;
                                                gtox[n*blksize+i] += exps[i] * lx * xpows[lx-1] * ypows[ly] * zpows[lz];
                                                gtoy[n*blksize+i] += exps[i] * ly * xpows[lx] * ypows[ly-1] * zpows[lz];
                                                gtoz[n*blksize+i] += exps[i] * lz * xpows[lx] * ypows[ly] * zpows[lz-1];
                                        } }
                                } else {
                                        for (n = 0; n < degen; n++) {
                                                gtox[n*blksize+i] = 0;
                                                gtoy[n*blksize+i] = 0;
                                                gtoz[n*blksize+i] = 0;
                                        }
                                }
                        }
                        exps    += blksize;
                        exps_2a += blksize;
                        gtox    += blksize * degen;
                        gtoy    += blksize * degen;
                        gtoz    += blksize * degen;
                }
        }
}

void GTOval_cart(int nao, int ngrids,
                 int blksize, int bastart, int bascount,
                 double *ao, double *coord, char *non0table,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_cart_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                         param, nao, ngrids, blksize, bastart, bascount,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_sph(int nao, int ngrids,
                int blksize, int bastart, int bascount,
                double *ao, double *coord, char *non0table,
                int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        GTOeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                        param, nao, ngrids, blksize, bastart, bascount,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void GTOval_ip_cart(int nao, int ngrids,
                    int blksize, int bastart, int bascount,
                    double *ao, double *coord, char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_cart_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1,
                         param, nao, ngrids, blksize, bastart, bascount,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void GTOval_ip_sph(int nao, int ngrids,
                   int blksize, int bastart, int bascount,
                   double *ao, double *coord, char *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 3};
        GTOeval_sph_drv(GTOshell_eval_grid_ip_cart, GTOcontract_exp1,
                        param, nao, ngrids, blksize, bastart, bascount,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}


