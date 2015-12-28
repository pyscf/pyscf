/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"
#include "vhf/fblas.h"

// 128s42p21d12f8g6h4i3j 
#define NCTR_CART      128
//  72s24p14d10f8g6h5i4j 
#define NCTR_SPH        72
#define NPRIMAX         64
#define BLKSIZE         96
#define EXPCUTOFF       50  // 1e-22
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define NOTZERO(e)      ((e)>1e-18 || (e)<-1e-18)

double CINTcommon_fac_sp(int l);
double exp_cephes(double);

static int _len_cart[] = {
        1, 3, 6, 10, 15, 21, 28, 36
};

// exps, np can be used for both primitive or contract case
// for contracted case, np stands for num contracted functions, exps are
// contracted factors = \sum c_{i} exp(-a_i*r_i**2)
static void grid_cart_gto0(double *gto, double *coord, double *exps,
                           int l, int np, int blksize)
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
                for (k = 0; k < np; k++) {
                        for (i = 0; i < blksize; i++) {
                                gto[i] = exps[i];
                        }
                        exps += blksize;
                        gto += blksize;
                }
                break;
        case 1:
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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

static void grid_cart_gto1(double *gto, double *coord, double *exps,
                           int l, int np, int blksize)
{
        const int degen = _len_cart[l];
        const int gtosize = np*degen*blksize;
        int lx, ly, lz, i, j, k, n;
        char mask[blksize+16];
        int *imask = (int *)mask;
        double xinv, yinv, zinv;
        double ax, ay, az, tmp, tmp1;
        double ce[6];
        double xpows[64];
        double ypows[64];
        double zpows[64];
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
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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
                for (k = 0; k < np; k++) {
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
                        gto[n*blksize+i*4+j] = tmp1;
                        gtox[n*blksize+i*4+j] = (xinv + tmp*gridx[i*4+j]) * tmp1;
                        gtoy[n*blksize+i*4+j] = (yinv + tmp*gridy[i*4+j]) * tmp1;
                        gtoz[n*blksize+i*4+j] = (zinv + tmp*gridz[i*4+j]) * tmp1;
                }
        } }
                                } else {
        for (n = 0; n < degen; n++) {
                for (j = 0; j < 4; j++) {
                        gto [n*blksize+i*4+j] = 0;
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
                                                xinv = lx/(gridx[i]+1e-200);
                                                yinv = ly/(gridy[i]+1e-200);
                                                zinv = lz/(gridz[i]+1e-200);
                                                tmp = exps_2a[i]/(exps[i]+1e-200);
                                                tmp1 = xpows[lx] * ypows[ly]
                                                     * zpows[lz] * exps[i];
                                                gto[n*blksize+i] = tmp1;
                                                gtox[n*blksize+i] = (xinv + tmp*gridx[i]) * tmp1;
                                                gtoy[n*blksize+i] = (yinv + tmp*gridy[i]) * tmp1;
                                                gtoz[n*blksize+i] = (zinv + tmp*gridz[i]) * tmp1;
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

static void derivative(double *fx1, double *fy1, double *fz1,
                       double *fx0, double *fy0, double *fz0, double a, int l)
{
        int i;
        double a2 = -2 * a;
        fx1[0] = a2*fx0[1];
        fy1[0] = a2*fy0[1];
        fz1[0] = a2*fz0[1];
        for (i = 1; i <= l; i++) {
                fx1[i] = i*fx0[i-1] + a2*fx0[i+1];
                fy1[i] = i*fy0[i-1] + a2*fy0[i+1];
                fz1[i] = i*fz0[i-1] + a2*fz0[i+1];
        }
}

static void grid_cart_gto0_slow(double *cgto, double *coord, double *exps,
                                double *alpha, double *coeff,
                                int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
        const int mblksize = blksize * degen;
        const int gtosize = np * mblksize;
        double gtobuf[gtosize];
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        grid_cart_gto0(gtobuf, coord, exps, l, np, blksize);

        dgemm_(&TRANS_N, &TRANS_N, &mblksize, &nc, &np,
               &D1, gtobuf, &mblksize, coeff, &np, &D0, cgto, &mblksize);
}

static void grid_cart_gto1_slow(double *cgto, double *coord, double *exps,
                                double *alpha, double *coeff,
                                int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
        const int mblksize = blksize * degen;
        const int gtosize = np * mblksize;
        int lx, ly, lz, i, k, n;
        double fx0[16];
        double fy0[16];
        double fz0[16];
        double fx1[16];
        double fy1[16];
        double fz1[16];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double gtobuf[gtosize*4];
        double *gto = gtobuf;
        double *gtox = gto + gtosize;
        double *gtoy = gto + gtosize * 2;
        double *gtoz = gto + gtosize * 3;

        for (k = 0; k < np; k++) {
                for (i = 0; i < blksize; i++) {
                        if (NOTZERO(exps[i])) {
        fx0[0] = 1;
        fy0[0] = 1;
        fz0[0] = 1;
        for (lx = 1; lx <= l+1; lx++) {
                fx0[lx] = fx0[lx-1] * gridx[i];
                fy0[lx] = fy0[lx-1] * gridy[i];
                fz0[lx] = fz0[lx-1] * gridz[i];
        }
        derivative(fx1, fy1, fz1, fx0, fy0, fz0, alpha[k], l);
        for (lx = l, n = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, n++) {
                lz = l - lx - ly;
                gto  [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz0[lz];
                gtox [n*blksize+i] = exps[i] * fx1[lx] * fy0[ly] * fz0[lz];
                gtoy [n*blksize+i] = exps[i] * fx0[lx] * fy1[ly] * fz0[lz];
                gtoz [n*blksize+i] = exps[i] * fx0[lx] * fy0[ly] * fz1[lz];
        } }
                        } else {
                                for (n = 0; n < degen; n++) {
                                        gto  [n*blksize+i] = 0;
                                        gtox [n*blksize+i] = 0;
                                        gtoy [n*blksize+i] = 0;
                                        gtoz [n*blksize+i] = 0;
                                }
                        }
                }
                exps  += blksize;
                gto   += mblksize;
                gtox  += mblksize;
                gtoy  += mblksize;
                gtoz  += mblksize;
        }

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        for (k = 0; k < 4; k++) {
                dgemm_(&TRANS_N, &TRANS_N, &mblksize, &nc, &np,
                       &D1, gtobuf+gtosize*k, &mblksize, coeff, &np,
                       &D0, cgto+nc*mblksize*k, &mblksize);
        }
}

static void grid_cart_gto2_slow(double *cgto, double *coord, double *exps,
                                double *alpha, double *coeff,
                                int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
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
        derivative(fx1, fy1, fz1, fx0, fy0, fz0, alpha[k], l+1);
        derivative(fx2, fy2, fz2, fx1, fy1, fz1, alpha[k], l);
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

static void grid_cart_gto3_slow(double *cgto, double *coord, double *exps,
                                double *alpha, double *coeff,
                                int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
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
        derivative(fx1, fy1, fz1, fx0, fy0, fz0, alpha[k], l+2);
        derivative(fx2, fy2, fz2, fx1, fy1, fz1, alpha[k], l+1);
        derivative(fx3, fy3, fz3, fx2, fy2, fz2, alpha[k], l);
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

static void grid_cart_gto4_slow(double *cgto, double *coord, double *exps,
                                double *alpha, double *coeff,
                                int l, int np, int nc, int blksize)
{
        const int degen = _len_cart[l];
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
        derivative(fx1, fy1, fz1, fx0, fy0, fz0, alpha[k], l+3);
        derivative(fx2, fy2, fz2, fx1, fy1, fz1, alpha[k], l+2);
        derivative(fx3, fy3, fz3, fx2, fy2, fz2, alpha[k], l+1);
        derivative(fx4, fy4, fz4, fx3, fy3, fz3, alpha[k], l);
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

void (*grid_cart_gto[])() = {
        grid_cart_gto0,
        grid_cart_gto1,
        NULL,
        NULL,
};

void (*grid_cart_gto_slow[])() = {
        grid_cart_gto0_slow,
        grid_cart_gto1_slow,
        grid_cart_gto2_slow,
        grid_cart_gto3_slow,
        grid_cart_gto4_slow,
};

static int _contract_exp(double *ectr, double *coord, double *alpha, double *coeff,
                         int l, int nprim, int nctr, int blksize, int deriv)
{
        int i, j, k;
        double arr, maxc;
        double fac = CINTcommon_fac_sp(l);
        double eprim[nprim*blksize];
        double logcoeff[nprim];
        double rr[blksize];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double *peprim = eprim;
        int not0 = 0;

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
                double d2 = 1;
                double *ectr_2a = ectr;
                double coeff_a[nprim*nctr];
                dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                       &D1, eprim, &nprim, coeff, &nprim, &D0, ectr, &blksize);

                // -2 alpha_i C_ij exp(-alpha_i r_k^2)
                for (k = 1; k <= deriv; k++) {
                        if (k == 1) {
                                for (i = 0; i < nctr; i++) {
                                for (j = 0; j < nprim; j++) {
                                        coeff_a[i*nprim+j] = coeff[i*nprim+j]*alpha[j];
                                } }
                        } else {
                                for (i = 0; i < nctr; i++) {
                                for (j = 0; j < nprim; j++) {
                                        coeff_a[i*nprim+j] *= alpha[j];
                                } }
                        }

                        ectr_2a += NPRIMAX*blksize;
                        d2 *= -2;
                        dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                               &d2, eprim, &nprim, coeff_a, &nprim,
                               &D0, ectr_2a, &blksize);
                }
        }

        return not0;
}

static int _prim_exp(double *eprim, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, int blksize, int deriv)
{
        int i, j;
        double arr, maxc;
        double fac = CINTcommon_fac_sp(l);
        double logcoeff[nprim];
        double rr[blksize];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
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

        for (j = 0; j < nprim; j++) {
                for (i = 0; i < blksize; i++) {
                        arr = alpha[j] * rr[i];
                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                eprim[j*blksize+i] = exp_cephes(-arr) * fac;
                                not0 = 1;
                        } else {
                                eprim[j*blksize+i] = 0;
                        }
                }
        }
        return not0;
}


// grid2atm[atm_id,xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *coord, int blksize,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int atm_id, ig;
        double *r_atm;
        for (atm_id = 0; atm_id < natm; atm_id++) {
                r_atm = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                for (ig = 0; ig < blksize; ig++) {
                        grid2atm[0*blksize+ig] = coord[ig*3+0] - r_atm[0];
                        grid2atm[1*blksize+ig] = coord[ig*3+1] - r_atm[1];
                        grid2atm[2*blksize+ig] = coord[ig*3+2] - r_atm[2];
                }
                grid2atm += 3*blksize;
        }
}


static void _trans(double *ao, double *aobuf, int nao, int blksize, int counts)
{
        int i, j, k;
        if (blksize == BLKSIZE) {
                for (k = 0; k < BLKSIZE; k+=16) {
                        for (i = 0; i < counts; i++) {
                                for (j = k; j < k+16; j++) {
                                        ao[j*nao+i] = aobuf[i*BLKSIZE+j];
                                }
                        }
                }
        } else if ((blksize % 16) == 0) {
                for (k = 0; k < blksize; k+=16) {
                        for (i = 0; i < counts; i++) {
                                for (j = k; j < k+16; j++) {
                                        ao[j*nao+i] = aobuf[i*blksize+j];
                                }
                        }
                }
        } else {
                for (i = 0; i < counts; i++) {
                        for (j = 0; j < blksize; j++) {
                                ao[j*nao+i] = aobuf[j];
                        }
                        aobuf += blksize;
                }
        }
}

static void _set0(double *ao, int nao, int blksize, int counts)
{
        int i, j;
        for (j = 0; j < blksize; j++) {
                for (i = 0; i < counts; i++) {
                        ao[j*nao+i] = 0;
                }
        }
}

/*
 * ao in Fortran contiguous
 * deriv 0 ao[:nao,:ngrids]
 * deriv 1 ao[:nao,:ngrids,:4] ~ ao, ao_dx, ao_dy, ao_dz
 * deriv 2 ao[:nao,:ngrids,:10] ~ ao, ao_dx, ao_dy, ao_dz, xx, xy, xz, yy, yz, zz
 * deriv 3 ao[:nao,:ngrids,:20] ~ ao, ao_dx, ao_dy, ao_dz, xx, xy, xz, yy, yz, zz, xxx, xxy, ...
 */
void VXCeval_nr_iter(int nao, int ngrids, int deriv, int blksize,
                     int bastart, int bascount, double *ao, double *coord,
                     char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ndd = (deriv+1) * (deriv+2) * (deriv+3) / 6;
        const int basend = bastart + bascount;
        const int atmstart = bas[bastart*BAS_SLOTS+ATOM_OF];
        const int atmend = bas[(basend-1)*BAS_SLOTS+ATOM_OF]+1;
        const int atmcount = atmend - atmstart;
        int i, k, l, np, nc, atm_id, bas_id, deg;
        int ao_id = 0;
        double *p_exp, *pcoeff, *pcoord, *pcart;
        double ectr[NPRIMAX*blksize*(deriv+1)];
        double cart_gto[NCTR_CART*blksize * ndd];
        double aobuf[NCTR_SPH*blksize * ndd];
        double grid2atm[atmcount*3*blksize]; // [atm_id,xyz,grid]
        double *paobuf;
        int (*fexp)();
        void (*fgrid)();
        if (deriv < 2) {
                fexp = _contract_exp;
                fgrid = grid_cart_gto[deriv];
        } else {
                fexp = _prim_exp;
                fgrid = grid_cart_gto_slow[deriv];
        }

        _fill_grid2atm(grid2atm, coord, blksize,
                       atm+atmstart*ATM_SLOTS, atmcount, bas, nbas, env);

        for (bas_id = bastart; bas_id < basend; bas_id++) {
                np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                deg = l * 2 + 1;
                p_exp = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF] - atmstart;
                pcoord = grid2atm + atm_id * 3*blksize;
                if (non0table[bas_id] &&
                    (*fexp)(ectr, pcoord, p_exp, pcoeff,
                            l, np, nc, blksize, deriv)) {
                        if (deriv < 2) {
                                (*fgrid)(cart_gto, pcoord, ectr,
                                         l, nc, blksize);
                        } else {
                                (*fgrid)(cart_gto, pcoord, ectr, p_exp, pcoeff,
                                         l, np, nc, blksize);
                        }
                        for (i = 0; i < ndd; i++) {
                                pcart = cart_gto + i*nc*_len_cart[l]*blksize;
                                if (l < 2) { // s, p functions
                                        _trans(ao+i*nao*ngrids+ao_id, pcart,
                                               nao, blksize, nc*deg);
                                } else {
                                        paobuf = aobuf;
                                        for (k = 0; k < nc; k++) {
                                                CINTc2s_ket_sph(paobuf, blksize,
                                                                pcart, l);
                                                pcart += _len_cart[l] * blksize;
                                                paobuf += deg * blksize;
                                        }
                                        _trans(ao+i*nao*ngrids+ao_id, aobuf,
                                               nao, blksize, nc*deg);
                                }
                        }
                } else {
                        for (i = 0; i < ndd; i++) {
                                _set0(ao+i*nao*ngrids+ao_id, nao, blksize, nc*deg);
                        }
                }
                ao_id += deg * nc;
        }
}

/*
 * blksize <= 1024 to avoid stack overflow
 * non0table[ngrids/blksize,natm] is the T/F table for ao values
 * It used to screen the ao evaluation for each shells
 */
void VXCeval_ao_drv(int deriv, int nao, int ngrids,
                    int bastart, int bascount, int blksize,
                    double *ao, double *coord, char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        ao += CINTtot_cgto_spheric(bas, bastart);

        const int nblk = (ngrids+blksize-1) / blksize;

        int ip, ib;
#pragma omp parallel default(none) \
        shared(deriv, nao, ngrids, bastart, bascount, blksize, \
               ao, coord, non0table, atm, natm, bas, nbas, env) \
        private(ip, ib)
        {
#pragma omp for nowait schedule(dynamic, 1)
                for (ib = 0; ib < nblk; ib++) {
                        ip = ib * blksize;
                        VXCeval_nr_iter(nao, ngrids, deriv, MIN(ngrids-ip, blksize),
                                        bastart, bascount, ao+ip*nao, coord+ip*3,
                                        non0table+ib*nbas,
                                        atm, natm, bas, nbas, env);
                }
        }
}

void VXCnr_ao_screen(signed char *non0table, double *coord,
                     int ngrids, int blksize,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ib, i, j;
        int np, nc, atm_id, bas_id;
        double rr, arr, maxc;
        double logcoeff[NPRIMAX];
        double dr[3];
        double *p_exp, *pcoeff, *pcoord, *ratm;

        memset(non0table, 0, sizeof(signed char) * nblk*nbas);

        for (bas_id = 0; bas_id < nbas; bas_id++) {
                np = bas[NPRIM_OF];
                nc = bas[NCTR_OF ];
                p_exp = env + bas[PTR_EXP];
                pcoeff = env + bas[PTR_COEFF];
                atm_id = bas[ATOM_OF];
                ratm = env + atm[atm_id*ATM_SLOTS+PTR_COORD];

                for (j = 0; j < np; j++) {
                        maxc = 0;
                        for (i = 0; i < nc; i++) {
                                maxc = MAX(maxc, fabs(pcoeff[i*np+j]));
                        }
                        logcoeff[j] = log(maxc);
                }

                pcoord = coord;
                for (ib = 0; ib < nblk; ib++) {
                        for (i = 0; i < MIN(ngrids-ib*blksize, blksize); i++) {
                                dr[0] = pcoord[i*3+0] - ratm[0];
                                dr[1] = pcoord[i*3+1] - ratm[1];
                                dr[2] = pcoord[i*3+2] - ratm[2];
                                rr = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                                for (j = 0; j < np; j++) {
                                        arr = p_exp[j] * rr;
                                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                                non0table[ib*nbas+bas_id] = 1;
                                                goto next_blk;
                                        }
                                }
                        }
next_blk:
                        pcoord += blksize*3;
                }
                bas += BAS_SLOTS;
        }
}

void VXCoriginal_becke(double *out, double *g, int n)
{
        int i;
        double s;
#pragma omp parallel default(none) \
        shared(out, g, n) private(i, s)
{
#pragma omp for nowait schedule(static)
        for (i = 0; i < n; i++) {
                s = (3 - g[i]*g[i]) * g[i] * .5;
                s = (3 - s*s) * s * .5;
                out[i] = (3 - s*s) * s * .5;
        }
}
}
