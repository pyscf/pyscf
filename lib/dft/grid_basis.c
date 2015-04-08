/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"
#include "vhf/fblas.h"

// l = 6, nprim can reach 9
#define NPRIM_CART      256
#define NPRIMAX         64
#define BLKSIZE         224
#define EXPCUTOFF       30  // 1e-13
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
static void grid_cart_gto(double *gto, double *coord, double *exps,
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

static void grid_cart_gto_grad(double *gto, double *coord, double *exps,
                               int l, int np, int blksize)
{
        const int gtosize = np*_len_cart[l]*blksize;
        int lx, ly, lz, i, k;
        double xinv, yinv, zinv;
        double ax, ay, az, tmp;
        double ce[6];
        double xpows[8*blksize];
        double ypows[8*blksize];
        double zpows[8*blksize];
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
                                                xinv = 1/(gridx[i]+1e-200);
                                                yinv = 1/(gridy[i]+1e-200);
                                                zinv = 1/(gridz[i]+1e-200);
                                                tmp = exps_2a[i]/(exps[i]+1e-200);
                                                ax = tmp * gridx[i];
                                                ay = tmp * gridy[i];
                                                az = tmp * gridz[i];
                                                gto[i] = xpows[i*8+lx]
                                                       * ypows[i*8+ly]
                                                       * zpows[i*8+lz]*exps[i];
                                                gtox[i] = (lx * xinv + ax) * gto[i];
                                                gtoy[i] = (ly * yinv + ay) * gto[i];
                                                gtoz[i] = (lz * zinv + az) * gto[i];
                                        } else {
                                                gto [i] = 0;
                                                gtox[i] = 0;
                                                gtoy[i] = 0;
                                                gtoz[i] = 0;
                                        }
                                }
                                gto  += blksize;
                                gtox += blksize;
                                gtoy += blksize;
                                gtoz += blksize;
                        } }
                        exps    += blksize;
                        exps_2a += blksize;
                }
        }
}

// ectr["exps",ngrid]
static int _contract_exp(double *ectr, double *coord, double *alpha,
                         double *coeff,
                         int l, int nprim, int nctr, int blksize)
{
        int i, j;
        double rr, arr, maxc;
        double fac = CINTcommon_fac_sp(l);
        double eprim[nprim*blksize];
        double logcoeff[nprim];
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
                rr = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
                for (j = 0; j < nprim; j++) {
                        arr = alpha[j] * rr;
                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                peprim[j] = exp_cephes(-arr) * fac;
                                not0 = 1;
                        } else {
                                peprim[j] = 0;
                        }
                }
                peprim += nprim;
                coord += 3;
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

static int _contract_exp_grad(double *ectr, double *coord, double *alpha,
                              double *coeff,
                              int l, int nprim, int nctr, int blksize)
{
        int i, j;
        double rr, arr, maxc;
        double fac = CINTcommon_fac_sp(l);
        double eprim[nprim*blksize];
        double logcoeff[nprim];
        double coeff_a[nprim*nctr];
        double *gridx = coord;
        double *gridy = coord+blksize;
        double *gridz = coord+blksize*2;
        double *ectr_2a = ectr + NPRIMAX*blksize;
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
                rr = gridx[i]*gridx[i] + gridy[i]*gridy[i] + gridz[i]*gridz[i];
                for (j = 0; j < nprim; j++) {
                        arr = alpha[j] * rr;
                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                peprim[j] = exp_cephes(-arr) * fac;
                                not0 = 1;
                        } else {
                                peprim[j] = 0;
                        }
                }
                peprim += nprim;
                coord += 3;
        }

        if (not0) {
                const char TRANS_T = 'T';
                const char TRANS_N = 'N';
                const double D0 = 0;
                const double D1 = 1;
                const double D2 = -2;
                dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                       &D1, eprim, &nprim, coeff, &nprim, &D0, ectr, &blksize);

                for (i = 0; i < nctr; i++) {
                        for (j = 0; j < nprim; j++) {
                                coeff_a[i*nprim+j] = coeff[i*nprim+j]*alpha[j];
                        }
                }
                dgemm_(&TRANS_T, &TRANS_N, &blksize, &nctr, &nprim,
                       &D2, eprim, &nprim, coeff_a, &nprim,
                       &D0, ectr_2a, &blksize);
        } else {
                memset(ectr   , 0, sizeof(double)*nctr*blksize);
                memset(ectr_2a, 0, sizeof(double)*nctr*blksize);
        }

        return not0;
}


// grid2atm[atm_id,xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *coord,
                           int blksize, int bastart, int bascount,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int atm_id, ig;
        double *r_atm;
        bas += bastart * BAS_SLOTS;
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

// ao[:nao,:ngrids] in Fortran-order
void VXCeval_nr_gto(int nao, int ngrids, int blksize,
                    int bastart, int bascount, double *ao, double *coord,
                    char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int k, l, np, nc, atm_id, bas_id, deg;
        int counts = 0;
        int *pbas = bas;
        double *p_exp, *pcoeff, *pcoord, *pgto, *pcart;
        double *ectr = malloc(sizeof(double) * NPRIMAX*blksize);
        double *cart_gto = malloc(sizeof(double) * NPRIM_CART*blksize);
        double *aobuf = malloc(sizeof(double) * nao * blksize);
        double *paobuf = aobuf;
        double *grid2atm = malloc(sizeof(double) * natm*3*blksize); // [atm_id,xyz,grid]
        _fill_grid2atm(grid2atm, coord, blksize, bastart, bascount,
                       atm, natm, bas, nbas, env);

        pbas = bas + bastart * BAS_SLOTS;
        for (bas_id = bastart; bas_id < bastart+bascount; bas_id++) {
                np = pbas[NPRIM_OF];
                nc = pbas[NCTR_OF ];
                l  = pbas[ANG_OF  ];
                deg = l * 2 + 1;
                p_exp = env + pbas[PTR_EXP];
                pcoeff = env + pbas[PTR_COEFF];
                atm_id = pbas[ATOM_OF];
                pcoord = grid2atm + atm_id * blksize*3;
                if (non0table[bas_id] &&
                    _contract_exp(ectr, pcoord, p_exp, pcoeff,
                                  l, np, nc, blksize)) {
                        grid_cart_gto(cart_gto, pcoord, ectr, l, nc, blksize);
                        pcart = cart_gto;
                        for (k = 0; k < nc; k++) {
                                pgto = CINTc2s_ket_sph(paobuf, blksize, pcart, l);
                                if (pgto != paobuf) { // s,p functions
                                        memcpy(paobuf, pcart, sizeof(double)*deg*blksize);
                                }
                                pcart += _len_cart[l] * blksize;
                                paobuf += deg * blksize;
                        }
                } else {
                        memset(paobuf, 0, sizeof(double) * nc*deg*blksize);
                        paobuf += nc * deg * blksize;
                }
                pbas += BAS_SLOTS;
                counts += deg * nc;
        }
        _trans(ao, aobuf, nao, blksize, counts);
        free(ectr);
        free(cart_gto);
        free(grid2atm);
        free(aobuf);
}

// in ao[:nao,:ngrids,:4] in Fortran-order, [:4] ~ ao, ao_dx, ao_dy, ao_dz
void VXCeval_nr_gto_grad(int nao, int ngrids, int blksize,
                         int bastart, int bascount, double *ao, double *coord,
                         char *non0table,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, k, l, np, nc, atm_id, bas_id, deg;
        int counts = 0;
        int *pbas = bas;
        double *p_exp, *pcoeff, *pcoord, *pgto, *pcart;
        double *ectr = malloc(sizeof(double) * NPRIMAX*blksize*2);
        double *cart_gto = malloc(sizeof(double) * NPRIM_CART*blksize * 4);
        double *aobuf = malloc(sizeof(double) * nao*blksize * 4);
        double *paobuf = aobuf;
        double *paobuf1;
        double *grid2atm = malloc(sizeof(double) * natm*3*blksize); // [atm_id,xyz,grid]
        _fill_grid2atm(grid2atm, coord, blksize, bastart, bascount,
                       atm, natm, bas, nbas, env);

        pbas = bas + bastart * BAS_SLOTS;
        for (bas_id = bastart; bas_id < bastart+bascount; bas_id++) {
                np = pbas[NPRIM_OF];
                nc = pbas[NCTR_OF ];
                l  = pbas[ANG_OF  ];
                deg = l * 2 + 1;
                p_exp = env + pbas[PTR_EXP];
                pcoeff = env + pbas[PTR_COEFF];
                atm_id = pbas[ATOM_OF];
                pcoord = grid2atm + atm_id * 3*blksize;
                if (non0table[bas_id] &&
                    _contract_exp_grad(ectr, pcoord, p_exp, pcoeff,
                                       l, np, nc, blksize)) {
                        grid_cart_gto_grad(cart_gto, pcoord, ectr, l, nc, blksize);
                        for (i = 0; i < 4; i++) {
                                pcart = cart_gto + i*nc*_len_cart[l]*blksize;
                                paobuf1 = paobuf + i*nao*blksize;
                                for (k = 0; k < nc; k++) {
                                        pgto = CINTc2s_ket_sph(paobuf1, blksize,
                                                               pcart, l);
                                        if (pgto != paobuf1) { // s,p functions
                                                memcpy(paobuf1, pcart,
                                                       sizeof(double)*deg*blksize);
                                        }
                                        pcart += _len_cart[l] * blksize;
                                        paobuf1 += deg * blksize;
                                }
                        }
                } else {
                        for (i = 0; i < 4; i++) {
                                paobuf1 = paobuf + i*nao*blksize;
                                memset(paobuf1, 0, sizeof(double)*nc*deg*blksize);
                        }
                }
                paobuf += nc * deg * blksize;
                pbas += BAS_SLOTS;
                counts += deg * nc;
        }
        for (i = 0; i < 4; i++) {
                // note the structure of ao[4,ngrids,nao]
                _trans(ao+i*nao*ngrids, aobuf+i*nao*blksize, nao, blksize, counts);
        }
        free(ectr);
        free(cart_gto);
        free(aobuf);
        free(grid2atm);
}

/*
 * blksize <= 1024 to avoid stack overflow
 * non0table[ngrids/blksize,natm] is the T/F table for ao values
 * It used to screen the ao evaluation for each shells
 */
void VXCeval_ao_drv(void (*eval_gto)(),
                    int nao, int ngrids, int bastart, int bascount, int blksize,
                    double *ao, double *coord, char *non0table,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        ao += CINTtot_cgto_spheric(bas, bastart);

        const int nblk = (ngrids+blksize-1) / blksize;

        int ip, ib;
#pragma omp parallel default(none) \
        shared(eval_gto, nao, ngrids, bastart, bascount, blksize, \
               ao, coord, non0table, atm, natm, bas, nbas, env) \
        private(ip, ib)
        {
#pragma omp for nowait schedule(dynamic, 1)
                for (ib = 0; ib < nblk; ib++) {
                        ip = ib * blksize;
                        (*eval_gto)(nao, ngrids, MIN(ngrids-ip, blksize),
                                    bastart, bascount, ao+ip*nao, coord+ip*3,
                                    non0table+ib*nbas,
                                    atm, natm, bas, nbas, env);
                }
        }
}

void VXCnr_ao_screen(char *non0table, double *coord, int ngrids, int blksize,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ib, i, j;
        int np, nc, atm_id, bas_id;
        double rr, arr, maxc;
        double logcoeff[NPRIMAX];
        double dr[3];
        double *p_exp, *pcoeff, *pcoord, *ratm;

        memset(non0table, 0, sizeof(char) * nblk*nbas);

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

