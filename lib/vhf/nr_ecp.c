/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "cint.h"
#include "fblas.h"

#define RADI_POWER      3
#define ECP_LMAX        4
#define CART_MAX        128 // ~ lmax = 14
#define THIRD2          .6666666666666666666666666666
#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
#define CART_CUM        (455+1) // upto l = 12

double CINTcommon_fac_sp(int);

static double _factorial[] = {
        1.0, 1.0, 2.0, 6.0, 24.,
        1.2e+2, 7.2e+2, 5.04e+3, 4.032e+4, 3.6288e+5,
        3.6288e+6, 3.99168e+7, 4.790016e+8, 6.2270208e+9, 8.71782912e+10,
        1.307674368e+12, 2.0922789888e+13, 3.55687428096e+14,
        6.402373705728e+15, 1.21645100408832e+17,
        2.43290200817664e+18, 5.109094217170944e+19,
        1.1240007277776077e+21, 2.5852016738884978e+22,
};

static double _factorial2[] = {
        1., 1., 2., 3., 8.,
        15., 48., 105., 384., 945.,
        3840., 10395., 46080., 135135., 645120.,
        2027025., 10321920., 34459425., 185794560., 654729075.,
        3715891200., 13749310575., 81749606400., 316234143225., 1961990553600.,
        7905853580625., 51011754393600., 213458046676875.,
        1428329123020800., 6190283353629376.,
        42849873690624000., 1.9189878396251069e+17,
        1.371195958099968e+18, 6.3326598707628524e+18,
        4.6620662575398912e+19, 2.2164309547669976e+20,
        1.6783438527143608e+21, 8.2007945326378929e+21,
        6.3777066403145712e+22, 3.1983098677287775e+23,
};
static double factorial2(int n)
{
        if (n < 0) {
                return 1;
        } else {
                return _factorial2[n];
        }
}

static double _binom[] = {1,
                          1, 1,
                          1, 2, 1,
                          1, 3, 3, 1,
                          1, 4, 6, 4, 1,
                          1, 5, 10, 10, 5, 1,
                          1, 6, 15, 20, 15, 6, 1,
                          1, 7, 21, 35, 35, 21, 7, 1,
                          1, 8, 28, 56, 70, 56, 28, 8, 1,
                          1, 9, 36, 84, 126, 126, 84, 36, 9, 1,};
static double binom(int n, int m)
{
        if (n < 10) {
                return _binom[n*(n+1)/2+m];
        } else {
                return _factorial[n] / (_factorial[m]*_factorial[n-m]);
        }
}

static int _cart_powxyz[] = {
        0, 0, 0, // s
        1, 0, 0,
        0, 1, 0,
        0, 0, 1, // p
        2, 0, 0,
        1, 1, 0,
        1, 0, 1,
        0, 2, 0,
        0, 1, 1,
        0, 0, 2, // d
        3, 0, 0,
        2, 1, 0,
        2, 0, 1,
        1, 2, 0,
        1, 1, 1,
        1, 0, 2,
        0, 3, 0,
        0, 2, 1,
        0, 1, 2,
        0, 0, 3, // f
        4, 0, 0, 3, 1, 0, 3, 0, 1, 2, 2, 0, 2, 1, 1,
        2, 0, 2, 1, 3, 0, 1, 2, 1, 1, 1, 2, 1, 0, 3,
        0, 4, 0, 0, 3, 1, 0, 2, 2, 0, 1, 3, 0, 0, 4, // g
        5, 0, 0, 4, 1, 0, 4, 0, 1, 3, 2, 0, 3, 1, 1,
        3, 0, 2, 2, 3, 0, 2, 2, 1, 2, 1, 2, 2, 0, 3,
        1, 4, 0, 1, 3, 1, 1, 2, 2, 1, 1, 3, 1, 0, 4,
        0, 5, 0, 0, 4, 1, 0, 3, 2, 0, 2, 3, 0, 1, 4,
        0, 0, 5,
        6, 0, 0, 5, 1, 0, 5, 0, 1, 4, 2, 0, 4, 1, 1,
        4, 0, 2, 3, 3, 0, 3, 2, 1, 3, 1, 2, 3, 0, 3,
        2, 4, 0, 2, 3, 1, 2, 2, 2, 2, 1, 3, 2, 0, 4,
        1, 5, 0, 1, 4, 1, 1, 3, 2, 1, 2, 3, 1, 1, 4,
        1, 0, 5, 0, 6, 0, 0, 5, 1, 0, 4, 2, 0, 3, 3,
        0, 2, 4, 0, 1, 5, 0, 0, 6,
        7, 0, 0, 6, 1, 0, 6, 0, 1, 5, 2, 0, 5, 1, 1,
        5, 0, 2, 4, 3, 0, 4, 2, 1, 4, 1, 2, 4, 0, 3,
        3, 4, 0, 3, 3, 1, 3, 2, 2, 3, 1, 3, 3, 0, 4,
        2, 5, 0, 2, 4, 1, 2, 3, 2, 2, 2, 3, 2, 1, 4,
        2, 0, 5, 1, 6, 0, 1, 5, 1, 1, 4, 2, 1, 3, 3,
        1, 2, 4, 1, 1, 5, 1, 0, 6, 0, 7, 0, 0, 6, 1,
        0, 5, 2, 0, 4, 3, 0, 3, 4, 0, 2, 5, 0, 1, 6,
        0, 0, 7,
        8, 0, 0, 7, 1, 0, 7, 0, 1, 6, 2, 0, 6, 1, 1,
        6, 0, 2, 5, 3, 0, 5, 2, 1, 5, 1, 2, 5, 0, 3,
        4, 4, 0, 4, 3, 1, 4, 2, 2, 4, 1, 3, 4, 0, 4,
        3, 5, 0, 3, 4, 1, 3, 3, 2, 3, 2, 3, 3, 1, 4,
        3, 0, 5, 2, 6, 0, 2, 5, 1, 2, 4, 2, 2, 3, 3,
        2, 2, 4, 2, 1, 5, 2, 0, 6, 1, 7, 0, 1, 6, 1,
        1, 5, 2, 1, 4, 3, 1, 3, 4, 1, 2, 5, 1, 1, 6,
        1, 0, 7, 0, 8, 0, 0, 7, 1, 0, 6, 2, 0, 5, 3,
        0, 4, 4, 0, 3, 5, 0, 2, 6, 0, 1, 7, 0, 0, 8,
        9, 0, 0, 8, 1, 0, 8, 0, 1, 7, 2, 0, 7, 1, 1,
        7, 0, 2, 6, 3, 0, 6, 2, 1, 6, 1, 2, 6, 0, 3,
        5, 4, 0, 5, 3, 1, 5, 2, 2, 5, 1, 3, 5, 0, 4,
        4, 5, 0, 4, 4, 1, 4, 3, 2, 4, 2, 3, 4, 1, 4,
        4, 0, 5, 3, 6, 0, 3, 5, 1, 3, 4, 2, 3, 3, 3,
        3, 2, 4, 3, 1, 5, 3, 0, 6, 2, 7, 0, 2, 6, 1,
        2, 5, 2, 2, 4, 3, 2, 3, 4, 2, 2, 5, 2, 1, 6,
        2, 0, 7, 1, 8, 0, 1, 7, 1, 1, 6, 2, 1, 5, 3,
        1, 4, 4, 1, 3, 5, 1, 2, 6, 1, 1, 7, 1, 0, 8,
        0, 9, 0, 0, 8, 1, 0, 7, 2, 0, 6, 3, 0, 5, 4,
        0, 4, 5, 0, 3, 6, 0, 2, 7, 0, 1, 8, 0, 0, 9,
        10, 0, 0, 9, 1, 0, 9, 0, 1, 8, 2, 0, 8, 1, 1,
        8, 0, 2, 7, 3, 0, 7, 2, 1, 7, 1, 2, 7, 0, 3,
        6, 4, 0, 6, 3, 1, 6, 2, 2, 6, 1, 3, 6, 0, 4,
        5, 5, 0, 5, 4, 1, 5, 3, 2, 5, 2, 3, 5, 1, 4,
        5, 0, 5, 4, 6, 0, 4, 5, 1, 4, 4, 2, 4, 3, 3,
        4, 2, 4, 4, 1, 5, 4, 0, 6, 3, 7, 0, 3, 6, 1,
        3, 5, 2, 3, 4, 3, 3, 3, 4, 3, 2, 5, 3, 1, 6,
        3, 0, 7, 2, 8, 0, 2, 7, 1, 2, 6, 2, 2, 5, 3,
        2, 4, 4, 2, 3, 5, 2, 2, 6, 2, 1, 7, 2, 0, 8,
        1, 9, 0, 1, 8, 1, 1, 7, 2, 1, 6, 3, 1, 5, 4,
        1, 4, 5, 1, 3, 6, 1, 2, 7, 1, 1, 8, 1, 0, 9,
        0, 10, 0, 0, 9, 1, 0, 8, 2, 0, 7, 3, 0, 6, 4,
        0, 5, 5, 0, 4, 6, 0, 3, 7, 0, 2, 8, 0, 1, 9,
        0, 0, 10,
        11, 0, 0, 10, 1, 0, 10, 0, 1, 9, 2, 0, 9, 1, 1,
        9, 0, 2, 8, 3, 0, 8, 2, 1, 8, 1, 2, 8, 0, 3,
        7, 4, 0, 7, 3, 1, 7, 2, 2, 7, 1, 3, 7, 0, 4,
        6, 5, 0, 6, 4, 1, 6, 3, 2, 6, 2, 3, 6, 1, 4,
        6, 0, 5, 5, 6, 0, 5, 5, 1, 5, 4, 2, 5, 3, 3,
        5, 2, 4, 5, 1, 5, 5, 0, 6, 4, 7, 0, 4, 6, 1,
        4, 5, 2, 4, 4, 3, 4, 3, 4, 4, 2, 5, 4, 1, 6,
        4, 0, 7, 3, 8, 0, 3, 7, 1, 3, 6, 2, 3, 5, 3,
        3, 4, 4, 3, 3, 5, 3, 2, 6, 3, 1, 7, 3, 0, 8,
        2, 9, 0, 2, 8, 1, 2, 7, 2, 2, 6, 3, 2, 5, 4,
        2, 4, 5, 2, 3, 6, 2, 2, 7, 2, 1, 8, 2, 0, 9,
        1, 10, 0, 1, 9, 1, 1, 8, 2, 1, 7, 3, 1, 6, 4,
        1, 5, 5, 1, 4, 6, 1, 3, 7, 1, 2, 8, 1, 1, 9,
        1, 0, 10, 0, 11, 0, 0, 10, 1, 0, 9, 2, 0, 8, 3,
        0, 7, 4, 0, 6, 5, 0, 5, 6, 0, 4, 7, 0, 3, 8,
        0, 2, 9, 0, 1, 10, 0, 0, 11,
        12, 0, 0, 11, 1, 0, 11, 0, 1, 10, 2, 0, 10, 1, 1,
        10, 0, 2, 9, 3, 0, 9, 2, 1, 9, 1, 2, 9, 0, 3,
        8, 4, 0, 8, 3, 1, 8, 2, 2, 8, 1, 3, 8, 0, 4,
        7, 5, 0, 7, 4, 1, 7, 3, 2, 7, 2, 3, 7, 1, 4,
        7, 0, 5, 6, 6, 0, 6, 5, 1, 6, 4, 2, 6, 3, 3,
        6, 2, 4, 6, 1, 5, 6, 0, 6, 5, 7, 0, 5, 6, 1,
        5, 5, 2, 5, 4, 3, 5, 3, 4, 5, 2, 5, 5, 1, 6,
        5, 0, 7, 4, 8, 0, 4, 7, 1, 4, 6, 2, 4, 5, 3,
        4, 4, 4, 4, 3, 5, 4, 2, 6, 4, 1, 7, 4, 0, 8,
        3, 9, 0, 3, 8, 1, 3, 7, 2, 3, 6, 3, 3, 5, 4,
        3, 4, 5, 3, 3, 6, 3, 2, 7, 3, 1, 8, 3, 0, 9,
        2, 10, 0, 2, 9, 1, 2, 8, 2, 2, 7, 3, 2, 6, 4,
        2, 5, 5, 2, 4, 6, 2, 3, 7, 2, 2, 8, 2, 1, 9,
        2, 0, 10, 1, 11, 0, 1, 10, 1, 1, 9, 2, 1, 8, 3,
        1, 7, 4, 1, 6, 5, 1, 5, 6, 1, 4, 7, 1, 3, 8,
        1, 2, 9, 1, 1, 10, 1, 0, 11, 0, 12, 0, 0, 11, 1,
        0, 10, 2, 0, 9, 3, 0, 8, 4, 0, 7, 5, 0, 6, 6,
        0, 5, 7, 0, 4, 8, 0, 3, 9, 0, 2, 10, 0, 1, 11,
        0, 0, 12,
};
static int _offset_cart[] = {0, 1, 4, 10, 20, 35, 56, 84, 120,
                             165, 220, 286, 364, 455, 560};
#define LOOP_CART(l, i, pxyz)   pxyz = _cart_powxyz + _offset_cart[l] * 3; \
                        for (i = 0; i < _offset_cart[l+1]-_offset_cart[l]; i++, pxyz+=3)
#define LOOP_XYZ(i, j, k, pxyz) \
        for (i = 0; i <= pxyz[0]; i++) \
        for (j = 0; j <= pxyz[1]; j++) \
        for (k = 0; k <= pxyz[2]; k++)


/*
 * exponentially scaled modified spherical Bessel function of the first kind
 *
 * JCC, 27, 1009
 */
void ECPsph_ine(double *out, int order, double z)
{
        int i, k;
        if (z < 1e-7) {
                // (1-z) * z^l / (2l+1)!!
                out[0] = 1. - z;
                for (i = 1; i <= order; i++) {
                        out[i] = out[i-1] * z / (i*2+1);
                }
        } else if (z > 16) {
                // R_l(z) = \sum_k (l+k)!/(k!(l-k)!(2x)^k)
                double z2 = -.5 / z;
                double ti;
                for (i = 0; i <= order; i++) {
                        ti = .5 / z;
                        out[i] = ti;
                        for (k = 1; k <= i; k++) {
                                ti *= z2;
                                out[i] += ti * _factorial[i+k]
                                        / (_factorial[k] * _factorial[i-k]);
                        }
                }
        } else {
                // z^l e^{-z} \sum (z^2/2)^k/(k!(2k+2l+1)!!)
                double z2 = .5 * z * z;
                double t0 = exp(-z);
                double ti, next;
                for (i = 0; i <= order; i++) {
                        ti = t0;
                        out[i] = ti;
                        for (k = 1;; k++) {
                                ti *= z2 / (k * (k*2+i*2+1));
                                next = out[i] + ti;
                                if (next == out[i]) {
                                        break;
                                } else {
                                        out[i] = next;
                                }
                        }
                        t0 *= z/(i*2+3);  // k = 0
                }
        }

}

void ECPsph_ine_a(double *out, int order, double *rs, int n)
{
        int i;
        for (i = 0; i < n; i++) {
                ECPsph_ine(out, order, rs[i]);
                out += order+1;
        }
}

void ECPgauss_chebyshev(double *rs, double *ws, int n)
{
        int i;
        double step = 1./(n+1);
        double fac = 16 * step / 3;
        double xinc = M_PI * step;
        double x1 = 0;
        double x2, x3, x4, xi;
        for (i = 0; i < n; i++) {
                x1 += xinc;
                x2 = sin(x1);
                x3 = sin(x1*2);
                x4 = x2 * x2;
                xi = (n-i*2-1) * step + M_1_PI * (1+THIRD2*x4) * x3;
                rs[i] = 1 - log(1+xi) * M_LOG2E;  // 1/ln2
                ws[i] = fac * x4 * x4 * M_LOG2E / (1+xi);
        }
}

void ECPrad_part(double *ur, double *rs, int nrs, int *ecpshls, int *ecpbas,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        double ubuf[nrs];
        double r2[nrs];
        double *ak, *ck;
        double *pur = ur;
        int npk;
        int ish, i, kp;

        for (i = 0; i < nrs; i++) {
                r2[i] = rs[i] * rs[i];
        }

        for (ish = 0; ecpshls[ish] != -1; ish++) {
                npk = ecpbas[ecpshls[ish]*BAS_SLOTS+NPRIM_OF];
                ak = env + ecpbas[ecpshls[ish]*BAS_SLOTS+PTR_EXP];
                ck = env + ecpbas[ecpshls[ish]*BAS_SLOTS+PTR_COEFF];

                for (i = 0; i < nrs; i++) {
                        pur[i] = ck[0] * exp(-ak[0]*r2[i]);
                }
                for (kp = 1; kp < npk; kp++) {
                        for (i = 0; i < nrs; i++) {
                                pur[i] += ck[kp] * exp(-ak[kp]*r2[i]);
                        }
                }
                switch (ecpbas[ecpshls[ish]*BAS_SLOTS+RADI_POWER]) {
                case 1:
                        for (i = 0; i < nrs; i++) {
                                pur[i] *= rs[i];
                        }
                        break;
                case 2:
                        for (i = 0; i < nrs; i++) {
                                pur[i] *= r2[i];
                        }
                        break;
                }

                if (ish == 0) {
                        pur = ubuf;
                } else {
                        for (i = 0; i < nrs; i++) {
                                ur[i] += pur[i];
                        }
                }
        }
}

static double int_unit_xyz(int i, int j, int k)
{
        if (i % 2 || j % 2 || k % 2) {
                return 0;
        } else {
                return (factorial2(i-1) * factorial2(j-1) *
                        factorial2(k-1) / factorial2(i+j+k+1));
        }
}
/*
 * Angular part integration then transform back to cartesian basis
 */
double *CINTc2s_bra_sph(double *gsph, int nket, double *gcart, int l);
double *CINTs2c_bra_sph(double *gsph, int nket, double *gcart, int l);
static void ang_nuc_in_cart(double *omega, int l, double *r)
{
        double buf[CART_MAX];
        double xx[16];
        double yy[16];
        double zz[16];
        int i, j, k, n;

        switch (l) {
        case 0:
                omega[0] = 0.07957747154594767;
                break;
        case 1:
                omega[0] = r[0] * 0.2387324146378430;
                omega[1] = r[1] * 0.2387324146378430;
                omega[2] = r[2] * 0.2387324146378430;
                break;
        default:
                xx[0] = 1;
                yy[0] = 1;
                zz[0] = 1;
                for (i = 1; i <= l; i++) {
                        xx[i] = xx[i-1] * r[0];
                        yy[i] = yy[i-1] * r[1];
                        zz[i] = zz[i-1] * r[2];
                }
                for (n = 0, i = l; i >= 0; i--) {
                        for (j = l-i; j >= 0; j--, n++) {
                                k = l - i - j;
                                omega[n] = xx[i] * yy[j] * zz[k];
                        }
                }
                CINTc2s_bra_sph(buf, 1, omega, l);
                CINTs2c_bra_sph(buf, 1, omega, l);
        }
}

static void cache_3dfac(double *facs, int l, double *r)
{
        int l1 = l + 1;
        double *facx = facs;
        double *facy = facs + l1*l1;
        double *facz = facy + l1*l1;
        double xx[16];
        double yy[16];
        double zz[16];
        double bfac;
        int i, j, off;
        xx[0] = 1;
        yy[0] = 1;
        zz[0] = 1;
        for (i = 1; i <= l; i++) {
                xx[i] = xx[i-1] * r[0];
                yy[i] = yy[i-1] * r[1];
                zz[i] = zz[i-1] * r[2];
        }
        for (i = 0; i <= l; i++) {
                for (j = 0; j <= i; j++) {
                        bfac = binom(i,j);
                        off = i*l1+j;
                        facx[off] = bfac * xx[i-j];
                        facy[off] = bfac * yy[i-j];
                        facz[off] = bfac * zz[i-j];
                }
        }
}

void type2_facs_ang(double *facs, int li, int lc, double *ri)
{
        double unitr[3];
        if (ri[0] == 0 && ri[1] == 0 && ri[2] == 0) {
                unitr[0] = 0;
                unitr[1] = 0;
                unitr[2] = 0;
        } else {
                double norm_ri = -1/sqrt(SQUARE(ri));
                unitr[0] = ri[0] * norm_ri;
                unitr[1] = ri[1] * norm_ri;
                unitr[2] = ri[2] * norm_ri;
        }

        const int li1 = li + 1;
        const int dlc = lc * 2 + 1;
        const int dlambda = li + lc + 1;
        double omega_nuc[CART_CUM];
        double *pnuc;
        int m, n, i, j, k, lmb, mi;
        for (i = 0; i <= li+lc; i++) {
                pnuc = omega_nuc + _offset_cart[i];
                ang_nuc_in_cart(pnuc, i, unitr);
        }
        for (i = 0; i < _offset_cart[li+lc+1]; i++) {
                omega_nuc[i] *= 4 * M_PI;
        }

        double buf[CART_MAX];
        double omega[li1*li1*li1*dlambda*dlc];
        double *pomega;
        int dlclmb = dlambda * dlc;
        int need_even;
        int need_odd;
        int *puvw, *prst;
        for (i = 0; i <= li; i++) {
        for (j = 0; j <= li-i; j++) {
        for (k = 0; k <= li-i-j; k++) {
                // use need_even to ensure (lc+a+b+c+lmb) is even
                need_even = (lc+i+j+k)%2;
                pomega = omega + (i*li1*li1+j*li1+k)*dlclmb+need_even*dlc;
                for (lmb = need_even; lmb <= li+lc; lmb+=2) {
                        pnuc = omega_nuc + _offset_cart[lmb];
                        LOOP_CART(lc, m, puvw) {
                                buf[m] = 0;
                                LOOP_CART(lmb, n, prst) {
                                        buf[m] += pnuc[n] *
                                                int_unit_xyz(i+puvw[0]+prst[0],
                                                             j+puvw[1]+prst[1],
                                                             k+puvw[2]+prst[2]);
                                }
                        }
                        switch (lc) {
                        case 0:
                                pomega[0] = buf[0] * 0.282094791773878143;
                                break;
                        case 1:
                                pomega[0] = buf[0] * 0.488602511902919921;
                                pomega[1] = buf[1] * 0.488602511902919921;
                                pomega[2] = buf[2] * 0.488602511902919921;
                                break;
                        default:
                                CINTc2s_bra_sph(pomega, 1, buf, lc);
                        }
                        pomega += dlc*2;
                }

                need_odd = need_even ^ 1;
                pomega = omega + (i*li1*li1+j*li1+k)*dlclmb+need_odd*dlc;
                for (lmb = need_odd; lmb <= li+lc; lmb+=2) {
                        for (m = 0; m < dlc; m++) {
                                pomega[m] = 0;
                        }
                        pomega += dlc*2;
                }
        } } }

        const int nfi = _offset_cart[li+1]-_offset_cart[li];//(li+1)*(li+2)/2;
        double fac3d[3*li1*li1];
        double *fac3dx = fac3d;
        double *fac3dy = fac3dx + li1*li1;
        double *fac3dz = fac3dy + li1*li1;
        double *pfacs;
        double fac;
        cache_3dfac(fac3d, li, ri);

        memset(facs, 0, sizeof(double)*li1*nfi*dlclmb);
        LOOP_CART(li, mi, prst) {
                LOOP_XYZ(i, j, k, prst) {
                        need_even = (lc+i+j+k)%2;
                        fac = fac3dx[prst[0]*li1+i] * fac3dy[prst[1]*li1+j] *
                              fac3dz[prst[2]*li1+k];
                        pomega = omega + (i*li1*li1+j*li1+k)*dlclmb;
                        pfacs = facs + ((i+j+k)*nfi+mi)*dlclmb;
                        for (m = 0; m < dlc; m++) {
                        for (n = need_even; n < dlambda; n+=2) {
                                pfacs[m*dlambda+n] += fac * pomega[n*dlc+m];
                        } }
                }
        }
}

void type2_facs_rad(double *facs, int ish, int lc, double rca,
                    double *rs, int nrs,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const int li = bas[ANG_OF  +ish*BAS_SLOTS];
        const int np = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int nc = bas[NCTR_OF +ish*BAS_SLOTS];
        const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
        const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
        int ip, i, j;
        double ka;
        double r2[nrs];
        double buf[np*nrs*(li+lc+1)];
        double t1;
        double *pbuf = buf;

        for (i = 0; i < nrs; i++) {
                t1 = rs[i] - rca;
                r2[i] = -t1 * t1;
        }
        for (ip = 0; ip < np; ip++) {
                ka = 2 * ai[ip] * rca;
                for (i = 0; i < nrs; i++) {
                        t1 = exp(r2[i] * ai[ip]);
                        ECPsph_ine(pbuf, li+lc, ka*rs[i]);
                        for (j = 0; j <= li+lc; j++) {
                                pbuf[j] *= t1;
                        }
                        pbuf += li + lc + 1;
                }
        }

        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int m = nrs * (li+lc+1);
        dgemm_(&TRANS_N, &TRANS_N, &m, &nc, &np,
               &D1, buf, &m, ci, &np, &D0, facs, &m);
}

void type1_static_facs(double *facs, int li, double *ri)
{
        const int d1 = li + 1;
        const int d2 = d1 * d1;
        const int d3 = d2 * d1;
        double fac3d[3*d1*d1];
        double *fac3dx = fac3d;
        double *fac3dy = fac3dx + d1*d1;
        double *fac3dz = fac3dy + d1*d1;
        double *pfacs;
        cache_3dfac(fac3d, li, ri);
        int mi, i, j, k;
        int *pxyz;

        LOOP_CART(li, mi, pxyz) {
                pfacs = facs + mi * d3;
                LOOP_XYZ(i, j, k, pxyz) {
                        pfacs[i*d2+j*d1+k] = fac3dx[pxyz[0]*d1+i]
                                           * fac3dy[pxyz[1]*d1+j]
                                           * fac3dz[pxyz[2]*d1+k];
                }
        }
}

void type1_rad_ang(double *rad_ang, int lmax, double *r, double *rad_all)
{
        double unitr[3];
        if (r[0] == 0 && r[1] == 0 && r[2] == 0) {
                unitr[0] = 0;
                unitr[1] = 0;
                unitr[2] = 0;
        } else {
                double norm_r = -1/sqrt(SQUARE(r));
                unitr[0] = r[0] * norm_r;
                unitr[1] = r[1] * norm_r;
                unitr[2] = r[2] * norm_r;
        }

        double omega_nuc[CART_CUM];
        double *pnuc;
        int n, i, j, k, lmb;
        for (i = 0; i <= lmax; i++) {
                pnuc = omega_nuc + _offset_cart[i];
                ang_nuc_in_cart(pnuc, i, unitr);
        }
//        for (i = 0; i < _offset_cart[lmax+1]; i++) {
//                omega_nuc[i] *= 4 * M_PI;
//        }

        const int d1 = lmax + 1;
        const int d2 = d1 * d1;
        const int d3 = d2 * d1;
        int need_even;
        double tmp;
        int *prst;
        double *pout, *prad;
        memset(rad_ang, 0, sizeof(double)*d3);
        for (i = 0; i <= lmax; i++) {
        for (j = 0; j <= lmax-i; j++) {
        for (k = 0; k <= lmax-i-j; k++) {
                pout = rad_ang + i*d2+j*d1+k;
                prad = rad_all + (i+j+k)*d1;
                // need_even to ensure (a+b+c+lmb) is even
                need_even = (i+j+k)%2;
                for (lmb = need_even; lmb <= lmax; lmb+=2) {
                        tmp = 0;
                        pnuc = omega_nuc + _offset_cart[lmb];
                        LOOP_CART(lmb, n, prst) {
                                tmp += pnuc[n] * int_unit_xyz(i+prst[0],
                                                              j+prst[1],
                                                              k+prst[2]);
                        }
                        *pout += prad[lmb] * tmp;
                }
        } } }
}

static void search_ecpatms(int *ecpatmlst, int natm, int *ecpbas, int necpbas)
{
        int mask[natm];
        int i, k;
        memset(mask, 0, sizeof(int)*natm);
        for (i = 0; i < necpbas; i++) {
                mask[ecpbas[ATOM_OF+i*BAS_SLOTS]] = 1;
        }
        for (k = 0, i = 0; i < natm; i++) {
                if (mask[i]) {
                        ecpatmlst[k] = i;
                        k++;
                }
        }
        ecpatmlst[k] = -1;
}

static void search_ecpshls(int *ecpshls, int atm_id, int lc,
                           int *ecpbas, int necpbas)
{
        int i, k;
        for (k = 0, i = 0; i < necpbas; i++) {
                if (ecpbas[ATOM_OF+i*BAS_SLOTS] == atm_id &&
                    ecpbas[ANG_OF +i*BAS_SLOTS] == lc) {
                        ecpshls[k] = i;
                        k++;
                }
        }
        ecpshls[k] = -1;
}

int ECPtype2_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                  int *atm, int natm, int *bas, int nbas, double *env,
                  void *opt)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const int di = nfi * nci;
        const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

        const int nrs = 500;
        double rs[nrs];
        double ws[nrs];
        ECPgauss_chebyshev(rs, ws, nrs);

        int ecpatmlst[natm+1];
        search_ecpatms(ecpatmlst, natm, ecpbas, necpbas);

        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double common_fac = CINTcommon_fac_sp(li) *
                                  CINTcommon_fac_sp(lj) * 16 * M_PI * M_PI;
        int ecpshls[necpbas+1];
        int ia, atm_id, lc, lab, lilc1, ljlc1, dlc, im, mq;
        int i, j, n, ic, jc;
        double ur[nrs];
        double rur[nrs*(li+lj+1)];
        double radi[nci*(li+ECP_LMAX+1)*nrs];
        double radj[ncj*(lj+ECP_LMAX+1)*nrs];
        double angi[(li+1)*nfi*(ECP_LMAX*2+1)*(li+ECP_LMAX+1)];
        double angj[(lj+1)*nfj*(ECP_LMAX*2+1)*(lj+ECP_LMAX+1)];
        double rad_all[(li+lj+1)*(li+ECP_LMAX+1)*(lj+ECP_LMAX+1)];
        double rca[3];
        double rcb[3];
        double buf[nfi*(ECP_LMAX*2+1)*(lj+ECP_LMAX+1)];
        double dca, dcb;
        double *rc, *prad, *pradi, *pradj, *prur;

        memset(gctr, 0, sizeof(double)*nci*ncj*nfi*nfj);

        for (ia = 0; ecpatmlst[ia] != -1; ia++) {
                atm_id = ecpatmlst[ia];
                rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                for (lc = 0; lc <= ECP_LMAX; lc++) { // upto g function
                        search_ecpshls(ecpshls, atm_id, lc, ecpbas, necpbas);
                        if (ecpshls[0] == -1) {
                                continue;
                        }

        rca[0] = rc[0] - ri[0];
        rca[1] = rc[1] - ri[1];
        rca[2] = rc[2] - ri[2];
        rcb[0] = rc[0] - rj[0];
        rcb[1] = rc[1] - rj[1];
        rcb[2] = rc[2] - rj[2];
        dca = sqrt(SQUARE(rca));
        dcb = sqrt(SQUARE(rcb));
        ECPrad_part(ur, rs, nrs, ecpshls, ecpbas, atm, natm, bas, nbas, env);
        for (i = 0; i < nrs; i++) {
                ur[i] *= ws[i];
                rur[i] = ur[i];
                for (lab = 1; lab <= li+lj; lab++) {
                        rur[nrs*lab+i] = rur[nrs*(lab-1)+i] * rs[i];
                }
        }

        type2_facs_rad(radi, ish, lc, dca, rs, nrs, atm, natm, bas, nbas, env);
        type2_facs_rad(radj, jsh, lc, dcb, rs, nrs, atm, natm, bas, nbas, env);
        type2_facs_ang(angi, li, lc, rca);
        type2_facs_ang(angj, lj, lc, rcb);

        dlc = lc * 2 + 1;
        lilc1 = li + lc + 1;
        ljlc1 = lj + lc + 1;
        im = nfi * dlc;
        mq = dlc * ljlc1;
        for (ic = 0; ic < nci; ic++) {
        for (jc = 0; jc < ncj; jc++) {
                pradi = radi + ic * nrs * lilc1;
                pradj = radj + jc * nrs * ljlc1;
                for (lab = 0; lab <= li+lj; lab++) {
                        prur = rur + lab * nrs;
                        prad = rad_all + lab*lilc1*ljlc1;
                        memset(prad, 0, sizeof(double)*lilc1*ljlc1);
                        for (n = 0; n < nrs; n++) {
                        for (i = 0; i < lilc1; i++) {
                        for (j = 0; j < ljlc1; j++) {
                                prad[i*ljlc1+j] += prur[n] *
                                        pradi[n*lilc1+i] * pradj[n*ljlc1+j];
                        } } }
                }

                for (i = 0; i <= li; i++) {
                for (j = 0; j <= lj; j++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ljlc1, &im, &lilc1,
                               &D1, rad_all+(i+j)*lilc1*ljlc1, &ljlc1,
                               angi+i*nfi*dlc*lilc1, &lilc1, &D0, buf, &ljlc1);
                        dgemm_(&TRANS_T, &TRANS_N, &nfi, &nfj, &mq,
                               &common_fac, buf, &mq, angj+j*nfj*dlc*ljlc1, &mq,
                               &D1, gctr+jc*nfj*di+ic*nfi, &di);
                } }
        } }
                }
        }
        return 1;
}

static void scale_coeff(double *cei, const double *ci, const double *ai,
                        const double r2ca, const int npi, const int nci, const int li)
{
        int ip, ic;
        double tmp;
        double common_fac = CINTcommon_fac_sp(li) * 4 * M_PI;
        for (ip = 0; ip < npi; ip++) {
                tmp = exp(-ai[ip] * r2ca) * common_fac;
                for (ic = 0; ic < nci; ic++) {
                        cei[ic*npi+ip] = ci[ic*npi+ip] * tmp;
                }
        }
}

void type1_rad_part(double *rad_all, int lmax, double k, double aij,
                    double *ur, double *rs, int nrs)
{
        const int lmax1 = lmax + 1;
        double rur[nrs];
        double bval[nrs*lmax1];
        int lab, i, n;
        double kaij, fac, tmp;
        double *prad;

        memset(rad_all, 0, sizeof(double)*lmax1*lmax1);

        kaij = k / (2*aij);
        fac = exp(kaij*kaij*aij);
        for (n = 0; n < nrs; n++) {
                tmp = rs[n] - kaij;
                rur[n] = ur[n] * exp(-aij*tmp*tmp) * fac;
                ECPsph_ine(bval+n*lmax1, lmax, k*rs[n]);
        }

        for (lab = 0; lab <= lmax; lab++) {
                if (lab > 0) {
                        for (n = 0; n < nrs; n++) {
                                rur[n] *= rs[n];
                        }
                }
                prad = rad_all + lab * lmax1;
                for (i = lab%2; i <= lmax; i+=2) {
                        for (n = 0; n < nrs; n++) {
                                prad[i] += rur[n] * bval[n*lmax1+i];
                        }
                }
        }
}

int ECPtype1_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                  int *atm, int natm, int *bas, int nbas, double *env,
                  void *opt)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
        const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
        const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
        const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
        const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

        const int nrs = 500;
        double rs[nrs];
        double ws[nrs];
        ECPgauss_chebyshev(rs, ws, nrs);

        int ecpatmlst[natm+1];
        search_ecpatms(ecpatmlst, natm, ecpbas, necpbas);

        const int lilj1 = li + lj + 1;
        const int d1 = lilj1;
        const int d2 = d1 * d1;
        const int d3 = d2 * d1;
        const int di1 = li + 1;
        const int di2 = di1 * di1;
        const int di3 = di2 * di1;
        const int dj1 = lj + 1;
        const int dj2 = dj1 * dj1;
        const int dj3 = dj2 * dj1;
        int ecpshls[necpbas+1];
        int ia, atm_id;
        int n, ip, jp, ic, jc, mi, mj;
        int i1, i2, i3, j1, j2, j3;
        int *ixyz, *jxyz;
        double ur[nrs];
        double rad_all[d2];
        double rad_ang[d3];
        double rad_ang_all[nci*ncj*d3];
        double ifac[nfi*di3];
        double jfac[nfj*dj3];
        double cei[npi*nci];
        double cej[npj*ncj];
        double rca[3];
        double rcb[3];
        double rij[3];
        double r2ca, r2cb, dca, dcb, fac;
        double *rc, *prad, *pifac, *pjfac, *pout;

        memset(gctr, 0, sizeof(double)*nci*ncj*nfi*nfj);
        memset(rad_ang, 0, sizeof(double)*d3);

        for (ia = 0; ecpatmlst[ia] != -1; ia++) {
                atm_id = ecpatmlst[ia];
                search_ecpshls(ecpshls, atm_id, -1, ecpbas, necpbas);
                if (ecpshls[0] == -1) {
                        continue;
                }

        rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
        rca[0] = rc[0] - ri[0];
        rca[1] = rc[1] - ri[1];
        rca[2] = rc[2] - ri[2];
        rcb[0] = rc[0] - rj[0];
        rcb[1] = rc[1] - rj[1];
        rcb[2] = rc[2] - rj[2];
        r2ca = SQUARE(rca);
        r2cb = SQUARE(rcb);
        dca = sqrt(r2ca);
        dcb = sqrt(r2cb);
        scale_coeff(cei, ci, ai, r2ca, npi, nci, li);
        scale_coeff(cej, cj, aj, r2cb, npj, ncj, lj);

        ECPrad_part(ur, rs, nrs, ecpshls, ecpbas, atm, natm, bas, nbas, env);
        for (n = 0; n < nrs; n++) {
                ur[n] *= ws[n];
        }
        memset(rad_ang_all, 0, sizeof(double)*nci*ncj*d3);

        for (ip = 0; ip < npi; ip++) {
        for (jp = 0; jp < npj; jp++) {
                rij[0] = ai[ip] * rca[0] + aj[jp] * rcb[0];
                rij[1] = ai[ip] * rca[1] + aj[jp] * rcb[1];
                rij[2] = ai[ip] * rca[2] + aj[jp] * rcb[2];
                type1_rad_part(rad_all, li+lj, sqrt(SQUARE(rij))*2,
                               ai[ip]+aj[jp], ur, rs, nrs);
                type1_rad_ang(rad_ang, li+lj, rij, rad_all);
                for (ic = 0; ic < nci; ic++) {
                for (jc = 0; jc < ncj; jc++) {
                        fac = cei[ic*npi+ip] * cej[jc*npj+jp];
                        prad = rad_ang_all + (ic*ncj+jc)*d3;
                        for (n = 0; n < d3; n++) {
                                prad[n] += fac * rad_ang[n];
                        }
                } }
        } }

        type1_static_facs(ifac, li, rca);
        type1_static_facs(jfac, lj, rcb);
        for (ic = 0; ic < nci; ic++) {
        for (jc = 0; jc < ncj; jc++) {
                prad = rad_ang_all + (ic*ncj+jc)*d3;
                LOOP_CART(li, mi, ixyz) {
                LOOP_CART(lj, mj, jxyz) {
                        pifac = ifac + mi * di3;
                        pjfac = jfac + mj * dj3;
                        pout = gctr + (jc*nfj+mj) * nci*nfi + ic*nfi+mi;
                        LOOP_XYZ(i1, i2, i3, ixyz) {
                        LOOP_XYZ(j1, j2, j3, jxyz) {
                                *pout += pifac[i1*di2+i2*di1+i3] *
                                         pjfac[j1*dj2+j2*dj1+j3] *
                                         prad[(i1+j1)*d2+(i2+j2)*d1+i3+j3];
                        } }
                } }
        } }

        }
        return 1;
}

static int c2s_factory(double *gctr, int *shls, int *ecpbas, int necpbas,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       void *opt, int (*fcart)())
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        int has_value;

        if (li < 2 && lj < 2) {
                return (*fcart)(gctr, shls, ecpbas, necpbas,
                                atm, natm, bas, nbas, env, opt);
        }

        int j;
        int di = nfi * nci;
        int dji = di * (lj*2+1);
        double *gcart, *gtmp;
        gcart = malloc(sizeof(double) * nfi*nfj*nci*ncj*2);
        gtmp = gcart + nfi*nfj*nci*ncj;
        has_value = (*fcart)(gcart, shls, ecpbas, necpbas,
                             atm, natm, bas, nbas, env, opt);

        if (li < 2) {
                for (j = 0; j < ncj; j++) {
                        CINTc2s_ket_sph(gctr+j*dji, di, gcart+j*nfj*di, lj);
                }
        } else if (lj < 2 ) {
                CINTc2s_bra_sph(gctr, (lj*2+1)*nci*ncj, gcart, li);
        } else {
                for (j = 0; j < ncj; j++) {
                        CINTc2s_ket_sph(gtmp+j*dji, di, gcart+j*nfj*di, lj);
                }
                CINTc2s_bra_sph(gctr, (lj*2+1)*nci*ncj, gtmp, li);
        }
        free(gcart);
        return has_value;
}

int ECPtype1_sph(double *gctr, int *shls, int *ecpbas, int necpbas,
                 int *atm, int natm, int *bas, int nbas, double *env,
                 void *opt)
{
        return c2s_factory(gctr, shls, ecpbas, necpbas,
                           atm, natm, bas, nbas, env, opt, ECPtype1_cart);
}

int ECPtype2_sph(double *gctr, int *shls, int *ecpbas, int necpbas,
                 int *atm, int natm, int *bas, int nbas, double *env,
                 void *opt)
{
        return c2s_factory(gctr, shls, ecpbas, necpbas,
                           atm, natm, bas, nbas, env, opt, ECPtype2_cart);
}

