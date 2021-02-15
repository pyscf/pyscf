/*
 * Modified based on rotate.f from MOPAC 7-1.11
 */

#include <math.h>

extern int MOPAC_NATORB[107];
extern double MOPAC_CORE[107];

void MOPAC_repp(int ni, int nj, double rij, double ev, double *ri, double *core,
                double *dd, double *qq, double *am, double *ad, double *aq);

static double boron1[] = {
        .182613, .118587, -.07328, 0., 0., 0., 0., 0., 0., 0.,
        .412253,-.149917, 0.     , 0., 0., 0., 0., 0., 0., 0.,
        .261751, .050275, 0.     , 0., 0., 0., 0., 0., 0., 0.,
        .359244, .074729, 0.     , 0., 0., 0., 0., 0., 0., 0.,
};
static double boron2[] = {
        6. , 6., 5., 0., 0., 0., 0., 0., 0., 0.,
        10., 6., 0., 0., 0., 0., 0., 0., 0., 0.,
        8. , 5., 0., 0., 0., 0., 0., 0., 0., 0.,
        9. , 9., 0., 0., 0., 0., 0., 0., 0., 0.,
};
static double boron3[] = {
        0.727592, 1.466639, 1.570975, 0., 0., 0., 0., 0., 0., 0.,
        0.832586, 1.18622 , 0.      , 0., 0., 0., 0., 0., 0., 0.,
        1.063995, 1.936492, 0.      , 0., 0., 0., 0., 0., 0., 0.,
        0.819351, 1.574414, 0.      , 0., 0., 0., 0., 0., 0., 0.,
};

#define AM1_MODEL 2
#define PM3_MODEL 3

static void _set_fn_ptr(double **pfn1, double **pfn2, double **pfn3,
                        double *fn1, double *fn2, double *fn3,
                        int model, int atom_type, int atom_other)
{
/*   LOAD IN AM1 BORON GAUSSIANS */
    if (model == AM1_MODEL && atom_type == 5) { // Boron
        switch (atom_other) {
        case 1:
            *pfn1 = boron1 + 10;
            *pfn2 = boron2 + 10;
            *pfn3 = boron3 + 10;
            break;
        case 6:
            *pfn1 = boron1 + 20;
            *pfn2 = boron2 + 20;
            *pfn3 = boron3 + 20;
            break;
        case 9: case 17: case 35: case 53:
            *pfn1 = boron1 + 30;
            *pfn2 = boron2 + 30;
            *pfn3 = boron3 + 30;
            break;
        default:
            *pfn1 = boron1;
            *pfn2 = boron2;
            *pfn3 = boron3;
        }
    } else {
        *pfn1 = fn1 + atom_type * 10;
        *pfn2 = fn2 + atom_type * 10;
        *pfn3 = fn3 + atom_type * 10;
    }
}

void MOPAC_rotate(int ni, int nj, double *xi, double *xj, double *w,
                  double *e1b, double *e2a, double *enuc, double *alp,
                  double *dd, double *qq, double *am, double *ad, double *aq,
                  double *fn1, double *fn2, double *fn3, int model)
{
    double d__1;
    double a;
    int i, ig;
    double ri[22];
    int si, sj;
    double ax;
    int nt;
    double gam, rij, xx11, xx21, xx22, xx31, xx32, xx33, yy11, yy21,
           yy22, zz11, zz21, zz22, zz31, zz32, zz33, xy11, xy21, xy22, xy31,
           xy32, xz11, xz21, xz22, xz31, xz32, xz33, yz11, yz21, yz22, yz31,
           yz32;
    double scale;
    double yyzz11, yyzz21, yyzz22;

/* *********************************************************************** */

/* ..IMPROVED SCALAR VERSION */
/* ..WRITTEN BY ERNEST R. DAVIDSON, INDIANA UNIVERSITY. */


/*   ROTATE CALCULATES THE TWO-PARTICLE INTERACTIONS. */

/*   ON INPUT  NI     = ATOMIC NUMBER OF FIRST ATOM. */
/*             NJ     = ATOMIC NUMBER OF SECOND ATOM. */
/*             XI     = COORDINATE OF FIRST ATOM. */
/*             XJ     = COORDINATE OF SECOND ATOM. */

/* ON OUTPUT W      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS. */
/*           E1B,E2A= ARRAY OF ELECTRON-NUCLEAR ATTRACTION INTEGRALS, */
/*                    E1B = ELECTRON ON ATOM NI ATTRACTING NUCLEUS OF NJ. */
/*           ENUC   = NUCLEAR-NUCLEAR REPULSION TERM. */


/* *** THIS ROUTINE COMPUTES THE REPULSION AND NUCLEAR ATTRACTION */
/*     INTEGRALS OVER MOLECULAR-FRAME COORDINATES.  THE INTEGRALS OVER */
/*     LOCAL FRAME COORDINATES ARE EVALUATED BY SUBROUTINE REPP AND */
/*     STORED AS FOLLOWS (WHERE P-SIGMA = O,   AND P-PI = P AND P* ) */
/*     IN RI */
/*     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5, */
/*     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10, */
/*     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15, */
/*     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20, */
/*     (PP/P*P*)=21,   (P*P/P*P)=22. */

/* *********************************************************************** */
    /* Parameter adjustments */
    --e2a;
    --e1b;
    --w;

    /* Function Body */
    double x[3];
    double y[3];
    double z[3];
    x[0] = xi[0] - xj[0];
    x[1] = xi[1] - xj[1];
    x[2] = xi[2] - xj[2];
    rij = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];

    if (rij < 2e-5) {

        for (i = 1; i <= 10; ++i) {
            e1b[i] = 0.;
            e2a[i] = 0.;
/* L10: */
        }
        *enuc = 0.;

    } else {

/*     MNDO AND AM1 CASES */

/* *** THE REPULSION INTEGRALS OVER MOLECULAR FRAME (W) ARE STORED IN THE */
/*     ORDER IN WHICH THEY WILL LATER BE USED.  IE.  (I,J/K,L) WHERE */
/*     J.LE.I  AND  L.LE.K     AND L VARIES MOST RAPIDLY AND I LEAST */
/*     RAPIDLY.  (ANTI-NORMAL COMPUTER STORAGE) */

        rij = sqrt(rij);

/* *** COMPUTE INTEGRALS IN DIATOMIC FRAME */

        double ccore[8];
        double unit = 1.;  // in a.u. was eV (27.21) in MOPAC
        MOPAC_repp(ni, nj, rij, unit, ri, ccore, dd, qq, am, ad, aq);
        double css1  = ccore[0];
        double csp1  = ccore[1];
        double cpps1 = ccore[2];
        double cppp1 = ccore[3];
        double css2  = ccore[4];
        double csp2  = ccore[5];
        double cpps2 = ccore[6];
        double cppp2 = ccore[7];

        gam = ri[0];
        a = 1. / rij;
        x[0] *= a;
        x[1] *= a;
        x[2] *= a;
        if (fabs(x[2]) > .99999999) {
            y[0] = 0.;
            y[1] = 1.;
            y[2] = 0.;
            z[0] = 1.;
            z[1] = 0.;
            z[2] = 0.;
        } else {
            z[2] = sqrt(1. - x[2] * x[2]);
            a = 1. / z[2];
            if (x[0] > 0) {
                y[0] = -a * x[1];
            } else {
                y[0] = a * x[1];
            }
            y[1] = fabs(a * x[0]);
            y[2] = 0.;
            z[0] = -a * x[0] * x[2];
            z[1] = -a * x[1] * x[2];
        }
        si = MOPAC_NATORB[ni] > 1;
        sj = MOPAC_NATORB[nj] > 1;
        if (si || sj) {
            xx11 = x[0] * x[0];
            xx21 = x[1] * x[0];
            xx22 = x[1] * x[1];
            xx31 = x[2] * x[0];
            xx32 = x[2] * x[1];
            xx33 = x[2] * x[2];
            yy11 = y[0] * y[0];
            yy21 = y[1] * y[0];
            yy22 = y[1] * y[1];
            zz11 = z[0] * z[0];
            zz21 = z[1] * z[0];
            zz22 = z[1] * z[1];
            zz31 = z[2] * z[0];
            zz32 = z[2] * z[1];
            zz33 = z[2] * z[2];
            yyzz11 = yy11 + zz11;
            yyzz21 = yy21 + zz21;
            yyzz22 = yy22 + zz22;
            xy11 = x[0] * 2. * y[0];
            xy21 = x[0] * y[1] + x[1] * y[0];
            xy22 = x[1] * 2. * y[1];
            xy31 = x[2] * y[0];
            xy32 = x[2] * y[1];
            xz11 = x[0] * 2. * z[0];
            xz21 = x[0] * z[1] + x[1] * z[0];
            xz22 = x[1] * 2. * z[1];
            xz31 = x[0] * z[2] + x[2] * z[0];
            xz32 = x[1] * z[2] + x[2] * z[1];
            xz33 = x[2] * 2. * z[2];
            yz11 = y[0] * 2. * z[0];
            yz21 = y[0] * z[1] + y[1] * z[0];
            yz22 = y[1] * 2. * z[1];
            yz31 = y[0] * z[2];
            yz32 = y[1] * z[2];
        }

/*     (S S/S S) */
        w[1] = ri[0];
        if (sj) {
/*     (S S/PX S) */
            w[2] = ri[4] * x[0];
/*     (S S/PX PX) */
            w[3] = ri[10] * xx11 + ri[11] * yyzz11;
/*     (S S/PY S) */
            w[4] = ri[4] * x[1];
/*     (S S/PY PX) */
            w[5] = ri[10] * xx21 + ri[11] * yyzz21;
/*     (S S/PY PY) */
            w[6] = ri[10] * xx22 + ri[11] * yyzz22;
/*     (S S/PZ S) */
            w[7] = ri[4] * x[2];
/*     (S S/PZ PX) */
            w[8] = ri[10] * xx31 + ri[11] * zz31;
/*     (S S/PZ PY) */
            w[9] = ri[10] * xx32 + ri[11] * zz32;
/*     (S S/PZ PZ) */
            w[10] = ri[10] * xx33 + ri[11] * zz33;
        }

        if (si) {
/*     (PX S/S S) */
            w[11] = ri[1] * x[0];
            if (sj) {
/*     (PX S/PX S) */
                w[12] = ri[5] * xx11 + ri[6] * yyzz11;
/*     (PX S/PX PX) */
                w[13] = x[0] * (ri[12] * xx11 + ri[13] * yyzz11) + ri[14] * (y[0] * xy11 + z[0] * xz11);
/*     (PX S/PY S) */
                w[14] = ri[5] * xx21 + ri[6] * yyzz21;
/*     (PX S/PY PX) */
                w[15] = x[0] * (ri[12] * xx21 + ri[13] * yyzz21) + ri[14] * (y[0] * xy21 + z[0] * xz21);
/*     (PX S/PY PY) */
                w[16] = x[0] * (ri[12] * xx22 + ri[13] * yyzz22) + ri[14] * (y[0] * xy22 + z[0] * xz22);
/*     (PX S/PZ S) */
                w[17] = ri[5] * xx31 + ri[6] * zz31;
/*     (PX S/PZ PX) */
                w[18] = x[0] * (ri[12] * xx31 + ri[13] * zz31) + ri[14] * (y[0] * xy31 + z[0] * xz31);
/*     (PX S/PZ PY) */
                w[19] = x[0] * (ri[12] * xx32 + ri[13] * zz32) + ri[14] * (y[0] * xy32 + z[0] * xz32);
/*     (PX S/PZ PZ) */
                w[20] = x[0] * (ri[12] * xx33 + ri[13] * zz33) + ri[14] * (z[0] * xz33);
/*     (PX PX/S S) */
                w[21] = ri[2] * xx11 + ri[3] * yyzz11;
/*     (PX PX/PX S) */
                w[22] = x[0] * (ri[7] * xx11 + ri[8] * yyzz11) + ri[9] * (y[0] * xy11 + z[0] * xz11);
/*     (PX PX/PX PX) */
                w[23] = (ri[15] * xx11 + ri[16] * yyzz11) * xx11 + ri[17] *
                        xx11 * yyzz11 + ri[18] * (yy11 * yy11 + zz11 * zz11)
                        + ri[19] * (xy11 * xy11 + xz11 * xz11) + ri[20] * (
                        yy11 * zz11 + zz11 * yy11) + ri[21] * yz11 * yz11;
/*     (PX PX/PY S) */
                w[24] = x[1] * (ri[7] * xx11 + ri[8] * yyzz11) + ri[9] * (y[1] * xy11 + z[1] * xz11);
/*     (PX PX/PY PX) */
                w[25] = (ri[15] * xx11 + ri[16] * yyzz11) * xx21 + ri[17] *
                        xx11 * yyzz21 + ri[18] * (yy11 * yy21 + zz11 * zz21)
                        + ri[19] * (xy11 * xy21 + xz11 * xz21) + ri[20] * (
                        yy11 * zz21 + zz11 * yy21) + ri[21] * yz11 * yz21;
/*     (PX PX/PY PY) */
                w[26] = (ri[15] * xx11 + ri[16] * yyzz11) * xx22 + ri[17] *
                        xx11 * yyzz22 + ri[18] * (yy11 * yy22 + zz11 * zz22)
                        + ri[19] * (xy11 * xy22 + xz11 * xz22) + ri[20] * (
                        yy11 * zz22 + zz11 * yy22) + ri[21] * yz11 * yz22;
/*     (PX PX/PZ S) */
                w[27] = x[2] * (ri[7] * xx11 + ri[8] * yyzz11) + ri[9] * (z[2] * xz11);
/*     (PX PX/PZ PX) */
                w[28] = (ri[15] * xx11 + ri[16] * yyzz11) * xx31 + (ri[17] *
                        xx11 + ri[18] * zz11 + ri[20] * yy11) * zz31 + ri[19]
                        * (xy11 * xy31 + xz11 * xz31) + ri[21] * yz11 * yz31;
/*     (PX PX/PZ PY) */
                w[29] = (ri[15] * xx11 + ri[16] * yyzz11) * xx32 + (ri[17] *
                        xx11 + ri[18] * zz11 + ri[20] * yy11) * zz32 + ri[19]
                        * (xy11 * xy32 + xz11 * xz32) + ri[21] * yz11 * yz32;
/*     (PX PX/PZ PZ) */
                w[30] = (ri[15] * xx11 + ri[16] * yyzz11) * xx33 + (ri[17] *
                        xx11 + ri[18] * zz11 + ri[20] * yy11) * zz33 + ri[19]
                        * xz11 * xz33;
/*     (PY S/S S) */
                w[31] = ri[1] * x[1];
/*     (PY S/PX S) */
                w[32] = ri[5] * xx21 + ri[6] * yyzz21;
/*     (PY S/PX PX) */
                w[33] = x[1] * (ri[12] * xx11 + ri[13] * yyzz11) + ri[14] * (y[1] * xy11 + z[1] * xz11);
/*     (PY S/PY S) */
                w[34] = ri[5] * xx22 + ri[6] * yyzz22;
/*     (PY S/PY PX) */
                w[35] = x[1] * (ri[12] * xx21 + ri[13] * yyzz21) + ri[14] * (y[1] * xy21 + z[1] * xz21);
/*     (PY S/PY PY) */
                w[36] = x[1] * (ri[12] * xx22 + ri[13] * yyzz22) + ri[14] * (y[1] * xy22 + z[1] * xz22);
/*     (PY S/PZ S) */
                w[37] = ri[5] * xx32 + ri[6] * zz32;
/*     (PY S/PZ PX) */
                w[38] = x[1] * (ri[12] * xx31 + ri[13] * zz31) + ri[14] * (y[1] * xy31 + z[1] * xz31);
/*     (PY S/PZ PY) */
                w[39] = x[1] * (ri[12] * xx32 + ri[13] * zz32) + ri[14] * (y[1] * xy32 + z[1] * xz32);
/*     (PY S/PZ PZ) */
                w[40] = x[1] * (ri[12] * xx33 + ri[13] * zz33) + ri[14] * (z[1] * xz33);
/*     (PY PX/S S) */
                w[41] = ri[2] * xx21 + ri[3] * yyzz21;
/*     (PY PX/PX S) */
                w[42] = x[0] * (ri[7] * xx21 + ri[8] * yyzz21) + ri[9] * (y[0] * xy21 + z[0] * xz21);
/*     (PY PX/PX PX) */
                w[43] = (ri[15] * xx21 + ri[16] * yyzz21) * xx11 + ri[17] *
                        xx21 * yyzz11 + ri[18] * (yy21 * yy11 + zz21 * zz11)
                        + ri[19] * (xy21 * xy11 + xz21 * xz11) + ri[20] * (
                        yy21 * zz11 + zz21 * yy11) + ri[21] * yz21 * yz11;
/*     (PY PX/PY S) */
                w[44] = x[1] * (ri[7] * xx21 + ri[8] * yyzz21) + ri[9] * (y[1] * xy21 + z[1] * xz21);
/*     (PY PX/PY PX) */
                w[45] = (ri[15] * xx21 + ri[16] * yyzz21) * xx21 + ri[17] *
                        xx21 * yyzz21 + ri[18] * (yy21 * yy21 + zz21 * zz21)
                        + ri[19] * (xy21 * xy21 + xz21 * xz21) + ri[20] * (
                        yy21 * zz21 + zz21 * yy21) + ri[21] * yz21 * yz21;
/*     (PY PX/PY PY) */
                w[46] = (ri[15] * xx21 + ri[16] * yyzz21) * xx22 + ri[17] *
                        xx21 * yyzz22 + ri[18] * (yy21 * yy22 + zz21 * zz22)
                        + ri[19] * (xy21 * xy22 + xz21 * xz22) + ri[20] * (
                        yy21 * zz22 + zz21 * yy22) + ri[21] * yz21 * yz22;
/*     (PY PX/PZ S) */
                w[47] = x[2] * (ri[7] * xx21 + ri[8] * yyzz21) + ri[9] * (z[2] * xz21);
/*      (PY PX/PZ PX) */
                w[48] = (ri[15] * xx21 + ri[16] * yyzz21) * xx31 + (ri[17] *
                        xx21 + ri[18] * zz21 + ri[20] * yy21) * zz31 + ri[19]
                        * (xy21 * xy31 + xz21 * xz31) + ri[21] * yz21 * yz31;
/*      (PY PX/PZ PY) */
                w[49] = (ri[15] * xx21 + ri[16] * yyzz21) * xx32 + (ri[17] *
                        xx21 + ri[18] * zz21 + ri[20] * yy21) * zz32 + ri[19]
                        * (xy21 * xy32 + xz21 * xz32) + ri[21] * yz21 * yz32;
/*      (PY PX/PZ PZ) */
                w[50] = (ri[15] * xx21 + ri[16] * yyzz21) * xx33 + (ri[17] *
                        xx21 + ri[18] * zz21 + ri[20] * yy21) * zz33 + ri[19]
                        * xz21 * xz33;
/*     (PY PY/S S) */
                w[51] = ri[2] * xx22 + ri[3] * yyzz22;
/*     (PY PY/PX S) */
                w[52] = x[0] * (ri[7] * xx22 + ri[8] * yyzz22) + ri[9] * (y[0] * xy22 + z[0] * xz22);
/*      (PY PY/PX PX) */
                w[53] = (ri[15] * xx22 + ri[16] * yyzz22) * xx11 + ri[17] *
                        xx22 * yyzz11 + ri[18] * (yy22 * yy11 + zz22 * zz11)
                        + ri[19] * (xy22 * xy11 + xz22 * xz11) + ri[20] * (
                        yy22 * zz11 + zz22 * yy11) + ri[21] * yz22 * yz11;
/*     (PY PY/PY S) */
                w[54] = x[1] * (ri[7] * xx22 + ri[8] * yyzz22) + ri[9] * (y[1] * xy22 + z[1] * xz22);
/*      (PY PY/PY PX) */
                w[55] = (ri[15] * xx22 + ri[16] * yyzz22) * xx21 + ri[17] *
                        xx22 * yyzz21 + ri[18] * (yy22 * yy21 + zz22 * zz21)
                        + ri[19] * (xy22 * xy21 + xz22 * xz21) + ri[20] * (
                        yy22 * zz21 + zz22 * yy21) + ri[21] * yz22 * yz21;
/*      (PY PY/PY PY) */
                w[56] = (ri[15] * xx22 + ri[16] * yyzz22) * xx22 + ri[17] *
                        xx22 * yyzz22 + ri[18] * (yy22 * yy22 + zz22 * zz22)
                        + ri[19] * (xy22 * xy22 + xz22 * xz22) + ri[20] * (
                        yy22 * zz22 + zz22 * yy22) + ri[21] * yz22 * yz22;
/*     (PY PY/PZ S) */
                w[57] = x[2] * (ri[7] * xx22 + ri[8] * yyzz22) + ri[9] * (z[2] * xz22);
/*      (PY PY/PZ PX) */
                w[58] = (ri[15] * xx22 + ri[16] * yyzz22) * xx31 + (ri[17] *
                        xx22 + ri[18] * zz22 + ri[20] * yy22) * zz31 + ri[19]
                        * (xy22 * xy31 + xz22 * xz31) + ri[21] * yz22 * yz31;
/*      (PY PY/PZ PY) */
                w[59] = (ri[15] * xx22 + ri[16] * yyzz22) * xx32 + (ri[17] *
                        xx22 + ri[18] * zz22 + ri[20] * yy22) * zz32 + ri[19]
                        * (xy22 * xy32 + xz22 * xz32) + ri[21] * yz22 * yz32;
/*      (PY PY/PZ PZ) */
                w[60] = (ri[15] * xx22 + ri[16] * yyzz22) * xx33 + (ri[17] *
                        xx22 + ri[18] * zz22 + ri[20] * yy22) * zz33 + ri[19]
                        * xz22 * xz33;
/*     (PZ S/SS) */
                w[61] = ri[1] * x[2];
/*     (PZ S/PX S) */
                w[62] = ri[5] * xx31 + ri[6] * zz31;
/*     (PZ S/PX PX) */
                w[63] = x[2] * (ri[12] * xx11 + ri[13] * yyzz11) + ri[14] * (z[2] * xz11);
/*     (PZ S/PY S) */
                w[64] = ri[5] * xx32 + ri[6] * zz32;
/*     (PZ S/PY PX) */
                w[65] = x[2] * (ri[12] * xx21 + ri[13] * yyzz21) + ri[14] * (z[2] * xz21);
/*     (PZ S/PY PY) */
                w[66] = x[2] * (ri[12] * xx22 + ri[13] * yyzz22) + ri[14] * (z[2] * xz22);
/*     (PZ S/PZ S) */
                w[67] = ri[5] * xx33 + ri[6] * zz33;
/*     (PZ S/PZ PX) */
                w[68] = x[2] * (ri[12] * xx31 + ri[13] * zz31) + ri[14] * (z[2] * xz31);
/*     (PZ S/PZ PY) */
                w[69] = x[2] * (ri[12] * xx32 + ri[13] * zz32) + ri[14] * (z[2] * xz32);
/*     (PZ S/PZ PZ) */
                w[70] = x[2] * (ri[12] * xx33 + ri[13] * zz33) + ri[14] * (z[2] * xz33);
/*     (PZ PX/S S) */
                w[71] = ri[2] * xx31 + ri[3] * zz31;
/*     (PZ PX/PX S) */
                w[72] = x[0] * (ri[7] * xx31 + ri[8] * zz31) + ri[9] * (y[0] * xy31 + z[0] * xz31);
/*      (PZ PX/PX PX) */
                w[73] = (ri[15] * xx31 + ri[16] * zz31) * xx11 + ri[17] *
                        xx31 * yyzz11 + ri[18] * zz31 * zz11 + ri[19] * (xy31
                        * xy11 + xz31 * xz11) + ri[20] * zz31 * yy11 + ri[21]
                        * yz31 * yz11;
/*     (PZ PX/PY S) */
                w[74] = x[1] * (ri[7] * xx31 + ri[8] * zz31) + ri[9]
                        * (y[1] * xy31 + z[1] * xz31);
/*      (PZ PX/PY PX) */
                w[75] = (ri[15] * xx31 + ri[16] * zz31) * xx21 + ri[17] *
                        xx31 * yyzz21 + ri[18] * zz31 * zz21 + ri[19] * (xy31
                        * xy21 + xz31 * xz21) + ri[20] * zz31 * yy21 + ri[21]
                        * yz31 * yz21;
/*      (PZ PX/PY PY) */
                w[76] = (ri[15] * xx31 + ri[16] * zz31) * xx22 + ri[17] *
                        xx31 * yyzz22 + ri[18] * zz31 * zz22 + ri[19] * (xy31
                        * xy22 + xz31 * xz22) + ri[20] * zz31 * yy22 + ri[21]
                        * yz31 * yz22;
/*     (PZ PX/PZ S) */
                w[77] = x[2] * (ri[7] * xx31 + ri[8] * zz31) + ri[9]
                        * (z[2] * xz31);
/*     (PZ PX/PZ PX) */
                w[78] = (ri[15] * xx31 + ri[16] * zz31) * xx31 + (ri[17] *
                        xx31 + ri[18] * zz31) * zz31 + ri[19] * (xy31 * xy31
                        + xz31 * xz31) + ri[21] * yz31 * yz31;
/*      (PZ PX/PZ PY) */
                w[79] = (ri[15] * xx31 + ri[16] * zz31) * xx32 + (ri[17] *
                        xx31 + ri[18] * zz31) * zz32 + ri[19] * (xy31 * xy32
                        + xz31 * xz32) + ri[21] * yz31 * yz32;
/*      (PZ PX/PZ PZ) */
                w[80] = (ri[15] * xx31 + ri[16] * zz31) * xx33 + (ri[17] *
                        xx31 + ri[18] * zz31) * zz33 + ri[19] * xz31 * xz33;
/*     (PZ PY/S S) */
                w[81] = ri[2] * xx32 + ri[3] * zz32;
/*     (PZ PY/PX S) */
                w[82] = x[0] * (ri[7] * xx32 + ri[8] * zz32) + ri[9] * (y[0] * xy32 + z[0] * xz32);
/*      (PZ PY/PX PX) */
                w[83] = (ri[15] * xx32 + ri[16] * zz32) * xx11 + ri[17] *
                        xx32 * yyzz11 + ri[18] * zz32 * zz11 + ri[19] * (xy32
                        * xy11 + xz32 * xz11) + ri[20] * zz32 * yy11 + ri[21]
                        * yz32 * yz11;
/*     (PZ PY/PY S) */
                w[84] = x[1] * (ri[7] * xx32 + ri[8] * zz32) + ri[9] * (y[1] * xy32 + z[1] * xz32);
/*      (PZ PY/PY PX) */
                w[85] = (ri[15] * xx32 + ri[16] * zz32) * xx21 + ri[17] *
                        xx32 * yyzz21 + ri[18] * zz32 * zz21 + ri[19] * (xy32
                        * xy21 + xz32 * xz21) + ri[20] * zz32 * yy21 + ri[21]
                        * yz32 * yz21;
/*      (PZ PY/PY PY) */
                w[86] = (ri[15] * xx32 + ri[16] * zz32) * xx22 + ri[17] *
                        xx32 * yyzz22 + ri[18] * zz32 * zz22 + ri[19] * (xy32
                        * xy22 + xz32 * xz22) + ri[20] * zz32 * yy22 + ri[21]
                        * yz32 * yz22;
/*     (PZ PY/PZ S) */
                w[87] = x[2] * (ri[7] * xx32 + ri[8] * zz32) + ri[9] * (z[2] * xz32);
/*      (PZ PY/PZ PX) */
                w[88] = (ri[15] * xx32 + ri[16] * zz32) * xx31 + (ri[17] *
                        xx32 + ri[18] * zz32) * zz31 + ri[19] * (xy32 * xy31
                        + xz32 * xz31) + ri[21] * yz32 * yz31;
/*      (PZ PY/PZ PY) */
                w[89] = (ri[15] * xx32 + ri[16] * zz32) * xx32 + (ri[17] *
                        xx32 + ri[18] * zz32) * zz32 + ri[19] * (xy32 * xy32
                        + xz32 * xz32) + ri[21] * yz32 * yz32;
/*       (PZ PY/PZ PZ) */
                w[90] = (ri[15] * xx32 + ri[16] * zz32) * xx33 + (ri[17] *
                        xx32 + ri[18] * zz32) * zz33 + ri[19] * xz32 * xz33;
/*     (PZ PZ/S S) */
                w[91] = ri[2] * xx33 + ri[3] * zz33;
/*     (PZ PZ/PX S) */
                w[92] = x[0] * (ri[7] * xx33 + ri[8] * zz33) + ri[9] * (z[0] * xz33);
/*       (PZ PZ/PX PX) */
                w[93] = (ri[15] * xx33 + ri[16] * zz33) * xx11 + ri[17] *
                        xx33 * yyzz11 + ri[18] * zz33 * zz11 + ri[19] * xz33 *
                         xz11 + ri[20] * zz33 * yy11;
/*     (PZ PZ/PY S) */
                w[94] = x[1] * (ri[7] * xx33 + ri[8] * zz33) + ri[9] * (z[1] * xz33);
/*       (PZ PZ/PY PX) */
                w[95] = (ri[15] * xx33 + ri[16] * zz33) * xx21 + ri[17] *
                        xx33 * yyzz21 + ri[18] * zz33 * zz21 + ri[19] * xz33 *
                         xz21 + ri[20] * zz33 * yy21;
/*       (PZ PZ/PY PY) */
                w[96] = (ri[15] * xx33 + ri[16] * zz33) * xx22 + ri[17] *
                        xx33 * yyzz22 + ri[18] * zz33 * zz22 + ri[19] * xz33 *
                         xz22 + ri[20] * zz33 * yy22;
/*     (PZ PZ/PZ S) */
                w[97] = x[2] * (ri[7] * xx33 + ri[8] * zz33) + ri[9] * (z[2] * xz33);
/*       (PZ PZ/PZ PX) */
                w[98] = (ri[15] * xx33 + ri[16] * zz33) * xx31 + (ri[17] *
                        xx33 + ri[18] * zz33) * zz31 + ri[19] * xz33 * xz31;
/*       (PZ PZ/PZ PY) */
                w[99] = (ri[15] * xx33 + ri[16] * zz33) * xx32 + (ri[17] *
                        xx33 + ri[18] * zz33) * zz32 + ri[19] * xz33 * xz32;
/*       (PZ PZ/PZ PZ) */
                w[100] = (ri[15] * xx33 + ri[16] * zz33) * xx33 + (ri[17] *
                        xx33 + ri[18] * zz33) * zz33 + ri[19] * xz33 * xz33;
            } else {
/*     (PX S/S S) */
                w[2] = ri[1] * x[0];
/*     (PX PX/S S) */
                w[3] = ri[2] * xx11 + ri[3] * yyzz11;
/*     (PY S/S S) */
                w[4] = ri[1] * x[1];
/*     (PY PX/S S) */
                w[5] = ri[2] * xx21 + ri[3] * yyzz21;
/*     (PY PY/S S) */
                w[6] = ri[2] * xx22 + ri[3] * yyzz22;
/*     (PZ S/SS) */
                w[7] = ri[1] * x[2];
/*     (PZ PX/S S) */
                w[8] = ri[2] * xx31 + ri[3] * zz31;
/*     (PZ PY/S S) */
                w[9] = ri[2] * xx32 + ri[3] * zz32;
/*     (PZ PZ/S S) */
                w[10] = ri[2] * xx33 + ri[3] * zz33;
            }
        }

/* *** NOW ROTATE THE NUCLEAR ATTRACTION INTEGRALS. */
/* *** THE STORAGE OF THE NUCLEAR ATTRACTION INTEGRALS  CORE(KL/IJ) IS */
/*     (SS/)=1,   (SO/)=2,   (OO/)=3,   (PP/)=4 */

        e1b[1] = -css1;
        if (MOPAC_NATORB[ni] == 4) {
            e1b[2] = -csp1 * x[0];
            e1b[3] = -cpps1 * xx11 - cppp1 * yyzz11;
            e1b[4] = -csp1 * x[1];
            e1b[5] = -cpps1 * xx21 - cppp1 * yyzz21;
            e1b[6] = -cpps1 * xx22 - cppp1 * yyzz22;
            e1b[7] = -csp1 * x[2];
            e1b[8] = -cpps1 * xx31 - cppp1 * zz31;
            e1b[9] = -cpps1 * xx32 - cppp1 * zz32;
            e1b[10] = -cpps1 * xx33 - cppp1 * zz33;
        }
        e2a[1] = -css2;
        if (MOPAC_NATORB[nj] == 4) {
            e2a[2] = -csp2 * x[0];
            e2a[3] = -cpps2 * xx11 - cppp2 * yyzz11;
            e2a[4] = -csp2 * x[1];
            e2a[5] = -cpps2 * xx21 - cppp2 * yyzz21;
            e2a[6] = -cpps2 * xx22 - cppp2 * yyzz22;
            e2a[7] = -csp2 * x[2];
            e2a[8] = -cpps2 * xx31 - cppp2 * zz31;
            e2a[9] = -cpps2 * xx32 - cppp2 * zz32;
            e2a[10] = -cpps2 * xx33 - cppp2 * zz33;
        }
        if (fabs(MOPAC_CORE[ni]) > 20. && fabs(MOPAC_CORE[nj]) > 20.) {
/* SPARKLE-SPARKLE INTERACTION */
            *enuc = 0.;
            return;
        } else if (rij < 1. && MOPAC_NATORB[ni] * MOPAC_NATORB[nj] == 0) {
            *enuc = 0.;
            return;
        }
        scale = 1. + exp(-alp[ni] * rij) + exp(-alp[nj] * rij);

//        if (ni == 24 && nj == 24) {
//            scale = exp(-alptm[ni] * rij) + exp(-alptm[nj] * rij);
//        }

        nt = ni + nj;
        if (nt == 8 || nt == 9) {
            if (ni == 7 || ni == 8) {
                scale += (rij - 1.) * exp(-alp[ni] * rij);
            }
            if (nj == 7 || nj == 8) {
                scale += (rij - 1.) * exp(-alp[nj] * rij);
            }
        }

        scale *= MOPAC_CORE[ni] * MOPAC_CORE[nj] * gam;

        if (model == AM1_MODEL || model == PM3_MODEL) {
            double *pfn1, *pfn2, *pfn3;
            _set_fn_ptr(&pfn1, &pfn2, &pfn3, fn1, fn2, fn3, model, ni, nj);
            for (ig = 0; ig < 10; ++ig) {
                if (fabs(pfn1[ig]) > 0.) {
                    d__1 = rij - pfn3[ig];
                    ax = pfn2[ig] * (d__1 * d__1);
                    if (ax <= 25.) {
                        scale += MOPAC_CORE[ni] * MOPAC_CORE[nj] / rij * pfn1[ig] * exp(-ax);
                    }
                }
            }
            _set_fn_ptr(&pfn1, &pfn2, &pfn3, fn1, fn2, fn3, model, nj, ni);
            for (ig = 0; ig < 10; ++ig) {
                if (fabs(pfn1[ig]) > 0.) {
                    d__1 = rij - pfn3[ig];
                    ax = pfn2[ig] * (d__1 * d__1);
                    if (ax <= 25.) {
                        scale += MOPAC_CORE[ni] * MOPAC_CORE[nj] / rij * pfn1[ig] * exp(-ax);
                    }
                }
            }
        }
        *enuc = scale;
    }
}


