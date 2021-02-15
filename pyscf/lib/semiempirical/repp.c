/*
 * Modified based on repp.f from MOPAC 7-1.11
 */

#include <math.h>

/*
 *  NATORB IS THE NUMBER OF ATOMIC ORBITALS PER ATOM.
 */
int MOPAC_NATORB[] = {0,
        1, 1,
        4, 4, 4, 4, 4, 4, 4, 4,
        0, 4, 4, 4, 4, 4, 4, 4,
        0, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4,
        2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
};

/*
 *  CORE IS THE CHARGE ON THE ATOM AS SEEN BY THE ELECTRONS
 */
double MOPAC_CORE[] = {0.,
        1.,0.,
        1.,2.,3.,4.,5.,6.,7.,0.,
        1.,2.,3.,4.,5.,6.,7.,0.,
        1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,2.,3.,4.,5.,6.,7.,0.,
        1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,2.,3.,4.,5.,6.,7.,0.,
        1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,3.,4.,5.,6.,7.,8.,9.,10.,11.,2.,3.,4.,5.,6.,7.,0.,
        0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,2.,1.,-2.,-1.,0.,
};



// Note rij needs to be in atomic unit
// double ev = 27.21;
void MOPAC_repp(int ni, int nj, double rij, double ev, double *ri, double *core,
                double *dd, double *qq, double *am, double *ad, double *aq)
{
    /* Initialized data */

    double ev1 = ev / 2;
    double ev2 = ev / 4;
    double ev3 = ev / 8;
    double ev4 = ev / 16;

    /* Local variables */
    int i__;
    double da, db, ee, qa, qb;
    int si, sj;
    double ade, aee, aed, adq, aqe, aeq, aqd, arg[72], aqq, dze, edz, axx,
           sqr[72], rsq, www, xxx, yyy, zzz, dxdx, dzdz, qxxe, eqxx, qzze,
           eqzz, dzqxx, qxxdz, dxqxz, qxzdx, dzqzz, qzzdz, qxxqxx, qxxqyy,
           qxxqzz, qzzqxx, qxzqxz, qzzqzz;

/* *********************************************************************** */

/* ..VECTOR VERSION WRITTEN BY ERNEST R. DAVIDSON, INDIANA UNIVERSITY */


/*  REPP CALCULATES THE TWO-ELECTRON REPULSION INTEGRALS AND THE */
/*       NUCLEAR ATTRACTION INTEGRALS. */

/*     ON INPUT RIJ     = INTERATOMIC DISTANCE */
/*              NI      = ATOM NUMBER OF FIRST ATOM */
/*              NJ      = ATOM NUMBER OF SECOND ATOM */
/*    (REF)     ADD     = ARRAY OF GAMMA, OR TWO-ELECTRON ONE-CENTER, */
/*                        INTEGRALS. */
/*    (REF)     TORE    = ARRAY OF NUCLEAR CHARGES OF THE ELEMENTS */
/*    (REF)     DD      = ARRAY OF DIPOLE CHARGE SEPARATIONS */
/*    (REF)     QQ      = ARRAY OF QUADRUPOLE CHARGE SEPARATIONS */

/*     THE COMMON BLOCKS ARE INITIALIZED IN BLOCK-DATA, AND NEVER CHANGED */

/*    ON OUTPUT RI      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS */
/*              CORE    = 4 X 2 ARRAY OF ELECTRON-CORE ATTRACTION */
/*                        INTEGRALS */


/* *** THIS ROUTINE COMPUTES THE TWO-CENTRE REPULSION INTEGRALS AND THE */
/* *** NUCLEAR ATTRACTION INTEGRALS. */
/* *** THE TWO-CENTRE REPULSION INTEGRALS (OVER LOCAL COORDINATES) ARE */
/* *** STORED AS FOLLOWS (WHERE P-SIGMA = O,  AND P-PI = P AND P* ) */
/*     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5, */
/*     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10, */
/*     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15, */
/*     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20, */
/*     (PP/P*P*)=21,   (P*P/P*P)=22. */
/* *** THE STORAGE OF THE NUCLEAR ATTRACTION INTEGRALS  CORE(KL/IJ) IS */
/*     (SS/)=1,   (SO/)=2,   (OO/)=3,   (PP/)=4 */
/*     WHERE IJ=1 IF THE ORBITALS CENTRED ON ATOM I,  =2 IF ON ATOM J. */
/* *** NI AND NJ ARE THE ATOMIC NUMBERS OF THE TWO ELEMENTS. */

/* *********************************************************************** */
    /* Function Body */

/*     ATOMIC UNITS ARE USED IN THE CALCULATION, */
/*     FINAL RESULTS ARE CONVERTED TO EV */

    si = MOPAC_NATORB[ni] >= 3;
    sj = MOPAC_NATORB[nj] >= 3;

    if (! si && ! sj) {

/*     HYDROGEN - HYDROGEN  (SS/SS) */

        aee = .5 / am[ni] + .5 / am[nj];
        aee *= aee;
        ri[0] = ev / sqrt(rij * rij + aee);
        core[0] = MOPAC_CORE[nj] * ri[0];
        core[4] = MOPAC_CORE[ni] * ri[0];

    } else if (si && ! sj) {

/*     HEAVY ATOM - HYDROGEN */

        aee = .5 / am[ni] + .5 / am[nj];
        aee *= aee;
        da = dd[ni];
        qa = qq[ni] * 2.;
        ade = .5 / ad[ni] + .5 / am[nj];
        ade *= ade;
        aqe = .5 / aq[ni] + .5 / am[nj];
        aqe *= aqe;
        rsq = rij * rij;
        arg[0] = rsq + aee;
        xxx = rij + da;
        arg[1] = xxx * xxx + ade;
        xxx = rij - da;
        arg[2] = xxx * xxx + ade;
        xxx = rij + qa;
        arg[3] = xxx * xxx + aqe;
        xxx = rij - qa;
        arg[4] = xxx * xxx + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
/* $DOIT ASIS */
        for (i__ = 1; i__ <= 7; ++i__) {
            sqr[i__ - 1] = sqrt(arg[i__ - 1]);
/* L10: */
        }
        ee = ev / sqr[0];
        ri[0] = ee;
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
        core[0] = MOPAC_CORE[nj] * ri[0];
        core[4] = MOPAC_CORE[ni] * ri[0];
        core[1] = MOPAC_CORE[nj] * ri[1];
        core[2] = MOPAC_CORE[nj] * ri[2];
        core[3] = MOPAC_CORE[nj] * ri[3];

    } else if (! si && sj) {

/*     HYDROGEN - HEAVY ATOM */

        aee = .5 / am[ni] + .5 / am[nj];
        aee *= aee;
        db = dd[nj];
        qb = qq[nj] * 2.;
        aed = .5 / am[ni] + .5 / ad[nj];
        aed *= aed;
        aeq = .5 / am[ni] + .5 / aq[nj];
        aeq *= aeq;
        rsq = rij * rij;
        arg[0] = rsq + aee;
        xxx = rij - db;
        arg[1] = xxx * xxx + aed;
        xxx = rij + db;
        arg[2] = xxx * xxx + aed;
        xxx = rij - qb;
        arg[3] = xxx * xxx + aeq;
        xxx = rij + qb;
        arg[4] = xxx * xxx + aeq;
        arg[5] = rsq + aeq;
        arg[6] = arg[5] + qb * qb;
/* $DOIT ASIS */
        for (i__ = 1; i__ <= 7; ++i__) {
            sqr[i__ - 1] = sqrt(arg[i__ - 1]);
/* L20: */
        }
        ee = ev / sqr[0];
        ri[0] = ee;
        ri[4] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5];
        core[0] = MOPAC_CORE[nj] * ri[0];
        core[4] = MOPAC_CORE[ni] * ri[0];
        core[5] = MOPAC_CORE[ni] * ri[4];
        core[6] = MOPAC_CORE[ni] * ri[10];
        core[7] = MOPAC_CORE[ni] * ri[11];

    } else {

/*     HEAVY ATOM - HEAVY ATOM */

/*     DEFINE CHARGE SEPARATIONS. */
        da = dd[ni];
        db = dd[nj];
        qa = qq[ni] * 2.;
        qb = qq[nj] * 2.;

        aee = .5 / am[ni] + .5 / am[nj];
        aee *= aee;

        ade = .5 / ad[ni] + .5 / am[nj];
        ade *= ade;
        aqe = .5 / aq[ni] + .5 / am[nj];
        aqe *= aqe;
        aed = .5 / am[ni] + .5 / ad[nj];
        aed *= aed;
        aeq = .5 / am[ni] + .5 / aq[nj];
        aeq *= aeq;
        axx = .5 / ad[ni] + .5 / ad[nj];
        axx *= axx;
        adq = .5 / ad[ni] + .5 / aq[nj];
        adq *= adq;
        aqd = .5 / aq[ni] + .5 / ad[nj];
        aqd *= aqd;
        aqq = .5 / aq[ni] + .5 / aq[nj];
        aqq *= aqq;
        rsq = rij * rij;
        arg[0] = rsq + aee;
        xxx = rij + da;
        arg[1] = xxx * xxx + ade;
        xxx = rij - da;
        arg[2] = xxx * xxx + ade;
        xxx = rij - qa;
        arg[3] = xxx * xxx + aqe;
        xxx = rij + qa;
        arg[4] = xxx * xxx + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
        xxx = rij - db;
        arg[7] = xxx * xxx + aed;
        xxx = rij + db;
        arg[8] = xxx * xxx + aed;
        xxx = rij - qb;
        arg[9] = xxx * xxx + aeq;
        xxx = rij + qb;
        arg[10] = xxx * xxx + aeq;
        arg[11] = rsq + aeq;
        arg[12] = arg[11] + qb * qb;
        xxx = da - db;
        arg[13] = rsq + axx + xxx * xxx;
        xxx = da + db;
        arg[14] = rsq + axx + xxx * xxx;
        xxx = rij + da - db;
        arg[15] = xxx * xxx + axx;
        xxx = rij - da + db;
        arg[16] = xxx * xxx + axx;
        xxx = rij - da - db;
        arg[17] = xxx * xxx + axx;
        xxx = rij + da + db;
        arg[18] = xxx * xxx + axx;
        xxx = rij + da;
        arg[19] = xxx * xxx + adq;
        arg[20] = arg[19] + qb * qb;
        xxx = rij - da;
        arg[21] = xxx * xxx + adq;
        arg[22] = arg[21] + qb * qb;
        xxx = rij - db;
        arg[23] = xxx * xxx + aqd;
        arg[24] = arg[23] + qa * qa;
        xxx = rij + db;
        arg[25] = xxx * xxx + aqd;
        arg[26] = arg[25] + qa * qa;
        xxx = rij + da - qb;
        arg[27] = xxx * xxx + adq;
        xxx = rij - da - qb;
        arg[28] = xxx * xxx + adq;
        xxx = rij + da + qb;
        arg[29] = xxx * xxx + adq;
        xxx = rij - da + qb;
        arg[30] = xxx * xxx + adq;
        xxx = rij + qa - db;
        arg[31] = xxx * xxx + aqd;
        xxx = rij + qa + db;
        arg[32] = xxx * xxx + aqd;
        xxx = rij - qa - db;
        arg[33] = xxx * xxx + aqd;
        xxx = rij - qa + db;
        arg[34] = xxx * xxx + aqd;
        arg[35] = rsq + aqq;
        xxx = qa - qb;
        arg[36] = arg[35] + xxx * xxx;
        xxx = qa + qb;
        arg[37] = arg[35] + xxx * xxx;
        arg[38] = arg[35] + qa * qa;
        arg[39] = arg[35] + qb * qb;
        arg[40] = arg[38] + qb * qb;
        xxx = rij - qb;
        arg[41] = xxx * xxx + aqq;
        arg[42] = arg[41] + qa * qa;
        xxx = rij + qb;
        arg[43] = xxx * xxx + aqq;
        arg[44] = arg[43] + qa * qa;
        xxx = rij + qa;
        arg[45] = xxx * xxx + aqq;
        arg[46] = arg[45] + qb * qb;
        xxx = rij - qa;
        arg[47] = xxx * xxx + aqq;
        arg[48] = arg[47] + qb * qb;
        xxx = rij + qa - qb;
        arg[49] = xxx * xxx + aqq;
        xxx = rij + qa + qb;
        arg[50] = xxx * xxx + aqq;
        xxx = rij - qa - qb;
        arg[51] = xxx * xxx + aqq;
        xxx = rij - qa + qb;
        arg[52] = xxx * xxx + aqq;
        qa = qq[ni];
        qb = qq[nj];
        xxx = da - qb;
        xxx *= xxx;
        yyy = rij - qb;
        yyy *= yyy;
        zzz = da + qb;
        zzz *= zzz;
        www = rij + qb;
        www *= www;
        arg[53] = xxx + yyy + adq;
        arg[54] = xxx + www + adq;
        arg[55] = zzz + yyy + adq;
        arg[56] = zzz + www + adq;
        xxx = qa - db;
        xxx *= xxx;
        yyy = qa + db;
        yyy *= yyy;
        zzz = rij + qa;
        zzz *= zzz;
        www = rij - qa;
        www *= www;
        arg[57] = zzz + xxx + aqd;
        arg[58] = www + xxx + aqd;
        arg[59] = zzz + yyy + aqd;
        arg[60] = www + yyy + aqd;
        xxx = qa - qb;
        xxx *= xxx;
        arg[61] = arg[35] + 2. * xxx;
        yyy = qa + qb;
        yyy *= yyy;
        arg[62] = arg[35] + 2. * yyy;
        arg[63] = arg[35] + 2. * (qa * qa + qb * qb);
        zzz = rij + qa - qb;
        zzz *= zzz;
        arg[64] = zzz + xxx + aqq;
        arg[65] = zzz + yyy + aqq;
        zzz = rij + qa + qb;
        zzz *= zzz;
        arg[66] = zzz + xxx + aqq;
        arg[67] = zzz + yyy + aqq;
        zzz = rij - qa - qb;
        zzz *= zzz;
        arg[68] = zzz + xxx + aqq;
        arg[69] = zzz + yyy + aqq;
        zzz = rij - qa + qb;
        zzz *= zzz;
        arg[70] = zzz + xxx + aqq;
        arg[71] = zzz + yyy + aqq;
        for (i__ = 1; i__ <= 72; ++i__) {
            sqr[i__ - 1] = sqrt(arg[i__ - 1]);
        }
        ee = ev / sqr[0];
        dze = -ev1 / sqr[1] + ev1 / sqr[2];
        qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        qxxe = ev1 / sqr[6] - ev1 / sqr[5];
        edz = -ev1 / sqr[7] + ev1 / sqr[8];
        eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
        eqxx = ev1 / sqr[12] - ev1 / sqr[11];
        dxdx = ev1 / sqr[13] - ev1 / sqr[14];
        dzdz = ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18];
        dzqxx = ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22];
        qxxdz = ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26];
        dzqzz =-ev3 / sqr[27] + ev3 / sqr[28] - ev3 / sqr[29] + ev3 / sqr[30]
              - ev2 / sqr[21] + ev2 / sqr[19];
        qzzdz = -ev3 / sqr[31] + ev3 / sqr[32] - ev3 / sqr[33] + ev3 / sqr[34]
               + ev2 / sqr[23] - ev2 / sqr[25];
        qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38] - ev2 / sqr[39]
               + ev2 / sqr[35];
        qxxqyy = ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35];
        qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41] - ev3 / sqr[43]
               - ev2 / sqr[38] + ev2 / sqr[35];
        qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45] - ev3 / sqr[47]
               - ev2 / sqr[39] + ev2 / sqr[35];
        qzzqzz = ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51] + ev4 / sqr[52]
               - ev3 / sqr[47] - ev3 / sqr[45] - ev3 / sqr[41] - ev3 / sqr[43] + ev2 / sqr[35];
        dxqxz = -ev2 / sqr[53] + ev2 / sqr[54] + ev2 / sqr[55] - ev2 / sqr[56];
        qxzdx = -ev2 / sqr[57] + ev2 / sqr[58] + ev2 / sqr[59] - ev2 / sqr[60];
        qxzqxz = ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68] + ev3 / sqr[70]
               - ev3 / sqr[65] + ev3 / sqr[67] + ev3 / sqr[69] - ev3 / sqr[71];
        ri[0] = ee;
        ri[1] = -dze;
        ri[2] = ee + qzze;
        ri[3] = ee + qxxe;
        ri[4] = -edz;
        ri[5] = dzdz;
        ri[6] = dxdx;
        ri[7] = -edz - qzzdz;
        ri[8] = -edz - qxxdz;
        ri[9] = -qxzdx;
        ri[10] = ee + eqzz;
        ri[11] = ee + eqxx;
        ri[12] = -dze - dzqzz;
        ri[13] = -dze - dzqxx;
        ri[14] = -dxqxz;
        ri[15] = ee + eqzz + qzze + qzzqzz;
        ri[16] = ee + eqzz + qxxe + qxxqzz;
        ri[17] = ee + eqxx + qzze + qzzqxx;
        ri[18] = ee + eqxx + qxxe + qxxqxx;
        ri[19] = qxzqxz;
        ri[20] = ee + eqxx + qxxe + qxxqyy;
        ri[21] = .5 * (qxxqxx - qxxqyy);

/*     CALCULATE CORE-ELECTRON ATTRACTIONS. */

        core[0] = MOPAC_CORE[nj] * ri[0];
        core[1] = MOPAC_CORE[nj] * ri[1];
        core[2] = MOPAC_CORE[nj] * ri[2];
        core[3] = MOPAC_CORE[nj] * ri[3];
        core[4] = MOPAC_CORE[ni] * ri[0];
        core[5] = MOPAC_CORE[ni] * ri[4];
        core[6] = MOPAC_CORE[ni] * ri[10];
        core[7] = MOPAC_CORE[ni] * ri[11];
    }
}

