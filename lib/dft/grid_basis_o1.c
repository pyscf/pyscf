/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <math.h>
#include "cint.h"
#include "vhf/fblas.h"

// at least up to l = 8, nprim = 28
#define NPRIM_CART 1024

double CINTcommon_fac_sp(int l);

static void grid_cart_gto(double *gto, double coord[3], int l, double alpha)
{
        int lx, ly, lz, i;
        double rr = coord[0]*coord[0] + coord[1]*coord[1] + coord[2]*coord[2];
        double e = exp(-alpha * rr) * CINTcommon_fac_sp(l);
        switch (l) {
        case 0:
                *gto = e;
                break;
        case 1:
                gto[0] = coord[0] * e;
                gto[1] = coord[1] * e;
                gto[2] = coord[2] * e;
                break;
        case 2:
                gto[0] = coord[0] * coord[0] * e; // xx
                gto[1] = coord[0] * coord[1] * e; // xy
                gto[2] = coord[0] * coord[2] * e; // xz
                gto[3] = coord[1] * coord[1] * e; // yy
                gto[4] = coord[1] * coord[2] * e; // yz
                gto[5] = coord[2] * coord[2] * e; // zz
                break;
        case 3:
                gto[0] = coord[0] * coord[0] * coord[0] * e; // xxx
                gto[1] = coord[0] * coord[0] * coord[1] * e; // xxy
                gto[2] = coord[0] * coord[0] * coord[2] * e; // xxz
                gto[3] = coord[0] * coord[1] * coord[1] * e; // xyy
                gto[4] = coord[0] * coord[1] * coord[2] * e; // xyz
                gto[5] = coord[0] * coord[2] * coord[2] * e; // xzz
                gto[6] = coord[1] * coord[1] * coord[1] * e; // yyy
                gto[7] = coord[1] * coord[1] * coord[2] * e; // yyz
                gto[8] = coord[1] * coord[2] * coord[2] * e; // yzz
                gto[9] = coord[2] * coord[2] * coord[2] * e; // zzz
                break;
        default:
                for (i = 0, lx = l; lx >= 0; lx--) {
                        for (ly = l - lx; ly >= 0; ly--) {
                                lz = l - lx - ly;
                                gto[i] = pow(coord[0], lx)
                                        * pow(coord[1], ly)
                                        * pow(coord[2], lz) * e;
                                i++;
                        }
                }
        }
}

static void grid_cart_gto_grad(double *gto, int ngto, double coord[3],
                               int l, double alpha)
{
        int lx, ly, lz, i;
        double rr = coord[0]*coord[0] + coord[1]*coord[1] + coord[2]*coord[2];
        double e = exp(-alpha * rr) * CINTcommon_fac_sp(l);
        double *gtox = gto + ngto;
        double *gtoy = gto + ngto * 2;
        double *gtoz = gto + ngto * 3;
        double ax = -2 * alpha * coord[0];
        double ay = -2 * alpha * coord[1];
        double az = -2 * alpha * coord[2];
        double xinv, yinv, zinv;
        switch (l) {
        case 0:
                *gto  = e;
                *gtox = ax * e;
                *gtoy = ay * e;
                *gtoz = az * e;
                break;
        case 1:
                gto[0] = coord[0] * e;
                gto[1] = coord[1] * e;
                gto[2] = coord[2] * e;
                gtox[0] = ax * gto[0] + e;
                gtox[1] = ax * gto[1];
                gtox[2] = ax * gto[2];
                gtoy[0] = ay * gto[0];
                gtoy[1] = ay * gto[1] + e;
                gtoy[2] = ay * gto[2];
                gtoz[0] = az * gto[0];
                gtoz[1] = az * gto[1];
                gtoz[2] = az * gto[2] + e;
                break;
        case 2:
                gto[0] = coord[0] * coord[0] * e; // xx
                gto[1] = coord[0] * coord[1] * e; // xy
                gto[2] = coord[0] * coord[2] * e; // xz
                gto[3] = coord[1] * coord[1] * e; // yy
                gto[4] = coord[1] * coord[2] * e; // yz
                gto[5] = coord[2] * coord[2] * e; // zz
                gtox[0] = ax * gto[0] + 2 * coord[0] * e;
                gtox[1] = ax * gto[1] + coord[1] * e;
                gtox[2] = ax * gto[2] + coord[2] * e;
                gtox[3] = ax * gto[3];
                gtox[4] = ax * gto[4];
                gtox[5] = ax * gto[5];
                gtoy[0] = ay * gto[0];
                gtoy[1] = ay * gto[1] + coord[0] * e;
                gtoy[2] = ay * gto[2];
                gtoy[3] = ay * gto[3] + 2 * coord[1] * e;
                gtoy[4] = ay * gto[4] + coord[2] * e;
                gtoy[5] = ay * gto[5];
                gtoz[0] = az * gto[0];
                gtoz[1] = az * gto[1];
                gtoz[2] = az * gto[2] + coord[0] * e;
                gtoz[3] = az * gto[3];
                gtoz[4] = az * gto[4] + coord[1] * e;
                gtoz[5] = az * gto[5] + 2 * coord[2] * e;
                break;
        case 3:
                gto[0] = coord[0] * coord[0] * coord[0] * e; // xxx
                gto[1] = coord[0] * coord[0] * coord[1] * e; // xxy
                gto[2] = coord[0] * coord[0] * coord[2] * e; // xxz
                gto[3] = coord[0] * coord[1] * coord[1] * e; // xyy
                gto[4] = coord[0] * coord[1] * coord[2] * e; // xyz
                gto[5] = coord[0] * coord[2] * coord[2] * e; // xzz
                gto[6] = coord[1] * coord[1] * coord[1] * e; // yyy
                gto[7] = coord[1] * coord[1] * coord[2] * e; // yyz
                gto[8] = coord[1] * coord[2] * coord[2] * e; // yzz
                gto[9] = coord[2] * coord[2] * coord[2] * e; // zzz
                gtox[0] = ax * gto[0] + 3 * coord[0] * coord[0] * e;
                gtox[1] = ax * gto[1] + 2 * coord[0] * coord[1] * e;
                gtox[2] = ax * gto[2] + 2 * coord[0] * coord[2] * e;
                gtox[3] = ax * gto[3] + coord[1] * coord[1] * e;
                gtox[4] = ax * gto[4] + coord[1] * coord[2] * e;
                gtox[5] = ax * gto[5] + coord[2] * coord[2] * e;
                gtox[6] = ax * gto[6];
                gtox[7] = ax * gto[7];
                gtox[8] = ax * gto[8];
                gtox[9] = ax * gto[9];
                gtoy[0] = ay * gto[0];
                gtoy[1] = ay * gto[1] + coord[0] * coord[0] * e;
                gtoy[2] = ay * gto[2];
                gtoy[3] = ay * gto[3] + 2 * coord[0] * coord[1] * e;
                gtoy[4] = ay * gto[4] + coord[0] * coord[2] * e;
                gtoy[5] = ay * gto[5];
                gtoy[6] = ay * gto[6] + 3 * coord[1] * coord[1] * e;
                gtoy[7] = ay * gto[7] + 2 * coord[1] * coord[2] * e;
                gtoy[8] = ay * gto[8] + coord[2] * coord[2] * e;
                gtoy[9] = ay * gto[9];
                gtoz[0] = az * gto[0];
                gtoz[1] = az * gto[1];
                gtoz[2] = az * gto[2] + coord[0] * coord[0] * e;
                gtoz[3] = az * gto[3];
                gtoz[4] = az * gto[4] + coord[0] * coord[1] * e;
                gtoz[5] = az * gto[5] + 2 * coord[0] * coord[2] * e;
                gtoz[6] = az * gto[6];
                gtoz[7] = az * gto[7] + coord[1] * coord[1] * e;
                gtoz[8] = az * gto[8] + 2 * coord[1] * coord[2] * e;
                gtoz[9] = az * gto[9] + 3 * coord[2] * coord[2] * e;
                break;
        default:
                xinv = 1/(coord[0]+1e-80);
                yinv = 1/(coord[1]+1e-80);
                zinv = 1/(coord[2]+1e-80);
                for (i = 0, lx = l; lx >= 0; lx--) {
                        for (ly = l - lx; ly >= 0; ly--) {
                                lz = l - lx - ly;
                                gto[i] = pow(coord[0], lx)
                                        * pow(coord[1], ly)
                                        * pow(coord[2], lz) * e;
                                gtox[i] = (lx * xinv + ax) * gto[i];
                                gtoy[i] = (ly * yinv + ay) * gto[i];
                                gtoz[i] = (lz * zinv + az) * gto[i];
                                i++;
                        }
                }
        }
}

inline void _contract(int m, int n, int k, double *a, double *b, double *c)
{
        int i, j, u;
        if (k == 0) {
                for (i = 0; i < m; i++) {
                        c[i] = a[i] * b[0];
                }
        } else {
                for (j = 0; j < n; j++) {
                        for (i = 0; i < m; i++) {
                                c[j*m+i] = 0;
                        }
                        for (u = 0; u < k; u++) {
                        for (i = 0; i < m; i++) {
                                c[j*m+i] += a[u*m+i] * b[j*k+u];
                        } }
                }
        }
}

void VXCvalue_nr_gto(int nao, int ngrids, double *ao, double *coord,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, k, ig;
        int l, np, nc, ncart, atm_id, bas_id, deg;
        int *pbas;
        double *r_atm, *p_exp, *pcoeff;
        double dcoord[3];
        double cart_gto[NPRIM_CART*ngrids];
        double pgto_buf[NPRIM_CART*ngrids];
        double *pgto;
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                pbas = bas + bas_id*BAS_SLOTS;
                np = pbas[NPRIM_OF];
                nc = pbas[NCTR_OF ];
                l  = pbas[ANG_OF  ];
                ncart = CINTlen_cart(l);
                deg = l * 2 + 1;
                p_exp = &env[pbas[PTR_EXP]];
                pcoeff = &env[pbas[PTR_COEFF]];
                atm_id = pbas[ATOM_OF];
                r_atm = &env[atm[PTR_COORD+atm_id*ATM_SLOTS]];
                for (k = 0, ig = 0; ig < ngrids; ig++) {
                        dcoord[0] = coord[ig*3+0] - r_atm[0];
                        dcoord[1] = coord[ig*3+1] - r_atm[1];
                        dcoord[2] = coord[ig*3+2] - r_atm[2];
                        for (i = 0; i < np; i++, k++) {
                                grid_cart_gto(cart_gto+k*ncart, dcoord, l,
                                              p_exp[i]);
                        }
                }
                pgto = CINTc2s_bra_sph(pgto_buf, np*ngrids, cart_gto, l);
                for (ig = 0; ig < ngrids; ig++) {
                        //dgemm_(&TRANS_N, &TRANS_N, &deg, &nc, &np,
                        //       &D1, pgto+np*deg*ig, &deg, pcoeff, &np,
                        //       &D0, ao+nao*ig, &deg);
                        _contract(deg, nc, np, pgto+np*deg*ig, pcoeff,
                                  ao+nao*ig);
                }
                ao += nc * deg;
        }
}

void VXCvalue_nr_gto_grad(int nao, int ngrids, double *ao, double *coord,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, k, ig;
        int l, np, nc, ncart, atm_id, bas_id, deg;
        int *pbas;
        double *r_atm, *p_exp, *pcoeff;
        double dcoord[3];
        double cart_gto[NPRIM_CART*ngrids*4];
        double pgto_buf[NPRIM_CART*ngrids*4];
        double *pgto;
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                pbas = bas + bas_id*BAS_SLOTS;
                np = pbas[NPRIM_OF];
                nc = pbas[NCTR_OF ];
                l  = pbas[ANG_OF  ];
                ncart = CINTlen_cart(l);
                deg = l * 2 + 1;
                p_exp = &env[pbas[PTR_EXP]];
                pcoeff = &env[pbas[PTR_COEFF]];
                atm_id = pbas[ATOM_OF];
                r_atm = &env[atm[PTR_COORD+atm_id*ATM_SLOTS]];
                for (k = 0, ig = 0; ig < ngrids; ig++) {
                        dcoord[0] = coord[ig*3+0] - r_atm[0];
                        dcoord[1] = coord[ig*3+1] - r_atm[1];
                        dcoord[2] = coord[ig*3+2] - r_atm[2];
                        for (i = 0; i < np; i++, k++) {
                                grid_cart_gto_grad(cart_gto+k*ncart, ncart*np*ngrids,
                                                   dcoord, l, p_exp[i]);
                        }
                }
                for (i = 0; i < 4; i++) {
                        pgto = CINTc2s_bra_sph(pgto_buf, np*ngrids,
                                               cart_gto+ncart*np*ngrids*i, l);
                        for (ig = 0; ig < ngrids; ig++) {
                                //dgemm_(&TRANS_N, &TRANS_N, &deg, &nc, &np,
                                //       &D1, pgto+np*deg*ig, &deg, pcoeff, &np,
                                //       &D0, ao+nao*(ngrids*i+ig), &deg);
                                _contract(deg, nc, np, pgto+np*deg*ig, pcoeff,
                                          ao+nao*(ngrids*i+ig));
                        }
                }
                ao += nc * deg;
        }
}
