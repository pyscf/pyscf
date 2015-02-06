/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"
#include "vhf/fblas.h"

// l = 6, nprim can reach 36
#define NPRIM_CART 1024
#define NPRIMAX    64

double CINTcommon_fac_sp(int l);

// exps, np can be used for both primitive or contract case
// for contracted case, np stands for num contracted functions, exps are
// contracted factors = \sum c_{i} exp(-a_i*r_i**2)
static void grid_cart_gto(double *gto, double coord[3], double *exps,
                          int l, int np)
{
        int ncart;
        int lx, ly, lz, i, k;
        double e;
        switch (l) {
        case 0:
                for (k = 0; k < np; k++) {
                        gto[k] = exps[k];
                }
                break;
        case 1:
                for (k = 0; k < np; k++) {
                        gto[0] = coord[0] * exps[k];
                        gto[1] = coord[1] * exps[k];
                        gto[2] = coord[2] * exps[k];
                        gto += 3;
                }
                break;
        case 2:
                for (k = 0; k < np; k++) {
                        e = exps[k];
                        gto[0] = coord[0] * coord[0] * e; // xx
                        gto[1] = coord[0] * coord[1] * e; // xy
                        gto[2] = coord[0] * coord[2] * e; // xz
                        gto[3] = coord[1] * coord[1] * e; // yy
                        gto[4] = coord[1] * coord[2] * e; // yz
                        gto[5] = coord[2] * coord[2] * e; // zz
                        gto += 6;
                }
                break;
        case 3:
                for (k = 0; k < np; k++) {
                        e = exps[k];
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
                        gto += 10;
                }
                break;
        default:
                ncart = CINTlen_cart(l);
                for (k = 0; k < np; k++) {
                        e = exps[k];
                        for (i = 0, lx = l; lx >= 0; lx--) {
                                for (ly = l - lx; ly >= 0; ly--) {
                                        lz = l - lx - ly;
                                        gto[i] = pow(coord[0], lx)
                                                * pow(coord[1], ly)
                                                * pow(coord[2], lz) * e;
                                        i++;
                                }
                        }
                        gto += ncart;
                }
        }
}

static void grid_cart_gto_grad(double *gto, double coord[3], double *exps,
                               int l, int np, int ngto)
{
        int ncart;
        int lx, ly, lz, i, k;
        double xinv, yinv, zinv;
        double ax, ay, az, e, tmp;
        double *gtox = gto + ngto;
        double *gtoy = gto + ngto * 2;
        double *gtoz = gto + ngto * 3;
        double *exps_2a = exps + NPRIMAX;
        switch (l) {
        case 0:
                for (k = 0; k < np; k++) {
                        gto [k] = exps[k];
                        gtox[k] = exps_2a[k] * coord[0];
                        gtoy[k] = exps_2a[k] * coord[1];
                        gtoz[k] = exps_2a[k] * coord[2];
                }
                break;
        case 1:
                for (k = 0; k < np; k++) {
                        ax = exps_2a[k] * coord[0];
                        ay = exps_2a[k] * coord[1];
                        az = exps_2a[k] * coord[2];
                        e = exps[k];
                        gto[0] = coord[0] * e;
                        gto[1] = coord[1] * e;
                        gto[2] = coord[2] * e;
                        gtox[0] = ax * coord[0] + e;
                        gtox[1] = ax * coord[1];
                        gtox[2] = ax * coord[2];
                        gtoy[0] = ay * coord[0];
                        gtoy[1] = ay * coord[1] + e;
                        gtoy[2] = ay * coord[2];
                        gtoz[0] = az * coord[0];
                        gtoz[1] = az * coord[1];
                        gtoz[2] = az * coord[2] + e;
                        gto  += 3;
                        gtox += 3;
                        gtoy += 3;
                        gtoz += 3;
                }
                break;
        case 2:
                for (k = 0; k < np; k++) {
                        tmp = exps_2a[k]/(exps[k]+1e-200);
                        ax = tmp * coord[0];
                        ay = tmp * coord[1];
                        az = tmp * coord[2];
                        e = exps[k];
                        gto[0] = coord[0] * coord[0] * e; // xx
                        gto[1] = coord[0] * coord[1] * e; // xy
                        gto[2] = coord[0] * coord[2] * e; // xz
                        gto[3] = coord[1] * coord[1] * e; // yy
                        gto[4] = coord[1] * coord[2] * e; // yz
                        gto[5] = coord[2] * coord[2] * e; // zz
                        gtox[0] = ax * gto[0] + 2 * coord[0] * e;
                        gtox[1] = ax * gto[1] +     coord[1] * e;
                        gtox[2] = ax * gto[2] +     coord[2] * e;
                        gtox[3] = ax * gto[3];
                        gtox[4] = ax * gto[4];
                        gtox[5] = ax * gto[5];
                        gtoy[0] = ay * gto[0];
                        gtoy[1] = ay * gto[1] +     coord[0] * e;
                        gtoy[2] = ay * gto[2];
                        gtoy[3] = ay * gto[3] + 2 * coord[1] * e;
                        gtoy[4] = ay * gto[4] +     coord[2] * e;
                        gtoy[5] = ay * gto[5];
                        gtoz[0] = az * gto[0];
                        gtoz[1] = az * gto[1];
                        gtoz[2] = az * gto[2] +     coord[0] * e;
                        gtoz[3] = az * gto[3];
                        gtoz[4] = az * gto[4] +     coord[1] * e;
                        gtoz[5] = az * gto[5] + 2 * coord[2] * e;
                        gto  += 6;
                        gtox += 6;
                        gtoy += 6;
                        gtoz += 6;
                }
                break;
        case 3:
                for (k = 0; k < np; k++) {
                        tmp = exps_2a[k]/(exps[k]+1e-200);
                        ax = tmp * coord[0];
                        ay = tmp * coord[1];
                        az = tmp * coord[2];
                        e = exps[k];
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
                        gtox[3] = ax * gto[3] +     coord[1] * coord[1] * e;
                        gtox[4] = ax * gto[4] +     coord[1] * coord[2] * e;
                        gtox[5] = ax * gto[5] +     coord[2] * coord[2] * e;
                        gtox[6] = ax * gto[6];
                        gtox[7] = ax * gto[7];
                        gtox[8] = ax * gto[8];
                        gtox[9] = ax * gto[9];
                        gtoy[0] = ay * gto[0];
                        gtoy[1] = ay * gto[1] +     coord[0] * coord[0] * e;
                        gtoy[2] = ay * gto[2];
                        gtoy[3] = ay * gto[3] + 2 * coord[0] * coord[1] * e;
                        gtoy[4] = ay * gto[4] +     coord[0] * coord[2] * e;
                        gtoy[5] = ay * gto[5];
                        gtoy[6] = ay * gto[6] + 3 * coord[1] * coord[1] * e;
                        gtoy[7] = ay * gto[7] + 2 * coord[1] * coord[2] * e;
                        gtoy[8] = ay * gto[8] +     coord[2] * coord[2] * e;
                        gtoy[9] = ay * gto[9];
                        gtoz[0] = az * gto[0];
                        gtoz[1] = az * gto[1];
                        gtoz[2] = az * gto[2] +     coord[0] * coord[0] * e;
                        gtoz[3] = az * gto[3];
                        gtoz[4] = az * gto[4] +     coord[0] * coord[1] * e;
                        gtoz[5] = az * gto[5] + 2 * coord[0] * coord[2] * e;
                        gtoz[6] = az * gto[6];
                        gtoz[7] = az * gto[7] +     coord[1] * coord[1] * e;
                        gtoz[8] = az * gto[8] + 2 * coord[1] * coord[2] * e;
                        gtoz[9] = az * gto[9] + 3 * coord[2] * coord[2] * e;
                        gto  += 10;
                        gtox += 10;
                        gtoy += 10;
                        gtoz += 10;
                }
                break;
        default:
                ncart = CINTlen_cart(l);
                for (k = 0; k < np; k++) {
                        xinv = 1/(coord[0]+1e-200);
                        yinv = 1/(coord[1]+1e-200);
                        zinv = 1/(coord[2]+1e-200);
                        tmp = exps_2a[k]/(exps[k]+1e-200);
                        ax = tmp * coord[0];
                        ay = tmp * coord[1];
                        az = tmp * coord[2];
                        e = exps[k];
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
                        gto  += ncart;
                        gtox += ncart;
                        gtoy += ncart;
                        gtoz += ncart;
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

static void _contract_exp(double *ectr, double *coord, double *alpha,
                          double *coeff, int l, int nprim, int nctr)
{
        int i, j;
        double rr = coord[0]*coord[0] + coord[1]*coord[1] + coord[2]*coord[2];
        double eprim[NPRIM_CART];
        double arr;
        double fac = CINTcommon_fac_sp(l);
        for (i = 0; i < nprim; i++) {
                arr = alpha[i] * rr;
                if (arr < 80) { // exp(-80) ~ 1e-35
                        eprim[i] = exp(-arr) * fac;
                } else {
                        eprim[i] = 0;
                }
        }
        for (i = 0; i < nctr; i++) {
                ectr[i] = 0;
                for (j = 0; j < nprim; j++) {
                        ectr[i] += eprim[j] * coeff[j];
                }
                coeff += nprim;
        }
}

static void _contract_exp_grad(double *ectr, double *coord, double *alpha,
                               double *coeff, int l, int nprim, int nctr)
{
        int i, j;
        double rr = coord[0]*coord[0] + coord[1]*coord[1] + coord[2]*coord[2];
        double arr, tmp;
        double fac = CINTcommon_fac_sp(l);
        double eprim[NPRIM_CART];
        double *ectr_2a = ectr + NPRIMAX;
        for (i = 0; i < nprim; i++) {
                arr = alpha[i] * rr;
                if (arr < 80) {
                        eprim[i] = exp(-arr) * fac;
                } else {
                        eprim[i] = 0;
                }
        }
        for (i = 0; i < nctr; i++) {
                ectr   [i] = 0;
                ectr_2a[i] = 0;
                for (j = 0; j < nprim; j++) {
                        tmp = eprim[j] * coeff[j];
                        ectr   [i] += tmp;
                        ectr_2a[i] += tmp * -2 * alpha[j];
                }
                coeff += nprim;
        }
}

static void _part_cp(double *out, double *in, int nrow, int ncol,
                     int ldo, int ldi)
{
        int i, j;
        for (i = 0; i < nrow; i++) {
                for (j = 0; j < ncol; j++) {
                        out[j] = in[j];
                }
                in  += ldi;
                out += ldo;
        }
}

// ao[:nao,:ngrids] in Fortran-order
void VXCeval_nr_gto(int nao, int ngrids, double *ao, double *coord,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int l, ig, np, nc, nc_cart, atm_id, bas_id, deg;
        int *pbas;
        double *r_atm, *p_exp, *pcoeff;
        double *pgto;
        double ectr[NPRIMAX];
        double *cart_gto = malloc(sizeof(double)*NPRIM_CART*ngrids);
        double *pgto_buf = malloc(sizeof(double)*NPRIM_CART*ngrids);
        double dcoord[3];
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                pbas = bas + bas_id*BAS_SLOTS;
                np = pbas[NPRIM_OF];
                nc = pbas[NCTR_OF ];
                l  = pbas[ANG_OF  ];
                nc_cart = nc * CINTlen_cart(l);
                deg = l * 2 + 1;
                p_exp = &env[pbas[PTR_EXP]];
                pcoeff = &env[pbas[PTR_COEFF]];
                atm_id = pbas[ATOM_OF];
                r_atm = &env[atm[PTR_COORD+atm_id*ATM_SLOTS]];
                for (ig = 0; ig < ngrids; ig++) {
                        dcoord[0] = coord[ig*3+0] - r_atm[0];
                        dcoord[1] = coord[ig*3+1] - r_atm[1];
                        dcoord[2] = coord[ig*3+2] - r_atm[2];
                        _contract_exp(ectr, dcoord, p_exp, pcoeff, l, np, nc);
                        grid_cart_gto(cart_gto+ig*nc_cart, dcoord, ectr, l, nc);
                }
                pgto = CINTc2s_bra_sph(pgto_buf, nc*ngrids, cart_gto, l);
                _part_cp(ao, pgto, ngrids, nc*deg, nao, nc*deg);
                ao += nc * deg;
        }
        free(cart_gto);
        free(pgto_buf);
}

// in ao[:nao,:ngrids,:4] in Fortran-order, [:4] ~ ao, ao_dx, ao_dy, ao_dz
void VXCeval_nr_gto_grad(int nao, int ngrids, double *ao, double *coord,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int i, ig;
        int l, np, nc, nc_cart, atm_id, bas_id, deg;
        int *pbas;
        double *r_atm, *p_exp, *pcoeff;
        double dcoord[3];
        double ectr[NPRIMAX*2];
        double *cart_gto = malloc(sizeof(double)*NPRIM_CART*ngrids*4);
        double *pgto_buf = malloc(sizeof(double)*NPRIM_CART*ngrids);
        double *pgto;
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                pbas = bas + bas_id*BAS_SLOTS;
                np = pbas[NPRIM_OF];
                nc = pbas[NCTR_OF ];
                l  = pbas[ANG_OF  ];
                nc_cart = nc * CINTlen_cart(l);
                deg = l * 2 + 1;
                p_exp = &env[pbas[PTR_EXP]];
                pcoeff = &env[pbas[PTR_COEFF]];
                atm_id = pbas[ATOM_OF];
                r_atm = &env[atm[PTR_COORD+atm_id*ATM_SLOTS]];
                for (ig = 0; ig < ngrids; ig++) {
                        dcoord[0] = coord[ig*3+0] - r_atm[0];
                        dcoord[1] = coord[ig*3+1] - r_atm[1];
                        dcoord[2] = coord[ig*3+2] - r_atm[2];
                        _contract_exp_grad(ectr, dcoord, p_exp, pcoeff, l, np, nc);
                        grid_cart_gto_grad(cart_gto+ig*nc_cart, dcoord, ectr,
                                           l, nc, nc_cart*ngrids);
                }
                for (i = 0; i < 4; i++) {
                        pgto = CINTc2s_bra_sph(pgto_buf, nc*ngrids,
                                               cart_gto+i*nc_cart*ngrids, l);
                        _part_cp(ao+i*nao*ngrids, pgto,
                                 ngrids, nc*deg, nao, nc*deg);
                }
                ao += nc * deg;
        }
        free(cart_gto);
        free(pgto_buf);
}
