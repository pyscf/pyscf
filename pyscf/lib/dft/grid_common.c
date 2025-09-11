/* Copyright 2021-2024 The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <limits.h>
#include "config.h"
#include "vhf/fblas.h"
#include "dft/multigrid.h"
#include "dft/grid_common.h"
#if defined(HAVE_LIBXSMM)
    #include "libxsmm.h"
#endif
#define EXPMIN          -700

int get_lmax(int ish0, int ish1, int* bas)
{
    int lmax = 0;
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        lmax = MAX(lmax, bas[ANG_OF+ish*BAS_SLOTS]);
    }
    return lmax;
}


int get_nprim_max(int ish0, int ish1, int* bas)
{
    int nprim_max = 1;
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        nprim_max = MAX(nprim_max, bas[NPRIM_OF+ish*BAS_SLOTS]);
    }
    return nprim_max;
}


int get_nctr_max(int ish0, int ish1, int* bas)
{
    int nctr_max = 1;
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        nctr_max = MAX(nctr_max, bas[NCTR_OF+ish*BAS_SLOTS]);
    }
    return nctr_max;
}


void get_cart2sph_coeff(double** contr_coeff, double** gto_norm, 
                        int ish0, int ish1, int* bas, double* env, int cart)
{
    int l;
    int lmax = get_lmax(ish0, ish1, bas);
    int nprim, ncart, nsph, nctr;
    int ptr_exp, ptr_coeff;
    int ish, ipgf, ic, i, j;

    double **c2s = (double**) malloc(sizeof(double*) * (lmax+1));
    for (l = 0; l <= lmax; l++) {
        ncart = _LEN_CART[l];
        if (l <= 1 || cart == 1) {
            c2s[l] = (double*) calloc(ncart*ncart, sizeof(double));
            for (i = 0; i < ncart; i++) {
                c2s[l][i*ncart + i] = 1;
            }
        }
        else {
            nsph = 2*l + 1;
            c2s[l] = (double*) calloc(nsph*ncart, sizeof(double));
            double* gcart = (double*) calloc(ncart*ncart, sizeof(double));
            for (i = 0; i < ncart; i++) {
                gcart[i*ncart + i] = 1;
            }
            CINTc2s_ket_sph(c2s[l], ncart, gcart, l);
            free(gcart);
        }
    }

#pragma omp parallel private (ish, ipgf, ic, i, j, l,\
                              ncart, nsph, nprim, nctr,\
                              ptr_exp, ptr_coeff)
{
    #pragma omp for schedule(dynamic) 
    for (ish = ish0; ish < ish1; ish++) {
        l = bas[ANG_OF+ish*BAS_SLOTS];
        ncart = _LEN_CART[l];
        nsph = cart == 1 ? ncart : 2*l+1;
        nprim = bas[NPRIM_OF+ish*BAS_SLOTS];
        nctr = bas[NCTR_OF+ish*BAS_SLOTS];

        ptr_exp = bas[PTR_EXP+ish*BAS_SLOTS];
        gto_norm[ish] = (double*) malloc(sizeof(double) * nprim);
        for (ipgf = 0; ipgf < nprim; ipgf++) {
            gto_norm[ish][ipgf] = CINTgto_norm(l, env[ptr_exp+ipgf]);
        }

        ptr_coeff = bas[PTR_COEFF+ish*BAS_SLOTS];
        double *buf = (double*) calloc(nctr*nprim, sizeof(double));
        for (ipgf = 0; ipgf < nprim; ipgf++) {
            double inv_norm = 1./gto_norm[ish][ipgf];
            daxpy_(&nctr, &inv_norm, env+ptr_coeff+ipgf, &nprim, buf+ipgf, &nprim);
        }

        contr_coeff[ish] = (double*) malloc(sizeof(double) * nprim*ncart*nctr*nsph);
        double* ptr_contr_coeff = contr_coeff[ish];
        for (ipgf = 0; ipgf < nprim; ipgf++) {
            for (i = 0; i < ncart; i++) {
                for (ic = 0; ic < nctr; ic++) {
                    for (j = 0; j < nsph; j++) {
                        *ptr_contr_coeff = buf[ic*nprim+ipgf] * c2s[l][j*ncart+i];
                        ptr_contr_coeff += 1;
                    }
                }
            }
        }
        free(buf);
    }
}

    for (l = 0; l <= lmax; l++) {
        free(c2s[l]);
    }
    free(c2s);
}


void del_cart2sph_coeff(double** contr_coeff, double** gto_norm, int ish0, int ish1)
{
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        if (contr_coeff[ish]) {
            free(contr_coeff[ish]);
        }
        if (gto_norm[ish]) {
            free(gto_norm[ish]);
        }
    }
    free(contr_coeff);
    free(gto_norm);
}


int get_max_num_grid_orth(double* dh, double radius)
{
    double dx = MIN(MIN(dh[0], dh[4]), dh[8]);
    int ngrid = 2 * (int) ceil(radius / dx) + 1;
    return ngrid;
}


int get_max_num_grid_nonorth(double* dh_inv, double radius)
{
    int i, j, k, ia;
    int lb[3], ub[3];
    for (i = 0; i < 3; i++) {
        lb[i] = INT_MAX;
        ub[i] = INT_MIN;
    }

    double r[3], r_frac[3];
    for (i = -1; i <= 1; i++) {
        r[0] = i * radius;
        for (j = -1; j <= 1; j++) {
            r[1] = j * radius;
            for (k = -1; k <= 1; k++) {
                r[2] = k * radius;

                get_lattice_coords(r_frac, r, dh_inv);
                for (ia = 0; ia < 3; ia++) {
                    lb[ia] = MIN(lb[ia], (int)floor(r_frac[ia]));
                    ub[ia] = MAX(ub[ia], (int) ceil(r_frac[ia]));
                }
            }
        }
    }

    int nx = ub[0] - lb[0];
    int ny = ub[1] - lb[1];
    int nz = ub[2] - lb[2];
    int nmax = MAX(MAX(nx, ny), nz) + 1;
    return nmax;
}


void get_grid_spacing(double* dh, double *dh_inv, double* a, double* b, int* mesh)
{
    int i, j;
    for (i = 0; i < 3; i++) {
        const int ni = mesh[i];
        for (j = 0; j < 3; j++) {
            dh[i*3+j] = a[i*3+j] / ni;
            dh_inv[j*3+i] = b[i*3+j] * ni;
        }
    }
}


static int _orth_components(double *xs_exp, int* bounds, double dx, double radius,
                            double xi, double xj, double ai, double aj,
                            int nx_per_cell, int topl, double *cache)
{
    double aij = ai + aj;
    double xij = (ai * xi + aj * xj) / aij;
    int x0_latt = (int) floor((xij - radius) / dx);
    int x1_latt = (int) ceil((xij + radius) / dx);
    int xij_latt = (int) rint(xij / dx);
    xij_latt = MAX(xij_latt, x0_latt);
    xij_latt = MIN(xij_latt, x1_latt);
    bounds[0] = x0_latt;
    bounds[1] = x1_latt;
    int ngridx = x1_latt - x0_latt;

    double base_x = dx * xij_latt;
    double x0xij = base_x - xij;
    double _x0x0 = -aij * x0xij * x0xij;
    if (_x0x0 < EXPMIN) {
        return 0;
    }

    double *gridx = cache;
    double *xs_all = xs_exp;
    if (ngridx > nx_per_cell) {
        xs_all = gridx + ngridx;
    }

    double _dxdx = -aij * dx * dx;
    double _x0dx = -2 * aij * x0xij * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_dxdx + _x0dx);
    double exp_x0x0_cache = exp(_x0x0);
    double exp_x0x0 = exp_x0x0_cache;

    int i;
    int istart = xij_latt - x0_latt;
    for (i = istart; i < ngridx; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp_x0x0_cache;
    for (i = istart-1; i >= 0; i--) {
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
        xs_all[i] = exp_x0x0;
    }

    if (topl > 0) {
        double x0xi = x0_latt * dx - xi;
        for (i = 0; i < ngridx; i++) {
            gridx[i] = x0xi + i * dx;
        }
        int l;
        double *px0;
        for (l = 1; l <= topl; l++) {
            px0 = xs_all + (l-1) * ngridx;
            for (i = 0; i < ngridx; i++) {
                px0[ngridx+i] = px0[i] * gridx[i];
            }
        }
    }

    // add up contributions from all images to the reference image
    if (ngridx > nx_per_cell) {
        memset(xs_exp, 0, (topl+1)*nx_per_cell*sizeof(double));
        int ix, l, lb, ub, size_x;
        for (ix = 0; ix < ngridx;) {
            lb = modulo(ix + x0_latt, nx_per_cell);
            ub = get_upper_bound(lb, nx_per_cell, ix, ngridx);
            size_x = ub - lb;
            double* __restrict ptr_xs_exp = xs_exp + lb;
            double* __restrict ptr_xs_all = xs_all + ix;
            for (l = 0; l <= topl; l++) {
                //#pragma omp simd
                PRAGMA_OMP_SIMD
                for (i = 0; i < size_x; i++) {
                    ptr_xs_exp[i] += ptr_xs_all[i];
                }
                ptr_xs_exp += nx_per_cell;
                ptr_xs_all += ngridx;
            }
            ix += size_x;
        }

        bounds[0] = 0;
        bounds[1] = nx_per_cell;
        ngridx = nx_per_cell;
    }
    return ngridx;
}


int init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                   int *grid_slice, double* dh, int* mesh, int topl, double radius,
                   double ai, double aj, double *ri, double *rj, double *cache)
{
    int l1 = topl + 1;
    *xs_exp = cache;
    *ys_exp = *xs_exp + l1 * mesh[0];
    *zs_exp = *ys_exp + l1 * mesh[1];
    int data_size = l1 * (mesh[0] + mesh[1] + mesh[2]);
    cache += data_size;

    int ngridx = _orth_components(*xs_exp, grid_slice, dh[0], radius,
                                  ri[0], rj[0], ai, aj, mesh[0], topl, cache);
    if (ngridx == 0) {
        return 0;
    }

    int ngridy = _orth_components(*ys_exp, grid_slice+2, dh[4], radius,
                                  ri[1], rj[1], ai, aj, mesh[1], topl, cache);
    if (ngridy == 0) {
        return 0;
    }

    int ngridz = _orth_components(*zs_exp, grid_slice+4, dh[8], radius,
                                  ri[2], rj[2], ai, aj, mesh[2], topl, cache);
    if (ngridz == 0) {
        return 0;
    }

    return data_size;
}


/*
static void _orth_bounds(int* bounds, int* rp_latt, double* roff,
                         double* rp, double* dh, double radius)
{
    double rx = rp[0];
    double ry = rp[1];
    double rz = rp[2];

    double dx = dh[0];
    double dy = dh[4];
    double dz = dh[8];

    bounds[0] = (int)floor((rx - radius) / dx);
    bounds[1] = (int) ceil((rx + radius) / dx);
    bounds[2] = (int)floor((ry - radius) / dy);
    bounds[3] = (int) ceil((ry + radius) / dy);
    bounds[4] = (int)floor((rz - radius) / dz);
    bounds[5] = (int) ceil((rz + radius) / dz);

    int cx = (int)rint(rx / dx);
    int cy = (int)rint(ry / dy);
    int cz = (int)rint(rz / dz);

    rp_latt[0] = cx;
    rp_latt[1] = cy;
    rp_latt[2] = cz;

    roff[0] = dx * cx - rx;
    roff[1] = dy * cy - ry;
    roff[2] = dz * cz - rz;
}


static void _nonorth_bounds(int* bounds, int* rp_latt, double* roff,
                            double* rp, double* dh_inv, double radius)
{
    int i, j, k, ia;
    double r[3], r_frac[3];

    get_lattice_coords(r_frac, rp, dh_inv);
    rp_latt[0] = (int)rint(r_frac[0]);
    rp_latt[1] = (int)rint(r_frac[1]);
    rp_latt[2] = (int)rint(r_frac[2]);

    roff[0] = rp_latt[0] - r_frac[0];
    roff[1] = rp_latt[1] - r_frac[1];
    roff[2] = rp_latt[2] - r_frac[2];

    for (i = 0; i < 6; i += 2) {
        bounds[i] = INT_MAX;
        bounds[i + 1] = INT_MIN;
    }

    // A parallelepiped containting the cube that contains the sphere.
    // TODO compute more precise bounds
    for (i = -1; i <= 1; i++) {
        r[0] = rp[0] + i * radius;
        for (j = -1; j <= 1; j++) {
            r[1] = rp[1] + j * radius;
            for (k = -1; k <= 1; k++) {
                r[2] = rp[2] + k * radius;

                get_lattice_coords(r_frac, r, dh_inv);
                for (ia = 0; ia < 3; ia++) {
                    bounds[ia * 2] = MIN(bounds[ia * 2], (int)floor(r_frac[ia]));
                    bounds[ia * 2 + 1] = MAX(bounds[ia * 2 + 1], (int)ceil(r_frac[ia]));
                }
            }
        }
    }
}
*/


static void _nonorth_bounds_tight(int* bounds, int* rp_latt, double* roff,
                                  double* rp, double* dh, double* dh_inv, double radius)
{
    int i, j, ia;
    double r_frac[3];

    get_lattice_coords(r_frac, rp, dh_inv);
    rp_latt[0] = (int)rint(r_frac[0]);
    rp_latt[1] = (int)rint(r_frac[1]);
    rp_latt[2] = (int)rint(r_frac[2]);

    roff[0] = rp_latt[0] - r_frac[0];
    roff[1] = rp_latt[1] - r_frac[1];
    roff[2] = rp_latt[2] - r_frac[2];

    for (i = 0; i < 6; i += 2) {
        bounds[i] = INT_MAX;
        bounds[i + 1] = INT_MIN;
    }

    double a_norm[3], e[3][3];
    for (i = 0; i < 3; i++) {
        double *a = dh + i * 3;
        a_norm[i] = vnorm(a);
        vscale(e[i], 1. / a_norm[i], a);
    }

    const int idx[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (i = 0; i < 3; i++) {
        int i1 = idx[i][0];
        int i2 = idx[i][1];
        double *a1 = dh + i1 * 3;
        double *a2 = dh + i2 * 3;
        double theta = .5 * acos(vdot(a1, a2) / (a_norm[i1] * a_norm[i2]));
        double r1 = radius / sin(theta);
        double r2 = r1 * cos(theta);

        double *e1 = e[i1], *e2 = e[i2];
        double e12[3];
        vadd(e12, e1, e2);
        double e12_norm = vnorm(e12);
        vscale(e12, 1. / e12_norm * r1, e12);

        double rp_plus_e12[3], rp_minus_e12[3];
        vadd(rp_plus_e12, rp, e12);
        vsub(rp_minus_e12, rp, e12);

        double e1_times_r2[3], e2_times_r2[3];
        vscale(e1_times_r2, r2, e1);
        vscale(e2_times_r2, r2, e2);

        //four points where the polygon is tangent to the circle
        double c[4][3];
        vsub(c[0], rp_plus_e12, e2_times_r2);
        vsub(c[1], rp_plus_e12, e1_times_r2);
        vadd(c[2], rp_minus_e12, e2_times_r2);
        vadd(c[3], rp_minus_e12, e1_times_r2);

        for (j = 0; j < 4; j++) {
            get_lattice_coords(r_frac, c[j], dh_inv);
            for (ia = 0; ia < 3; ia++) {
                bounds[ia * 2] = MIN(bounds[ia * 2], (int)floor(r_frac[ia]));
                bounds[ia * 2 + 1] = MAX(bounds[ia * 2 + 1], (int)ceil(r_frac[ia]));
            }
        }
    }
}


int get_max_num_grid_nonorth_tight(double*dh, double* dh_inv, double radius)
{
    int bounds[6];
    int rp_latt[3];
    double roff[3];
    double rp[3] = {0};

    _nonorth_bounds_tight(bounds, rp_latt, roff, rp, dh, dh_inv, radius);

    int nx = bounds[1] - bounds[0];
    int ny = bounds[3] - bounds[2];
    int nz = bounds[5] - bounds[4];
    int nmax = MAX(MAX(nx, ny), nz) + 1;
    return nmax;
}


static void _poly_exp(double *xs_all, int* bounds, double dx,
                      double xi, double xoff, int xp_latt, double ap,
                      int topl, double *cache)
{
    int x0_latt = bounds[0];
    int x1_latt = bounds[1];
    int ngridx = x1_latt - x0_latt;

    double _x0x0 = -ap * xoff * xoff;
    if (_x0x0 < EXPMIN) {
        return;
    }

    double _dxdx = -ap * dx * dx;
    double _x0dx = -2 * ap * xoff * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_dxdx + _x0dx);
    double exp_x0x0_cache = exp(_x0x0);
    double exp_x0x0 = exp_x0x0_cache;

    int i;
    int istart = xp_latt - x0_latt;
    for (i = istart; i < ngridx; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp_x0x0_cache;
    for (i = istart-1; i >= 0; i--) {
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
        xs_all[i] = exp_x0x0;
    }

    if (topl > 0) {
        double *gridx = cache;
        double x0xi = x0_latt * dx - xi;
        for (i = 0; i < ngridx; i++) {
            gridx[i] = x0xi + i * dx;
        }
        int l;
        double *px0 = xs_all;
        double *px1 = px0 + ngridx;
        for (l = 1; l <= topl; l++) {
            for (i = 0; i < ngridx; i++) {
                px1[i] = px0[i] * gridx[i];
            }
            px0 += ngridx;
            px1 += ngridx;
        }
    }
}


static void _nonorth_exp_i(double* out, int* bounds, int i0, double alpha)
{
    int i;
    int istart = bounds[0];
    int iend = bounds[1];
    const double c_exp = exp(alpha);

    out[0] = exp(alpha * (istart - i0));
    for (i = 1; i < (iend - istart); i++) {
        out[i] = out[i-1] * c_exp;
    }
}


static void _nonorth_exp_ij(double* out, int* bounds_i, int* bounds_j,
                            int i0, int j0, double alpha)
{
    int i, j;
    const int istart = bounds_i[0];
    const int iend = bounds_i[1];
    const int ni = iend - istart;
    const int jstart = bounds_j[0];
    const int jend = bounds_j[1];
    const int nj = jend - jstart;

    double c_exp = exp(alpha);
    double c_exp_i = exp(alpha * (istart - i0));
    double ctmp, c_exp_j;
    double *pout;

    ctmp = c_exp_j = 1.;
    for (j = j0 - jstart; j < nj; j++) {
        pout = out + ni * j;
        double ctmp1 = ctmp;
        for (i = 0; i < ni; i++) {
            pout[i] *= ctmp1;
            ctmp1 *= c_exp_j;
        }
        ctmp *= c_exp_i;
        c_exp_j *= c_exp;
    }

    c_exp_i = 1. / c_exp_i;
    c_exp = 1. / c_exp;
    ctmp = c_exp_i;
    c_exp_j = c_exp;
    for (j = j0 - jstart - 1; j >= 0; j--) {
        pout = out + ni * j;
        double ctmp1 = ctmp;
        for (i = 0; i < ni; i++) {
            pout[i] *= ctmp1;
            ctmp1 *= c_exp_j;
        }
        ctmp *= c_exp_i;
        c_exp_j *= c_exp;
    }
}


static void _nonorth_exp_correction(double* exp_corr, int* bounds,
                                    double* dh, double* roff, int* rp_latt,
                                    double ap, double* cache)
{
    const int idx[3][2] = {{1, 0}, {2, 1}, {2, 0}};
    const double c[3] = {
        //a1 * a2
        -2. * ap * (dh[0] * dh[3] + dh[1] * dh[4] + dh[2] * dh[5]),
        //a2 * a3
        -2. * ap * (dh[6] * dh[3] + dh[7] * dh[4] + dh[8] * dh[5]),
        //a3 * a1
        -2. * ap * (dh[0] * dh[6] + dh[1] * dh[7] + dh[2] * dh[8])
    };

    const int ng[3] = {
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    };
    const int nmax = MAX(MAX(ng[0], ng[1]), ng[2]);
    const int I1 = 1;
    double *exp1 = cache;
    double *exp2 = exp1 + nmax;

    int i;
    for (i = 0; i < 3; i++) {
        int i1 = idx[i][0];
        int i2 = idx[i][1];
        double c_exp = exp(c[i] * roff[i1] * roff[i2]);

        _nonorth_exp_i(exp1, bounds+i1*2, rp_latt[i1], c[i] * roff[i2]);
        _nonorth_exp_i(exp2, bounds+i2*2, rp_latt[i2], c[i] * roff[i1]);

        int n1 = ng[i1];
        int n2 = ng[i2];
        const size_t n12 = (size_t)n1 * n2;
        memset(exp_corr, 0, n12 * sizeof(double));
        dger_(&n1, &n2, &c_exp, exp1, &I1, exp2, &I1, exp_corr, &n1);

        _nonorth_exp_ij(exp_corr, bounds+i1*2, bounds+i2*2,
                        rp_latt[i1], rp_latt[i2], c[i]);
        exp_corr += n12;
    }
}


size_t init_nonorth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                         double **exp_corr, int *bounds,
                         double* dh, double* dh_inv,
                         int* mesh, int topl, double radius,
                         double ai, double aj, double *ri, double *rj, double *cache)
{
    int l1 = topl + 1;
    int rp_latt[3];
    double ap = ai + aj;
    double bas2[3], rp[3], roff[3], ri_frac[3];

    bas2[0] = dh[0] * dh[0] + dh[1] * dh[1] + dh[2] * dh[2];
    bas2[1] = dh[3] * dh[3] + dh[4] * dh[4] + dh[5] * dh[5];
    bas2[2] = dh[6] * dh[6] + dh[7] * dh[7] + dh[8] * dh[8];

    get_lattice_coords(ri_frac, ri, dh_inv);

    rp[0] = (ai * ri[0] + aj * rj[0]) / ap;
    rp[1] = (ai * ri[1] + aj * rj[1]) / ap;
    rp[2] = (ai * ri[2] + aj * rj[2]) / ap;

    //_nonorth_bounds(bounds, rp_latt, roff, rp, dh_inv, radius);
    _nonorth_bounds_tight(bounds, rp_latt, roff, rp, dh, dh_inv, radius);

    *xs_exp = cache;
    int ngridx = bounds[1] - bounds[0];
    if (ngridx <= 0) {
        return 0;
    }
    cache += l1 * ngridx;
    _poly_exp(*xs_exp, bounds, 1., ri_frac[0],
              roff[0], rp_latt[0], ap * bas2[0], topl, cache);

    *ys_exp = cache;
    int ngridy = bounds[3] - bounds[2];
    if (ngridy <= 0) {
        return 0;
    }
    cache += l1 * ngridy;
    _poly_exp(*ys_exp, bounds+2, 1., ri_frac[1],
              roff[1], rp_latt[1], ap * bas2[1], topl, cache);

    *zs_exp = cache;
    int ngridz = bounds[5] - bounds[4];
    if (ngridz <= 0) {
        return 0;
    }
    cache += l1 * ngridz;
    _poly_exp(*zs_exp, bounds+4, 1., ri_frac[2],
              roff[2], rp_latt[2], ap * bas2[2], topl, cache);

    *exp_corr = cache;
    size_t exp_corr_size = (size_t)ngridx * ngridy
                         + (size_t)ngridy * ngridz
                         + (size_t)ngridz * ngridx;
    cache += exp_corr_size;
    _nonorth_exp_correction(*exp_corr, bounds,
                            dh, roff, rp_latt,
                            ap, cache);

    size_t data_size = l1 * (ngridx + ngridy + ngridz) + exp_corr_size;
    return data_size;
}


void get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache)
{
    int l1 = lmax + 1;
    int l, lx;

    double *rx_pow = cache;
    double *ry_pow = rx_pow + l1;
    double *rz_pow = ry_pow + l1;

    rx_pow[0] = 1.0;
    ry_pow[0] = 1.0;
    rz_pow[0] = 1.0;
    for (lx = 1; lx <= lmax; lx++) {
        rx_pow[lx] = rx_pow[lx-1] * rij[0];
        ry_pow[lx] = ry_pow[lx-1] * rij[1];
        rz_pow[lx] = rz_pow[lx-1] * rij[2];
    }

    int dj = _LEN_CART[lmax];
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    for (l = 0; l <= lmax; l++){
        for (lx = 0; lx <= l; lx++) {
            pcx[lx] = BINOMIAL(l, lx) * rx_pow[l-lx];
            pcy[lx] = BINOMIAL(l, lx) * ry_pow[l-lx];
            pcz[lx] = BINOMIAL(l, lx) * rz_pow[l-lx];
        }
        pcx += l+1;
        pcy += l+1;
        pcz += l+1;
    }
}


void dm_to_dm_xyz(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache)
{
    int lx, ly, lz;
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    int jx, jy, jz;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int l1 = li + lj + 1;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj];
    cache += 3 * dj;

    get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

    double cx, cxy, cxyz;
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pdm = dm;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    for (jx = 0; jx <= lx_j; jx++) {
                        cx = pcx[jx+_LEN_CART0[lx_j]];
                        lx = lx_i + jx;
                        for (jy = 0; jy <= ly_j; jy++) {
                            cxy = cx * pcy[jy+_LEN_CART0[ly_j]];
                            ly = ly_i + jy;
                            for (jz = 0; jz <= lz_j; jz++) {
                                cxyz = cxy * pcz[jz+_LEN_CART0[lz_j]];
                                lz = lz_i + jz;
                                dm_xyz[lx*l1l1+ly*l1+lz] += cxyz * pdm[0];
                            }
                        }
                    }
                    pdm += 1;
                }
            }
        }
    }
}


void dm_xyz_to_dm_ijk(double* dm_ijk, double* dm_xyz, double* dh, int topl)
{
    if (topl == 0) {
        dm_ijk[0] = dm_xyz[0];
        return;
    }

    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    double dh_pow[l1][9];
    int i, l;
    for (i = 0; i < 9; i++) {
        dh_pow[0][i] = 1.;
        for (l = 1; l <= topl; l++) {
            dh_pow[l][i] = dh_pow[l - 1][i] * dh[i];
        }
    }

    int lx, ly, lz;
    int ix, jx, kx;
    int iy, jy, ky;
    int iz, jz, kz;
    for (lx = 0; lx <= topl; lx++) {
    for (ix = 0; ix <= lx; ix++) {
    for (jx = 0; jx <= lx-ix; jx++) {
        kx = lx - ix - jx;
        double cx = dh_pow[ix][0] * dh_pow[jx][3] * dh_pow[kx][6]
                  * fac(lx) / (fac(ix) * fac(jx) * fac(kx));

        for (ly = 0; ly <= topl - lx; ly++) {
        for (iy = 0; iy <= ly; iy++) {
        for (jy = 0; jy <= ly-iy; jy++) {
            ky = ly - iy - jy;
            double cxy = cx * dh_pow[iy][1] * dh_pow[jy][4] * dh_pow[ky][7]
                       * fac(ly) / (fac(iy) * fac(jy) * fac(ky)); 

            for (lz = 0; lz <= topl - lx - ly; lz++) {
                double dm_value = dm_xyz[lx*l1l1+ly*l1+lz];
            for (iz = 0; iz <= lz; iz++) {
            for (jz = 0; jz <= lz-iz; jz++) {
                kz = lz - iz - jz;
                double cxyz = cxy * dh_pow[iz][2] * dh_pow[jz][5] * dh_pow[kz][8]
                            * fac(lz) / (fac(iz) * fac(jz) * fac(kz));

                int li = ix + iy + iz;
                int lj = jx + jy + jz;
                int lk = kx + ky + kz;
                dm_ijk[li*l1l1+lj*l1+lk] += dm_value * cxyz;
            }}}
        }}}
    }}}
}


void dm_ijk_to_dm_xyz(double* dm_ijk, double* dm_xyz, double* dh, int topl)
{
    if (topl == 0) {
        dm_xyz[0] = dm_ijk[0];
        return;
    }

    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    double dh_pow[l1][9];
    int i, l;
    for (i = 0; i < 9; i++) {
        dh_pow[0][i] = 1.;
        for (l = 1; l <= topl; l++) {
            dh_pow[l][i] = dh_pow[l - 1][i] * dh[i];
        }
    }

    int lx, ly, lz;
    int ix, jx, kx;
    int iy, jy, ky;
    int iz, jz, kz;
    for (lx = 0; lx <= topl; lx++) {
    for (ix = 0; ix <= lx; ix++) {
    for (jx = 0; jx <= lx-ix; jx++) {
        kx = lx - ix - jx;
        double cx = dh_pow[ix][0] * dh_pow[jx][3] * dh_pow[kx][6]
                  * fac(lx) / (fac(ix) * fac(jx) * fac(kx));

        for (ly = 0; ly <= topl-lx; ly++) {
        for (iy = 0; iy <= ly; iy++) {
        for (jy = 0; jy <= ly-iy; jy++) {
            ky = ly - iy - jy;
            double cxy = cx * dh_pow[iy][1] * dh_pow[jy][4] * dh_pow[ky][7]
                       * fac(ly) / (fac(iy) * fac(jy) * fac(ky)); 

            for (lz = 0; lz <= topl-lx-ly ; lz++) {
                double *ptr_dm_xyz = dm_xyz + lx*l1l1+ly*l1+lz;
            for (iz = 0; iz <= lz; iz++) {
            for (jz = 0; jz <= lz-iz; jz++) {
                kz = lz - iz - jz;
                double cxyz = cxy * dh_pow[iz][2] * dh_pow[jz][5] * dh_pow[kz][8]
                            * fac(lz) / (fac(iz) * fac(jz) * fac(kz));

                int li = ix + iy + iz;
                int lj = jx + jy + jz;
                int lk = kx + ky + kz;

                *ptr_dm_xyz += dm_ijk[li*l1l1+lj*l1+lk] * cxyz;
            }}}
        }}}
    }}}
}


void dm_xyz_to_dm(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache)
{
    int lx, ly, lz;
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    int jx, jy, jz;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int l1 = li + lj + 1;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj];
    cache += 3 * dj;

    get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

    double cx, cy, cz;
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pdm = dm;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    for (jx = 0; jx <= lx_j; jx++) {
                        cx = pcx[jx+_LEN_CART0[lx_j]];
                        lx = lx_i + jx;
                        for (jy = 0; jy <= ly_j; jy++) {
                            cy = pcy[jy+_LEN_CART0[ly_j]];
                            ly = ly_i + jy;
                            for (jz = 0; jz <= lz_j; jz++) {
                                cz = pcz[jz+_LEN_CART0[lz_j]];
                                lz = lz_i + jz;
                                pdm[0] += cx*cy*cz * dm_xyz[lx*l1l1+ly*l1+lz];
                            }
                        }
                    }
                    pdm += 1;
                }
            }
        }
    }
}


void get_dm_pgfpair(double* dm_pgf, double* dm_cart, 
                    PGFPair* pgfpair, int* ish_bas, int* jsh_bas, int hermi)
{
    int ish = pgfpair->ish;
    int jsh = pgfpair->jsh;
    int ipgf = pgfpair->ipgf;
    int jpgf = pgfpair->jpgf;

    int li = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int di = _LEN_CART[li];
    int dj = _LEN_CART[lj];

    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int ncol = nprim_j * dj;
    double *pdm = dm_cart + (ipgf*di*ncol + jpgf*dj);
    double *pdm_pgf = dm_pgf;
    int i, j;
    for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
            pdm_pgf[j] = pdm[j];
        }
        pdm_pgf += dj;
        pdm += ncol;
    }

    /*
    if (hermi == 1 && ish == jsh) {
        assert(di == dj);
        for (i = 0; i < di; i++) {
            for (j = i+1; j < dj; j++) {
                dm_pgf[i*dj+j] *= 2;
                dm_pgf[j*dj+i] = 0;
            }
        }
    }*/
    if (hermi == 1 && ish != jsh) {
        pdm_pgf = dm_pgf;
        for (i = 0; i < di; i++) {
            for (j = 0; j < dj; j++) {
                pdm_pgf[j] *= 2;
            }
            pdm_pgf += dj;
        }
    }
}


void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc)
{
#if defined(HAVE_LIBXSMM)
    if (transa == 'N') {
        //libxsmm_dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        int prefetch = LIBXSMM_PREFETCH_AUTO;
        int flags = transb != 'T' ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B;
        libxsmm_dmmfunction kernel = libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc,
                                                         &alpha, &beta, &flags, &prefetch);
        if (kernel) {
            kernel(a,b,c,a,b,c);
            return;
        }
    }
#endif
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}


void get_gga_vrho_gs(double complex *vrho_gs, double complex *vsigma1_gs,
                     double *Gv, double fac, double weight, int ngrid)
{
    int i;
    int ngrid2 = 2 * ngrid;
    double complex zfac = -fac * _Complex_I;
#pragma omp parallel
{
    double complex v;
    #pragma omp for schedule(static)
    for (i = 0; i < ngrid; i++) {
        v = ( Gv[i*3]   * vsigma1_gs[i]
             +Gv[i*3+1] * vsigma1_gs[i+ngrid]
             +Gv[i*3+2] * vsigma1_gs[i+ngrid2]) * zfac;
        vrho_gs[i] += v;
        vrho_gs[i] *= weight;
    }
}
}
