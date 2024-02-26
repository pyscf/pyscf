/* Copyright 2021- The PySCF Developers. All Rights Reserved.

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

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "dft/multigrid.h"
#include "dft/grid_common.h"

#define EXPMIN         -700


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


void get_grid_spacing(double* dh, double* a, int* mesh)
{
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            dh[i*3+j] = a[i*3+j] / mesh[i];
        }
    }
}


int orth_components(double *xs_exp, int* bounds, double dx, double radius,
                    double xi, double xj, double ai, double aj,
                    int nx_per_cell, int topl, double *cache)
{
    double aij = ai + aj;
    double xij = (ai * xi + aj * xj) / aij;
    int x0_latt = (int) floor((xij - radius) / dx);
    int x1_latt = (int) ceil((xij + radius) / dx);
    int xij_latt = rint(xij / dx);
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
    if (ngridx >= nx_per_cell) {
        xs_all = gridx + ngridx;
    }

    double _dxdx = -aij * dx * dx;
    double _x0dx = -2 * aij * x0xij * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_x0dx + _dxdx);
    double exp_x0x0 = exp(_x0x0);

    int i;
    int istart = xij_latt - x0_latt;
    for (i = istart; i < ngridx; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp(_x0x0);
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

    // add up contributions from all images to the referece image
    if (ngridx >= nx_per_cell) {
        memset(xs_exp, 0, (topl+1)*nx_per_cell*sizeof(double));
        int ix, l, lb, ub, size_x;
        for (ix = 0; ix < ngridx; ix++) {
            lb = modulo(ix + x0_latt, nx_per_cell);
            ub = get_upper_bound(lb, nx_per_cell, ix, ngridx);
            size_x = ub - lb;
            double* __restrict ptr_xs_exp = xs_exp + lb;
            double* __restrict ptr_xs_all = xs_all + ix;
            for (l = 0; l <= topl; l++) {
                #pragma omp simd
                for (i = 0; i < size_x; i++) {
                    ptr_xs_exp[i] += ptr_xs_all[i];
                }
                ptr_xs_exp += nx_per_cell;
                ptr_xs_all += ngridx;
            }
            ix += size_x - 1;
        }

        bounds[0] = 0;
        bounds[1] = nx_per_cell;
        ngridx = nx_per_cell;
    }
    return ngridx;
}


int _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
                     double a, double b, double cutoff,
                     double xi, double xj, double ai, double aj,
                     int periodic, int nx_per_cell, int topl, double *cache)
{
    double aij = ai + aj;
    double xij = (ai * xi + aj * xj) / aij;
    double heights_inv = b;
    double xij_frac = xij * heights_inv;
    double edge0 = xij_frac - cutoff * heights_inv;
    double edge1 = xij_frac + cutoff * heights_inv;

    if (edge0 == edge1) {
        return 0;
    }

    int nimg0 = 0;
    int nimg1 = 1;
    if (periodic) {
        nimg0 = (int) floor(edge0);
        nimg1 = (int) ceil(edge1);
    }

    int nimg = nimg1 - nimg0;

    int nmx0 = nimg0 * nx_per_cell;
    int nmx1 = nimg1 * nx_per_cell;
    int nmx = nmx1 - nmx0;

    int nx0 = (int) floor(edge0 * nx_per_cell);
    int nx1 = (int) ceil(edge1 * nx_per_cell);
   
    int nx0_edge = nx0 - nmx0;
    int nx1_edge = nx1 - nmx0;

    if (periodic) {
        nx0 = nx0_edge % nx_per_cell;
        nx1 = nx1_edge % nx_per_cell;
        if (nx1 == 0) {
            nx1 = nx_per_cell;
        }
    }
    assert(nx0 == nx0_edge);

    img_slice[0] = nimg0;
    img_slice[1] = nimg1;
    grid_slice[0] = nx0;
    grid_slice[1] = nx1;

    int ngridx = _num_grids_on_x(nimg, nx0, nx1, nx_per_cell);
    if (ngridx == 0) {
        return 0;
    }

    int i, m, l;
    double *px0;

    double *gridx = cache;
    double *xs_all = cache + nmx;
    if (nimg == 1) {
        xs_all = xs_exp;
    }

    int grid_close_to_xij = rint(xij_frac * nx_per_cell) - nmx0;
    grid_close_to_xij = MIN(grid_close_to_xij, nx1_edge);
    grid_close_to_xij = MAX(grid_close_to_xij, nx0_edge);

    double img0_x = a * nimg0;
    double dx = a / nx_per_cell;
    double base_x = img0_x + dx * grid_close_to_xij;
    double x0xij = base_x - xij;
    double _x0x0 = -aij * x0xij * x0xij;
    if (_x0x0 < EXPMIN) {
        return 0;
    }

    double _dxdx = -aij * dx * dx;
    double _x0dx = -2 * aij * x0xij * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_x0dx + _dxdx);
    double exp_x0x0 = exp(_x0x0);

    for (i = grid_close_to_xij; i < nx1_edge; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp(_x0x0);
    for (i = grid_close_to_xij-1; i >= nx0_edge; i--) {
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
        xs_all[i] = exp_x0x0;
    }

    if (topl > 0) {
        double x0xi = img0_x - xi;
        for (i = nx0_edge; i < nx1_edge; i++) {
            gridx[i] = x0xi + i * dx;
        }
        for (l = 1; l <= topl; l++) {
            px0 = xs_all + (l-1) * nmx;
            for (i = nx0_edge; i < nx1_edge; i++) {
                px0[nmx+i] = px0[i] * gridx[i];
            }
        }
    }

    int idx1;
    if (nimg > 1) {
        for (l = 0; l <= topl; l++) {
            px0 = xs_all + l * nmx;
            for (i = nx0; i < nx_per_cell; i++) {
                xs_exp[l*nx_per_cell+i] = px0[i];
            }
            memset(xs_exp+l*nx_per_cell, 0, nx0*sizeof(double));
            for (m = 1; m < nimg; m++) {
                px0 = xs_all + l * nmx + m*nx_per_cell;
                idx1 = (m == nimg - 1) ? nx1 : nx_per_cell;
                for (i = 0; i < idx1; i++) {
                    xs_exp[l*nx_per_cell+i] += px0[i];
                }
            }
        }
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

    int ngridx = orth_components(*xs_exp, grid_slice, dh[0], radius,
                                 ri[0], rj[0], ai, aj, mesh[0], topl, cache);
    if (ngridx == 0) {
            return 0;
    }

    int ngridy = orth_components(*ys_exp, grid_slice+2, dh[4], radius,
                                 ri[1], rj[1], ai, aj, mesh[1], topl, cache);
    if (ngridy == 0) {
            return 0;
    }

    int ngridz = orth_components(*zs_exp, grid_slice+4, dh[8], radius,
                                 ri[2], rj[2], ai, aj, mesh[2], topl, cache);
    if (ngridz == 0) {
            return 0;
    }

    return data_size;
}


int _init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                    int *img_slice, int *grid_slice, int *mesh,
                    int topl, int dimension, double cutoff,
                    double ai, double aj, double *ri, double *rj,
                    double *a, double *b, double *cache)
{
        int l1 = topl + 1;
        *xs_exp = cache;
        *ys_exp = *xs_exp + l1 * mesh[0];
        *zs_exp = *ys_exp + l1 * mesh[1];
        int data_size = l1 * (mesh[0] + mesh[1] + mesh[2]);
        cache += data_size;

        int ngridx = _orth_components(*xs_exp, img_slice, grid_slice,
                                      a[0], b[0], cutoff, ri[0], rj[0], ai, aj,
                                      (dimension>=1), mesh[0], topl, cache);
        if (ngridx == 0) {
                return 0;
        }

        int ngridy = _orth_components(*ys_exp, img_slice+2, grid_slice+2,
                                      a[4], b[4], cutoff, ri[1], rj[1], ai, aj,
                                      (dimension>=2), mesh[1], topl, cache);
        if (ngridy == 0) {
                return 0;
        }

        int ngridz = _orth_components(*zs_exp, img_slice+4, grid_slice+4,
                                      a[8], b[8], cutoff, ri[2], rj[2], ai, aj,
                                      (dimension>=3), mesh[2], topl, cache);
        if (ngridz == 0) {
                return 0;
        }

        return data_size;
}


void _get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache)
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


void _dm_to_dm_xyz(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache)
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

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

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


void _dm_xyz_to_dm(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache)
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

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

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
