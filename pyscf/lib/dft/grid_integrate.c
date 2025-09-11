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
#include <math.h>
#include "config.h"
#include "dft/multigrid.h"
#include "dft/grid_common.h"
#include "vhf/fblas.h"


void transform_dm_inverse(double* dm_cart, double* dm, int comp,
                          double* ish_contr_coeff, double* jsh_contr_coeff,
                          int* ish_ao_loc, int* jsh_ao_loc,
                          int* ish_bas, int* jsh_bas, int ish, int jsh,
                          int ish0, int jsh0, int naoi, int naoj, double* cache)
{
    int i0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
    int i1 = ish_ao_loc[ish+1] - ish_ao_loc[ish0];
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];

    int nrow = i1 - i0;
    int ncol = j1 - j0;
    double* pdm = dm + ((size_t)naoj) * i0 + j0;

    int l_i = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int ncart_i = _LEN_CART[l_i];
    int nprim_i = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
    int nao_i = nprim_i*ncart_i;
    int l_j = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int ncart_j = _LEN_CART[l_j];
    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int nao_j = nprim_j*ncart_j;

    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double D0 = 0;
    double *buf = cache;

    int ic;
    for (ic = 0; ic < comp; ic++) {
        //einsum("pi,pq,qj->ij", coeff_i, dm_cart, coeff_j)
        dgemm_wrapper(TRANS_N, TRANS_N, ncol, nao_i, nao_j,
                      D1, jsh_contr_coeff, ncol, dm_cart, nao_j,
                      D0, buf, ncol);
        dgemm_wrapper(TRANS_N, TRANS_T, ncol, nrow, nao_i,
                      D1, buf, ncol, ish_contr_coeff, nrow,
                      D0, pdm, naoj);
        pdm += ((size_t)naoi) * naoj;
        dm_cart += nao_i * nao_j;
    }
}


static void fill_tril(double* mat, int comp, int* ish_ao_loc, int* jsh_ao_loc,
                      int ish, int jsh, int ish0, int jsh0, int naoi, int naoj)
{
    int i0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
    int i1 = ish_ao_loc[ish+1] - ish_ao_loc[ish0];
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];
    int ni = i1 - i0;
    int nj = j1 - j0;
    size_t nao2 = ((size_t)naoi) * naoj;

    double *pmat_up = mat + i0*((size_t)naoj) + j0;
    double *pmat_low = mat + j0*((size_t)naoj) + i0;
    int ic, i, j;
    for (ic = 0; ic < comp; ic++) {
        for (i = 0; i < ni; i++) {
            for (j = 0; j < nj; j++) {
                pmat_low[j*naoj+i] = pmat_up[i*naoj+j];
            }
        }
        pmat_up += nao2;
        pmat_low += nao2;
    }
}


static void integrate_submesh(double* out, double* weights,
                              double* xs_exp, double* ys_exp, double* zs_exp,
                              double fac, int topl,
                              int* mesh_lb, int* mesh_ub, int* submesh_lb,
                              const int* mesh, const int* submesh, double* cache)
{
    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    const int x0 = mesh_lb[0];
    const int y0 = mesh_lb[1];
    const int z0 = mesh_lb[2];

    const int nx = mesh_ub[0] - x0;
    const int ny = mesh_ub[1] - y0;
    const int nz = mesh_ub[2] - z0;

    const int x0_sub = submesh_lb[0];
    const int y0_sub = submesh_lb[1];
    const int z0_sub = submesh_lb[2];

    const size_t mesh_yz = ((size_t) mesh[1]) * mesh[2];

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;

    double *lzlyx = cache;
    double *zly = lzlyx + l1l1 * nx;
    double *ptr_weights = weights + x0 * mesh_yz + y0 * mesh[2] + z0;

    int ix;
    for (ix = 0; ix < nx; ix++) {
        dgemm_wrapper(TRANS_N, TRANS_N, nz, l1, ny,
                      D1, ptr_weights, mesh[2], ys_exp+y0_sub, submesh[1],
                      D0, zly, nz);
        dgemm_wrapper(TRANS_T, TRANS_N, l1, l1, nz,
                      D1, zs_exp+z0_sub, submesh[2], zly, nz,
                      D0, lzlyx+l1l1*ix, l1);
        ptr_weights += mesh_yz;
    }
    dgemm_wrapper(TRANS_N, TRANS_N, l1l1, l1, nx,
                  fac, lzlyx, l1l1, xs_exp+x0_sub, submesh[0],
                  D1, out, l1l1);
}


static void integrate_submesh_nonorth(
        double* out, double* weights,
        double* xs_exp, double* ys_exp, double* zs_exp, double* exp_corr,
        double fac, int topl, int* mesh_lb, int* mesh_ub, int* submesh_lb,
        const int* mesh, const int* submesh, double* cache)
{
    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    const int x0 = mesh_lb[0];
    const int y0 = mesh_lb[1];
    const int z0 = mesh_lb[2];

    const int nx = mesh_ub[0] - x0;
    const int ny = mesh_ub[1] - y0;
    const int nz = mesh_ub[2] - z0;
    const size_t nxy = (size_t)nx * ny;
    const size_t nyz = (size_t)ny * nz;

    const int x0_sub = submesh_lb[0];
    const int y0_sub = submesh_lb[1];
    const int z0_sub = submesh_lb[2];

    const size_t mesh_yz = (size_t)mesh[1] * mesh[2];

    const size_t submesh_xy = (size_t)submesh[0] * submesh[1];
    const size_t submesh_yz = (size_t)submesh[1] * submesh[2];

    double *exp_corr_ij = exp_corr;
    double *exp_corr_jk = exp_corr_ij + submesh_xy;
    double *exp_corr_ik = exp_corr_jk + submesh_yz;

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;

    weights += x0 * mesh_yz + y0 * mesh[2] + z0;
    exp_corr_ij += x0_sub * submesh[1] + y0_sub;
    exp_corr_jk += y0_sub * submesh[2] + z0_sub;
    exp_corr_ik += x0_sub * submesh[2] + z0_sub;

    xs_exp += x0_sub;
    ys_exp += y0_sub;
    zs_exp += z0_sub;

    int i, j, k;
    double *rho = cache;
    double *prho = rho;
    for (i = 0; i < nx; i++) {
        double *pw = weights + i * mesh_yz;
        double *ptmp = exp_corr_jk;
        for (j = 0; j < ny; j++) {
            double tmp = exp_corr_ij[j];
            for (k = 0; k < nz; k++) {
                prho[k] = pw[k] * tmp * ptmp[k] * exp_corr_ik[k];
            }
            pw += mesh[2];
            prho += nz;
            ptmp += submesh[2];
        }
        exp_corr_ij += submesh[1];
        exp_corr_ik += submesh[2];
    }

    double *xyr = rho + nx * nyz;
    dgemm_wrapper(TRANS_T, TRANS_N, l1, nxy, nz,
                  D1, zs_exp, submesh[2], rho, nz,
                  D0, xyr, l1);

    const int l1y = l1 * ny;
    double *xqr = xyr + l1 * nxy;
    for (i = 0; i < nx; i++) {
        dgemm_wrapper(TRANS_N, TRANS_N, l1, l1, ny,
                      D1, xyr + i * l1y, l1, ys_exp, submesh[1],
                      D0, xqr + i * l1l1, l1);
    }

    dgemm_wrapper(TRANS_N, TRANS_N, l1l1, l1, nx,
                  fac, xqr, l1l1, xs_exp, submesh[0],
                  D1, out, l1l1);
}


static void _orth_ints(double *out, double *weights, int topl, double fac,
                       double *xs_exp, double *ys_exp, double *zs_exp,
                       int *grid_slice, int *mesh, double *cache)
{// NOTE: out is accumulated
    const int nx0 = grid_slice[0];
    const int nx1 = grid_slice[1];
    const int ny0 = grid_slice[2];
    const int ny1 = grid_slice[3];
    const int nz0 = grid_slice[4];
    const int nz1 = grid_slice[5];
    const int ngridx = nx1 - nx0;
    const int ngridy = ny1 - ny0;
    const int ngridz = nz1 - nz0;
    if (ngridx == 0 || ngridy == 0 || ngridz == 0) {
        return;
    }

    const int submesh[3] = {ngridx, ngridy, ngridz};
    int lb[3], ub[3];
    int ix, iy, iz;
    for (ix = 0; ix < ngridx;) {
        lb[0] = modulo(ix + nx0, mesh[0]);
        ub[0] = get_upper_bound(lb[0], mesh[0], ix, ngridx);
        for (iy = 0; iy < ngridy;) {
            lb[1] = modulo(iy + ny0, mesh[1]);
            ub[1] = get_upper_bound(lb[1], mesh[1], iy, ngridy);
            for (iz = 0; iz < ngridz;) {
                lb[2] = modulo(iz + nz0, mesh[2]);
                ub[2] = get_upper_bound(lb[2], mesh[2], iz, ngridz);
                int lb_sub[3] = {ix, iy, iz};
                integrate_submesh(out, weights, xs_exp, ys_exp, zs_exp, fac, topl,
                                  lb, ub, lb_sub, mesh, submesh, cache);
                iz += ub[2] - lb[2];
            }
            iy += ub[1] - lb[1];
        }
        ix += ub[0] - lb[0];
    }
}


static void _nonorth_ints(double *out, double *weights, int topl, double fac,
                          double *xs_exp, double *ys_exp, double *zs_exp, double *exp_corr,
                          int *grid_slice, int *mesh, double *cache)
{
    const int nx = mesh[0];
    const int ny = mesh[1];
    const int nz = mesh[2];
    const int nx0 = grid_slice[0];
    const int ny0 = grid_slice[2];
    const int nz0 = grid_slice[4];
    const int ngridx = grid_slice[1] - nx0;
    const int ngridy = grid_slice[3] - ny0;
    const int ngridz = grid_slice[5] - nz0;
    if (ngridx == 0 || ngridy == 0 || ngridz == 0) {
        return;
    }

    const int submesh[3] = {ngridx, ngridy, ngridz};
    int lb[3], ub[3];
    int ix, iy, iz;
    for (ix = 0; ix < ngridx;) {
        lb[0] = modulo(ix + nx0, nx);
        ub[0] = get_upper_bound(lb[0], nx, ix, ngridx);
        for (iy = 0; iy < ngridy;) {
            lb[1] = modulo(iy + ny0, ny);
            ub[1] = get_upper_bound(lb[1], ny, iy, ngridy);
            for (iz = 0; iz < ngridz;) {
                lb[2] = modulo(iz + nz0, nz);
                ub[2] = get_upper_bound(lb[2], nz, iz, ngridz);
                int lb_sub[3] = {ix, iy, iz};
                integrate_submesh_nonorth(out, weights, xs_exp, ys_exp, zs_exp, exp_corr,
                                          fac, topl, lb, ub, lb_sub, mesh, submesh, cache);
                iz += ub[2] - lb[2];
            }
            iy += ub[1] - lb[1];
        }
        ix += ub[0] - lb[0];
    }
}


#define VRHO_LOOP_IP1(X, Y, Z) \
    int lx, ly, lz; \
    int jx, jy, jz; \
    int l##X##_i_m1 = l##X##_i - 1; \
    int l##X##_i_p1 = l##X##_i + 1; \
    double cx, cy, cz, cfac; \
    double fac_i = -2.0 * ai; \
    for (j##Y = 0; j##Y <= l##Y##_j; j##Y++) { \
        c##Y = pc##Y[j##Y+_LEN_CART0[l##Y##_j]]; \
        l##Y = l##Y##_i + j##Y; \
        for (j##Z = 0; j##Z <= l##Z##_j; j##Z++) { \
            c##Z = pc##Z[j##Z+_LEN_CART0[l##Z##_j]]; \
            l##Z = l##Z##_i + j##Z; \
            cfac = c##Y * c##Z; \
            for (j##X = 0; j##X <= l##X##_j; j##X++) { \
                if (l##X##_i > 0) { \
                    c##X = pc##X[j##X+_LEN_CART0[l##X##_j]] * l##X##_i; \
                    l##X = l##X##_i_m1 + j##X; \
                    pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
                } \
                c##X = pc##X[j##X+_LEN_CART0[l##X##_j]] * fac_i; \
                l##X = l##X##_i_p1 + j##X; \
                pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
            } \
        } \
    }


static void _vrho_loop_ip1_x(double* pv1, double* v1_xyz,
                             double* pcx, double* pcy, double* pcz,
                             double ai, double aj,
                             int lx_i, int ly_i, int lz_i,
                             int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VRHO_LOOP_IP1(x,y,z);
}


static void _vrho_loop_ip1_y(double* pv1, double* v1_xyz,
                             double* pcx, double* pcy, double* pcz,
                             double ai, double aj,
                             int lx_i, int ly_i, int lz_i,
                             int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VRHO_LOOP_IP1(y,x,z);
}


static void _vrho_loop_ip1_z(double* pv1, double* v1_xyz,
                             double* pcx, double* pcy, double* pcz,
                             double ai, double aj,
                             int lx_i, int ly_i, int lz_i,
                             int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VRHO_LOOP_IP1(z,x,y);
}


#define VSIGMA_LOOP(X, Y, Z) \
    int lx, ly, lz; \
    int jx, jy, jz; \
    int l##X##_i_m1 = l##X##_i - 1; \
    int l##X##_i_p1 = l##X##_i + 1; \
    int l##X##_j_m1 = l##X##_j - 1; \
    int l##X##_j_p1 = l##X##_j + 1; \
    double cx, cy, cz, cfac; \
    double fac_i = -2.0 * ai; \
    double fac_j = -2.0 * aj; \
    for (j##Y = 0; j##Y <= l##Y##_j; j##Y++) { \
        c##Y = pc##Y[j##Y+_LEN_CART0[l##Y##_j]]; \
        l##Y = l##Y##_i + j##Y; \
        for (j##Z = 0; j##Z <= l##Z##_j; j##Z++) { \
            c##Z = pc##Z[j##Z+_LEN_CART0[l##Z##_j]]; \
            l##Z = l##Z##_i + j##Z; \
            cfac = c##Y * c##Z; \
            for (j##X = 0; j##X <= l##X##_j_m1; j##X++) { \
                c##X = pc##X[j##X+_LEN_CART0[l##X##_j_m1]] * l##X##_j; \
                l##X = l##X##_i + j##X; \
                pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
            } \
            for (j##X = 0; j##X <= l##X##_j_p1; j##X++) { \
                c##X = pc##X[j##X+_LEN_CART0[l##X##_j_p1]] * fac_j; \
                l##X = l##X##_i + j##X; \
                pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
            } \
            for (j##X = 0; j##X <= l##X##_j; j##X++) { \
                if (l##X##_i > 0) { \
                    c##X = pc##X[j##X+_LEN_CART0[l##X##_j]] * l##X##_i; \
                    l##X = l##X##_i_m1 + j##X; \
                    pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
                } \
                c##X = pc##X[j##X+_LEN_CART0[l##X##_j]] * fac_i; \
                l##X = l##X##_i_p1 + j##X; \
                pv1[0] += c##X * cfac * v1_xyz[lx*l1l1+ly*l1+lz]; \
            } \
        } \
    }


static void _vsigma_loop_x(double* pv1, double* v1_xyz,
                           double* pcx, double* pcy, double* pcz,
                           double ai, double aj,
                           int lx_i, int ly_i, int lz_i,
                           int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VSIGMA_LOOP(x,y,z);
}


static void _vsigma_loop_y(double* pv1, double* v1_xyz,
                           double* pcx, double* pcy, double* pcz,
                           double ai, double aj,
                           int lx_i, int ly_i, int lz_i,
                           int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VSIGMA_LOOP(y,x,z);
}


static void _vsigma_loop_z(double* pv1, double* v1_xyz,
                           double* pcx, double* pcy, double* pcz,
                           double ai, double aj,
                           int lx_i, int ly_i, int lz_i,
                           int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    VSIGMA_LOOP(z,x,y);
}


static void _v1_xyz_to_v1(void (*_v1_loop)(), double* v1_xyz, double* v1,
                          int li, int lj, double ai, double aj,
                          double* ri, double* rj, double* cache)
{
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int l1 = li + lj + 2;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj+1];
    cache += 3 * dj;

    get_dm_to_dm_xyz_coeff(coeff, rij, lj+1, cache);

    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pv1 = v1;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    _v1_loop(pv1, v1_xyz, pcx, pcy, pcz, ai, aj,
                             lx_i, ly_i, lz_i, lx_j, ly_j, lz_j, l1, l1l1);
                    pv1 += 1;
                }
            }
        }
    }
}

/*
#define SUM_NABLA_I \
        if (lx_i > 0) { \
            pv1[0] += lx_i * cxyzj * v1x[(lx-1)*l1l1+ly*l1+lz]; \
        } \
        pv1[0] += fac_i * cxyzj * v1x[(lx+1)*l1l1+ly*l1+lz]; \
        if (ly_i > 0) { \
            pv1[0] += ly_i * cxyzj * v1y[lx*l1l1+(ly-1)*l1+lz]; \
        } \
        pv1[0] += fac_i * cxyzj * v1y[lx*l1l1+(ly+1)*l1+lz]; \
        if (lz_i > 0) { \
            pv1[0] += lz_i * cxyzj * v1z[lx*l1l1+ly*l1+lz-1]; \
        } \
        pv1[0] += fac_i * cxyzj * v1z[lx*l1l1+ly*l1+lz+1];
*/
/*
static void _vsigma_loop_ip1ip2_x(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    int lx, ly, lz;
    int jx, jy, jz;
    int lx_j_m1 = lx_j - 1;
    int lx_j_p1 = lx_j + 1;
    double cxj, cyj, czj, cyzj, cxyzj;
    double fac_i = -2.0 * ai;
    double fac_j = -2.0 * aj;

    for (jy = 0; jy <= ly_j; jy++) {
        cyj = pcy[jy+_LEN_CART0[ly_j]];
        ly = ly_i + jy;
        for (jz = 0; jz <= lz_j; jz++) {
            czj = pcz[jz+_LEN_CART0[lz_j]];
            lz = lz_i + jz;
            cyzj = cyj * czj;
            for (jx = 0; jx <= lx_j_m1; jx++) {
                cxj = pcx[jx+_LEN_CART0[lx_j_m1]] * lx_j;
                cxyzj = cxj * cyzj;
                lx = lx_i + jx;
                SUM_NABLA_I;
            }
            for (jx = 0; jx <= lx_j_p1; jx++) {
                cxj = pcx[jx+_LEN_CART0[lx_j_p1]] * fac_j;
                cxyzj = cxj * cyzj;
                lx = lx_i + jx;
                SUM_NABLA_I;
            }
        }
    }
}
*/

#define COMMON_INIT(x) \
    int l##x##_i; \
    int lx, ly, lz; \
    int jx, jy, jz; \
    int lx_j_m1 = lx_j - 1; \
    int lx_j_p1 = lx_j + 1; \
    int ly_j_m1 = ly_j - 1; \
    int ly_j_p1 = ly_j + 1; \
    int lz_j_m1 = lz_j - 1; \
    int lz_j_p1 = lz_j + 1; \
    double ci; \
    double cxj, cyj, czj; \
    double cyzj, cxzj, cxyj, cxyzj; \
    double fac_i = -2.0 * ai; \
    double fac_j = -2.0 * aj; \


#define SUM_NABLA_J(x, y, z) \
    for (j##y = 0; j##y <= l##y##_j; j##y++) { \
        c##y##j = pc##y[j##y+_LEN_CART0[l##y##_j]]; \
        l##y = l##y##_i + j##y; \
        for (j##z = 0; j##z <= l##z##_j; j##z++) { \
            c##z##j = pc##z[j##z+_LEN_CART0[l##z##_j]]; \
            l##z = l##z##_i + j##z; \
            c##y##z##j = c##y##j * c##z##j; \
            for (j##x = 0; j##x <= l##x##_j_m1; j##x++) { \
                c##x##j = pc##x[j##x+_LEN_CART0[l##x##_j_m1]] * l##x##_j; \
                cxyzj = c##x##j * c##y##z##j; \
                l##x = l##x##_i + j##x; \
                pv1[0] += ci * cxyzj * v1##x[lx*l1l1+ly*l1+lz]; \
            } \
            for (j##x = 0; j##x <= l##x##_j_p1; j##x++) { \
                c##x##j = pc##x[j##x+_LEN_CART0[l##x##_j_p1]] * fac_j; \
                cxyzj = c##x##j * c##y##z##j; \
                l##x = l##x##_i + j##x; \
                pv1[0] += ci * cxyzj * v1##x[lx*l1l1+ly*l1+lz]; \
            } \
        } \
    }


static void _vsigma_loop_ip1ip2_x(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i0, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    COMMON_INIT(x);

    lx_i = lx_i0 + 1;
    ci = fac_i;
    SUM_NABLA_J(x,y,z);
    SUM_NABLA_J(y,x,z);
    SUM_NABLA_J(z,x,y);

    if (lx_i0 > 0) {
        lx_i = lx_i0 - 1;
        ci = lx_i0;
        SUM_NABLA_J(x,y,z);
        SUM_NABLA_J(y,x,z);
        SUM_NABLA_J(z,x,y);
    }
}

/*
static void _vsigma_loop_ip1ip2_y(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    int lx, ly, lz;
    int jx, jy, jz;
    int ly_j_m1 = ly_j - 1;
    int ly_j_p1 = ly_j + 1;
    double cxj, cyj, czj, cxzj, cxyzj;
    double fac_i = -2.0 * ai;
    double fac_j = -2.0 * aj;

    for (jx = 0; jx <= lx_j; jx++) {
        cxj = pcx[jx+_LEN_CART0[lx_j]];
        lx = lx_i + jx;
        for (jz = 0; jz <= lz_j; jz++) {
            czj = pcz[jz+_LEN_CART0[lz_j]];
            lz = lz_i + jz;
            cxzj = cxj * czj;
            for (jy = 0; jy <= ly_j_m1; jy++) {
                cyj = pcy[jy+_LEN_CART0[ly_j_m1]] * ly_j;
                cxyzj = cyj * cxzj;
                ly = ly_i + jy;
                SUM_NABLA_I;
            }
            for (jy = 0; jy <= ly_j_p1; jy++) {
                cyj = pcy[jy+_LEN_CART0[ly_j_p1]] * fac_j;
                cxyzj = cyj * cxzj;
                ly = ly_i + jy;
                SUM_NABLA_I;
            }
        }
    }
}
*/

static void _vsigma_loop_ip1ip2_y(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i0, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    COMMON_INIT(y);

    ly_i = ly_i0 + 1;
    ci = fac_i;
    SUM_NABLA_J(x,y,z);
    SUM_NABLA_J(y,x,z);
    SUM_NABLA_J(z,x,y);

    if (ly_i0 > 0) {
        ly_i = ly_i0 - 1;
        ci = ly_i0;
        SUM_NABLA_J(x,y,z);
        SUM_NABLA_J(y,x,z);
        SUM_NABLA_J(z,x,y);
    }
}


/*
static void _vsigma_loop_ip1ip2_z(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    int lx, ly, lz;
    int jx, jy, jz;
    int lz_j_m1 = lz_j - 1;
    int lz_j_p1 = lz_j + 1;
    double cxj, cyj, czj, cxyj, cxyzj;
    double fac_i = -2.0 * ai;
    double fac_j = -2.0 * aj;

    for (jx = 0; jx <= lx_j; jx++) {
        cxj = pcx[jx+_LEN_CART0[lx_j]];
        lx = lx_i + jx;
        for (jy = 0; jy <= ly_j; jy++) {
            cyj = pcy[jy+_LEN_CART0[ly_j]];
            ly = ly_i + jy;
            cxyj = cxj * cyj;
            for (jz = 0; jz <= lz_j_m1; jz++) {
                czj = pcz[jz+_LEN_CART0[lz_j_m1]] * lz_j;
                cxyzj = czj * cxyj;
                lz = lz_i + jz;
                SUM_NABLA_I;
            }
            for (jz = 0; jz <= lz_j_p1; jz++) {
                czj = pcz[jz+_LEN_CART0[lz_j_p1]] * fac_j;
                cxyzj = czj * cxyj;
                lz = lz_i + jz;
                SUM_NABLA_I;
            }
        }
    }
}
*/

static void _vsigma_loop_ip1ip2_z(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i0,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    COMMON_INIT(z);

    lz_i = lz_i0 + 1;
    ci = fac_i;
    SUM_NABLA_J(x,y,z);
    SUM_NABLA_J(y,x,z);
    SUM_NABLA_J(z,x,y);

    if (lz_i0 > 0) {
        lz_i = lz_i0 - 1;
        ci = lz_i0;
        SUM_NABLA_J(x,y,z);
        SUM_NABLA_J(y,x,z);
        SUM_NABLA_J(z,x,y);
    }
}


static void _vsigma_ip1ip2(void (*_v1_loop)(), double* v1x,
                           double* v1y, double* v1z, double* v1,
                           int li, int lj, double ai, double aj,
                           double* ri, double* rj, double* cache)
{
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int topl = li + lj + 2;
    int l1 = topl + 1;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj+1];
    cache += 3 * dj;

    get_dm_to_dm_xyz_coeff(coeff, rij, lj+1, cache);

    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pv1 = v1;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    _v1_loop(pv1, v1x, v1y, v1z, pcx, pcy, pcz, ai, aj,
                             lx_i, ly_i, lz_i, lx_j, ly_j, lz_j, l1, l1l1);
                    pv1 += 1;
                }
            }
        }
    }
}


static void _vsigma_loop_lap1_x(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    int lx, ly, lz;
    int jx, jy, jz;
    double cxj, cyj, czj, cxyj, cxyzj;
    double fac_x;
    double fac_i = -2.0 * ai;

    for (jx = 0; jx <= lx_j; jx++) {
        cxj = pcx[jx+_LEN_CART0[lx_j]];
        lx = lx_i + jx;
        for (jy = 0; jy <= ly_j; jy++) {
            cyj = pcy[jy+_LEN_CART0[ly_j]];
            ly = ly_i + jy;
            cxyj = cxj * cyj;
            for (jz = 0; jz <= lz_j; jz++) {
                czj = pcz[jz+_LEN_CART0[lz_j]];
                lz = lz_i + jz;
                cxyzj = cxyj * czj;

                fac_x = lx_i + 1;
                pv1[0] += fac_x * fac_i * cxyzj * v1x[lx*l1l1+ly*l1+lz];
                if (lx_i - 1 > 0) {
                    fac_x = lx_i - 1;
                    pv1[0] += fac_x * lx_i * cxyzj * v1x[(lx-2)*l1l1+ly*l1+lz];
                }

                if (lx_i > 0) {
                    fac_x = lx_i;
                    if (ly_i > 0) {
                        pv1[0] += fac_x * ly_i * cxyzj * v1y[(lx-1)*l1l1+(ly-1)*l1+lz];
                    }
                    pv1[0] += fac_x * fac_i * cxyzj * v1y[(lx-1)*l1l1+(ly+1)*l1+lz];

                    if (lz_i > 0) {
                        pv1[0] += fac_x * lz_i * cxyzj * v1z[(lx-1)*l1l1+ly*l1+lz-1];
                    }
                    pv1[0] += fac_x * fac_i * cxyzj * v1z[(lx-1)*l1l1+ly*l1+lz+1];
                }

                fac_x = fac_i;
                if (lx_i > 0) {
                    pv1[0] += fac_x * lx_i * cxyzj * v1x[lx*l1l1+ly*l1+lz];
                }
                pv1[0] += fac_x * fac_i * cxyzj * v1x[(lx+2)*l1l1+ly*l1+lz];

                if (ly_i > 0) {
                    pv1[0] += fac_x * ly_i * cxyzj * v1y[(lx+1)*l1l1+(ly-1)*l1+lz];
                }
                pv1[0] += fac_x * fac_i * cxyzj * v1y[(lx+1)*l1l1+(ly+1)*l1+lz];

                if (lz_i > 0) {
                    pv1[0] += fac_x * lz_i * cxyzj * v1z[(lx+1)*l1l1+ly*l1+lz-1];
                }
                pv1[0] += fac_x * fac_i * cxyzj * v1z[(lx+1)*l1l1+ly*l1+lz+1];
            }
        }
    }
}


static void _vsigma_loop_lap1_y(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    int lx, ly, lz;
    int jx, jy, jz;
    double cxj, cyj, czj, cxyj, cxyzj;
    double fac_y;
    double fac_i = -2.0 * ai;

    for (jx = 0; jx <= lx_j; jx++) {
        cxj = pcx[jx+_LEN_CART0[lx_j]];
        lx = lx_i + jx;
        for (jy = 0; jy <= ly_j; jy++) {
            cyj = pcy[jy+_LEN_CART0[ly_j]];
            ly = ly_i + jy;
            cxyj = cxj * cyj;
            for (jz = 0; jz <= lz_j; jz++) {
                czj = pcz[jz+_LEN_CART0[lz_j]];
                lz = lz_i + jz;
                cxyzj = cxyj * czj;

                fac_y = ly_i + 1;
                pv1[0] += fac_y * fac_i * cxyzj * v1y[lx*l1l1+ly*l1+lz];
                if (ly_i - 1 > 0) {
                    fac_y = ly_i - 1;
                    pv1[0] += fac_y * ly_i * cxyzj * v1y[lx*l1l1+(ly-2)*l1+lz];
                }

                if (ly_i > 0) {
                    fac_y = ly_i;
                    if (lx_i > 0) {
                        pv1[0] += fac_y * lx_i * cxyzj * v1x[(lx-1)*l1l1+(ly-1)*l1+lz];
                    }
                    pv1[0] += fac_y * fac_i * cxyzj * v1x[(lx+1)*l1l1+(ly-1)*l1+lz];

                    if (lz_i > 0) {
                        pv1[0] += fac_y * lz_i * cxyzj * v1z[lx*l1l1+(ly-1)*l1+lz-1];
                    }
                    pv1[0] += fac_y * fac_i * cxyzj * v1z[lx*l1l1+(ly-1)*l1+lz+1];
                }

                fac_y = fac_i;
                if (lx_i > 0) {
                    pv1[0] += fac_y * lx_i * cxyzj * v1x[(lx-1)*l1l1+(ly+1)*l1+lz];
                }
                pv1[0] += fac_y * fac_i * cxyzj * v1x[(lx+1)*l1l1+(ly+1)*l1+lz];

                if (ly_i > 0) {
                    pv1[0] += fac_y * ly_i * cxyzj * v1y[lx*l1l1+ly*l1+lz];
                }
                pv1[0] += fac_y * fac_i * cxyzj * v1y[lx*l1l1+(ly+2)*l1+lz];

                if (lz_i > 0) {
                    pv1[0] += fac_y * lz_i * cxyzj * v1z[lx*l1l1+(ly+1)*l1+lz-1];
                }
                pv1[0] += fac_y * fac_i * cxyzj * v1z[lx*l1l1+(ly+1)*l1+lz+1];
            }
        }
    }
}


static void _vsigma_loop_lap1_z(double* pv1, double* v1x, double* v1y, double* v1z,
                       double* pcx, double* pcy, double* pcz,
                       double ai, double aj,
                       int lx_i, int ly_i, int lz_i,
                       int lx_j, int ly_j, int lz_j, int l1, int l1l1)
{
    int lx, ly, lz;
    int jx, jy, jz;
    double cxj, cyj, czj, cxyj, cxyzj;
    double fac_z;
    double fac_i = -2.0 * ai;

    for (jx = 0; jx <= lx_j; jx++) {
        cxj = pcx[jx+_LEN_CART0[lx_j]];
        lx = lx_i + jx;
        for (jy = 0; jy <= ly_j; jy++) {
            cyj = pcy[jy+_LEN_CART0[ly_j]];
            ly = ly_i + jy;
            cxyj = cxj * cyj;
            for (jz = 0; jz <= lz_j; jz++) {
                czj = pcz[jz+_LEN_CART0[lz_j]];
                lz = lz_i + jz;
                cxyzj = cxyj * czj;

                fac_z = lz_i + 1;
                pv1[0] += fac_z * fac_i * cxyzj * v1z[lx*l1l1+ly*l1+lz];
                if (lz_i - 1 > 0) {
                    fac_z = lz_i - 1;
                    pv1[0] += fac_z * lz_i * cxyzj * v1z[lx*l1l1+ly*l1+lz-2];
                }

                if (lz_i > 0) {
                    fac_z = lz_i;
                    if (lx_i > 0) {
                        pv1[0] += fac_z * lx_i * cxyzj * v1x[(lx-1)*l1l1+ly*l1+lz-1];
                    }
                    pv1[0] += fac_z * fac_i * cxyzj * v1x[(lx+1)*l1l1+ly*l1+lz-1];

                    if (ly_i > 0) {
                        pv1[0] += fac_z * ly_i * cxyzj * v1y[lx*l1l1+(ly-1)*l1+lz-1];
                    }
                    pv1[0] += fac_z * fac_i * cxyzj * v1y[lx*l1l1+(ly+1)*l1+lz-1];
                }

                fac_z = fac_i;
                if (lx_i > 0) {
                    pv1[0] += fac_z * lx_i * cxyzj * v1x[(lx-1)*l1l1+ly*l1+lz+1];
                }
                pv1[0] += fac_z * fac_i * cxyzj * v1x[(lx+1)*l1l1+ly*l1+lz+1];

                if (ly_i > 0) {
                    pv1[0] += fac_z * ly_i * cxyzj * v1y[lx*l1l1+(ly-1)*l1+lz+1];
                }
                pv1[0] += fac_z * fac_i * cxyzj * v1y[lx*l1l1+(ly+1)*l1+lz+1];

                if (lz_i > 0) {
                    pv1[0] += fac_z * lz_i * cxyzj * v1z[lx*l1l1+ly*l1+lz];
                }
                pv1[0] += fac_z * fac_i * cxyzj * v1z[lx*l1l1+ly*l1+lz+2];
            }
        }
    }
}


static void _vsigma_lap1(void (*_v1_loop)(), double* v1x,
                         double* v1y, double* v1z, double* v1,
                         int li, int lj, double ai, double aj,
                         double* ri, double* rj, double* cache)
{
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int topl = li + lj + 2;
    int l1 = topl + 1;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj];
    cache += 3 * dj;

    get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pv1 = v1;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    _v1_loop(pv1, v1x, v1y, v1z, pcx, pcy, pcz, ai, aj,
                             lx_i, ly_i, lz_i, lx_j, ly_j, lz_j, l1, l1l1);
                    pv1 += 1;
                }
            }
        }
    }
}


int eval_mat_lda_orth(double *weights, double *out, int comp,
                      int li, int lj, double ai, double aj,
                      double *ri, double *rj, double fac, double cutoff,
                      int dimension, double* dh, double *dh_inv,
                      int *mesh, double *cache)
{
    int topl = li + lj;
    int l1 = topl+1;
    int l1l1l1 = l1*l1*l1;
    int grid_slice[6];
    double *xs_exp, *ys_exp, *zs_exp;
    int data_size = init_orth_data(&xs_exp, &ys_exp, &zs_exp,
                                   grid_slice, dh, mesh, topl, cutoff,
                                   ai, aj, ri, rj, cache);

    if (data_size == 0) {
        return 0;
    }
    cache += data_size;

    double *dm_xyz = cache;
    cache += l1l1l1;

    memset(dm_xyz, 0, l1l1l1*sizeof(double));
    _orth_ints(dm_xyz, weights, topl, fac, xs_exp, ys_exp, zs_exp,
               grid_slice, mesh, cache);

    dm_xyz_to_dm(dm_xyz, out, li, lj, ri, rj, cache);
    return 1;
}


int eval_mat_lda_nonorth(double *weights, double *out, int comp,
                         int li, int lj, double ai, double aj,
                         double *ri, double *rj, double fac, double cutoff,
                         int dimension, double* dh, double *dh_inv,
                         int *mesh, double *cache)
{
    int topl = li + lj;
    int l1 = topl + 1;
    int l1l1l1 = l1 * l1 * l1;
    int grid_slice[6];
    double *xs_exp, *ys_exp, *zs_exp, *exp_corr;
    size_t data_size = init_nonorth_data(&xs_exp, &ys_exp, &zs_exp, &exp_corr,
                                         grid_slice, dh, dh_inv, mesh, topl, cutoff,
                                         ai, aj, ri, rj, cache);

    if (data_size == 0) {
        return 0;
    }
    cache += data_size;

    double *dm_xyz = cache;
    memset(dm_xyz, 0, l1l1l1 * sizeof(double));

    double *dm_ijk = dm_xyz + l1l1l1;
    memset(dm_ijk, 0, l1l1l1 * sizeof(double));

    cache = dm_ijk + l1l1l1;
    _nonorth_ints(dm_ijk, weights, topl, fac, xs_exp, ys_exp, zs_exp, exp_corr,
                  grid_slice, mesh, cache);

    dm_ijk_to_dm_xyz(dm_ijk, dm_xyz, dh, topl);

    cache = dm_xyz + l1l1l1;
    dm_xyz_to_dm(dm_xyz, out, li, lj, ri, rj, cache);
    return 1;
}


int eval_mat_lda_orth_ip1(double *weights, double *out, int comp,
                          int li, int lj, double ai, double aj,
                          double *ri, double *rj, double fac, double cutoff,
                          int dimension, double* dh, double *dh_inv,
                          int *mesh, double *cache)
{
        int dij = _LEN_CART[li] * _LEN_CART[lj];
        int topl = li + lj + 1;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = init_orth_data(&xs_exp, &ys_exp, &zs_exp,
                                       grid_slice, dh, mesh, topl, cutoff,
                                       ai, aj, ri, rj, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        double *mat_xyz = cache;
        cache += l1l1l1;
        double *pout_x = out;
        double *pout_y = pout_x + dij;
        double *pout_z = pout_y + dij;

        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, weights, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);
        return 1;
}


int eval_mat_lda_nonorth_ip1(double *weights, double *out, int comp,
                             int li, int lj, double ai, double aj,
                             double *ri, double *rj, double fac, double cutoff,
                             int dimension, double* dh, double *dh_inv,
                             int *mesh, double *cache)
{
    int dij = _LEN_CART[li] * _LEN_CART[lj];
    int topl = li + lj + 1;
    int l1 = topl+1;
    int l1l1l1 = l1*l1*l1;
    int grid_slice[6];
    double *xs_exp, *ys_exp, *zs_exp, *exp_corr;
    size_t data_size = init_nonorth_data(&xs_exp, &ys_exp, &zs_exp, &exp_corr,
                                         grid_slice, dh, dh_inv, mesh, topl, cutoff,
                                         ai, aj, ri, rj, cache);
    if (data_size == 0) {
            return 0;
    }
    cache += data_size;

    double *mat_xyz = cache;
    double *mat_ijk = mat_xyz + l1l1l1;
    cache = mat_ijk + l1l1l1;

    memset(mat_xyz, 0, l1l1l1*sizeof(double));
    memset(mat_ijk, 0, l1l1l1*sizeof(double));

    _nonorth_ints(mat_ijk, weights, topl, fac, xs_exp, ys_exp, zs_exp, exp_corr,
                  grid_slice, mesh, cache);

    dm_ijk_to_dm_xyz(mat_ijk, mat_xyz, dh, topl);

    cache = mat_xyz + l1l1l1;
    double *pout_x = out;
    double *pout_y = pout_x + dij;
    double *pout_z = pout_y + dij;
    _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
    _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
    _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);
    return 1;
}


int eval_mat_gga_orth(double *weights, double *out, int comp,
                      int li, int lj, double ai, double aj,
                      double *ri, double *rj, double fac, double cutoff,
                      int dimension, double* dh, double *dh_inv,
                      int *mesh, double *cache)
{
        int topl = li + lj + 1;
        int l1 = topl+1;
        int l1l1l1 = l1 * l1 * l1;
        double *mat_xyz = cache;
        cache += l1l1l1;
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = init_orth_data(&xs_exp, &ys_exp, &zs_exp,
                                       grid_slice, dh, mesh, topl, cutoff,
                                       ai, aj, ri, rj, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];
        double *vx = weights + ngrids;
        double *vy = vx + ngrids;
        double *vz = vy + ngrids;

        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, weights, li+lj, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        dm_xyz_to_dm(mat_xyz, out, li, lj, ri, rj, cache);

        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, vx, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vsigma_loop_x, mat_xyz, out, li, lj, ai, aj, ri, rj, cache);

        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, vy, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vsigma_loop_y, mat_xyz, out, li, lj, ai, aj, ri, rj, cache);

        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, vz, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vsigma_loop_z, mat_xyz, out, li, lj, ai, aj, ri, rj, cache);

        return 1;
}


int eval_mat_gga_orth_ip1(double *weights, double *out, int comp,
                          int li, int lj, double ai, double aj,
                          double *ri, double *rj, double fac, double cutoff,
                          int dimension, double* dh, double *dh_inv,
                          int *mesh, double *cache)
{
        int dij = _LEN_CART[li] * _LEN_CART[lj];
        int topl = li + lj + 2;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = init_orth_data(&xs_exp, &ys_exp, &zs_exp,
                                       grid_slice, dh, mesh, topl, cutoff,
                                       ai, aj, ri, rj, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        double *mat_xyz = cache;
        double *mat_x = mat_xyz;
        double *mat_y = mat_x + l1l1l1;
        double *mat_z = mat_y + l1l1l1;
        cache += l1l1l1*3;
        double *pout_x = out;
        double *pout_y = pout_x + dij;
        double *pout_z = pout_y + dij;

        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];
        double *vx = weights + ngrids;
        double *vy = vx + ngrids;
        double *vz = vy + ngrids;

        //vrho part
        memset(mat_xyz, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_xyz, weights, topl-1, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);

        //vsigma part
        memset(mat_x, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_x, vx, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);

        memset(mat_y, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_y, vy, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);

        memset(mat_z, 0, l1l1l1*sizeof(double));
        _orth_ints(mat_z, vz, topl, fac, xs_exp, ys_exp, zs_exp,
                   grid_slice, mesh, cache);

        _vsigma_ip1ip2(_vsigma_loop_ip1ip2_x, mat_x, mat_y, mat_z,
                       pout_x, li, lj, ai, aj, ri, rj, cache);
        _vsigma_ip1ip2(_vsigma_loop_ip1ip2_y, mat_x, mat_y, mat_z,
                       pout_y, li, lj, ai, aj, ri, rj, cache);
        _vsigma_ip1ip2(_vsigma_loop_ip1ip2_z, mat_x, mat_y, mat_z,
                       pout_z, li, lj, ai, aj, ri, rj, cache);

        _vsigma_lap1(_vsigma_loop_lap1_x, mat_x, mat_y, mat_z,
                     pout_x, li, lj, ai, aj, ri, rj, cache);
        _vsigma_lap1(_vsigma_loop_lap1_y, mat_x, mat_y, mat_z,
                     pout_y, li, lj, ai, aj, ri, rj, cache);
        _vsigma_lap1(_vsigma_loop_lap1_z, mat_x, mat_y, mat_z,
                     pout_z, li, lj, ai, aj, ri, rj, cache);
        return 1;
}


static void _apply_ints(int (*eval_ints)(), double *weights, double *mat,
                        PGFPair *pgfpair, int comp, double fac, int dimension,
                        double *dh, double *dh_inv, int *mesh,
                        double *ish_gto_norm, double *jsh_gto_norm,
                        int *ish_atm, int *ish_bas, double *ish_env,
                        int *jsh_atm, int *jsh_bas, double *jsh_env,
                        double *Ls, double *cache)
{
    int i_sh = pgfpair->ish;
    int j_sh = pgfpair->jsh;
    int ipgf = pgfpair->ipgf;
    int jpgf = pgfpair->jpgf;
    int iL = pgfpair->iL;
    double cutoff = pgfpair->radius;

    int li = ish_bas[ANG_OF+i_sh*BAS_SLOTS];
    int lj = jsh_bas[ANG_OF+j_sh*BAS_SLOTS];
    int di = _LEN_CART[li];
    int dj = _LEN_CART[lj];

    int ish_nprim = ish_bas[NPRIM_OF+i_sh*BAS_SLOTS];
    int jsh_nprim = jsh_bas[NPRIM_OF+j_sh*BAS_SLOTS];
    int naoi = ish_nprim * di;
    int naoj = jsh_nprim * dj;

    double *ri = ish_env + ish_atm[PTR_COORD+ish_bas[ATOM_OF+i_sh*BAS_SLOTS]*ATM_SLOTS];
    double *rj = jsh_env + jsh_atm[PTR_COORD+jsh_bas[ATOM_OF+j_sh*BAS_SLOTS]*ATM_SLOTS];
    double *rL = Ls + iL*3;
    double rjL[3];
    rjL[0] = rj[0] + rL[0];
    rjL[1] = rj[1] + rL[1];
    rjL[2] = rj[2] + rL[2];

    double ai = ish_env[ish_bas[PTR_EXP+i_sh*BAS_SLOTS]+ipgf];
    double aj = jsh_env[jsh_bas[PTR_EXP+j_sh*BAS_SLOTS]+jpgf];
    double ci = ish_gto_norm[ipgf];
    double cj = jsh_gto_norm[jpgf];
    double aij = ai + aj;
    double rrij = CINTsquare_dist(ri, rjL);
    double eij = (ai * aj / aij) * rrij;
    if (eij > EIJCUTOFF) {
        return;
    }
    fac *= exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
    if (fac < ish_env[PTR_EXPDROP] && fac < jsh_env[PTR_EXPDROP]) {
        return;
    }

    double *out = cache;
    memset(out, 0, comp*di*dj*sizeof(double));
    cache += comp * di * dj;

    int value = (*eval_ints)(weights, out, comp, li, lj, ai, aj, ri, rjL,
                             fac, cutoff, dimension, dh, dh_inv, mesh, cache);

    double *pmat = mat + ipgf*di*naoj + jpgf*dj;
    if (value != 0) {
        int i, j, ic;
        for (ic = 0; ic < comp; ic++) {
            for (i = 0; i < di; i++) {
                //#pragma omp simd
                PRAGMA_OMP_SIMD
                for (j = 0; j < dj; j++) {
                    pmat[i*naoj+j] += out[i*dj+j];
                } 
            }
            pmat += naoi * naoj;
            out += di * dj;
        }
    }
}


static size_t _orth_ints_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* dh, int comp)
{
    size_t size = 0;
    size_t nmx = get_max_num_grid_orth(dh, radius);
    int max_mesh = MAX(MAX(mesh[0], mesh[1]), mesh[2]);
    int l1 = 2 * l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1]; // use l1 to be safe

    size += comp * nprim * nprim * ncart * ncart; // dm_cart
    size += comp * ncart * ncart; // out
    size += l1 * (mesh[0] + mesh[1] + mesh[2]); // xs_exp, ys_exp, zs_exp

    size_t size_orth_components = l1 * nmx + nmx; // orth_components
    size += l1l1 * l1; // dm_xyz
    size += 3 * (ncart + l1); // dm_xyz_to_dm

    size_t size_orth_ints = 0;
    if (nmx < max_mesh) {
        size_orth_ints = (l1 + l1l1) * nmx;
    } else {
        size_orth_ints = l1*mesh[2] + l1l1*mesh[0];
    }
    size += MAX(size_orth_components, size_orth_ints);
    size += nctr * ncart * nprim * ncart;
    //size += 1000000;
    //printf("Memory allocated per thread for make_mat: %ld MB.\n", size*sizeof(double) / 1000000);
    return size;
}


static size_t _nonorth_ints_cache_size(int l, int nprim, int nctr, int* mesh, double radius,
                                       double* dh, double* dh_inv, int comp)
{
    size_t size = 0;

    //size_t nmx = get_max_num_grid_nonorth(dh_inv, radius);
    size_t nmx = get_max_num_grid_nonorth_tight(dh, dh_inv, radius);
    size_t nmx2 = nmx * nmx;
    int l1 = 2 * l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1]; // use l1 to be safe

    size += comp * nprim * nprim * ncart * ncart; // dm_cart
    size += comp * ncart * ncart; // out
    size += l1 * nmx * 3; // xs_exp, ys_exp, zs_exp
    size += nmx2 * 3; //exp_corr

    size_t tmp = nmx * 2;
    if (l > 0) {
        tmp += nmx;
    }
    size_t tmp1 = l1l1 * l1 * 2; // dm_xyz, dm_ijk
    tmp1 += nmx2 * nmx + l1 * nmx2 + l1l1 * nmx; // _nonorth_ints
    tmp1 = MAX(tmp1, 3 * (ncart + l1)); // dm_xyz_to_dm
    size += MAX(tmp, tmp1);

    size += nctr * ncart * nprim * ncart;
    return size;
}


static size_t _ints_cache_size(int l, int nprim, int nctr, int* mesh, double radius,
                               double* dh, double* dh_inv, int comp, bool orth)
{
    if (orth) {
        return _orth_ints_cache_size(l, nprim, nctr, mesh, radius, dh, comp);
    } else {
        return _nonorth_ints_cache_size(l, nprim, nctr, mesh, radius, dh, dh_inv, comp);
    }
}


static size_t _orth_ints_core_cache_size(int* mesh, double radius, double* dh, int comp)
{
    size_t size = 0;
    size_t nmx = get_max_num_grid_orth(dh, radius);
    int max_mesh = MAX(MAX(mesh[0], mesh[1]), mesh[2]);
    const int l = 0;
    int l1 = l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1];

    size_t size_orth_components = l1 * nmx + nmx;
    size_t size_orth_ints = 0;
    if (nmx < max_mesh) {
        size_orth_ints = (l1 + l1l1) * nmx;
    } else {
        size_orth_ints = l1*mesh[2] + l1l1*mesh[0];
    }
    size += MAX(size_orth_components, size_orth_ints);
    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1l1 * l1;
    size += 3 * (ncart + l1);
    return size;
}


static size_t _nonorth_ints_core_cache_size(int* mesh, double radius, double* dh, double* dh_inv, int comp)
{
    size_t size = 0;
    //size_t nmx = get_max_num_grid_nonorth(dh_inv, radius);
    size_t nmx = get_max_num_grid_nonorth_tight(dh, dh_inv, radius);
    size_t nmx2 = nmx * nmx;
    const int l = 0;
    int l1 = l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1];

    size += l1 * nmx * 3; // xs_exp, ys_exp, zs_exp
    size += nmx2 * 3; //exp_corr

    size_t tmp = nmx * 2;
    if (l > 0) {
        tmp += nmx;
    }
    size_t tmp1 = l1l1 * l1 * 2; // dm_xyz, dm_ijk
    tmp1 += nmx2 * nmx + l1 * nmx2 + l1l1 * nmx; // _nonorth_ints
    tmp1 = MAX(tmp1, 3 * (ncart + l1)); // dm_xyz_to_dm
    size += MAX(tmp, tmp1);
    return size;
}


static size_t _ints_core_cache_size(int* mesh, double radius, double* dh, double *dh_inv, int comp, bool orth)
{
    if (orth) {
        return _orth_ints_core_cache_size(mesh, radius, dh, comp);
    } else {
        return _nonorth_ints_core_cache_size(mesh, radius, dh, dh_inv, comp);
    }
}


void grid_integrate_drv(int (*eval_ints)(), double* mat, double* weights, TaskList** task_list,
                        int comp, int hermi, int grid_level, 
                        int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                        int dimension, double* Ls, double* a, double* b,
                        int* ish_atm, int* ish_bas, double* ish_env,
                        int* jsh_atm, int* jsh_bas, double* jsh_env,
                        int cart, bool orth)
{
    TaskList* tl = *task_list;
    GridLevel_Info* gridlevel_info = tl->gridlevel_info;
    Task *task = (tl->tasks)[grid_level];
    int ntasks = task->ntasks;
    if (ntasks <= 0) {
        return;
    }
    double max_radius = task->radius;
    PGFPair **pgfpairs = task->pgfpairs;
    int* mesh = gridlevel_info->mesh + grid_level*3;

    double dh[9], dh_inv[9];
    get_grid_spacing(dh, dh_inv, a, b, mesh);

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    //const int nijsh = nish * njsh;
    const int naoi = ish_ao_loc[ish1] - ish_ao_loc[ish0];
    const int naoj = jsh_ao_loc[jsh1] - jsh_ao_loc[jsh0];

    int ish_lmax = get_lmax(ish0, ish1, ish_bas);
    int jsh_lmax = ish_lmax;
    if (hermi != 1) {
        jsh_lmax = get_lmax(jsh0, jsh1, jsh_bas);
    }

    int ish_nprim_max = get_nprim_max(ish0, ish1, ish_bas);
    int jsh_nprim_max = ish_nprim_max;
    if (hermi != 1) {
        jsh_nprim_max = get_nprim_max(jsh0, jsh1, jsh_bas);
    }

    int ish_nctr_max = get_nctr_max(ish0, ish1, ish_bas);
    int jsh_nctr_max = ish_nctr_max;
    if (hermi != 1) {
        jsh_nctr_max = get_nctr_max(jsh0, jsh1, jsh_bas);
    }

    double **gto_norm_i = (double**) malloc(sizeof(double*) * nish);
    double **cart2sph_coeff_i = (double**) malloc(sizeof(double*) * nish);
    get_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1, ish_bas, ish_env, cart);
    double **gto_norm_j = gto_norm_i;
    double **cart2sph_coeff_j = cart2sph_coeff_i;
    if (hermi != 1) {
        gto_norm_j = (double**) malloc(sizeof(double*) * njsh);
        cart2sph_coeff_j = (double**) malloc(sizeof(double*) * njsh);
        get_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1, jsh_bas, jsh_env, cart);
    }

    int *task_loc;
    int nblock = get_task_loc(&task_loc, pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);

    size_t cache_size = _ints_cache_size(MAX(ish_lmax,jsh_lmax),
                                         MAX(ish_nprim_max, jsh_nprim_max),
                                         MAX(ish_nctr_max, jsh_nctr_max), 
                                         mesh, max_radius, dh, dh_inv, comp, orth);

#pragma omp parallel
{
    int ish, jsh, itask, iblock;
    int li, lj, ish_nprim, jsh_nprim;
    PGFPair *pgfpair = NULL;
    double *ptr_gto_norm_i, *ptr_gto_norm_j;
    double *cache0 = malloc(sizeof(double) * cache_size);
    double *dm_cart = cache0;
    int len_dm_cart = comp*ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
    double *cache = dm_cart + len_dm_cart;

    #pragma omp for schedule(dynamic)
    for (iblock = 0; iblock < nblock; iblock+=2) {
        itask = task_loc[iblock];
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        jsh = pgfpair->jsh;
        ptr_gto_norm_i = gto_norm_i[ish];
        ptr_gto_norm_j = gto_norm_j[jsh];
        li = ish_bas[ANG_OF+ish*BAS_SLOTS];
        lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
        ish_nprim = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
        jsh_nprim = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
        len_dm_cart = comp*ish_nprim*_LEN_CART[li]*jsh_nprim*_LEN_CART[lj];
        memset(dm_cart, 0, len_dm_cart * sizeof(double));
        for (; itask < task_loc[iblock+1]; itask++) {
            pgfpair = pgfpairs[itask];
            _apply_ints(eval_ints, weights, dm_cart, pgfpair, comp, 1.0, dimension, dh, dh_inv, mesh,
                        ptr_gto_norm_i, ptr_gto_norm_j, ish_atm, ish_bas, ish_env,
                        jsh_atm, jsh_bas, jsh_env, Ls, cache);
        }
        transform_dm_inverse(dm_cart, mat, comp,
                             cart2sph_coeff_i[ish], cart2sph_coeff_j[jsh],
                             ish_ao_loc, jsh_ao_loc, ish_bas, jsh_bas,
                             ish, jsh, ish0, jsh0, naoi, naoj, cache);
        if (hermi == 1 && ish != jsh) {
            fill_tril(mat, comp, ish_ao_loc, jsh_ao_loc,
                      ish, jsh, ish0, jsh0, naoi, naoj);
        }
    }
    free(cache0);
}

    if (task_loc) {
        free(task_loc);
    }
    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (hermi != 1) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }
}


void int_gauss_charge_v_rs(int (*eval_ints)(), double* out, double* v_rs, int comp,
                           int* atm, int* bas, int nbas, double* env,
                           int* mesh, int dimension, double* a, double* b, double max_radius, bool orth)
{
    double dh[9], dh_inv[9];
    get_grid_spacing(dh, dh_inv, a, b, mesh);

    size_t cache_size = _ints_core_cache_size(mesh, max_radius, dh, dh_inv, comp, orth);

#pragma omp parallel
{
    int ia, ib;
    double alpha, coeff, charge, rad, fac;
    double *r0;
    double *cache = (double*) malloc(sizeof(double) * cache_size);
    #pragma omp for schedule(static)
    for (ib = 0; ib < nbas; ib++) {
        ia = bas[ib*BAS_SLOTS+ATOM_OF];
        alpha = env[bas[ib*BAS_SLOTS+PTR_EXP]];
        coeff = env[bas[ib*BAS_SLOTS+PTR_COEFF]];
        charge = (double)atm[ia*ATM_SLOTS+CHARGE_OF];
        r0 = env + atm[ia*ATM_SLOTS+PTR_COORD];
        fac = -charge * coeff;
        rad = env[atm[ia*ATM_SLOTS+PTR_RADIUS]];
        if (rad > 1e-15) {
            (*eval_ints)(v_rs, out+ia*comp, comp, 0, 0, alpha, 0.0, r0, r0,
                         fac, rad, dimension, dh, dh_inv, mesh, cache);
        }
    }
    free(cache);
}
}
