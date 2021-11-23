//#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "dft/multigrid.h"
#include "dft/grid_common.h"

#define MESH_BLK        4


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
        dgemm_(&TRANS_N, &TRANS_N, &ncol, &nao_i, &nao_j,
               &D1, jsh_contr_coeff, &ncol, dm_cart, &nao_j, &D0, buf, &ncol);
        dgemm_(&TRANS_N, &TRANS_T, &ncol, &nrow, &nao_i,
               &D1, buf, &ncol, ish_contr_coeff, &nrow, &D0, pdm, &naoj);
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


static void _orth_ints(double *out, double *weights,
                       int topl, double fac,
                       double *xs_exp, double *ys_exp, double *zs_exp,
                       int *img_slice, int *grid_slice,
                       int *mesh, double *cache)
{
        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgx = nimgx1 - nimgx0;
        int nimgy = nimgy1 - nimgy0;
        int nimgz = nimgz1 - nimgz0;
        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridx = _num_grids_on_x(nimgx, nx0, nx1, mesh[0]);
        int ngridy = _num_grids_on_x(nimgy, ny0, ny1, mesh[1]);
        int ngridz = _num_grids_on_x(nimgz, nz0, nz1, mesh[2]);
        int lx;

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        int xcols = mesh[1] * mesh[2];
        int ycols = mesh[2];
        double *weightyz = cache;
        double *weightz = weightyz + l1*xcols;
        double *weights_submesh = weightz + l1l1*ycols;

        int is_x_split = 0, is_y_split = 0, is_z_split = 0;
        int xmap[ngridx], ymap[ngridy], zmap[ngridz];
        int ix, iy, iz, nx, ny, nz;

        if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) is_x_split = 1;
        if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) is_y_split = 1;
        if (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) is_z_split = 1;

        if (ngridy * ngridz < mesh[1] * mesh[2] / MESH_BLK) {
            if (nimgx == 1) {
                for (ix = 0; ix < ngridx; ix++) {
                    xmap[ix] = ix + nx0;
                }
            } else if (is_x_split == 1) {
                for (ix = 0; ix < nx1; ix++) {
                    xmap[ix] = ix;
                }
                nx = nx0 - nx1;
                for (ix = nx1; ix < ngridx; ix++) {
                    xmap[ix] = ix + nx;
                }
            } else {
                for (ix = 0; ix < mesh[0]; ix++) {
                    xmap[ix] = ix;
                }
            }

            if (nimgy == 1) {
                for (iy = 0; iy < ngridy; iy++) {
                    ymap[iy] = iy + ny0;
                }
            } else if (is_y_split == 1) {
                for (iy = 0; iy < ny1; iy++) {
                    ymap[iy] = iy;
                }
                ny = ny0 - ny1;
                for (iy = ny1; iy < ngridy; iy++) {
                    ymap[iy] = iy + ny;
                }
            } else {
                for (iy = 0; iy < mesh[1]; iy++) {
                    ymap[iy] = iy;
                }
            }

            if (nimgz == 1) {
                for (iz = 0; iz < ngridz; iz++) {
                    zmap[iz] = iz + nz0;
                }
            } else if (is_z_split == 1) {
                for (iz = 0; iz < nz1; iz++) {
                    zmap[iz] = iz;
                }
                nz = nz0 - nz1;
                for (iz = nz1; iz < ngridz; iz++) {
                    zmap[iz] = iz + nz;
                }
            } else{
                for (iz = 0; iz < mesh[2]; iz++) {
                    zmap[iz] = iz;
                }
            }


            xcols = ngridy * ngridz;
            size_t mesh_yz = ((size_t)mesh[1]) * mesh[2];
            for (ix = 0; ix < ngridx; ix++) {
            for (iy = 0; iy < ngridy; iy++) {
            for (iz = 0; iz < ngridz; iz++) {
                weights_submesh[ix*xcols+iy*ngridz+iz] = weights[xmap[ix]*mesh_yz+ymap[iy]*mesh[2]+zmap[iz]];
            }}}

            if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights_submesh, &xcols, xs_exp+nx0, mesh,
                       &D0, weightyz, &xcols);
            } else if (is_x_split == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx1,
                       &fac, weights_submesh, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
                nx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx,
                       &fac, weights_submesh+nx1*xcols, &xcols, xs_exp+nx0, mesh,
                       &D1, weightyz, &xcols);
            } else {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
                       &fac, weights_submesh, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
            }

            if (nimgy == 1) {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1, &ngridy,
                               &D1, weightyz+lx*xcols, &ngridz, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ngridz, &ngridz);
                }
            } else if (is_y_split == 1) {
                ny = mesh[1] - ny0;
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1, &ny1,
                               &D1, weightyz+lx*xcols, &ngridz, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ngridz, &ngridz);
                        dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1, &ny,
                               &D1, weightyz+lx*xcols+ny1*ngridz, &ngridz, ys_exp+ny0, mesh+1,
                               &D1, weightz+lx*l1*ngridz, &ngridz);
                }
            } else {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1, mesh+1,
                               &D1, weightyz+lx*xcols, &ngridz, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ngridz, &ngridz);
                }
            }

            if (nimgz == 1) {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &ngridz,
                       &D1, zs_exp+nz0, mesh+2, weightz, &ngridz,
                       &D0, out, &l1);
            } else if (is_z_split == 1) {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &nz1,
                       &D1, zs_exp, mesh+2, weightz, &ngridz,
                       &D0, out, &l1);
                nz = mesh[2] - nz0;;
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &nz,
                       &D1, zs_exp+nz0, mesh+2, weightz+nz1, &ngridz,
                       &D1, out, &l1);
            } else {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, mesh+2,
                       &D1, zs_exp, mesh+2, weightz, mesh+2,
                       &D0, out, &l1);
            }
        }
        else{
            if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D0, weightyz, &xcols);
            } else if (is_x_split == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx1,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
                ngridx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D1, weightyz, &xcols);
            } else {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
            }

            if (nimgy == 1) {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
            } else if (is_y_split == 1) {
                ngridy = mesh[1] - ny0;
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ny1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D1, weightz+lx*l1*ycols, &ycols);
                }
            } else {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, mesh+1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
            }

            if (nimgz == 1) {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &ngridz,
                       &D1, zs_exp+nz0, mesh+2, weightz+nz0, mesh+2,
                       &D0, out, &l1);
            } else if (is_z_split == 1) {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &nz1,
                       &D1, zs_exp, mesh+2, weightz, mesh+2,
                       &D0, out, &l1);
                ngridz = mesh[2] - nz0;;
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &ngridz,
                       &D1, zs_exp+nz0, mesh+2, weightz+nz0, mesh+2,
                       &D1, out, &l1);
            } else {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, mesh+2,
                       &D1, zs_exp, mesh+2, weightz, mesh+2,
                       &D0, out, &l1);
            }
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

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj+1, cache);

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

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj+1, cache);

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

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

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


int eval_mat_gga_orth(double *weights, double *out, int comp,
                      int li, int lj, double ai, double aj,
                      double *ri, double *rj, double fac, double cutoff,
                      int dimension, double *a, double *b,
                      int *mesh, double *cache)
{
        int topl = li + lj + 1;
        int l1 = topl+1;
        int l1l1l1 = l1 * l1 * l1;
        double *mat_xyz = cache;
        cache += l1l1l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
                                        grid_slice, mesh,
                                        topl, dimension, cutoff,
                                        ai, aj, ri, rj, a, b, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];
        double *vx = weights + ngrids;
        double *vy = vx + ngrids;
        double *vz = vy + ngrids;

        _orth_ints(mat_xyz, weights, li+lj, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _dm_xyz_to_dm(mat_xyz, out, li, lj, ri, rj, cache);

        _orth_ints(mat_xyz, vx, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vsigma_loop_x, mat_xyz, out, li, lj, ai, aj, ri, rj, cache);

        _orth_ints(mat_xyz, vy, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vsigma_loop_y, mat_xyz, out, li, lj, ai, aj, ri, rj, cache);

        _orth_ints(mat_xyz, vz, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vsigma_loop_z, mat_xyz, out, li, lj, ai, aj, ri, rj, cache);

        return 1;
}


int eval_mat_lda_orth(double *weights, double *out, int comp,
                      int li, int lj, double ai, double aj,
                      double *ri, double *rj, double fac, double cutoff,
                      int dimension, double *a, double *b,
                      int *mesh, double *cache)
{
        int topl = li + lj;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
                                        grid_slice, mesh,
                                        topl, dimension, cutoff,
                                        ai, aj, ri, rj, a, b, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        double *dm_xyz = cache;
        cache += l1l1l1;

        _orth_ints(dm_xyz, weights, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);

        _dm_xyz_to_dm(dm_xyz, out, li, lj, ri, rj, cache);
        return 1;
}


int eval_mat_lda_orth_ip1(double *weights, double *out, int comp,
                          int li, int lj, double ai, double aj,
                          double *ri, double *rj, double fac, double cutoff,
                          int dimension, double *a, double *b,
                          int *mesh, double *cache)
{
        int dij = _LEN_CART[li] * _LEN_CART[lj];
        int topl = li + lj + 1;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
                                        grid_slice, mesh,
                                        topl, dimension, cutoff,
                                        ai, aj, ri, rj, a, b, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        double *mat_xyz = cache;
        cache += l1l1l1;
        double *pout_x = out;
        double *pout_y = pout_x + dij;
        double *pout_z = pout_y + dij;

        _orth_ints(mat_xyz, weights, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);
        return 1;
}


int eval_mat_gga_orth_ip1(double *weights, double *out, int comp,
                          int li, int lj, double ai, double aj,
                          double *ri, double *rj, double fac, double cutoff,
                          int dimension, double *a, double *b,
                          int *mesh, double *cache)
{
        int dij = _LEN_CART[li] * _LEN_CART[lj];
        int topl = li + lj + 2;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
                                        grid_slice, mesh,
                                        topl, dimension, cutoff,
                                        ai, aj, ri, rj, a, b, cache);
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
        _orth_ints(mat_xyz, weights, topl-1, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_x, mat_xyz, pout_x, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_y, mat_xyz, pout_y, li, lj, ai, aj, ri, rj, cache);
        _v1_xyz_to_v1(_vrho_loop_ip1_z, mat_xyz, pout_z, li, lj, ai, aj, ri, rj, cache);

        //vsigma part
        _orth_ints(mat_x, vx, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _orth_ints(mat_y, vy, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);
        _orth_ints(mat_z, vz, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);

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
                        PGFPair* pgfpair, int comp, double fac,
                        int dimension, double *a, double *b, int *mesh,
                        double* ish_gto_norm, double* jsh_gto_norm,
                        int *ish_atm, int *ish_bas, double *ish_env,
                        int *jsh_atm, int *jsh_bas, double *jsh_env,
                        double* Ls, double *cache)
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
                                 fac, cutoff, dimension, a, b, mesh, cache);

        double *pmat = mat + ipgf*di*naoj + jpgf*dj;
        if (value != 0) {
                int i, j, ic;
                for (ic = 0; ic < comp; ic++) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pmat[i*naoj+j] += out[i*dj+j];
                        } }
                        pmat += naoi * naoj;
                        out += di * dj;
                }
        }
}


static size_t _ints_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* a, int comp)
{
    size_t size = 0;
    size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];
    int l1 = 2 * l + 1;
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l];
    int nimgs = (int) ceil(MAX(MAX(radius/fabs(a[0]), radius/a[4]), radius/a[8])) + 1;
    int nmx = MAX(MAX(mesh[0], mesh[1]), mesh[2]) * nimgs;

    size += comp * nprim * nprim * ncart * ncart;
    size += comp * ncart * ncart;
    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1 * nmx + nmx;
    size += l1l1 * l1;
    size += 3 * (ncart + l1);
    size += l1 * mesh[1] * mesh[2];
    size += l1l1 * mesh[2];
    size += ngrids / MESH_BLK;
    size += nctr * ncart * nprim * ncart;
    size += 1000000;
    //printf("Memory allocated per thread for make_mat: %ld MB.\n", size*sizeof(double) / 1000000);
    return size;
}


void grid_integrate_drv(int (*eval_ints)(), double* mat, double* weights, TaskList** task_list,
                        int comp, int hermi, int grid_level, 
                        int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                        int dimension, double* Ls, double* a, double* b,
                        int* ish_atm, int* ish_bas, double* ish_env,
                        int* jsh_atm, int* jsh_bas, double* jsh_env, int cart)
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
                                         mesh, max_radius, a, comp);

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
            _apply_ints(eval_ints, weights, dm_cart, pgfpair, comp, 1.0, dimension, a, b, mesh,
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

    free(task_loc);
    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (hermi != 1) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }
}
