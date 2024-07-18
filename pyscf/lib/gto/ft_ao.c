/* Copyright 2021 The PySCF Developers. All Rights Reserved.

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
 * Fourier transformed AO pair
 * \int e^{-i Gv \cdot r}  i(r) * j(r) dr^3
 *
 * eval_gz, b, gxyz, gs:
 * - when eval_gz is    GTO_Gv_uniform_orth
 *   > b (reciprocal vectors) is diagonal 3x3 matrix
 *   > Gv k-space grids = dot(b.T,gxyz)
 *   > gxyz[3,nGv] = (kx[:nGv], ky[:nGv], kz[:nGv])
 *   > gs[3]: The number of G-vectors along each direction (nGv=gs[0]*gs[1]*gs[2]).
 * - when eval_gz is    GTO_Gv_uniform_nonorth
 *   > b is 3x3 matrix = 2\pi * scipy.linalg.inv(cell.lattice_vectors).T
 *   > Gv k-space grids = dot(b.T,gxyz)
 *   > gxyz[3,nGv] = (kx[:nGv], ky[:nGv], kz[:nGv])
 *   > gs[3]: The number of *positive* G-vectors along each direction.
 * - when eval_gz is    GTO_Gv_general
 *   only Gv is needed
 * - when eval_gz is    GTO_Gv_nonuniform_orth
 *   > b is the basic G value for each cartesian component
 *     Gx = b[:gs[0]]
 *     Gy = b[gs[0]:gs[0]+gs[1]]
 *     Gz = b[gs[0]+gs[1]:]
 *   > gs[3]: Number of basic G values along each direction.
 *   > gxyz[3,nGv] are used to index the basic G value
 *   > Gv is not used
 *
 * Note this implementation is an inplace version. The output tensor needs to be
 * zeroed in caller.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "gto/ft_ao.h"
#include "np_helper/np_helper.h"

#define SQRTPI          1.7724538509055160272981674833411451
#define EXPCUTOFF       60
#define MIN_EXPCUTOFF   40
#define OF_CMPLX        2
#define CART_MAX        136  // >= (ANG_MAX*(ANG_MAX+1)/2)
#define BLKSIZE         312

// functions implemented in libcint
double CINTsquare_dist(double *r1, double *r2);
double CINTcommon_fac_sp(int l);
void CINTcart_comp(int *nx, int *ny, int *nz, int lmax);

#define IINC            0
#define JINC            1
#define GSHIFT          4
#define POS_E1          5
#define RYS_ROOTS       6
#define TENSOR          7
void GTO_ft_init1e_envs(FTEnvVars *envs, int *ng, int *shls, double complex fac,
                        double *Gv, double *b, int *gxyz, int *gs,
                        int nGv, int block_size,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        envs->natm = natm;
        envs->nbas = nbas;
        envs->atm = atm;
        envs->bas = bas;
        envs->env = env;
        envs->shls = shls;

        int i_sh = shls[0];
        int j_sh = shls[1];
        envs->i_l = bas(ANG_OF, i_sh);
        envs->j_l = bas(ANG_OF, j_sh);
        envs->x_ctr[0] = bas(NCTR_OF, i_sh);
        envs->x_ctr[1] = bas(NCTR_OF, j_sh);
        envs->nfi = (envs->i_l+1)*(envs->i_l+2)/2;
        envs->nfj = (envs->j_l+1)*(envs->j_l+2)/2;
        envs->nf = envs->nfi * envs->nfj;
        if (env[PTR_EXPCUTOFF] == 0) {
                envs->expcutoff = EXPCUTOFF;
        } else {
                envs->expcutoff = MAX(MIN_EXPCUTOFF, env[PTR_EXPCUTOFF]);
        }

        envs->gbits = ng[GSHIFT];
        envs->ncomp_e1 = ng[POS_E1];
        envs->ncomp_tensor = ng[TENSOR];

        envs->li_ceil = envs->i_l + ng[IINC];
        envs->lj_ceil = envs->j_l + ng[JINC];
        envs->ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        envs->rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));

        int dli, dlj;
        if (envs->li_ceil >= envs->lj_ceil) {
                dli = envs->li_ceil + envs->lj_ceil + 1;
                dlj = envs->lj_ceil + 1;
                envs->rirj[0] = envs->ri[0] - envs->rj[0];
                envs->rirj[1] = envs->ri[1] - envs->rj[1];
                envs->rirj[2] = envs->ri[2] - envs->rj[2];
        } else {
                dli = envs->li_ceil + 1;
                dlj = envs->li_ceil + envs->lj_ceil + 1;
                envs->rirj[0] = envs->rj[0] - envs->ri[0];
                envs->rirj[1] = envs->rj[1] - envs->ri[1];
                envs->rirj[2] = envs->rj[2] - envs->ri[2];
        }
        envs->g_stride_i = 1;
        envs->g_stride_j = dli;
        envs->g_size     = dli * dlj;

        envs->common_factor = fac;
        envs->Gv = Gv;
        envs->b = b;
        envs->gxyz = gxyz;
        envs->gs = gs;
        envs->ngrids = nGv;
        envs->block_size = block_size;
}

static void _g2c_index_xyz(int *idx, FTEnvVars *envs)
{
        int i_l = envs->i_l;
        int j_l = envs->j_l;
        int nfi = envs->nfi;
        int nfj = envs->nfj;
        int di = envs->g_stride_i;
        int dj = envs->g_stride_j;
        int i, j, n;
        int ofx, ofjx;
        int ofy, ofjy;
        int ofz, ofjz;
        int i_nx[CART_MAX], i_ny[CART_MAX], i_nz[CART_MAX];
        int j_nx[CART_MAX], j_ny[CART_MAX], j_nz[CART_MAX];

        CINTcart_comp(i_nx, i_ny, i_nz, i_l);
        CINTcart_comp(j_nx, j_ny, j_nz, j_l);

        ofx = 0;
        ofy = envs->g_size;
        ofz = envs->g_size * 2;
        n = 0;
        for (j = 0; j < nfj; j++) {
                ofjx = ofx + dj * j_nx[j];
                ofjy = ofy + dj * j_ny[j];
                ofjz = ofz + dj * j_nz[j];
                for (i = 0; i < nfi; i++) {
                        idx[n+0] = ofjx + di * i_nx[i];
                        idx[n+1] = ofjy + di * i_ny[i];
                        idx[n+2] = ofjz + di * i_nz[i];
                        n += 3;
                }
        }
}

static void make_g1e_2d(double *g, double fac, double ai, double aj,
                        FTEnvVars *envs, FPtr_eval_gz eval_gz, double *cache)
{
        int nGv = envs->ngrids;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double aij = ai + aj;
        double *ri = envs->ri;
        double *rj = envs->rj;
        double rij[3];
        double *gxR = g;
        double *gyR = gxR + g_size;
        double *gzR = gyR + g_size;
        double *gxI = gzR + g_size;
        double *gyI = gxI + g_size;
        double *gzI = gyI + g_size;
        int i, j, n;

        rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
        rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
        rij[2] = (ai * ri[2] + aj * rj[2]) / aij;
        for (n = 0; n < bs; n++) {
                gxR[n] = 1;
                gyR[n] = 1;
                gxI[n] = 0;
                gyI[n] = 0;
        }
        (*eval_gz)(gzR, gzI, fac, aij, rij, envs, cache);

        double ia2;
        double a2 = .5 / aij;
        double *kx = envs->Gv;
        double *ky = kx + nGv;
        double *kz = ky + nGv;
        double *rirj = envs->rirj;
        double *rx;
        int nmax = envs->li_ceil + envs->lj_ceil;
        int lj, di, dj;
        size_t off0, off1, off2;
        if (envs->li_ceil >= envs->lj_ceil) {
                lj = envs->lj_ceil;
                di = envs->g_stride_i;
                dj = envs->g_stride_j;
                rx = envs->ri;
        } else {
                lj = envs->li_ceil;
                di = envs->g_stride_j;
                dj = envs->g_stride_i;
                rx = envs->rj;
        }
        double rijrx[3];
        rijrx[0] = rij[0] - rx[0];
        rijrx[1] = rij[1] - rx[1];
        rijrx[2] = rij[2] - rx[2];

        if (nmax > 0) {
#pragma GCC ivdep
                for (n = 0; n < bs; n++) {
                        // gx[di*bs+n] = (rijrx[0] - kx[n]*a2*_Complex_I) * gx[n];
                        gxR[di*bs+n] = rijrx[0] * gxR[n] + kx[n] * a2 * gxI[n];
                        gxI[di*bs+n] = rijrx[0] * gxI[n] - kx[n] * a2 * gxR[n];
                        gyR[di*bs+n] = rijrx[1] * gyR[n] + ky[n] * a2 * gyI[n];
                        gyI[di*bs+n] = rijrx[1] * gyI[n] - ky[n] * a2 * gyR[n];
                        gzR[di*bs+n] = rijrx[2] * gzR[n] + kz[n] * a2 * gzI[n];
                        gzI[di*bs+n] = rijrx[2] * gzI[n] - kz[n] * a2 * gzR[n];
                }
        }

        for (i = 1; i < nmax; i++) {
                off0 = (i-1) * di * bs;
                off1 =  i    * di * bs;
                off2 = (i+1) * di * bs;
                ia2 = i * a2;
#pragma GCC ivdep
                for (n = 0; n < bs; n++) {
                        // gx[off2+n] = ia2 * gx[off0+n] + (rijrx[0] - kx[n]*a2*_Complex_I) * gx[off1+n];
                        gxR[off2+n] = ia2 * gxR[off0+n] + rijrx[0] * gxR[off1+n] + kx[n] * a2 * gxI[off1+n];
                        gxI[off2+n] = ia2 * gxI[off0+n] + rijrx[0] * gxI[off1+n] - kx[n] * a2 * gxR[off1+n];
                        gyR[off2+n] = ia2 * gyR[off0+n] + rijrx[1] * gyR[off1+n] + ky[n] * a2 * gyI[off1+n];
                        gyI[off2+n] = ia2 * gyI[off0+n] + rijrx[1] * gyI[off1+n] - ky[n] * a2 * gyR[off1+n];
                        gzR[off2+n] = ia2 * gzR[off0+n] + rijrx[2] * gzR[off1+n] + kz[n] * a2 * gzI[off1+n];
                        gzI[off2+n] = ia2 * gzI[off0+n] + rijrx[2] * gzI[off1+n] - kz[n] * a2 * gzR[off1+n];
                }
        }

        for (j = 1; j <= lj; j++) {
        for (i = 0; i <= nmax - j; i++) {
                off0 = (i    * di + (j-1) * dj) * bs;
                off1 =((i+1) * di + (j-1) * dj) * bs;
                off2 = (i    * di +  j    * dj) * bs;
#pragma GCC ivdep
                for (n = 0; n < bs; n++) {
                        gxR[off2+n] = gxR[off1+n] + rirj[0] * gxR[off0+n];
                        gxI[off2+n] = gxI[off1+n] + rirj[0] * gxI[off0+n];
                        gyR[off2+n] = gyR[off1+n] + rirj[1] * gyR[off0+n];
                        gyI[off2+n] = gyI[off1+n] + rirj[1] * gyI[off0+n];
                        gzR[off2+n] = gzR[off1+n] + rirj[2] * gzR[off0+n];
                        gzI[off2+n] = gzI[off1+n] + rirj[2] * gzI[off0+n];
                }
        } }
}

static void inner_prod(double *gout, double *g, int *idx, FTEnvVars *envs, int empty)
{
        int ix, iy, iz, n, k;
        int bs = envs->block_size;
        double *gR = g;
        double *gI = gR + envs->g_size * bs * 3;
        double *goutR = gout;
        double *goutI = gout + envs->nf * bs;
        double xyR, xyI;
        if (empty) {
                for (n = 0; n < envs->nf; n++) {
                        ix = idx[n*3+0];
                        iy = idx[n*3+1];
                        iz = idx[n*3+2];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(goutR[n*bs+k], goutI[n*bs+k], g, g, g);
                        }
                }
        } else {
                for (n = 0; n < envs->nf; n++) {
                        ix = idx[n*3+0];
                        iy = idx[n*3+1];
                        iz = idx[n*3+2];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMAD(goutR[n*bs+k], goutI[n*bs+k], g, g, g);
                        }
                }
        }
}

static void prim_to_ctr(double *gc, size_t nf, double *gp,
                        int nprim, int nctr, double *coeff, int empty)
{
        size_t n, i;
        double c;
        double *gpR = gp;
        double *gpI = gp + nf;
        double *gcR = gc;
        double *gcI = gc + nf * nctr;

        if (empty) {
                for (n = 0; n < nctr; n++) {
                        c = coeff[nprim*n];
                        for (i = 0; i < nf; i++) {
                                gcR[n*nf+i] = gpR[i] * c;
                                gcI[n*nf+i] = gpI[i] * c;
                        }
                }
        } else {
                for (n = 0; n < nctr; n++) {
                        c = coeff[nprim*n];
                        if (c != 0) {
                                for (i = 0; i < nf; i++) {
                                        gcR[n*nf+i] += gpR[i] * c;
                                        gcI[n*nf+i] += gpI[i] * c;
                                }
                        }
                }
        }
}

static void transpose(double *out, double *in, int nf, int comp, int ngrids)
{
        size_t n, k, ic;
        double *inR = in;
        double *inI = in + nf * comp * ngrids;
        double *outR = out;
        double *outI = out + nf * comp * ngrids;
        double *pinR, *pinI;

        for (ic = 0; ic < comp; ic++) {
                for (n = 0; n < nf; n++) {
                        pinR = inR + (n*comp+ic) * ngrids;
                        pinI = inI + (n*comp+ic) * ngrids;
                        for (k = 0; k < ngrids; k++) {
                                outR[n*ngrids+k] = pinR[k];
                                outI[n*ngrids+k] = pinI[k];
                        }
                }
                outR += nf * ngrids;
                outI += nf * ngrids;
        }
}

int GTO_ft_aopair_loop(double *gctr, FTEnvVars *envs, FPtr_eval_gz eval_gz,
                       double *cache)
{
        int *shls  = envs->shls;
        int *bas = envs->bas;
        double *env = envs->env;
        int i_sh = shls[0];
        int j_sh = shls[1];
        int i_l = envs->i_l;
        int j_l = envs->j_l;
        int i_ctr = envs->x_ctr[0];
        int j_ctr = envs->x_ctr[1];
        int i_prim = bas(NPRIM_OF, i_sh);
        int j_prim = bas(NPRIM_OF, j_sh);
        int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        int nf = envs->nf;
        double *ri = envs->ri;
        double *rj = envs->rj;
        double *ai = env + bas(PTR_EXP, i_sh);
        double *aj = env + bas(PTR_EXP, j_sh);
        double *ci = env + bas(PTR_COEFF, i_sh);
        double *cj = env + bas(PTR_COEFF, j_sh);
        double fac1i, fac1j;
        double aij, dij, eij;
        int ip, jp;
        int empty[3] = {1, 1, 1};
        int *jempty = empty + 0;
        int *iempty = empty + 1;
        int *gempty = empty + 2;
        int block_size = envs->block_size;
        size_t leng = envs->g_size * 3 * (1<<envs->gbits) * block_size * OF_CMPLX;
        size_t len0 = nf * n_comp * block_size * OF_CMPLX;
        size_t leni = nf * i_ctr * n_comp * block_size * OF_CMPLX;
        double *g = cache;
        cache = g + leng;
        double *gout, *gctri, *gctrj;

        if (n_comp == 1) {
                gctrj = gctr;
        } else {
                gctrj = cache;
                cache += nf * i_ctr * j_ctr * n_comp * block_size * OF_CMPLX;
        }
        if (j_ctr == 1) {
                gctri = gctrj;
                iempty = jempty;
        } else {
                gctri = cache;
                cache += leni;
        }
        if (i_ctr == 1) {
                gout = gctri;
                gempty = iempty;
        } else {
                gout = cache;
                cache += len0;
        }

        int *idx = (int *)cache;
        cache += (envs->nf * 3 + 1) / 2;
        _g2c_index_xyz(idx, envs);

        double rrij = CINTsquare_dist(ri, rj);
        double fac1 = SQRTPI * M_PI * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l);

        *jempty = 1;
        for (jp = 0; jp < j_prim; jp++) {
                envs->aj[0] = aj[jp];
                if (j_ctr == 1) {
                        fac1j = fac1 * cj[jp];
                } else {
                        fac1j = fac1;
                        *iempty = 1;
                }
                for (ip = 0; ip < i_prim; ip++) {
                        envs->ai[0] = ai[ip];
                        aij = ai[ip] + aj[jp];
                        eij = (ai[ip] * aj[jp] / aij) * rrij;
                        if (eij > envs->expcutoff) {
                                continue;
                        }

                        dij = exp(-eij) / (aij * sqrt(aij));
                        if (i_ctr == 1) {
                                fac1i = fac1j * dij * ci[ip];
                        } else {
                                fac1i = fac1j * dij;
                        }
                        make_g1e_2d(g, fac1i, ai[ip], aj[jp], envs, eval_gz, cache);
                        (*envs->f_gout)(gout, g, idx, envs, *gempty);
                        if (i_ctr > 1) {
                                prim_to_ctr(gctri, nf*n_comp*block_size,
                                            gout, i_prim, i_ctr, ci+ip, *iempty);
                        }
                        *iempty = 0;
                }
                if (!*iempty) {
                        if (j_ctr > 1) {
                                prim_to_ctr(gctrj, i_ctr*nf*n_comp*block_size,
                                            gctri, j_prim, j_ctr, cj+jp, *jempty);
                        }
                        *jempty = 0;
                }
        }

        if (n_comp > 1 && !*jempty) {
                transpose(gctr, gctrj, nf*i_ctr*j_ctr, n_comp, block_size);
        }
        return !*jempty;
}

void GTO_Gv_general(double *gzR, double *gzI, double fac, double aij,
                    double *rij, FTEnvVars *envs, double *cache)
{
        int nGv = envs->ngrids;
        int bs = envs->block_size;
        double *kx = envs->Gv;
        double *ky = kx + nGv;
        double *kz = ky + nGv;
        double *kk = cache;
        double *kR = kk + bs;
        double cutoff = envs->expcutoff * aij * 4;
        double aij4 = -.25 / aij;
        double complex fac1 = fac * envs->common_factor;
        double complex exp_kr, s;
        int n;
#pragma GCC ivdep
        for (n = 0; n < bs; n++) {
                kk[n] = kx[n] * kx[n] + ky[n] * ky[n] + kz[n] * kz[n];
                kR[n] = kx[n] * rij[0] + ky[n] * rij[1] + kz[n] * rij[2];
        }
        for (n = 0; n < bs; n++) {
                if (kk[n] < cutoff) {
                        // fac * exp(-.25*kk/aij - kR * 1j);
                        exp_kr = cexp(aij4 * kk[n] - kR[n] * _Complex_I);
                        s = fac1 * exp_kr;
                        gzR[n] = creal(s);
                        gzI[n] = cimag(s);
                } else {
                        gzR[n] = 0;
                        gzI[n] = 0;
                }
        }
}

/*
 * Gv = dot(b.T,gxyz) + kpt
 * kk = dot(Gv, Gv)
 * kr = dot(rij, Gv) = dot(rij,b.T, gxyz) + dot(rij,kpt) = dot(br, gxyz) + dot(rij,kpt)
 * out = fac * exp(-.25 * kk / aij) * (cos(kr) - sin(kr) * _Complex_I);
 *
 * b: the first 9 elements are 2\pi*inv(a^T), then 3 elements for k_{ij},
 * followed by 3*NGv floats for Gbase
 */
void GTO_Gv_orth(double *gzR, double *gzI, double fac, double aij,
                 double *rij, FTEnvVars *envs, double *cache)
{
        int *gs = envs->gs;
        double *b = envs->b;
        int nx = gs[0];
        int ny = gs[1];
        int nz = gs[2];
        double br[3];  // dot(rij, b)
        br[0]  = rij[0] * b[0];
        br[1]  = rij[1] * b[4];
        br[2]  = rij[2] * b[8];
        double *kpt = b + 9;
        double kr[3];
        kr[0] = rij[0] * kpt[0];
        kr[1] = rij[1] * kpt[1];
        kr[2] = rij[2] * kpt[2];
        double *Gxbase = b + 12;
        double *Gybase = Gxbase + nx;
        double *Gzbase = Gybase + ny;

        int nGv = envs->ngrids;
        int block_size = envs->block_size;
        double *kx = envs->Gv;
        double *ky = kx + nGv;
        double *kz = ky + nGv;
        double *kkpool = cache;
        double *kkx = kkpool;
        double *kky = kkx + nx;
        double *kkz = kky + ny;
        double complex *csx = (double complex *)(kkz + nz);
        double complex *csy = csx + nx;
        double complex *csz = csy + ny;
        int *idx = envs->gxyz;
        int *idy = idx + nGv;
        int *idz = idy + nGv;

        double cutoff = EXPCUTOFF * aij * 4;
        double aij4 = .25 / aij;
        double complex fac1 = fac * envs->common_factor;
        int n, ix, iy, iz;
        double kk, Gr;
        double complex s;
        for (n = 0; n < nx+ny+nz; n++) {
                kkpool[n] = -1;
        }

        // TODO: determine ix, iy, iz range and initialize csx here.

        for (n = 0; n < block_size; n++) {
                ix = idx[n];
                iy = idy[n];
                iz = idz[n];
                if (kkx[ix] < 0) {  // < 0 if not initialized
                        Gr = Gxbase[ix] * br[0] + kr[0];
                        kk = aij4 * kx[n] * kx[n];
                        kkx[ix] = kk;
                        csx[ix] = cexp(-kk - Gr * _Complex_I);
                }
                if (kky[iy] < 0) {
                        Gr = Gybase[iy] * br[1] + kr[1];
                        kk = aij4 * ky[n] * ky[n];
                        kky[iy] = kk;
                        csy[iy] = cexp(-kk - Gr * _Complex_I);
                }
                if (kkz[iz] < 0) {
                        Gr = Gzbase[iz] * br[2] + kr[2];
                        kk = aij4 * kz[n] * kz[n];
                        kkz[iz] = kk;
                        csz[iz] = fac1 * cexp(-kk - Gr * _Complex_I);
                }
                if (kkx[ix] + kky[iy] + kkz[iz] < cutoff) {
                        s = csx[ix] * csy[iy] * csz[iz];
                        gzR[n] = creal(s);
                        gzI[n] = cimag(s);
                } else {
                        gzR[n] = 0;
                        gzI[n] = 0;
                }
        }
}

void GTO_Gv_nonorth(double *gzR, double *gzI, double fac, double aij,
                    double *rij, FTEnvVars *envs, double *cache)
{
        int *gs = envs->gs;
        double *b = envs->b;
        int nx = gs[0];
        int ny = gs[1];
        int nz = gs[2];
        double br[3];  // dot(rij, b)
        // FIXME: cache b * Gxbase + kpt
        br[0] = rij[0] * b[0] + rij[1] * b[1] + rij[2] * b[2];
        br[1] = rij[0] * b[3] + rij[1] * b[4] + rij[2] * b[5];
        br[2] = rij[0] * b[6] + rij[1] * b[7] + rij[2] * b[8];
        double *kpt = b + 9;
        double kr[3];
        kr[0] = rij[0] * kpt[0];
        kr[1] = rij[1] * kpt[1];
        kr[2] = rij[2] * kpt[2];
        double *Gxbase = b + 12;
        double *Gybase = Gxbase + nx;
        double *Gzbase = Gybase + ny;

        int nGv = envs->ngrids;
        int block_size = envs->block_size;
        double *kx = envs->Gv;
        double *ky = kx + nGv;
        double *kz = ky + nGv;
        double complex *csx = (double complex *)cache;
        double complex *csy = csx + nx;
        double complex *csz = csy + ny;
        int n;
        int8_t *empty = (int8_t *)(csz + nz);
        int8_t *xempty = empty;
        int8_t *yempty = xempty + nx;
        int8_t *zempty = yempty + ny;
        for (n = 0; n < nx+ny+nz; n++) {
                empty[n] = 1;
        }
        int *idx = envs->gxyz;
        int *idy = idx + nGv;
        int *idz = idy + nGv;

        double cutoff = EXPCUTOFF * aij * 4;
        double aij4 = -.25 / aij;
        double complex fac1 = fac * envs->common_factor;
        int ix, iy, iz;
        double Gr, kk;
        double complex s;

        // TODO: determine ix, iy, iz range and initialize csx here.

        for (n = 0; n < block_size; n++) {
                kk = kx[n] * kx[n] + ky[n] * ky[n] + kz[n] * kz[n];
                if (kk < cutoff) {
                        ix = idx[n];
                        iy = idy[n];
                        iz = idz[n];
                        if (xempty[ix]) {
                                Gr = Gxbase[ix] * br[0] + kr[0];
                                csx[ix] = cexp(-Gr*_Complex_I);
                                xempty[ix] = 0;
                        }
                        if (yempty[iy]) {
                                Gr = Gybase[iy] * br[1] + kr[1];
                                csy[iy] = cexp(-Gr*_Complex_I);
                                yempty[iy] = 0;
                        }
                        if (zempty[iz]) {
                                Gr = Gzbase[iz] * br[2] + kr[2];
                                csz[iz] = fac1 * cexp(-Gr*_Complex_I);
                                zempty[iz] = 0;
                        }
                        s = exp(aij4 * kk) * csx[ix]*csy[iy]*csz[iz];
                        gzR[n] = creal(s);
                        gzI[n] = cimag(s);
                } else {
                        gzR[n] = 0;
                        gzI[n] = 0;
                }
        }
}


static void daxpy_ij(double *out, double *gctr,
                     int bs, int mi, int mj, int ni, size_t ngrids)
{
        int i, j, k;
        for (j = 0; j < mj; j++) {
                for (i = 0; i < mi; i++) {
                        for (k = 0; k < bs; k++) {
                                out[i*ngrids+k] += gctr[i*bs+k];
                        }
                }
                out  += ni * ngrids;
                gctr += mi * bs;
        }
}

void GTO_ft_c2s_cart(double *out, double *gctr, int *dims,
                     FTEnvVars *envs, double *cache)
{
        int i_ctr = envs->x_ctr[0];
        int j_ctr = envs->x_ctr[1];
        int bs = envs->block_size;
        int nfi = envs->nfi;
        int nfj = envs->nfj;
        int mi = nfi*i_ctr;
        int mj = nfj*j_ctr;
        int ni = dims[1];
        size_t ng = dims[0];
        size_t nf = envs->nf;
        int ic, jc;
        size_t off;

        for (jc = 0; jc < mj; jc += nfj) {
        for (ic = 0; ic < mi; ic += nfi) {
                off = (ni * jc + ic) * ng;
                daxpy_ij(out+off, gctr, bs, nfi, nfj, ni, ng);
                gctr += nf * bs;
        } }
}

void GTO_ft_c2s_sph(double *out, double *gctr, int *dims,
                    FTEnvVars *envs, double *cache)
{
        int i_l = envs->i_l;
        int j_l = envs->j_l;
        int i_ctr = envs->x_ctr[0];
        int j_ctr = envs->x_ctr[1];
        int bs = envs->block_size;
        int di = i_l * 2 + 1;
        int dj = j_l * 2 + 1;
        int nfi = envs->nfi;
        int mi = di*i_ctr;
        int mj = dj*j_ctr;
        int ni = dims[1];
        size_t ng = dims[0];
        size_t nf = envs->nf;
        int ic, jc, k;
        size_t off;
        int buflen = nfi*dj;
        double *buf1 = cache;
        double *buf2 = buf1 + buflen * bs;
        double *pij, *buf;

        for (jc = 0; jc < mj; jc += dj) {
        for (ic = 0; ic < mi; ic += di) {
                buf = CINTc2s_ket_sph(buf1, nfi*bs, gctr, j_l);
                pij = CINTc2s_ket_sph(buf2, bs, buf, i_l);
                for (k = bs; k < dj*bs; k+=bs) {
                        CINTc2s_ket_sph(buf2+k*di, bs, buf+k*nfi, i_l);
                }

                off = (ni * jc + ic) * ng;
                daxpy_ij(out+off, pij, bs, di, dj, ni, ng);
                gctr += nf * bs;
        } }
}

/*************************************************
 *
 * eval_gz is one of GTO_Gv_general, GTO_Gv_nonorth, GTO_Gv_orth
 *
 *************************************************/

static int ft_aopair_cache_size(FTEnvVars *envs)
{
        int i_ctr = envs->x_ctr[0];
        int j_ctr = envs->x_ctr[1];
        int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        int block_size = envs->block_size;
        int *gs = envs->gs;
        int ngs = gs[0] + gs[1] + gs[2];
        if (ngs == 0) {
                ngs = envs->ngrids;
        }
        int leng = envs->g_size * 3 * (1<<envs->gbits) * OF_CMPLX;
        int len0 = envs->nf * n_comp * OF_CMPLX;
        int nc = envs->nf * i_ctr * j_ctr;
        int cache_size = leng+len0+nc*OF_CMPLX*n_comp*3 +
                (ngs*3 + envs->nf*3) / block_size + 3;
        return cache_size;
}

int GTO_ft_aopair_drv(double *outR, double *outI, int *dims,
                      FPtr_eval_gz eval_gz, double *cache, void (*f_c2s)(),
                      FTEnvVars *envs)
{
        if (eval_gz == NULL) {
                eval_gz = GTO_Gv_general;
        }
        if (eval_gz != GTO_Gv_general) {
                assert(envs->gxyz != NULL);
        }

        int i_ctr = envs->x_ctr[0];
        int j_ctr = envs->x_ctr[1];
        int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        int block_size = envs->block_size;
        size_t nc = envs->nf * i_ctr * j_ctr * block_size;
        if (outR == NULL) {
                return ft_aopair_cache_size(envs);
        }

        double *stack = NULL;
        if (cache == NULL) {
                size_t cache_size = ft_aopair_cache_size(envs) * (size_t)block_size;
                stack = malloc(sizeof(double)*cache_size);
                if (stack == NULL) {
                        fprintf(stderr, "gctr = malloc(%zu) failed in GTO_ft_aopair_drv\n",
                                sizeof(double) * cache_size);
                        return 0;
                }
                cache = stack;
        }
        double *gctrR = cache;
        double *gctrI = gctrR + nc * n_comp;
        cache = gctrI + nc * n_comp;

        int has_value = GTO_ft_aopair_loop(gctrR, envs, eval_gz, cache);

        int counts[3];
        if (dims == NULL) {
                if (f_c2s == &GTO_ft_c2s_sph) {
                        counts[0] = block_size;
                        counts[1] = (envs->i_l*2+1) * i_ctr;
                        counts[2] = (envs->j_l*2+1) * j_ctr;
                } else {  // f_c2s == &GTO_ft_c2s_cart
                        counts[0] = block_size;
                        counts[1] = envs->nfi * i_ctr;
                        counts[2] = envs->nfj * j_ctr;
                }
                dims = counts;
        }
        size_t nout = (size_t)dims[0] * dims[1] * dims[2];
        int n;
        if (has_value) {
                for (n = 0; n < n_comp; n++) {
                        (*f_c2s)(outR+nout*n, gctrR+nc*n, dims, envs, cache);
                        (*f_c2s)(outI+nout*n, gctrI+nc*n, dims, envs, cache);
                }
        }
        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

int GTO_ft_ovlp_cart(double *outR, double *outI, int *shls, int *dims,
                     FPtr_eval_gz eval_gz, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                     int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_cart, &envs);
}

int GTO_ft_ovlp_sph(double *outR, double *outI, int *shls, int *dims,
                    FPtr_eval_gz eval_gz, double complex fac,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                    int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_sph, &envs);
}

// TODO: put kkpool in opt??
void GTO_ft_ovlp_optimizer()
{
}


/*************************************************
 *
 *************************************************/

void GTO_ft_dfill_s1(FPtrIntor intor, FPtr_eval_gz eval_gz,
                     double *out, int comp, int ish, int jsh, double *buf,
                     int *shls_slice, int *ao_loc, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        ish += ish0;
        jsh += jsh0;
        int ioff = ao_loc[ish] - ao_loc[ish0];
        int joff = ao_loc[jsh] - ao_loc[jsh0];
        size_t ni = ao_loc[ish1] - ao_loc[ish0];
        size_t nj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nij = ni * nj;
        int shls[2] = {ish, jsh};
        int dims[3] = {nGv, ni, nj};
        int grid0, grid1, dg;
        size_t off;
        double *outR = out;
        double *outI = outR + comp * nij * nGv;

        for (grid0 = 0; grid0 < nGv; grid0 += BLKSIZE) {
                grid1 = MIN(grid0+BLKSIZE, nGv);
                dg = grid1 - grid0;
                off = (joff * ni + ioff) * nGv + grid0;
                (*intor)(outR+off, outI+off, shls, dims, eval_gz,
                         fac, Gv+grid0, b, gxyz+grid0, gs, nGv, dg,
                         atm, natm, bas, nbas, env, buf);
        }
}

void GTO_ft_dfill_s1hermi(FPtrIntor intor, FPtr_eval_gz eval_gz,
                          double *out, int comp, int ish, int jsh, double *buf,
                          int *shls_slice, int *ao_loc, double complex fac,
                          double *Gv, double *b, int *gxyz, int *gs, int nGv,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        ish += ish0;
        jsh += jsh0;
        int ioff = ao_loc[ish] - ao_loc[ish0];
        int joff = ao_loc[jsh] - ao_loc[jsh0];
        if (ioff < joff) {
                return;
        }

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        size_t ni = ao_loc[ish1] - ao_loc[ish0];
        size_t nj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nij = ni * nj;
        size_t NGv = nGv;
        int shls[2] = {ish, jsh};
        int dims[3] = {nGv, ni, nj};
        int grid0, grid1, dg;
        int i, j, n, ic;
        size_t off, ij, ji;
        double *outR = out;
        double *outI = outR + comp * nij * nGv;

        for (grid0 = 0; grid0 < nGv; grid0 += BLKSIZE) {
                grid1 = MIN(grid0+BLKSIZE, nGv);
                dg = grid1 - grid0;
                off = (joff * ni + ioff) * nGv + grid0;
                if ((*intor)(outR+off, outI+off, shls, dims, eval_gz,
                             fac, Gv+grid0, b, gxyz+grid0, gs, nGv, dg,
                             atm, natm, bas, nbas, env, buf)) {
                        if (ioff == joff) {
                                continue;
                        }
for (ic = 0; ic < comp; ic++) {
        off = nij * NGv * ic + grid0;
        for (j = 0; j < dj; j++) {
        for (i = 0; i < di; i++) {
                ij = off + ((j+joff)*nj+i+ioff) * NGv;
                ji = off + ((i+ioff)*nj+j+joff) * NGv;
#pragma GCC ivdep
                for (n = 0; n < dg; n++) {
                        outR[ji+n] = outR[ij+n];
                        outI[ji+n] = outI[ij+n];
                }
        } }
}
                }
        }
}

void GTO_ft_zfill_s1(FPtrIntor intor, FPtr_eval_gz eval_gz,
                     double *out, int comp, int ish, int jsh, double *buf,
                     int *shls_slice, int *ao_loc, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        ish += ish0;
        jsh += jsh0;
        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int ni = ao_loc[ish1] - ao_loc[ish0];
        int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int ioff = ao_loc[ish] - ao_loc[ish0];
        int joff = ao_loc[jsh] - ao_loc[jsh0];
        size_t nij = ni * nj;
        size_t off = joff * ni + ioff;
        size_t NGv = nGv;
        int shls[2] = {ish, jsh};
        double *bufR = buf;
        double *bufI = bufR + dij * BLKSIZE * comp;
        double *cache = bufI + dij * BLKSIZE * comp;
        int grid0, grid1, dg, dijg;
        int i, j, n, ic, ij;
        double *pout, *pbufR, *pbufI;

        for (grid0 = 0; grid0 < nGv; grid0 += BLKSIZE) {
                grid1 = MIN(grid0+BLKSIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg;
                NPdset0(bufR, dijg * comp);
                NPdset0(bufI, dijg * comp);
                if ((*intor)(bufR, bufI, shls, NULL, eval_gz,
                            fac, Gv+grid0, b, gxyz+grid0, gs, nGv, dg,
                            atm, natm, bas, nbas, env, cache)) {
for (ic = 0; ic < comp; ic++) {
        pout = out + ((off + ic * nij) * NGv + grid0) * OF_CMPLX;
        for (j = 0; j < dj; j++) {
        for (i = 0; i < di; i++) {
                pbufR = bufR + ic * dijg + dg * (j*di+i);
                pbufI = bufI + ic * dijg + dg * (j*di+i);
                ij = j * ni + i;
                for (n = 0; n < dg; n++) {
                        pout[(ij*NGv+n)*OF_CMPLX  ] += pbufR[n];
                        pout[(ij*NGv+n)*OF_CMPLX+1] += pbufI[n];
                }
        } }
}
                }
        }
}

void GTO_ft_zfill_s1hermi(FPtrIntor intor, FPtr_eval_gz eval_gz,
                          double *out, int comp, int ish, int jsh, double *buf,
                          int *shls_slice, int *ao_loc, double complex fac,
                          double *Gv, double *b, int *gxyz, int *gs, int nGv,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        ish += ish0;
        jsh += jsh0;
        int ioff = ao_loc[ish] - ao_loc[ish0];
        int joff = ao_loc[jsh] - ao_loc[jsh0];
        if (ioff < joff) {
                return;
        }

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int ni = ao_loc[ish1] - ao_loc[ish0];
        int nj = ao_loc[jsh1] - ao_loc[jsh0];
        size_t nij = ni * nj;
        size_t ij_off = joff * ni + ioff;
        size_t ji_off = ioff * ni + joff;
        size_t NGv = nGv;
        int shls[2] = {ish, jsh};
        double *bufR = buf;
        double *bufI = bufR + dij * BLKSIZE * comp;
        double *cache = bufI + dij * BLKSIZE * comp;
        int grid0, grid1, dg, dijg;
        int i, j, n, ic, ij, ji;
        double *pout_ij, *pout_ji, *pbufR, *pbufI;

        for (grid0 = 0; grid0 < nGv; grid0 += BLKSIZE) {
                grid1 = MIN(grid0+BLKSIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg;
                NPdset0(bufR, dijg * comp);
                NPdset0(bufI, dijg * comp);
                if ((*intor)(bufR, bufI, shls, NULL, eval_gz,
                             fac, Gv+grid0, b, gxyz+grid0, gs, nGv, dg,
                             atm, natm, bas, nbas, env, cache)) {

if (ioff == joff) {
        for (ic = 0; ic < comp; ic++) {
                pout_ij = out + ((ij_off + ic * nij) * NGv + grid0) * OF_CMPLX;
                for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        pbufR = bufR + ic * dijg + dg * (j*di+i);
                        pbufI = bufI + ic * dijg + dg * (j*di+i);
                        ij = j * ni + i;
                        for (n = 0; n < dg; n++) {
                                pout_ij[(ij*NGv+n)*OF_CMPLX  ] += pbufR[n];
                                pout_ij[(ij*NGv+n)*OF_CMPLX+1] += pbufI[n];
                        }
                } }
        }
} else {
        for (ic = 0; ic < comp; ic++) {
                pout_ij = out + ((ij_off + ic * nij) * NGv + grid0) * OF_CMPLX;
                pout_ji = out + ((ji_off + ic * nij) * NGv + grid0) * OF_CMPLX;
                for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        pbufR = bufR + ic * dijg + dg * (j*di+i);
                        pbufI = bufI + ic * dijg + dg * (j*di+i);
                        ij = j * nj + i;
                        ji = i * nj + j;
                        for (n = 0; n < dg; n++) {
                                pout_ij[(ij*NGv+n)*OF_CMPLX  ] += pbufR[n];
                                pout_ij[(ij*NGv+n)*OF_CMPLX+1] += pbufI[n];
                                pout_ji[(ji*NGv+n)*OF_CMPLX  ] += pbufR[n];
                                pout_ji[(ji*NGv+n)*OF_CMPLX+1] += pbufI[n];
                        }
                } }
        }
}
                }
        }
}

void GTO_ft_zfill_s2(FPtrIntor intor, FPtr_eval_gz eval_gz,
                     double *out, int comp, int ish, int jsh, double *buf,
                     int *shls_slice, int *ao_loc, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        ish += ish0;
        jsh += jsh0;
        int ioff = ao_loc[ish] - ao_loc[ish0];
        int joff = ao_loc[jsh] - ao_loc[jsh0];
        if (ioff < joff) {
                return;
        }

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int i0 = ao_loc[ish0];
        size_t off0 = i0 * (i0 + 1) / 2;
        size_t off = ioff * (ioff + 1) / 2 - off0 + joff;
        size_t nij = ao_loc[ish1] * (ao_loc[ish1] + 1) / 2 - off0;
        size_t NGv = nGv;
        int shls[2] = {ish, jsh};
        double *bufR = buf;
        double *bufI = bufR + dij * BLKSIZE * comp;
        double *cache = bufI + dij * BLKSIZE * comp;
        int grid0, grid1, dg, dijg;
        int i, j, n, ic, ip1;
        double *pout, *pbufR, *pbufI;

        for (grid0 = 0; grid0 < nGv; grid0 += BLKSIZE) {
                grid1 = MIN(grid0+BLKSIZE, nGv);
                dg = grid1 - grid0;
                dijg = dij * dg;
                NPdset0(bufR, dijg * comp);
                NPdset0(bufI, dijg * comp);
                if ((*intor)(bufR, bufI, shls, NULL, eval_gz,
                             fac, Gv+grid0, b, gxyz+grid0, gs, nGv, dg,
                             atm, natm, bas, nbas, env, cache)) {

if (ioff == joff) {
        ip1 = ioff + 1;
        for (ic = 0; ic < comp; ic++) {
                pout = out + ((off + ic * nij) * NGv + grid0) * OF_CMPLX;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                pbufR = bufR + ic * dijg + dg * (j*di+i);
                                pbufI = bufI + ic * dijg + dg * (j*di+i);
                                for (n = 0; n < dg; n++) {
                                        pout[(j*NGv+n)*OF_CMPLX  ] += pbufR[n];
                                        pout[(j*NGv+n)*OF_CMPLX+1] += pbufI[n];
                                }
                        }
                        pout += (ip1 + i) * NGv * OF_CMPLX;
                }
        }
} else {
        ip1 = ioff + 1;
        for (ic = 0; ic < comp; ic++) {
                pout = out + ((off + ic * nij) * NGv + grid0) * OF_CMPLX;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pbufR = bufR + ic * dijg + dg * (j*di+i);
                                pbufI = bufI + ic * dijg + dg * (j*di+i);
                                for (n = 0; n < dg; n++) {
                                        pout[(j*NGv+n)*OF_CMPLX  ] += pbufR[n];
                                        pout[(j*NGv+n)*OF_CMPLX+1] += pbufI[n];
                                }
                        }
                        pout += (ip1 + i) * NGv * OF_CMPLX;
                }
        }
}
                }
        }
}

static size_t max_cache_size(FPtrIntor intor, FPtr_eval_gz eval_gz, int *shls_slice,
                             double *Gv, double *b, int *gxyz, int *gs, int nGv,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        double complex fac = 0.;
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int sh0 = MIN(ish0, jsh0);
        int sh1 = MAX(ish1, jsh1);
        int blksize = MIN(nGv, BLKSIZE);
        int shls[2];
        int i, cache_size;
        size_t max_size = 0;
        for (i = sh0; i < sh1; i++) {
                shls[0] = i;
                shls[1] = i;
                cache_size = (*intor)(NULL, NULL, shls, NULL, eval_gz,
                                      fac, Gv, b, gxyz, gs, nGv, blksize,
                                      atm, natm, bas, nbas, env, NULL);
                max_size = MAX(max_size, cache_size);
        }
        return max_size * blksize;
}

/*
 * Fourier transform AO pairs and add to out (inplace)
 */
void GTO_ft_fill_drv(FPtrIntor intor, FPtr_eval_gz eval_gz, void (*fill)(),
                     double *out, int8_t *ovlp_mask, int comp,
                     int *shls_slice, int *ao_loc, double phase,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int nish = ish1 - ish0;
        int njsh = jsh1 - jsh0;
        double complex fac = cos(phase) + sin(phase)*_Complex_I;
        size_t di = GTOmax_shell_dim(ao_loc, shls_slice  , 2);
        size_t cache_size = max_cache_size(intor, eval_gz, shls_slice,
                                           Gv, b, gxyz, gs, nGv,
                                           atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int i, j, ij;
        double *buf = malloc(sizeof(double) * (cache_size +
                                               di*di*comp*BLKSIZE * OF_CMPLX));
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                j = ij / nish;
                i = ij % nish;
                if (!ovlp_mask[ij]) {
                        continue;
                }
                (*fill)(intor, eval_gz, out,
                        comp, i, j, buf, shls_slice, ao_loc, fac,
                        Gv, b, gxyz, gs, nGv, atm, natm, bas, nbas, env);
        }
        free(buf);
}
}


static const int _LEN_CART[] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};

/*
 * WHEREX_IF_L_INC1 = [xyz2addr(x,y,z) for x,y,z in loopcart(L_MAX) if x > 0]
 * WHEREY_IF_L_INC1 = [xyz2addr(x,y,z) for x,y,z in loopcart(L_MAX) if y > 0]
 * WHEREZ_IF_L_INC1 = [xyz2addr(x,y,z) for x,y,z in loopcart(L_MAX) if z > 0]
 */
static const int _UPIDY[] = {
        1,
        3, 4,
        6, 7, 8,
        10, 11, 12, 13,
        15, 16, 17, 18, 19,
        21, 22, 23, 24, 25, 26,
        28, 29, 30, 31, 32, 33, 34,
        36, 37, 38, 39, 40, 41, 42, 43,
        45, 46, 47, 48, 49, 50, 51, 52, 53,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        91, 92, 93, 94, 95, 96, 97, 98, 99,100,101,102,103,
       105,106,107,108,109,110,111,112,113,114,115,116,117,118,
       120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,
};
static const int _UPIDZ[] = {
        2,
        4, 5,
        7, 8, 9,
        11, 12, 13, 14,
        16, 17, 18, 19, 20,
        22, 23, 24, 25, 26, 27,
        29, 30, 31, 32, 33, 34, 35,
        37, 38, 39, 40, 41, 42, 43, 44,
        46, 47, 48, 49, 50, 51, 52, 53, 54,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        92, 93, 94, 95, 96, 97, 98, 99,100,101,102,103,104,
       106,107,108,109,110,111,112,113,114,115,116,117,118,119,
       121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,
};

#define WHEREX_IF_L_INC1(i)     i
#define WHEREY_IF_L_INC1(i)     _UPIDY[i]
#define WHEREZ_IF_L_INC1(i)     _UPIDZ[i]
#define STARTX_IF_L_DEC1(i)     0
#define STARTY_IF_L_DEC1(i)     ((i<2)?0:_LEN_CART[i-2])
#define STARTZ_IF_L_DEC1(i)     (_LEN_CART[i-1]-1)


/*
 * Reversed vrr2d. They are used by numint_uniform_grid.c
 */
void GTOplain_vrr2d_ket_inc1(double *out, double *g,
                             double *rirj, int li, int lj)
{
        if (lj == 0) {
                NPdcopy(out, g, _LEN_CART[li]);
                return;
        }
        int row_10 = _LEN_CART[li+1];
        int row_00 = _LEN_CART[li  ];
        int col_00 = _LEN_CART[lj-1];
        double *g00 = g;
        double *g10 = g + row_00*col_00;
        int i, j;
        double *p00, *p10;
        double *p01 = out;

        for (j = STARTX_IF_L_DEC1(lj); j < _LEN_CART[lj-1]; j++) {
                for (i = 0; i < row_00; i++) {
                        p00 = g00 + (j*row_00+i);
                        p10 = g10 + (j*row_10+WHEREX_IF_L_INC1(i));
                        p01[i] = p10[0] + rirj[0] * p00[0];
                }
                p01 += row_00;
        }
        for (j = STARTY_IF_L_DEC1(lj); j < _LEN_CART[lj-1]; j++) {
                for (i = 0; i < row_00; i++) {
                        p00 = g00 + (j*row_00+i);
                        p10 = g10 + (j*row_10+WHEREY_IF_L_INC1(i));
                        p01[i] = p10[0] + rirj[1] * p00[0];
                }
                p01 += row_00;
        }
        j = STARTZ_IF_L_DEC1(lj);
        if (j < _LEN_CART[lj-1]) {
                for (i = 0; i < row_00; i++) {
                        p00 = g00 + (j*row_00+i);
                        p10 = g10 + (j*row_10+WHEREZ_IF_L_INC1(i));
                        p01[i] = p10[0] + rirj[2] * p00[0];
                }
        }
}

void GTOreverse_vrr2d_ket_inc1(double *g01, double *g00,
                               double *rirj, int li, int lj)
{
        int row_10 = _LEN_CART[li+1];
        int row_00 = _LEN_CART[li  ];
        int col_00 = _LEN_CART[lj-1];
        double *g10 = g00 + row_00*col_00;
        double *p00, *p10;
        int i, j;

        for (j = STARTX_IF_L_DEC1(lj); j < _LEN_CART[lj-1]; j++) {
                for (i = 0; i < row_00; i++) {
                        p00 = g00 + (j*row_00+i);
                        p10 = g10 + (j*row_10+WHEREX_IF_L_INC1(i));
                        p10[0] += g01[i];
                        p00[0] += g01[i] * rirj[0];
                }
                g01 += row_00;
        }
        for (j = STARTY_IF_L_DEC1(lj); j < _LEN_CART[lj-1]; j++) {
                for (i = 0; i < row_00; i++) {
                        p00 = g00 + (j*row_00+i);
                        p10 = g10 + (j*row_10+WHEREY_IF_L_INC1(i));
                        p10[0] += g01[i];
                        p00[0] += g01[i] * rirj[1];
                }
                g01 += row_00;
        }
        j = STARTZ_IF_L_DEC1(lj);
        if (j < _LEN_CART[lj-1]) {
                for (i = 0; i < row_00; i++) {
                        p00 = g00 + (j*row_00+i);
                        p10 = g10 + (j*row_10+WHEREZ_IF_L_INC1(i));
                        p10[0] += g01[i];
                        p00[0] += g01[i] * rirj[2];
                }
        }
}
