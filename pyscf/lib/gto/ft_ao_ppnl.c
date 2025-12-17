/* Copyright 2025 The PySCF Developers. All Rights Reserved.
  
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
 * Authors: Christopher Hillenbrand <chillenbrand15@gmail.com>,
 *          Qiming Sun <osirpt.sun@gmail.com>
 *          
 */


#include <stdlib.h>
#include <complex.h>
#include "cint.h"
#include "gto/ft_ao.h"


#define G1E_R_I(f, g, li, lj, lk) f = g + bs * envs->g_stride_i;
#define G1E_R_J(f, g, li, lj, lk) f = g + bs * envs->g_stride_j;
#define G1E_R_K(f, g, li, lj, lk) f = g + bs * envs->g_stride_k;

/*
 * Compare: CINTgout1e_int1e_r2_origi in libcint,
 * file cint1e_a.c
 * ( r dot r \| )
 */

static void inner_prod_r2_origi(double *gout, double *g, int *idx, FTEnvVars *envs, int empty)
{
        int nf = envs->nf;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double *g0R = g;
        double *g0I = g0R + g_size * 3;
        double *g1R = g0I + g_size * 3;
        double *g1I = g1R + g_size * 3;
        double *g2R = g1I + g_size * 3;
        double *g2I = g2R + g_size * 3;
        double *g3R = g2I + g_size * 3;
        double *g3I = g3R + g_size * 3;
        double *goutR = gout;
        double *goutI = gout + nf * bs;
        double xyR, xyI, sR, sI;
        int ix, iy, iz, n, k;
        G1E_R_I(g1R, g0R, envs->i_l+1, envs->j_l, 0);
        G1E_R_I(g3R, g1R, envs->i_l+0, envs->j_l, 0);

        G1E_R_I(g1I, g0I, envs->i_l+1, envs->j_l, 0);
        G1E_R_I(g3I, g1I, envs->i_l+0, envs->j_l, 0);

        if (empty) {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR, sI, g3, g0, g0);
                                ZMAD(sR, sI, g0, g3, g0);
                                ZMAD(sR, sI, g0, g0, g3);
                                goutR[n*bs+k] = sR;
                                goutI[n*bs+k] = sI;
                        }
                }
        } else {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR, sI, g3, g0, g0);
                                ZMAD(sR, sI, g0, g3, g0);
                                ZMAD(sR, sI, g0, g0, g3);
                                goutR[n*bs+k] += sR;
                                goutI[n*bs+k] += sI;
                        }
                }
        }
}


int GTO_ft_r2_origi_cart(double *outR, double *outI, int *shls, int *dims,
                      FPtr_eval_gz eval_gz, double complex fac,
                      double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                      int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {2, 0, 0, 0, 2, 1, 1, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_r2_origi;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_cart, &envs);
}

int GTO_ft_r2_origi_sph(double *outR, double *outI, int *shls, int *dims,
                     FPtr_eval_gz eval_gz, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                     int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {2, 0, 0, 0, 2, 1, 1, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_r2_origi;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_sph, &envs);
}




/*
 * Compare: CINTgout1e_int1e_r4_origi in libcint,
 * file cint1e_a.c
 * ( r dot r r dot r \| )
 */

static void inner_prod_r4_origi(double *gout, double *g, int *idx, FTEnvVars *envs, int empty)
{
        int nf = envs->nf;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double *g0R = g;
        double *g0I = g0R + g_size * 3;
        double *g1R = g0I + g_size * 3;
        double *g1I = g1R + g_size * 3;
        double *g2R = g1I + g_size * 3;
        double *g2I = g2R + g_size * 3;
        double *g3R = g2I + g_size * 3;
        double *g3I = g3R + g_size * 3;
        double *g4R = g3I + g_size * 3;
        double *g4I = g4R + g_size * 3;
        double *g5R = g4I + g_size * 3;
        double *g5I = g5R + g_size * 3;
        double *g6R = g5I + g_size * 3;
        double *g6I = g6R + g_size * 3;
        double *g7R = g6I + g_size * 3;
        double *g7I = g7R + g_size * 3;
        double *g8R = g7I + g_size * 3;
        double *g8I = g8R + g_size * 3;
        double *g9R = g8I + g_size * 3;
        double *g9I = g9R + g_size * 3;
        double *g10R = g9I + g_size * 3;
        double *g10I = g10R + g_size * 3;
        double *g11R = g10I + g_size * 3;
        double *g11I = g11R + g_size * 3;
        double *g12R = g11I + g_size * 3;
        double *g12I = g12R + g_size * 3;
        double *g13R = g12I + g_size * 3;
        double *g13I = g13R + g_size * 3;
        double *g14R = g13I + g_size * 3;
        double *g14I = g14R + g_size * 3;
        double *g15R = g14I + g_size * 3;
        double *g15I = g15R + g_size * 3;

        double *goutR = gout;
        double *goutI = gout + nf * bs;
        double xyR, xyI, sR, sI;
        int ix, iy, iz, n, k;
        G1E_R_I(g1R, g0R, envs->i_l+3, envs->j_l, 0);
        G1E_R_I(g3R, g1R, envs->i_l+2, envs->j_l, 0);
        G1E_R_I(g4R, g0R, envs->i_l+1, envs->j_l, 0);
        G1E_R_I(g7R, g3R, envs->i_l+1, envs->j_l, 0);
        G1E_R_I(g12R, g4R, envs->i_l+0, envs->j_l, 0);
        G1E_R_I(g15R, g7R, envs->i_l+0, envs->j_l, 0);

        G1E_R_I(g1I, g0I, envs->i_l+3, envs->j_l, 0);
        G1E_R_I(g3I, g1I, envs->i_l+2, envs->j_l, 0);
        G1E_R_I(g4I, g0I, envs->i_l+1, envs->j_l, 0);
        G1E_R_I(g7I, g3I, envs->i_l+1, envs->j_l, 0);
        G1E_R_I(g12I, g4I, envs->i_l+0, envs->j_l, 0);
        G1E_R_I(g15I, g7I, envs->i_l+0, envs->j_l, 0);

        if (empty) {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR, sI, g15, g0, g0);
                                ZMAD_MUL(sR, sI, g12, g3, g0, 2.0);
                                ZMAD_MUL(sR, sI, g12, g0, g3, 2.0);
                                ZMAD(sR, sI, g0, g15, g0);
                                ZMAD_MUL(sR, sI, g0, g12, g3, 2.0);
                                ZMAD(sR, sI, g0, g0, g15);

                                goutR[n*bs+k] = sR;
                                goutI[n*bs+k] = sI;
                        }
                }
        } else {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR, sI, g15, g0, g0);
                                ZMAD_MUL(sR, sI, g12, g3, g0, 2.0);
                                ZMAD_MUL(sR, sI, g12, g0, g3, 2.0);
                                ZMAD(sR, sI, g0, g15, g0);
                                ZMAD_MUL(sR, sI, g0, g12, g3, 2.0);
                                ZMAD(sR, sI, g0, g0, g15);

                                goutR[n*bs+k] += sR;
                                goutI[n*bs+k] += sI;
                        }
                }
        }
}

int GTO_ft_r4_origi_cart(double *outR, double *outI, int *shls, int *dims,
                      FPtr_eval_gz eval_gz, double complex fac,
                      double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                      int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {4, 0, 0, 0, 4, 1, 1, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_r4_origi;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_cart, &envs);
}

int GTO_ft_r4_origi_sph(double *outR, double *outI, int *shls, int *dims,
                     FPtr_eval_gz eval_gz, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                     int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {4, 0, 0, 0, 4, 1, 1, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_r4_origi;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_sph, &envs);
}

/*
 * ( x^1 i | j )
 * ri is the shift from the center R_O to the center of |i>
 * r - R_O = (r-R_i) + ri, ri = R_i - R_O
 * Note: this is called separately for real and imaginary parts.
 */
void GTO_ft_x1i(double *f, double *g, double ri[3], int li, int lj,
                FTEnvVars *envs) {
  int dj = envs->g_stride_j;
  int bs = envs->block_size;
  size_t g_size = envs->g_size * bs;
  double *gx = g;
  double *gy = gx + g_size;
  double *gz = gy + g_size;
  double *fx = f;
  double *fy = fx + g_size;
  double *fz = fy + g_size;
  int i, j, k, ptr;


  for (j = 0; j <= lj; j++) {
    ptr = dj * j;
#pragma GCC ivdep
    for (i = ptr; i <= ptr + li; i++) {
#pragma GCC ivdep
      for (k = 0; k < bs; k++) {
        fx[i * bs + k] = gx[(i + 1) * bs + k] + ri[0] * gx[i * bs + k];
        fy[i * bs + k] = gy[(i + 1) * bs + k] + ri[1] * gy[i * bs + k];
        fz[i * bs + k] = gz[(i + 1) * bs + k] + ri[2] * gz[i * bs + k];
      }
    }
  }
}

void GTO_ft_x1j(double *f, double *g, double rj[3], int li, int lj,
                FTEnvVars *envs) {
  int dj = envs->g_stride_j;
  int bs = envs->block_size;
  size_t g_size = envs->g_size * bs;
  double *gx = g;
  double *gy = gx + g_size;
  double *gz = gy + g_size;
  double *fx = f;
  double *fy = fx + g_size;
  double *fz = fy + g_size;
  int i, j, k, ptr;

  for (j = 0; j <= lj; j++) {
    ptr = dj * j;
#pragma GCC ivdep
    for (i = ptr; i <= ptr + li; i++) {
#pragma GCC ivdep
      for (k = 0; k < bs; k++) {
        fx[i * bs + k] = gx[(i + dj) * bs + k] + rj[0] * gx[i * bs + k];
        fy[i * bs + k] = gy[(i + dj) * bs + k] + rj[1] * gy[i * bs + k];
        fz[i * bs + k] = gz[(i + dj) * bs + k] + rj[2] * gz[i * bs + k];
      }
    }
  }
}
