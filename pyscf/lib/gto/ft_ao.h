/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
  
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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include "cint.h"

#ifndef HAVE_DEFINED_FTENVVARS_H
#define HAVE_DEFINED_FTENVVARS_H
typedef struct {
        int *atm;
        int *bas;
        double *env;
        int *shls;
        int natm;
        int nbas;

        int i_l;
        int j_l;
        int nfi;  // number of cartesian components
        int nfj;
        int nf;  // = nfi*nfj
        int ngrids;  // number of grids or planewaves
        int x_ctr[2];

        int gbits;
        int ncomp_e1; // = 1 if spin free, = 4 when spin included, it
        int ncomp_tensor;

        int li_ceil; // power of x, == i_l if nabla is involved, otherwise == i_l
        int lj_ceil;
        int g_stride_i;
        int g_stride_j;
        int g_size;  // ref to cint2e.c g = malloc(sizeof(double)*g_size)
        double expcutoff;
        double ai[1];
        double aj[1];
        double rirj[3];
        double *rx_in_rijrx;
        double *ri;
        double *rj;
        void (*f_gout)();

        double *Gv;
        double *b;
        int *gxyz;
        int *gs;
        double complex common_factor;
        int block_size;
} FTEnvVars;
#endif

typedef void (*FPtr_eval_gz)(double *gzR, double *gzI, double fac, double aij,
                             double *rij, FTEnvVars *envs, double *cache);

typedef int (*FPtrIntor)(double *outR, double *outI, int *shls, int *dims,
                         FPtr_eval_gz eval_gz, double complex fac,
                         double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                         int *atm, int natm, int *bas, int nbas, double *env, double *cache);

void GTO_ft_init1e_envs(FTEnvVars *envs, int *ng, int *shls, double complex fac,
                        double *Gv, double *b, int *gxyz, int *gs,
                        int nGv, int block_size,
                        int *atm, int natm, int *bas, int nbas, double *env);

int GTO_ft_aopair_drv(double *outR, double *outI, int *dims,
                      FPtr_eval_gz eval_gz, double *cache, void (*f_c2s)(),
                      FTEnvVars *envs);

void GTO_ft_c2s_cart(double *out, double *gctr, int *dims,
                     FTEnvVars *envs, double *cache);
void GTO_ft_c2s_sph(double *out, double *gctr, int *dims,
                    FTEnvVars *envs, double *cache);

int GTO_ft_ovlp_cart(double *outR, double *outI, int *shls, int *dims,
                     FPtr_eval_gz eval_gz, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                     int *atm, int natm, int *bas, int nbas, double *env, double *cache);
int GTO_ft_ovlp_sph(double *outR, double *outI, int *shls, int *dims,
                    FPtr_eval_gz eval_gz, double complex fac,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                    int *atm, int natm, int *bas, int nbas, double *env, double *cache);

void GTO_ft_x1i(double *f, double *g, double ri[3], int li, int lj,
                FTEnvVars *envs);
void GTO_ft_x1j(double *f, double *g, double rj[3], int li, int lj,
                FTEnvVars *envs);

#define ZMUL(outR, outI, gx, gy, gz) \
        xyR = gx##R[ix*bs+k] * gy##R[iy*bs+k] - gx##I[ix*bs+k] * gy##I[iy*bs+k]; \
        xyI = gx##R[ix*bs+k] * gy##I[iy*bs+k] + gx##I[ix*bs+k] * gy##R[iy*bs+k]; \
        outR = xyR * gz##R[iz*bs+k] - xyI * gz##I[iz*bs+k]; \
        outI = xyR * gz##I[iz*bs+k] + xyI * gz##R[iz*bs+k];

#define ZMAD(outR, outI, gx, gy, gz) \
        xyR = gx##R[ix*bs+k] * gy##R[iy*bs+k] - gx##I[ix*bs+k] * gy##I[iy*bs+k]; \
        xyI = gx##R[ix*bs+k] * gy##I[iy*bs+k] + gx##I[ix*bs+k] * gy##R[iy*bs+k]; \
        outR += xyR * gz##R[iz*bs+k] - xyI * gz##I[iz*bs+k]; \
        outI += xyR * gz##I[iz*bs+k] + xyI * gz##R[iz*bs+k];

#define ZMAD_MUL(outR, outI, gx, gy, gz, factor) \
        xyR = gx##R[ix*bs+k] * gy##R[iy*bs+k] - gx##I[ix*bs+k] * gy##I[iy*bs+k]; \
        xyI = gx##R[ix*bs+k] * gy##I[iy*bs+k] + gx##I[ix*bs+k] * gy##R[iy*bs+k]; \
        outR += factor * (xyR * gz##R[iz*bs+k] - xyI * gz##I[iz*bs+k]); \
        outI += factor * (xyR * gz##I[iz*bs+k] + xyI * gz##R[iz*bs+k]);
