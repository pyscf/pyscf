/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
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

#if !defined HAVE_DEFINED_CINTENVVARS_H
#define HAVE_DEFINED_CINTENVVARS_H
typedef struct {
        int *atm;
        int *bas;
        double *env;
        int *shls;
        int natm;
        int nbas;

        int i_l;
        int j_l;
        int k_l;
        int l_l;
        int nfi;  // number of cartesion components
        int nfj;
        int nfk;
        int nfl;
        int nf;  // = nfi*nfj*nfk*nfl;
        int _padding;
        int x_ctr[4];

        int gbits;
        int ncomp_e1; // = 1 if spin free, = 4 when spin included, it
        int ncomp_e2; // corresponds to POSX,POSY,POSZ,POS1, see cint_const.h
        int ncomp_tensor; // e.g. = 3 for gradients

        /* values may diff based on the g0_2d4d algorithm */
        int li_ceil; // power of x, == i_l if nabla is involved, otherwise == i_l
        int lj_ceil;
        int lk_ceil;
        int ll_ceil;
        int g_stride_i; // nrys_roots * shift of (i++,k,l,j)
        int g_stride_k; // nrys_roots * shift of (i,k++,l,j)
        int g_stride_l; // nrys_roots * shift of (i,k,l++,j)
        int g_stride_j; // nrys_roots * shift of (i,k,l,j++)
        int nrys_roots;
        int g_size;  // ref to cint2e.c g = malloc(sizeof(double)*g_size)

        int g2d_ijmax;
        int g2d_klmax;
        double common_factor;
        double _padding1;
        double rirj[3]; // diff by sign in different g0_2d4d algorithm
        double rkrl[3];
        double *rx_in_rijrx;
        double *rx_in_rklrx;

        double *ri;
        double *rj;
        double *rk;
        double *rl;

        void (*f_g0_2e)();
        void (*f_g0_2d4d)();
        void (*f_gout)();

        int *idx;
        double ai;
        double aj;
        double ak;
        double al;

// Other definitions in CINTEnvVars are different in libcint and qcint.
// They should not used in this function.
} CINTEnvVars;
#endif

void GTO_ft_init1e_envs(CINTEnvVars *envs, const int *ng, const int *shls,
                        const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env);

int GTO_ft_aopair_drv(double complex *out, int *dims,
                      int (*eval_aopair)(), void (*eval_gz)(), void (*c2s)(),
                      double complex fac, double *Gv, double *b, int *gxyz,
                      int *gs, size_t NGv, CINTEnvVars *envs);

void GTO_ft_c2s_cart(double complex *out, double complex *gctr,
                     int *dims, CINTEnvVars *envs, size_t NGv);
void GTO_ft_c2s_sph(double complex *out, double complex *gctr,
                    int *dims, CINTEnvVars *envs, size_t NGv);

int GTO_aopair_lazy_contract(double complex *gctr, CINTEnvVars *envs,
                             void (*eval_gz)(), double complex fac,
                             double *Gv, double *b, int *gxyz, int *gs,size_t NGv);

int GTO_ft_ovlp_cart(double complex *out, int *shls, int *dims,
                     int (*eval_aopair)(), void (*eval_gz)(), double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env);
int GTO_ft_ovlp_sph(double complex *out, int *shls, int *dims,
                    int (*eval_aopair)(), void (*eval_gz)(), double complex fac,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env);
