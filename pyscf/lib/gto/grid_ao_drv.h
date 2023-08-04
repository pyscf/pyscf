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
#include <stdint.h>
#include <complex.h>
#include <math.h>
#include "cint.h"

// 2 slots of int param[]
#define POS_E1   0
#define TENSOR   1

#define LMAX            ANG_MAX
#define SIMDD           8
// 128s42p21d12f8g6h4i3j
#define NCTR_CART       128
#define NPRIMAX         40
#define BLKSIZE         56
#define EXPCUTOFF       50  // 1e-22
#define NOTZERO(e)      (fabs(e)>1e-18)

#ifndef HAVE_NONZERO_EXP
#define HAVE_NONZERO_EXP
inline static int _nonzero_in(double *exps, int count) {
        int n;
        int val = 0;
        for (n = 0; n < count; n++) {
                if NOTZERO(exps[n]) {
                        val = 1;
                        break;
                }
        }
        return val;
}
#endif

typedef int (*FPtr_exp)(double *ectr, double *coord, double *alpha, double *coeff,
                        int l, int nprim, int nctr, size_t ngrids, double fac);
typedef void (*FPtr_eval)(double *gto, double *ri, double *exps,
                          double *coord, double *alpha, double *coeff,
                          double *env, int l, int np, int nc,
                          size_t nao, size_t ngrids, size_t blksize);

void GTOnabla1(double *fx1, double *fy1, double *fz1,
               double *fx0, double *fy0, double *fz0, int l, double a);
void GTOx1(double *fx1, double *fy1, double *fz1,
           double *fx0, double *fy0, double *fz0, int l, double *ri);
int GTOprim_exp(double *eprim, double *coord, double *alpha, double *coeff,
                int l, int nprim, int nctr, size_t ngrids, double fac);
int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac);
int GTOcontract_exp1(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac);

void GTOeval_sph_drv(FPtr_eval feval, FPtr_exp fexp, double fac,
                     int ngrids, int param[], int *shls_slice, int *ao_loc,
                     double *ao, double *coord, uint8_t *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env);

void GTOeval_cart_drv(FPtr_eval feval, FPtr_exp fexp, double fac,
                      int ngrids, int param[], int *shls_slice, int *ao_loc,
                      double *ao, double *coord, uint8_t *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env);

void GTOeval_spinor_drv(FPtr_eval feval, FPtr_exp fexp, void (*c2s)(), double fac,
                        int ngrids, int param[], int *shls_slice, int *ao_loc,
                        double complex *ao, double *coord, uint8_t *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env);

#define GTO_D_I(o, i, l) \
        GTOnabla1(fx##o, fy##o, fz##o, fx##i, fy##i, fz##i, l, alpha[k])
/* r-R_0, R_0 is (0,0,0) */
#define GTO_R0I(o, i, l) \
        GTOx1(fx##o, fy##o, fz##o, fx##i, fy##i, fz##i, l, ri)
/* r-R_C, R_C is common origin */
#define GTO_RCI(o, i, l) \
        GTOx1(fx##o, fy##o, fz##o, fx##i, fy##i, fz##i, l, dri)
/* origin from center of each basis
 * x1(fx1, fy1, fz1, fx0, fy0, fz0, l, 0, 0, 0) */
#define GTO_R_I(o, i, l) \
        fx##o = fx##i + SIMDD; \
        fy##o = fy##i + SIMDD; \
        fz##o = fz##i + SIMDD

#define ALIGN8_UP(buf) (void *)(((uintptr_t)buf + 7) & (-(uintptr_t)8))
