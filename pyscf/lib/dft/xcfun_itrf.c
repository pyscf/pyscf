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
 *
 * xcfun Library from
 * https://github.com/dftlibs/xcfun
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <XCFun/xcfun.h>
#include "config.h"

int XCFUN_max_deriv_order = XCFUN_MAX_DERIV_ORDER;

static int eval_xc(xcfun_t* fun, int deriv, xcfun_vars vars,
                   int np, int ncol, int outlen, double *rho, double *output)
{
        int err = xcfun_eval_setup(fun, vars, XC_PARTIAL_DERIVATIVES, deriv);
        if (err != 0) {
                fprintf(stderr, "Failed to initialize xcfun %d\n", err);
                return err;
        }
        assert(ncol == xcfun_input_length(fun));

        //xcfun_eval_vec(fun, np, rho, ncol, output, outlen);
#pragma omp parallel default(none) \
        shared(fun, rho, output, np, ncol, outlen)
{
        int i;
#pragma omp for nowait schedule(static)
        for (i=0; i < np; i++) {
                xcfun_eval(fun, rho+i*ncol, output+i*outlen);
        }
}
        return 0;
}

int XCFUN_eval_xc(int nfn, int *fn_id, double *fac, double *omega,
                  int spin, int deriv, int nvar, int np, int outlen,
                  double *rho_u, double *output)
{
        int i, err;
        double *rho_d = rho_u + np * nvar;
        double *rho;
        double *gxu, *gyu, *gzu, *gxd, *gyd, *gzd, *tau_u, *tau_d;
        const char *name;

        assert(xcfun_is_compatible_library() == true);
        xcfun_t* fun = xcfun_new();
        for (i = 0; i < nfn; i++) {
                name = xcfun_enumerate_parameters(fn_id[i]);
                xcfun_set(fun, name, fac[i]);

                if (omega[i] != 0) {
                        xcfun_set(fun, "RANGESEP_MU", omega[i]);
                }
                //xcfun_set(fun, "CAM_ALPHA", val);
                //xcfun_set(fun, "CAM_BETA", val);
        }

        if (spin == 0) {
                if (xcfun_is_metagga(fun)) {
                        rho = malloc(sizeof(double) * np*3);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        tau_u = rho_u + np * 4;
                        for (i = 0; i < np; i++) {
                                rho[i*3+0] = rho_u[i];
                                rho[i*3+1] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                rho[i*3+2] = tau_u[i];
                        }
                        err = eval_xc(fun, deriv, XC_N_GNN_TAUN, np, 3, outlen, rho, output);
                        free(rho);
                } else if (xcfun_is_gga(fun)) {
                        rho = malloc(sizeof(double) * np*2);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        }
                        err = eval_xc(fun, deriv, XC_N_GNN, np, 2, outlen, rho, output);
                        free(rho);
                } else { // LDA
                        rho = rho_u;
                        err = eval_xc(fun, deriv, XC_N, np, 1, outlen, rho, output);
                }
// xcfun computed rho*Exc[rho] for zeroth order derivative instead of Exc[rho]
                for (i = 0; i < np; i++) {
                        output[i*outlen] /= rho_u[i] + 1e-150;
                }
        } else {
                if (xcfun_is_metagga(fun)) {
                        rho = malloc(sizeof(double) * np*7);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        gxd = rho_d + np;
                        gyd = rho_d + np * 2;
                        gzd = rho_d + np * 3;
                        tau_u = rho_u + np * 4;
                        tau_d = rho_d + np * 4;
                        for (i = 0; i < np; i++) {
                                rho[i*7+0] = rho_u[i];
                                rho[i*7+1] = rho_d[i];
                                rho[i*7+2] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                rho[i*7+3] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                rho[i*7+4] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                                rho[i*7+5] = tau_u[i];
                                rho[i*7+6] = tau_d[i];
                        }
                        err = eval_xc(fun, deriv, XC_A_B_GAA_GAB_GBB_TAUA_TAUB, np, 7, outlen, rho, output);
                        free(rho);
                } else if (xcfun_is_gga(fun)) {
                        rho = malloc(sizeof(double) * np*5);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        gxd = rho_d + np;
                        gyd = rho_d + np * 2;
                        gzd = rho_d + np * 3;
                        for (i = 0; i < np; i++) {
                                rho[i*5+0] = rho_u[i];
                                rho[i*5+1] = rho_d[i];
                                rho[i*5+2] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                rho[i*5+3] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                rho[i*5+4] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        }
                        err = eval_xc(fun, deriv, XC_A_B_GAA_GAB_GBB, np, 5, outlen, rho, output);
                        free(rho);
                } else { // LDA
                        rho = malloc(sizeof(double) * np*2);
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                        }
                        err = eval_xc(fun, deriv, XC_A_B, np, 2, outlen, rho, output);
                        free(rho);
                }
                for (i = 0; i < np; i++) {
                        output[i*outlen] /= rho_u[i] + rho_d[i] + 1e-150;
                }
        }
        xcfun_delete(fun);
        return err;
}

/*
 * XC_LDA      0 // Local density
 * XC_GGA      1 // Local density & gradient
 * XC_MGGA     2 // Local density, gradient and kinetic energy density
 */
int XCFUN_xc_type(int fn_id)
{
        xcfun_t* fun = xcfun_new();
        const char *name = xcfun_enumerate_parameters(fn_id);
        xcfun_set(fun, name, 1.);
        int type = 0;
        if (xcfun_is_metagga(fun)) {
                type = 2;
        } else if (xcfun_is_gga(fun)) {
                type = 1;
        }
        xcfun_delete(fun);
        return type;
}
