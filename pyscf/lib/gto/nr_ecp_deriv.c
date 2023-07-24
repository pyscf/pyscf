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
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "gto/nr_ecp.h"

int ECPtype1_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                  int *atm, int natm, int *bas, int nbas, double *env,
                  ECPOpt *opt, double *cache);
int ECPtype2_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                  int *atm, int natm, int *bas, int nbas, double *env,
                  ECPOpt *opt, double *cache);
int ECPscalar_c2s_factory(Function_cart fcart, double *gctr, int comp, int *shls,
                          int *ecpbas, int necpbas, int *atm, int natm,
                          int *bas, int nbas, double *env, ECPOpt *opt,
                          double *cache);
void ECPscalar_distribute(double *out, double *gctr, const int *dims,
                          const int comp, const int di, const int dj);
void ECPscalar_distribute0(double *out, const int *dims,
                           const int comp, const int di, const int dj);
int ECPscalar_cache_size(int comp, int *shls,
                         int *atm, int natm, int *bas, int nbas, double *env);

/*
static int _x_addr[] = {
  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
 15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
 30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
 45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
 60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
 75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
 90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
}; */
static int _y_addr[] = {
  1,   3,   4,   6,   7,   8,  10,  11,  12,  13,  15,  16,  17,  18,  19,
 21,  22,  23,  24,  25,  26,  28,  29,  30,  31,  32,  33,  34,  36,  37,
 38,  39,  40,  41,  42,  43,  45,  46,  47,  48,  49,  50,  51,  52,  53,
 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  66,  67,  68,  69,  70,
 71,  72,  73,  74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  85,  86,
 87,  88,  89,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102,
103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
};
static int _z_addr[] = {
  2,   4,   5,   7,   8,   9,  11,  12,  13,  14,  16,  17,  18,  19,  20,
 22,  23,  24,  25,  26,  27,  29,  30,  31,  32,  33,  34,  35,  37,  38,
 39,  40,  41,  42,  43,  44,  46,  47,  48,  49,  50,  51,  52,  53,  54,
 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  67,  68,  69,  70,  71,
 72,  73,  74,  75,  76,  77,  79,  80,  81,  82,  83,  84,  85,  86,  87,
 88,  89,  90,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
};

static int _cart_pow_y[] = {
        0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1,
        0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5,
        4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,12,11,
       10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,13,12,11,10, 9, 8, 7, 6, 5,
        4, 3, 2, 1, 0,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};
static int _cart_pow_z[] = {
        0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
        5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
        4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 0, 1,
        2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9,10,11,12,13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
};

// ecpbas needs to be grouped according to atom Id. Searching for the first
// shell that matches the atm_id
static int _one_shell_ecpbas(int *nsh, int atm_id,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        int *all_ecp = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int necpbas = (int)env[AS_NECPBAS];
        int i;
        int n = 0;
        int shl_id = -1;
        for (i = 0; i < necpbas; i++) {
                if (atm_id == all_ecp[ATOM_OF+i*BAS_SLOTS]) {
                        shl_id = i;
                        break;
                }
        }
        for (; i < necpbas; i++) {
                if (atm_id == all_ecp[ATOM_OF+i*BAS_SLOTS]) {
                        n++;
                } else {
                        break;
                }
        }
        *nsh = n;
        return shl_id;
}

static void _uncontract_bas(int *fakbas, int *shls,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
        int i;
        for (i = 0; i < npi; i++) {
                fakbas[i*BAS_SLOTS+ATOM_OF  ] = bas[ish*BAS_SLOTS+ATOM_OF];
                fakbas[i*BAS_SLOTS+ANG_OF   ] = bas[ish*BAS_SLOTS+ANG_OF ];
                fakbas[i*BAS_SLOTS+NPRIM_OF ] = 1;
                fakbas[i*BAS_SLOTS+NCTR_OF  ] = 1;
                fakbas[i*BAS_SLOTS+PTR_EXP  ] = bas[ish*BAS_SLOTS+PTR_EXP] + i;
                fakbas[i*BAS_SLOTS+PTR_COEFF] = bas[ish*BAS_SLOTS+PTR_EXP] + i;
        }
        fakbas += npi * BAS_SLOTS;
        for (i = 0; i < npj; i++) {
                fakbas[i*BAS_SLOTS+ATOM_OF  ] = bas[jsh*BAS_SLOTS+ATOM_OF];
                fakbas[i*BAS_SLOTS+ANG_OF   ] = bas[jsh*BAS_SLOTS+ANG_OF ];
                fakbas[i*BAS_SLOTS+NPRIM_OF ] = 1;
                fakbas[i*BAS_SLOTS+NCTR_OF  ] = 1;
                fakbas[i*BAS_SLOTS+PTR_EXP  ] = bas[jsh*BAS_SLOTS+PTR_EXP] + i;
                fakbas[i*BAS_SLOTS+PTR_COEFF] = bas[jsh*BAS_SLOTS+PTR_EXP] + i;
        }
}

static void _l_down(double *out, double *buf1,
                    double fac, double ai, int li, int nfj)
{
        const int nfi = (li+1) * (li+2) / 2;
        const int nfi1 = (li+2) * (li+3) / 2;
        int i, j;
        double *outx = out;
        double *outy = outx + nfi*nfj;
        double *outz = outy + nfi*nfj;

        if (li == 0) {
                fac *= -2./sqrt(3.) * ai;
        } else if (li == 1) {
                fac *= -2.*0.488602511902919921 * ai;
        } else {
                fac *= -2. * ai;
        }
        for (j = 0; j < nfj; j++) {
                for (i = 0; i < nfi; i++) {
                        outx[j*nfi+i] = fac * buf1[j*nfi1+        i ];
                        outy[j*nfi+i] = fac * buf1[j*nfi1+_y_addr[i]];
                        outz[j*nfi+i] = fac * buf1[j*nfi1+_z_addr[i]];
                }
        }
}

static void _l_up(double *out, double *buf1, double fac, int li, int nfj)
{
        const int nfi = (li+1) * (li+2) / 2;
        const int nfi0 = li * (li+1) / 2;
        int i, j;
        double *outx = out;
        double *outy = outx + nfi*nfj;
        double *outz = outy + nfi*nfj;
        double xfac, yfac, zfac;

        if (li == 1) {
                fac *= sqrt(3.);
        } else if (li == 2) {
                fac *= 1./0.488602511902919921;
        }
        for (i = 0; i < nfi0; i++) {
                yfac = fac * (_cart_pow_y[i] + 1);
                zfac = fac * (_cart_pow_z[i] + 1);
                xfac = fac * (li-1 - _cart_pow_y[i] - _cart_pow_z[i] + 1);
                for (j = 0; j < nfj; j++) {
                        outx[j*nfi+        i ] += xfac * buf1[j*nfi0+i];
                        outy[j*nfi+_y_addr[i]] += yfac * buf1[j*nfi0+i];
                        outz[j*nfi+_z_addr[i]] += zfac * buf1[j*nfi0+i];
                }
        }
}

static int _deriv1_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        ECPOpt *opt, double *cache)
{
        if (necpbas == 0) {
                return 0;
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const int nfi0 = li * (li+1) / 2;
        const int nfi1 = (li+2) * (li+3) / 2;
        const int di = nfi * nci;
        const int dj = nfj * ncj;
        const int dij = di * dj;
        const double *expi = env + bas[PTR_EXP+ish*BAS_SLOTS];
        const double *expj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
        const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
        const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
        int nfakbas = npi + npj;
        int *fakbas;
        MALLOC_INSTACK(fakbas, (npi+npj) * BAS_SLOTS);
        _uncontract_bas(fakbas, shls, atm, natm, bas, nbas, env);
        double *buf1;
        MALLOC_INSTACK(buf1, (nfi1*nfj + nfi*nfj*3));
        double *gprim = buf1 + nfi1*nfj;

        int has_value = 0;
        int shls1[2];
        double fac;

        int i, j, ip, jp, ic, jc, n;
        double *gctrx = gctr;
        double *gctry = gctrx + dij;
        double *gctrz = gctry + dij;
        double *gpx = gprim;
        double *gpy = gpx + nfi*nfj;
        double *gpz = gpy + nfi*nfj;

        for (jp = 0; jp < npj; jp++) {
        for (ip = 0; ip < npi; ip++) {
                shls1[0] = ip;
                shls1[1] = npi + jp;
/* divide (expi[ip] * expj[jp]) because the exponents were used as normalization
 * coefficients for primitive GTOs in function _uncontract_bas */
                fac = 1. / (expi[ip] * expj[jp]);
                fakbas[ip*BAS_SLOTS+ANG_OF] = li + 1;
                NPdset0(buf1, nfi1*nfj);
                has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                _l_down(gprim, buf1, fac, expi[ip], li, nfj);

                if (li > 0) {
                        fakbas[ip*BAS_SLOTS+ANG_OF] = li - 1;
                        NPdset0(buf1, nfi0*nfj);
                        has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                   fakbas, nfakbas, env, opt, cache) | has_value);
                        has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                   fakbas, nfakbas, env, opt, cache) | has_value);
                        _l_up(gprim, buf1, fac, li, nfj);
                }

                for (jc = 0; jc < ncj; jc++) {
                for (ic = 0; ic < nci; ic++) {
                        fac = ci[ic*npi+ip] * cj[jc*npj+jp];
                        n = jc*nfj*di + ic*nfi;
                        for (j = 0; j < nfj; j++) {
                        for (i = 0; i < nfi; i++) {
                                gctrx[n+j*di+i] += fac * gpx[j*nfi+i];
                                gctry[n+j*di+i] += fac * gpy[j*nfi+i];
                                gctrz[n+j*di+i] += fac * gpz[j*nfi+i];
                        } }
                } }
        } }

        return has_value;
}

static int _cart_factory(Function_cart intor_cart, double *out, int comp,
                         int *dims, int *shls, int *ecpbas, int necpbas,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         ECPOpt *opt, double *cache)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int di = (li+1) * (li+2) / 2 * bas[NCTR_OF+ish*BAS_SLOTS];;
        const int dj = (lj+1) * (lj+2) / 2 * bas[NCTR_OF+jsh*BAS_SLOTS];
        const int dij = di * dj;

        if (out == NULL) {
                int cache_size = ECPscalar_cache_size(comp*2, shls,
                                                      atm, natm, bas, nbas, env);
                return cache_size;
        }
        double *stack = NULL;
        if (cache == NULL) {
                int cache_size = ECPscalar_cache_size(comp*2, shls,
                                                      atm, natm, bas, nbas, env);
                stack = malloc(sizeof(double) * cache_size);
                cache = stack;
        }

        int ngcart = dij * comp;
        double *buf;
        MALLOC_INSTACK(buf, ngcart);
        NPdset0(buf, ngcart);
        int has_value;
        has_value = intor_cart(buf, shls, ecpbas, necpbas,
                               atm, natm, bas, nbas, env, opt, cache);
        if (has_value) {
                ECPscalar_distribute(out, buf, dims, comp, di, dj);
        } else {
                ECPscalar_distribute0(out, dims, comp, di, dj);
        }

        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

int ECPscalar_iprinv_cart(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        if (out == NULL) {
                int cache_size = _cart_factory(_deriv1_cart, out, 3,
                                               dims, shls, NULL, necpbas,
                                               atm, natm, bas, nbas, env, opt, cache);
                cache_size += necpbas * BAS_SLOTS;
                return cache_size;
        } else {
                int atm_id = (int)env[AS_RINV_ORIG_ATOM];
                int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
                int shl_id = _one_shell_ecpbas(&necpbas, atm_id, atm, natm, bas, nbas, env);
                if (shl_id < 0) {
                        return 0;
                }
                ecpbas += shl_id * BAS_SLOTS;
                ECPOpt opt1;
                if (opt != NULL) {
                        // iprinv requires potential on a specific atom.
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = opt->u_ecp + shl_id * (1 << LEVEL_MAX);
                        opt = &opt1;
                }
                int has_value = _cart_factory(_deriv1_cart, out, 3,
                                              dims, shls, ecpbas, necpbas,
                                              atm, natm, bas, nbas, env, opt, cache);
                return has_value;
        }
}

int ECPscalar_ipnuc_cart(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _cart_factory(_deriv1_cart, out, 3,
                                      dims, shls, ecpbas, necpbas,
                                      atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

static int _sph_factory(Function_cart intor_cart, double *out, int comp,
                        int *dims, int *shls, int *ecpbas, int necpbas,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        ECPOpt *opt, double *cache)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int di = (li*2+1) * bas[NCTR_OF+ish*BAS_SLOTS];
        const int dj = (lj*2+1) * bas[NCTR_OF+jsh*BAS_SLOTS];
        const int dij = di * dj;

        if (out == NULL) {
                int cache_size = ECPscalar_cache_size(comp*2+2, shls,
                                                      atm, natm, bas, nbas, env);
                return cache_size;
        }
        double *stack = NULL;
        if (cache == NULL) {
                int cache_size = ECPscalar_cache_size(comp*2+2, shls,
                                                      atm, natm, bas, nbas, env);
                stack = malloc(sizeof(double) * cache_size);
                cache = stack;
        }

        double *buf = cache;
        cache += dij * comp;
        int has_value = 0;
        has_value = ECPscalar_c2s_factory(intor_cart, buf, comp, shls, ecpbas, necpbas,
                                          atm, natm, bas, nbas, env, opt, cache);
        if (has_value) {
                ECPscalar_distribute(out, buf, dims, comp, di, dj);
        } else {
                ECPscalar_distribute0(out, dims, comp, di, dj);
        }

        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

int ECPscalar_iprinv_sph(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        if (out == NULL) {
                int cache_size = _sph_factory(_deriv1_cart, out, 3,
                                              dims, shls, NULL, necpbas,
                                              atm, natm, bas, nbas, env, opt, cache);
                cache_size += necpbas * BAS_SLOTS;
                return cache_size;
        } else {
                int atm_id = (int)env[AS_RINV_ORIG_ATOM];
                int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
                int shl_id = _one_shell_ecpbas(&necpbas, atm_id, atm, natm, bas, nbas, env);
                if (shl_id < 0) {
                        return 0;
                }
                ecpbas += shl_id * BAS_SLOTS;
                ECPOpt opt1;
                if (opt != NULL) {
                        // iprinv requires potential on a specific atom.
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = opt->u_ecp + shl_id * (1 << LEVEL_MAX);
                        opt = &opt1;
                }
                int has_value = _sph_factory(_deriv1_cart, out, 3,
                                             dims, shls, ecpbas, necpbas,
                                             atm, natm, bas, nbas, env, opt, cache);
                return has_value;
        }
}

int ECPscalar_ipnuc_sph(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _sph_factory(_deriv1_cart, out, 3,
                                     dims, shls, ecpbas, necpbas,
                                     atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

static int _ipipv_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       ECPOpt *opt, double *cache)
{
        if (necpbas == 0) {
                return 0;
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const int nfi_1 = li * (li-1) / 2;
        const int nfi1 = (li+2) * (li+3) / 2;
        const int nfi2 = (li+3) * (li+4) / 2;
        const int nff = nfi * nfj;
        const int di = nfi * nci;
        const int dj = nfj * ncj;
        const int dij = di * dj;
        const double *expi = env + bas[PTR_EXP+ish*BAS_SLOTS];
        const double *expj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
        const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
        const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
        int nfakbas = npi + npj;
        int *fakbas;
        MALLOC_INSTACK(fakbas, (npi+npj) * BAS_SLOTS);
        _uncontract_bas(fakbas, shls, atm, natm, bas, nbas, env);
        double *buf1;
        MALLOC_INSTACK(buf1, (nfi2*nfj + nfi1*nfj*3 + nfi*nfj*9));
        double *buf = buf1 + nfi2*nfj;
        double *gprim = buf + nfi1*nfj * 3;

        int has_value = 0;
        int shls1[2];
        double fac;

        int i, j, k, ip, jp, ic, jc, n;
        for (i = 0; i < dij*9; i++) {
                gctr[i] = 0;
        }

        for (jp = 0; jp < npj; jp++) {
        for (ip = 0; ip < npi; ip++) {
                shls1[0] = ip;
                shls1[1] = npi + jp;
                fac = 1. / (expi[ip] * expj[jp]);
                fakbas[ip*BAS_SLOTS+ANG_OF] = li + 2;
                NPdset0(buf1, nfi2*nfj);
                has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                _l_down(buf, buf1, fac, expi[ip], li+1, nfj);

                fakbas[ip*BAS_SLOTS+ANG_OF] = li;
                NPdset0(buf1, nfi*nfj);
                has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                _l_up(buf, buf1, fac, li+1, nfj);
                _l_down(gprim, buf, 1., expi[ip], li, nfj*3);

                if (li > 0) {
                        _l_down(buf, buf1, fac, expi[ip], li-1, nfj);

                        if (li > 1) {
                                fakbas[ip*BAS_SLOTS+ANG_OF] = li - 2;
                                NPdset0(buf1, nfi_1*nfj);
                                has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                           fakbas, nfakbas, env, opt, cache) | has_value);
                                has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                           fakbas, nfakbas, env, opt, cache) | has_value);
                                _l_up(buf, buf1, fac, li-1, nfj);
                        }
                        _l_up(gprim, buf, 1., li, nfj*3);
                }

                for (jc = 0; jc < ncj; jc++) {
                for (ic = 0; ic < nci; ic++) {
                        fac = ci[ic*npi+ip] * cj[jc*npj+jp];
                        n = jc*nfj*di + ic*nfi;
                        for (k = 0; k < 9; k++) {
                        for (j = 0; j < nfj; j++) {
                        for (i = 0; i < nfi; i++) {
                                gctr[k*dij+n+j*di+i] += fac * gprim[k*nff+j*nfi+i];
                        } } }
                } }
        } }

        return has_value;
}

int ECPscalar_ipiprinv_cart(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        if (out == NULL) {
                int cache_size = _cart_factory(_ipipv_cart, out, 9,
                                               dims, shls, NULL, necpbas,
                                               atm, natm, bas, nbas, env, opt, cache);
                cache_size += necpbas * BAS_SLOTS;
                return cache_size;
        } else {
                int atm_id = (int)env[AS_RINV_ORIG_ATOM];
                int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
                int shl_id = _one_shell_ecpbas(&necpbas, atm_id, atm, natm, bas, nbas, env);
                if (shl_id < 0) {
                        return 0;
                }
                ecpbas += shl_id * BAS_SLOTS;
                ECPOpt opt1;
                if (opt != NULL) {
                        // iprinv requires potential on a specific atom.
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = opt->u_ecp + shl_id * (1 << LEVEL_MAX);
                        opt = &opt1;
                }
                int has_value = _cart_factory(_ipipv_cart, out, 9,
                                              dims, shls, ecpbas, necpbas,
                                              atm, natm, bas, nbas, env, opt, cache);
                return has_value;
        }
}

int ECPscalar_ipipnuc_cart(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _cart_factory(_ipipv_cart, out, 9,
                                      dims, shls, ecpbas, necpbas,
                                      atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

int ECPscalar_ipiprinv_sph(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        if (out == NULL) {
                int cache_size = _sph_factory(_ipipv_cart, out, 9,
                                              dims, shls, NULL, necpbas,
                                              atm, natm, bas, nbas, env, opt, cache);
                cache_size += necpbas * BAS_SLOTS;
                return cache_size;
        } else {
                int atm_id = (int)env[AS_RINV_ORIG_ATOM];
                int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
                int shl_id = _one_shell_ecpbas(&necpbas, atm_id, atm, natm, bas, nbas, env);
                if (shl_id < 0) {
                        return 0;
                }
                ecpbas += shl_id * BAS_SLOTS;
                ECPOpt opt1;
                if (opt != NULL) {
                        // iprinv requires potential on a specific atom.
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = opt->u_ecp + shl_id * (1 << LEVEL_MAX);
                        opt = &opt1;
                }
                int has_value = _sph_factory(_ipipv_cart, out, 9,
                                             dims, shls, ecpbas, necpbas,
                                             atm, natm, bas, nbas, env, opt, cache);
                return has_value;
        }
}

int ECPscalar_ipipnuc_sph(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _sph_factory(_ipipv_cart, out, 9,
                                     dims, shls, ecpbas, necpbas,
                                     atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

static int _ipvip_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       ECPOpt *opt, double *cache)
{
        if (necpbas == 0) {
                return 0;
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const int nfi0 = li * (li+1) / 2;
        const int nfj0 = lj * (lj+1) / 2;
        const int nfi1 = (li+2) * (li+3) / 2;
        const int nfj1 = (lj+2) * (lj+3) / 2;
        const int nff = nfi * nfj;
        const int di = nfi * nci;
        const int dj = nfj * ncj;
        const int dij = di * dj;
        const double *expi = env + bas[PTR_EXP+ish*BAS_SLOTS];
        const double *expj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
        const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
        const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
        int nfakbas = npi + npj;
        int *fakbas;
        MALLOC_INSTACK(fakbas, (npi+npj) * BAS_SLOTS);
        _uncontract_bas(fakbas, shls, atm, natm, bas, nbas, env);
        double *buf1;
        MALLOC_INSTACK(buf1, (nfi1*nfj1 + nfi*nfj1*3 + nfi*nfj*9));
        double *buf = buf1 + nfi1*nfj1;
        double *gprim = buf + nfi*nfj1 * 3;
        double *pg, *pbuf;

        int has_value = 0;
        int shls1[2];
        double fac, xfac, yfac, zfac;

        int i, j, k, ip, jp, ic, jc, n;
        for (i = 0; i < dij*9; i++) {
                gctr[i] = 0;
        }

        for (jp = 0; jp < npj; jp++) {
        for (ip = 0; ip < npi; ip++) {
                shls1[0] = ip;
                shls1[1] = npi + jp;
                fac = 1. / (expi[ip] * expj[jp]);
                fakbas[(npi+jp)*BAS_SLOTS+ANG_OF] = lj + 1;
                fakbas[ip*BAS_SLOTS+ANG_OF] = li + 1;
                NPdset0(buf1, nfi1*nfj1);
                has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                           fakbas, nfakbas, env, opt, cache) | has_value);
                _l_down(buf, buf1, fac, expi[ip], li, nfj1);

                if (li > 0) {
                        fakbas[ip*BAS_SLOTS+ANG_OF] = li - 1;
                        NPdset0(buf1, nfi0*nfj1);
                        has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                   fakbas, nfakbas, env, opt, cache) | has_value);
                        has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                   fakbas, nfakbas, env, opt, cache) | has_value);
                        _l_up(buf, buf1, fac, li, nfj1);
                }
                if (lj == 0) {
                        fac = -2./sqrt(3.) * expj[jp];
                } else if (lj == 1) {
                        fac = -2.*0.488602511902919921 * expj[jp];
                } else {
                        fac = -2. * expj[jp];
                }
                for (k = 0; k < 3; k++) {
                        pg = gprim + k * nff * 3;
                        pbuf = buf + k * nfi*nfj1;
                        for (j = 0; j < nfj; j++) {
                        for (i = 0; i < nfi; i++) {
                                pg[      j*nfi+i] = fac * pbuf[        j *nfi+i];
                                pg[  nff+j*nfi+i] = fac * pbuf[_y_addr[j]*nfi+i];
                                pg[2*nff+j*nfi+i] = fac * pbuf[_z_addr[j]*nfi+i];
                        } }
                }

                if (lj > 0) {
                        fac = 1. / (expi[ip] * expj[jp]);
                        fakbas[(npi+jp)*BAS_SLOTS+ANG_OF] = lj - 1;
                        fakbas[ip*BAS_SLOTS+ANG_OF] = li + 1;
                        NPdset0(buf1, nfi1*nfj0);
                        has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                   fakbas, nfakbas, env, opt, cache) | has_value);
                        has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                   fakbas, nfakbas, env, opt, cache) | has_value);
                        _l_down(buf, buf1, fac, expi[ip], li, nfj0);

                        if (li > 0) {
                                fakbas[ip*BAS_SLOTS+ANG_OF] = li - 1;
                                NPdset0(buf1, nfi0*nfj0);
                                has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                           fakbas, nfakbas, env, opt, cache) | has_value);
                                has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                           fakbas, nfakbas, env, opt, cache) | has_value);
                                _l_up(buf, buf1, fac, li, nfj0);
                        }
                        if (lj == 1) {
                                fac = sqrt(3.);
                        } else if (lj == 2) {
                                fac = 1./0.488602511902919921;
                        } else {
                                fac = 1;
                        }
                        for (k = 0; k < 3; k++) {
                                pg = gprim + k * nff * 3;
                                pbuf = buf + k * nfi*nfj0;
                                for (j = 0; j < nfj0; j++) {
                                        yfac = fac * (_cart_pow_y[j] + 1);
                                        zfac = fac * (_cart_pow_z[j] + 1);
                                        xfac = fac * (lj-1 - _cart_pow_y[j] - _cart_pow_z[j] + 1);
                                        for (i = 0; i < nfi; i++) {
                                                pg[              j *nfi+i] += xfac * pbuf[j*nfi+i];
                                                pg[  nff+_y_addr[j]*nfi+i] += yfac * pbuf[j*nfi+i];
                                                pg[2*nff+_z_addr[j]*nfi+i] += zfac * pbuf[j*nfi+i];
                                        }
                                }
                        }
                }

                for (jc = 0; jc < ncj; jc++) {
                for (ic = 0; ic < nci; ic++) {
                        fac = ci[ic*npi+ip] * cj[jc*npj+jp];
                        n = jc*nfj*di + ic*nfi;
                        for (k = 0; k < 9; k++) {
                        for (j = 0; j < nfj; j++) {
                        for (i = 0; i < nfi; i++) {
                                gctr[k*dij+n+j*di+i] += fac * gprim[k*nff+j*nfi+i];
                        } } }
                } }
        } }

        return has_value;
}

int ECPscalar_iprinvip_cart(double *out, int *dims, int *shls, int *atm, int natm,
                            int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        if (out == NULL) {
                int cache_size = _cart_factory(_ipvip_cart, out, 9,
                                               dims, shls, NULL, necpbas,
                                               atm, natm, bas, nbas, env, opt, cache);
                cache_size += necpbas * BAS_SLOTS;
                return cache_size;
        } else {
                int atm_id = (int)env[AS_RINV_ORIG_ATOM];
                int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
                int shl_id = _one_shell_ecpbas(&necpbas, atm_id, atm, natm, bas, nbas, env);
                if (shl_id < 0) {
                        return 0;
                }
                ecpbas += shl_id * BAS_SLOTS;
                ECPOpt opt1;
                if (opt != NULL) {
                        // iprinv requires potential on a specific atom.
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = opt->u_ecp + shl_id * (1 << LEVEL_MAX);
                        opt = &opt1;
                }
                int has_value = _cart_factory(_ipvip_cart, out, 9,
                                              dims, shls, ecpbas, necpbas,
                                              atm, natm, bas, nbas, env, opt, cache);
                return has_value;
        }
}

int ECPscalar_ipnucip_cart(double *out, int *dims, int *shls, int *atm, int natm,
                           int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _cart_factory(_ipvip_cart, out, 9,
                                      dims, shls, ecpbas, necpbas,
                                      atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

int ECPscalar_iprinvip_sph(double *out, int *dims, int *shls, int *atm, int natm,
                           int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        if (out == NULL) {
                int cache_size = _sph_factory(_ipvip_cart, out, 9,
                                              dims, shls, NULL, necpbas,
                                              atm, natm, bas, nbas, env, opt, cache);
                cache_size += necpbas * BAS_SLOTS;
                return cache_size;
        } else {
                int atm_id = (int)env[AS_RINV_ORIG_ATOM];
                int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
                int shl_id = _one_shell_ecpbas(&necpbas, atm_id, atm, natm, bas, nbas, env);
                if (shl_id < 0) {
                        return 0;
                }
                ecpbas += shl_id * BAS_SLOTS;
                ECPOpt opt1;
                if (opt != NULL) {
                        // iprinv requires potential on a specific atom.
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = opt->u_ecp + shl_id * (1 << LEVEL_MAX);
                        opt = &opt1;
                }
                int has_value = _sph_factory(_ipvip_cart, out, 9,
                                             dims, shls, ecpbas, necpbas,
                                             atm, natm, bas, nbas, env, opt, cache);
                return has_value;
        }
}

int ECPscalar_ipnucip_sph(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _sph_factory(_ipvip_cart, out, 9,
                                     dims, shls, ecpbas, necpbas,
                                     atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

static int _igv_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                     int *atm, int natm, int *bas, int nbas, double *env,
                     ECPOpt *opt, double *cache)
{
        if (necpbas == 0) {
                return 0;
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
        const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int nfi = (li+1) * (li+2) / 2;
        const int nfj = (lj+1) * (lj+2) / 2;
        const int nfi1 = (li+2) * (li+3) / 2;
        const int di = nfi * nci;
        const int dj = nfj * ncj;
        const int dij = di * dj;
        int ngcart = nfi1*nci*nfj*ncj;
        const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        double *buf1;
        MALLOC_INSTACK(buf1, ngcart * 2);
        double *buf2 = buf1 + ngcart;

        int i, j;
        double *gctrx = gctr;
        double *gctry = gctrx + dij;
        double *gctrz = gctry + dij;
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];

        int fakbas[2*BAS_SLOTS];
        for (i = 0; i < BAS_SLOTS; i++) {
                fakbas[          i] = bas[ish*BAS_SLOTS+i];
                fakbas[BAS_SLOTS+i] = bas[jsh*BAS_SLOTS+i];
        }
        int has_value = 0;
        int shls1[2] = {0, 1};
        double fac, vx, vy, vz;

        fakbas[ANG_OF] = li + 1;
        NPdset0(buf1, ngcart);
        has_value = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                   fakbas, 2, env, opt, cache) | has_value);
        has_value = (ECPtype2_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                   fakbas, 2, env, opt, cache) | has_value);
        NPdset0(buf2, dij);
        has_value = (ECPtype1_cart(buf2, shls, ecpbas, necpbas, atm, natm,
                                   bas, nbas, env, opt, cache) | has_value);
        has_value = (ECPtype2_cart(buf2, shls, ecpbas, necpbas, atm, natm,
                                   bas, nbas, env, opt, cache) | has_value);
        if (!has_value) {
                return has_value;
        }

        if (li == 0) {
                fac = 1./sqrt(3.);
        } else if (li == 1) {
                fac = 0.488602511902919921;
        } else {
                fac = 1.;
        }
        for (j = 0; j < nci*nfj*ncj; j++) {
                for (i = 0; i < nfi; i++) {
                        vx = fac * buf1[j*nfi1+        i ] + ri[0] * buf2[j*nfi+i];
                        vy = fac * buf1[j*nfi1+_y_addr[i]] + ri[1] * buf2[j*nfi+i];
                        vz = fac * buf1[j*nfi1+_z_addr[i]] + ri[2] * buf2[j*nfi+i];
                        gctrx[j*nfi+i] = -.5 * (rirj[1] * vz - rirj[2] * vy);
                        gctry[j*nfi+i] = -.5 * (rirj[2] * vx - rirj[0] * vz);
                        gctrz[j*nfi+i] = -.5 * (rirj[0] * vy - rirj[1] * vx);
                }
        }
        return has_value;
}

int ECPscalar_ignuc_cart(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int comp = 3;
        int has_value = _cart_factory(_igv_cart, out, comp,
                                      dims, shls, ecpbas, necpbas,
                                      atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

int ECPscalar_ignuc_sph(double *out, int *dims, int *shls, int *atm, int natm,
                        int *bas, int nbas, double *env, ECPOpt *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int comp = 3;
        int has_value = _sph_factory(_igv_cart, out, comp,
                                     dims, shls, ecpbas, necpbas,
                                     atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

void ECPscalar_optimizer(ECPOpt **opt, int *atm, int natm, int *bas, int nbas, double *env);
#define make_optimizer(fname) \
void ECPscalar_##fname##_optimizer(ECPOpt **opt, int *atm, int natm, \
                                   int *bas, int nbas, double *env) \
{ \
        ECPscalar_optimizer(opt, atm, natm, bas, nbas, env); \
}
#define make_empty_optimizer(fname) \
void ECPscalar_##fname##_optimizer(ECPOpt **opt, int *atm, int natm, \
                                   int *bas, int nbas, double *env) \
{ \
        *opt = NULL; \
}
make_optimizer(ignuc)
make_optimizer(ipnuc)
make_optimizer(ipipnuc)
make_optimizer(ipnucip)
make_empty_optimizer(iprinv)
make_empty_optimizer(ipiprinv)
make_empty_optimizer(iprinvip)
