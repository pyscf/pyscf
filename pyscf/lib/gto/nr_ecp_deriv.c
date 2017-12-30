/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cint.h"
#include "vhf/fblas.h"

#define CART_MAX        128 // ~ lmax = 14
#define SIM_ZERO        1e-50
#define EXPCUTOFF       39   // 1e-17
#define CUTOFF          460  // ~ 1e200
#define CLOSE_ENOUGH(x, y)      (fabs(x-y) < 1e-10*fabs(y) || fabs(x-y) < 1e-10)
#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
#define CART_CUM        (455+1) // upto l = 12

// Held in env, to get *ecpbas, necpbas
#define AS_RINV_ORIG_ATOM       17
#define AS_ECPBAS_OFFSET        18
#define AS_NECPBAS              19

int ECPtype1_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                  int *atm, int natm, int *bas, int nbas, double *env, double *cache);
int ECPtype2_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                  int *atm, int natm, int *bas, int nbas, double *env, double *cache);
int ECPscalar_c2s_factory(double *gctr, int comp, int *shls,
                          int *ecpbas, int necpbas,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          double *cache, int (*fcart)());
void ECPscalar_distribute(double *out, double *gctr, const int *dims,
                          const int comp, const int di, const int dj);

static int _x_addr[] = {
  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
 15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
 30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
 45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
 60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
 75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
 90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
};
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

static int _one_shell_ecpbas(int *ecpbas, int atm_id,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        int *all_ecp = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int necpbas = (int)env[AS_NECPBAS];
        int i, j;
        int n = 0;
        for (i = 0; i < necpbas; i++) {
                if (atm_id == all_ecp[ATOM_OF+i*BAS_SLOTS]) {
                        for (j = 0; j < BAS_SLOTS; j++) {
                                ecpbas[n*BAS_SLOTS+j] = all_ecp[i*BAS_SLOTS+j];
                        }
                        n += 1;
                }
        }
        return n;
}

static void _uncontract_bas(int *fakbas, int *shls,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
        const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
        int i, j;
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

static int _deriv1_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                        int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
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
        const int comp = 3;
        const double *expi = env + bas[PTR_EXP+ish*BAS_SLOTS];
        const double *expj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
        const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
        const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
        int nfakbas = npi + npj;
        int *fakbas = malloc(sizeof(int) * (npi+npj) * BAS_SLOTS);
        _uncontract_bas(fakbas, shls, atm, natm, bas, nbas, env);
        double *buf1 = malloc(sizeof(double) * (nfi1*nfj * 2 + nfi*nfj*comp));
        double *buf2 = buf1 + nfi1*nfj;
        double *gprim = buf2 + nfi1*nfj;
        double *gpx = gprim;
        double *gpy = gpx + nfi*nfj;
        double *gpz = gpy + nfi*nfj;

        int has_value = 0;
        int has_value1 = 0;
        int shls1[2];
        double fac;

        int i, j, ip, jp, ic, jc, n, lx, ly, lz;
        for (i = 0; i < dij*comp; i++) {
                gctr[i] = 0;
        }
        double *gctrx = gctr;
        double *gctry = gctrx + dij;
        double *gctrz = gctry + dij;

        for (jp = 0; jp < npj; jp++) {
        for (ip = 0; ip < npi; ip++) {
                shls1[0] = ip;
                shls1[1] = npi + jp;
                fakbas[ip*BAS_SLOTS+ANG_OF] = li + 1;
                has_value1 = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                            fakbas, nfakbas, env, cache) |
                              ECPtype2_cart(buf2, shls1, ecpbas, necpbas, atm, natm,
                                            fakbas, nfakbas, env, cache));

                has_value = has_value | has_value1;
                if (li == 0) {
/* divide (expi[ip] * expj[jp]) because the exponents were used as normalization
 * coefficients for primitive GTOs in function _uncontract_bas */
                        fac = -2./sqrt(3.) * expi[ip] / expi[ip] / expj[jp];
                } else if (li == 1) {
                        fac = -2.*0.488602511902919921 * expi[ip] / expi[ip] / expj[jp];
                } else {
                        fac = -2. * expi[ip] / expi[ip] / expj[jp];
                }
                for (i = 0; i < nfi1 * nfj * comp; i++) {
                        buf1[i] = fac * (buf1[i] + buf2[i]);
                }
                for (j = 0; j < nfj; j++) {
                        for (i = 0; i < nfi; i++) {
                                gpx[j*nfi+i] = buf1[j*nfi1+        i ];
                                gpy[j*nfi+i] = buf1[j*nfi1+_y_addr[i]];
                                gpz[j*nfi+i] = buf1[j*nfi1+_z_addr[i]];
                        }
                }

                if (li > 0) {
                        fakbas[ip*BAS_SLOTS+ANG_OF] = li - 1;
                        has_value1 = (ECPtype1_cart(buf1, shls1, ecpbas, necpbas, atm, natm,
                                                    fakbas, nfakbas, env, cache) |
                                      ECPtype2_cart(buf2, shls1, ecpbas, necpbas, atm, natm,
                                                    fakbas, nfakbas, env, cache));
                        has_value = has_value | has_value1;

                        if (li == 1) {
                                fac = sqrt(3.) / (expi[ip] * expj[jp]);
                        } else if (li == 2) {
                                fac = 1./(0.488602511902919921 * expi[ip] * expj[jp]);
                        } else {
                                fac = 1. / (expi[ip] * expj[jp]);
                        }
                        for (i = 0; i < nfi0*nfj; i++) {
                                buf1[i] = fac * (buf1[i] + buf2[i]);
                        }
                        for (j = 0; j < nfj; j++) {
                        for (i = 0; i < nfi0; i++) {
                                ly = _cart_pow_y[i] + 1;
                                lz = _cart_pow_z[i] + 1;
                                lx = li - _cart_pow_y[i] - _cart_pow_z[i] + 1;
                                gpx[j*nfi+        i ] += lx * buf1[j*nfi0+i];
                                gpy[j*nfi+_y_addr[i]] += ly * buf1[j*nfi0+i];
                                gpz[j*nfi+_z_addr[i]] += lz * buf1[j*nfi0+i];
                        } }
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

        free(fakbas);
        free(buf1);
        return has_value;
}

static int _ecp_ipv_cart(double *out, int *dims, int *shls, int *ecpbas, int necpbas,
                         int *atm, int natm, int *bas, int nbas, double *env,
                         void *opt, double *cache)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int di = (li+1) * (li+2) / 2 * bas[NCTR_OF+ish*BAS_SLOTS];;
        const int dj = (lj+1) * (lj+2) / 2 * bas[NCTR_OF+jsh*BAS_SLOTS];
        const int dij = di * dj;
        const int comp = 3;

        if (out == NULL) {
                return dij * 2 * comp;
        }
        double *stack = NULL;
        if (cache == NULL) {
                stack = malloc(sizeof(double) * dij * 2 * comp);
                cache = stack;
        }

        double *buf = cache;
        cache += dij;
        int has_value;
        has_value = _deriv1_cart(buf, shls, ecpbas, necpbas,
                                 atm, natm, bas, nbas, env, cache);
        ECPscalar_distribute(out, buf, dims, comp, di, dj);

        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

int ECPscalar_iprinv_cart(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, void *opt, double *cache)
{
        int atm_id = (int)env[AS_RINV_ORIG_ATOM];
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = malloc(sizeof(int) * necpbas * BAS_SLOTS);
        necpbas = _one_shell_ecpbas(ecpbas, atm_id, atm, natm, bas, nbas, env);
        int has_value = _ecp_ipv_cart(out, dims, shls, ecpbas, necpbas,
                                      atm, natm, bas, nbas, env, opt, cache);
        free(ecpbas);
        return has_value;
}

int ECPscalar_ipnuc_cart(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, void *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _ecp_ipv_cart(out, dims, shls, ecpbas, necpbas,
                                      atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}

static int _ecp_ipv_sph(double *out, int *dims, int *shls, int *ecpbas, int necpbas,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        void *opt, double *cache)
{
        const int ish = shls[0];
        const int jsh = shls[1];
        const int li = bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = bas[ANG_OF+jsh*BAS_SLOTS];
        const int di = (li*2+1) * bas[NCTR_OF+ish*BAS_SLOTS];
        const int dj = (lj*2+1) * bas[NCTR_OF+jsh*BAS_SLOTS];
        const int dij = di * dj;
        const int comp = 3;

        if (out == NULL) {
                const int nfi = (li+1) * (li+2) / 2;
                const int nfj = (lj+1) * (lj+2) / 2;
                const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
                const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
                return dij*2*comp + nfi*nfj*nci*ncj*2;
        }
        double *stack = NULL;
        if (cache == NULL) {
                const int nfi = (li+1) * (li+2) / 2;
                const int nfj = (lj+1) * (lj+2) / 2;
                const int nci = bas[NCTR_OF+ish*BAS_SLOTS];
                const int ncj = bas[NCTR_OF+jsh*BAS_SLOTS];
                stack = malloc(sizeof(double) * (dij*2*comp+nfi*nfj*nci*ncj*2));
                cache = stack;
        }

        double *buf = cache;
        cache += dij;
        int has_value;
        has_value = ECPscalar_c2s_factory(buf, 3, shls, ecpbas, necpbas,
                                          atm, natm, bas, nbas, env, cache, _deriv1_cart);
        ECPscalar_distribute(out, buf, dims, comp, di, dj);

        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

int ECPscalar_iprinv_sph(double *out, int *dims, int *shls, int *atm, int natm,
                          int *bas, int nbas, double *env, void *opt, double *cache)
{
        int atm_id = (int)env[AS_RINV_ORIG_ATOM];
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = malloc(sizeof(int) * necpbas * BAS_SLOTS);
        necpbas = _one_shell_ecpbas(ecpbas, atm_id, atm, natm, bas, nbas, env);
        int has_value = _ecp_ipv_sph(out, dims, shls, ecpbas, necpbas,
                                     atm, natm, bas, nbas, env, opt, cache);
        free(ecpbas);
        return has_value;
}

int ECPscalar_ipnuc_sph(double *out, int *dims, int *shls, int *atm, int natm,
                         int *bas, int nbas, double *env, void *opt, double *cache)
{
        int necpbas = (int)env[AS_NECPBAS];
        int *ecpbas = bas + ((int)env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int has_value = _ecp_ipv_sph(out, dims, shls, ecpbas, necpbas,
                                     atm, natm, bas, nbas, env, opt, cache);
        return has_value;
}
