/*
 * File: g1e.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <string.h>
#include <math.h>
#include <assert.h>
#include "cint_bas.h"
#include "misc.h"
#include "g1e.h"

#define MAX(X,Y) (X)>(Y)?(X):(Y)

int CINTinit_int3c1e_EnvVars(CINTEnvVars *envs, const int *ng, const int *shls,
                             const int *atm, const int natm,
                             const int *bas, const int nbas, const double *env)
{
        envs->natm = natm;
        envs->nbas = nbas;
        envs->atm = atm;
        envs->bas = bas;
        envs->env = env;
        envs->shls = shls;

        const int i_sh = shls[0];
        const int j_sh = shls[1];
        const int k_sh = shls[2];
        envs->i_l = bas(ANG_OF, i_sh);
        envs->j_l = bas(ANG_OF, j_sh);
        envs->k_l = bas(ANG_OF, k_sh);
        envs->i_prim = bas(NPRIM_OF, i_sh);
        envs->j_prim = bas(NPRIM_OF, j_sh);
        envs->k_prim = bas(NPRIM_OF, k_sh);
        envs->i_ctr = bas(NCTR_OF, i_sh);
        envs->j_ctr = bas(NCTR_OF, j_sh);
        envs->k_ctr = bas(NCTR_OF, k_sh);
        envs->nfi = CINTlen_cart(envs->i_l);
        envs->nfj = CINTlen_cart(envs->j_l);
        envs->nfk = CINTlen_cart(envs->k_l);
        envs->nf = envs->nfi * envs->nfj * envs->nfk;

        envs->ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
        envs->rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
        envs->rk = env + atm(PTR_COORD, bas(ATOM_OF, k_sh));

        envs->gbits = ng[GSHIFT];
        envs->ncomp_e1 = ng[POS_E1];
        envs->ncomp_tensor = ng[TENSOR];

        envs->li_ceil = envs->i_l + ng[IINC];
        envs->lj_ceil = envs->j_l + ng[JINC];
        envs->lk_ceil = envs->k_l + ng[KINC];

        envs->common_factor = SQRTPI * M_PI
                * CINTcommon_fac_sp(envs->i_l) * CINTcommon_fac_sp(envs->j_l)
                * CINTcommon_fac_sp(envs->k_l);

        int dli = envs->li_ceil + 1;
        int dlj = envs->lj_ceil + envs->lk_ceil + 1;
        int dlk = envs->lk_ceil + 1;
        envs->g_stride_i = 1;
        envs->g_stride_j = dli;
        envs->g_stride_k = dli * dlj;
        int nmax = envs->li_ceil + dlj;
        envs->g_size     = MAX(dli*dlj*dlk, dli*nmax);

        envs->rirj[0] = envs->ri[0] - envs->rj[0];
        envs->rirj[1] = envs->ri[1] - envs->rj[1];
        envs->rirj[2] = envs->ri[2] - envs->rj[2];
        return 0;
}

void CINTg3c1e_index_xyz(int *idx, const CINTEnvVars *envs)
{
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int k_l = envs->k_l;
        const int nfi = envs->nfi;
        const int nfj = envs->nfj;
        const int nfk = envs->nfk;
        const int dj = envs->g_stride_j;
        const int dk = envs->g_stride_k;
        int i, j, k, n;
        int ofx, ofjx, ofkx;
        int ofy, ofjy, ofky;
        int ofz, ofjz, ofkz;
        int i_nx[CART_MAX], i_ny[CART_MAX], i_nz[CART_MAX];
        int j_nx[CART_MAX], j_ny[CART_MAX], j_nz[CART_MAX];
        int k_nx[CART_MAX], k_ny[CART_MAX], k_nz[CART_MAX];

        CINTcart_comp(i_nx, i_ny, i_nz, i_l);
        CINTcart_comp(j_nx, j_ny, j_nz, j_l);
        CINTcart_comp(k_nx, k_ny, k_nz, k_l);

        ofx = 0;
        ofy = envs->g_size;
        ofz = envs->g_size * 2;
        n = 0;
        for (k = 0; k < nfk; k++) {
                ofkx = ofx + dk * k_nx[k];
                ofky = ofy + dk * k_ny[k];
                ofkz = ofz + dk * k_nz[k];
                for (j = 0; j < nfj; j++) {
                        ofjx = ofkx + dj * j_nx[j];
                        ofjy = ofky + dj * j_ny[j];
                        ofjz = ofkz + dj * j_nz[j];
                        for (i = 0; i < nfi; i++) {
                                idx[n+0] = ofjx + i_nx[i];
                                idx[n+1] = ofjy + i_ny[i];
                                idx[n+2] = ofjz + i_nz[i];
                                n += 3;
                        }
                }
        }
}


void CINTg3c1e_ovlp(double *g, double ai, double aj, double ak,
                    double fac, const CINTEnvVars *envs)
{
        const int li = envs->li_ceil;
        const int lj = envs->lj_ceil;
        const int lk = envs->lk_ceil;
        const int nmax = li + lj + lk;
        const int mmax = lj + lk;
        double *gx = g;
        double *gy = g + envs->g_size;
        double *gz = g + envs->g_size * 2;
        gx[0] = 1;
        gy[0] = 1;
        gz[0] = fac;
        if (nmax == 0) {
                return;
        }

        int dj = li + 1;
        const int dk = envs->g_stride_k;
        const double aijk = ai + aj + ak;
        const double aijk1 = .5 / aijk;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        const double *rk = envs->rk;
        int i, j, k, off;
        const double *rirj = envs->rirj;
        double rjrk[3], rjrijk[3];

        rjrk[0] = rj[0] - rk[0];
        rjrk[1] = rj[1] - rk[1];
        rjrk[2] = rj[2] - rk[2];

        rjrijk[0] = rj[0] - (ai * ri[0] + aj * rj[0] + ak * rk[0]) / aijk;
        rjrijk[1] = rj[1] - (ai * ri[1] + aj * rj[1] + ak * rk[1]) / aijk;
        rjrijk[2] = rj[2] - (ai * ri[2] + aj * rj[2] + ak * rk[2]) / aijk;

        gx[dj] = -rjrijk[0] * gx[0];
        gy[dj] = -rjrijk[1] * gy[0];
        gz[dj] = -rjrijk[2] * gz[0];

        double *p0x = gx + dj;
        double *p0y = gy + dj;
        double *p0z = gz + dj;
        double *p1x = gx - dj;
        double *p1y = gy - dj;
        double *p1z = gz - dj;
        double *p2x, *p2y, *p2z;
        for (j = 1; j < nmax; j++) {
                p0x[j*dj] = aijk1 * j * p1x[j*dj] - rjrijk[0] * gx[j*dj];
                p0y[j*dj] = aijk1 * j * p1y[j*dj] - rjrijk[1] * gy[j*dj];
                p0z[j*dj] = aijk1 * j * p1z[j*dj] - rjrijk[2] * gz[j*dj];
        }

        for (i = 1; i <= li; i++) {
                for (j = 0; j <= nmax-i; j++) { // upper limit lj+lk
                        gx[i+j*dj] = p0x[i-1+j*dj] - rirj[0] * gx[i-1+j*dj];
                        gy[i+j*dj] = p0y[i-1+j*dj] - rirj[1] * gy[i-1+j*dj];
                        gz[i+j*dj] = p0z[i-1+j*dj] - rirj[2] * gz[i-1+j*dj];
                }
        }

        dj = envs->g_stride_j;
        for (k = 1; k <= lk; k++) {
                for (j = 0; j <= mmax-k; j++) {
                        off = k * dk + j * dj;
                        p0x = gx + off;
                        p0y = gy + off;
                        p0z = gz + off;
                        p1x = gx + off + dj - dk;
                        p1y = gy + off + dj - dk;
                        p1z = gz + off + dj - dk;
                        p2x = gx + off - dk;
                        p2y = gy + off - dk;
                        p2z = gz + off - dk;
                        for (i = 0; i <= li; i++) {
                                p0x[i] = p1x[i] + rjrk[0] * p2x[i];
                                p0y[i] = p1y[i] + rjrk[1] * p2y[i];
                                p0z[i] = p1z[i] + rjrk[2] * p2z[i];
                        }
                }
        }
}

