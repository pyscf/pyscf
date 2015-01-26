/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <xc.h>
#include "cint.h"
#include "vhf/fblas.h"
#include "vxc.h"
#include "grid_basis.h"
#include <assert.h>

#define MIN(X,Y)        ((X)>(Y)?(Y):(X))
#define BOX_SIZE        56

struct _VXCEnvs {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;

        int num_grids;
        int nao;
        double *coords;
        double *weights;
        xc_func_type *func_x;
        xc_func_type *func_c;
};

double nr_accumlator(double (*fgrid)(), struct _VXCEnvs *envs,
                     double *e, double *v, const double *dm, const int stride)
{
        const int INC1 = 1;
        const double D1 = 1;
        int id;
        double nelec = 0;
        double nelec_priv, e_priv;
        double *v_priv;
        const int nn = envs->nao * envs->nao;
        *e = 0;
        memset(v, 0, sizeof(double)*nn);
#pragma omp parallel default(none) \
        shared(nelec, envs, e, v, dm, fgrid) \
        private(id, nelec_priv, e_priv, v_priv)
        {
                nelec_priv = 0;
                e_priv = 0;
                v_priv = malloc(sizeof(double)*nn);
                memset(v_priv, 0, sizeof(double)*nn);
#pragma omp for nowait schedule(dynamic, 1)
                for (id = 0; id < envs->num_grids; id+=stride) {
                        nelec_priv += (*fgrid)(id, MIN(envs->num_grids-id,stride),
                                               envs, &e_priv, v_priv, dm);
                }
#pragma omp critical
                {
                        nelec += nelec_priv;
                        *e += e_priv;
                        daxpy_(&nn, &D1, v_priv, &INC1, v, &INC1);
                }
                free(v_priv);
        }
        return nelec;
}

/* mat is n x n matrix */
static void _dot_vmv(int n, int nv, double *v1, double *mat, double *v2,
                     double *res)
{
        const int INC1 = 1;
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        double *mv = malloc(sizeof(double) * n*nv);
        int i;
        dgemm_(&TRANS_N, &TRANS_N, &n, &nv, &n,
               &D1, mat, &n, v2, &n, &D0, mv, &n);
        for (i = 0; i < nv; i++) {
                res[i] = ddot_(&n, v1+n*i, &INC1, mv+n*i, &INC1);
        }
        free(mv);
}

/* v1 is n x nv matrix, m1 is n x n matrix, m2 is n x nv x nvs matrix
 * v is nv x nvs matrix
 * v[k,l] = v1[i,k] * m1[i,j] * m2[j,k,l] */
static void _dot_vmm(int n, int nv, int nvs,
                     double *v1, double *m1, double *m2, double *v)
{
        const int INC1 = 1;
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        double *vm = malloc(sizeof(double) * n*nv);
        int i;
        int nnv = n * nv;
        dgemm_(&TRANS_T, &TRANS_N, &n, &nv, &n,
               &D1, m1, &n, v1, &n, &D0, vm, &n);
        for (i = 0; i < nv; i++) {
                dgemv_(&TRANS_T, &n, &nvs, &D1, m2+n*i, &nnv,
                       vm+n*i, &INC1, &D0, v+i, &nv);
        }
        free(vm);
}

static int _xc_has_gga(xc_func_type *func_x, xc_func_type *func_c)
{
        /* in xc.h, XC families are defined bitwise
           #define XC_FAMILY_LDA           1
           #define XC_FAMILY_GGA           2
           #define XC_FAMILY_MGGA          4
           #define XC_FAMILY_LCA           8
           #define XC_FAMILY_OEP          16
           #define XC_FAMILY_HYB_GGA      32
           #define XC_FAMILY_HYB_MGGA     64 */
        int code = func_x->info->family & (~XC_FAMILY_LDA); // screen the LDA bit
        if (func_c->info) {
                code |= func_c->info->family & (~XC_FAMILY_LDA);
        }
        return code;
}

static void nr_rho_sf(int id, int np, struct _VXCEnvs *envs, double *dm,
                      double *ao, double *rho, double *sigma)
{
        int nao = envs->nao;

        if (_xc_has_gga(envs->func_x, envs->func_c)) {
                VXCvalue_nr_gto_grad(nao, np, ao, envs->coords+id*3,
                                     envs->atm, envs->natm,
                                     envs->bas, envs->nbas, envs->env);
                _dot_vmm(nao, np, 4, ao, dm, ao, rho);
                int i;
                double *rho1 = rho  + np;
                double *rho2 = rho1 + np;
                double *rho3 = rho2 + np;
                for (i = 0; i < np; i++) {
                        rho1[i] *= 2; // * 2 for  +c.c.
                        rho2[i] *= 2;
                        rho3[i] *= 2;
                        sigma[i] = rho1[i]*rho1[i] + rho2[i]*rho2[i]
                                 + rho3[i]*rho3[i];
                }
        } else {
                VXCvalue_nr_gto(nao, np, ao, envs->coords+id*3,
                                envs->atm, envs->natm,
                                envs->bas, envs->nbas, envs->env);
                _dot_vmv(nao, np, ao, dm, ao, rho);
        }
}

static double nr_vmat_sf(int id, int np, struct _VXCEnvs *envs, double *ao,
                         double *exc, double *rho, double *vrho,
                         double *vsigma, double *e, double *v)
{
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        int nao = envs->nao;
        int i, j;
        double weight;
        double *aow = malloc(sizeof(double)* nao*np);
        for (j = 0; j < np; j++) {
                // *.5 since dsyr2k_ x*y'+y*x'
                weight = envs->weights[id+j] * vrho[j] * .5;
                for (i = 0; i < nao; i++) {
                        aow[j*nao+i] = ao[j*nao+i] * weight;
                }
        }

        if (_xc_has_gga(envs->func_x, envs->func_c)) {
                double *ao1 = ao  + nao*np;
                double *ao2 = ao1 + nao*np;
                double *ao3 = ao2 + nao*np;
                double *rho1 = rho  + np;
                double *rho2 = rho1 + np;
                double *rho3 = rho2 + np;
                for (j = 0; j < np; j++) {
                        weight = envs->weights[id+j] * vsigma[j] * 2;
                        for (i = 0; i < nao; i++) {
                                aow[j*nao+i] += weight
                                        *(ao1[j*nao+i] * rho1[j]
                                        + ao2[j*nao+i] * rho2[j]
                                        + ao3[j*nao+i] * rho3[j]);
                        }
                }
        }
        // ao * nabla_ao + nabla_ao * ao
        dsyr2k_(&UP, &TRANS_N, &nao, &np, &D1, ao, &nao,
                aow, &nao, &D1, v, &nao);

        double nelec = 0;
        for (i = 0; i < np; i++) {
                *e += envs->weights[id+i] * rho[i] * exc[i];
                nelec += rho[i] * envs->weights[id+i];
        }
        free(aow);

        return nelec;
}


/* Extracted from comments of libxc:gga.c
 
    sigma_st       = grad rho_s . grad rho_t
    zk             = energy density per unit particle
 
    vrho_s         = d zk / d rho_s
    vsigma_st      = d n*zk / d sigma_st
    
    v2rho2_st      = d^2 n*zk / d rho_s d rho_t
    v2rhosigma_svx = d^2 n*zk / d rho_s d sigma_tv
    v2sigma2_stvx  = d^2 n*zk / d sigma_st d sigma_vx
 
 if nspin == 2
    rho(2)        = (u, d)
    sigma(3)      = (uu, du, dd)
 
    vrho(2)       = (u, d)
    vsigma(3)     = (uu, du, dd)
 
    v2rho2(3)     = (uu, du, dd)
    v2rhosigma(6) = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
    v2sigma2(6)   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
 */
/* spin unpolarized */
static double nr_spin0(int id, int np, struct _VXCEnvs *envs,
                       double *e, double *v, double *dm)
{
        const int INC1 = 1;
        int nao = envs->nao;
        double rho[8*np], sigma[3*np];
        double vrho[2*np], vcrho[2*np], vsigma[3*np], vcsigma[3*np];
        double exc[np], ec[np];
        double *ao = malloc(sizeof(double)* nao*np*4);
        double nelec;
        int i;

        nr_rho_sf(id, np, envs, dm, ao, rho, sigma);
        if (dasum_(&np, rho, &INC1) < 1e-14) {
                nelec = 0;
                goto _normal_end;
        }

        switch (envs->func_x->info->family) {
        case XC_FAMILY_LDA:
                // exc is the energy density
                // note libxc have added exc/ec to vrho/vcrho
                xc_lda_exc_vxc(envs->func_x, np, rho, exc, vrho);
                memset(vsigma, 0, sizeof(double)*np);
                break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
                xc_gga_exc_vxc(envs->func_x, np, rho, sigma, exc,
                               vrho, vsigma);
                break;
        default:
                fprintf(stderr, "X functional %d '%s' is not implmented\n",
                        envs->func_x->info->number, envs->func_x->info->name);
                exit(1);
        }

        if (envs->func_x->info->kind == XC_EXCHANGE) {
                switch (envs->func_c->info->family) {
                case XC_FAMILY_LDA:
                        xc_lda_exc_vxc(envs->func_c, np, rho, ec, vcrho);
                        break;
                case XC_FAMILY_GGA:
                        xc_gga_exc_vxc(envs->func_c, np, rho, sigma, ec,
                                       vcrho, vcsigma);
                        for (i = 0; i < np; i++) {
                                vsigma[i] += vcsigma[i];
                        }
                        break;
                default:
                        fprintf(stderr, "C functional %d '%s' is not implmented\n",
                                envs->func_c->info->number,
                                envs->func_c->info->name);
                        exit(1);
                }
                for (i = 0; i < np; i++) {
                        exc[i] += ec[i];
                        vrho[i] += vcrho[i];
                }
        }

        nelec = nr_vmat_sf(id, np, envs, ao, exc, rho, vrho, vsigma, e, v);

_normal_end:
        free(ao);
        return nelec;
}


// e ~ energy,  n ~ num electrons,  v ~ XC potential matrix
double VXCnr_vxc(int x_id, int c_id, int spin, int relativity,
                 double *dm, double *exc, double *v,
                 int num_grids, double *coords, double *weights,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        xc_func_type func_x = {};
        xc_func_type func_c = {};
        VXCinit_libxc(&func_x, &func_c, x_id, c_id, spin, relativity);
        struct _VXCEnvs envs = {natm, nbas, atm, bas, env,
                                num_grids, CINTtot_cgto_spheric(bas, nbas),
                                coords, weights, &func_x, &func_c};

        double n = nr_accumlator(nr_spin0, &envs, exc, v, dm, BOX_SIZE);

        VXCdel_libxc(&func_x, &func_c);
        return n;
}
