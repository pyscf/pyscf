/*
 * Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include "config.h"

#if !defined HAVE_DEFINED_CINTENVVARS_H
#define HAVE_DEFINED_CINTENVVARS_H
// ref to CINTinit_int1e_EnvVars, CINTinit_int2e_EnvVars
typedef struct {
        const FINT *atm;
        const FINT *bas;
        const double *env;
        const FINT *shls;
        FINT natm;
        FINT nbas;

        FINT i_l;
        FINT j_l;
        FINT k_l;
        FINT l_l;
        FINT i_prim;
        FINT j_prim;
        FINT k_prim;
        FINT l_prim;
        FINT i_ctr;
        FINT j_ctr;
        FINT k_ctr;
        FINT l_ctr;
        FINT nfi;  // number of cartesion components
        FINT nfj;
        FINT nfk;
        FINT nfl;
        FINT nf;  // = nfi*nfj*nfk*nfl;
        FINT _padding1;
        const double *ri;
        const double *rj;
        const double *rk;
        const double *rl;
        double common_factor;

        FINT gbits;
        FINT ncomp_e1; // = 1 if spin free, = 4 when spin included, it
        FINT ncomp_e2; // corresponds to POSX,POSY,POSZ,POS1, see cint_const.h
        FINT ncomp_tensor; // e.g. = 3 for gradients

        /* values may diff based on the g0_2d4d algorithm */
        FINT li_ceil; // power of x, == i_l if nabla is involved, otherwise == i_l
        FINT lj_ceil;
        FINT lk_ceil;
        FINT ll_ceil;
        FINT g_stride_i; // nrys_roots * shift of (i++,k,l,j)
        FINT g_stride_k; // nrys_roots * shift of (i,k++,l,j)
        FINT g_stride_l; // nrys_roots * shift of (i,k,l++,j)
        FINT g_stride_j; // nrys_roots * shift of (i,k,l,j++)
        FINT nrys_roots;
        FINT g_size;  // ref to cint2e.c g = malloc(sizeof(double)*g_size)

        FINT g2d_ijmax;
        FINT g2d_klmax;
        const double *rx_in_rijrx;
        const double *rx_in_rklrx;
        double rirj[3]; // diff by an sign in different g0_2d4d algorithm
        double rkrl[3];

        void (*f_g0_2d4d)();

        /* */
        void (*f_gout)();

        /* values are assigned during calculation */
        FINT *idx;
        double ai;
        double aj;
        double ak;
        double al;
        double rij[3];
        double rijrx[3];
        double aij;
        double rkl[3];
        double rklrx[3];
        double akl;
} CINTEnvVars;
#endif

FINT CINTinit_int1e_EnvVars(CINTEnvVars *envs, const FINT *ng, const FINT *shls,
                           const FINT *atm, const FINT natm,
                           const FINT *bas, const FINT nbas, const double *env);

void CINTg1e_index_xyz(FINT *idx, const CINTEnvVars *envs);

void CINTg_ovlp(double *g, const double ai, const double aj,
                const double fac, const CINTEnvVars *envs);

void CINTg_nuc(double *g, const double aij, const double *rij,
               const double *cr, const double t2, const double fac,
               const CINTEnvVars *envs);

void CINTnabla1i_1e(double *f, const double *g,
                    const FINT li, const FINT lj, const CINTEnvVars *envs);

void CINTnabla1j_1e(double *f, const double *g,
                    const FINT li, const FINT lj, const CINTEnvVars *envs);

void CINTx1i_1e(double *f, const double *g, const double ri[3],
                const FINT li, const FINT lj, const CINTEnvVars *envs);

void CINTx1j_1e(double *f, const double *g, const double rj[3],
                const FINT li, const FINT lj, const CINTEnvVars *envs);

void CINTprim_to_ctr(double *gc, const FINT nf, const double *gp,
                     const FINT inc, const FINT nprim,
                     const FINT nctr, const double *pcoeff);
void CINTprim_to_ctr_0(double *gc, const FINT nf, const double *gp,
                       const FINT nprim, const FINT nctr, const double *coeff);
void CINTprim_to_ctr_1(double *gc, const FINT nf, const double *gp,
                       const FINT nprim, const FINT nctr, const double *coeff);
void CINTprim_to_ctr_opt(double *gc, const FINT nf, const double *gp,
                         double *non0coeff, FINT *non0idx, FINT non0ctr);

double CINTcommon_fac_sp(FINT l);

#define G1E_D_I(f, g, li, lj)   CINTnabla1i_1e(f, g, li, lj, envs)
#define G1E_D_J(f, g, li, lj)   CINTnabla1j_1e(f, g, li, lj, envs)
/* r-R_0, R_0 is (0,0,0) */
#define G1E_R0I(f, g, li, lj)   CINTx1i_1e(f, g, ri, li, lj, envs)
#define G1E_R0J(f, g, li, lj)   CINTx1j_1e(f, g, rj, li, lj, envs)
/* r-R_C, R_C is common origin */
#define G1E_RCI(f, g, li, lj)   CINTx1i_1e(f, g, dri, li, lj, envs)
#define G1E_RCJ(f, g, li, lj)   CINTx1j_1e(f, g, drj, li, lj, envs)
/* origin from center of each basis
 * x1[ij]_1e(f, g, ng, li, lj, 0d0) */
#define G1E_R_I(f, g, li, lj)   f = g + 1
#define G1E_R_J(f, g, li, lj)   f = g + envs->g_stride_j
