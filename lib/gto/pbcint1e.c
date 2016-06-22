#include <stdlib.h>
#include "cint.h"

typedef struct {
        const int *atm;
        const int *bas;
        const double *env;
        const int *shls;
        int natm;
        int nbas;

        int i_l;
        int j_l;
        int k_l;
        int l_l;
        int i_prim;
        int j_prim;
        int k_prim;
        int l_prim;
        int i_ctr;
        int j_ctr;
        int k_ctr;
        int l_ctr;
        int nfi;  // number of cartesion components
        int nfj;
        int nfk;
        int nfl;
        int nf;  // = nfi*nfj*nfk*nfl;
        int _padding1;
        const double *ri;
        const double *rj;
        const double *rk;
        const double *rl;
        double common_factor;

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
        const double *rx_in_rijrx;
        const double *rx_in_rklrx;
        double rirj[3]; // diff by an sign in different g0_2d4d algorithm
        double rkrl[3];

        void (*f_g0_2d4d)();

        /* */
        void (*f_gout)();

        /* values are assigned during calculation */
        double ai;
        double aj;
        double ak;
        double al;
        double aij;
        double akl;
        double rij[3];
        double rijrx[3];
        double rkl[3];
        double rklrx[3];
        int *idx;
} CINTEnvVars;

int CINTinit_int1e_EnvVars(CINTEnvVars *envs, const int *ng, const int *shls,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);
int CINT1e_drv(double *opij, CINTEnvVars *envs, double fac,
               void (*const f_c2s)());
int CINTinit_int3c1e_EnvVars(CINTEnvVars *envs, const int *ng, const int *shls,
                             const int *atm, const int natm,
                             const int *bas, const int nbas, const double *env);
int CINT3c1e_spheric_drv(double *opijk, CINTEnvVars *envs, const CINTOpt *opt,
                         void (*const f_e1_c2s)(), int is_ssc);
void c2s_sph_1e(double *opij, const double *gctr, CINTEnvVars *envs);
void c2s_sph_3c1e(double *fijkl, const double *gctr, CINTEnvVars *envs);

#define G1E_R_I(f, g, li, lj)   f = g + 1
#define G3C1E_R_K(f, g, li, lj, lk)   f = g + envs->g_stride_k

/* <R DOT R i|OVLP |j> */
static void PBCgout1e_cint1e_r2_origi_sph(double *g,
        double *gout, const int *idx, const CINTEnvVars *envs)
{
        const double *env = envs->env;
        const int nf = envs->nf;
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        int ix, iy, iz, n;
        double *g0 = g;
        double *g1 = g0  + envs->g_size * 3;
        double *g2 = g1  + envs->g_size * 3;
        double s[3];
        G1E_R_I(g1, g0, i_l+1, j_l);
        G1E_R_I(g2, g1, i_l+0, j_l);
        for (n = 0; n < nf; n++, idx+=3) {
                ix = idx[0];
                iy = idx[1];
                iz = idx[2];
                s[0] = g2[ix] * g0[iy] * g0[iz];
                s[1] = g0[ix] * g2[iy] * g0[iz];
                s[2] = g0[ix] * g0[iy] * g2[iz];
                gout[0] += + s[0] + s[1] + s[2];
                gout += 1;
        }
}
int cint1e_pbc_r2_origi_sph(double *opij, const int *shls,
                            const int *atm, const int natm,
                            const int *bas, const int nbas, const double *env)
{
        int ng[] = {2, 0, 0, 0, 2, 1, 0, 1};
        CINTEnvVars envs;
        CINTinit_int1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &PBCgout1e_cint1e_r2_origi_sph;
        return CINT1e_drv(opij, &envs, 1, &c2s_sph_1e);
}


/* <R DOT R R DOT R i|OVLP |j> */
static void PBCgout1e_cint1e_r4_origi_sph(double *g,
        double *gout, const int *idx, const CINTEnvVars *envs)
{
        const double *env = envs->env;
        const int nf = envs->nf;
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        int ix, iy, iz, n;
        double *g0 = g;
        double *g1 = g0  + envs->g_size * 3;
        double *g2 = g1  + envs->g_size * 3;
        double *g3 = g2  + envs->g_size * 3;
        double *g4 = g3  + envs->g_size * 3;
        double *g5 = g4  + envs->g_size * 3;
        double *g6 = g5  + envs->g_size * 3;
        double s[6];
        G1E_R_I(g1, g0, i_l+3, j_l);
        G1E_R_I(g2, g1, i_l+2, j_l);
        G1E_R_I(g3, g0, i_l+1, j_l);
        G1E_R_I(g4, g2, i_l+1, j_l);
        G1E_R_I(g5, g3, i_l+0, j_l);
        G1E_R_I(g6, g4, i_l+0, j_l);
        for (n = 0; n < nf; n++, idx+=3) {
                ix = idx[0];
                iy = idx[1];
                iz = idx[2];
                s[0] = g6[ix] * g0[iy] * g0[iz];
                s[1] = g5[ix] * g2[iy] * g0[iz];
                s[2] = g5[ix] * g0[iy] * g2[iz];
                s[3] = g0[ix] * g6[iy] * g0[iz];
                s[4] = g0[ix] * g5[iy] * g2[iz];
                s[5] = g0[ix] * g0[iy] * g6[iz];
                gout[0] += s[0] + (2*s[1]) + (2*s[2]) + s[3] + (2*s[4]) + s[5];
                gout += 1;
        }
}
int cint1e_pbc_r4_origi_sph(double *opij, const int *shls,
                            const int *atm, const int natm,
                            const int *bas, const int nbas, const double *env)
{
        int ng[] = {4, 0, 0, 0, 4, 1, 0, 1};
        CINTEnvVars envs;
        CINTinit_int1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &PBCgout1e_cint1e_r4_origi_sph;
        return CINT1e_drv(opij, &envs, 1, &c2s_sph_1e);
}



/* (i j|CCC1E |R DOT R k) */
void PBCgout3c1e_cint3c1e_r2_origk_sph(double *g,
        double *gout, const int *idx, const CINTEnvVars *envs, int gout_empty)
{
        const double *env = envs->env;
        const int nf = envs->nf;
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int k_l = envs->k_l;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        const double *rk = envs->rk;
        int ix, iy, iz, i, n;
        double *g0 = g;
        double *g1 = g0 + envs->g_size * 3;
        double *g2 = g1 + envs->g_size * 3;
        double s[3];
        G3C1E_R_K(g1, g0, i_l+0, j_l+0, k_l+1);
        G3C1E_R_K(g2, g1, i_l+0, j_l+0, k_l+0);
        for (n = 0; n < nf; n++, idx+=3) {
                ix = idx[0];
                iy = idx[1];
                iz = idx[2];
                s[0] = g2[ix] * g0[iy] * g0[iz];
                s[1] = g0[ix] * g2[iy] * g0[iz];
                s[2] = g0[ix] * g0[iy] * g2[iz];
                if (gout_empty) {
                        gout[0] = + s[0] + s[1] + s[2];
                        gout += 1;
                } else {
                        gout[0] += + s[0] + s[1] + s[2];
                        gout += 1;
                }}
}
void cint3c1e_pbc_r2_origk_sph_optimizer(CINTOpt **opt, const int *atm, const int natm,
                                         const int *bas, const int nbas, const double *env)
{
        *opt = NULL;
}
int cint3c1e_pbc_r2_origk_sph(double *opijkl, const int *shls,
                              const int *atm, const int natm,
                              const int *bas, const int nbas, const double *env, CINTOpt *opt)
{
        int ng[] = {0, 0, 2, 0, 2, 1, 1, 1};
        CINTEnvVars envs;
        CINTinit_int3c1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &PBCgout3c1e_cint3c1e_r2_origk_sph;
        envs.common_factor *= 1;
        return CINT3c1e_spheric_drv(opijkl, &envs, opt, &c2s_sph_3c1e, 0);
}


/* (i j|CCC1E |R DOT R R DOT R k) */
void PBCgout3c1e_cint3c1e_r4_origk_sph(double *g,
        double *gout, const int *idx, const CINTEnvVars *envs, int gout_empty)
{
        const double *env = envs->env;
        const int nf = envs->nf;
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int k_l = envs->k_l;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        const double *rk = envs->rk;
        int ix, iy, iz, i, n;
        double *g0 = g;
        double *g1 = g0 + envs->g_size * 3;
        double *g2 = g1 + envs->g_size * 3;
        double *g3 = g2 + envs->g_size * 3;
        double *g4 = g3 + envs->g_size * 3;
        double *g5 = g4 + envs->g_size * 3;
        double *g6 = g5 + envs->g_size * 3;
        double s[6];
        G3C1E_R_K(g1, g0, i_l+0, j_l+0, k_l+3);
        G3C1E_R_K(g2, g1, i_l+0, j_l+0, k_l+2);
        G3C1E_R_K(g3, g0, i_l+0, j_l+0, k_l+1);
        G3C1E_R_K(g4, g2, i_l+0, j_l+0, k_l+1);
        G3C1E_R_K(g5, g3, i_l+0, j_l+0, k_l+0);
        G3C1E_R_K(g6, g4, i_l+0, j_l+0, k_l+0);
        for (n = 0; n < nf; n++, idx+=3) {
                ix = idx[0];
                iy = idx[1];
                iz = idx[2];
                s[0] = g6[ix] * g0[iy] * g0[iz];
                s[1] = g5[ix] * g2[iy] * g0[iz];
                s[2] = g5[ix] * g0[iy] * g2[iz];
                s[3] = g0[ix] * g6[iy] * g0[iz];
                s[4] = g0[ix] * g5[iy] * g2[iz];
                s[5] = g0[ix] * g0[iy] * g6[iz];
                if (gout_empty) {
                        gout[0] = s[0] + (2*s[1]) + (2*s[2]) + s[3] + (2*s[4]) + s[5];
                        gout += 1;
                } else {
                        gout[0] += s[0] + (2*s[1]) + (2*s[2]) + s[3] + (2*s[4]) + s[5];
                        gout += 1;
                }}
}
void cint3c1e_pbc_r4_origk_sph_optimizer(CINTOpt **opt, const int *atm, const int natm,
                                         const int *bas, const int nbas, const double *env)
{
        *opt = NULL;
}
int cint3c1e_pbc_r4_origk_sph(double *opijkl, const int *shls,
                              const int *atm, const int natm,
                              const int *bas, const int nbas, const double *env, CINTOpt *opt)
{
        int ng[] = {0, 0, 4, 0, 4, 1, 1, 1};
        CINTEnvVars envs;
        CINTinit_int3c1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &PBCgout3c1e_cint3c1e_r4_origk_sph;
        envs.common_factor *= 1;
        return CINT3c1e_spheric_drv(opijkl, &envs, opt, &c2s_sph_3c1e, 0);
}


/* (i j|CCC1E |R DOT R R DOT R R DOT R k) */
void PBCgout3c1e_cint3c1e_r6_origk_sph(double *g,
        double *gout, const int *idx, const CINTEnvVars *envs, int gout_empty)
{
        const double *env = envs->env;
        const int nf = envs->nf;
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int k_l = envs->k_l;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        const double *rk = envs->rk;
        int ix, iy, iz, i, n;
        double *g0 = g;
        double *g1 = g0 + envs->g_size * 3;
        double *g2 = g1 + envs->g_size * 3;
        double *g3 = g2 + envs->g_size * 3;
        double *g4 = g3 + envs->g_size * 3;
        double *g5 = g4 + envs->g_size * 3;
        double *g6 = g5 + envs->g_size * 3;
        double *g7 = g6 + envs->g_size * 3;
        double *g8 = g7 + envs->g_size * 3;
        double *g9 = g8 + envs->g_size * 3;
        double *g10 = g9 + envs->g_size * 3;
        double *g11 = g10 + envs->g_size * 3;
        double *g12 = g11 + envs->g_size * 3;
        double s[10];
        G3C1E_R_K(g1, g0, i_l+0, j_l+0, k_l+5);
        G3C1E_R_K(g2, g1, i_l+0, j_l+0, k_l+4);
        G3C1E_R_K(g3, g0, i_l+0, j_l+0, k_l+3);
        G3C1E_R_K(g4, g2, i_l+0, j_l+0, k_l+3);
        G3C1E_R_K(g5, g3, i_l+0, j_l+0, k_l+2);
        G3C1E_R_K(g6, g4, i_l+0, j_l+0, k_l+2);
        G3C1E_R_K(g7, g0, i_l+0, j_l+0, k_l+1);
        G3C1E_R_K(g8, g5, i_l+0, j_l+0, k_l+1);
        G3C1E_R_K(g9, g6, i_l+0, j_l+0, k_l+1);
        G3C1E_R_K(g10, g7, i_l+0, j_l+0, k_l+0);
        G3C1E_R_K(g11, g8, i_l+0, j_l+0, k_l+0);
        G3C1E_R_K(g12, g9, i_l+0, j_l+0, k_l+0);
        for (n = 0; n < nf; n++, idx+=3) {
                ix = idx[0];
                iy = idx[1];
                iz = idx[2];
                s[0] = g12[ix] * g0 [iy] * g0 [iz];
                s[1] = g11[ix] * g2 [iy] * g0 [iz];
                s[2] = g11[ix] * g0 [iy] * g2 [iz];
                s[3] = g10[ix] * g6 [iy] * g0 [iz];
                s[4] = g10[ix] * g5 [iy] * g2 [iz];
                s[5] = g10[ix] * g0 [iy] * g6 [iz];
                s[6] = g0 [ix] * g12[iy] * g0 [iz];
                s[7] = g0 [ix] * g11[iy] * g2 [iz];
                s[8] = g0 [ix] * g10[iy] * g6 [iz];
                s[9] = g0 [ix] * g0 [iy] * g12[iz];
                if (gout_empty) {
                        gout[0] = + s[0] + (3*s[1]) + (3*s[2]) + (3*s[3]) + (6*s[4]) + (3*s[5]) + s[6] + (3*s[7]) + (3*s[8]) + s[9];
                        gout += 1;
                } else {
                        gout[0] += + s[0] + (3*s[1]) + (3*s[2]) + (3*s[3]) + (6*s[4]) + (3*s[5]) + s[6] + (3*s[7]) + (3*s[8]) + s[9];
                        gout += 1;
                }}
}
void cint3c1e_pbc_r6_origk_sph_optimizer(CINTOpt **opt, const int *atm, const int natm,
                                         const int *bas, const int nbas, const double *env)
{
        *opt = NULL;
}
int cint3c1e_pbc_r6_origk_sph(double *opijkl, const int *shls,
                              const int *atm, const int natm,
                              const int *bas, const int nbas, const double *env, CINTOpt *opt) {
        int ng[] = {0, 0, 6, 0, 6, 1, 1, 1};
        CINTEnvVars envs;
        CINTinit_int3c1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &PBCgout3c1e_cint3c1e_r6_origk_sph;
        envs.common_factor *= 1;
        return CINT3c1e_spheric_drv(opijkl, &envs, opt, &c2s_sph_3c1e, 0);
}

