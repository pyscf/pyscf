/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"

// 2 slots of int param[]
#define POS_E1   0
#define TENSOR   1

// 128s42p21d12f8g6h4i3j 
#define NCTR_CART      128
//  72s24p14d10f8g6h5i4j 
#define NCTR_SPH        72
#define NPRIMAX         64
#define BLKSIZE         96
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))

static int _len_cart[] = {
        1, 3, 6, 10, 15, 21, 28, 36
};

double CINTcommon_fac_sp(int l);
int GTOcontract_exp0(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, int blksize, double fac);
int GTOcontract_exp1(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, int blksize, double fac);
int GTOprim_exp(double *ectr, double *coord, double *alpha, double *coeff,
                int l, int nprim, int nctr, int blksize, double fac);
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff,
                             int l, int np, int nc, int blksize);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize);
void GTOshell_eval_grid_cart_deriv2(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize);
void GTOshell_eval_grid_cart_deriv3(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize);
void GTOshell_eval_grid_cart_deriv4(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    int l, int np, int nc, int blksize);

static void axpy(double complex **out, double *ao0, double complex *expLk,
                 int nkpts, size_t off, int ngrids, int blksize, int ncol)
{
        int i, j, ik;
        double complex *out_ik;
        for (ik = 0; ik < nkpts; ik++) {
                out_ik = out[ik] + off;
                for (j = 0; j < ncol; j++) {
                for (i = 0; i < blksize; i++) {
                        out_ik[j*ngrids+i] += ao0[j*blksize+i] * expLk[ik];
                } }
        }
}

// grid2atm[xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *coord, double *L,
                           int blksize, int atm_id, int *atm, double *env)
{
        int ig;
        double *r_atm = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
        double rL[3];
        rL[0] = r_atm[0] + L[0];
        rL[1] = r_atm[1] + L[1];
        rL[2] = r_atm[2] + L[2];
        for (ig = 0; ig < blksize; ig++) {
                grid2atm[0*blksize+ig] = coord[ig*3+0] - rL[0];
                grid2atm[1*blksize+ig] = coord[ig*3+1] - rL[1];
                grid2atm[2*blksize+ig] = coord[ig*3+2] - rL[2];
        }
}


void PBCeval_sph_iter(void (*feval)(),  int (*fexp)(),
                      int param[], int ish, int ngrids, int blksize,
                      double *Ls, int nimgs, double complex *expLk, int nkpts,
                      int *shls_slice, int *ao_loc,
                      double complex **ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int nao = ao_loc[ish1] - ao_loc[ish0];
        const int sh_id = ish0 + ish;
        const int ncomp = param[TENSOR];
        const int atm_id = bas[sh_id*BAS_SLOTS+ATOM_OF];
        const int np = bas[sh_id*BAS_SLOTS+NPRIM_OF];
        const int nc = bas[sh_id*BAS_SLOTS+NCTR_OF ];
        const int l  = bas[sh_id*BAS_SLOTS+ANG_OF  ];
        const int deg = l * 2 + 1;
        const double fac = CINTcommon_fac_sp(l);
        const double *p_exp  = env + bas[sh_id*BAS_SLOTS+PTR_EXP];
        const double *pcoeff = env + bas[sh_id*BAS_SLOTS+PTR_COEFF];
        const int di = ao_loc[sh_id+1] - ao_loc[sh_id];
        size_t off0 = (ao_loc[sh_id] - ao_loc[ish0]) * ngrids;
        size_t off;

        int i, k, m, grid0, ngrid_blk, blk_id;
        double *pcart, *ri;
        double *paobuf;
        double eprim[NPRIMAX*blksize*2];
        double cart_gto[NCTR_CART*blksize * ncomp];
        double aobuf[NCTR_SPH*blksize * ncomp];
        double grid2atm[3*blksize]; // [atm_id,xyz,grid]

        for (grid0 = 0; grid0 < ngrids; grid0 += blksize) {
        for (m = 0; m < nimgs; m++) {
                blk_id = grid0 / blksize;
                ngrid_blk = MIN(ngrids-grid0, blksize);

                _fill_grid2atm(grid2atm, coord+grid0*3, Ls+m*3, ngrid_blk,
                               atm_id, atm, env);
                if (non0table[blk_id*nbas+sh_id] &&
                    (*fexp)(eprim, grid2atm, p_exp, pcoeff, l, np, nc,
                            ngrid_blk, fac)) {
                        ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                        (*feval)(cart_gto, ri, eprim, grid2atm, p_exp, pcoeff,
                                 l, np, nc, ngrid_blk);

                        for (i = 0; i < ncomp; i++) {
                                off = off0 + ngrids*nao*i;
                                pcart = cart_gto + i*nc*_len_cart[l]*ngrid_blk;
                                if (l < 2) { // s, p functions
                                        axpy(ao, pcart, expLk+m*nkpts, nkpts,
                                             off+grid0, ngrids, ngrid_blk, di);
                                } else {
                                        paobuf = aobuf;
                                        for (k = 0; k < nc; k++) {
                                                CINTc2s_ket_sph(paobuf, ngrid_blk,
                                                                pcart, l);
                                                pcart += _len_cart[l] * ngrid_blk;
                                                paobuf += deg * ngrid_blk;
                                        }
                                        axpy(ao, aobuf, expLk+m*nkpts, nkpts,
                                             off+grid0, ngrids, ngrid_blk, di);
                                }
                        }
                }
        } }
}


/*
 * blksize <= 1024 to avoid stack overflow
 *
 * non0table[ngrids/blksize,natm] is the T/F table for ao values to
 * screen the ao evaluation for each shell
 */
void PBCeval_sph_drv(void (*feval)(), int (*fexp)(),
                     int param[], int ngrids, int blksize,
                     double *Ls, int nimgs, double complex *expLk, int nkpts,
                     int *shls_slice, int *ao_loc,
                     double complex **ao, double *coord, char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];

#pragma omp parallel default(none) \
        shared(feval, fexp, param, ngrids, blksize, \
               Ls, nimgs, expLk, nkpts, shls_slice, ao_loc, \
               ao, coord, non0table, atm, natm, bas, nbas, env)
{
        int ish;
#pragma omp for nowait schedule(dynamic, 2)
        for (ish = 0; ish < ish1-ish0; ish++) {
                PBCeval_sph_iter(feval, fexp, param, ish, ngrids, blksize,
                                 Ls, nimgs, expLk, nkpts, shls_slice, ao_loc,
                                 ao, coord, non0table, atm, natm, bas, nbas, env);
        }
}
}

void PBCval_sph_deriv0(int ngrids, int blksize, double *Ls, int nimgs,
                       double complex *expLk, int nkpts, int *shls_slice, int *ao_loc,
                       double complex **ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        PBCeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                        param, ngrids, blksize, Ls, nimgs, expLk, nkpts,
                        shls_slice, ao_loc, ao, coord, non0table,
                        atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv1(int ngrids, int blksize, double *Ls, int nimgs,
                       double complex *expLk, int nkpts, int *shls_slice, int *ao_loc,
                       double complex **ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                        param, ngrids, blksize, Ls, nimgs, expLk, nkpts,
                        shls_slice, ao_loc, ao, coord, non0table,
                        atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv2(int ngrids, int blksize, double *Ls, int nimgs,
                       double complex *expLk, int nkpts, int *shls_slice, int *ao_loc,
                       double complex **ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                        param, ngrids, blksize, Ls, nimgs, expLk, nkpts,
                        shls_slice, ao_loc, ao, coord, non0table,
                        atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv3(int ngrids, int blksize, double *Ls, int nimgs,
                       double complex *expLk, int nkpts, int *shls_slice, int *ao_loc,
                       double complex **ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                        param, ngrids, blksize, Ls, nimgs, expLk, nkpts,
                        shls_slice, ao_loc, ao, coord, non0table,
                        atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv4(int ngrids, int blksize, double *Ls, int nimgs,
                       double complex *expLk, int nkpts, int *shls_slice, int *ao_loc,
                       double complex **ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                        param, ngrids, blksize, Ls, nimgs, expLk, nkpts,
                        shls_slice, ao_loc, ao, coord, non0table,
                        atm, natm, bas, nbas, env);
}


