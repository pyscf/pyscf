/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "gto/grid_ao_drv.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define ALL_IMAGES      255

double CINTcommon_fac_sp(int l);
void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff, double *env,
                             int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv2(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv3(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);
void GTOshell_eval_grid_cart_deriv4(double *cgto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff, double *env,
                                    int l, int np, int nc, size_t nao, size_t ngrids, size_t bgrids);

/*
 * Extend the meaning of non0table:  given shell ID and block ID,
 * non0table is the number of images in Ls that does not vanish.
 * Ls should be sorted based on the distance to center cell.
 */
void PBCnr_ao_screen(unsigned char *non0table, double *coords, int ngrids,
                     double *Ls, int nimgs, 
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;

#pragma omp parallel default(none) \
        shared(Ls, nimgs, coords, ngrids, non0table, atm, natm, bas, nbas, env)
{
        int i, j, m;
        int np, nc, atm_id;
        size_t bas_id, ib;
        double rr, arr, maxc;
        double logcoeff[NPRIMAX];
        double dr[3];
        double rL[3];
        double *p_exp, *pcoeff, *ratm;
#pragma omp for nowait schedule(dynamic)
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                np = bas[NPRIM_OF+bas_id*BAS_SLOTS];
                nc = bas[NCTR_OF +bas_id*BAS_SLOTS];
                p_exp = env + bas[PTR_EXP+bas_id*BAS_SLOTS];
                pcoeff = env + bas[PTR_COEFF+bas_id*BAS_SLOTS];
                atm_id = bas[ATOM_OF+bas_id*BAS_SLOTS];
                ratm = env + atm[atm_id*ATM_SLOTS+PTR_COORD];

                for (j = 0; j < np; j++) {
                        maxc = 0;
                        for (i = 0; i < nc; i++) {
                                maxc = MAX(maxc, fabs(pcoeff[i*np+j]));
                        }
                        logcoeff[j] = log(maxc);
                }

                for (ib = 0; ib < nblk; ib++) {
                for (m = nimgs-1; m >= 0; m--) {
                        rL[0] = ratm[0] + Ls[m*3+0];
                        rL[1] = ratm[1] + Ls[m*3+1];
                        rL[2] = ratm[2] + Ls[m*3+2];
                        for (i = ib*BLKSIZE; i < MIN(ngrids, (ib+1)*BLKSIZE); i++) {
                                dr[0] = coords[0*ngrids+i] - rL[0];
                                dr[1] = coords[1*ngrids+i] - rL[1];
                                dr[2] = coords[2*ngrids+i] - rL[2];
                                rr = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                                for (j = 0; j < np; j++) {
                                        arr = p_exp[j] * rr;
                                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                                non0table[ib*nbas+bas_id] = MIN(ALL_IMAGES, m+1);
                                                goto next_blk;
                                        }
                                }
                        }
                }
                non0table[ib*nbas+bas_id] = 0;
next_blk:;
                }
        }
}
}


static void axpy(double complex **out, double *ao0, double complex *expLk,
                 int nkpts, size_t off, size_t ngrids, size_t bgrids, int ncol)
{
        size_t i, j, ik;
        double complex *out_ik;
        for (ik = 0; ik < nkpts; ik++) {
                out_ik = out[ik] + off;
                for (j = 0; j < ncol; j++) {
                for (i = 0; i < bgrids; i++) {
                        out_ik[j*ngrids+i] += ao0[j*BLKSIZE+i] * expLk[ik];
                } }
        }
}
static void set0(double complex **out,
                 int nkpts, size_t off, size_t ngrids, size_t bgrids, int ncol)
{
        size_t i, j, ik;
        double complex *out_ik;
        for (ik = 0; ik < nkpts; ik++) {
                out_ik = out[ik] + off;
                for (j = 0; j < ncol; j++) {
                for (i = 0; i < bgrids; i++) {
                        out_ik[j*ngrids+i] = 0;
                } }
        }
}

// grid2atm[xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *coord, double *L,
                           int bgrids, int ngrids,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int atm_id, ig;
        double *r_atm;
        double rL[3];
        for (atm_id = 0; atm_id < natm; atm_id++) {
                r_atm = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
                rL[0] = r_atm[0] + L[0];
                rL[1] = r_atm[1] + L[1];
                rL[2] = r_atm[2] + L[2];
                for (ig = 0; ig < bgrids; ig++) {
                        grid2atm[0*BLKSIZE+ig] = coord[0*ngrids+ig] - rL[0];
                        grid2atm[1*BLKSIZE+ig] = coord[1*ngrids+ig] - rL[1];
                        grid2atm[2*BLKSIZE+ig] = coord[2*ngrids+ig] - rL[2];
                }
                grid2atm += 3*BLKSIZE;
        }
}


void PBCeval_sph_iter(FPtr_eval feval,  FPtr_exp fexp,
                      size_t nao, size_t ngrids, size_t bgrids, size_t offao,
                      int param[], int *shls_slice, int *ao_loc, double *buf,
                      double *Ls, int nimgs, double complex *expLk, int nkpts,
                      double complex **ao, double *coord, unsigned char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ncomp = param[TENSOR];
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const int atmstart = bas[sh0*BAS_SLOTS+ATOM_OF];
        const int atmend = bas[(sh1-1)*BAS_SLOTS+ATOM_OF]+1;
        const int atmcount = atmend - atmstart;
        int i, k, l, m, np, nc, atm_id, bas_id, deg, dcart, di, ao_id;
        size_t off;
        double fac;
        double *p_exp, *pcoeff, *pcoord, *pcart, *ri, *pao;
        double *grid2atm = buf; // [atm_id,xyz,grid]
        double *eprim = grid2atm + atmcount*3*BLKSIZE;
        double *cart_gto = eprim + NPRIMAX*BLKSIZE*2;
        double *aobuf = cart_gto + BLKSIZE*NCTR_CART*ncomp*param[POS_E1];

        for (i = 0; i < ncomp; i++) {
                off = (i*nao+ao_loc[sh0])*ngrids + offao;
                set0(ao, nkpts, offao, ngrids, bgrids, ao_loc[sh1]-ao_loc[sh0]);
        }
        for (m = 0; m < nimgs; m++) {
                _fill_grid2atm(grid2atm, coord, Ls+m*3, bgrids, ngrids,
                               atm+atmstart*ATM_SLOTS, atmcount, bas, nbas, env);

                for (bas_id = sh0; bas_id < sh1; bas_id++) {
                        np = bas[bas_id*BAS_SLOTS+NPRIM_OF];
                        nc = bas[bas_id*BAS_SLOTS+NCTR_OF ];
                        l  = bas[bas_id*BAS_SLOTS+ANG_OF  ];
                        deg = l * 2 + 1;
                        fac = CINTcommon_fac_sp(l);
                        p_exp  = env + bas[bas_id*BAS_SLOTS+PTR_EXP];
                        pcoeff = env + bas[bas_id*BAS_SLOTS+PTR_COEFF];
                        atm_id = bas[bas_id*BAS_SLOTS+ATOM_OF];
                        pcoord = grid2atm + (atm_id - atmstart) * 3*BLKSIZE;
                        ao_id = ao_loc[bas_id] - ao_loc[sh0];
                        if ((m < non0table[bas_id] || non0table[bas_id] == ALL_IMAGES) &&
                            (*fexp)(eprim, pcoord, p_exp, pcoeff, l, np, nc, bgrids, fac)) {
                                dcart = (l+1)*(l+2)/2;
                                ri = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
        if (l <= 1) { // s, p functions
                (*feval)(aobuf, ri, eprim, pcoord, p_exp, pcoeff, env,
                         l, np, nc, nc*dcart, BLKSIZE, bgrids);
        } else {
                (*feval)(cart_gto, ri, eprim, pcoord, p_exp, pcoeff, env,
                         l, np, nc, nc*dcart, bgrids, bgrids);
                pcart = cart_gto;
                pao = aobuf;
                for (i = 0; i < ncomp; i++) {
                        for (k = 0; k < nc; k++) {
                                CINTc2s_ket_sph1(pao, pcart, BLKSIZE, bgrids, l);
                                pao += deg * BLKSIZE;
                                pcart += dcart * bgrids;
                        }
                }
        }
        di = nc * deg;
        for (i = 0; i < ncomp; i++) {
                off = (i*nao+ao_id)*ngrids + offao;
                pao = aobuf + i*di*BLKSIZE;
                axpy(ao, pao, expLk+m*nkpts, nkpts, off, ngrids, bgrids, di);
        }
                        }
                }
        }
}


int GTOshloc_by_atom(int *shloc, int *shls_slice, int *ao_loc, int *atm, int *bas);
/*
 * blksize <= 1024 to avoid stack overflow
 *
 * non0table[ngrids/blksize,natm] is the T/F table for ao values to
 * screen the ao evaluation for each shell
 */
void PBCeval_loop(void (*fiter)(), FPtr_eval feval, FPtr_exp fexp,
                  int ngrids, int param[], int *shls_slice, int *ao_loc,
                  double *Ls, int nimgs, double complex *expLk, int nkpts,
                  double complex **ao, double *coord, unsigned char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        int shloc[shls_slice[1]-shls_slice[0]+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t Ngrids = ngrids;

#pragma omp parallel default(none) \
        shared(fiter, feval, fexp, param, ngrids, \
               Ls, nimgs, expLk, nkpts, shls_slice, ao_loc, \
               ao, coord, non0table, atm, natm, bas, nbas, env, shloc)
{
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t aoff, bgrids;
        int ncart = NCTR_CART * param[TENSOR] * param[POS_E1];
        double *buf = malloc(sizeof(double) * BLKSIZE*(NPRIMAX*2+ncart*2));
#pragma omp for nowait schedule(static)
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ish = shloc[iloc];
                ib = k - iloc * nblk;
                ip = ib * BLKSIZE;
                aoff = (ao_loc[ish] - ao_loc[sh0]) * Ngrids + ip;
                bgrids = MIN(ngrids-ip, BLKSIZE);
                (*fiter)(feval, fexp, nao, Ngrids, bgrids, aoff,
                         param, shloc+iloc, ao_loc, buf, Ls, nimgs, expLk, nkpts,
                         ao, coord+ip, non0table+ib*nbas,
                         atm, natm, bas, nbas, env);
        }
        free(buf);
}
}

void PBCeval_sph_drv(FPtr_eval feval, FPtr_exp fexp,
                     int ngrids, int param[], int *shls_slice, int *ao_loc,
                     double *Ls, int nimgs, double complex *expLk, int nkpts,
                     double complex **ao, double *coord, unsigned char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        PBCeval_loop(PBCeval_sph_iter, feval, fexp,
                     ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                     ao, coord, non0table, atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv0(int ngrids, int *shls_slice, int *ao_loc,
                       double *Ls, int nimgs, double complex *expLk, int nkpts,
                       double complex **ao, double *coord, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 1};
        PBCeval_sph_drv(GTOshell_eval_grid_cart, GTOcontract_exp0,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *Ls, int nimgs, double complex *expLk, int nkpts,
                       double complex **ao, double *coord, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv2(int ngrids, int *shls_slice, int *ao_loc,
                       double *Ls, int nimgs, double complex *expLk, int nkpts,
                       double complex **ao, double *coord, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 10};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv2, GTOprim_exp,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv3(int ngrids, int *shls_slice, int *ao_loc,
                       double *Ls, int nimgs, double complex *expLk, int nkpts,
                       double complex **ao, double *coord, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 20};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv3, GTOprim_exp,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void PBCval_sph_deriv4(int ngrids, int *shls_slice, int *ao_loc,
                       double *Ls, int nimgs, double complex *expLk, int nkpts,
                       double complex **ao, double *coord, unsigned char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 35};
        PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv4, GTOprim_exp,
                        ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

