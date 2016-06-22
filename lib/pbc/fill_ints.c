/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include "config.h"
#include "cint.h"

#define NCTRMAX         72

static void axpy_s1(double complex **out, double *in,
                    double complex *expLk, int nkpts, int comp, size_t off,
                    size_t ni, size_t nij, size_t nijk,
                    size_t di, size_t dj, size_t dk)
{
        const size_t dij = di * dj;
        int i, j, k, ic, ik;
        double complex *out_ik, *pout;
        double *pin;
        for (ic = 0; ic < comp; ic++) {
                for (ik = 0; ik < nkpts; ik++) {
                        out_ik = out[ik] + off;
                        for (k = 0; k < dk; k++) {
                                pout = out_ik + k * nij;
                                pin  = in     + k * dij;
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++) {
                                        pout[j*ni+i] += pin[j*di+i] * expLk[ik];
                                } }
                        }
                }
                off += nijk;
                in  += dij * dk;
        }
}
/*
 * out[naoi,naoj,naok,comp] in F-order
 */
void PBCnr3c_fill_s1(int (*intor)(), double complex **out,
                     double complex *expLk, int nkpts, int comp,
                     int jsh, int ksh, double *buf,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t nij = naoi * naoj;
        const size_t nijk = nij * naok;

        jsh += jsh0;
        ksh += ksh0;
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const int kp = ao_loc[ksh] - ao_loc[ksh0];
        const size_t off = kp * nij + jp * naoi;

        int ish, di, i0;
        int shls[3];

        shls[1] = jsh;
        shls[2] = ksh;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                i0 = ao_loc[ish  ] - ao_loc[ish0];
                di = ao_loc[ish+1] - ao_loc[ish];
                if ((*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        axpy_s1(out, buf, expLk, nkpts, comp, off+i0,
                                naoi, nij, nijk, di, dj, dk);
                }
        }
}


void PBCnr3c_loop(int (*intor)(), void (*fill)(), double complex **eri,
                  double complex *expLk, int nkpts, int comp,
                  int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const int njsh = jsh1 - jsh0;
        const int nksh = ksh1 - ksh0;

#pragma omp parallel default(none) \
        shared(intor, fill, eri, expLk, nkpts, comp, \
               shls_slice, ao_loc, cintopt, atm, natm, bas, nbas, env)
{
        int jsh, ksh, jk;
        double *buf = (double *)malloc(sizeof(double)*NCTRMAX*NCTRMAX*NCTRMAX*comp);
#pragma omp for schedule(dynamic)
        for (jk = 0; jk < njsh*nksh; jk++) {
                ksh = jk / njsh;
                jsh = jk % njsh;
                (*fill)(intor, eri, expLk, nkpts, comp,
                        jsh, ksh, buf, shls_slice, ao_loc,
                        cintopt, atm, natm, bas, nbas, env);
        }
        free(buf);
}
}

static void shift_bas(double *xyz, int *ptr_coords, double *L, int nxyz, double *env)
{
        int i, p;
        for (i = 0; i < nxyz; i++) {
                p = ptr_coords[i];
                env[p+0] = xyz[i*3+0] + L[0];
                env[p+1] = xyz[i*3+1] + L[1];
                env[p+2] = xyz[i*3+2] + L[2];
        }
}

void PBCnr3c_drv(int (*intor)(), void (*fill)(), double complex **eri,
                 double *xyz, int *ptr_coords, int nxyz, double *Ls, int nimgs,
                 double complex *expLk, int nkpts, int comp,
                 int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int m;
        for (m = 0; m < nimgs; m++) {
                shift_bas(xyz, ptr_coords, Ls+m*3, nxyz, env);
                PBCnr3c_loop(intor, fill, eri, expLk+m*nkpts, nkpts, comp,
                             shls_slice, ao_loc, cintopt,
                             atm, natm, bas, nbas, env);
        }
}



void PBCnr2c2e_fill_s1(int (*intor)(), double complex **out,
                     double complex *expLk, int nkpts, int comp,
                     int jsh, int ksh, double *buf,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nij = naoi * naoj;

        jsh += jsh0;
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const size_t off = jp * naoi;

        int ish, di, i0;
        int shls[2];
        shls[1] = jsh;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                i0 = ao_loc[ish  ] - ao_loc[ish0];
                di = ao_loc[ish+1] - ao_loc[ish];
                if ((*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        axpy_s1(out, buf, expLk, nkpts, comp, off+i0,
                                naoi, nij, nij, di, dj, 1);
                }
        }
}

void PBCnr2c2e_fill_s2(int (*intor)(), double complex **out,
                     double complex *expLk, int nkpts, int comp,
                     int jsh, int ksh, double *buf,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nij = naoi * naoj;

        const int dj = ao_loc[jsh+jsh0+1] - ao_loc[jsh+jsh0];
        const int jp = ao_loc[jsh+jsh0] - ao_loc[jsh0];
        const size_t off = jp * naoi;
        int i, ish, di, i0;
        int shls[2];
        shls[1] = jsh + jsh0;

        for (i = 0; i <= jsh; i++) {
                ish = i + ish0;
                shls[0] = ish;
                i0 = ao_loc[ish  ] - ao_loc[ish0];
                di = ao_loc[ish+1] - ao_loc[ish];
                if ((*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        axpy_s1(out, buf, expLk, nkpts, comp, off+i0,
                                naoi, nij, nij, di, dj, 1);
                }
        }
}


void PBCnr2c_fill_s1(int (*intor)(), double complex **out,
                     double complex *expLk, int nkpts, int comp,
                     int jsh, int ksh, double *buf,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nij = naoi * naoj;

        jsh += jsh0;
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        const size_t off = jp * naoi;

        int ish, di, i0;
        int shls[2];
        shls[1] = jsh;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                i0 = ao_loc[ish  ] - ao_loc[ish0];
                di = ao_loc[ish+1] - ao_loc[ish];
                if ((*intor)(buf, shls, atm, natm, bas, nbas, env)) {
                        axpy_s1(out, buf, expLk, nkpts, comp, off+i0,
                                naoi, nij, nij, di, dj, 1);
                }
        }
}

void PBCnr2c_fill_s2(int (*intor)(), double complex **out,
                     double complex *expLk, int nkpts, int comp,
                     int jsh, int ksh, double *buf,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t nij = naoi * naoj;

        const int dj = ao_loc[jsh+jsh0+1] - ao_loc[jsh+jsh0];
        const int jp = ao_loc[jsh+jsh0] - ao_loc[jsh0];
        const size_t off = jp * naoi;
        int i, ish, di, i0;
        int shls[2];
        shls[1] = jsh + jsh0;

        for (i = 0; i <= jsh; i++) {
                ish = i + ish0;
                shls[0] = ish;
                i0 = ao_loc[ish  ] - ao_loc[ish0];
                di = ao_loc[ish+1] - ao_loc[ish];
                if ((*intor)(buf, shls, atm, natm, bas, nbas, env)) {
                        axpy_s1(out, buf, expLk, nkpts, comp, off+i0,
                                naoi, nij, nij, di, dj, 1);
                }
        }
}

void PBCnr2c_drv(int (*intor)(), void (*fill)(), double complex **out,
                 double *xyz, int *ptr_coords, int nxyz, double *Ls, int nimgs,
                 double complex *expLk, int nkpts, int comp,
                 int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
        int shls_slice_3c[6];
        shls_slice_3c[0] = shls_slice[0];
        shls_slice_3c[1] = shls_slice[1];
        shls_slice_3c[2] = shls_slice[2];
        shls_slice_3c[3] = shls_slice[3];
        shls_slice_3c[4] = 0;
        shls_slice_3c[5] = 1;

        int m;
        for (m = 0; m < nimgs; m++) {
                shift_bas(xyz, ptr_coords, Ls+m*3, nxyz, env);
                PBCnr3c_loop(intor, fill, out, expLk+m*nkpts, nkpts, comp,
                             shls_slice_3c, ao_loc, cintopt,
                             atm, natm, bas, nbas, env);
        }
}
