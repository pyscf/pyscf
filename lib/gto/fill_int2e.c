/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"


#define NCTRMAX         64

/*
 *************************************************
 * 2e AO integrals in s4, s2ij, s2kl, s1
 */

void GTOnr2e_fill_s1(int (*intor)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ishp, int jshp,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * nj;
        size_t nkl = nk * nl;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0 * nj + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        double *eri0, *peri, *buf0, *pbuf;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        dijk = dij * dk;
                        dijkl = dijk * dl;
                        eri0 = eri + k0*nl+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[k*nl+l] = pbuf[l*dijk];
                                        } }
                                } }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*nl+l0;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++) {
                                                for (l = 0; l < dl; l++) {
                                                        peri[k*nl+l] = 0;
                                                }
                                        }
                                } }
                                eri0 += neri;
                        }
                }
        } }
        free(buf);
}

void GTOnr2e_fill_s2ij(int (*intor)(), int (*fprescreen)(),
                       double *eri, int ncomp, int ishp, int jshp,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        if (ishp < jshp) {
                return;
        }

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        //int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        //int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * (ni+1) / 2;
        size_t nkl = nk * nl;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0*(i0+1)/2 + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        double *eri0, *peri0, *peri, *buf0, *pbuf;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        dijk = dij * dk;
                        dijkl = dijk * dl;
                        eri0 = eri + k0*nl+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                peri0 = eri0;
                                if (ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[k*nl+l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[k*nl+l] = pbuf[l*dijk];
                                        } }
                                } }
                                }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*nl+l0;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                peri0 = eri0;
                                if (ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[k*nl+l] = 0;
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[k*nl+l] = 0;
                                        } }
                                } }
                                }
                                eri0 += neri;
                        }
                }
        } }
        free(buf);
}

void GTOnr2e_fill_s2kl(int (*intor)(), int (*fprescreen)(),
                       double *eri, int ncomp, int ishp, int jshp,
                       int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        //int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        //int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * nj;
        size_t nkl = nk * (nk+1) / 2;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0 * nj + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh, kshp, lshp;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        double *eri0, *peri, *buf0, *pbuf;

        shls[0] = ish;
        shls[1] = jsh;

        for (kshp = 0; kshp < ksh1-ksh0; kshp++) {
        for (lshp = 0; lshp <= kshp; lshp++) {
                ksh = kshp + ksh0;
                lsh = lshp + lsh0;
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        dijk = dij * dk;
                        dijkl = dijk * dl;
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                if (kshp > lshp) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                if (kshp > lshp) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        peri = eri0 + nkl*(i*nj+j);
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                }
                                eri0 += neri;
                        }
                }
        } }
        free(buf);
}

void GTOnr2e_fill_s4(int (*intor)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ishp, int jshp,
                     int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        if (ishp < jshp) {
                return;
        }

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        //int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        //int lsh1 = shls_slice[7];
        int ni = ao_loc[ish1] - ao_loc[ish0];
        //int nj = ao_loc[jsh1] - ao_loc[jsh0];
        int nk = ao_loc[ksh1] - ao_loc[ksh0];
        //int nl = ao_loc[lsh1] - ao_loc[lsh0];
        size_t nij = ni * (ni+1) / 2;
        size_t nkl = nk * (nk+1) / 2;
        size_t neri = nij * nkl;

        int ish = ishp + ish0;
        int jsh = jshp + jsh0;
        int i0 = ao_loc[ish] - ao_loc[ish0];
        int j0 = ao_loc[jsh] - ao_loc[jsh0];
        eri += nkl * (i0*(i0+1)/2 + j0);

        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int dij = di * dj;
        int k0, l0, dk, dl, dijk, dijkl;
        int i, j, k, l, icomp;
        int ksh, lsh, kshp, lshp;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        double *eri0, *peri0, *peri, *buf0, *pbuf;

        shls[0] = ish;
        shls[1] = jsh;

        for (kshp = 0; kshp < ksh1-ksh0; kshp++) {
        for (lshp = 0; lshp <= kshp; lshp++) {
                ksh = kshp + ksh0;
                lsh = lshp + lsh0;
                shls[2] = ksh;
                shls[3] = lsh;
                k0 = ao_loc[ksh] - ao_loc[ksh0];
                l0 = ao_loc[lsh] - ao_loc[lsh0];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, shls, atm, natm, bas, nbas, env, cintopt)) {
                        dijk = dij * dk;
                        dijkl = dijk * dl;
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                peri0 = eri0;
                                if (kshp > lshp && ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else if (ish > jsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else if (ksh > lsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (pbuf = buf0 + k*dij + j*di + i,
                                             l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        } }
                                } }
                                }
                                buf0 += dijkl;
                                eri0 += neri;
                        }
                } else {
                        dijk = dij * dk;
                        dijkl = dijk * dl;
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                peri0 = eri0;
                                if (kshp > lshp && ishp > jshp) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else if (ish > jsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j < dj; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else if (ksh > lsh) {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                } else {
                                for (i = 0; i < di; i++, peri0+=nkl*(i0+i)) {
                                for (j = 0; j <= i; j++) {
                                        peri = peri0 + nkl*j;
                                        for (k = 0; k < dk; k++, peri+=k0+k) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        } }
                                } }
                                }
                                eri0 += neri;
                        }
                }
        } }
        free(buf);
}

static int no_prescreen()
{
        return 1;
}

void GTOnr2e_fill_drv(int (*intor)(), void (*fill)(), int (*fprescreen)(),
                      double *eri, int ncomp,
                      int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        if (fprescreen == NULL) {
                fprescreen = no_prescreen;
        }

        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;

#pragma omp parallel default(none) \
        shared(fill, fprescreen, eri, intor, ncomp, \
               shls_slice, ao_loc, cintopt, atm, natm, bas, nbas, env)
{
        int ij, i, j;
#pragma omp for nowait schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                (*fill)(intor, fprescreen, eri, ncomp, i, j, shls_slice,
                        ao_loc, cintopt, atm, natm, bas, nbas, env);
        }
}
}

