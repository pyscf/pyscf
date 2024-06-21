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
#include <math.h>
#include "config.h"
#include "cint.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter)
{
        int i;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        int di = 0;
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        for (i = i0; i < i1; i++) {
                di = MAX(di, ao_loc[i+1]-ao_loc[i]);
        }
        return di;
}
size_t GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        int i;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        size_t (*f)() = (size_t (*)())intor;
        size_t cache_size = 0;
        size_t n;
        int shls[4];
        for (i = i0; i < i1; i++) {
                shls[0] = i;
                shls[1] = i;
                shls[2] = i;
                shls[3] = i;
                n = (*f)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

/*
 *************************************************
 * 2e AO integrals in s4, s2ij, s2kl, s1
 */

void GTOnr2e_fill_s1(int (*intor)(), int (*fprescreen)(),
                     double *eri, double *buf, int comp, int ishp, int jshp,
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
        double *eri0, *peri, *buf0, *pbuf, *cache;

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
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*nl+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
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
                        for (icomp = 0; icomp < comp; icomp++) {
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
}

void GTOnr2e_fill_s2ij(int (*intor)(), int (*fprescreen)(),
                       double *eri, double *buf, int comp, int ishp, int jshp,
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
        double *eri0, *peri0, *peri, *buf0, *pbuf, *cache;

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
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*nl+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
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
                        for (icomp = 0; icomp < comp; icomp++) {
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
}

void GTOnr2e_fill_s2kl(int (*intor)(), int (*fprescreen)(),
                       double *eri, double *buf, int comp, int ishp, int jshp,
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
        double *eri0, *peri, *buf0, *pbuf, *cache;

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
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
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
                        for (icomp = 0; icomp < comp; icomp++) {
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
}

void GTOnr2e_fill_s4(int (*intor)(), int (*fprescreen)(),
                     double *eri, double *buf, int comp, int ishp, int jshp,
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
        double *eri0, *peri0, *peri, *buf0, *pbuf, *cache;

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
                dijk = dij * dk;
                dijkl = dijk * dl;
                cache = buf + dijkl * comp;
                if ((*fprescreen)(shls, atm, bas, env) &&
                    (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
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
                        eri0 = eri + k0*(k0+1)/2+l0;
                        buf0 = buf;
                        for (icomp = 0; icomp < comp; icomp++) {
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
}

static int no_prescreen()
{
        return 1;
}

void GTOnr2e_fill_drv(int (*intor)(), void (*fill)(), int (*fprescreen)(),
                      double *eri, int comp,
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
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                 atm, natm, bas, nbas, env);

#pragma omp parallel
{
        int ij, i, j;
        double *buf = malloc(sizeof(double) * (di*di*di*di*comp + cache_size));
#pragma omp for nowait schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                (*fill)(intor, fprescreen, eri, buf, comp, i, j, shls_slice,
                        ao_loc, cintopt, atm, natm, bas, nbas, env);
        }
        free(buf);
}
}

