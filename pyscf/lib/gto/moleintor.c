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
#include <complex.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define NCTRMAX         64

static void cart_or_sph(int (*intor)(), int (*num_cgto)(),
                        double *mat, int ncomp, int hermi,
                        int *bralst, int nbra, int *ketlst, int nket,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish;
        int ilocs[nbra+1];
        int naoi = 0;
        int naoj = 0;
        for (ish = 0; ish < nbra; ish++) {
                ilocs[ish] = naoi;
                naoi += (*num_cgto)(bralst[ish], bas);
        }
        ilocs[nbra] = naoi;
        for (ish = 0; ish < nket; ish++) {
                naoj += (*num_cgto)(ketlst[ish], bas);
        }

#pragma omp parallel default(none) \
        shared(intor, num_cgto, mat, ncomp, hermi, bralst, nbra, ketlst, nket,\
               atm, natm, bas, nbas, env, naoi, naoj, ilocs) \
        private(ish)
{
        int jsh, jsh1, i, j, i0, j0, icomp;
        int di, dj, iloc, jloc;
        int shls[2];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*ncomp);
        double *pmat, *pbuf;
#pragma omp for nowait schedule(dynamic)
        for (ish = 0; ish < nbra; ish++) {
                iloc = ilocs[ish];
                di = ilocs[ish+1] - iloc;
                if (hermi == PLAIN) {
                        jsh1 = nket;
                } else {
                        jsh1 = ish + 1;
                }
                for (jloc = 0, jsh = 0; jsh < jsh1; jsh++, jloc+=dj) {
                        dj = (*num_cgto)(ketlst[jsh], bas);
                        shls[0] = bralst[ish];
                        shls[1] = ketlst[jsh];
                        (*intor)(buf, shls, atm, natm, bas, nbas, env);
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                pmat = mat + icomp*naoi*naoj;
                                pbuf = buf + icomp*di*dj;
                                for (i0=iloc, i=0; i < di; i++, i0++) {
                                for (j0=jloc, j=0; j < dj; j++, j0++) {
                                        pmat[i0*naoj+j0] = pbuf[j*di+i];
                                } }
                        }
                }
        }
        free(buf);
}
}

void GTO1eintor_sph(int (*intor)(), double *mat, int ncomp, int hermi,
                    int *bralst, int nbra, int *ketlst, int nket,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        cart_or_sph(intor, CINTcgto_spheric, mat, ncomp, hermi,
                    bralst, nbra, ketlst, nket, atm, natm, bas, nbas, env);
}

void GTO1eintor_cart(int (*intor)(), double *mat, int ncomp, int hermi,
                     int *bralst, int nbra, int *ketlst, int nket,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        cart_or_sph(intor, CINTcgto_cart, mat, ncomp, hermi,
                    bralst, nbra, ketlst, nket, atm, natm, bas, nbas, env);
}

void GTO1eintor_spinor(int (*intor)(), double complex *mat, int ncomp, int hermi,
                       int *bralst, int nbra, int *ketlst, int nket,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish;
        int ilocs[nbra+1];
        int naoi = 0;
        int naoj = 0;
        for (ish = 0; ish < nbra; ish++) {
                ilocs[ish] = naoi;
                naoi += CINTcgto_spinor(bralst[ish], bas);
        }
        ilocs[nbra] = naoi;
        for (ish = 0; ish < nket; ish++) {
                naoj += CINTcgto_spinor(ketlst[ish], bas);
        }

#pragma omp parallel default(none) \
        shared(intor, mat, ncomp, hermi, bralst, nbra, ketlst, nket,\
               atm, natm, bas, nbas, env, naoi, naoj, ilocs) \
        private(ish)
{
        int jsh, jsh1, i, j, i0, j0, icomp;
        int di, dj, iloc, jloc;
        int shls[2];
        double complex *buf = malloc(sizeof(double complex)*NCTRMAX*NCTRMAX*4*ncomp);
        double complex *pmat, *pbuf;
#pragma omp for nowait schedule(dynamic)
        for (ish = 0; ish < nbra; ish++) {
                iloc = ilocs[ish];
                di = CINTcgto_spinor(bralst[ish], bas);
                if (hermi == PLAIN) {
                        jsh1 = nket;
                } else {
                        jsh1 = ish + 1;
                }
                for (jloc = 0, jsh = 0; jsh < jsh1; jsh++, jloc+=dj) {
                        dj = CINTcgto_spinor(ketlst[jsh], bas);
                        shls[0] = bralst[ish];
                        shls[1] = ketlst[jsh];
                        (*intor)(buf, shls, atm, natm, bas, nbas, env);
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                pmat = mat + icomp*naoi*naoj;
                                pbuf = buf + icomp*di*dj;
                                for (i0=iloc, i=0; i < di; i++, i0++) {
                                for (j0=jloc, j=0; j < dj; j++, j0++) {
                                        pmat[i0*naoj+j0] = pbuf[j*di+i];
                                } }
                        }
                }
        }
        free(buf);
}
}

void GTO1e_intor_drv(int (*intor)(), double *mat, size_t ijoff,
                     int *basrange, int naoi, int naoj, int *iloc, int *jloc,
                     int ncomp, int hermi, CINTOpt *cintopt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
#pragma omp parallel default(none) \
        shared(intor, mat, ijoff, basrange, naoi, naoj, iloc, jloc, \
               ncomp, hermi, cintopt, atm, natm, bas, nbas, env)
{
        int ish, jsh, i, j, i0, j0, icomp;
        int brastart = basrange[0];
        int bracount = basrange[1];
        int ketstart = basrange[2];
        int ketcount = basrange[3];
        int di, dj;
        int shls[2];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*ncomp);
        double *pmat, *pbuf;
#pragma omp for nowait schedule(dynamic)
        for (ish = 0; ish < bracount; ish++) {
                di = iloc[ish+1] - iloc[ish];
                for (jsh = 0; jsh < ketcount; jsh++) {
                        if (hermi != PLAIN && iloc[ish] < jloc[jsh]) {
                                continue;
                        }
                        dj = jloc[jsh+1] - jloc[jsh];
                        shls[0] = brastart + ish;
                        shls[1] = ketstart + jsh;
                        (*intor)(buf, shls, atm, natm, bas, nbas, env);
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                pmat = mat + icomp*naoi*naoj + ijoff;
                                pbuf = buf + icomp*di*dj;
                                for (i0=iloc[ish], i=0; i < di; i++, i0++) {
                                for (j0=jloc[jsh], j=0; j < dj; j++, j0++) {
                                        pmat[i0*naoj+j0] = pbuf[j*di+i];
                                } }
                        }
                }
        }
        free(buf);
}
}

void GTO1e_spinor_drv(int (*intor)(), double complex *mat, size_t ijoff,
                      int *basrange, int naoi, int naoj, int *iloc, int *jloc,
                      int ncomp, int hermi, CINTOpt *cintopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
#pragma omp parallel default(none) \
        shared(intor, mat, ijoff, basrange, naoi, naoj, iloc, jloc, \
               ncomp, hermi, cintopt, atm, natm, bas, nbas, env)
{
        int ish, jsh, i, j, i0, j0, icomp;
        int brastart = basrange[0];
        int bracount = basrange[1];
        int ketstart = basrange[2];
        int ketcount = basrange[3];
        int di, dj;
        int shls[2];
        double complex *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*ncomp);
        double complex *pmat, *pbuf;
#pragma omp for nowait schedule(dynamic)
        for (ish = 0; ish < bracount; ish++) {
                di = iloc[ish+1] - iloc[ish];
                for (jsh = 0; jsh < ketcount; jsh++) {
                        if (hermi != PLAIN && iloc[ish] < jloc[jsh]) {
                                continue;
                        }
                        dj = jloc[jsh+1] - jloc[jsh];
                        shls[0] = brastart + ish;
                        shls[1] = ketstart + jsh;
                        (*intor)(buf, shls, atm, natm, bas, nbas, env);
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                pmat = mat + icomp*naoi*naoj + ijoff;
                                pbuf = buf + icomp*di*dj;
                                for (i0=iloc[ish], i=0; i < di; i++, i0++) {
                                for (j0=jloc[jsh], j=0; j < dj; j++, j0++) {
                                        pmat[i0*naoj+j0] = pbuf[j*di+i];
                                } }
                        }
                }
        }
        free(buf);
}
}

