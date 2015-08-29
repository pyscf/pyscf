/*
 * numpy helper
 */

#include <stdlib.h>
#include <complex.h>
#include "cint.h"

#define PLAIN        0
#define HERMITIAN    1
#define ANTIHERMI    2

static void cart_or_sph(int (*intor)(), int (*num_cgto)(),
                        double *mat, int ncomp, int hermi,
                        int *bralst, int nbra, int *ketlst, int nket,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int ish, jsh, jsh1, i, j, i0, j0, icomp;
        int di, dj, iloc, jloc;
        int shls[2];
        double *pmat, *pbuf;
        int naoi = 0;
        int naoj = 0;
        for (ish = 0; ish < nbra; ish++) {
                naoi += (*num_cgto)(bralst[ish], bas);
        }
        for (ish = 0; ish < nket; ish++) {
                naoj += (*num_cgto)(ketlst[ish], bas);
        }
        double *buf = malloc(sizeof(double)*naoi*naoj*ncomp);

        for (iloc = 0, ish = 0; ish < nbra; ish++, iloc+=di) {
                if (hermi == PLAIN) {
                        jsh1 = nket;
                } else {
                        jsh1 = ish + 1;
                }
                di = (*num_cgto)(bralst[ish], bas);
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
        int ish, jsh, jsh1, i, j, i0, j0, icomp;
        int di, dj, iloc, jloc;
        int shls[2];
        double complex *pmat, *pbuf;
        int naoi = 0;
        int naoj = 0;
        for (ish = 0; ish < nbra; ish++) {
                naoi += CINTcgto_spinor(bralst[ish], bas);
        }
        for (ish = 0; ish < nket; ish++) {
                naoj += CINTcgto_spinor(ketlst[ish], bas);
        }
        double complex *buf = malloc(sizeof(double complex)*naoi*naoj*ncomp);

        for (iloc = 0, ish = 0; ish < nbra; ish++, iloc+=di) {
                if (hermi == PLAIN) {
                        jsh1 = nket;
                } else {
                        jsh1 = ish + 1;
                }
                di = CINTcgto_spinor(bralst[ish], bas);
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

