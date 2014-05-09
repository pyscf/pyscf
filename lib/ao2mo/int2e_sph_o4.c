/*
 * File: int2e_sph_o4.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "cint.h"
#include "int2e_ao2mo.h"

//#define ERI_CUTOFF 1e-14
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

void int2e_sph_o4(double *eri, const int *atm, const int natm,
                  const int *bas, const int nbas, const double *env)
{
        int ish, jsh, ksh, lsh;
        int di, dj, dk, dl;
        int i, j, k, l, ij, kl, ijkl, shls_ij, shls_kl, shls_kl_max;
        int i0, j0, k0, l0;
        int *ishls = malloc(sizeof(int)*nbas*nbas);
        int *jshls = malloc(sizeof(int)*nbas*nbas);
        int shls[4];
        int ao_loc[nbas];
        double *fijkl;
        for (i = 0, ij = 0; i < nbas; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        ishls[ij] = i;
                        jshls[ij] = j;
                }
        }

        CINTshells_spheric_offset(ao_loc, bas, nbas);
        CINTOpt *opt = NULL;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);

        double *qijij = malloc(sizeof(double)*nbas*nbas);
        double v;
        for (ish = 0; ish < nbas; ish++) {
                for (jsh = 0; jsh <= ish; jsh++) {
                        di = CINTcgtos_spheric(ish, bas);
                        dj = CINTcgtos_spheric(jsh, bas);
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ish;
                        shls[3] = jsh;
                        fijkl = malloc(sizeof(double) * di*dj*di*dj);
                        cint2e_sph(fijkl, shls, atm, natm, bas, nbas, env, opt);
                        shls_ij = LOWERTRI_INDEX(ish, jsh);
                        qijij[shls_ij] = 0;
                        for (i = ao_loc[ish], i0 = 0; i0 < di; i++, i0++)
                        for (j = ao_loc[jsh], j0 = 0; j0 < dj; j++, j0++)
                        for (k = ao_loc[ish], k0 = 0; k0 < di; k++, k0++)
                        for (l = ao_loc[jsh], l0 = 0; l0 < dj; l++, l0++) {
                                ij = LOWERTRI_INDEX(i, j);
                                kl = LOWERTRI_INDEX(k, l);
                                v = fijkl[i0+di*j0+di*dj*k0+di*dj*di*l0];
                                if (i >= j && k >= l && ij >= kl) {
                                        ijkl = LOWERTRI_INDEX(ij, kl);
                                        eri[ijkl] = v;
                                }
                                qijij[shls_ij] = MAX(qijij[shls_ij], fabs(v));
                        }
                        qijij[shls_ij] = sqrt(qijij[shls_ij]);
                        free(fijkl);
                }
        }

#pragma omp parallel default(none) \
        shared(eri, atm, bas, env, ishls, jshls, ao_loc, qijij, opt) \
        private(ish, jsh, ksh, lsh, di, dj, dk, dl, \
                i, j, k, l, i0, j0, k0, l0, \
                shls, fijkl, shls_ij, shls_kl, shls_kl_max, ij, kl, ijkl)
#pragma omp for nowait schedule(dynamic, 2)
        for (shls_ij = 0; shls_ij < nbas*(nbas+1)/2; shls_ij++) {
                ish = ishls[shls_ij];
                jsh = jshls[shls_ij];
                di = CINTcgtos_spheric(ish, bas);
                dj = CINTcgtos_spheric(jsh, bas);
                if (di != 1) {
                        // when ksh==ish, there exists k<i, so it's
                        // possible shls_kl>shls_ij
                        shls_kl_max = (ish+1)*(ish+2)/2;
                } else {
                        shls_kl_max = shls_ij;
                }
                for (shls_kl = 0; shls_kl < shls_kl_max; shls_kl++) {
                        if (shls_ij == shls_kl) {
                                continue;
                        }
                        ksh = ishls[shls_kl];
                        lsh = jshls[shls_kl];
                        dk = CINTcgtos_spheric(ksh, bas);
                        dl = CINTcgtos_spheric(lsh, bas);
                        if (qijij[shls_ij]*qijij[shls_kl] < 1e-13) {
                                for (i = ao_loc[ish]; i < ao_loc[ish]+di; i++)
                                for (j = ao_loc[jsh]; j < ao_loc[jsh]+dj; j++)
                                for (k = ao_loc[ksh]; k < ao_loc[ksh]+dk; k++)
                                for (l = ao_loc[lsh]; l < ao_loc[lsh]+dl; l++) {
                                        ij = LOWERTRI_INDEX(i, j);
                                        kl = LOWERTRI_INDEX(k, l);
                                        ijkl = LOWERTRI_INDEX(ij, kl);
                                        if (i >= j && k >= l && ij >= kl) {
                                                eri[ijkl] = 0;
                                        }
                                }
                                continue;
                        }
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ksh;
                        shls[3] = lsh;
                        fijkl = malloc(sizeof(double) * di*dj*dk*dl);
                        cint2e_sph(fijkl, shls, atm, natm, bas, nbas, env, opt);
                        for (i0 = ao_loc[ish], i = 0; i < di; i++,i0++)
                        for (j0 = ao_loc[jsh], j = 0; j < dj; j++,j0++)
                        for (k0 = ao_loc[ksh], k = 0; k < dk; k++,k0++)
                        for (l0 = ao_loc[lsh], l = 0; l < dl; l++,l0++) {
                                ij = LOWERTRI_INDEX(i0, j0);
                                kl = LOWERTRI_INDEX(k0, l0);
                                ijkl = LOWERTRI_INDEX(ij, kl);
                                if (i0 >= j0 && k0 >= l0 && ij >= kl) {
                                        eri[ijkl] = fijkl[i+di*j+di*dj*k+di*dj*dk*l];
#if defined ERI_CUTOFF
                                        if (fabs(eri[ijkl]) < ERI_CUTOFF) {
                                                eri[ijkl] = 0;
                                        }
#endif
                                }
                        } // ijkl, i0,j0,k0,l0
                        free(fijkl);
                } // shls_kl
        } // shls_ij
        free(ishls);
        free(jshls);
        free(qijij);
        CINTdel_2e_optimizer(&opt);
}
