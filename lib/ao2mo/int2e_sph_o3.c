/*
 * File: int2e_sph_o3.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "cint.h"
#include "int2e_ao2mo.h"

//#define ERI_CUTOFF 1e-14

void int2e_sph_o3(double *eri, const int *atm, const int natm,
                  const int *bas, const int nbas, const double *env)
{
        int ish, jsh, ksh, lsh;
        int di, dj, dk, dl;
        unsigned int i, j, k, l, ij, kl, ijkl, shls_ij;
        unsigned int i0, j0, k0, l0;
        int *ishls = malloc(sizeof(int)*nbas*nbas);
        int *jshls = malloc(sizeof(int)*nbas*nbas);
        unsigned int shls[4];
        int ao_loc[nbas];
        double *fijkl;
        for (i = 0, ij = 0; i < nbas; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        ishls[ij] = i;
                        jshls[ij] = j;
                }
        }

        CINTshells_spheric_offset(ao_loc, bas, nbas);

        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);

#pragma omp parallel default(none) \
        shared(eri, atm, bas, env, ishls, jshls, ao_loc, opt) \
        private(ish, jsh, ksh, lsh, di, dj, dk, dl, \
                i, j, k, l, i0, j0, k0, l0, \
                shls, fijkl, shls_ij, ij, kl, ijkl)
#pragma omp for nowait schedule(dynamic, 2)
        for (shls_ij = 0; shls_ij < nbas*(nbas+1)/2; shls_ij++) {
                ish = ishls[shls_ij];
                jsh = jshls[shls_ij];
                for (ksh = 0; ksh <= ish; ksh++)
                for (lsh = 0; lsh <= ksh; lsh++) {
                        di = CINTcgtos_spheric(ish, bas);
                        dj = CINTcgtos_spheric(jsh, bas);
                        dk = CINTcgtos_spheric(ksh, bas);
                        dl = CINTcgtos_spheric(lsh, bas);
                        fijkl = malloc(sizeof(double) * di*dj*dk*dl);
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ksh;
                        shls[3] = lsh;
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
                        }
                        free(fijkl);
                }
        }
        free(ishls);
        free(jshls);
        CINTdel_2e_optimizer(&opt);
}
