/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"

void RInr_fill2c2e_sph(double *eri, int auxstart, int auxcount,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nbasnaux = auxstart + auxcount;
        int *ao_loc = malloc(sizeof(int)*(nbasnaux+1));
        CINTshells_spheric_offset(ao_loc, bas, nbasnaux);
        ao_loc[nbasnaux] = ao_loc[nbasnaux-1] + CINTcgto_spheric(nbasnaux-1, bas);
        const int naoaux = ao_loc[nbasnaux] - ao_loc[auxstart];
        double *buf;

        int ish, jsh, di, dj;
        int i, j, i0, j0;
        int shls[2];
        CINTOpt *cintopt = NULL;
        cint2c2e_sph_optimizer(&cintopt, atm, natm, bas, nbas, env);

#pragma omp parallel default(none) \
        shared(eri, auxstart, auxcount, atm, natm, bas, nbas, env, \
               ao_loc, cintopt) \
        private(ish, jsh, di, dj, i, j, i0, j0, shls, buf)
#pragma omp for nowait schedule(dynamic)
        for (ish = auxstart; ish < nbasnaux; ish++) {
        for (jsh = auxstart; jsh <= ish; jsh++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                buf = (double *)malloc(sizeof(double) * di * dj);
                if (cint2c2e_sph(buf, shls, atm, natm, bas, nbas, env,
                                 cintopt)) {
                        for (i0 = ao_loc[ish]-ao_loc[auxstart], i = 0; i < di; i++, i0++) {
                        for (j0 = ao_loc[jsh]-ao_loc[auxstart], j = 0; j < dj; j++, j0++) {
                                eri[i0*naoaux+j0] = buf[j*di+i];
                        } }
                }
                free(buf);
        } }

        for (i = 0; i < naoaux; i++) {
                for (j = 0; j < i; j++) {
                        eri[j*naoaux+i] = eri[i*naoaux+j];
                }
        }

        free(ao_loc);
        CINTdel_optimizer(&cintopt);
}


