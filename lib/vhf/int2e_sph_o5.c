/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cvhf.h"
#include "nr_direct.h"
#include "optimizer.h"

static void store_ij(double *eri, int ish, int jsh,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int *ao_loc = envs->ao_loc;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);
        int i, j, i0, j0, kl;
        unsigned long ij0;
        double *peri, *pbuf;

        CVHFfill_nr_s8(cint2e_sph, CVHFunpack_nrblock2tril, vhfopt->fprescreen,
                       eribuf, 1, ish, jsh, cintopt, vhfopt, envs);
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++) {
        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                if (i0 >= j0) {
                        ij0 = i0*(i0+1)/2 + j0;
                        peri = eri + ij0*(ij0+1)/2;
                        pbuf = eribuf + (j*di+i) * nao*nao;
                        for (kl = 0; kl <= ij0; kl++) {
                                peri[kl] = pbuf[kl];
                        }
                }
        } }

        free(eribuf);
}

void int2e_sph_o5(double *eri, int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        int *ao_loc = malloc(sizeof(int)*(nbas+1));
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        ao_loc[nbas] = nao;

        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao,
                                ao_loc};
        CINTOpt *cintopt;
        cint2e_optimizer(&cintopt, atm, natm, bas, nbas, env);
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, atm, natm, bas, nbas, env);
        vhfopt->fprescreen = CVHFnr_schwarz_cond;

        int i, j, ij;

#pragma omp parallel default(none) \
        shared(eri, nbas, envs, cintopt, vhfopt) \
        private(ij, i, j)
#pragma omp for nowait schedule(dynamic, 2)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - (i*(i+1)/2);
                store_ij(eri, i, j, cintopt, vhfopt, &envs);
        }

        free(ao_loc);
        CVHFdel_optimizer(&vhfopt);
        CINTdel_optimizer(&cintopt);
}
