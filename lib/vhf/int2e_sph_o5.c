/*
 * File: int2e_sph_o5.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "cint.h"
#include "misc.h"
#include "optimizer.h"
#include "nr_vhf_direct.h"

//#define ERI_CUTOFF 1e-14
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define LOWERTRI_INDEX(I,J)     ((I) > (J) ? ((I)*((I)+1)/2+(J)) : ((J)*((J)+1)/2+(I)))

struct _VHFEnvs {
        int natm;
        int nbas;
        const int *atm;
        const int *bas;
        const double *env;
        int nao;
        int *ao_loc;
        int *idx_tri;
};

static void store_ij(double *eri, int ish, int jsh, struct _VHFEnvs *envs,
                     CINTOpt *opt, CVHFOpt *vhfopt)
{
        const int nao = envs->nao;
        const int di = CINTcgto_spheric(ish, envs->bas);
        const int dj = CINTcgto_spheric(jsh, envs->bas);
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);
        const int *ao_loc = envs->ao_loc;
        const int *idx_tri = envs->idx_tri;
        int i, j, i0, j0, ij, kl0;
        unsigned long ij0;
        double *peri;

        if (CVHFfill_nr_eri_o2(eribuf, ish, jsh, ish+1,
                               envs->atm, envs->natm, envs->bas, envs->nbas,
                               envs->env, opt, vhfopt)) {
                for (i0 = ao_loc[ish], i = 0; i < di; i++,i0++) {
                for (j0 = ao_loc[jsh], j = 0; j < dj; j++,j0++) {
                if (i0 >= j0) {
                        ij = j * di + i;
                        ij0 = i0*(i0+1)/2 + j0;
                        peri = eri + ij0*(ij0+1)/2;
                        for (kl0 = 0; kl0 <= ij0; kl0++) {
                                peri[kl0] = eribuf[idx_tri[kl0]*di*dj+ij];
                        }
                } } }
        } else {
                for (i0 = ao_loc[ish], i = 0; i < di; i++,i0++) {
                for (j0 = ao_loc[jsh], j = 0; j < dj; j++,j0++) {
                if (i0 >= j0) {
                        ij0 = i0*(i0+1)/2 + j0;
                        peri = eri + ij0*(ij0+1)/2;
                        memset(peri, 0, sizeof(double)*ij0);
                } } }
        }

        free(eribuf);
}

void int2e_sph_o5(double *eri, const int *atm, const int natm,
                  const int *bas, const int nbas, const double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        int *ij2i = malloc(sizeof(int)*nbas*nbas);
        int *idx_tri = malloc(sizeof(int)*nao*nao);
        CVHFset_ij2i(ij2i, nbas);
        int *ao_loc = malloc(sizeof(int)*nbas);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        CVHFindex_blocks2tri(idx_tri, ao_loc, bas, nbas);

        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao,
                                ao_loc, idx_tri};
        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, atm, natm, bas, nbas, env);
        vhfopt->fprescreen = &CVHFnr_schwarz_cond;

        int i, j, ij;

#pragma omp parallel default(none) \
        shared(eri, ij2i, envs, opt, vhfopt) \
        private(ij, i, j)
#pragma omp for nowait schedule(guided)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                i = ij2i[ij];
                j = ij - (i*(i+1)/2);
                store_ij(eri, i, j, &envs, opt, vhfopt);
        }

        free(idx_tri);
        free(ij2i);
        free(ao_loc);
        CVHFdel_optimizer(&vhfopt);
        CINTdel_optimizer(&opt);
}
