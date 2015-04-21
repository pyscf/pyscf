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


#define MAXCGTO         64

static void unpack_block2tril(double *buf, double *eri,
                              int ish, int jsh, int dkl, int nao, int *ao_loc)
{
        size_t nao2 = nao * nao;
        int iloc = ao_loc[ish];
        int jloc = ao_loc[jsh];
        int di = ao_loc[ish+1] - iloc;
        int dj = ao_loc[jsh+1] - jloc;
        int i, j, kl;
        eri += iloc*(iloc+1)/2 + jloc;
        double *eri0 = eri;

        if (ish > jsh) {
                for (kl = 0; kl < dkl; kl++) {
                        eri0 = eri + nao2 * kl;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        eri0[j] = buf[j*di+i];
                                }
                                eri0 += ao_loc[ish] + i + 1;
                        }
                        buf += di*dj;
                }
        } else { // ish == jsh
                for (kl = 0; kl < dkl; kl++) {
                        eri0 = eri + nao2 * kl;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        eri0[j] = buf[j*di+i];
                                }
                                // row ao_loc[ish]+i has ao_loc[ish]+i+1 elements
                                eri0 += ao_loc[ish] + i + 1;
                        }
                        buf += di*dj;
                }
        }
}

static int fill_s2(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ncomp, int ksh, int lsh, int ish_count,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int *ao_loc = envs->ao_loc;
        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int ish, jsh, di, dj;
        int empty = 1;
        int shls[4];
        double *buf = malloc(sizeof(double)*MAXCGTO*MAXCGTO*dk*dl*ncomp);

        shls[2] = ksh;
        shls[3] = lsh;

        for (ish = 0; ish < ish_count; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)) {
                        empty = !(*intor)(buf, shls, envs->atm, envs->natm,
                                          envs->bas, envs->nbas, envs->env,
                                          cintopt)
                                && empty;
                } else {
                        memset(buf, 0, sizeof(double)*di*dj*dk*dl*ncomp);
                }
                (*funpack)(buf, eri, ish, jsh, dk*dl*ncomp, nao, ao_loc);
        } }

        free(buf);
        return !empty;
}
static int fillnr_s8(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ksh, int lsh,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ksh >= lsh) {
                // 8-fold symmetry, k>=l, k>=i>=j, 
                return fill_s2(intor, funpack, fprescreen, eri, ncomp,
                               ksh, lsh, ksh+1, cintopt, vhfopt, envs);
        } else {
                return 0;
        }
}

static void store_ij(double *eri, int ish, int jsh,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);
        int i, j, i0, j0, kl;
        size_t ij0;
        double *peri, *pbuf;

        fillnr_s8(cint2e_sph, unpack_block2tril, vhfopt->fprescreen,
                  eribuf, 1, ish, jsh, cintopt, vhfopt, envs);
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++) {
        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                if (i0 >= j0) {
                        ij0 = i0*(i0+1)/2 + j0;
                        peri = eri + ij0*(ij0+1)/2;
                        pbuf = eribuf + nao2 * (j*di+i);
                        for (kl = 0; kl <= ij0; kl++) {
                                peri[kl] = pbuf[kl];
                        }
                }
        } }

        free(eribuf);
}

void int2e_sph(double *eri, int *atm, int natm, int *bas, int nbas,double *env)
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
