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


#define NCTRMAX         64

static void unpack_block2tril(double *buf, double *eri,
                              int ish, int jsh, int dkl, size_t nao2, int *ao_loc)
{
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

/*
 * 8-fold symmetry, k>=l, k>=i>=j, 
 */
static void fillnr_s8(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                      double *eri, int ksh, int lsh,
                      CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ksh < lsh) {
                return;
        }
        const int *ao_loc = envs->ao_loc;
        const size_t nao2 = ao_loc[ksh+1] * (ao_loc[ksh+1]+1) / 2;
        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int ish, jsh, di, dj;
        int shls[4];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*dk*dl);

        shls[2] = ksh;
        shls[3] = lsh;

        for (ish = 0; ish < ksh+1; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)) {
                        (*intor)(buf, shls, envs->atm, envs->natm,
                                 envs->bas, envs->nbas, envs->env, cintopt);
                } else {
                        memset(buf, 0, sizeof(double)*di*dj*dk*dl);
                }
                (*funpack)(buf, eri, ish, jsh, dk*dl, nao2, ao_loc);
        } }

        free(buf);
}

static void store_ij(int (*intor)(), double *eri, int ish, int jsh,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int *ao_loc = envs->ao_loc;
        const size_t nao2 = ao_loc[ish+1] * (ao_loc[ish+1]+1) / 2;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*nao2);
        int i, j, i0, j0, kl;
        size_t ij0;
        double *peri, *pbuf;

        fillnr_s8(intor, unpack_block2tril, vhfopt->fprescreen,
                  eribuf, ish, jsh, cintopt, vhfopt, envs);
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

void GTO2e_cart_or_sph(int (*intor)(), int (*cgto_in_shell)(), double *eri,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int *ao_loc = malloc(sizeof(int)*(nbas+1));
        int i, j, ij;
        int nao = 0;
        for (i = 0; i < nbas; i++) {
                ao_loc[i] = nao;
                nao += (*cgto_in_shell)(i, bas);
        }
        ao_loc[nbas] = nao;

        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao,
                                ao_loc};
        CINTOpt *cintopt;
        cint2e_optimizer(&cintopt, atm, natm, bas, nbas, env);
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, atm, natm, bas, nbas, env);
        vhfopt->fprescreen = CVHFnr_schwarz_cond;

#pragma omp parallel default(none) \
        shared(intor, eri, nbas, envs, cintopt, vhfopt) \
        private(ij, i, j)
#pragma omp for nowait schedule(dynamic, 2)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - (i*(i+1)/2);
                store_ij(intor, eri, i, j, cintopt, vhfopt, &envs);
        }

        free(ao_loc);
        CVHFdel_optimizer(&vhfopt);
        CINTdel_optimizer(&cintopt);
}

void int2e_sph(double *eri,
               int *atm, int natm, int *bas, int nbas, double *env)
{
        GTO2e_cart_or_sph(cint2e_sph, CINTcgto_spheric, eri,
                          atm, natm, bas, nbas, env);
}

