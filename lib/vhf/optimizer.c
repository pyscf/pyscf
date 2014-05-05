/*
 *
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "cint.h"
#include "optimizer.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

void CVHFinit_optimizer(CVHFOpt **opt, const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env)
{
        CVHFOpt *opt0 = (CVHFOpt *)malloc(sizeof(CVHFOpt));
        opt0->nbas = nbas;
        opt0->direct_scf_cutoff = 1e-13;
        opt0->q_cond = NULL;
        opt0->dm_cond = NULL;
        *opt = opt0;
}

void CVHFdel_optimizer(CVHFOpt **opt)
{
        CVHFOpt *opt0 = *opt;
        if (!opt0) {
                return;
        }

        if (!opt0->q_cond) {
                free(opt0->q_cond);
                opt0->q_cond = NULL;
        }
        if (!opt0->dm_cond) {
                free(opt0->dm_cond);
                opt0->dm_cond = NULL;
        }

        free(opt0);
        *opt = NULL;
}

void CVHFset_direct_scf(CVHFOpt *opt, const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env)
{
        if (!opt->q_cond) {
                opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        }

        double *buf;
        double qtmp;
        unsigned int i, j, di, dj, ish, jsh;
        unsigned int shls[4];
        for (ish = 0; ish < nbas; ish++) {
                di = CINTcgto_spheric(ish, bas);
                for (jsh = 0; jsh <= ish; jsh++) {
                        dj = CINTcgto_spheric(jsh, bas);
                        buf = (double *)malloc(sizeof(double) * di*dj*di*dj);
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ish;
                        shls[3] = jsh;
                        qtmp = 0;
                        if (0 != cint2e_sph(buf, shls, atm, natm, bas, nbas, env, NULL)) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        qtmp = MAX(qtmp, fabs(buf[i+di*j+di*dj*i+di*dj*di*j]));
                                } }
                        }
                        qtmp = sqrt(qtmp);
                        opt->q_cond[ish*nbas+jsh] = qtmp;
                        opt->q_cond[jsh*nbas+ish] = qtmp;
                        free(buf);

                }
        }

        if (!opt->dm_cond) {
                opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        }
}

void CVHFset_direct_scf_dm(CVHFOpt *opt, double *dm, const int nset,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env)
{
        int *ao_loc = malloc(sizeof(unsigned int) * nbas);
        unsigned int nao = CINTtot_cgto_spheric(bas, nbas);
        CINTshells_spheric_offset(ao_loc, bas, nbas);

        double dmax;
        unsigned int i, j, di, dj, ish, jsh, iloc, jloc;
        unsigned int iset;
        double *pdm;
        for (ish = 0; ish < nbas; ish++) {
                iloc = ao_loc[ish];
                di = CINTcgto_spheric(ish, bas);
                for (jsh = 0; jsh < nbas; jsh++) {
                        jloc = ao_loc[jsh];
                        dj = CINTcgto_spheric(jsh, bas);
                        dmax = 0;
                        for (iset = 0; iset < nset; iset++) {
                                pdm = dm + nao*nao*iset;
                                for (i = iloc; i < iloc+di; i++) {
                                for (j = jloc; j < jloc+dj; j++) {
                                        dmax = MAX(dmax, fabs(pdm[i*nao+j]));
                                } }
                        }
                        opt->dm_cond[ish*nbas+jsh] = dmax * .25l;
                }
        }
        free(ao_loc);
}

