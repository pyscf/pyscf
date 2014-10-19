/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "cint.h"
#include "cvhf.h"
#include "optimizer.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))


void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        CVHFOpt *opt0 = (CVHFOpt *)malloc(sizeof(CVHFOpt));
        opt0->nbas = nbas;
        opt0->direct_scf_cutoff = 1e-14;
        opt0->q_cond = NULL;
        opt0->dm_cond = NULL;
        opt0->fprescreen = &CVHFnoscreen;
        opt0->r_vkscreen = &CVHFr_vknoscreen;
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

int CVHFnoscreen(int *shls, CVHFOpt *opt,
                 int *atm, int *bas, double *env)
{
        return 1;
}

int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt,
                        int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        return qijkl > opt->direct_scf_cutoff;
}

int CVHFnrs8_prescreen(int *shls, CVHFOpt *opt,
                       int *atm, int *bas, double *env)
{
        if (!opt) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff * qijkl;
        return (4*opt->dm_cond[j*n+i] > dmin)
             | (4*opt->dm_cond[l*n+k] > dmin)
             | (  opt->dm_cond[j*n+k] > dmin)
             | (  opt->dm_cond[j*n+l] > dmin)
             | (  opt->dm_cond[i*n+k] > dmin)
             | (  opt->dm_cond[i*n+l] > dmin);
}

// return flag to decide whether transpose01324
int CVHFr_vknoscreen(int *shls, CVHFOpt *opt,
                     double **dms_cond, int n_dm, double *dm_atleast,
                     int *atm, int *bas, double *env)
{
        int idm;
        for (idm = 0; idm < n_dm; idm++) {
                dms_cond[idm] = NULL;
        }
        *dm_atleast = 0;
        return 1;
}

void CVHFset_direct_scf_cutoff(CVHFOpt *opt, double cutoff)
{
        opt->direct_scf_cutoff = cutoff;
}
double CVHFget_direct_scf_cutoff(CVHFOpt *opt)
{
        return opt->direct_scf_cutoff;
}


void CVHFsetnr_direct_scf(CVHFOpt *opt, int *atm, int natm,
                          int *bas, int nbas, double *env)
{
        /* This memory is released in void CVHFdel_optimizer, Don't know
         * why valgrind raises memory leak here */
        if (!opt->q_cond) {
                opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        }

        double *buf;
        double qtmp;
        int i, j, di, dj, ish, jsh;
        int shls[4];
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
                        qtmp = 1./sqrt(qtmp);
                        opt->q_cond[ish*nbas+jsh] = qtmp;
                        opt->q_cond[jsh*nbas+ish] = qtmp;
                        free(buf);

                }
        }

        if (!opt->dm_cond) {
                opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
                memset(opt->dm_cond, 0, sizeof(double)*nbas*nbas);
        }
}

void CVHFsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        int *ao_loc = malloc(sizeof(int) * (nbas+1));
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        ao_loc[nbas] = ao_loc[nbas-1] + CINTcgto_spheric(nbas-1, bas);
        int nao = ao_loc[nbas];

        double dmax;
        int i, j, ish, jsh;
        int iset;
        double *pdm;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh < nbas; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        pdm = dm + nao*nao*iset;
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dmax = MAX(dmax, fabs(pdm[i*nao+j]));
                        } }
                }
                opt->dm_cond[ish*nbas+jsh] = dmax;
        } }
        free(ao_loc);
}



/*
 *************************************************
 */
void CVHFnr_optimizer(CVHFOpt **vhfopt, int *atm, int natm,
                      int *bas, int nbas, double *env)
{
        CVHFinit_optimizer(vhfopt, atm, natm, bas, nbas, env);
        (*vhfopt)->fprescreen = &CVHFnrs8_prescreen;
        CVHFsetnr_direct_scf(*vhfopt, atm, natm, bas, nbas, env);
}
