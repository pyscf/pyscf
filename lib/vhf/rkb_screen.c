/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include "cint.h"
#include "cvhf.h"
#include "optimizer.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

#define LL 0
#define SS 1
#define SL 2
#define LS 3


int cint2e();
int cint2e_spsp1spsp2();

int CVHFrkbllll_prescreen(int *shls, CVHFOpt *opt,
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
        return (opt->dm_cond[j*n+i] > dmin)
             | (opt->dm_cond[l*n+k] > dmin)
             | (opt->dm_cond[j*n+k] > dmin)
             | (opt->dm_cond[j*n+l] > dmin)
             | (opt->dm_cond[i*n+k] > dmin)
             | (opt->dm_cond[i*n+l] > dmin);
}

int CVHFrkbssll_prescreen(int *shls, CVHFOpt *opt,
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
        double *dmsl = opt->dm_cond + n*n*SL;
        double qijkl = opt->q_cond[n*n*SS+i*n+j] * opt->q_cond[k*n+l];
        double dmin = opt->direct_scf_cutoff * qijkl;
        return (opt->dm_cond[n*n*SS+j*n+i] > dmin)
             | (opt->dm_cond[l*n+k] > dmin)
             | (dmsl[j*n+k] > dmin)
             | (dmsl[j*n+l] > dmin)
             | (dmsl[i*n+k] > dmin)
             | (dmsl[i*n+l] > dmin);
}


static void set_qcond(int (*intor)(), double *qcond,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        double complex *buf;
        double qtmp;
        int i, j, di, dj, ish, jsh;
        int shls[4];
        for (ish = 0; ish < nbas; ish++) {
                di = CINTcgto_spinor(ish, bas);
                for (jsh = 0; jsh <= ish; jsh++) {
                        dj = CINTcgto_spinor(jsh, bas);
                        buf = malloc(sizeof(double complex) * di*dj*di*dj);
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ish;
                        shls[3] = jsh;
                        qtmp = 0;
                        if (0 != (*intor)(buf, shls, atm, natm, bas, nbas, env, NULL)) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        qtmp = MAX(qtmp, cabs(buf[i+di*j+di*dj*i+di*dj*di*j]));
                                } }
                        }
                        qtmp = 1./sqrt(qtmp);
                        qcond[ish*nbas+jsh] = qtmp;
                        qcond[jsh*nbas+ish] = qtmp;
                        free(buf);

                }
        }
}

void CVHFrkbllll_direct_scf(CVHFOpt *opt, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (!opt->q_cond) {
                opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        }

        set_qcond(cint2e, opt->q_cond, atm, natm, bas, nbas, env);

        if (!opt->dm_cond) {
                opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
                memset(opt->dm_cond, 0, sizeof(double)*nbas*nbas);
        }
}

void CVHFrkbssss_direct_scf(CVHFOpt *opt, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (!opt->q_cond) {
                opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        }

        const int INC1 = 1;
        int nn = nbas * nbas;
        double c1 = .25 / (env[PTR_LIGHT_SPEED]*env[PTR_LIGHT_SPEED]);
        set_qcond(cint2e_spsp1spsp2, opt->q_cond, atm, natm, bas, nbas, env);
        dscal_(&nn, &c1, opt->q_cond, &INC1);

        if (!opt->dm_cond) {
                opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas);
                memset(opt->dm_cond, 0, sizeof(double)*nbas*nbas);
        }
}


void CVHFrkbssll_direct_scf(CVHFOpt *opt, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (!opt->q_cond) {
                opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas*2);
        }

        const int INC1 = 1;
        int nn = nbas * nbas;
        double c1 = .25 / (env[PTR_LIGHT_SPEED]*env[PTR_LIGHT_SPEED]);
        set_qcond(cint2e, opt->q_cond, atm, natm, bas, nbas, env);
        set_qcond(cint2e_spsp1spsp2, opt->q_cond+nbas*nbas,
                  atm, natm, bas, nbas, env);
        dscal_(&nn, &c1, opt->q_cond+nbas*nbas, &INC1);

        if (!opt->dm_cond) {
                opt->dm_cond = (double *)malloc(sizeof(double) * nbas*nbas*4);
                memset(opt->dm_cond, 0, sizeof(double)*nbas*nbas*4);
        }
}

static void set_dmcond(double *dmcond, double complex *dm,
                       double direct_scf_cutoff, int nset,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int *ao_loc = malloc(sizeof(int) * (nbas+1));
        CINTshells_spinor_offset(ao_loc, bas, nbas);
        ao_loc[nbas] = ao_loc[nbas-1] + CINTcgto_spinor(nbas-1, bas);
        int nao = ao_loc[nbas];

        double dmax;
        int i, j, ish, jsh;
        int iset;
        double complex *pdm;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh < nbas; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        pdm = dm + nao*nao*iset;
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dmax = MAX(dmax, cabs(pdm[i*nao+j]));
                        } }
                }
                dmcond[ish*nbas+jsh] = dmax;
        } }
        free(ao_loc);
}


void CVHFrkbllll_direct_scf_dm(CVHFOpt *opt, double complex *dm, int nset,
                               int *atm, int natm, int *bas, int nbas, double *env)
{
        set_dmcond(opt->dm_cond, dm, opt->direct_scf_cutoff, nset,
                   atm, natm, bas, nbas, env);
}

void CVHFrkbssss_direct_scf_dm(CVHFOpt *opt, double complex *dm, int nset,
                               int *atm, int natm, int *bas, int nbas,
                               double *env)
{
        set_dmcond(opt->dm_cond, dm, opt->direct_scf_cutoff, nset,
                   atm, natm, bas, nbas, env);
}

void CVHFrkbssll_direct_scf_dm(CVHFOpt *opt, double complex *dm, int nset,
                               int *atm, int natm, int *bas, int nbas,
                               double *env)
{
        int n2c = CINTtot_cgto_spinor(bas, nbas);
        double *dmcondll = opt->dm_cond + nbas*nbas*LL;
        double *dmcondss = opt->dm_cond + nbas*nbas*SS;
        double *dmcondsl = opt->dm_cond + nbas*nbas*SL;
        //double *dmcondls = opt->dm_cond + nbas*nbas*LS;
        nset = nset / 3;
        double complex *dmll = dm + n2c*n2c*LL*nset;
        double complex *dmss = dm + n2c*n2c*SS*nset;
        double complex *dmsl = dm + n2c*n2c*SL*nset;
        //double complex *dmls = dm + n2c*n2c*LS*nset;

        set_dmcond(dmcondll, dmll, opt->direct_scf_cutoff, nset,
                   atm, natm, bas, nbas, env);
        set_dmcond(dmcondss, dmss, opt->direct_scf_cutoff, nset,
                   atm, natm, bas, nbas, env);
        set_dmcond(dmcondsl, dmsl, opt->direct_scf_cutoff, nset,
                   atm, natm, bas, nbas, env);
}
