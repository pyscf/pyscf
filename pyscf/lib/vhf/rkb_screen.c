/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include "cint.h"
#include "cvhf.h"
#include "optimizer.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

#define LL 0
#define SS 1
#define SL 2
#define LS 3


int int2e_spinor();
int int2e_spsp1spsp2_spinor();

int CVHFrkbllll_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((opt->dm_cond[j*n+i] > dmin)
            || (opt->dm_cond[l*n+k] > dmin)
            || (opt->dm_cond[j*n+k] > dmin)
            || (opt->dm_cond[j*n+l] > dmin)
            || (opt->dm_cond[i*n+k] > dmin)
            || (opt->dm_cond[i*n+l] > dmin));
}

int CVHFrkbllll_vkscreen(int *shls, CVHFOpt *opt,
                         double **dms_cond, int n_dm, double *dm_atleast,
                         int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int nbas = opt->nbas;
        int idm;
        double qijkl = opt->q_cond[i*nbas+j] * opt->q_cond[k*nbas+l];
        double *pdmscond = opt->dm_cond + nbas*nbas;
        for (idm = 0; idm < (n_dm+1)/2; idm++) {
// note in _vhf.rdirect_mapdm, J and K share the same DM
                dms_cond[idm*2+0] = pdmscond + idm*nbas*nbas; // for vj
                dms_cond[idm*2+1] = pdmscond + idm*nbas*nbas; // for vk
        }
        *dm_atleast = opt->direct_scf_cutoff / qijkl;
        return 1;
}

int CVHFrkbssll_prescreen(int *shls, CVHFOpt *opt,
                          int *atm, int *bas, double *env)
{
        if (opt == NULL) {
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
        double dmin = opt->direct_scf_cutoff / qijkl;
        return qijkl > opt->direct_scf_cutoff
            &&((opt->dm_cond[n*n*SS+j*n+i] > dmin)
            || (opt->dm_cond[l*n+k] > dmin)
            || (dmsl[j*n+k] > dmin)
            || (dmsl[j*n+l] > dmin)
            || (dmsl[i*n+k] > dmin)
            || (dmsl[i*n+l] > dmin));
}

// be careful with the order in dms_cond, the current order (dmll, dmss, dmsl)
// is consistent to the function _call_veff_ssll in dhf.py
int CVHFrkbssll_vkscreen(int *shls, CVHFOpt *opt,
                         double **dms_cond, int n_dm, double *dm_atleast,
                         int *atm, int *bas, double *env)
{
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int nbas = opt->nbas;
        int idm;
        double qijkl = opt->q_cond[nbas*nbas*SS+i*nbas+j] * opt->q_cond[k*nbas+l];
        double *pdmscond = opt->dm_cond + 4*nbas*nbas;
        int nset = (n_dm+2) / 3;
        double *dmscondll = pdmscond + nset*nbas*nbas*LL;
        double *dmscondss = pdmscond + nset*nbas*nbas*SS;
        double *dmscondsl = pdmscond + nset*nbas*nbas*SL;
        for (idm = 0; idm < nset; idm++) {
                dms_cond[nset*0+idm] = dmscondll + idm*nbas*nbas;
                dms_cond[nset*1+idm] = dmscondss + idm*nbas*nbas;
                dms_cond[nset*2+idm] = dmscondsl + idm*nbas*nbas;
        }
        *dm_atleast = opt->direct_scf_cutoff / qijkl;
        return 1;
}


static void set_qcond(int (*intor)(), CINTOpt *cintopt, double *qcond,
                      int *ao_loc, int *atm, int natm,
                      int *bas, int nbas, double *env)
{
        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
#pragma omp parallel
{
        double qtmp, tmp;
        int i, j, ij, di, dj, ish, jsh;
        int shls[4];
        double *cache = malloc(sizeof(double) * cache_size);
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double complex *buf = malloc(sizeof(double complex) * di*di*di*di);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                jsh = ij - ish*(ish+1)/2;
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ish;
                shls[3] = jsh;
                qtmp = 1e-100;
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, cache)) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                tmp = cabs(buf[i+di*j+di*dj*i+di*dj*di*j]);
                                qtmp = MAX(qtmp, tmp);
                        } }
                        qtmp = sqrt(qtmp);
                }
                qcond[ish*nbas+jsh] = qtmp;
                qcond[jsh*nbas+ish] = qtmp;
        }
        free(buf);
        free(cache);
}
}

void CVHFrkbllll_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);

        assert(intor == &int2e_spinor);
        set_qcond(intor, cintopt, opt->q_cond, ao_loc, atm, natm, bas, nbas, env);
}

void CVHFrkbssss_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas);

        assert(intor == &int2e_spsp1spsp2_spinor);
        set_qcond(intor, cintopt, opt->q_cond, ao_loc, atm, natm, bas, nbas, env);
}


void CVHFrkbssll_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        opt->q_cond = (double *)malloc(sizeof(double) * nbas*nbas*2);

        set_qcond(&int2e_spinor, NULL, opt->q_cond, ao_loc, atm, natm, bas, nbas, env);
        set_qcond(&int2e_spsp1spsp2_spinor, NULL, opt->q_cond+nbas*nbas, ao_loc,
                  atm, natm, bas, nbas, env);
}

static void set_dmcond(double *dmcond, double *dmscond, double complex *dm,
                       double direct_scf_cutoff, int nset, int *ao_loc,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        double dmax, dmaxi, tmp;
        int i, j, ish, jsh;
        int iset;
        double complex *pdm;

        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        dmaxi = 0;
                        pdm = dm + nao*nao*iset;
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                tmp = cabs(pdm[i*nao+j]) + cabs(pdm[j*nao+i]);
                                dmaxi = MAX(dmaxi, tmp);
                        } }
                        dmscond[iset*nbas*nbas+ish*nbas+jsh] = .5 * dmaxi;
                        dmscond[iset*nbas*nbas+jsh*nbas+ish] = .5 * dmaxi;
                        dmax = MAX(dmax, dmaxi);
                }
                dmcond[ish*nbas+jsh] = .5 * dmax;
                dmcond[jsh*nbas+ish] = .5 * dmax;
        } }
}

//  dm_cond ~ 1+nset, dm_cond + dms_cond
void CVHFrkbllll_direct_scf_dm(CVHFOpt *opt, double complex *dm, int nset,
                               int *ao_loc, int *atm, int natm,
                               int *bas, int nbas, double *env)
{
        if (opt->dm_cond != NULL) { // NOT reuse opt->dm_cond because nset may be diff in different call
                free(opt->dm_cond);
        }
        opt->dm_cond = (double *)malloc(sizeof(double)*nbas*nbas*(1+nset));
        NPdset0(opt->dm_cond, ((size_t)nbas)*nbas*(1+nset));
        // dmcond followed by dmscond which are max matrix element for each dm
        set_dmcond(opt->dm_cond, opt->dm_cond+nbas*nbas, dm,
                   opt->direct_scf_cutoff, nset, ao_loc, atm, natm, bas, nbas, env);
}

void CVHFrkbssss_direct_scf_dm(CVHFOpt *opt, double complex *dm, int nset,
                               int *ao_loc, int *atm, int natm,
                               int *bas, int nbas, double *env)
{
        if (opt->dm_cond != NULL) {
                free(opt->dm_cond);
        }
        opt->dm_cond = (double *)malloc(sizeof(double)*nbas*nbas*(1+nset));
        NPdset0(opt->dm_cond, ((size_t)nbas)*nbas*(1+nset));
        set_dmcond(opt->dm_cond, opt->dm_cond+nbas*nbas, dm,
                   opt->direct_scf_cutoff, nset, ao_loc, atm, natm, bas, nbas, env);
}

// the current order of dmscond (dmll, dmss, dmsl) is consistent to the
// function _call_veff_ssll in dhf.py
void CVHFrkbssll_direct_scf_dm(CVHFOpt *opt, double complex *dm, int nset,
                               int *ao_loc, int *atm, int natm,
                               int *bas, int nbas, double *env)
{
        if (opt->dm_cond != NULL) {
                free(opt->dm_cond);
        }
        if (nset < 4) {
                fprintf(stderr, "4 sets of DMs (dmll,dmss,dmsl,dmls) are "
                        "required to set rkb prescreening\n");
                exit(1);
        }
        nset = nset / 4;
        int n2c = CINTtot_cgto_spinor(bas, nbas);
        size_t nbas2 = nbas * nbas;
        opt->dm_cond = (double *)malloc(sizeof(double)*nbas2*4*(1+nset));
        NPdset0(opt->dm_cond, nbas2*4*(1+nset));

        // 4 types of dmcond (LL,SS,SL,LS) followed by 4 types of dmscond
        double *dmcondll = opt->dm_cond + nbas2*LL;
        double *dmcondss = opt->dm_cond + nbas2*SS;
        double *dmcondsl = opt->dm_cond + nbas2*SL;
        double *dmcondls = opt->dm_cond + nbas2*LS;
        double *pdmscond = opt->dm_cond + nbas2*4;
        double *dmscondll = pdmscond + nset*nbas2*LL;
        double *dmscondss = pdmscond + nset*nbas2*SS;
        double *dmscondsl = pdmscond + nset*nbas2*SL;
        double *dmscondls = pdmscond + nset*nbas2*LS;
        double complex *dmll = dm + n2c*n2c*LL*nset;
        double complex *dmss = dm + n2c*n2c*SS*nset;
        double complex *dmsl = dm + n2c*n2c*SL*nset;
        double complex *dmls = dm + n2c*n2c*LS*nset;

        set_dmcond(dmcondll, dmscondll, dmll,
                   opt->direct_scf_cutoff, nset, ao_loc, atm, natm, bas, nbas, env);
        set_dmcond(dmcondss, dmscondss, dmss,
                   opt->direct_scf_cutoff, nset, ao_loc, atm, natm, bas, nbas, env);
        set_dmcond(dmcondsl, dmscondsl, dmsl,
                   opt->direct_scf_cutoff, nset, ao_loc, atm, natm, bas, nbas, env);
        set_dmcond(dmcondls, dmscondls, dmls,
                   opt->direct_scf_cutoff, nset, ao_loc, atm, natm, bas, nbas, env);

        // aggregate dmcondls to dmcondsl
        int i, j, n;
        for (i = 0; i < nbas; i++) {
        for (j = 0; j < nbas; j++) {
                dmcondsl[i*nbas+j] = MAX(dmcondsl[i*nbas+j], dmcondls[j*nbas+i]);
        } }
        for (n = 0; n < nset; n++) {
                for (i = 0; i < nbas; i++) {
                for (j = 0; j < nbas; j++) {
                        dmscondsl[i*nbas+j] = MAX(dmscondsl[i*nbas+j], dmscondls[j*nbas+i]);
                } }
                dmscondsl += nbas2;
                dmscondls += nbas2;
        }
}
