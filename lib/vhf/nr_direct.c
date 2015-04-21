/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "nr_direct.h"

// 9f or 7g or 5h functions should be enough
#define MAXCGTO         64


/*
 * for given ksh, lsh, loop all ish, jsh
 */
void CVHFdot_nrs1(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                 int n_dm, int ncomp, int ish, int jsh,
                 CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int i0 = ao_loc[ish];
        const int j0 = ao_loc[jsh];
        const int i1 = ao_loc[ish+1];
        const int j1 = ao_loc[jsh+1];
        const int di = i1 - i0;
        const int dj = j1 - j0;
        int idm, icomp;
        int ksh, lsh, k0, k1, l0, l1, dk, dl, dijkl;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*MAXCGTO*MAXCGTO*ncomp);
        double *pv;
        void (*pf)();
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < envs->nbas; ksh++) {
        for (lsh = 0; lsh < envs->nbas; lsh++) {
                k0 = ao_loc[ksh];
                l0 = ao_loc[lsh];
                k1 = ao_loc[ksh+1];
                l1 = ao_loc[lsh+1];
                dk = k1 - k0;
                dl = l1 - l0;
                shls[2] = ksh;
                shls[3] = lsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)
                    && (*intor)(buf, shls, envs->atm, envs->natm,
                                envs->bas, envs->nbas, envs->env,
                                cintopt)) {
                        dijkl = di * dj * dk * dl;
                        pv = vjk;
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = fjk[idm];
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                        (*pf)(buf+dijkl*icomp, dms[idm], pv,
                                              i0, i1, j0, j1, k0, k1, l0, l1, nao);
                                        pv += nao2;
                                }
                        }
                }
        } }
        free(buf);
}

/*
 * for given ish, jsh, loop all ksh > lsh
 */
static void dot_nrs2sub(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                        int n_dm, int ncomp, int ish, int jsh,
                        CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int i0 = ao_loc[ish];
        const int j0 = ao_loc[jsh];
        const int i1 = ao_loc[ish+1];
        const int j1 = ao_loc[jsh+1];
        const int di = i1 - i0;
        const int dj = j1 - j0;
        int idm, icomp;
        int ksh, lsh, k0, k1, l0, l1, dk, dl, dijkl;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*MAXCGTO*MAXCGTO*ncomp);
        double *pv;
        void (*pf)();
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < envs->nbas; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                k0 = ao_loc[ksh];
                l0 = ao_loc[lsh];
                k1 = ao_loc[ksh+1];
                l1 = ao_loc[lsh+1];
                dk = k1 - k0;
                dl = l1 - l0;
                shls[2] = ksh;
                shls[3] = lsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)
                    && (*intor)(buf, shls, envs->atm, envs->natm,
                                envs->bas, envs->nbas, envs->env,
                                cintopt)) {
                        dijkl = di * dj * dk * dl;
                        pv = vjk;
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = fjk[idm];
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                        (*pf)(buf+dijkl*icomp, dms[idm], pv,
                                              i0, i1, j0, j1, k0, k1, l0, l1, nao);
                                        pv += nao2;
                                }
                        }
                }
        } }
        free(buf);
}

void CVHFdot_nrs2ij(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                    int n_dm, int ncomp, int ish, int jsh,
                    CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ish >= jsh) {
                CVHFdot_nrs1(intor, fjk, dms, vjk, n_dm, ncomp,
                             ish, jsh, cintopt, vhfopt, envs);
        }
}

void CVHFdot_nrs2kl(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                    int n_dm, int ncomp, int ish, int jsh,
                    CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        dot_nrs2sub(intor, fjk, dms, vjk, n_dm, ncomp,
                    ish, jsh, cintopt, vhfopt, envs);
}

void CVHFdot_nrs4(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                  int n_dm, int ncomp, int ish, int jsh,
                  CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ish >= jsh) {
                dot_nrs2sub(intor, fjk, dms, vjk, n_dm, ncomp,
                            ish, jsh, cintopt, vhfopt, envs);
        }
}

void CVHFdot_nrs8(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                  int n_dm, int ncomp, int ish, int jsh,
                  CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ish < jsh) {
                return;
        }
        const int nao = envs->nao;
        const int nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int i0 = ao_loc[ish];
        const int j0 = ao_loc[jsh];
        const int i1 = ao_loc[ish+1];
        const int j1 = ao_loc[jsh+1];
        const int di = i1 - i0;
        const int dj = j1 - j0;
        int idm, icomp;
        int ksh, lsh, k0, k1, l0, l1, dk, dl, dijkl;
        int shls[4];
        double *buf = malloc(sizeof(double)*di*dj*MAXCGTO*MAXCGTO*ncomp);
        double *pv;
        void (*pf)();
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

// to make fjk compatible to C-contiguous dm array, put ksh, lsh inner loop
        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh <= ish; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
/* when ksh==ish, (lsh<jsh) misses some integrals (eg k<i&&l>j).
 * These integrals are calculated in the next (ish,jsh) pair. To show
 * that, we just need to prove that every elements in shell^4 appeared
 * only once in fjk_s8.  */
                if ((ksh == ish) && (lsh > jsh)) {
                        break;
                }
                k0 = ao_loc[ksh];
                l0 = ao_loc[lsh];
                k1 = ao_loc[ksh+1];
                l1 = ao_loc[lsh+1];
                dk = k1 - k0;
                dl = l1 - l0;
                shls[2] = ksh;
                shls[3] = lsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)
                    && (*intor)(buf, shls, envs->atm, envs->natm,
                                envs->bas, envs->nbas, envs->env,
                                cintopt)) {
                        dijkl = di * dj * dk * dl;
                        pv = vjk;
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = fjk[idm];
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                        (*pf)(buf+dijkl*icomp, dms[idm], pv,
                                              i0, i1, j0, j1, k0, k1, l0, l1, nao);
                                        pv += nao2;
                                }
                        }
                }
        } }
        free(buf);
}


/*
 * drv loop over ij, generate eris of kl for given ij, call fjk to
 * calculate vj, vk.
 * 
 * n_dm is the number of dms for one [array(ij|kl)],
 * ncomp is the number of components that produced by intor
 */
void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        double *v_priv;
        int i, j, ij;
        int *ao_loc = malloc(sizeof(int)*(nbas+1));
        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao, ao_loc};

        memset(vjk, 0, sizeof(double)*nao*nao*n_dm*ncomp);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        ao_loc[nbas] = nao;

#pragma omp parallel default(none) \
        shared(intor, fdot, fjk, \
               dms, vjk, n_dm, ncomp, nbas, cintopt, vhfopt, envs) \
        private(ij, i, j, v_priv)
        {
                v_priv = malloc(sizeof(double)*nao*nao*n_dm*ncomp);
                memset(v_priv, 0, sizeof(double)*nao*nao*n_dm*ncomp);
#pragma omp for nowait schedule(dynamic, 2)
                for (ij = 0; ij < nbas*nbas; ij++) {
                        i = ij / nbas;
                        j = ij - i * nbas;
                        (*fdot)(intor, fjk, dms, v_priv, n_dm, ncomp, i, j,
                                cintopt, vhfopt, &envs);
                }
#pragma omp critical
                {
                        for (i = 0; i < nao*nao*n_dm*ncomp; i++) {
                                vjk[i] += v_priv[i];
                        }
                }
                free(v_priv);
        }

        free(ao_loc);
}

