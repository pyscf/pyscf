#include <stdlib.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"
#include "pbc/pbc.h"

#define DBLMIN  1e-200

void NPfcondense(float (*op)(float *, int, int, int), float *out, float *a,
                 int *loc_x, int *loc_y, int nloc_x, int nloc_y);
float NP_fmax(float *a, int nd, int di, int dj);

int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc,
                             int block_size);

void CVHFdot_sr_nrs1(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                     double **dms, double *buf, double *cache, int n_dm,
                     int *ishls, int *jshls, int *kshls, int *lshls,
                     CVHFOpt *vhfopt, IntorEnvs *envs)
{
        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        int *ao_loc = envs->ao_loc;
        CINTOpt *cintopt = envs->cintopt;
        int ish0 = ishls[0];
        int ish1 = ishls[1];
        int jsh0 = jshls[0];
        int jsh1 = jshls[1];
        int ksh0 = kshls[0];
        int ksh1 = kshls[1];
        int lsh0 = lshls[0];
        int lsh1 = lshls[1];
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *q_ijij = (float *)vhfopt->logq_cond;
        float *q_iijj = q_ijij + Nbas2;
        float *s_index = q_iijj + Nbas2;
        float *xij_cond = s_index + Nbas2;
        float *yij_cond = xij_cond + Nbas2;
        float *zij_cond = yij_cond + Nbas2;
        float *dm_cond = (float *)vhfopt->dm_cond;
        float kl_cutoff, jl_cutoff, il_cutoff;
        float log_cutoff = vhfopt->log_cutoff;
        float omega = env[PTR_RANGE_OMEGA];
        float omega2 = omega * omega;
        float dm_max0, dm_max, log_dm;
        float theta, theta_ij, theta_r2, skl_cutoff;
        float xij, yij, zij, xkl, ykl, zkl, dx, dy, dz, r2;
        int shls[4];
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls,
                   int i0, int i1, int j0, int j1,
                   int k0, int k1, int l0, int l1);
        int notempty;
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;
        double ai, aj, ak, al, aij, akl;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                ai = env[bas(PTR_EXP,ish) + bas(NPRIM_OF,ish)-1];

                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        if (q_ijij[ish*Nbas+jsh] < log_cutoff) {
                                continue;
                        }
                        shls[1] = jsh;
                        aj = env[bas(PTR_EXP,jsh) + bas(NPRIM_OF,jsh)-1];
                        aij = ai + aj;
                        theta_ij = omega2*aij / (omega2 + aij);
                        kl_cutoff = log_cutoff - q_ijij[ish*Nbas+jsh];
                        xij = xij_cond[ish * Nbas + jsh];
                        yij = yij_cond[ish * Nbas + jsh];
                        zij = zij_cond[ish * Nbas + jsh];
                        skl_cutoff = log_cutoff - s_index[ish * Nbas + jsh];

for (ksh = ksh0; ksh < ksh1; ksh++) {
        if (q_iijj[ish*Nbas+ksh] < log_cutoff ||
            q_iijj[jsh*Nbas+ksh] < log_cutoff) {
                continue;
        }
        shls[2] = ksh;
        ak = env[bas(PTR_EXP,ksh) + bas(NPRIM_OF,ksh)-1];
        jl_cutoff = log_cutoff - q_iijj[ish*Nbas+ksh];
        il_cutoff = log_cutoff - q_iijj[jsh*Nbas+ksh];

        dm_max0 = dm_cond[ish*nbas+jsh];
        dm_max0 = MAX(dm_max0, dm_cond[ish*nbas+ksh]);
        dm_max0 = MAX(dm_max0, dm_cond[jsh*nbas+ksh]);

        for (lsh = lsh0; lsh < lsh1; lsh++) {
                dm_max = dm_max0 + dm_cond[ksh*nbas+lsh] +
                         dm_cond[ish*nbas+lsh] + dm_cond[jsh*nbas+lsh];
                log_dm = logf(dm_max);
                if (q_ijij[ksh*Nbas+lsh] + log_dm < kl_cutoff ||
                    q_iijj[jsh*Nbas+lsh] + log_dm < jl_cutoff ||
                    q_iijj[ish*Nbas+lsh] + log_dm < il_cutoff) {
                        continue;
                }

                al = env[bas(PTR_EXP,lsh) + bas(NPRIM_OF,lsh)-1];
                akl = ak + al;
                // theta = 1/(1/aij+1/akl+1/omega2);
                theta = theta_ij*akl / (theta_ij + akl);

                xkl = xij_cond[ksh * Nbas + lsh];
                ykl = yij_cond[ksh * Nbas + lsh];
                zkl = zij_cond[ksh * Nbas + lsh];
                dx = xij - xkl;
                dy = yij - ykl;
                dz = zij - zkl;
                r2 = dx * dx + dy * dy + dz * dz;
                theta_r2 = logf(r2 + 1e-30f) + theta * r2 - log_dm;
                if (theta_r2 + skl_cutoff > s_index[ksh*Nbas+lsh]) {
                        continue;
                }
                shls[3] = lsh;
                notempty = (*intor)(buf, NULL, shls,
                                    atm, natm, bas, nbas, env, cintopt, cache);
                if (notempty) {
                        i0 = ao_loc[ish];
                        j0 = ao_loc[jsh];
                        k0 = ao_loc[ksh];
                        l0 = ao_loc[lsh];
                        i1 = ao_loc[ish+1];
                        j1 = ao_loc[jsh+1];
                        k1 = ao_loc[ksh+1];
                        l1 = ao_loc[lsh+1];
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = jkop[idm]->contract;
                                (*pf)(buf, dms[idm], vjk[idm], shls,
                                      i0, i1, j0, j1, k0, k1, l0, l1);
                        }
                }
        }
}
                }
        }
}

void CVHFdot_sr_nrs2ij(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                       double **dms, double *buf, double *cache, int n_dm,
                       int *ishls, int *jshls, int *kshls, int *lshls,
                       CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > jshls[0]) {
                return CVHFdot_sr_nrs1(intor, jkop, vjk, dms, buf, cache, n_dm,
                                       ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[0] < jshls[0]) {
                return;
        }

        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        int *ao_loc = envs->ao_loc;
        CINTOpt *cintopt = envs->cintopt;
        int ish0 = ishls[0];
        int ish1 = ishls[1];
        int jsh0 = jshls[0];
        int jsh1 = jshls[1];
        int ksh0 = kshls[0];
        int ksh1 = kshls[1];
        int lsh0 = lshls[0];
        int lsh1 = lshls[1];
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *q_ijij = (float *)vhfopt->logq_cond;
        float *q_iijj = q_ijij + Nbas2;
        float *s_index = q_iijj + Nbas2;
        float *xij_cond = s_index + Nbas2;
        float *yij_cond = xij_cond + Nbas2;
        float *zij_cond = yij_cond + Nbas2;
        float *dm_cond = (float *)vhfopt->dm_cond;
        float kl_cutoff, jl_cutoff, il_cutoff;
        float log_cutoff = vhfopt->log_cutoff;
        float omega = env[PTR_RANGE_OMEGA];
        float omega2 = omega * omega;
        float dm_max0, dm_max, log_dm;
        float theta, theta_ij, theta_r2, skl_cutoff;
        float xij, yij, zij, xkl, ykl, zkl, dx, dy, dz, r2;
        int shls[4];
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls,
                   int i0, int i1, int j0, int j1,
                   int k0, int k1, int l0, int l1);
        int notempty;
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;
        double ai, aj, ak, al, aij, akl;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                ai = env[bas(PTR_EXP,ish) + bas(NPRIM_OF,ish)-1];

                for (jsh = jsh0; jsh <= ish; jsh++) {
                        if (q_ijij[ish*Nbas+jsh] < log_cutoff) {
                                continue;
                        }
                        shls[1] = jsh;
                        aj = env[bas(PTR_EXP,jsh) + bas(NPRIM_OF,jsh)-1];
                        aij = ai + aj;
                        theta_ij = omega2*aij / (omega2 + aij);
                        kl_cutoff = log_cutoff - q_ijij[ish*Nbas+jsh];
                        xij = xij_cond[ish * Nbas + jsh];
                        yij = yij_cond[ish * Nbas + jsh];
                        zij = zij_cond[ish * Nbas + jsh];
                        skl_cutoff = log_cutoff - s_index[ish * Nbas + jsh];

for (ksh = ksh0; ksh < ksh1; ksh++) {
        if (q_iijj[ish*Nbas+ksh] < log_cutoff ||
            q_iijj[jsh*Nbas+ksh] < log_cutoff) {
                continue;
        }
        shls[2] = ksh;
        ak = env[bas(PTR_EXP,ksh) + bas(NPRIM_OF,ksh)-1];
        jl_cutoff = log_cutoff - q_iijj[ish*Nbas+ksh];
        il_cutoff = log_cutoff - q_iijj[jsh*Nbas+ksh];

        dm_max0 = dm_cond[ish*nbas+jsh];
        dm_max0 = MAX(dm_max0, dm_cond[ish*nbas+ksh]);
        dm_max0 = MAX(dm_max0, dm_cond[jsh*nbas+ksh]);

        for (lsh = lsh0; lsh < lsh1; lsh++) {
                dm_max = dm_max0 + dm_cond[ksh*nbas+lsh] +
                         dm_cond[ish*nbas+lsh] + dm_cond[jsh*nbas+lsh];
                log_dm = logf(dm_max);
                if (q_ijij[ksh*Nbas+lsh] + log_dm < kl_cutoff ||
                    q_iijj[jsh*Nbas+lsh] + log_dm < jl_cutoff ||
                    q_iijj[ish*Nbas+lsh] + log_dm < il_cutoff) {
                        continue;
                }

                al = env[bas(PTR_EXP,lsh) + bas(NPRIM_OF,lsh)-1];
                akl = ak + al;
                // theta = 1/(1/aij+1/akl+1/omega2);
                theta = theta_ij*akl / (theta_ij + akl);

                xkl = xij_cond[ksh * Nbas + lsh];
                ykl = yij_cond[ksh * Nbas + lsh];
                zkl = zij_cond[ksh * Nbas + lsh];
                dx = xij - xkl;
                dy = yij - ykl;
                dz = zij - zkl;
                r2 = dx * dx + dy * dy + dz * dz;
                theta_r2 = logf(r2 + 1e-30f) + theta * r2 - log_dm;
                if (theta_r2 + skl_cutoff > s_index[ksh*Nbas+lsh]) {
                        continue;
                }
                shls[3] = lsh;
                notempty = (*intor)(buf, NULL, shls,
                                    atm, natm, bas, nbas, env, cintopt, cache);
                if (notempty) {
                        i0 = ao_loc[ish];
                        j0 = ao_loc[jsh];
                        k0 = ao_loc[ksh];
                        l0 = ao_loc[lsh];
                        i1 = ao_loc[ish+1];
                        j1 = ao_loc[jsh+1];
                        k1 = ao_loc[ksh+1];
                        l1 = ao_loc[lsh+1];
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = jkop[idm]->contract;
                                (*pf)(buf, dms[idm], vjk[idm], shls,
                                      i0, i1, j0, j1, k0, k1, l0, l1);
                        }
                }
        }
}
                }
        }
}

void CVHFdot_sr_nrs2kl(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                       double **dms, double *buf, double *cache, int n_dm,
                       int *ishls, int *jshls, int *kshls, int *lshls,
                       CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (kshls[0] > lshls[0]) {
                return CVHFdot_sr_nrs1(intor, jkop, vjk, dms, buf, cache, n_dm,
                                       ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (kshls[0] < lshls[0]) {
                return;
        }

        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        int *ao_loc = envs->ao_loc;
        CINTOpt *cintopt = envs->cintopt;
        int ish0 = ishls[0];
        int ish1 = ishls[1];
        int jsh0 = jshls[0];
        int jsh1 = jshls[1];
        int ksh0 = kshls[0];
        int ksh1 = kshls[1];
        int lsh0 = lshls[0];
        int lsh1 = lshls[1];
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *q_ijij = (float *)vhfopt->logq_cond;
        float *q_iijj = q_ijij + Nbas2;
        float *s_index = q_iijj + Nbas2;
        float *xij_cond = s_index + Nbas2;
        float *yij_cond = xij_cond + Nbas2;
        float *zij_cond = yij_cond + Nbas2;
        float *dm_cond = (float *)vhfopt->dm_cond;
        float kl_cutoff, jl_cutoff, il_cutoff;
        float log_cutoff = vhfopt->log_cutoff;
        float omega = env[PTR_RANGE_OMEGA];
        float omega2 = omega * omega;
        float dm_max0, dm_max, log_dm;
        float theta, theta_ij, theta_r2, skl_cutoff;
        float xij, yij, zij, xkl, ykl, zkl, dx, dy, dz, r2;
        int shls[4];
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls,
                   int i0, int i1, int j0, int j1,
                   int k0, int k1, int l0, int l1);
        int notempty;
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;
        double ai, aj, ak, al, aij, akl;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                ai = env[bas(PTR_EXP,ish) + bas(NPRIM_OF,ish)-1];

                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        if (q_ijij[ish*Nbas+jsh] < log_cutoff) {
                                continue;
                        }
                        shls[1] = jsh;
                        aj = env[bas(PTR_EXP,jsh) + bas(NPRIM_OF,jsh)-1];
                        aij = ai + aj;
                        theta_ij = omega2*aij / (omega2 + aij);
                        kl_cutoff = log_cutoff - q_ijij[ish*Nbas+jsh];
                        xij = xij_cond[ish * Nbas + jsh];
                        yij = yij_cond[ish * Nbas + jsh];
                        zij = zij_cond[ish * Nbas + jsh];
                        skl_cutoff = log_cutoff - s_index[ish * Nbas + jsh];

for (ksh = ksh0; ksh < ksh1; ksh++) {
        if (q_iijj[ish*Nbas+ksh] < log_cutoff ||
            q_iijj[jsh*Nbas+ksh] < log_cutoff) {
                continue;
        }
        shls[2] = ksh;
        ak = env[bas(PTR_EXP,ksh) + bas(NPRIM_OF,ksh)-1];
        jl_cutoff = log_cutoff - q_iijj[ish*Nbas+ksh];
        il_cutoff = log_cutoff - q_iijj[jsh*Nbas+ksh];

        dm_max0 = dm_cond[ish*nbas+jsh];
        dm_max0 = MAX(dm_max0, dm_cond[ish*nbas+ksh]);
        dm_max0 = MAX(dm_max0, dm_cond[jsh*nbas+ksh]);

        for (lsh = lsh0; lsh <= ksh; lsh++) {
                dm_max = dm_max0 + dm_cond[ksh*nbas+lsh] +
                         dm_cond[ish*nbas+lsh] + dm_cond[jsh*nbas+lsh];
                log_dm = logf(dm_max);
                if (q_ijij[ksh*Nbas+lsh] + log_dm < kl_cutoff ||
                    q_iijj[jsh*Nbas+lsh] + log_dm < jl_cutoff ||
                    q_iijj[ish*Nbas+lsh] + log_dm < il_cutoff) {
                        continue;
                }

                al = env[bas(PTR_EXP,lsh) + bas(NPRIM_OF,lsh)-1];
                akl = ak + al;
                // theta = 1/(1/aij+1/akl+1/omega2);
                theta = theta_ij*akl / (theta_ij + akl);

                xkl = xij_cond[ksh * Nbas + lsh];
                ykl = yij_cond[ksh * Nbas + lsh];
                zkl = zij_cond[ksh * Nbas + lsh];
                dx = xij - xkl;
                dy = yij - ykl;
                dz = zij - zkl;
                r2 = dx * dx + dy * dy + dz * dz;
                theta_r2 = logf(r2 + 1e-30f) + theta * r2 - log_dm;
                if (theta_r2 + skl_cutoff > s_index[ksh*Nbas+lsh]) {
                        continue;
                }
                shls[3] = lsh;
                notempty = (*intor)(buf, NULL, shls,
                                    atm, natm, bas, nbas, env, cintopt, cache);
                if (notempty) {
                        i0 = ao_loc[ish];
                        j0 = ao_loc[jsh];
                        k0 = ao_loc[ksh];
                        l0 = ao_loc[lsh];
                        i1 = ao_loc[ish+1];
                        j1 = ao_loc[jsh+1];
                        k1 = ao_loc[ksh+1];
                        l1 = ao_loc[lsh+1];
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = jkop[idm]->contract;
                                (*pf)(buf, dms[idm], vjk[idm], shls,
                                      i0, i1, j0, j1, k0, k1, l0, l1);
                        }
                }
        }
}
                }
        }
}

void CVHFdot_sr_nrs4(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                     double **dms, double *buf, double *cache, int n_dm,
                     int *ishls, int *jshls, int *kshls, int *lshls,
                     CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] < jshls[0] || kshls[0] < lshls[0]) {
                return;
        }

        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        int *ao_loc = envs->ao_loc;
        CINTOpt *cintopt = envs->cintopt;
        int ish0 = ishls[0];
        int ish1 = ishls[1];
        int jsh0 = jshls[0];
        int jsh1 = jshls[1];
        int ksh0 = kshls[0];
        int ksh1 = kshls[1];
        int lsh0 = lshls[0];
        int lsh1 = lshls[1];
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *q_ijij = (float *)vhfopt->logq_cond;
        float *q_iijj = q_ijij + Nbas2;
        float *s_index = q_iijj + Nbas2;
        float *xij_cond = s_index + Nbas2;
        float *yij_cond = xij_cond + Nbas2;
        float *zij_cond = yij_cond + Nbas2;
        float *dm_cond = (float *)vhfopt->dm_cond;
        float kl_cutoff, jl_cutoff, il_cutoff;
        float log_cutoff = vhfopt->log_cutoff;
        float omega = env[PTR_RANGE_OMEGA];
        float omega2 = omega * omega;
        float dm_max0, dm_max, log_dm;
        float theta, theta_ij, theta_r2, skl_cutoff;
        float xij, yij, zij, xkl, ykl, zkl, dx, dy, dz, r2;
        int shls[4];
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls,
                   int i0, int i1, int j0, int j1,
                   int k0, int k1, int l0, int l1);
        int notempty;
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;
        double ai, aj, ak, al, aij, akl;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                ai = env[bas(PTR_EXP,ish) + bas(NPRIM_OF,ish)-1];

                for (jsh = jsh0; jsh < MIN(jsh1, ish+1); jsh++) {
                        if (q_ijij[ish*Nbas+jsh] < log_cutoff) {
                                continue;
                        }
                        shls[1] = jsh;
                        aj = env[bas(PTR_EXP,jsh) + bas(NPRIM_OF,jsh)-1];
                        aij = ai + aj;
                        theta_ij = omega2*aij / (omega2 + aij);
                        kl_cutoff = log_cutoff - q_ijij[ish*Nbas+jsh];
                        xij = xij_cond[ish * Nbas + jsh];
                        yij = yij_cond[ish * Nbas + jsh];
                        zij = zij_cond[ish * Nbas + jsh];
                        skl_cutoff = log_cutoff - s_index[ish * Nbas + jsh];

for (ksh = ksh0; ksh < ksh1; ksh++) {
        if (q_iijj[ish*Nbas+ksh] < log_cutoff ||
            q_iijj[jsh*Nbas+ksh] < log_cutoff) {
                continue;
        }
        shls[2] = ksh;
        ak = env[bas(PTR_EXP,ksh) + bas(NPRIM_OF,ksh)-1];
        jl_cutoff = log_cutoff - q_iijj[ish*Nbas+ksh];
        il_cutoff = log_cutoff - q_iijj[jsh*Nbas+ksh];

        dm_max0 = dm_cond[ish*nbas+jsh];
        dm_max0 = MAX(dm_max0, dm_cond[ish*nbas+ksh]);
        dm_max0 = MAX(dm_max0, dm_cond[jsh*nbas+ksh]);

        for (lsh = lsh0; lsh < MIN(lsh1, ksh+1); lsh++) {
                dm_max = dm_max0 + dm_cond[ksh*nbas+lsh] +
                         dm_cond[ish*nbas+lsh] + dm_cond[jsh*nbas+lsh];
                log_dm = logf(dm_max);
                if (q_ijij[ksh*Nbas+lsh] + log_dm < kl_cutoff ||
                    q_iijj[jsh*Nbas+lsh] + log_dm < jl_cutoff ||
                    q_iijj[ish*Nbas+lsh] + log_dm < il_cutoff) {
                        continue;
                }

                al = env[bas(PTR_EXP,lsh) + bas(NPRIM_OF,lsh)-1];
                akl = ak + al;
                // theta = 1/(1/aij+1/akl+1/omega2);
                theta = theta_ij*akl / (theta_ij + akl);

                xkl = xij_cond[ksh * Nbas + lsh];
                ykl = yij_cond[ksh * Nbas + lsh];
                zkl = zij_cond[ksh * Nbas + lsh];
                dx = xij - xkl;
                dy = yij - ykl;
                dz = zij - zkl;
                r2 = dx * dx + dy * dy + dz * dz;
                theta_r2 = logf(r2 + 1e-30f) + theta * r2 - log_dm;
                if (theta_r2 + skl_cutoff > s_index[ksh*Nbas+lsh]) {
                        continue;
                }
                shls[3] = lsh;
                notempty = (*intor)(buf, NULL, shls,
                                    atm, natm, bas, nbas, env, cintopt, cache);
                if (notempty) {
                        i0 = ao_loc[ish];
                        j0 = ao_loc[jsh];
                        k0 = ao_loc[ksh];
                        l0 = ao_loc[lsh];
                        i1 = ao_loc[ish+1];
                        j1 = ao_loc[jsh+1];
                        k1 = ao_loc[ksh+1];
                        l1 = ao_loc[lsh+1];
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = jkop[idm]->contract;
                                (*pf)(buf, dms[idm], vjk[idm], shls,
                                      i0, i1, j0, j1, k0, k1, l0, l1);
                        }
                }
        }
}
                }
        }
}

void CVHFdot_sr_nrs8(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                     double **dms, double *buf, double *cache, int n_dm,
                     int *ishls, int *jshls, int *kshls, int *lshls,
                     CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > kshls[0]) {
                return CVHFdot_sr_nrs4(intor, jkop, vjk, dms, buf, cache, n_dm,
                                       ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[0] < kshls[0]) {
                return;
        } else if ((ishls[1] <= jshls[0]) || (kshls[1] <= lshls[0])) {
                assert(ishls[1] == kshls[1]);
                return;
        }

        // else i == k && i >= j && k >= l
        assert(ishls[1] == kshls[1]);

        int *atm = envs->atm;
        int *bas = envs->bas;
        double *env = envs->env;
        int natm = envs->natm;
        int nbas = envs->nbas;
        int *ao_loc = envs->ao_loc;
        CINTOpt *cintopt = envs->cintopt;
        int ish0 = ishls[0];
        int ish1 = ishls[1];
        int jsh0 = jshls[0];
        int jsh1 = jshls[1];
        int ksh0 = kshls[0];
        int lsh0 = lshls[0];
        int lsh1 = lshls[1];
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *q_ijij = (float *)vhfopt->logq_cond;
        float *q_iijj = q_ijij + Nbas2;
        float *s_index = q_iijj + Nbas2;
        float *xij_cond = s_index + Nbas2;
        float *yij_cond = xij_cond + Nbas2;
        float *zij_cond = yij_cond + Nbas2;
        float *dm_cond = (float *)vhfopt->dm_cond;
        float kl_cutoff, jl_cutoff, il_cutoff;
        float log_cutoff = vhfopt->log_cutoff;
        float dm_max0, dm_max, log_dm;
        float omega = env[PTR_RANGE_OMEGA];
        float omega2 = omega * omega;
        float theta, theta_ij, theta_r2, skl_cutoff;
        float xij, yij, zij, xkl, ykl, zkl, dx, dy, dz, r2;
        int shls[4];
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls,
                   int i0, int i1, int j0, int j1,
                   int k0, int k1, int l0, int l1);
        int notempty;
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;
        double ai, aj, ak, al, aij, akl;

        for (ish = ish0; ish < ish1; ish++) {
                shls[0] = ish;
                ai = env[bas(PTR_EXP,ish) + bas(NPRIM_OF,ish)-1];

                for (jsh = jsh0; jsh < MIN(jsh1, ish+1); jsh++) {
                        if (q_ijij[ish*Nbas+jsh] < log_cutoff) {
                                continue;
                        }
                        shls[1] = jsh;
                        aj = env[bas(PTR_EXP,jsh) + bas(NPRIM_OF,jsh)-1];
                        aij = ai + aj;
                        theta_ij = omega2*aij / (omega2 + aij);
                        kl_cutoff = log_cutoff - q_ijij[ish*Nbas+jsh];
                        xij = xij_cond[ish * Nbas + jsh];
                        yij = yij_cond[ish * Nbas + jsh];
                        zij = zij_cond[ish * Nbas + jsh];
                        skl_cutoff = log_cutoff - s_index[ish * Nbas + jsh];

for (ksh = ksh0; ksh <= ish; ksh++) {
        if (q_iijj[ish*Nbas+ksh] < log_cutoff ||
            q_iijj[jsh*Nbas+ksh] < log_cutoff) {
                continue;
        }
        shls[2] = ksh;
        ak = env[bas(PTR_EXP,ksh) + bas(NPRIM_OF,ksh)-1];
        jl_cutoff = log_cutoff - q_iijj[ish*Nbas+ksh];
        il_cutoff = log_cutoff - q_iijj[jsh*Nbas+ksh];

        dm_max0 = dm_cond[ish*nbas+jsh];
        dm_max0 = MAX(dm_max0, dm_cond[ish*nbas+ksh]);
        dm_max0 = MAX(dm_max0, dm_cond[jsh*nbas+ksh]);

        for (lsh = lsh0; lsh < MIN(lsh1, ksh+1); lsh++) {
                if ((ksh == ish) && (lsh > jsh)) {
                        break;
                }
                dm_max = dm_max0 + dm_cond[ksh*nbas+lsh] +
                         dm_cond[ish*nbas+lsh] + dm_cond[jsh*nbas+lsh];
                log_dm = logf(dm_max);
                if (q_ijij[ksh*Nbas+lsh] + log_dm < kl_cutoff ||
                    q_iijj[jsh*Nbas+lsh] + log_dm < jl_cutoff ||
                    q_iijj[ish*Nbas+lsh] + log_dm < il_cutoff) {
                        continue;
                }

                al = env[bas(PTR_EXP,lsh) + bas(NPRIM_OF,lsh)-1];
                akl = ak + al;
                // theta = 1/(1/aij+1/akl+1/omega2);
                theta = theta_ij*akl / (theta_ij + akl);

                xkl = xij_cond[ksh * Nbas + lsh];
                ykl = yij_cond[ksh * Nbas + lsh];
                zkl = zij_cond[ksh * Nbas + lsh];
                dx = xij - xkl;
                dy = yij - ykl;
                dz = zij - zkl;
                r2 = dx * dx + dy * dy + dz * dz;
                theta_r2 = logf(r2 + 1e-30f) + theta * r2 - log_dm;
                if (theta_r2 + skl_cutoff > s_index[ksh*Nbas+lsh]) {
                        continue;
                }
                shls[3] = lsh;
                notempty = (*intor)(buf, NULL, shls,
                                    atm, natm, bas, nbas, env, cintopt, cache);
                if (notempty) {
                        i0 = ao_loc[ish];
                        j0 = ao_loc[jsh];
                        k0 = ao_loc[ksh];
                        l0 = ao_loc[lsh];
                        i1 = ao_loc[ish+1];
                        j1 = ao_loc[jsh+1];
                        k1 = ao_loc[ksh+1];
                        l1 = ao_loc[lsh+1];
                        for (idm = 0; idm < n_dm; idm++) {
                                pf = jkop[idm]->contract;
                                (*pf)(buf, dms[idm], vjk[idm], shls,
                                      i0, i1, j0, j1, k0, k1, l0, l1);
                        }
                }
        }
}
                }
        }
}

void CVHFnr_sr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                          double **dms, double **vjk, int n_dm, int ncomp,
                          int *shls_slice, int *ao_loc,
                          CINTOpt *cintopt, CVHFOpt *vhfopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};
        int idm;
        double *tile_dms[n_dm];
        for (idm = 0; idm < n_dm; idm++) {
                CVHFzero_out_vjk(vjk[idm], jkop[idm], shls_slice, ao_loc, ncomp);
                tile_dms[idm] = CVHFallocate_and_reorder_dm(jkop[idm], dms[idm],
                                                            shls_slice, ao_loc);
        }

        size_t di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                              atm, natm, bas, nbas, env);
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int nish = ish1 - ish0;
        int njsh = jsh1 - jsh0;
        int nksh = ksh1 - ksh0;
        int nlsh = lsh1 - lsh0;
        assert(njsh == nish);
        assert(nksh == nish);
        assert(nlsh == nish);
        int *block_loc = malloc(sizeof(int) * (nish+1));
        uint32_t nblock = CVHFshls_block_partition(block_loc, shls_slice, ao_loc, AO_BLOCK_SIZE);
        uint32_t nblock2 = nblock * nblock;
        uint32_t nblock3 = nblock2 * nblock;
        // up to 1.6 GB per thread
        int size_limit = (200000000 - di*di*di*di*ncomp - cache_size) / n_dm;

        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *q_ijij = (float *)vhfopt->logq_cond;
        float *q_iijj = q_ijij + Nbas2;
        float *dm_cond = (float *)vhfopt->dm_cond;
        float *bq_ijij = malloc(sizeof(float) * nblock2*3);
        float *bq_iijj = bq_ijij + nblock2;
        float *bdm_cond = bq_iijj + nblock2;
        NPfcondense(NP_fmax, bq_ijij, q_ijij, block_loc, block_loc, nblock, nblock);
        NPfcondense(NP_fmax, bq_iijj, q_iijj, block_loc, block_loc, nblock, nblock);
        NPfcondense(NP_fmax, bdm_cond, dm_cond, block_loc, block_loc, nblock, nblock);

#pragma omp parallel
{
        int ioff = ao_loc[ish0];
        int joff = ao_loc[jsh0];
        int koff = ao_loc[ksh0];
        int loff = ao_loc[lsh0];
        float log_cutoff = vhfopt->log_cutoff;
        float ij_cutoff, ik_cutoff, il_cutoff;
        float dm_max0, dm_max, log_dm;
        int i, j, k, l, n, r, blk_id;
        JKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = CVHFallocate_JKArray(jkop[i], shls_slice, ao_loc,
                                                 ncomp, nblock, size_limit);
        }
        double *buf = malloc(sizeof(double) * (di*di*di*di*ncomp + di*di*2 + cache_size));
        double *cache = buf + di*di*di*di*ncomp;
#pragma omp for nowait schedule(dynamic, 1)
        for (blk_id = 0; blk_id < nblock3; blk_id++) {
                r = blk_id;
                j = r / nblock2; r = r % nblock2;
                k = r / nblock ; r = r % nblock;
                l = r;
                if (bq_ijij[k*nblock+l] < log_cutoff ||
                    bq_iijj[j*nblock+k] < log_cutoff ||
                    bq_iijj[j*nblock+l] < log_cutoff) {
                        continue;
                }
                ij_cutoff = log_cutoff - bq_ijij[k*nblock+l];
                il_cutoff = log_cutoff - bq_iijj[j*nblock+k];
                ik_cutoff = log_cutoff - bq_iijj[j*nblock+l];

                dm_max0 = bdm_cond[k*nblock+l];
                dm_max0 = MAX(dm_max0, bdm_cond[j*nblock+l]);
                dm_max0 = MAX(dm_max0, bdm_cond[j*nblock+k]);

                int j0 = ao_loc[block_loc[j]];
                int k0 = ao_loc[block_loc[k]];
                int l0 = ao_loc[block_loc[l]];
                int j1 = ao_loc[block_loc[j+1]];
                int k1 = ao_loc[block_loc[k+1]];
                int l1 = ao_loc[block_loc[l+1]];
                for (n = 0; n < n_dm; n++) {
                        JKArray *pv = v_priv[n];
                        pv->ao_off[1] = j0 - joff;
                        pv->ao_off[2] = k0 - koff;
                        pv->ao_off[3] = l0 - loff;
                        pv->shape[1] = j1 - j0;
                        pv->shape[2] = k1 - k0;
                        pv->shape[3] = l1 - l0;
                        pv->block_quartets[1] = j;
                        pv->block_quartets[2] = k;
                        pv->block_quartets[3] = l;
                }
                for (i = 0; i < nblock; i++) {
                        dm_max = MAX(dm_max0, bdm_cond[i*nblock+j]);
                        dm_max = MAX(dm_max , bdm_cond[i*nblock+k]);
                        dm_max = MAX(dm_max , bdm_cond[i*nblock+l]);
                        log_dm = logf(dm_max);
                        if (bq_ijij[i*nblock+j] + log_dm < ij_cutoff ||
                            bq_iijj[i*nblock+k] + log_dm < il_cutoff ||
                            bq_iijj[i*nblock+l] + log_dm < ik_cutoff) {
                                continue;
                        }

                        int i0 = ao_loc[block_loc[i]];
                        int i1 = ao_loc[block_loc[i+1]];
                        for (n = 0; n < n_dm; n++) {
                                JKArray *pv = v_priv[n];
                                pv->ao_off[0] = i0 - ioff;
                                pv->shape[0] = i1 - i0;
                                pv->block_quartets[0] = i;
                        }
                        (*fdot)(intor, jkop, v_priv, tile_dms, buf, cache, n_dm,
                                block_loc+i, block_loc+j, block_loc+k, block_loc+l,
                                vhfopt, &envs);
                }
                for (n = 0; n < n_dm; n++) {
                        if (v_priv[n]->stack_size >= size_limit) {
#pragma omp critical
        (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
                               block_loc, block_loc, block_loc, block_loc);
                        }
                }
        }
        for (n = 0; n < n_dm; n++) {
                if (v_priv[n]->stack_size > 0) {
#pragma omp critical
        (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
                               block_loc, block_loc, block_loc, block_loc);
                }
                CVHFdeallocate_JKArray(v_priv[n]);
        }
        free(buf);
}
        for (idm = 0; idm < n_dm; idm++) {
                free(tile_dms[idm]);
        }
        free(block_loc);
        free(bq_ijij);
}

// sqrt(-log(1e-9))
#define R_GUESS_FAC     4.5f

void CVHFnr_sr_int2e_q_cond(int (*intor)(), CINTOpt *cintopt, float *q_cond,
                            int *ao_loc, int *atm, int natm,
                            int *bas, int nbas, double *env)
{
        size_t Nbas = nbas;
        size_t Nbas2 = Nbas * Nbas;
        float *qijij = q_cond;
        float *qiijj = q_cond + Nbas2;
        float *s_index = qiijj + Nbas2;
        float *xij_cond = s_index + Nbas2;
        float *yij_cond = xij_cond + Nbas2;
        float *zij_cond = yij_cond + Nbas2;

        int shls_slice[] = {0, nbas};
        const int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                                 atm, natm, bas, nbas, env);
        float *exps = malloc(sizeof(float) * nbas * 5);
        float *cs = exps + nbas;
        float *rx = cs + nbas;
        float *ry = rx + nbas;
        float *rz = ry + nbas;

        for (int n = 0; n < nbas; n++) {
                int ia = bas[ATOM_OF+n*BAS_SLOTS];
                int nprim = bas[NPRIM_OF+n*BAS_SLOTS];
                int nctr = bas[NCTR_OF+n*BAS_SLOTS];
                int ptr_coord = atm[PTR_COORD+ia*ATM_SLOTS];
                int ptr_coeff = bas[PTR_COEFF+n*BAS_SLOTS];
                exps[n] = env[bas[PTR_EXP+n*BAS_SLOTS] + nprim-1];
                rx[n] = env[ptr_coord+0];
                ry[n] = env[ptr_coord+1];
                rz[n] = env[ptr_coord+2];

                float c_max = fabs(env[ptr_coeff + nprim-1]);
                for (int m = 1; m < nctr; m++) {
                        float c1 = fabs(env[ptr_coeff + (m+1)*nprim-1]);
                        c_max = MAX(c_max, c1);
                }
                cs[n] = c_max;
        }
        float omega = env[PTR_RANGE_OMEGA];
        float omega2 = omega * omega;

#pragma omp parallel
{
        float fac_guess = .5f - logf(omega2)/4;
        int ish, jsh, li, lj;
        int ij, i, j, di, dj, dij, di2, dj2;
        float ai, aj, aij, ai_aij, a1, ci, cj;
        float xi, yi, zi, xj, yj, zj, xij, yij, zij;
        float dx, dy, dz, r2, v, log_fac, r_guess, theta, theta_r;
        double qtmp, tmp;
        float log_qmax;
        int shls[4];
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *cache = malloc(sizeof(double) * (di*di*di*di + cache_size));
        double *buf = cache + cache_size;

#pragma omp for schedule(dynamic, 1)
        for (ish = 0; ish < nbas; ish++) {
                li = bas[ANG_OF+ish*BAS_SLOTS];
                ai = exps[ish];
                ci = cs[ish];
                xi = rx[ish];
                yi = ry[ish];
                zi = rz[ish];
                di = ao_loc[ish+1] - ao_loc[ish];
                di2 = di * di;
#pragma GCC ivdep
                for (jsh = 0; jsh <= ish; jsh++) {
                        lj = bas[ANG_OF+jsh*BAS_SLOTS];
                        aj = exps[jsh];
                        cj = cs[jsh];
                        xj = rx[jsh];
                        yj = ry[jsh];
                        zj = rz[jsh];
                        dx = xj - xi;
                        dy = yj - yi;
                        dz = zj - zi;
                        aij = ai + aj;
                        ai_aij = ai / aij;
                        a1 = ai_aij * aj;
                        xij = xi + ai_aij * dx;
                        yij = yi + ai_aij * dy;
                        zij = zi + ai_aij * dz;

                        theta = omega2/(omega2+aij);
                        r_guess = R_GUESS_FAC / sqrtf(aij * theta);
                        theta_r = theta * r_guess;
                        // log(ci*cj * ((2*li+1)*(2*lj+1))**.5/(4*pi) * (pi/aij)**1.5)
                        log_fac = logf(ci*cj * sqrtf((2*li+1.f)*(2*lj+1.f))/(4*M_PI))
                                + 1.5f*logf(M_PI/aij) + fac_guess;
                        r2 = dx * dx + dy * dy + dz * dz;
                        v = (li+lj)*logf(MAX(theta_r, 1.f)) - a1*r2 + log_fac;
                        s_index[ish*Nbas+jsh] = v;
                        s_index[jsh*Nbas+ish] = v;
                        xij_cond[ish*Nbas+jsh] = xij;
                        yij_cond[ish*Nbas+jsh] = yij;
                        zij_cond[ish*Nbas+jsh] = zij;
                        xij_cond[jsh*Nbas+ish] = xij;
                        yij_cond[jsh*Nbas+ish] = yij;
                        zij_cond[jsh*Nbas+ish] = zij;

                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        dij = di * dj;
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ish;
                        shls[3] = jsh;
                        qtmp = DBLMIN;
                        if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                          cintopt, cache)) {
                                for (ij = 0; ij < dij; ij++) {
                                        // buf[i,j,i,j]
                                        tmp = fabsf(buf[ij+dij*ij]);
                                        qtmp = MAX(qtmp, tmp);
                                }
                        }
                        log_qmax = log(qtmp) * .5;
                        qijij[ish*Nbas+jsh] = log_qmax;
                        qijij[jsh*Nbas+ish] = log_qmax;

                        shls[0] = ish;
                        shls[1] = ish;
                        shls[2] = jsh;
                        shls[3] = jsh;
                        dj2 = dj * dj;
                        qtmp = DBLMIN;
                        if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                          cintopt, cache)) {
                                for (j = 0; j < dj2; j+=dj+1) {
                                for (i = 0; i < di2; i+=di+1) {
                                        // buf[i,i,j,j]
                                        tmp = fabsf(buf[i+di2*j]);
                                        qtmp = MAX(qtmp, tmp);
                                } }
                        }
                        log_qmax = log(qtmp) * .5;
                        qiijj[ish*Nbas+jsh] = log_qmax;
                        qiijj[jsh*Nbas+ish] = log_qmax;
                }
        }
        free(cache);
}
        free(exps);
}

void CVHFsetnr_sr_direct_scf(int (*intor)(), CINTOpt *cintopt, float *q_cond,
                             int *ao_loc, int *atm, int natm,
                             int *bas, int nbas, double *env)
{
        CVHFnr_sr_int2e_q_cond(intor, cintopt, q_cond, ao_loc,
                               atm, natm, bas, nbas, env);
}
