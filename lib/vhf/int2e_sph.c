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


/*
 *************************************************
 * 2e AO integrals in s4, s2ij, s2kl, s1
 */
static void s1_copy(double *eri, double *buf,
                    int ncomp, int nao, int naoi, int naoj,
                    int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int i0 = ishloc[ish];
        int j0 = jshloc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int di = ishloc[ish+1] - i0;
        int dj = jshloc[jsh+1] - j0;
        int dk = ao_loc[ksh+1] - k0;
        int dl = ao_loc[lsh+1] - l0;
        int dij = di * dj;
        int dijk = dij * dk;
        int dijkl = dijk * dl;
        int i, j, k, l, icomp;
        size_t nao2 = nao * nao;
        size_t neri = nao2 * naoi * naoj;
        double *peri, *pbuf;
        eri += nao2 * (i0*naoj+j0) + k0*nao+l0;

        for (icomp = 0; icomp < ncomp; icomp++) {
                for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                        peri = eri + nao2*(i*naoj+j);
                        for (k = 0; k < dk; k++) {
                                pbuf = buf + k*dij + j*di + i;
                                for (l = 0; l < dl; l++) {
                                        peri[k*nao+l] = pbuf[l*dijk];
                                }
                        }
                } }
                buf += dijkl;
                eri += neri;
        }
}

static void s1_set0(double *eri, double *nop,
                    int ncomp, int nao, int naoi, int naoj,
                    int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int i, j, k, l, icomp;
        size_t nao2 = nao * nao;
        size_t neri = nao2 * naoi * naoj;
        double *peri;

        for (icomp = 0; icomp < ncomp; icomp++) {
                for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                for (j = jshloc[jsh]; j < jshloc[jsh+1]; j++) {
                        peri = eri + nao2*(i*naoj+j);
                        for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                        for (l = ao_loc[lsh]; l < ao_loc[lsh+1]; l++) {
                                peri[k*nao+l] = 0;
                        } }
                } }
                eri += neri;
        }
}

static void s2ij_copy(double *eri, double *buf,
                      int ncomp, int nao, int naoi, int naoj,
                      int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int i0 = ishloc[ish];
        int j0 = jshloc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int di = ishloc[ish+1] - i0;
        int dj = jshloc[jsh+1] - j0;
        int dk = ao_loc[ksh+1] - k0;
        int dl = ao_loc[lsh+1] - l0;
        int dij = di * dj;
        int dijk = dij * dk;
        int dijkl = dijk * dl;
        int i, j, k, l, icomp;
        size_t nao2 = nao * nao;
        size_t neri = nao2 * naoi*(naoi+1)/2;
        double *peri, *pbuf;
        eri += k0*nao+l0;

        if (ish > jsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = ishloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jshloc[jsh], j = 0; j < dj; j++, j0++) {
                                peri = eri + nao2*(i0*(i0+1)/2+j0);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l < dl; l++) {
                                                peri[k*nao+l] = pbuf[l*dijk];
                                        }
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        } else {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = ishloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jshloc[jsh], j = 0; j <= i; j++, j0++) {
                                peri = eri + nao2*(i0*(i0+1)/2+j0);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l < dl; l++) {
                                                peri[k*nao+l] = pbuf[l*dijk];
                                        }
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        }
}

static void s2ij_set0(double *eri, double *nop,
                      int ncomp, int nao, int naoi, int naoj,
                      int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int i, j, k, l, icomp;
        size_t nao2 = nao * nao;
        size_t neri = nao2 * naoi*(naoi+1)/2;
        double *peri;

        if (ish > jsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j < jshloc[jsh+1]; j++) {
                                peri = eri + nao2*(i*(i+1)/2+j);
                                for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh+1]; l++) {
                                        peri[k*nao+l] = 0;
                                } }
                        } }
                        eri += neri;
                }
        } else {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j <= i; j++) {
                                peri = eri + nao2*(i*(i+1)/2+j);
                                for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh+1]; l++) {
                                        peri[k*nao+l] = 0;
                                } }
                        } }
                        eri += neri;
                }
        }
}

static void s2kl_copy(double *eri, double *buf,
                      int ncomp, int nao, int naoi, int naoj,
                      int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int i0 = ishloc[ish];
        int j0 = jshloc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int di = ishloc[ish+1] - i0;
        int dj = jshloc[jsh+1] - j0;
        int dk = ao_loc[ksh+1] - k0;
        int dl = ao_loc[lsh+1] - l0;
        int dij = di * dj;
        int dijk = dij * dk;
        int dijkl = dijk * dl;
        int i, j, k, l, icomp;
        size_t nao2 = nao*(nao+1)/2;
        size_t neri = nao2 * naoi * naoj;
        double *peri, *pbuf;
        eri += nao2 * (i0*naoj+j0) + k0*(k0+1)/2+l0;

        if (ksh > lsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                peri = eri + nao2*(i*naoj+j);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        } else {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                peri = eri + nao2*(i*naoj+j);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        }
}

static void s2kl_set0(double *eri, double *nop,
                      int ncomp, int nao, int naoi, int naoj,
                      int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int dk = ao_loc[ksh+1] - k0;
        int dl = ao_loc[lsh+1] - l0;
        int i, j, k, l, icomp;
        size_t nao2 = nao*(nao+1)/2;
        size_t neri = nao2 * naoi * naoj;
        double *peri;
        eri += k0*(k0+1)/2+l0;

        if (ksh > lsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j < jshloc[jsh+1]; j++) {
                                peri = eri + nao2*(i*nao+j);
                                for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        eri += neri;
                }
        } else {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j < jshloc[jsh+1]; j++) {
                                peri = eri + nao2*(i*nao+j);
                                for (k = 0; k < dk; k++) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        eri += neri;
                }
        }
}

static void s4_copy(double *eri, double *buf,
                    int ncomp, int nao, int naoi, int naoj,
                    int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int i0 = ishloc[ish];
        int j0 = jshloc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int di = ishloc[ish+1] - i0;
        int dj = jshloc[jsh+1] - j0;
        int dk = ao_loc[ksh+1] - k0;
        int dl = ao_loc[lsh+1] - l0;
        int dij = di * dj;
        int dijk = dij * dk;
        int dijkl = dijk * dl;
        int i, j, k, l, icomp;
        size_t nao2 = nao*(nao+1)/2;
        size_t neri = nao2 * naoi*(naoi+1)/2;
        double *peri, *pbuf;
        eri += k0*(k0+1)/2+l0;

        if (ksh > lsh && ish > jsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = ishloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jshloc[jsh], j = 0; j < dj; j++, j0++) {
                                peri = eri + nao2*(i0*(i0+1)/2+j0);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        } else if (ish > jsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = ishloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jshloc[jsh], j = 0; j < dj; j++, j0++) {
                                peri = eri + nao2*(i0*(i0+1)/2+j0);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        } else if (ksh > lsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = ishloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jshloc[jsh], j = 0; j <= i; j++, j0++) {
                                peri = eri + nao2*(i0*(i0+1)/2+j0);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        } else {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = ishloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jshloc[jsh], j = 0; j <= i; j++, j0++) {
                                peri = eri + nao2*(i0*(i0+1)/2+j0);
                                for (k = 0; k < dk; k++) {
                                        pbuf = buf + k*dij + j*di + i;
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = pbuf[l*dijk];
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        buf += dijkl;
                        eri += neri;
                }
        }
}

static void s4_set0(double *eri, double *nop,
                    int ncomp, int nao, int naoi, int naoj,
                    int *shls, int *ao_loc, int *ishloc, int *jshloc)
{
        int ish = shls[0];
        int jsh = shls[1];
        int ksh = shls[2];
        int lsh = shls[3];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        int dk = ao_loc[ksh+1] - k0;
        int dl = ao_loc[lsh+1] - l0;
        int i, j, k, l, icomp;
        size_t nao2 = nao*(nao+1)/2;
        size_t neri = nao2 * nao2;
        double *peri;
        eri += k0*(k0+1)/2+l0;

        if (ksh > lsh && ish > jsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j < jshloc[jsh+1]; j++) {
                                peri = eri + nao2*(i*(i+1)/2+j);
                                for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        eri += neri;
                }
        } else if (ish > jsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j < jshloc[jsh+1]; j++) {
                                peri = eri + nao2*(i*(i+1)/2+j);
                                for (k = 0; k < dk; k++) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        eri += neri;
                }
        } else if (ksh > lsh) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j <= i; j++) {
                                peri = eri + nao2*(i*(i+1)/2+j);
                                for (k = 0; k < dk; k++) {
                                        for (l = 0; l < dl; l++) {
                                                peri[l] = 0;
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        eri += neri;
                }
        } else {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = ishloc[ish]; i < ishloc[ish+1]; i++) {
                        for (j = jshloc[jsh]; j <= i; j++) {
                                peri = eri + nao2*(i*(i+1)/2+j);
                                for (k = 0; k < dk; k++) {
                                        for (l = 0; l <= k; l++) {
                                                peri[l] = 0;
                                        }
                                        peri += k0 + k + 1;
                                }
                        } }
                        eri += neri;
                }
        }
}

#define DISTR_INTS_BY(fcopy, fset0) \
        if ((*fprescreen)(shls, envs->vhfopt, envs->atm, envs->bas, envs->env) && \
            (*intor)(buf, shls, envs->atm, envs->natm, \
                     envs->bas, envs->nbas, envs->env, envs->cintopt)) { \
                fcopy(eri, buf, ncomp, envs->nao, naoi, naoj, \
                      shls, envs->ao_loc, ishloc, jshloc); \
        } else { \
                fset0(eri, buf, ncomp, envs->nao, naoi, naoj, \
                      shls, envs->ao_loc, ishloc, jshloc); \
        }

void GTOnr2e_fill_s1(int (*intor)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ish, int jsh,
                     struct _VHFEnvs *envs,
                     int *ishloc, int *jshloc, int naoi, int naoj)
{
        int di = ishloc[ish+1] - ishloc[ish];
        int dj = jshloc[jsh+1] - jshloc[jsh];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        int ksh, lsh;
        int shls[4];

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < envs->nbas; ksh++) {
        for (lsh = 0; lsh < envs->nbas; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                DISTR_INTS_BY(s1_copy, s1_set0);
        } }
        free(buf);
}

void GTOnr2e_fill_s2ij(int (*intor)(), int (*fprescreen)(),
                       double *eri, int ncomp, int ish, int jsh,
                       struct _VHFEnvs *envs,
                       int *ishloc, int *jshloc, int naoi, int naoj)
{
        if (ish < jsh) {
                return;
        }
        int di = ishloc[ish+1] - ishloc[ish];
        int dj = jshloc[jsh+1] - jshloc[jsh];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        int shls[4];
        int ksh, lsh;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < envs->nbas; ksh++) {
        for (lsh = 0; lsh < envs->nbas; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                DISTR_INTS_BY(s2ij_copy, s2ij_set0);
        } }
        free(buf);
}

void GTOnr2e_fill_s2kl(int (*intor)(), int (*fprescreen)(),
                       double *eri, int ncomp, int ish, int jsh,
                       struct _VHFEnvs *envs,
                       int *ishloc, int *jshloc, int naoi, int naoj)
{
        int di = ishloc[ish+1] - ishloc[ish];
        int dj = jshloc[jsh+1] - jshloc[jsh];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        int shls[4];
        int ksh, lsh;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < envs->nbas; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                DISTR_INTS_BY(s2kl_copy, s2kl_set0);
        } }
        free(buf);
}

void GTOnr2e_fill_s4(int (*intor)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ish, int jsh,
                     struct _VHFEnvs *envs,
                     int *ishloc, int *jshloc, int naoi, int naoj)
{
        if (ish < jsh) {
                return;
        }
        int di = ishloc[ish+1] - ishloc[ish];
        int dj = jshloc[jsh+1] - jshloc[jsh];
        double *buf = malloc(sizeof(double)*di*dj*NCTRMAX*NCTRMAX*ncomp);
        int shls[4];
        int ksh, lsh;

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = 0; ksh < envs->nbas; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                shls[2] = ksh;
                shls[3] = lsh;
                DISTR_INTS_BY(s4_copy, s4_set0);
        } }
        free(buf);
}

void GTOnr2e_fill_drv(int (*intor)(), int (*cgto_in_shell)(), void (*fill)(),
                      double *eri, int ncomp,
                      int *ishlst, int nbra, int *jshlst, int nket,
                      CINTOpt *cintopt, CVHFOpt *vhfopt,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        int *ao_loc = malloc(sizeof(int)*(nbas+1));
        int *ishloc = malloc(sizeof(int)*(nbas+1));
        int *jshloc = malloc(sizeof(int)*(nbas+1));
        int i, j, ij;
        int nao = 0;
        for (i = 0; i < nbas; i++) {
                ao_loc[i] = nao;
                nao += (*cgto_in_shell)(i, bas);
        }
        ao_loc[nbas] = nao;
        memcpy(ishloc, ao_loc, sizeof(int)*(nbas+1));
        memcpy(jshloc, ao_loc, sizeof(int)*(nbas+1));

// ishloc (jshloc) is fake ao_loc.  For shell x which is given by ishlst[*],
// ishloc stores the actual offset in ishloc[x] and the stop offset in ishloc[x+1] 
        int naoi = 0;
        int naoj = 0;
        for (i = 0; i < nbra; i++) {
                j = ishlst[i];
                ishloc[j] = naoi;
                naoi += ao_loc[j+1] - ao_loc[j];
                ishloc[j+1] = naoi;
        }
        for (i = 0; i < nket; i++) {
                j = jshlst[i];
                jshloc[j] = naoj;
                naoj += ao_loc[j+1] - ao_loc[j];
                jshloc[j+1] = naoj;
        }

        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao,
                                ao_loc, NULL, vhfopt, cintopt};
        int (*fprescreen)();
        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

#pragma omp parallel default(none) \
        shared(fill, fprescreen, eri, envs, intor, ncomp, nbra, nket, \
               naoi, naoj, ishlst, jshlst, ishloc, jshloc) \
        private(i, j, ij)
#pragma omp for nowait schedule(dynamic)
        for (ij = 0; ij < nbra*nket; ij++) {
                i = ij / nket;
                j = ij % nket;
                (*fill)(intor, fprescreen, eri, ncomp, ishlst[i], jshlst[j],
                        &envs, ishloc, jshloc, naoi, naoj);
        }

        free(ao_loc);
        free(ishloc);
        free(jshloc);
}

