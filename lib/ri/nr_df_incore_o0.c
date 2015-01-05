/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

struct _AO2MOEnvs {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        int nao;
        int ksh_start;
        int ksh_count;
        int bra_start;
        int bra_count;
        int ket_start;
        int ket_count;
        int ncomp;
        int *ao_loc;
        double *mo_coeff;
};

void RIfill_s1_auxe2(int (*intor)(), double *eri,
                     int ish, int jsh, int bastart, int auxstart, int auxcount,
                     CINTOpt *cintopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int *ao_loc = envs->ao_loc;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int nbasnaux = auxstart + auxcount;
        const int naoaux = ao_loc[nbasnaux] - nao;
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*naoaux);

        int ksh, dk;
        int i, j, k, i0, j0, k0;
        int shls[3];
        unsigned long ij0;
        double *peri, *pbuf;

        shls[0] = ish;
        shls[1] = jsh;
        for (ksh = auxstart; ksh < nbasnaux; ksh++) {
                shls[2] = ksh;
                if ((*intor)(eribuf, shls, envs->atm, envs->natm,
                             envs->bas, envs->nbas, envs->env, cintopt)) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        i0 = ao_loc[ish] - ao_loc[bastart];
                        for (i = 0; i < di; i++, i0++) {
                        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                                ij0 = i0 * nao + j0;
                                k0 = ao_loc[ksh] - nao;
                                peri = eri + ij0 * naoaux + k0;
                                pbuf = eribuf + j * di + i;
                                for (k = 0; k < dk; k++) {
                                        peri[k] = pbuf[k*dij];
                                }
                        } }
                }
        }
        free(eribuf);
}

void RIfill_s2ij_auxe2(int (*intor)(), double *eri,
                       int ish, int jsh, int bastart, int auxstart, int auxcount,
                       CINTOpt *cintopt, struct _VHFEnvs *envs)
{
        if (ish < jsh) {
                return;
        }

        const int nao = envs->nao;
        const int *ao_loc = envs->ao_loc;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int dij = di * dj;
        const int ijoff = ao_loc[bastart] * (ao_loc[bastart] + 1) / 2;
        const int nbasnaux = auxstart + auxcount;
        const int naoaux = ao_loc[nbasnaux] - nao;
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*naoaux);

        int ksh, dk;
        int i, j, k, i0, j0, k0;
        int shls[3];
        unsigned long ij0;
        double *peri, *pbuf;

        shls[0] = ish;
        shls[1] = jsh;
        for (ksh = auxstart; ksh < nbasnaux; ksh++) {
                shls[2] = ksh;
                if ((*intor)(eribuf, shls, envs->atm, envs->natm,
                             envs->bas, envs->nbas, envs->env, cintopt)) {
                        dk = ao_loc[ksh+1] - ao_loc[ksh];
                        if (ish == jsh) {
                                for (i0 = ao_loc[ish],i = 0; i < di; i++, i0++) {
                                for (j0 = ao_loc[jsh],j = 0; j0 <= i0; j++, j0++) {
                                        ij0 = i0*(i0+1)/2 + j0 - ijoff;
                                        k0 = ao_loc[ksh] - nao;
                                        peri = eri + ij0 * naoaux + k0;
                                        pbuf = eribuf + j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                peri[k] = pbuf[k*dij];
                                        }
                                } }
                        } else {
                                for (i0 = ao_loc[ish], i = 0; i < di; i++,i0++) {
                                for (j0 = ao_loc[jsh], j = 0; j < dj; j++,j0++) {
                                        ij0 = i0*(i0+1)/2 + j0 - ijoff;
                                        k0 = ao_loc[ksh] - nao;
                                        peri = eri + ij0 * naoaux + k0;
                                        pbuf = eribuf + j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                peri[k] = pbuf[k*dij];
                                        }
                                } }
                        }
                }
        }
        free(eribuf);
}


/*
 * fill can be one of RIfill_s1_auxe2 and RIfill_s2ij_auxe2
 * NOTE nbas is the number of normal basis, the number of auxiliary basis is auxcount;
 * bastart and bascount to fill a range of basis;
 * auxstart is the end of normal basis, so it equals to the number of
 * normal basis
 */
void RInr_3c2e_auxe2_drv(int (*intor)(), void (*fill)(), double *eri,
                         int bastart, int bascount, int auxstart, int auxcount,
                         int ncomp, CINTOpt *cintopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nbasnaux = auxstart + auxcount;
        int *ao_loc = malloc(sizeof(int)*(nbasnaux+1));
        CINTshells_spheric_offset(ao_loc, bas, nbasnaux);
        ao_loc[nbasnaux] = ao_loc[nbasnaux-1] + CINTcgto_spheric(nbasnaux-1, bas);
        const int nao = ao_loc[auxstart];
        const int nshell = auxstart;

        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao, ao_loc};

        int i, j, ij;

#pragma omp parallel default(none) \
        shared(eri, intor, fill, bastart, bascount, auxstart, auxcount, \
               envs, cintopt) \
        private(ij, i, j)
#pragma omp for nowait schedule(dynamic, 2)
        for (ij = 0; ij < nshell*bascount; ij++) {
                i = ij / nshell;
                j = ij - i * nshell;
                (*fill)(intor, eri,
                        i, j, bastart, auxstart, auxcount,
                        cintopt, &envs);
        }

        free(ao_loc);
}

/*
void RInr_int3c2e_auxe1(int (*intor)(), void (*fill)(), double *eri,
                        int bastart, int bascount, int auxstart, int auxcount,
                        int ncomp, CINTOpt *cintopt,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nbasnaux = auxstart + auxcount;
        int *ao_loc = malloc(sizeof(int)*(nbasnaux+1));
        CINTshells_spheric_offset(ao_loc, bas, nbasnaux);
        ao_loc[nbasnaux] = ao_loc[nbasnaux-1] + CINTcgto_spheric(nbasnaux-1, bas);
        const int nao = ao_loc[auxstart];
        const int nshell = auxstart;

        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao, ao_loc};

        int i, j, ij;

#pragma omp parallel default(none) \
        shared(eri, intor, fill, bastart, bascount, auxstart, auxcount, \
               envs, cintopt) \
        private(ij, i, j)
#pragma omp for nowait schedule(dynamic, 2)
        for (ksh = 0; ksh < auxcount; ksh++) {
                (*fill)(intor, eri,
                        ksh, bastart, auxstart, auxcount,
                        cintopt, &envs);
        }

        free(ao_loc);
}
*/

/*
 * transform bra, s1 to label AO symmetry
 */
int RIhalfmmm_nr_s1_bra(double *vout, double *vin, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->nao;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        double *mo_coeff = envs->mo_coeff;

        dgemm_(&TRANS_N, &TRANS_N, &nao, &i_count, &nao,
               &D1, vin, &nao, mo_coeff+i_start*nao, &nao,
               &D0, vout, &nao);
        return 0;
}

/*
 * transform ket, s1 to label AO symmetry
 */
int RIhalfmmm_nr_s1_ket(double *vout, double *vin, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;

        dgemm_(&TRANS_T, &TRANS_N, &j_count, &nao, &nao,
               &D1, mo_coeff+j_start*nao, &nao, vin, &nao,
               &D0, vout, &j_count);
        return 0;
}

/*
 * transform bra, s2 to label AO symmetry
 */
int RIhalfmmm_nr_s2_bra(double *vout, double *vin, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->nao;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        double *mo_coeff = envs->mo_coeff;

        dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
               &D1, vin, &nao, mo_coeff+i_start*nao, &nao,
               &D0, vout, &nao);
        return 0;
}

/*
 * transform ket, s2 to label AO symmetry
 */
int RIhalfmmm_nr_s2_ket(double *vout, double *vin, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*j_count);
        int i, j;

        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, vin, &nao, mo_coeff+j_start*nao, &nao,
               &D0, buf, &nao);
        for (j = 0; j < nao; j++) {
                for (i = 0; i < j_count; i++) {
                        vout[i] = buf[i*nao+j];
                }
                vout += j_count;
        }
        free(buf);
        return 0;
}

/*
 * unpack the AO integrals and copy to vout, s2 to label AO symmetry
 */
int RImmm_nr_s2_copy(double *vout, double *vin, struct _AO2MOEnvs *envs,
                     int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->nao;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        int nao = envs->nao;
        int i, j;
        for (i = 0; i < nao; i++) {
                for (j = 0; j < i; j++) {
                        vout[i*nao+j] = vin[j];
                        vout[j*nao+i] = vin[j];
                }
                vout[i*nao+i] = vin[i];
                vin += nao;
        }
        return 0;
}
