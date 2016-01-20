/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * auxe2: (ij|P) where P is the auxiliary basis
 */

#include <stdlib.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "np_helper/np_helper.h"
#include "ao2mo/nr_ao2mo.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define OUTPUTIJ        1
#define INPUT_IJ        2


void RIfill_s1_auxe2(int (*intor)(), double *eri, size_t ijkoff,
                     int ncomp, int ish, int jsh, int naoaux,
                     int *basrange, int *iloc, int *jloc, int *kloc,
                     CINTOpt *cintopt, struct _VHFEnvs *envs)
{
        const int brastart = basrange[0];
        const int ketstart = basrange[2];
        const int auxstart = basrange[4];
        const int auxcount = basrange[5];
        const int nao = envs->nao;
        const int di = iloc[ish+1] - iloc[ish];
        const int dj = jloc[jsh+1] - jloc[jsh];
        const int dij = di * dj;
        const size_t nao2 = nao * nao;
        const size_t neri = nao2 * naoaux;
        double *eribuf = (double *)malloc(sizeof(double)*dij*naoaux*ncomp);

        int ksh, dk;
        int i, j, k, i0, j0, k0, icomp;
        int shls[3];
        size_t ij0;
        double *peri, *pbuf;

        shls[0] = brastart + ish;
        shls[1] = ketstart + jsh;

        for (ksh = 0; ksh < auxcount; ksh++) {
                shls[2] = auxstart + ksh;
                k0 = kloc[ksh  ];
                dk = kloc[ksh+1] - kloc[ksh];
                if ((*intor)(eribuf, shls, envs->atm, envs->natm,
                             envs->bas, envs->nbas, envs->env, cintopt)) {
                        for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = iloc[ish], i = 0; i < di; i++, i0++) {
                        for (j0 = jloc[jsh], j = 0; j < dj; j++, j0++) {
                                ij0 = i0 * nao + j0;
                                peri = eri + neri * icomp
                                     + ij0 * naoaux + k0 - ijkoff;
                                pbuf = eribuf + j * di + i;
                                for (k = 0; k < dk; k++) {
                                        peri[k] = pbuf[k*dij];
                                }
                        } } }
                } else {
                        for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i0 = iloc[ish]; i0 < iloc[ish+1]; i0++) {
                        for (j0 = jloc[jsh]; j0 < jloc[jsh+1]; j0++) {
                                ij0 = i0 * nao + j0;
                                peri = eri + neri * icomp
                                     + ij0 * naoaux + k0 - ijkoff;
                                for (k = 0; k < dk; k++) {
                                        peri[k] = 0;
                                }
                        } } }
                }
        }
        free(eribuf);
}

/*
 * [ \      ]
 * [  \     ]
 * [ ...    ]
 * [ ....   ]
 * [ .....  ]
 * [      \ ]
 */
void RIfill_s2ij_auxe2(int (*intor)(), double *eri, size_t ijkoff,
                       int ncomp, int ish, int jsh, int naoaux,
                       int *basrange, int *iloc, int *jloc, int *kloc,
                       CINTOpt *cintopt, struct _VHFEnvs *envs)
{
        if (iloc[ish] < jloc[jsh]) {
                return;
        }

        const int brastart = basrange[0];
        const int ketstart = basrange[2];
        const int auxstart = basrange[4];
        const int auxcount = basrange[5];
        const int di = iloc[ish+1] - iloc[ish];
        const int dj = jloc[jsh+1] - jloc[jsh];
        const int dij = di * dj;
        const int nao = envs->nao;
        const size_t nao2 = nao*(nao+1)/2;
        const size_t neri = nao2 * naoaux;
        double *eribuf = (double *)malloc(sizeof(double)*dij*naoaux*ncomp);

        int ksh, dk;
        int i, j, k, i0, j0, k0, icomp;
        int shls[3];
        size_t ij0;
        double *peri, *pbuf;

        shls[0] = brastart + ish;
        shls[1] = ketstart + jsh;

        for (ksh = 0; ksh < auxcount; ksh++) {
                shls[2] = auxstart + ksh;
                k0 = kloc[ksh  ];
                dk = kloc[ksh+1] - kloc[ksh];
                if ((*intor)(eribuf, shls, envs->atm, envs->natm,
                             envs->bas, envs->nbas, envs->env, cintopt)) {
                        if (iloc[ish] != jloc[jsh]) {
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i0 = iloc[ish], i = 0; i < di; i++,i0++) {
                                for (j0 = jloc[jsh], j = 0; j < dj; j++,j0++) {
                                        ij0 = i0*(i0+1)/2 + j0;
                                        peri = eri + neri * icomp
                                             + ij0 * naoaux + k0 - ijkoff;
                                        pbuf = eribuf + j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                peri[k] = pbuf[k*dij];
                                        }
                                } } }
                        } else {
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i0 = iloc[ish],i = 0; i < di; i++, i0++) {
                                for (j0 = jloc[jsh],j = 0; j0 <= i0; j++, j0++) {
                                        ij0 = i0*(i0+1)/2 + j0;
                                        peri = eri + neri * icomp
                                             + ij0 * naoaux + k0 - ijkoff;
                                        pbuf = eribuf + j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                peri[k] = pbuf[k*dij];
                                        }
                                } } }
                        }
                } else {
                        if (iloc[ish] != jloc[jsh]) {
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i0 = iloc[ish]; i0 < iloc[ish+1]; i0++) {
                                for (j0 = jloc[jsh]; j0 < jloc[jsh+1]; j0++) {
                                        ij0 = i0*(i0+1)/2 + j0;
                                        peri = eri + neri * icomp
                                             + ij0 * naoaux + k0 - ijkoff;
                                        for (k = 0; k < dk; k++) {
                                                peri[k] = 0;
                                        }
                                } } }
                        } else {
                                for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i0 = iloc[ish]; i0 < iloc[ish+1]; i0++) {
                                for (j0 = jloc[jsh]; j0 <= i0; j0++) {
                                        ij0 = i0*(i0+1)/2 + j0;
                                        peri = eri + ij0 * naoaux + k0 - ijkoff;
                                        for (k = 0; k < dk; k++) {
                                                peri[k] = 0;
                                        }
                                } } }
                        }
                }
        }
        free(eribuf);
}


/*
 * fill can be one of RIfill_s1_auxe2 and RIfill_s2ij_auxe2
 */
void RInr_3c2e_auxe2_drv(int (*intor)(), void (*fill)(), double *eri,
                         size_t ijkoff, int nao, int naoaux,
                         int *basrange, int *iloc, int *jloc, int *kloc,
                         int ncomp, CINTOpt *cintopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        assert(fill == &RIfill_s1_auxe2 || fill == &RIfill_s2ij_auxe2);
        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao};
#pragma omp parallel default(none) \
        shared(eri, intor, fill, basrange, iloc, jloc, kloc, \
               ijkoff, ncomp, naoaux, envs, cintopt)
{
        int i, j, ij;
        int bracount = basrange[1];
        int ketcount = basrange[3];
#pragma omp for nowait schedule(dynamic, 2)
        for (ij = 0; ij < bracount*ketcount; ij++) {
                i = ij / ketcount;
                j = ij - i * ketcount;
                (*fill)(intor, eri, ijkoff, ncomp, i, j, naoaux,
                        basrange, iloc, jloc, kloc, cintopt, &envs);
        }
}
}

/*
 * transform bra, s1 to label AO symmetry
 */
int RIhalfmmm_nr_s1_bra(double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
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
int RIhalfmmm_nr_s1_ket(double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->nao * envs->ket_count;
                case INPUT_IJ: return envs->nao * envs->nao;
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
int RIhalfmmm_nr_s2_bra(double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->bra_count * envs->nao;
                case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
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
int RIhalfmmm_nr_s2_ket(double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->nao * envs->ket_count;
                case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
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
        return 0;
}

/*
 * unpack the AO integrals and copy to vout, s2 to label AO symmetry
 */
int RImmm_nr_s2_copy(double *vout, double *vin, double *buf,
                     struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->nao * envs->nao;
                case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
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

