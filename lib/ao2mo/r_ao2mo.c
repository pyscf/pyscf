/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <assert.h>

//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))


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
        double complex *mo_coeff;
};






/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 * s1, s2 here to label the AO symmetry
 */
int AO2MOmmm_r_s1_iltj(double complex *vout, double complex *eri, 
                       struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double complex Z0 = 0;
        const double complex Z1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i;
        double complex *mo_coeff = envs->mo_coeff;
        double complex *buf = malloc(sizeof(double complex)*n2c*i_count);
        double complex *mo1 = malloc(sizeof(double complex)*n2c*i_count);

        for (i = i_start*n2c; i < n2c*(i_start+i_count); i++) {
                mo1[i] = conj(mo_coeff[i]);
        }

        // C_pi (pq| = (iq|, where (pq| is in C-order
        zgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &Z1, eri, &n2c, mo1, &n2c,
               &Z0, buf, &n2c);
        zgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &Z1, mo_coeff+j_start*n2c, &n2c, buf, &n2c,
               &Z0, vout, &j_count);
        free(buf);
        free(mo1);
        return 0;
}
/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 */
int AO2MOmmm_r_s1_igtj(double complex *vout, double complex *eri,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double complex Z0 = 0;
        const double complex Z1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i;
        double complex *mo_coeff = envs->mo_coeff;
        double complex *buf = malloc(sizeof(double complex)*n2c*j_count);
        double complex *mo1 = malloc(sizeof(double complex)*n2c*i_count);

        for (i = i_start*n2c; i < n2c*(i_start+i_count); i++) {
                mo1[i] = conj(mo_coeff[i]);
        }

        // C_qj (pq| = (pj|, where (pq| is in C-order
        zgemm_(&TRANS_T, &TRANS_N, &j_count, &n2c, &n2c,
               &Z1, mo_coeff+j_start*n2c, &n2c, eri, &n2c,
               &Z0, buf, &j_count);
        zgemm_(&TRANS_N, &TRANS_N, &j_count, &i_count, &n2c,
               &Z1, buf, &j_count, mo1, &n2c,
               &Z0, vout, &j_count);
        free(buf);
        free(mo1);
        return 0;
}

/*
 * s2-AO integrals to s2-MO integrals
 * shape requirements:
 *      vout[:,bra_count*(bra_count+1)/2] and bra_count==ket_count,
 *      eri[:,nao*(nao+1)/2]
 * first s2 is the AO symmetry, second s2 is the MO symmetry
 */
int AO2MOmmm_r_s2_s2(double complex *vout, double complex *eri,
                     struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: assert(envs->bra_count == envs->ket_count);
                        return envs->bra_count * (envs->bra_count+1) / 2;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
//TODO: use time-reversal symmetry to unpack AO integrals
        return 0;
}

/*
 * s2-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*(nao+1)/2]
 */
int AO2MOmmm_r_s2_iltj(double complex *vout, double complex *eri,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
//TODO: use time-reversal symmetry to unpack AO integrals
        return 0;
}

/*
 * s2-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*(nao+1)/2]
 */
int AO2MOmmm_r_s2_igtj(double complex *vout, double complex *eri,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
//TODO: use time-reversal symmetry to unpack AO integrals
        return 0;
}





/*
 * ************************************************
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
void AO2MOtranse2_r_s1(int (*fmmm)(),
                       double complex *vout, double complex *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        unsigned long ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        unsigned long nao2 = envs->nao * envs->nao;
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, envs, 0);
}
void AO2MOtranse2_r_s2ij(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_r_s1(fmmm, vout, vin, row_id, envs);
}

void AO2MOtranse2_r_s2kl(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        unsigned long ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        unsigned long nao2 = nao*(nao+1)/2;
        double complex *buf = malloc(sizeof(double complex) * nao*nao);
//        NPzunpack_tril(nao, vin+nao2*row_id, buf, 0);
//FIXME: use time-reversal symmetry to unpack AO integrals
//        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
}
void AO2MOtranse2_r_s4(int (*fmmm)(),
                       double complex *vout, double complex *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_r_s2kl(fmmm, vout, vin, row_id, envs);
}


void AO2MOr_e2_drv(void (*ftranse2)(), int (*fmmm)(),
                   double complex *vout, double complex *vin,
                   double complex *mo_coeff,
                   int nijcount, int nao,
                   int i_start, int i_count, int j_start, int j_count)
{
        struct _AO2MOEnvs envs;
        envs.bra_start = i_start;
        envs.bra_count = i_count;
        envs.ket_start = j_start;
        envs.ket_count = j_count;
        envs.nao = nao;
        envs.mo_coeff = mo_coeff;

        int i;
#pragma omp parallel default(none) \
        shared(ftranse2, fmmm, vout, vin, nijcount, envs) \
        private(i)
#pragma omp for nowait schedule(static)
        for (i = 0; i < nijcount; i++) {
                (*ftranse2)(fmmm, vout, vin, i, &envs);
        }
}



