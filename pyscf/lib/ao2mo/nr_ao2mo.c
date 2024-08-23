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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
//#define NDEBUG

//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "nr_ao2mo.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))
// 9f or 7g or 5h functions should be enough
#define NCTRMAX         64
#define OUTPUTIJ        1
#define INPUT_IJ        2

/*
 * Denoting 2e integrals (ij|kl),
 * AO2MOnr_e1_drv transforms ij for ksh_start <= k shell < ksh_end.
 * The transformation C_pi C_qj (pq|k*) coefficients are stored in
 * mo_coeff, C_pi and C_qj are offset by i_start and i_count, j_start and j_count.
 * The output eri is an 2D array, ordered as (kl-AO-pair,ij-MO-pair) in
 * C-order.  Transposing is needed before calling AO2MOnr_e2_drv.
 *
 * AO2MOnr_e2_drv transforms kl for nijcount of ij pairs.
 * vin is assumed to be an C-array of (ij-MO-pair, kl-AO-pair)
 * vout is an C-array of (ij-MO-pair, kl-MO-pair)
 *
 * ftranse1 and ftranse2
 * ---------------------
 * AO2MOtranse1_nr_s4, AO2MOtranse1_nr_s2ij, AO2MOtranse1_nr_s2kl AO2MOtranse1_nr_s1
 * AO2MOtranse2_nr_s4, AO2MOtranse2_nr_s2ij, AO2MOtranse2_nr_s2kl AO2MOtranse2_nr_s1
 * Labels s4, s2, s1 are used to label the AO integral symmetry.  The
 * symmetry of transformed integrals are controled by function fmmm
 *
 * fmmm
 * ----
 * fmmm dim requirements:
 *                      | vout                          | eri
 * ---------------------+-------------------------------+-------------------
 *  AO2MOmmm_nr_s2_s2   | [:,bra_count*(bra_count+1)/2] | [:,nao*(nao+1)/2]
 *                      |    and bra_count==ket_count   |
 *  AO2MOmmm_nr_s2_iltj | [:,bra_count*ket_count]       | [:,nao*nao]
 *  AO2MOmmm_nr_s2_igtj | [:,bra_count*ket_count]       | [:,nao*nao]
 *  AO2MOmmm_nr_s1_iltj | [:,bra_count*ket_count]       | [:,nao*nao]
 *  AO2MOmmm_nr_s1_igtj | [:,bra_count*ket_count]       | [:,nao*nao]
 *
 * AO2MOmmm_nr_s1_iltj, AO2MOmmm_nr_s1_igtj, AO2MOmmm_nr_s2_s2,
 * AO2MOmmm_nr_s2_iltj, AO2MOmmm_nr_s2_igtj
 * Pick a proper function from the 5 kinds of AO2MO transformation.
 * 1. AO integral I_ij != I_ji, use
 *    AO2MOmmm_nr_s1_iltj or AO2MOmmm_nr_s1_igtj
 * 2. AO integral I_ij == I_ji, but the MO coefficients for bra and ket
 *    are different, use
 *    AO2MOmmm_nr_s2_iltj or AO2MOmmm_nr_s2_igtj
 * 3. AO integral I_ij == I_ji, and the MO coefficients are the same for
 *    bra and ket, use
 *    AO2MOmmm_nr_s2_s2
 *
 *      ftrans           |     allowed fmmm
 * ----------------------+---------------------
 *  AO2MOtranse1_nr_s4   |  AO2MOmmm_nr_s2_s2
 *  AO2MOtranse1_nr_s2ij |  AO2MOmmm_nr_s2_iltj
 *  AO2MOtranse2_nr_s4   |  AO2MOmmm_nr_s2_igtj
 *  AO2MOtranse2_nr_s2kl |
 * ----------------------+---------------------
 *  AO2MOtranse1_nr_s2kl |  AO2MOmmm_nr_s2_s2
 *  AO2MOtranse2_nr_s2ij |  AO2MOmmm_nr_s2_igtj
 *                       |  AO2MOmmm_nr_s2_iltj
 * ----------------------+---------------------
 *  AO2MOtranse1_nr_s1   |  AO2MOmmm_nr_s1_iltj
 *  AO2MOtranse2_nr_s1   |  AO2MOmmm_nr_s1_igtj
 *
 */


/* for m > n
 * calculate the upper triangle part (of Fortran order matrix)
 *   _        |------- n -------| _
 *   diag_off [ . . . . . . . . ] |
 *   _        [ . . . . . . . . ] m
 *            [   . . . . . . . ] |
 *            [     . . . . . . ] _
 */
void AO2MOdtriumm_o1(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int BLK = 48;
        int mstart = m - MAX(0, (m-diag_off)/BLK)*BLK;
        int nstart = mstart - diag_off;
        int nleft;

        dgemm_(&TRANS_T, &TRANS_N, &mstart, &n, &k,
               &D1, a, &k, b, &k, &D0, c, &m);

        for (; mstart < m; mstart+=BLK, nstart+=BLK) {
                nleft = n - nstart;

                dgemm_(&TRANS_T, &TRANS_N, &BLK, &nleft, &k,
                       &D1, a+mstart*k, &k, b+nstart*k, &k,
                       &D0, c+nstart*m+mstart, &m);
        }
}

/* for m < n
 * calculate the upper triangle part (of Fortran order matrix)
 *   _        |------- n -------| _
 *   diag_off [ . . . . . . . . ] |
 *   _        [ . . . . . . . . ] m
 *            [   . . . . . . . ] |
 *            [     . . . . . . ] _
 */
void AO2MOdtriumm_o2(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int BLK = 48;
        int nstart, nleft;
        int mend = diag_off;

        for (nstart = 0; nstart < m-diag_off-BLK; nstart+=BLK) {
                mend += BLK;
                dgemm_(&TRANS_T, &TRANS_N, &mend, &BLK, &k,
                       &D1, a, &k, b+nstart*k, &k,
                       &D0, c+nstart*m, &m);
        }
        nleft = n - nstart;
        dgemm_(&TRANS_T, &TRANS_N, &m, &nleft, &k,
               &D1, a, &k, b+nstart*k, &k,
               &D0, c+nstart*m, &m);
}


/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 * s1, s2 here to label the AO symmetry
 */
int AO2MOmmm_nr_s1_iltj(double *vout, double *eri, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->bra_count * envs->ket_count;
                case INPUT_IJ: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;

        // C_pi (pq| = (iq|, where (pq| is in C-order
        dgemm_(&TRANS_N, &TRANS_N, &nao, &i_count, &nao,
               &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
               &D0, buf, &nao);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &nao,
               &D1, mo_coeff+j_start*nao, &nao, buf, &nao,
               &D0, vout, &j_count);
        return 0;
}
/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 */
int AO2MOmmm_nr_s1_igtj(double *vout, double *eri, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->bra_count * envs->ket_count;
                case INPUT_IJ: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;

        // C_qj (pq| = (pj|, where (pq| is in C-order
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &nao, &nao,
               &D1, mo_coeff+j_start*nao, &nao, eri, &nao,
               &D0, buf, &j_count);
        dgemm_(&TRANS_N, &TRANS_N, &j_count, &i_count, &nao,
               &D1, buf, &j_count, mo_coeff+i_start*nao, &nao,
               &D0, vout, &j_count);
        return 0;
}

/*
 * s2-AO integrals to s2-MO integrals
 * shape requirements:
 *      vout[:,bra_count*(bra_count+1)/2] and bra_count==ket_count,
 *      eri[:,nao*(nao+1)/2]
 * first s2 is the AO symmetry, second s2 is the MO symmetry
 */
int AO2MOmmm_nr_s2_s2(double *vout, double *eri, double *buf,
                      struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: assert(envs->bra_count == envs->ket_count);
                               return envs->bra_count * (envs->bra_count+1) / 2;
                case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf1 = buf + nao*i_count;
        int i, j, ij;

        // C_pi (pq| = (iq|, where (pq| is in C-order
        dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
               &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
               &D0, buf, &nao);
        AO2MOdtriumm_o1(j_count, i_count, nao, 0,
                        mo_coeff+j_start*nao, buf, buf1);

        for (i = 0, ij = 0; i < i_count; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        vout[ij] = buf1[j];
                }
                buf1 += j_count;
        }
        return 0;
}

/*
 * s2-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*(nao+1)/2]
 */
int AO2MOmmm_nr_s2_iltj(double *vout, double *eri, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->bra_count * envs->ket_count;
                case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;

        // C_pi (pq| = (iq|, where (pq| is in C-order
        dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
               &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
               &D0, buf, &nao);
        // C_qj (iq| = (ij|
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &nao,
               &D1, mo_coeff+j_start*nao, &nao, buf, &nao,
               &D0, vout, &j_count);
        return 0;
}

/*
 * s2-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*(nao+1)/2]
 */
int AO2MOmmm_nr_s2_igtj(double *vout, double *eri, double *buf,
                        struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case OUTPUTIJ: return envs->bra_count * envs->ket_count;
                case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;

        // C_qj (pq| = (pj|, where (pq| is in C-order
        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, eri, &nao, mo_coeff+j_start*nao, &nao,
               &D0, buf, &nao);
        // C_pi (pj| = (ij|
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &nao,
               &D1, buf, &nao, mo_coeff+i_start*nao, &nao,
               &D0, vout, &j_count);
        return 0;
}

/*
 * transform bra, s1 to label AO symmetry
 */
int AO2MOmmm_bra_nr_s1(double *vout, double *vin, double *buf,
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
int AO2MOmmm_ket_nr_s1(double *vout, double *vin, double *buf,
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
int AO2MOmmm_bra_nr_s2(double *vout, double *vin, double *buf,
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
int AO2MOmmm_ket_nr_s2(double *vout, double *vin, double *buf,
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
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 * eris[ncomp,nkl,nao_pair_ij]
 */
static void s4_copy(double *eri, double *ints, int di, int dj, int dk, int dl,
                    int istride, size_t nao2)
{
        int i, j, k, l;
        double *pints, *peri, *peri1;
        switch (di) {
        case 1:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        pints = ints + di * dj * (l*dk+k);
                        for (j = 0; j < dj; j++) {
                                eri[j] = pints[j];
                        }
                        eri += nao2;
                } }
                break;
        case 2:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        pints = ints + di * dj * (l*dk+k);
                        peri = eri + istride;
                        for (j = 0; j < dj;j++) {
                                eri [j] = pints[j*2+0];
                                peri[j] = pints[j*2+1];
                        }
                        eri += nao2;
                } }
                break;
        case 3:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        pints = ints + di * dj * (l*dk+k);
                        peri  = eri + istride;
                        peri1 = peri + istride + 1;
                        for (j = 0; j < dj;j++) {
                                eri  [j] = pints[j*3+0];
                                peri [j] = pints[j*3+1];
                                peri1[j] = pints[j*3+2];
                        }
                        eri += nao2;
                } }
                break;
        default:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        pints = ints + di * dj * (l*dk+k);
                        peri = eri;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
//TODO: call nontemporal write to avoid write-allocate
                                        peri[j] = pints[j*di+i];
                                }
                                peri += istride + i;
                        }
                        eri += nao2;
                } }
        }
}
static void s4_set0(double *eri, double *nop,
                    int di, int dj, int dk, int dl,
                    int istride, size_t nao2)
{
        int i, j, k, l;
        double *peri, *peri1;
        switch (di) {
        case 1:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        for (j = 0; j < dj; j++) {
                                eri[j] = 0;
                        }
                        eri += nao2;
                } }
                break;
        case 2:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        peri = eri + istride;
                        for (j = 0; j < dj; j++) {
                                eri [j] = 0;
                                peri[j] = 0;
                        }
                        eri += nao2;
                } }
                break;
        case 3:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        peri  = eri + istride;
                        peri1 = peri + istride + 1;
                        for (j = 0; j < dj; j++) {
                                eri  [j] = 0;
                                peri [j] = 0;
                                peri1[j] = 0;
                        }
                        eri += nao2;
                } }
                break;
        default:
                for (k = 0; k < dk; k++) {
                for (l = 0; l < dl; l++) {
                        peri = eri;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
//TODO: call nontemporal write to avoid write-allocate
                                        peri[j] = 0;
                                }
                                peri += istride + i;
                        }
                        eri += nao2;
                } }
        }
}

static void s4_copy_keql(double *eri, double *ints,
                         int di, int dj, int dk, int dl,
                         int istride, size_t nao2)
{
        int i, j, k, l;
        double *pints, *peri;
        for (k = 0; k < dk; k++) {
        for (l = 0; l <= k; l++) {
                pints = ints + di * dj * (l*dk+k);
                peri = eri;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                peri[j] = pints[j*di+i];
                        }
                        peri += istride + i;
                }
                eri += nao2;
        } }
}
static void s4_set0_keql(double *eri, double *nop,
                         int di, int dj, int dk, int dl,
                         int istride, size_t nao2)
{
        int i, j, k, l;
        double *peri;
        for (k = 0; k < dk; k++) {
        for (l = 0; l <= k; l++) {
                peri = eri;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                peri[j] = 0;
                        }
                        peri += istride + i;
                }
                eri += nao2;
        } }
}
static void s4_copy_ieqj(double *eri, double *ints,
                         int di, int dj, int dk, int dl,
                         int istride, size_t nao2)
{
        int i, j, k, l;
        double *pints, *peri;
        for (k = 0; k < dk; k++) {
        for (l = 0; l < dl; l++) {
                pints = ints + di * dj * (l*dk+k);
                peri = eri;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                peri[j] = pints[j*di+i];
                        }
                        peri += istride + i;
                }
                eri += nao2;
        } }
}
static void s4_set0_ieqj(double *eri, double *nop,
                         int di, int dj, int dk, int dl,
                         int istride, size_t nao2)
{
        int i, j, k, l;
        double *peri;
        for (k = 0; k < dk; k++) {
        for (l = 0; l < dl; l++) {
                peri = eri;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                peri[j] = 0;
                        }
                        peri += istride + i;
                }
                eri += nao2;
        } }
}
static void s4_copy_keql_ieqj(double *eri, double *ints,
                              int di, int dj, int dk, int dl,
                              int istride, size_t nao2)
{
        int i, j, k, l;
        double *pints, *peri;
        for (k = 0; k < dk; k++) {
        for (l = 0; l <= k; l++) {
                pints = ints + di * dj * (l*dk+k);
                peri = eri;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                peri[j] = pints[j*di+i];
                        }
                        peri += istride + i;
                }
                eri += nao2;
        } }
}
static void s4_set0_keql_ieqj(double *eri, double *nop,
                              int di, int dj, int dk, int dl,
                              int istride, size_t nao2)
{
        int i, j, k, l;
        double *peri;
        for (k = 0; k < dk; k++) {
        for (l = 0; l <= k; l++) {
                peri = eri;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                peri[j] = 0;
                        }
                        peri += istride + i;
                }
                eri += nao2;
        } }
}
static void s2kl_copy_keql(double *eri, double *ints,
                           int di, int dj, int dk, int dl,
                           int istride, size_t nao2)
{
        int i, j, k, l;
        double *pints;
        for (k = 0; k < dk; k++) {
        for (l = 0; l <= k; l++) {
                pints = ints + di * dj * (l*dk+k);
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                eri[i*istride+j] = pints[j*di+i];
                        }
                }
                eri += nao2;
        } }
}
static void s2kl_set0_keql(double *eri, double *nop,
                           int di, int dj, int dk, int dl,
                           int istride, size_t nao2)
{
        int i, j, k, l;
        for (k = 0; k < dk; k++) {
        for (l = 0; l <= k; l++) {
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                eri[i*istride+j] = 0;
                        }
                }
                eri += nao2;
        } }
}
static void s1_copy(double *eri, double *ints,
                    int di, int dj, int dk, int dl,
                    int istride, size_t nao2)
{
        int i, j, k, l;
        double *pints;
        for (k = 0; k < dk; k++) {
        for (l = 0; l < dl; l++) {
                pints = ints + di * dj * (l*dk+k);
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                eri[i*istride+j] = pints[j*di+i];
                        }
                }
                eri += nao2;
        } }
}
static void s1_set0(double *eri, double *nop,
                    int di, int dj, int dk, int dl,
                    int istride, size_t nao2)
{
        int i, j, k, l;
        for (k = 0; k < dk; k++) {
        for (l = 0; l < dl; l++) {
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                eri[i*istride+j] = 0;
                        }
                }
                eri += nao2;
        } }
}

#define DISTR_INTS_BY(fcopy, fset0, istride) \
        if ((*fprescreen)(shls, envs->vhfopt, envs->atm, envs->bas, envs->env) && \
            (*intor)(buf, NULL, shls, envs->atm, envs->natm, \
                     envs->bas, envs->nbas, envs->env, envs->cintopt, NULL)) { \
                pbuf = buf; \
                for (icomp = 0; icomp < envs->ncomp; icomp++) { \
                        peri = eri + nao2 * nkl * icomp + ioff + ao_loc[jsh]; \
                        fcopy(peri, pbuf, di, dj, dk, dl, istride, nao2); \
                        pbuf += di * dj * dk * dl; \
                } \
        } else { \
                for (icomp = 0; icomp < envs->ncomp; icomp++) { \
                        peri = eri + nao2 * nkl * icomp + ioff + ao_loc[jsh]; \
                        fset0(peri, pbuf, di, dj, dk, dl, istride, nao2); \
                } \
        }

void AO2MOfill_nr_s1(int (*intor)(), int (*fprescreen)(),
                     double *eri, double *buf,
                     int nkl, int ish, struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int ioff = ao_loc[ish] * nao;
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp;
        int shls[4];
        double *pbuf, *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {
                // kl = k * (k+1) / 2 + l
                ksh = kl / envs->nbas;
                lsh = kl - ksh * envs->nbas;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;

                for (jsh = 0; jsh < envs->nbas; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        DISTR_INTS_BY(s1_copy, s1_set0, nao);
                }
                eri += nao2 * dk * dl;
        }
}

void AO2MOfill_nr_s2ij(int (*intor)(), int (*fprescreen)(),
                       double *eri, double *buf,
                       int nkl, int ish, struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * (nao+1) / 2;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int ioff = ao_loc[ish] * (ao_loc[ish]+1) / 2;
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp;
        int shls[4];
        double *pbuf = buf;
        double *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {
                // kl = k * (k+1) / 2 + l
                ksh = kl / envs->nbas;
                lsh = kl - ksh * envs->nbas;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;

                for (jsh = 0; jsh < ish; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        DISTR_INTS_BY(s4_copy, s4_set0, ao_loc[ish]+1);
                }

                jsh = ish;
                dj = di;
                shls[1] = jsh;
                DISTR_INTS_BY(s4_copy_ieqj, s4_set0_ieqj, ao_loc[ish]+1);
                eri += nao2 * dk * dl;
        }
}

void AO2MOfill_nr_s2kl(int (*intor)(), int (*fprescreen)(),
                       double *eri, double *buf,
                       int nkl, int ish, struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int ioff = ao_loc[ish] * nao;
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp;
        int shls[4];
        double *pbuf = buf;
        double *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {

        // kl = k * (k+1) / 2 + l
        ksh = (int)(sqrt(2*kl+.25) - .5 + 1e-7);
        lsh = kl - ksh * (ksh+1) / 2;
        dk = ao_loc[ksh+1] - ao_loc[ksh];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        shls[2] = ksh;
        shls[3] = lsh;

        if (ksh == lsh) {
                for (jsh = 0; jsh < envs->nbas; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        DISTR_INTS_BY(s2kl_copy_keql, s2kl_set0_keql, nao);
                }
                eri += nao2 * dk*(dk+1)/2;

        } else {

                for (jsh = 0; jsh < envs->nbas; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        DISTR_INTS_BY(s1_copy, s1_set0, nao);
                }
                eri += nao2 * dk * dl;
        } }
}

void AO2MOfill_nr_s4(int (*intor)(), int (*fprescreen)(),
                     double *eri, double *buf,
                     int nkl, int ish, struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * (nao+1) / 2;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int ioff = ao_loc[ish] * (ao_loc[ish]+1) / 2;
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp;
        int shls[4];
        double *pbuf = buf;
        double *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {

        // kl = k * (k+1) / 2 + l
        ksh = (int)(sqrt(2*kl+.25) - .5 + 1e-7);
        lsh = kl - ksh * (ksh+1) / 2;
        dk = ao_loc[ksh+1] - ao_loc[ksh];
        dl = ao_loc[lsh+1] - ao_loc[lsh];
        shls[2] = ksh;
        shls[3] = lsh;

        if (ksh == lsh) {
                for (jsh = 0; jsh < ish; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        DISTR_INTS_BY(s4_copy_keql, s4_set0_keql,
                                      ao_loc[ish]+1);
                }

                jsh = ish;
                dj = di;
                shls[1] = ish;
                DISTR_INTS_BY(s4_copy_keql_ieqj, s4_set0_keql_ieqj,
                              ao_loc[ish]+1);
                eri += nao2 * dk*(dk+1)/2;

        } else {

                for (jsh = 0; jsh < ish; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        DISTR_INTS_BY(s4_copy, s4_set0, ao_loc[ish]+1);
                }

                jsh = ish;
                dj = di;
                shls[1] = ish;
                DISTR_INTS_BY(s4_copy_ieqj, s4_set0_ieqj, ao_loc[ish]+1);
                eri += nao2 * dk * dl;
        } }
}

/*
 * ************************************************
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
void AO2MOtranse1_nr_s1(int (*fmmm)(), int row_id,
                        double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs)
{
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        size_t nao2 = envs->nao * envs->nao;
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, buf, envs, 0);
}

void AO2MOtranse1_nr_s2ij(int (*fmmm)(), int row_id,
                          double *vout, double *vin, double *buf,
                          struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        size_t nao2 = nao*(nao+1)/2;
        NPdunpack_tril(nao, vin+nao2*row_id, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
}
void AO2MOtranse1_nr_s2(int (*fmmm)(), int row_id,
                        double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_nr_s2ij(fmmm, row_id, vout, vin, buf, envs);
}

void AO2MOtranse1_nr_s2kl(int (*fmmm)(), int row_id,
                          double *vout, double *vin, double *buf,
                          struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_nr_s1(fmmm, row_id, vout, vin, buf, envs);
}

void AO2MOtranse1_nr_s4(int (*fmmm)(), int row_id,
                        double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_nr_s2ij(fmmm, row_id, vout, vin, buf, envs);
}


/*
 * ************************************************
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
void AO2MOtranse2_nr_s1(int (*fmmm)(), int row_id,
                        double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs)
{
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        size_t nao2 = (*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, buf, envs, 0);
}

void AO2MOtranse2_nr_s2ij(int (*fmmm)(), int row_id,
                          double *vout, double *vin, double *buf,
                          struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s1(fmmm, row_id, vout, vin, buf, envs);
}

void AO2MOtranse2_nr_s2kl(int (*fmmm)(), int row_id,
                          double *vout, double *vin, double *buf,
                          struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        size_t nao2 = (*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);
        NPdunpack_tril(nao, vin+nao2*row_id, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
}
void AO2MOtranse2_nr_s2(int (*fmmm)(), int row_id,
                        double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s2kl(fmmm, row_id, vout, vin, buf, envs);
}

void AO2MOtranse2_nr_s4(int (*fmmm)(), int row_id,
                        double *vout, double *vin, double *buf,
                        struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s2kl(fmmm, row_id, vout, vin, buf, envs);
}



/*
 * ************************************************
 * sort (shell-based) integral blocks then transform
 */
void AO2MOsortranse2_nr_s1(int (*fmmm)(), int row_id,
                           double *vout, double *vin, double *buf,
                           struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        size_t nao2 = (*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);
        int ish, jsh, di, dj;
        int i, j, ij;
        double *pbuf;

        vin += nao2 * row_id;
        ij = 0;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                for (jsh = 0; jsh < envs->nbas; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        pbuf = buf + ao_loc[ish] * nao + ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++, ij++) {
                                pbuf[i*nao+j] = vin[ij];
                        } }
                }
        }

        (*fmmm)(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
}

void AO2MOsortranse2_nr_s2ij(int (*fmmm)(), int row_id,
                             double *vout, double *vin, double *buf,
                             struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s1(fmmm, row_id, vout, vin, buf, envs);
}

void AO2MOsortranse2_nr_s2kl(int (*fmmm)(), int row_id,
                             double *vout, double *vin, double *buf,
                             struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        size_t ij_pair = (*fmmm)(NULL, NULL, buf, envs, OUTPUTIJ);
        size_t nao2 = (*fmmm)(NULL, NULL, buf, envs, INPUT_IJ);
        int ish, jsh, di, dj;
        int i, j, ij;
        double *pbuf;

        vin += nao2 * row_id;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                for (jsh = 0; jsh < ish; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        pbuf = buf + ao_loc[ish] * nao + ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pbuf[i*nao+j] = vin[i*dj+j];
                        } }
                        vin += di * dj;
                }

                // lower triangle block when ish == jsh
                pbuf = buf + ao_loc[ish] * nao + ao_loc[ish];
                for (ij = 0, i = 0; i < di; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        pbuf[i*nao+j] = vin[ij];
                } }
                vin += di * (di+1) / 2;
        }

        (*fmmm)(vout+ij_pair*row_id, buf, buf+nao*nao, envs, 0);
}
void AO2MOsortranse2_nr_s2(int (*fmmm)(), int row_id,
                           double *vout, double *vin, double *buf,
                           struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s2kl(fmmm, row_id, vout, vin, buf, envs);
}

void AO2MOsortranse2_nr_s4(int (*fmmm)(), int row_id,
                           double *vout, double *vin, double *buf,
                           struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s2kl(fmmm, row_id, vout, vin, buf, envs);
}

/*
 * ************************************************
 * combine ftrans and fmmm
 */

void AO2MOtrans_nr_s1_iltj(void *nop, int row_id,
                           double *vout, double *eri, double *buf,
                           struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s1(AO2MOmmm_nr_s1_iltj, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_s1_igtj(void *nop, int row_id,
                           double *vout, double *eri, double *buf,
                           struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s1(AO2MOmmm_nr_s1_igtj, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_sorts1_iltj(void *nop, int row_id,
                               double *vout, double *eri, double *buf,
                               struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s1(AO2MOmmm_nr_s1_iltj, row_id, vout, eri, buf,envs);
}

void AO2MOtrans_nr_sorts1_igtj(void *nop, int row_id,
                               double *vout, double *eri, double *buf,
                               struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s1(AO2MOmmm_nr_s1_igtj, row_id, vout, eri, buf,envs);
}

void AO2MOtrans_nr_s2_iltj(void *nop, int row_id,
                           double *vout, double *eri, double *buf,
                           struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s2kl(AO2MOmmm_nr_s2_iltj, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_s2_igtj(void *nop, int row_id,
                           double *vout, double *eri, double *buf,
                           struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s2kl(AO2MOmmm_nr_s2_igtj, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_s2_s2(void *nop, int row_id,
                         double *vout, double *eri, double *buf,
                         struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s2kl(AO2MOmmm_nr_s2_s2, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_sorts2_iltj(void *nop, int row_id,
                               double *vout, double *eri, double *buf,
                               struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s2kl(AO2MOmmm_nr_s2_iltj, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_sorts2_igtj(void *nop, int row_id,
                               double *vout, double *eri, double *buf,
                               struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s2kl(AO2MOmmm_nr_s2_igtj, row_id, vout, eri, buf, envs);
}

void AO2MOtrans_nr_sorts2_s2(void *nop, int row_id,
                             double *vout, double *eri, double *buf,
                             struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_nr_s2kl(AO2MOmmm_nr_s2_s2, row_id, vout, eri, buf,envs);
}

/*
 * ************************************************
 * Denoting 2e integrals (ij|kl),
 * transform ij for ksh_start <= k shell < ksh_end.
 * The transformation C_pi C_qj (pq|k*) coefficients are stored in
 * mo_coeff, C_pi and C_qj are offset by i_start and i_count, j_start and j_count
 *
 * The output eri is an 2D array, ordered as (kl-AO-pair,ij-MO-pair) in
 * C-order.  Transposing is needed before calling AO2MOnr_e2_drv.
 * eri[ncomp,nkl,mo_i,mo_j]
 */
void AO2MOnr_e1_drv(int (*intor)(), void (*fill)(), void (*ftrans)(), int (*fmmm)(),
                    double *eri, double *mo_coeff,
                    int klsh_start, int klsh_count, int nkl, int ncomp,
                    int *orbs_slice, int *ao_loc,
                    CINTOpt *cintopt, CVHFOpt *vhfopt,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int nao = ao_loc[nbas];
        double *eri_ao = malloc(sizeof(double) * nao*nao*nkl*ncomp);
        if (eri_ao == NULL) {
                fprintf(stderr, "malloc(%zu) failed in AO2MOnr_e1_drv\n",
                        sizeof(double) * nao*nao*nkl*ncomp);
                exit(1);
        }
        AO2MOnr_e1fill_drv(intor, fill, eri_ao, klsh_start, klsh_count,
                           nkl, ncomp, ao_loc, cintopt, vhfopt,
                           atm, natm, bas, nbas, env);
        AO2MOnr_e2_drv(ftrans, fmmm, eri, eri_ao, mo_coeff,
                       nkl*ncomp, nao, orbs_slice, ao_loc, nbas);
        free(eri_ao);
}

void AO2MOnr_e2_drv(void (*ftrans)(), int (*fmmm)(),
                    double *vout, double *vin, double *mo_coeff,
                    int nij, int nao, int *orbs_slice, int *ao_loc, int nbas)
{
        struct _AO2MOEnvs envs;
        envs.bra_start = orbs_slice[0];
        envs.bra_count = orbs_slice[1] - orbs_slice[0];
        envs.ket_start = orbs_slice[2];
        envs.ket_count = orbs_slice[3] - orbs_slice[2];
        envs.nao = nao;
        envs.nbas = nbas;
        envs.ao_loc = ao_loc;
        envs.mo_coeff = mo_coeff;

#pragma omp parallel default(none) \
        shared(ftrans, fmmm, vout, vin, nij, envs, nao, orbs_slice)
{
        int i;
        int i_count = envs.bra_count;
        int j_count = envs.ket_count;
        double *buf = malloc(sizeof(double) * (nao+i_count) * (nao+j_count));
#pragma omp for schedule(dynamic)
        for (i = 0; i < nij; i++) {
                (*ftrans)(fmmm, i, vout, vin, buf, &envs);
        }
        free(buf);
}
}

/*
 * The size of eri is ncomp*nkl*nao*nao, note the upper triangular part
 * may not be filled
 */
void AO2MOnr_e1fill_drv(int (*intor)(), void (*fill)(), double *eri,
                        int klsh_start, int klsh_count, int nkl, int ncomp,
                        int *ao_loc, CINTOpt *cintopt, CVHFOpt *vhfopt,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int i;
        int nao = ao_loc[nbas];
        int dmax = 0;
        for (i= 0; i< nbas; i++) {
                dmax = MAX(dmax, ao_loc[i+1]-ao_loc[i]);
        }
        struct _AO2MOEnvs envs = {natm, nbas, atm, bas, env, nao,
                                  klsh_start, klsh_count, 0, 0, 0, 0,
                                  ncomp, ao_loc, NULL, cintopt, vhfopt};
        int (*fprescreen)();
        if (vhfopt != NULL) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

#pragma omp parallel default(none) \
        shared(fill, fprescreen, eri, envs, intor, nkl, nbas, dmax, ncomp)
{
        int ish;
        double *buf = malloc(sizeof(double)*dmax*dmax*dmax*dmax*ncomp);
#pragma omp for schedule(dynamic, 1)
        for (ish = 0; ish < nbas; ish++) {
                (*fill)(intor, fprescreen, eri, buf, nkl, ish, &envs);
        }
        free(buf);
}
}

