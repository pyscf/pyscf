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
#include <complex.h>
#include <math.h>
#include <assert.h>

//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "r_ao2mo.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))
#define NCTRMAX         128


/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 * s1, s2 here to label the AO symmetry
 */
int AO2MOmmm_r_iltj(double complex *vout, double complex *eri, 
                    struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i;
        double *buf1 = malloc(sizeof(double)*n2c*i_count*3);
        double *buf2 = buf1 + n2c*i_count;
        double *buf3 = buf2 + n2c*i_count;
        double *bufr, *bufi;
        double *mo1 = malloc(sizeof(double) * n2c*MAX(i_count,j_count)*2);
        double *mo2, *mo_r, *mo_i;
        double *eri_r = malloc(sizeof(double) * n2c*n2c*3);
        double *eri_i = eri_r + n2c*n2c;
        double *eri1  = eri_i + n2c*n2c;
        double *vout1, *vout2, *vout3;

        // Gauss complex multiplication, C_pi^* (pq| = (iq|, where (pq| is in C-order
        mo_r = envs->mo_r + i_start * n2c;
        mo_i = envs->mo_i + i_start * n2c;
        mo2 = mo1 + n2c*i_count;
        for (i = 0; i < n2c*i_count; i++) {
                mo1[i] = mo_r[i] - mo_i[i];
                mo2[i] =-mo_i[i] - mo_r[i];
        }
        for (i = 0; i < n2c*n2c; i++) {
                eri_r[i] = creal(eri[i]);
                eri_i[i] = cimag(eri[i]);
                eri1 [i] = eri_r[i] + eri_i[i];
        }
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri1, &n2c, mo_r, &n2c, &D0, buf1, &n2c);
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri_r, &n2c, mo2, &n2c, &D0, buf2, &n2c);
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri_i, &n2c, mo1, &n2c, &D0, buf3, &n2c);
        free(eri_r);

        // C_qj^* (iq| = (ij|
        bufr = buf3;
        bufi = buf2;
        for (i = 0; i < n2c*i_count; i++) {
                buf3[i] = buf1[i] - buf3[i];
                buf2[i] = buf1[i] + buf2[i];
        }
        for (i = 0; i < n2c*i_count; i++) {
                buf1[i] = bufr[i] + bufi[i];
        }
        mo_r = envs->mo_r + j_start * n2c;
        mo_i = envs->mo_i + j_start * n2c;
        mo2 = mo1 + n2c*j_count;
        for (i = 0; i < n2c*j_count; i++) {
                mo1[i] = mo_r[i] + mo_i[i];
                mo2[i] = mo_i[i] - mo_r[i];
        }
        vout1 = malloc(sizeof(double)*i_count*j_count*3);
        vout2 = vout1 + i_count * j_count;
        vout3 = vout2 + i_count * j_count;
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo_r, &n2c, buf1, &n2c, &D0, vout1, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo2, &n2c, bufr, &n2c, &D0, vout2, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo1, &n2c, bufi, &n2c, &D0, vout3, &j_count);
        for (i = 0; i < i_count*j_count; i++) {
                vout[i] = (vout1[i]-vout3[i]) + (vout1[i]+vout2[i])*_Complex_I;
        }
        free(vout1);
        free(buf1);
        free(mo1);
        return 0;
}
int AO2MOmmm_r_s1_iltj(double complex *vout, double complex *eri, 
                       struct _AO2MOEnvs *envs, int seekdim)
{
        return AO2MOmmm_r_iltj(vout, eri, envs, seekdim);
}

/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 */
int AO2MOmmm_r_igtj(double complex *vout, double complex *eri,
                    struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i;
        double *buf1 = malloc(sizeof(double)*n2c*j_count*3);
        double *buf2 = buf1 + n2c*j_count;
        double *buf3 = buf2 + n2c*j_count;
        double *bufr, *bufi;
        double *mo1 = malloc(sizeof(double) * n2c*MAX(i_count,j_count)*2);
        double *mo2, *mo_r, *mo_i;
        double *eri_r = malloc(sizeof(double) * n2c*n2c*3);
        double *eri_i = eri_r + n2c*n2c;
        double *eri1  = eri_i + n2c*n2c;
        double *vout1, *vout2, *vout3;

        // Gauss complex multiplication, C_qj (pq| = (pj|, where (pq| is in C-order
        for (i = 0; i < n2c*n2c; i++) {
                eri_r[i] = creal(eri[i]);
                eri_i[i] = cimag(eri[i]);
                eri1 [i] = eri_r[i] + eri_i[i];
        }
        mo_r = envs->mo_r + j_start * n2c;
        mo_i = envs->mo_i + j_start * n2c;
        mo2 = mo1 + n2c*j_count;
        for (i = 0; i < n2c*j_count; i++) {
                mo1[i] = mo_r[i] + mo_i[i];
                mo2[i] = mo_i[i] - mo_r[i];
        }

        dgemm_(&TRANS_T, &TRANS_N, &j_count, &n2c, &n2c,
               &D1, mo_r, &n2c, eri1, &n2c, &D0, buf1, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &n2c, &n2c,
               &D1, mo2, &n2c, eri_r, &n2c, &D0, buf2, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &n2c, &n2c,
               &D1, mo1, &n2c, eri_i, &n2c, &D0, buf3, &j_count);
        free(eri_r);

        bufr = buf3;
        bufi = buf2;
        for (i = 0; i < n2c*j_count; i++) {
                buf3[i] = buf1[i] - buf3[i];
                buf2[i] = buf1[i] + buf2[i];
        }
        for (i = 0; i < n2c*j_count; i++) {
                buf1[i] = bufr[i] + bufi[i];
        }
        mo_r = envs->mo_r + i_start * n2c;
        mo_i = envs->mo_i + i_start * n2c;
        mo2 = mo1 + n2c*i_count;
        for (i = 0; i < n2c*i_count; i++) {
                mo1[i] = mo_r[i] - mo_i[i];
                mo2[i] =-mo_i[i] - mo_r[i];
        }
        vout1 = malloc(sizeof(double)*i_count*j_count*3);
        vout2 = vout1 + i_count * j_count;
        vout3 = vout2 + i_count * j_count;
        dgemm_(&TRANS_N, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, buf1, &j_count, mo_r, &n2c, &D0, vout1, &j_count);
        dgemm_(&TRANS_N, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, bufr, &j_count, mo2, &n2c, &D0, vout2, &j_count);
        dgemm_(&TRANS_N, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, bufi, &j_count, mo1, &n2c, &D0, vout3, &j_count);
        for (i = 0; i < i_count*j_count; i++) {
                vout[i] = (vout1[i]-vout3[i]) + (vout1[i]+vout2[i])*_Complex_I;
        }
        free(vout1);
        free(buf1);
        free(mo1);
        return 0;
}
int AO2MOmmm_r_s1_igtj(double complex *vout, double complex *eri,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        return AO2MOmmm_r_igtj(vout, eri, envs, seekdim);
}


/*
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 * eris[ncomp,nkl,nao*nao]
 */
static void fill_s1(int (*intor)(), int (*fprescreen)(), double complex *eri,
                    int nkl, int ish, int jshtot, struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp, i, j, k, l, n;
        int shls[4];
        double complex *buf = malloc(sizeof(double complex)
                                     *di*nao*NCTRMAX*NCTRMAX*envs->ncomp);
        assert(buf);
        double complex *pbuf, *pbuf1, *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {
                ksh = kl / envs->nbas;
                lsh = kl - ksh * envs->nbas;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;

                pbuf = buf;
                for (jsh = 0; jsh < jshtot; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        n = di * dj * dk * dl * envs->ncomp;
                        if ((*fprescreen)(shls, envs->vhfopt,
                                          envs->atm, envs->bas, envs->env)) {
                                (*intor)(pbuf, NULL, shls, envs->atm, envs->natm,
                                         envs->bas, envs->nbas, envs->env,
                                         envs->cintopt, NULL);
                        } else {
                                NPzset0(pbuf, n);
                        }
                        pbuf += n;
                }

                pbuf = buf;
                for (jsh = 0; jsh < jshtot; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        for (icomp = 0; icomp < envs->ncomp; icomp++) {
                                peri = eri + nao2 * nkl * icomp
                                     + ao_loc[ish] * nao + ao_loc[jsh];
                                for (k = 0; k < dk; k++) {
                                for (l = 0; l < dl; l++) {
                                        pbuf1 = pbuf + di * dj * (l*dk+k);
                                        for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                                peri[i*nao+j] = pbuf1[j*di+i];
                                        } }
                                        peri += nao2;
                                } }
                                pbuf += di * dj * dk * dl;
                        }
                }
                eri += nao2 * dk * dl;
        }
        free(buf);
}

static void fill_s2(int (*intor)(), int (*fprescreen)(), double complex *eri,
                    int nkl, int ish, int jshtot, struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp, i, j, k, l, n;
        int shls[4];
        double complex *buf = malloc(sizeof(double complex)
                                     *di*nao*nkl*envs->ncomp);
        assert(buf);
        double complex *pbuf, *pbuf1, *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {
                // kl = k * (k+1) / 2 + l
                ksh = (int)(sqrt(2*kl+.25) - .5 + 1e-7);
                lsh = kl - ksh * (ksh+1) / 2;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;

                pbuf = buf;
                for (jsh = 0; jsh < jshtot; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        n = di * dj * dk * dl * envs->ncomp;
                        if ((*fprescreen)(shls, envs->vhfopt,
                                          envs->atm, envs->bas, envs->env)) {
                                (*intor)(pbuf, NULL, shls, envs->atm, envs->natm,
                                         envs->bas, envs->nbas, envs->env,
                                         envs->cintopt, NULL);
                        } else {
                                NPzset0(pbuf, n);
                        }
                        pbuf += n;
                }

                pbuf = buf;
                for (jsh = 0; jsh < jshtot; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        for (icomp = 0; icomp < envs->ncomp; icomp++) {
                                peri = eri + nao2 * nkl * icomp
                                     + ao_loc[ish] * nao + ao_loc[jsh];
                                for (k = 0; k < dk; k++) {
                                for (l = 0; l < dl; l++) {
                                        pbuf1 = pbuf + di * dj * (l*dk+k);
                                        for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                                peri[i*nao+j] = pbuf1[j*di+i];
                                        } }
                                        peri += nao2;
                                } }
                                pbuf += di * dj * dk * dl;
                        }
                }
                eri += nao2 * dk * dl;
        }
        free(buf);
}

void AO2MOfill_r_s1(int (*intor)(), int (*fprescreen)(),
                    double complex *eri, int nkl, int ish,
                    struct _AO2MOEnvs *envs)
{
        fill_s1(intor, fprescreen, eri, nkl, ish, envs->nbas, envs);
}

void AO2MOfill_r_s2ij(int (*intor)(), int (*fprescreen)(),
                      double complex *eri, int nkl, int ish,
                      struct _AO2MOEnvs *envs)
{
        fill_s1(intor, fprescreen, eri, nkl, ish, ish+1, envs);
}

void AO2MOfill_r_s2kl(int (*intor)(), int (*fprescreen)(),
                      double complex *eri, int nkl, int ish,
                      struct _AO2MOEnvs *envs)
{
        fill_s2(intor, fprescreen, eri, nkl, ish, envs->nbas, envs);
}

void AO2MOfill_r_s4(int (*intor)(), int (*fprescreen)(),
                    double complex *eri, int nkl, int ish,
                    struct _AO2MOEnvs *envs)
{
        fill_s2(intor, fprescreen, eri, nkl, ish, ish+1, envs);
}

void AO2MOfill_r_a2ij(int (*intor)(), int (*fprescreen)(),
                      double complex *eri, int nkl, int ish,
                      struct _AO2MOEnvs *envs)
{
        fill_s1(intor, fprescreen, eri, nkl, ish, ish+1, envs);
}

void AO2MOfill_r_a2kl(int (*intor)(), int (*fprescreen)(),
                      double complex *eri, int nkl, int ish,
                      struct _AO2MOEnvs *envs)
{
        fill_s2(intor, fprescreen, eri, nkl, ish, envs->nbas, envs);
}

void AO2MOfill_r_a4ij(int (*intor)(), int (*fprescreen)(),
                      double complex *eri, int nkl, int ish,
                      struct _AO2MOEnvs *envs)
{
        fill_s2(intor, fprescreen, eri, nkl, ish, ish+1, envs);
}

void AO2MOfill_r_a4kl(int (*intor)(), int (*fprescreen)(),
                      double complex *eri, int nkl, int ish,
                      struct _AO2MOEnvs *envs)
{
        fill_s2(intor, fprescreen, eri, nkl, ish, ish+1, envs);
}

void AO2MOfill_r_a4(int (*intor)(), int (*fprescreen)(),
                    double complex *eri, int nkl, int ish,
                    struct _AO2MOEnvs *envs)
{
        fill_s2(intor, fprescreen, eri, nkl, ish, ish+1, envs);
}


/*
 * time reversal symmetry for AOs
 * tao index start from 1
 */
#define BeginTimeRevLoop(I, J) \
        for (I##0 = I##start; I##0 < I##end;) { \
                I##1 = abs(tao[I##0]); \
                for (J##0 = J##start; J##0 < J##end;) { \
                        J##1 = abs(tao[J##0]);
#define EndTimeRevLoop(I, J) \
                        J##0 = J##1; } \
                I##0 = I##1; }
static void timerev_mat(double complex *mat, int *tao, int *ao_loc, int nbas)
{
        int nao = ao_loc[nbas];
        int ish, jsh, istart, iend, jstart, jend;
        int i, j, i0, j0, i1, j1;
        double complex *pmat, *pmat1, *pbuf, *pbuf1;

        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh < ish; jsh++) {
                istart = ao_loc[ish  ];
                iend   = ao_loc[ish+1];
                jstart = ao_loc[jsh  ];
                jend   = ao_loc[jsh+1];
                if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                        pbuf  = mat + j0 * nao + i0;
                        pbuf1 = pbuf + nao;
                        pmat  = mat + (i1-1)*nao + (j1-1);
                        pmat1 = pmat - nao;
                        for (j = 0; j < j1-j0; j+=2) {
                        for (i = 0; i < i1-i0; i+=2) {
                                pbuf [j*nao+i  ] = pmat [-i*nao-j  ];
                                pbuf1[j*nao+i  ] =-pmat [-i*nao-j-1];
                                pbuf [j*nao+i+1] =-pmat1[-i*nao-j  ];
                                pbuf1[j*nao+i+1] = pmat1[-i*nao-j-1];
                        } }
EndTimeRevLoop(i, j);
                } else {
BeginTimeRevLoop(i, j);
                        pbuf  = mat + j0 * nao + i0;
                        pbuf1 = pbuf + nao;
                        pmat  = mat + (i1-1)*nao + (j1-1);
                        pmat1 = pmat - nao;
                        for (j = 0; j < j1-j0; j+=2) {
                        for (i = 0; i < i1-i0; i+=2) {
                                pbuf [j*nao+i  ] =-pmat [-i*nao-j  ];
                                pbuf1[j*nao+i  ] = pmat [-i*nao-j-1];
                                pbuf [j*nao+i+1] = pmat1[-i*nao-j  ];
                                pbuf1[j*nao+i+1] =-pmat1[-i*nao-j-1];
                        } }
EndTimeRevLoop(i, j);
                }
        } }
}

static void atimerev_mat(double complex *mat, int *tao, int *ao_loc, int nbas)
{
        int nao = ao_loc[nbas];
        int ish, jsh, istart, iend, jstart, jend;
        int i, j, i0, j0, i1, j1;
        double complex *pmat, *pmat1, *pbuf, *pbuf1;

        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh < ish; jsh++) {
                istart = ao_loc[ish  ];
                iend   = ao_loc[ish+1];
                jstart = ao_loc[jsh  ];
                jend   = ao_loc[jsh+1];
                if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                        pbuf  = mat + j0 * nao + i0;
                        pbuf1 = pbuf + nao;
                        pmat  = mat + (i1-1)*nao + (j1-1);
                        pmat1 = pmat - nao;
                        for (j = 0; j < j1-j0; j+=2) {
                        for (i = 0; i < i1-i0; i+=2) {
                                pbuf [j*nao+i  ] =-pmat [-i*nao-j  ];
                                pbuf1[j*nao+i  ] = pmat [-i*nao-j-1];
                                pbuf [j*nao+i+1] = pmat1[-i*nao-j  ];
                                pbuf1[j*nao+i+1] =-pmat1[-i*nao-j-1];
                        } }
EndTimeRevLoop(i, j);
                } else {
BeginTimeRevLoop(i, j);
                        pbuf  = mat + j0 * nao + i0;
                        pbuf1 = pbuf + nao;
                        pmat  = mat + (i1-1)*nao + (j1-1);
                        pmat1 = pmat - nao;
                        for (j = 0; j < j1-j0; j+=2) {
                        for (i = 0; i < i1-i0; i+=2) {
                                pbuf [j*nao+i  ] = pmat [-i*nao-j  ];
                                pbuf1[j*nao+i  ] =-pmat [-i*nao-j-1];
                                pbuf [j*nao+i+1] =-pmat1[-i*nao-j  ];
                                pbuf1[j*nao+i+1] = pmat1[-i*nao-j-1];
                        } }
EndTimeRevLoop(i, j);
                }
        } }
}

static void copy_mat(double complex *buf, double complex *mat,
                     int *ao_loc, int nbas)
{
        int nao = ao_loc[nbas];
        int ish, istart, iend, i, j;

        for (ish = 0; ish < nbas; ish++) {
                istart = ao_loc[ish  ];
                iend   = ao_loc[ish+1];
                for (i = istart; i < iend; i++) {
                for (j = 0; j < iend; j++) {
                        buf[i*nao+j] = mat[i*nao+j];
                } }
        }
}


/*
 * ************************************************
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
void AO2MOtranse1_r_s1(int (*fmmm)(),
                       double complex *vout, double complex *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = envs->nao * envs->nao;
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, envs, 0);
}

void AO2MOtranse1_r_s2ij(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = nao * nao;
        double complex *buf = malloc(sizeof(double complex) * nao*nao);
        copy_mat(buf, vin+nao2*row_id, envs->ao_loc, envs->nbas);
        timerev_mat(buf, envs->tao, envs->ao_loc, envs->nbas);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
}

void AO2MOtranse1_r_s2kl(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_s1(fmmm, vout, vin, row_id, envs);
}

void AO2MOtranse1_r_s4(int (*fmmm)(),
                       double complex *vout, double complex *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_s2ij(fmmm, vout, vin, row_id, envs);
}

void AO2MOtranse1_r_a2ij(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = nao * nao;
        double complex *buf = malloc(sizeof(double complex) * nao*nao);
        copy_mat(buf, vin+nao2*row_id, envs->ao_loc, envs->nbas);
        atimerev_mat(buf, envs->tao, envs->ao_loc, envs->nbas);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
}

void AO2MOtranse1_r_a2kl(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_s1(fmmm, vout, vin, row_id, envs);
}

// anti-time-reversal between ij and time-reversal between kl
void AO2MOtranse1_r_a4ij(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_a2ij(fmmm, vout, vin, row_id, envs);
}

// time-reversal between ij and anti-time-reversal between kl
void AO2MOtranse1_r_a4kl(int (*fmmm)(),
                         double complex *vout, double complex *vin, int row_id,
                         struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_s2ij(fmmm, vout, vin, row_id, envs);
}

// anti-time-reversal between ij and anti-time-reversal between kl
void AO2MOtranse1_r_a4(int (*fmmm)(),
                       double complex *vout, double complex *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_a2ij(fmmm, vout, vin, row_id, envs);
}


void AO2MOtranse2_r_s1(int (*fmmm)(),
                       double complex *vout, double complex *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        AO2MOtranse1_r_s1(fmmm, vout, vin, row_id, envs);
}


/*
 * ************************************************
 * sort (shell-based) integral blocks then transform
 */
void AO2MOsortranse2_r_s1(int (*fmmm)(),
                          double complex *vout, double complex *vin,
                          int row_id, struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = envs->nao * envs->nao;
        int ish, jsh, di, dj;
        int i, j;
        double complex *buf = malloc(sizeof(double complex) * nao2);
        double complex *pbuf;

        vin += nao2 * row_id;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                for (jsh = 0; jsh < envs->nbas; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        pbuf = buf + ao_loc[ish] * nao + ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pbuf[i*nao+j] = vin[i*dj+j];
                        } }
                        vin += di * dj;
                }
        }

        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);

        free(buf);
}

void AO2MOsortranse2_r_s2ij(int (*fmmm)(),
                            double complex *vout, double complex *vin,
                            int row_id, struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_r_s1(fmmm, vout, vin, row_id, envs);
}

void AO2MOsortranse2_r_s2kl(int (*fmmm)(),
                            double complex *vout, double complex *vin,
                            int row_id, struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = 0;
        int ish, jsh, di, dj;
        int i, j;
        double complex *buf = malloc(sizeof(double complex) * nao * nao);
        double complex *pbuf;

        nao2 = nao * (nao+1) / 2;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                nao2 += di * (di-1) / 2; // upper triangle for diagonal shells
        }

        vin += nao2 * row_id;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                for (jsh = 0; jsh <= ish; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        pbuf = buf + ao_loc[ish] * nao + ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pbuf[i*nao+j] = vin[i*dj+j];
                        } }
                        vin += di * dj;
                }
        }

        timerev_mat(buf, envs->tao, envs->ao_loc, envs->nbas);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);

        free(buf);
}

void AO2MOsortranse2_r_s4(int (*fmmm)(),
                          double complex *vout, double complex *vin, 
                          int row_id, struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_r_s2kl(fmmm, vout, vin, row_id, envs);
}

void AO2MOsortranse2_r_a2ij(int (*fmmm)(),
                            double complex *vout, double complex *vin,
                            int row_id, struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_r_s1(fmmm, vout, vin, row_id, envs);
}

void AO2MOsortranse2_r_a2kl(int (*fmmm)(),
                            double complex *vout, double complex *vin,
                            int row_id, struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = 0;
        int ish, jsh, di, dj;
        int i, j;
        double complex *buf = malloc(sizeof(double complex) * nao * nao);
        double complex *pbuf;

        nao2 = nao * (nao+1) / 2;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                nao2 += di * (di-1) / 2; // upper triangle for diagonal shells
        }

        vin += nao2 * row_id;
        for (ish = 0; ish < envs->nbas; ish++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                for (jsh = 0; jsh <= ish; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        pbuf = buf + ao_loc[ish] * nao + ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pbuf[i*nao+j] = vin[i*dj+j];
                        } }
                        vin += di * dj;
                }
        }

        atimerev_mat(buf, envs->tao, envs->ao_loc, envs->nbas);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);

        free(buf);
}

// anti-time-reversal between ij and time-reversal between kl
void AO2MOsortranse2_r_a4ij(int (*fmmm)(),
                            double complex *vout, double complex *vin, 
                            int row_id, struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_r_s2kl(fmmm, vout, vin, row_id, envs);
}

// time-reversal between ij and anti-time-reversal between kl
void AO2MOsortranse2_r_a4kl(int (*fmmm)(),
                            double complex *vout, double complex *vin, 
                            int row_id, struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_r_a2kl(fmmm, vout, vin, row_id, envs);
}

// anti-time-reversal between ij and anti-time-reversal between kl
void AO2MOsortranse2_r_a4(int (*fmmm)(),
                          double complex *vout, double complex *vin, 
                          int row_id, struct _AO2MOEnvs *envs)
{
        AO2MOsortranse2_r_a2kl(fmmm, vout, vin, row_id, envs);
}


/*
 * Kramers pair should not be assumed
 */

void AO2MOr_e1_drv(int (*intor)(), void (*fill)(),
                   void (*ftrans)(), int (*fmmm)(),
                   double complex *eri, double complex *mo_coeff,
                   int klsh_start, int klsh_count, int nkl, int ncomp,
                   int *orbs_slice, int *tao, int *ao_loc,
                   CINTOpt *cintopt, CVHFOpt *vhfopt,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        const int i_start = orbs_slice[0];
        const int i_count = orbs_slice[1] - orbs_slice[0];
        const int j_start = orbs_slice[2];
        const int j_count = orbs_slice[3] - orbs_slice[2];
        int nao = ao_loc[nbas];
        int nmo = MAX(orbs_slice[1], orbs_slice[3]);
        int i;
        double *mo_r = malloc(sizeof(double) * nao * nmo);
        double *mo_i = malloc(sizeof(double) * nao * nmo);
        for (i = 0; i < nao*nmo; i++) {
                mo_r[i] = creal(mo_coeff[i]);
                mo_i[i] = cimag(mo_coeff[i]);
        }
        struct _AO2MOEnvs envs = {natm, nbas, atm, bas, env, nao,
                                  klsh_start, klsh_count,
                                  i_start, i_count, j_start, j_count,
                                  ncomp, tao, ao_loc, mo_coeff,
                                  mo_r, mo_i, cintopt, vhfopt};

        double complex *eri_ao = malloc(sizeof(double complex)
                                        * nao*nao*nkl*ncomp);
        if (eri_ao == NULL) {
                fprintf(stderr, "malloc(%zu) failed in AO2MOr_e1_drv\n",
                        sizeof(double complex) * nao*nao*nkl*ncomp);
                exit(1);
        }
        int ish, kl;
        int (*fprescreen)();
        if (vhfopt != NULL) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

#pragma omp parallel default(none) \
        shared(fill, fprescreen, eri_ao, envs, intor, nkl, nbas) \
        private(ish)
#pragma omp for nowait schedule(dynamic)
        for (ish = 0; ish < nbas; ish++) {
                (*fill)(intor, fprescreen, eri_ao, nkl, ish, &envs, 0);
        }

#pragma omp parallel default(none) \
        shared(ftrans, fmmm, eri, eri_ao, nkl, ncomp, envs) \
        private(kl)
#pragma omp for nowait schedule(static)
        for (kl = 0; kl < nkl*ncomp; kl++) {
                (*ftrans)(fmmm, eri, eri_ao, kl, &envs);
        }

        free(eri_ao);
        free(mo_r);
        free(mo_i);
}

void AO2MOr_e2_drv(void (*ftrans)(), int (*fmmm)(),
                   double complex *vout, double complex *vin,
                   double complex *mo_coeff,
                   int nijcount, int nao,
                   int *orbs_slice, int *tao, int *ao_loc, int nbas)
{
        int nmo = MAX(orbs_slice[1], orbs_slice[3]);
        int i;
        double *mo_r = malloc(sizeof(double) * nao * nmo);
        double *mo_i = malloc(sizeof(double) * nao * nmo);
        for (i = 0; i < nao*nmo; i++) {
                mo_r[i] = creal(mo_coeff[i]);
                mo_i[i] = cimag(mo_coeff[i]);
        }
        struct _AO2MOEnvs envs;
        envs.bra_start = orbs_slice[0];
        envs.bra_count = orbs_slice[1] - orbs_slice[0];
        envs.ket_start = orbs_slice[2];
        envs.ket_count = orbs_slice[3] - orbs_slice[2];
        envs.nao = nao;
        envs.nbas = nbas;
        envs.tao = tao;
        envs.ao_loc = ao_loc;
        envs.mo_coeff = mo_coeff;
        envs.mo_r = mo_r;
        envs.mo_i = mo_i;

#pragma omp parallel default(none) \
        shared(ftrans, fmmm, vout, vin, nijcount, envs) \
        private(i)
#pragma omp for nowait schedule(static)
        for (i = 0; i < nijcount; i++) {
                (*ftrans)(fmmm, vout, vin, i, &envs);
        }
        free(mo_r);
        free(mo_i);
}

