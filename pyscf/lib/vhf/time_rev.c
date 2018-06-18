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
#include <complex.h>
#include "cint.h"
#include "time_rev.h"

/* index start from 1 */
void CVHFtimerev_map(int *tao, int *bas, int nbas)
{
        int k, l, n, m, k0, dj, ib, kpa;

        k0 = 0;
        k = 0;
        for (ib = 0; ib < nbas; ib++) {
                l = bas(ANG_OF,ib);
                kpa = bas(KAPPA_OF,ib);
                if (l%2 == 0) {
                        for (n = 0; n < bas(NCTR_OF,ib); n++) {
                                if (kpa >= 0) {
                                        dj = 2 * l;
                                        k0 = k;
                                        for (m = 0; m < dj; m+=2) {
                                                tao[k  ] =-(k0+dj-m);
                                                tao[k+1] = (k0+dj-m-1);
                                                k += 2;
                                        }
                                }
                                if (kpa <= 0) {
                                        dj = 2 * l + 2;
                                        k0 = k;
                                        for (m = 0; m < dj; m+=2) {
                                                tao[k  ] =-(k0+dj-m);
                                                tao[k+1] = (k0+dj-m-1);
                                                k += 2;
                                        }
                                }
                        }
                } else {
                        for (n = 0; n < bas(NCTR_OF,ib); n++) {
                                if (kpa >= 0) {
                                        dj = 2 * l;
                                        k0 = k;
                                        for (m = 0; m < dj; m+=2) {
                                                tao[k  ] = (k0+dj-m);
                                                tao[k+1] =-(k0+dj-m-1);
                                                k += 2;
                                        }
                                }
                                if (kpa <= 0) {
                                        dj = 2 * l + 2;
                                        k0 = k;
                                        for (m = 0; m < dj; m+=2) {
                                                tao[k  ] = (k0+dj-m);
                                                tao[k+1] =-(k0+dj-m-1);
                                                k += 2;
                                        }
                                }
                        }
                }
        }
}

/*
 * time reverse mat_{i,j} to block_{Tj,Ti}
 *      mat[istart:iend,jstart:jend] -> block[:dj,:di]
 */
static void timerev_block_o1(double complex *block, double complex *mat, int *tao,
                             int istart, int iend, int jstart, int jend, int nao)
{
        //const int di = iend - istart;
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj + (j0-jstart);
                pblock1 = pblock + dj;
                pmat = mat + (i1-1)*nao + (j1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [i*dj+j  ] = pmat [-i*nao-j  ];
                        pblock [i*dj+j+1] =-pmat [-i*nao-j-1];
                        pblock1[i*dj+j  ] =-pmat1[-i*nao-j  ];
                        pblock1[i*dj+j+1] = pmat1[-i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj + (j0-jstart);
                pblock1 = pblock + dj;
                pmat = mat + (i1-1)*nao + (j1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [i*dj+j  ] =-pmat [-i*nao-j  ];
                        pblock [i*dj+j+1] = pmat [-i*nao-j-1];
                        pblock1[i*dj+j  ] = pmat1[-i*nao-j  ];
                        pblock1[i*dj+j+1] =-pmat1[-i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        }
}

void CVHFtimerev_block(double complex *block, double complex *mat, int *tao,
                       int istart, int iend, int jstart, int jend, int nao)
{
        timerev_block_o1(block, mat, tao, istart, iend, jstart, jend, nao);
}

void CVHFtimerev_blockT(double complex *block, double complex *mat, int *tao,
                        int istart, int iend, int jstart, int jend, int nao)
{
        const int di = iend - istart;
        //const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart) + (j0-jstart)*di;
                pblock1 = pblock + di;
                pmat = mat + (i1-1)*nao + (j1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [j*di+i  ] = pmat [-i*nao-j  ];
                        pblock1[j*di+i  ] =-pmat [-i*nao-j-1];
                        pblock [j*di+i+1] =-pmat1[-i*nao-j  ];
                        pblock1[j*di+i+1] = pmat1[-i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart) + (j0-jstart)*di;
                pblock1 = pblock + di;
                pmat = mat + (i1-1)*nao + (j1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [j*di+i  ] =-pmat [-i*nao-j  ];
                        pblock1[j*di+i  ] = pmat [-i*nao-j-1];
                        pblock [j*di+i+1] = pmat1[-i*nao-j  ];
                        pblock1[j*di+i+1] =-pmat1[-i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        }
}


void CVHFtimerev_i(double complex *block, double complex *mat, int *tao,
                   int istart, int iend, int jstart, int jend, int nao)
{
        //const int di = iend - istart;
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if (tao[istart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj+(j0-jstart);
                pblock1 = pblock + dj;
                pmat = mat + (i1-1)*nao+j0;
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pblock [i*dj+j] = pmat [-i*nao+j];
                        pblock1[i*dj+j] =-pmat1[-i*nao+j];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj+(j0-jstart);
                pblock1 = pblock + dj;
                pmat = mat + (i1-1)*nao+j0;
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pblock [i*dj+j] =-pmat [-i*nao+j];
                        pblock1[i*dj+j] = pmat1[-i*nao+j];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_iT(double complex *block, double complex *mat, int *tao,
                    int istart, int iend, int jstart, int jend, int nao)
{
        const int di = iend - istart;
        //const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat, *pmat1;

        if (tao[istart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)+(j0-jstart)*di;
                pmat = mat + (i1-1)*nao+j0;
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pblock[j*di+i  ] = pmat [-i*nao+j];
                        pblock[j*di+i+1] =-pmat1[-i*nao+j];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)+(j0-jstart)*di;
                pmat = mat + (i1-1)*nao+j0;
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pblock[j*di+i  ] =-pmat [-i*nao+j];
                        pblock[j*di+i+1] = pmat1[-i*nao+j];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_j(double complex *block, double complex *mat, int *tao,
                   int istart, int iend, int jstart, int jend, int nao)
{
        //const int di = iend - istart;
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;

        if (tao[jstart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj+(j0-jstart);
                pmat = mat + i0*nao+(j1-1);
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock[i*dj+j  ] = pmat[i*nao-j  ];
                        pblock[i*dj+j+1] =-pmat[i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj+(j0-jstart);
                pmat = mat + i0*nao+(j1-1);
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock[i*dj+j  ] =-pmat[i*nao-j  ];
                        pblock[i*dj+j+1] = pmat[i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_jT(double complex *block, double complex *mat, int *tao,
                    int istart, int iend, int jstart, int jend, int nao)
{
        const int di = iend - istart;
        //const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pblock1, *pmat;

        if (tao[jstart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)+(j0-jstart)*di;
                pblock1 = pblock + di;
                pmat = mat + i0*nao+(j1-1);
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [j*di+i] = pmat[i*nao-j  ];
                        pblock1[j*di+i] =-pmat[i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)+(j0-jstart)*di;
                pblock1 = pblock + di;
                pmat = mat + i0*nao+(j1-1);
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [j*di+i] =-pmat[i*nao-j  ];
                        pblock1[j*di+i] = pmat[i*nao-j-1];
                } }
EndTimeRevLoop(i, j);
        }
}

/*
 * mat_{i,j} += mat_{Tj,Ti}
 */
void CVHFtimerev_ijplus(double complex *block, double complex *mat, int *tao,
                        int istart, int iend, int jstart, int jend, int nao)
{
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *mat0, *mat1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj + (j0-jstart);
                pblock1 = pblock + dj;
                mat0 = mat + i0*nao+j0;
                mat1 = mat0 + nao;
                pmat = mat + (j1-1)*nao + (i1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [i*dj+j  ] = mat0[i*nao+j  ] + pmat [-j*nao-i  ];
                        pblock [i*dj+j+1] = mat0[i*nao+j+1] - pmat1[-j*nao-i  ];
                        pblock1[i*dj+j  ] = mat1[i*nao+j  ] - pmat [-j*nao-i-1];
                        pblock1[i*dj+j+1] = mat1[i*nao+j+1] + pmat1[-j*nao-i-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj + (j0-jstart);
                pblock1 = pblock + dj;
                mat0 = mat + i0*nao+j0;
                mat1 = mat0 + nao;
                pmat = mat + (j1-1)*nao + (i1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [i*dj+j  ] = mat0[i*nao+j  ] - pmat [-j*nao-i  ];
                        pblock [i*dj+j+1] = mat0[i*nao+j+1] + pmat1[-j*nao-i  ];
                        pblock1[i*dj+j  ] = mat1[i*nao+j  ] + pmat [-j*nao-i-1];
                        pblock1[i*dj+j+1] = mat1[i*nao+j+1] - pmat1[-j*nao-i-1];
                } }
EndTimeRevLoop(i, j);
        }
}
/*
 * mat_{i,j} -= mat_{Tj,Ti}
 */
void CVHFtimerev_ijminus(double complex *block, double complex *mat, int *tao,
                         int istart, int iend, int jstart, int jend, int nao)
{
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *mat0, *mat1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj + (j0-jstart);
                pblock1 = pblock + dj;
                mat0 = mat + i0*nao+j0;
                mat1 = mat0 + nao;
                pmat = mat + (j1-1)*nao + (i1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [i*dj+j  ] = mat0[i*nao+j  ] - pmat [-j*nao-i  ];
                        pblock [i*dj+j+1] = mat0[i*nao+j+1] + pmat1[-j*nao-i  ];
                        pblock1[i*dj+j  ] = mat1[i*nao+j  ] + pmat [-j*nao-i-1];
                        pblock1[i*dj+j+1] = mat1[i*nao+j+1] - pmat1[-j*nao-i-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj + (j0-jstart);
                pblock1 = pblock + dj;
                mat0 = mat + i0*nao+j0;
                mat1 = mat0 + nao;
                pmat = mat + (j1-1)*nao + (i1-1);
                pmat1 = pmat - nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pblock [i*dj+j  ] = mat0[i*nao+j  ] + pmat [-j*nao-i  ];
                        pblock [i*dj+j+1] = mat0[i*nao+j+1] - pmat1[-j*nao-i  ];
                        pblock1[i*dj+j  ] = mat1[i*nao+j  ] - pmat [-j*nao-i-1];
                        pblock1[i*dj+j+1] = mat1[i*nao+j+1] + pmat1[-j*nao-i-1];
                } }
EndTimeRevLoop(i, j);
        }
}

void CVHFtimerev_adbak_block(double complex *block, double complex *mat, int *tao,
                             int istart, int iend, int jstart, int jend, int nao)
{
        //const int di = iend - istart;
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1)*dj + (j1-jstart-1);
                pblock1 = pblock - dj;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat [i*nao+j  ] += pblock [-i*dj-j  ];
                        pmat [i*nao+j+1] -= pblock [-i*dj-j-1];
                        pmat1[i*nao+j  ] -= pblock1[-i*dj-j  ];
                        pmat1[i*nao+j+1] += pblock1[-i*dj-j-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1)*dj + (j1-jstart-1);
                pblock1 = pblock - dj;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat [i*nao+j  ] -= pblock [-i*dj-j  ];
                        pmat [i*nao+j+1] += pblock [-i*dj-j-1];
                        pmat1[i*nao+j  ] += pblock1[-i*dj-j  ];
                        pmat1[i*nao+j+1] -= pblock1[-i*dj-j-1];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_adbak_blockT(double complex *block, double complex *mat, int *tao,
                              int istart, int iend, int jstart, int jend, int nao)
{
        const int di = iend - istart;
        //const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if ((tao[jstart]<0) == (tao[istart]<0)) {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1) + (j1-jstart-1)*di;
                pblock1 = pblock - di;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat [i*nao+j  ] += pblock [-j*di-i  ];
                        pmat [i*nao+j+1] -= pblock1[-j*di-i  ];
                        pmat1[i*nao+j  ] -= pblock [-j*di-i-1];
                        pmat1[i*nao+j+1] += pblock1[-j*di-i-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1) + (j1-jstart-1)*di;
                pblock1 = pblock - di;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat [i*nao+j  ] -= pblock [-j*di-i  ];
                        pmat [i*nao+j+1] += pblock1[-j*di-i  ];
                        pmat1[i*nao+j  ] += pblock [-j*di-i-1];
                        pmat1[i*nao+j+1] -= pblock1[-j*di-i-1];
                } }
EndTimeRevLoop(i, j);
        }
}


void CVHFtimerev_adbak_i(double complex *block, double complex *mat, int *tao,
                         int istart, int iend, int jstart, int jend, int nao)
{
        //const int di = iend - istart;
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;
        double complex *pblock1, *pmat1;

        if (tao[istart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1)*dj+(j0-jstart);
                pblock1 = pblock - dj;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pmat [i*nao+j] -= pblock [-i*dj+j];
                        pmat1[i*nao+j] += pblock1[-i*dj+j];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1)*dj+(j0-jstart);
                pblock1 = pblock - dj;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pmat [i*nao+j] += pblock [-i*dj+j];
                        pmat1[i*nao+j] -= pblock1[-i*dj+j];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_adbak_iT(double complex *block, double complex *mat, int *tao,
                          int istart, int iend, int jstart, int jend, int nao)
{
        const int di = iend - istart;
        //const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat, *pmat1;

        if (tao[istart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1)+(j0-jstart)*di;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pmat [i*nao+j] -= pblock[j*di-i  ];
                        pmat1[i*nao+j] += pblock[j*di-i-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i1-istart-1)+(j0-jstart)*di;
                pmat = mat + i0*nao + j0;
                pmat1 = pmat + nao;
                for (i = 0; i < i1-i0; i+=2) {
                for (j = 0; j < j1-j0; j++) {
                        pmat [i*nao+j] += pblock[j*di-i  ];
                        pmat1[i*nao+j] -= pblock[j*di-i-1];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_adbak_j(double complex *block, double complex *mat, int *tao,
                         int istart, int iend, int jstart, int jend, int nao)
{
        //const int di = iend - istart;
        const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pmat;

        if (tao[jstart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj+(j1-jstart-1);
                pmat = mat + i0*nao + j0;
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat[i*nao+j  ] -= pblock[i*dj-j  ];
                        pmat[i*nao+j+1] += pblock[i*dj-j-1];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)*dj+(j1-jstart-1);
                pmat = mat + i0*nao + j0;
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat[i*nao+j  ] += pblock[i*dj-j  ];
                        pmat[i*nao+j+1] -= pblock[i*dj-j-1];
                } }
EndTimeRevLoop(i, j);
        }
}
void CVHFtimerev_adbak_jT(double complex *block, double complex *mat, int *tao,
                          int istart, int iend, int jstart, int jend, int nao)
{
        const int di = iend - istart;
        //const int dj = jend - jstart;
        int i, j, i0, j0, i1, j1;
        double complex *pblock, *pblock1, *pmat;

        if (tao[jstart] < 0) {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)+(j1-jstart-1)*di;
                pblock1 = pblock - di;
                pmat = mat + i0*nao + j0;
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat[i*nao+j  ] -= pblock [i-j*di];
                        pmat[i*nao+j+1] += pblock1[i-j*di];
                } }
EndTimeRevLoop(i, j);
        } else {
BeginTimeRevLoop(i, j);
                pblock = block + (i0-istart)+(j1-jstart-1)*di;
                pblock1 = pblock - di;
                pmat = mat + i0*nao + j0;
                for (i = 0; i < i1-i0; i++) {
                for (j = 0; j < j1-j0; j+=2) {
                        pmat[i*nao+j  ] += pblock [i-j*di];
                        pmat[i*nao+j+1] -= pblock1[i-j*di];
                } }
EndTimeRevLoop(i, j);
        }
}

/* reorder the matrix elements, the smallest elements in the new matrix
 * is time-reversal block. So the new matrix may have time-reversal
 * symmetry
 *      b_{block(ib,jb)[j,i]} = a_{i0,j0}
 * in each time-reversal symmetry block, elements are ordered in F-contiguous.
 */
void CVHFreblock_mat(double complex *a, double complex *b,
                     int *bloc, int nbloc, int nao)
{
        int ib, jb, i0, j0, di, dj, i, j;
        double complex *pb, *pa;

        for (ib = 0; ib < nbloc; ib++) {
        for (jb = 0; jb < nbloc; jb++) {
                i0 = bloc[ib];
                j0 = bloc[jb];
                di = bloc[ib+1] - i0;
                dj = bloc[jb+1] - j0;
                pa = a + i0*nao + j0;
                pb = b + i0*nao + di*j0;
                for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                        pb[j*di+i] = pa[i*nao+j];
                } }
        } }
}
/*
 * a_{i0,j0} = b_{block(ib,jb)[j,i]}
 */
void CVHFunblock_mat(double complex *a, double complex *b,
                     int *bloc, int nbloc, int nao)
{
        int ib, jb, i0, j0, di, dj, i, j;
        double complex *pb, *pa;

        for (ib = 0; ib < nbloc; ib++) {
        for (jb = 0; jb < nbloc; jb++) {
                i0 = bloc[ib];
                j0 = bloc[jb];
                di = bloc[ib+1] - i0;
                dj = bloc[jb+1] - j0;
                pa = a + i0*nao + j0;
                pb = b + i0*nao + di*j0;
                for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                        pa[i*nao+j] = pb[j*di+i];
                } }
        } }
}
