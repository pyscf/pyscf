/*  Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
   
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
 *  Author: Oliver J. Backhouse <olbackhouse@gmail.com>
 *          Alejandro Santana-Bonilla <alejandro.santana_bonilla@kcl.ac.uk>
 *          George H. Booth <george.booth@kcl.ac.uk>
 */

#include<stdlib.h>
#include<stdbool.h>
#include<assert.h>
#include<math.h>
#include<stdio.h>

//#include "omp.h"
#include "config.h"
#include "vhf/fblas.h"
#include "ragf2.h"



/*
 *  Capital indices indicate the opposite spin to the lower case index
 */

/*
 *  exact ERI
 *  vv_xy = (xi|ja) [(yi|ja) + (yi|JA) - (yi|ja)]
 *  vev_xy = (xi|ja) [(yi|ja) - (yj|ia)] (ei + ej - ea) + (xi|ja) (yi|JA) (ei + eJ - eA)
 */
void AGF2uee_vv_vev_islice(double *xija,
                           double *xiJA,
                           double *e_i,
                           double *e_I,
                           double *e_a,
                           double *e_A,
                           double os_factor,
                           double ss_factor,
                           int nmo,
                           int noa,
                           int nob,
                           int nva,
                           int nvb,
                           int istart,
                           int iend,
                           double *vv,
                           double *vev)
{
    const double D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    const int nja = noa * nva;
    const int nJA = nob * nvb;
    const int nxi = nmo * noa;

#pragma omp parallel
{
    double *eja = calloc(noa*nva, sizeof(double));
    double *eJA = calloc(nob*nvb, sizeof(double));
    double *xia = calloc(nmo*noa*nva, sizeof(double));
    double *xja = calloc(nmo*noa*nva, sizeof(double));
    double *xJA = calloc(nmo*nob*nvb, sizeof(double));
    double *exJA = calloc(nmo*nob*nvb, sizeof(double));

    double *vv_priv = calloc(nmo*nmo, sizeof(double));
    double *vev_priv = calloc(nmo*nmo, sizeof(double));

    int i;

#pragma omp for
    for (i = istart; i < iend; i++) {
        // build xija
        AGF2slice_0i2(xija, nmo, noa, nja, i, xja);

        // build xiJA
        AGF2slice_0i2(xiJA, nmo, noa, nJA, i, xJA);

        // build xjia
        AGF2slice_0i2(xija, nxi, noa, nva, i, xia);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_i, e_a, noa, nva, eja);

        // build eiJA = ei + eJ - eA
        AGF2sum_inplace_ener(e_i[i], e_I, e_A, nob, nvb, eJA);

        // inplace xjia = xija - xjia
        AGF2sum_inplace(xja, xia, nmo*nja, ss_factor, -ss_factor);

        // vv_xy += xija * (yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vv_priv, &nmo);

        // vv_xy += xiJA * yiJA
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nJA, &os_factor, xJA, &nJA, xJA, &nJA, &D1, vv_priv, &nmo);

        // inplace xija = eija * xija
        AGF2prod_inplace_ener(eja, xja, nmo, nja);

        // outplace xiJA = eiJA * xiJA
        AGF2prod_outplace_ener(eJA, xJA, nmo, nJA, exJA);

        // vev_xy += xija * eija * (yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vev_priv, &nmo);

        // vev_xy += xiJA * eiJA * yiJA
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nJA, &os_factor, xJA, &nJA, exJA, &nJA, &D1, vev_priv, &nmo);
    }

    free(eja);
    free(eJA);
    free(xia);
    free(xja);
    free(xJA);
    free(exJA);

#pragma omp critical
    for (i = 0; i < (nmo*nmo); i++) {
        vv[i] += vv_priv[i];
        vev[i] += vev_priv[i];
    }

    free(vv_priv);
    free(vev_priv);
}
}


/*
 *  density fitting
 *  (xi|ja) = (xi|Q)(Q|ja)
 *  vv_xy = (xi|ja) [(yi|ja) + (yi|JA) - (yi|ja)]
 *  vev_xy = (xi|ja) [(yi|ja) - (yj|ia)] (ei + ej - ea) + (xi|ja) (yi|JA) (ei + eJ - eA)
 */
void AGF2udf_vv_vev_islice(double *qxi,
                           double *qja,
                           double *qJA,
                           double *e_i,
                           double *e_I,
                           double *e_a,
                           double *e_A,
                           double os_factor,
                           double ss_factor,
                           int nmo,
                           int noa,
                           int nob,
                           int nva,
                           int nvb,
                           int naux,
                           int istart,
                           int iend,
                           double *vv,
                           double *vev)
{
    const double D0 = 0.0;
    const double D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    const int nxi = nmo * noa;
    const int nja = noa * nva;
    const int nJA = nob * nvb;

#pragma omp parallel
{
    double *qa = calloc(naux*nva, sizeof(double));
    double *qx = calloc(naux*nmo, sizeof(double));
    double *eja = calloc(noa*nva, sizeof(double));
    double *eJA = calloc(nob*nvb, sizeof(double));
    double *xia = calloc(nmo*noa*nva, sizeof(double));
    double *xja = calloc(nmo*noa*nva, sizeof(double));
    double *xJA = calloc(nmo*nob*nvb, sizeof(double));
    double *exJA = calloc(nmo*nob*nvb, sizeof(double));

    double *vv_priv = calloc(nmo*nmo, sizeof(double));
    double *vev_priv = calloc(nmo*nmo, sizeof(double));

    int i;

#pragma omp for
    for (i = istart; i < iend; i++) {
        // build qx
        AGF2slice_01i(qxi, naux, nmo, noa, i, qx);

        // build qa
        AGF2slice_0i2(qja, naux, noa, nva, i, qa);

        // build xija = xq * qja
        dgemm_(&TRANS_N, &TRANS_T, &nja, &nmo, &naux, &D1, qja, &nja, qx, &nmo, &D0, xja, &nja);

        // build xiJA = xq * qJA
        dgemm_(&TRANS_N, &TRANS_T, &nJA, &nmo, &naux, &D1, qJA, &nJA, qx, &nmo, &D0, xJA, &nJA);

        // build xjia = xiq * qa
        dgemm_(&TRANS_N, &TRANS_T, &nva, &nxi, &naux, &D1, qa, &nva, qxi, &nxi, &D0, xia, &nva);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_i, e_a, noa, nva, eja);

        // build eiJA = ei + eJ - eA
        AGF2sum_inplace_ener(e_i[i], e_I, e_A, nob, nvb, eJA);

        // inplace xjia = xija - xjia
        AGF2sum_inplace(xja, xia, nmo*nja, ss_factor, -ss_factor);

        // vv_xy += xija * (yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vv_priv, &nmo);

        // vv_xy += xiJA * yiJA
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nJA, &os_factor, xJA, &nJA, xJA, &nJA, &D1, vv_priv, &nmo);

        // inplace xija = eija * xija
        AGF2prod_inplace_ener(eja, xja, nmo, nja);

        // outplace xiJA = eiJA * xiJA
        AGF2prod_outplace_ener(eJA, xJA, nmo, nJA, exJA);

        // vev_xy += xija * eija * (yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vev_priv, &nmo);

        // vev_xy += xiJA * eiJA * yiJA
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nJA, &os_factor, xJA, &nJA, exJA, &nJA, &D1, vev_priv, &nmo);
    }

    free(qa);
    free(qx);
    free(eja);
    free(eJA);
    free(xia);
    free(xja);
    free(xJA);
    free(exJA);

#pragma omp critical
    for (i = 0; i < (nmo*nmo); i++) {
        vv[i] += vv_priv[i];
        vev[i] += vev_priv[i];
    }

    free(vv_priv);
    free(vev_priv);
}
}


/*
 *  Removes an index from DGEMM and into a for loop to reduce the
 *  thread-private memory overhead, at the cost of serial speed
 */
void AGF2udf_vv_vev_islice_lowmem(double *qxi,
                                  double *qja,
                                  double *qJA,
                                  double *e_i,
                                  double *e_I,
                                  double *e_a,
                                  double *e_A,
                                  double os_factor,
                                  double ss_factor,
                                  int nmo,
                                  int noa,
                                  int nob,
                                  int nva,
                                  int nvb,
                                  int naux,
                                  int start,
                                  int end,
                                  double *vv,
                                  double *vev)
{
    const double D0 = 0.0;
    const double D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const int one = 1;

#pragma omp parallel
{
    double *qx_i = calloc(naux*nmo, sizeof(double));
    double *qx_j = calloc(naux*nmo, sizeof(double));
    double *qa_i = calloc(naux*nva, sizeof(double));
    double *qa_j = calloc(naux*nva, sizeof(double));
    double *qA_i = calloc(naux*nvb, sizeof(double));
    double *qA_j = calloc(naux*nvb, sizeof(double));
    double *xa_i = calloc(nmo*nva, sizeof(double));
    double *xa_j = calloc(nmo*nva, sizeof(double));
    double *xA_i = calloc(nmo*nvb, sizeof(double));
    double *xA_j = calloc(nmo*nvb, sizeof(double));
    double *ea = calloc(nva, sizeof(double));
    double *eA = calloc(nvb, sizeof(double));
    double *exA_i = calloc(nmo*nvb, sizeof(double));

    double *vv_priv = calloc(nmo*nmo, sizeof(double));
    double *vev_priv = calloc(nmo*nmo, sizeof(double));

    bool do_os, do_ss;
    int i, j, ij;

#pragma omp for
    for (ij = start; ij < end; ij++) {
        // i = 0 -> noa
        // j = 0 -> max(noa, nob)
        i = ij / ((noa > nob) ? noa : nob);
        j = ij % ((noa > nob) ? noa : nob);

        do_os = j < nob;
        do_ss = j < noa;

        // build qx_i
        AGF2slice_01i(qxi, naux, nmo, noa, i, qx_i);

        // build qx_j
        AGF2slice_01i(qxi, naux, nmo, noa, j, qx_j);

        // build qa_i
        AGF2slice_0i2(qja, naux, noa, nva, i, qa_i);

        // build qa_j
        AGF2slice_0i2(qja, naux, noa, nva, j, qa_j);

        if (do_ss) {
            // build xija
            dgemm_(&TRANS_N, &TRANS_T, &nva, &nmo, &naux, &D1, qa_i, &nva, qx_j, &nmo, &D0, xa_i, &nva);

            // build xjia
            dgemm_(&TRANS_N, &TRANS_T, &nva, &nmo, &naux, &D1, qa_j, &nva, qx_i, &nmo, &D0, xa_j, &nva);

            // build eija
            AGF2sum_inplace_ener(e_i[i], &(e_i[j]), e_a, one, nva, ea);

            // inplace xjia = xija - xjia
            AGF2sum_inplace(xa_i, xa_j, nmo*nva, ss_factor, -ss_factor);

            // vv_xy += xija * (yija - yjia)
            dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nva, &D1, xa_j, &nva, xa_i, &nva, &D1, vv_priv, &nmo);

            // inplace xija = eija * xija
            AGF2prod_inplace_ener(ea, xa_i, nmo, nva);

            // vev_xy += xija * eija * (yija - yjia)
            dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nva, &D1, xa_j, &nva, xa_i, &nva, &D1, vev_priv, &nmo);
        }

        if (do_os) {
            // build qA_j
            AGF2slice_0i2(qJA, naux, nob, nvb, j, qA_j);

            // build xiJA
            dgemm_(&TRANS_N, &TRANS_T, &nvb, &nmo, &naux, &D1, qA_j, &nvb, qx_i, &nmo, &D0, xA_i, &nvb);

            // build eiJA
            AGF2sum_inplace_ener(e_i[i], &(e_I[j]), e_A, one, nvb, eA);

            // vv_xy += xiJA * yiJA
            dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nvb, &os_factor, xA_i, &nvb, xA_i, &nvb, &D1, vv_priv, &nmo);

            // outplace xiJA = eiJA * xiJA
            AGF2prod_outplace_ener(eA, xA_i, nmo, nvb, exA_i);

            // vev_xy += xiJA * eiJA * yiJA
            dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nvb, &os_factor, xA_i, &nvb, exA_i, &nvb, &D1, vev_priv, &nmo);
        }
    }

    free(qx_i);
    free(qx_j);
    free(qa_i);
    free(qa_j);
    free(qA_i);
    free(qA_j);
    free(xa_i);
    free(xa_j);
    free(xA_i);
    free(xA_j);
    free(ea);
    free(eA);
    free(exA_i);

#pragma omp critical
    for (i = 0; i < (nmo*nmo); i++) {
        vv[i] += vv_priv[i];
        vev[i] += vev_priv[i];
    }

    free(vv_priv);
    free(vev_priv);
}
}
