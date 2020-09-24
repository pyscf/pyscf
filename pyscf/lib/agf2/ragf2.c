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
 *          George H. Booth <george.booth@kcl.ac.uk>
 */

#include<stdlib.h>
#include<assert.h>
#include<math.h>

//#include "omp.h"
#include "config.h"
#include "vhf/fblas.h"



/*
 *  b_x = alpha * a_x + beta * b_x
 */
void AGF2sum_inplace(double *a,
                     double *b,
                     int x,
                     double alpha,
                     double beta)
{
    //TODO: can we just use blas for this?

    for (int i = 0; i < x; i++) {
        b[i] *= beta;
        b[i] += alpha * a[i];
    }
}


/*
 *  b_x = a_x * b_x
 */
void AGF2prod_inplace(double *a,
                      double *b,
                      int x)
{
    for (int i = 0; i < x; i++) {
        b[i] *= a[i];
    }
}


/*
 *  c_x = a_x * b_x
 */
void AGF2prod_outplace(double *a,
                       double *b,
                       int x,
                       double *c)
{
    for (int i = 0; i < x; i++) {
        c[i] = a[i] * b[i];
    }
}


/*
 *  b_xz = a_xiz
 */
void AGF2slice_0i2(double *a,
                   int x,
                   int y,
                   int z,
                   int idx,
                   double *b)
{
    double *pa, *pb;

    for (int i = 0; i < x; i++) {
        pb = b + i*z;
        pa = a + i*y*z + idx*z;
        for (int k = 0; k < z; k++) {
            pb[k] = pa[k];
        }
    }
}


/*
 *  b_xy = a_xyi
 */
void AGF2slice_01i(double *a,
                   int x,
                   int y,
                   int z,
                   int idx,
                   double *b)
{
    double *pa, *pb;

    for (int i = 0; i < x; i++) {
        pb = b + i*y;
        pa = a + i*y*z + idx;
        for (int j = 0; j < y; j++) {
            pb[j] = pa[j*z];
        }
    }
}


/*
 *  d_xy = a + b_x - c_y
 */
void AGF2sum_inplace_ener(double a,
                  double *b,
                  double *c,
                  int x,
                  int y,
                  double *d)
{
    double *pd;

    for (int i = 0; i < x; i++) {
        pd = d + i*y;
        for (int j = 0; j < y; j++) {
            pd[j] = a + b[i] - c[j];
        }
    }
}


/*
 *  b_xy = a_y * b_xy
 */
void AGF2prod_inplace_ener(double *a,
                           double *b,
                           int x,
                           int y)
{
    double *pb;

    for (int i = 0; i < x; i++) {
        pb = b + i*y;
        AGF2prod_inplace(a, pb, y);
    }
}


/*
 *   c_xy = a_y * b_xy
 */
void AGF2prod_outplace_ener(double *a,
                            double *b,
                            int x,
                            int y,
                            double *c)
{
    double *pb, *pc;

    for (int i = 0; i < x; i++) {
        pb = b + i*y;
        pc = c + i*y;
        AGF2prod_outplace(a, pb, y, pc);
    }
}


/*
 *  exact ERI
 *  vv_xy = (xi|ja) [2(yi|ja) - (yj|ia)]
 *  vev_xy = (xi|ja) [2(yi|ja) - (yj|ia)] (ei + ej - ea)
 */
void AGF2ee_vv_vev_islice(double *xija,
                          double *e_i,
                          double *e_a,
                          double os_factor,
                          double ss_factor,
                          int nmo,
                          int nocc,
                          int nvir,
                          int istart,
                          int iend,
                          double *vv,
                          double *vev)
{
    const double D1 = 1;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    const int nja = nocc * nvir;
    const int nxi = nmo * nocc;
    const double fpos = os_factor + ss_factor;
    const double fneg = -1.0 * ss_factor;

#pragma omp parallel
{
    double *eja = calloc(nocc*nvir, sizeof(double));
    double *xia = calloc(nmo*nocc*nvir, sizeof(double));
    double *xja = calloc(nmo*nocc*nvir, sizeof(double));

    double *vv_priv = calloc(nmo*nmo, sizeof(double));
    double *vev_priv = calloc(nmo*nmo, sizeof(double));

#pragma omp for
    for (int i = istart; i < iend; i++) {
        // build xija
        AGF2slice_0i2(xija, nmo, nocc, nja, i, xja);

        // build xjia
        AGF2slice_0i2(xija, nxi, nocc, nvir, i, xia);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_i, e_a, nocc, nvir, eja);

        // inplace xjia = 2 * xija - xjia
        AGF2sum_inplace(xja, xia, nmo*nja, fpos, fneg);

        // vv_xy += xija * (2 yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vv_priv, &nmo);

        // inplace xija = eija * xija
        AGF2prod_inplace_ener(eja, xja, nmo, nja);

        // vev_xy += xija * eija * (2 yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vev_priv, &nmo);
    }

    free(eja);
    free(xia);
    free(xja);

#pragma omp critical
    for (int i = 0; i < (nmo*nmo); i++) {
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
 *  vv_xy = (xi|ja) [2(yi|ja) - (yj|ia)]
 *  vev_xy = (xi|ja) [2(yi|ja) - (yj|ia)] (ei + ej - ea)
 */
void AGF2df_vv_vev_islice(double *qxi,
                          double *qja,
                          double *e_i,
                          double *e_a,
                          double os_factor,
                          double ss_factor,
                          int nmo,
                          int nocc,
                          int nvir,
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

    const int nja = nocc * nvir;
    const int nxi = nmo * nocc;
    const double fpos = os_factor + ss_factor;
    const double fneg = -1.0 * ss_factor;

#pragma omp parallel
{
    double *qa = calloc(naux*nvir, sizeof(double));
    double *qx = calloc(naux*nmo, sizeof(double));
    double *eja = calloc(nocc*nvir, sizeof(double));
    double *xia = calloc(nmo*nocc*nvir, sizeof(double));
    double *xja = calloc(nmo*nocc*nvir, sizeof(double));

    double *vv_priv = calloc(nmo*nmo, sizeof(double));
    double *vev_priv = calloc(nmo*nmo, sizeof(double));

#pragma omp for
    for (int i = istart; i < iend; i++) {
        // build qx
        AGF2slice_01i(qxi, naux, nmo, nocc, i, qx);

        // build qa
        AGF2slice_0i2(qja, naux, nocc, nvir, i, qa);

        // build xija = xq * qja
        dgemm_(&TRANS_N, &TRANS_T, &nja, &nmo, &naux, &D1, qja, &nja, qx, &nmo, &D0, xja, &nja);

        // build xjia = xiq * qa
        dgemm_(&TRANS_N, &TRANS_T, &nvir, &nxi, &naux, &D1, qa, &nvir, qxi, &nxi, &D0, xia, &nvir);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_i, e_a, nocc, nvir, eja);

        // inplace xjia = 2 * xija - xjia
        AGF2sum_inplace(xja, xia, nmo*nja, fpos, fneg);

        // vv_xy += xija * (2 yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vv_priv, &nmo);

        // inplace xija = eija * xija
        AGF2prod_inplace_ener(eja, xja, nmo, nja);

        // vev_xy += xija * eija * (2 yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, vev_priv, &nmo);
    }

    free(qa);
    free(qx);
    free(eja);
    free(xia);
    free(xja);

#pragma omp critical
    for (int i = 0; i < (nmo*nmo); i++) {
        vv[i] += vv_priv[i];
        vev[i] += vev_priv[i];
    }

    free(vv_priv);
    free(vev_priv);
}
}
