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
#include <string.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

/*
 * 4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
 * - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
 * - 2 * w.transpose(1,0,2)
 */
static void add_and_permute_emb(double *out, double *w, double *v, int n)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++) {
                v[i] += w[i];
        }

        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = v[i*nn+j*n+k] * 4
                                + v[j*nn+k*n+i]
                                + v[k*nn+i*n+j]
                                - v[k*nn+j*n+i] * 2
                                - v[i*nn+k*n+j] * 2
                                - v[j*nn+i*n+k] * 2;
        } } }
}

/*
 * t2T = t2.transpose(2,3,1,0)
 * ov = vv_op[:,nocc:]
 * oo = vv_op[:,:nocc]
 * w = numpy.einsum('if,fjk->ijk', ov, t2T[c])
 * w-= numpy.einsum('ijm,mk->ijk', vooo[a], t2T[c,b])
 * v = numpy.einsum('ij,k->ijk', oo, t1T[c]*.5)
 * v+= numpy.einsum('ij,k->ijk', t2T[b,a], fov[:,c]*.5)
 * v+= w
 */
static void get_wv_emb(double *w, double *wloc, double *v, double *cache, double *cache2,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T, double *t2Tloc,
                   int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        // Normal W
        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        // Local W
        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2Tloc+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache2, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2Tloc+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache2, &nocc);


        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                wloc[idx[n]] += cache2[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

double _ccsd_t_get_energy_emb(double *w, double *v, double *mo_energy, int nocc,
                          int a, int b, int c, double fac)
{
        int i, j, k, n;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double et = 0;

        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                et += fac * w[n] * v[n] / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc);
        } } }
        return et;
}

static double contract6_emb(int nocc, int nvir, int a, int b, int c,
                        double *mo_energy, double *t1T, double *t2T, double *t2Tloc,
                        int nirrep, int *o_ir_loc, int *v_ir_loc,
                        int *oo_ir_loc, int *orbsym, double *fvo,
                        double *vooo, double *cache1, double *cache2, void **cache,
                        int *permute_idx)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *wtmp = z0;
        // NEW
        double *w0loc = cache2;
        double *wtmp2 = cache2 + nooo;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                w0loc[i] = 0;
                v0[i] = 0;
        }

        if (nirrep == 1) {
                // NEW
                get_wv_emb(w0, w0loc, v0, wtmp, wtmp2, fvo, vooo, cache[0], t1T, t2T, t2Tloc, nocc, nvir, a, b, c, idx0);
                get_wv_emb(w0, w0loc, v0, wtmp, wtmp2, fvo, vooo, cache[1], t1T, t2T, t2Tloc, nocc, nvir, a, c, b, idx1);
                get_wv_emb(w0, w0loc, v0, wtmp, wtmp2, fvo, vooo, cache[2], t1T, t2T, t2Tloc, nocc, nvir, b, a, c, idx2);
                get_wv_emb(w0, w0loc, v0, wtmp, wtmp2, fvo, vooo, cache[3], t1T, t2T, t2Tloc, nocc, nvir, b, c, a, idx3);
                get_wv_emb(w0, w0loc, v0, wtmp, wtmp2, fvo, vooo, cache[4], t1T, t2T, t2Tloc, nocc, nvir, c, a, b, idx4);
                get_wv_emb(w0, w0loc, v0, wtmp, wtmp2, fvo, vooo, cache[5], t1T, t2T, t2Tloc, nocc, nvir, c, b, a, idx5);
        }
        add_and_permute_emb(z0, w0, v0, nocc);

        double et;
        if (a == c) {
                et = _ccsd_t_get_energy_emb(w0loc, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy_emb(w0loc, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy_emb(w0loc, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

size_t _ccsd_t_gen_jobs_emb(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride)
{
        size_t nov = nocc * (nocc+nvir) * stride;
        int da = a1 - a0;
        int db = b1 - b0;
        size_t m, a, b, c;

        if (b1 <= a0) {
                m = 0;
                for (a = a0; a < a1; a++) {
                for (b = b0; b < b1; b++) {
                        for (c = 0; c < b0; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b   );
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c   );
                                jobs[m].cache[2] = cache_col_a + nov*(da*(b)   +a-a0);
                                jobs[m].cache[3] = cache_row_b + nov*(b1*(b-b0)+c   );
                                jobs[m].cache[4] = cache_col_a + nov*(da*(c)   +a-a0);
                                jobs[m].cache[5] = cache_col_b + nov*(db*(c)   +b-b0);
                        }
                        for (c = b0; c <= b; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b   );
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c   );
                                jobs[m].cache[2] = cache_col_a + nov*(da*(b)   +a-a0);
                                jobs[m].cache[3] = cache_row_b + nov*(b1*(b-b0)+c   );
                                jobs[m].cache[4] = cache_col_a + nov*(da*(c)   +a-a0);
                                jobs[m].cache[5] = cache_row_b + nov*(b1*(c-b0)+b   );
                        }
                } }
        } else {
                m = 0;
                for (a = a0; a < a1; a++) {
                for (b = a0; b <= a; b++) {
                        for (c = 0; c < a0; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = cache_row_a + nov*(a1*(b-a0)+a);
                                jobs[m].cache[3] = cache_row_a + nov*(a1*(b-a0)+c);
                                jobs[m].cache[4] = cache_col_a + nov*(da*(c)+a-a0);
                                jobs[m].cache[5] = cache_col_a + nov*(da*(c)+b-a0);
                        }
                        for (c = a0; c <= b; c++, m++) {
                                jobs[m].a = a;
                                jobs[m].b = b;
                                jobs[m].c = c;
                                jobs[m].cache[0] = cache_row_a + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = cache_row_a + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = cache_row_a + nov*(a1*(b-a0)+a);
                                jobs[m].cache[3] = cache_row_a + nov*(a1*(b-a0)+c);
                                jobs[m].cache[4] = cache_row_a + nov*(a1*(c-a0)+a);
                                jobs[m].cache[5] = cache_row_a + nov*(a1*(c-a0)+b);
                        }
                } }
        }
        return m;
}

void _make_permute_indices_emb(int *idx, int n)
{
        const int nn = n * n;
        const int nnn = nn * n;
        int *idx0 = idx;
        int *idx1 = idx0 + nnn;
        int *idx2 = idx1 + nnn;
        int *idx3 = idx2 + nnn;
        int *idx4 = idx3 + nnn;
        int *idx5 = idx4 + nnn;
        int i, j, k, m;

        for (m = 0, i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++, m++) {
                idx0[m] = i * nn + j * n + k;
                idx1[m] = i * nn + k * n + j;
                idx2[m] = j * nn + i * n + k;
                idx3[m] = k * nn + i * n + j;
                idx4[m] = j * nn + k * n + i;
                idx5[m] = k * nn + j * n + i;
        } } }
}

void CCsd_t_contract_emb(double *e_tot,
                     double *mo_energy, double *t1T,
                     double *t2T, double *t2Tloc,
                     double *vooo, double *fvo,
                     int nocc, int nvir, int a0, int a1, int b0, int b1,
                     int nirrep, int *o_ir_loc, int *v_ir_loc,
                     int *oo_ir_loc, int *orbsym,
                     void *cache_row_a, void *cache_col_a,
                     void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs_emb(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices_emb(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, t2Tloc, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
        double *cache2 = malloc(sizeof(double) * (nocc*nocc*nocc*2));
        double *t1Thalf = malloc(sizeof(double) * nvir*nocc * 2);
        double *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6_emb(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T, t2Tloc,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, cache2, jobs[k].cache, permute_idx);
        }
        free(t1Thalf);
        free(cache1);
        free(cache2);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
}

//void _gen_permute_indices(int n, int *indices)
//{
//        const int nn = n * n;
//        const int nnn = nn * n;
//        int *indices0 = indices;
//        int *indices1 = indices0 + nnn;
//        int *indices2 = indices1 + nnn;
//        int *indices3 = indices2 + nnn;
//        int *indices4 = indices3 + nnn;
//        int *indices5 = indices4 + nnn;
//        int i;
//
//        for (int m = 0, i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//        for (int k = 0; k < n; k++, m++) {
//                indices0[m] = i*nn + j*n + k;
//                indices1[m] = i*nn + k*n + j;
//                indices2[m] = j*nn + i*n + k;
//                indices3[m] = k*nn + i*n + j;
//                indices4[m] = j*nn + k*n + i;
//                indices5[m] = k*nn + j*n + i;
//        }}}
//}

void get_paddings(int n, int order[], int pad[])
{
    int i;
    for (i = 0 ; i < 3 ; i++) {
        pad[i] = 1;
        if (order[i] == 0) {
            pad[i] = n*n;
        } else if (order[i] == 1) {
            pad[i] = n;
        }
    }
}


void _get_w(
        /* In */
        int a, int b, int c,
        int no, int nv,
        double *t2T,
        double *gvooo, double *gvvov,
        int worder[3],
        /* Out */
        double *w)
{
    //memset(w, 0, no*no*no * sizeof(double));
    //
    //
    int wpad[3];
    int i, j, k, d, l;
    get_paddings(no, worder, wpad);
    for (i = 0; i < no ; i++) {
    for (j = 0; j < no ; j++) {
    for (k = 0; k < no ; k++) {
        //w[i*no*no + j*no + k] = 0.0;
        int widx = i*wpad[0] + j*wpad[1] + k*wpad[2];
        for (d = 0; d < nv ; d++) {
            //w[i*no*no + j*no + k] +=
            w[widx] +=
                gvvov[a*nv*no*nv + b*no*nv + i*nv + d] * t2T[c*nv*no*no + d*no*no + k*no + j];
        }
        for (l = 0; l < no ; l++) {
            //w[i*no*no + j*no + k] -=
            w[widx] -=
                gvooo[a*no*no*no + i*no*no + j*no + l] * t2T[b*nv*no*no + c*no*no + l*no + k];
        }
    }}}
}

void _add_ws(int no,
        double *wabc, double *wacb, double *wbac,
        double *wbca, double *wcab, double *wcba,
        double *w)
{
    const int noo = no*no;
    int n, i, j, k;
    for (n = 0, i = 0; i < no ; i++) {
    for (j = 0; j < no ; j++) {
    for (k = 0; k < no ; k++, n++) {
        w[n] = wabc[n]
                + wacb[i*noo + k*no + j]
                + wbac[j*noo + i*no + k]
                + wbca[j*noo + k*no + i]
                + wcab[k*noo + i*no + j]
                + wcba[k*noo + j*no + i];
    }}}
}

void _get_v(
        /* In */
        int a, int b, int c,
        int no, int nv,
        double *t1T, double *t2T,
        double *fvo, double *gvvoo,
        /* Out */
        double *v)
{
    int i, j, k;
    for (i = 0; i < no ; i++) {
    for (j = 0; j < no ; j++) {
    for (k = 0; k < no ; k++) {
        v[i*no*no + j*no + k] =
            gvvoo[a*nv*no*no + b*no*no + i*no + j] * t1T[c*no + k];
        v[i*no*no + j*no + k] +=
            t2T[a*nv*no*no + b*no*no + i*no + j] * fvo[c*no + k];
    }}}
}


static void _permute(int no, double *in, double *out)
{
    const int noo = no*no;
    int i, j, k;
    for (i = 0; i < no; i++) {
    for (j = 0; j < no; j++) {
    for (k = 0; k < no; k++) {
        out[i*noo+j*no+k] = 4*in[i*noo + j*no + k]
                          +   in[j*noo + k*no + i]
                          +   in[k*noo + i*no + j]
                          - 2*in[k*noo + j*no + i]
                          - 2*in[i*noo + k*no + j]
                          - 2*in[j*noo + i*no + k];
    }}}
}

void _get_z(
        int a, int b, int c,
        int no, int nv,
        //double *w, double *v, double *e,
        double *t1T, double *t2T,
        double *fvo, double *gvvoo,
        double *w,
        //double *e,
        double *cache,
        /* Out */
        double *z)
{
    const int nooo = no*no*no;

    /* v -> cache. Cache does not need to be set to 0 */
    _get_v(a, b, c, no, nv, t1T, t2T, fvo, gvvoo, cache);

    /* Add w to cache */
    int ijk;
    for (ijk = 0; ijk < nooo; ijk++) {
        cache[ijk] = (w[ijk] + 0.5*cache[ijk]);
    }

    _permute(no, cache, z);
}

void _add_energy(
        int a, int b, int c,
        int no, int nv,
        int widcs[],
        double *eo, double *ev, double *w, double *z,
        /* Out */
        double *et)
{

    int wpad[3];
    get_paddings(no, widcs, wpad);

    double fac = 1.0;
    if (a == c) {
        fac = 1.0/6.0;
    } else if (a == b || b == c) {
        fac = 1.0/2.0;
    }
    const double eabc = ev[a] + ev[b] + ev[c];
    int n, i, j, k;
    for (n = 0, i = 0; i < no; i++) {
    for (j = 0; j < no; j++) {
    for (k = 0; k < no; k++, n++) {
        *et += (fac * w[i*wpad[0] + j*wpad[1] + k*wpad[2]] * z[n]) / (eo[i] + eo[j] + eo[k] - eabc);

    }}}
}


void ccsd_t_simple_emb(
        /* In */
        int no, int nv,
        double *mo_energy, double *t1T, double *t2T, double *t2locT,
        double *fvo,  double *gvvov, double *gvooo, double *gvvoo,
        /* Out */
        double *et)
{
    const int nooo = no*no*no;
    int order0[3] = {0, 1, 2};

    /* Occupied and virtual MO energies */
    double *eo = mo_energy;
    double *ev = (mo_energy + no);

#pragma omp parallel
    {
    double energy = 0.0;

    //double *wabc = malloc(nooo * sizeof(double));
    //double *wacb = malloc(nooo * sizeof(double));
    //double *wbac = malloc(nooo * sizeof(double));
    //double *wbca = malloc(nooo * sizeof(double));
    //double *wcab = malloc(nooo * sizeof(double));
    //double *wcba = malloc(nooo * sizeof(double));
    double *w = malloc((nooo) * sizeof(double));
    double *wsum = malloc(nooo * sizeof(double));
    double *z = malloc((nooo) * sizeof(double));
    double *cache = malloc(nooo * sizeof(double));

    int a, b, c;
#pragma omp for
    for (a = 0; a < nv ; a++) {
        for (b = 0; b <= a ; b++) {
            for (c = 0; c <= b ; c++) {

                /* Get ws */
                //memset(wabc, 0, nooo * sizeof(double));
                //memset(wacb, 0, nooo * sizeof(double));
                //memset(wbac, 0, nooo * sizeof(double));
                //memset(wbca, 0, nooo * sizeof(double));
                //memset(wcab, 0, nooo * sizeof(double));
                //memset(wcba, 0, nooo * sizeof(double));
                //_get_w(a, b, c, no, nv, t2T, gvooo, gvvov, order0, wabc);
                //_get_w(a, c, b, no, nv, t2T, gvooo, gvvov, order0, wacb);
                //_get_w(b, a, c, no, nv, t2T, gvooo, gvvov, order0, wbac);
                //_get_w(b, c, a, no, nv, t2T, gvooo, gvvov, order0, wbca);
                //_get_w(c, a, b, no, nv, t2T, gvooo, gvvov, order0, wcab);
                //_get_w(c, b, a, no, nv, t2T, gvooo, gvvov, order0, wcba);
                //_add_ws(no, wabc, wacb, wbac, wbca, wcab, wcba, wsum);

                memset(wsum, 0, nooo * sizeof(double));
                _get_w(a, b, c, no, nv, t2locT, gvooo, gvvov, (int[]){0, 1, 2}, wsum);
                _get_w(a, c, b, no, nv, t2locT, gvooo, gvvov, (int[]){0, 2, 1}, wsum);
                _get_w(b, a, c, no, nv, t2locT, gvooo, gvvov, (int[]){1, 0, 2}, wsum);
                _get_w(b, c, a, no, nv, t2locT, gvooo, gvvov, (int[]){1, 2, 0}, wsum);
                _get_w(c, a, b, no, nv, t2locT, gvooo, gvvov, (int[]){2, 0, 1}, wsum);
                _get_w(c, b, a, no, nv, t2locT, gvooo, gvvov, (int[]){2, 1, 0}, wsum);

                /* ABC */
                memset(w, 0, nooo * sizeof(double));
                _get_w(a, b, c, no, nv, t2T, gvooo, gvvov, order0, w);
                _get_z(a, b, c, no, nv, t1T, t2T, fvo, gvvoo, w, cache, z);
                _add_energy(a, b, c, no, nv, (int []){0, 1, 2}, eo, ev, wsum, z, &energy);
                /* ACB */
                memset(w, 0, nooo * sizeof(double));
                _get_w(a, c, b, no, nv, t2T, gvooo, gvvov, order0, w);
                _get_z(a, c, b, no, nv, t1T, t2T, fvo, gvvoo, w, cache, z);
                _add_energy(a, b, c, no, nv, (int []){0, 2, 1}, eo, ev, wsum, z, &energy);
                /* BAC */
                memset(w, 0, nooo * sizeof(double));
                _get_w(b, a, c, no, nv, t2T, gvooo, gvvov, order0, w);
                _get_z(b, a, c, no, nv, t1T, t2T, fvo, gvvoo, w, cache, z);
                _add_energy(a, b, c, no, nv, (int []){1, 0, 2}, eo, ev, wsum, z, &energy);
                /* BCA */
                memset(w, 0, nooo * sizeof(double));
                _get_w(b, c, a, no, nv, t2T, gvooo, gvvov, order0, w);
                _get_z(b, c, a, no, nv, t1T, t2T, fvo, gvvoo, w, cache, z);
                _add_energy(a, b, c, no, nv, (int []){1, 2, 0}, eo, ev, wsum, z, &energy);
                /* CAB */
                memset(w, 0, nooo * sizeof(double));
                _get_w(c, a, b, no, nv, t2T, gvooo, gvvov, order0, w);
                _get_z(c, a, b, no, nv, t1T, t2T, fvo, gvvoo, w, cache, z);
                _add_energy(a, b, c, no, nv, (int []){2, 0, 1}, eo, ev, wsum, z, &energy);
                /* CBA */
                memset(w, 0, nooo * sizeof(double));
                _get_w(c, b, a, no, nv, t2T, gvooo, gvvov, order0, w);
                _get_z(c, b, a, no, nv, t1T, t2T, fvo, gvvoo, w, cache, z);
                _add_energy(a, b, c, no, nv, (int []){2, 1, 0}, eo, ev, wsum, z, &energy);

            }
        }
    }

#pragma omp critical
    *et += 2.0*energy;

    //free(wabc);
    //free(wacb);
    //free(wbac);
    //free(wbca);
    //free(wcab);
    //free(wcba);
    free(w);
    free(wsum);
    free(z);
    free(cache);
    }
}
