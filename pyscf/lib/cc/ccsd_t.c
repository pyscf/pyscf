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
static void add_and_permute(double *out, double *w, double *v, int n, double fac)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++) {
                v[i] *= fac;
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
static void get_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
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

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static void sym_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int nirrep,
                   int *o_ir_loc, int *v_ir_loc, int *oo_ir_loc, int *orbsym,
                   int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int a_irrep = orbsym[nocc+a];
        int b_irrep = orbsym[nocc+b];
        int c_irrep = orbsym[nocc+c];
        int ab_irrep = a_irrep ^ b_irrep;
        int bc_irrep = c_irrep ^ b_irrep;
        int i, j, k, n;
        int fr, f0, f1, df, mr, m0, m1, dm, mk0;
        int ir, i0, i1, di, kr, k0, k1, dk, jr;
        int ijr, ij0, ij1, dij, jkr, jk0, jk1, djk;
        double *pt2T;

/* symmetry adapted
 * w = numpy.einsum('if,fjk->ijk', ov, t2T[c]) */
        pt2T = t2T + c * nvoo;
        for (ir = 0; ir < nirrep; ir++) {
                i0 = o_ir_loc[ir];
                i1 = o_ir_loc[ir+1];
                di = i1 - i0;
                if (di > 0) {
                        fr = ir ^ ab_irrep;
                        f0 = v_ir_loc[fr];
                        f1 = v_ir_loc[fr+1];
                        df = f1 - f0;
                        if (df > 0) {
                                jkr = fr ^ c_irrep;
                                jk0 = oo_ir_loc[jkr];
                                jk1 = oo_ir_loc[jkr+1];
                                djk = jk1 - jk0;
                                if (djk > 0) {

        dgemm_(&TRANS_N, &TRANS_N, &djk, &di, &df,
               &D1, pt2T+f0*noo+jk0, &noo, vv_op+i0*nmo+nocc+f0, &nmo,
               &D0, cache, &djk);
        for (n = 0, i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
        for (jr = 0; jr < nirrep; jr++) {
                kr = jkr ^ jr;
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[idx[i*noo+j*nocc+k]] += cache[n];
                } }
        } }
                                }
                        }
                }
        }

/* symmetry adapted
 * w-= numpy.einsum('ijm,mk->ijk', eris_vooo[a], t2T[c,b]) */
        pt2T = t2T + c * nvoo + b * noo;
        vooo += a * nooo;
        mk0 = oo_ir_loc[bc_irrep];
        for (mr = 0; mr < nirrep; mr++) {
                m0 = o_ir_loc[mr];
                m1 = o_ir_loc[mr+1];
                dm = m1 - m0;
                if (dm > 0) {
                        kr = mr ^ bc_irrep;
                        k0 = o_ir_loc[kr];
                        k1 = o_ir_loc[kr+1];
                        dk = k1 - k0;
                        if (dk > 0) {
                                ijr = mr ^ a_irrep;
                                ij0 = oo_ir_loc[ijr];
                                ij1 = oo_ir_loc[ijr+1];
                                dij = ij1 - ij0;
                                if (dij > 0) {

        dgemm_(&TRANS_N, &TRANS_N, &dk, &dij, &dm,
               &D1, pt2T+mk0, &dk, vooo+ij0*nocc+m0, &nocc,
               &D0, cache, &dk);
        for (n = 0, ir = 0; ir < nirrep; ir++) {
                jr = ijr ^ ir;
                for (i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[idx[i*noo+j*nocc+k]] -= cache[n];
                } }
        } }
                                }
                                mk0 += dm * dk;
                        }
                }
        }

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

double _ccsd_t_get_energy(double *w, double *v, double *mo_energy, int nocc,
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

static double contract6(int nocc, int nvir, int a, int b, int c,
                        double *mo_energy, double *t1T, double *t2T,
                        int nirrep, int *o_ir_loc, int *v_ir_loc,
                        int *oo_ir_loc, int *orbsym, double *fvo,
                        double *vooo, double *cache1, void **cache,
                        int *permute_idx, double fac)
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
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        if (nirrep == 1) {
                get_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        } else {
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx0);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx1);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx2);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx3);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx4);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx5);
        }
        add_and_permute(z0, w0, v0, nocc, fac);

        double et;
        if (a == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
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

void _make_permute_indices(int *idx, int n)
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

void CCsd_t_contract(double *e_tot,
                     double *mo_energy, double *t1T, double *t2T,
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
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
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
                e += contract6(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                               1.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}

void QCIsd_t_contract(double *e_tot,
                      double *mo_energy, double *t1T, double *t2T,
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
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
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
                e += contract6(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                               2.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}


/*
 * Complex version of all functions
 */
static void zadd_and_permute(double complex *out, double complex *w,
                             double complex *v, int n, double fac)
{
        int nn = n * n;
        int nnn = nn * n;
        int i, j, k;

        for (i = 0; i < nnn; i++) {
                v[i] *= fac;
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

static void zget_wv(double complex *w, double complex *v,
                    double complex *cache, double complex *fvohalf,
                    double complex *vooo, double complex *vv_op,
                    double complex *t1Thalf, double complex *t2T,
                    int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 =-1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;

        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

double _ccsd_t_zget_energy(double complex *w, double complex *v,
                           double *mo_energy, int nocc,
                           int a, int b, int c, double fac)
{
        int i, j, k, n;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double et = 0;

        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                et += fac / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc) * w[n] * conj(v[n]);
        } } }
        return et;
}

static double complex
zcontract6(int nocc, int nvir, int a, int b, int c,
           double *mo_energy, double complex *t1T, double complex *t2T,
           int nirrep, int *o_ir_loc, int *v_ir_loc,
           int *oo_ir_loc, int *orbsym, double complex *fvo,
           double complex *vooo, double complex *cache1, void **cache,
           int *permute_idx, double fac)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double complex *v0 = cache1;
        double complex *w0 = v0 + nooo;
        double complex *z0 = w0 + nooo;
        double complex *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        zget_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        zadd_and_permute(z0, w0, v0, nocc, fac);

        double complex et;
        if (a == c) {
                et = _ccsd_t_zget_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_zget_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_zget_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

void CCsd_t_zcontract(double complex *e_tot,
                      double *mo_energy, double complex *t1T, double complex *t2T,
                      double complex *vooo, double complex *fvo,
                      int nocc, int nvir, int a0, int a1, int b0, int b1,
                      int nirrep, int *o_ir_loc, int *v_ir_loc,
                      int *oo_ir_loc, int *orbsym,
                      void *cache_row_a, void *cache_col_a,
                      void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b,
                                        sizeof(double complex));

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) * (nocc*nocc*nocc*3+2));
        double complex *t1Thalf = malloc(sizeof(double complex) * nvir*nocc * 2);
        double complex *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double complex e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                               1.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}

void QCIsd_t_zcontract(double complex *e_tot,
                       double *mo_energy, double complex *t1T, double complex *t2T,
                       double complex *vooo, double complex *fvo,
                       int nocc, int nvir, int a0, int a1, int b0, int b1,
                       int nirrep, int *o_ir_loc, int *v_ir_loc,
                       int *oo_ir_loc, int *orbsym,
                       void *cache_row_a, void *cache_col_a,
                       void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b,
                                        sizeof(double complex));

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) * (nocc*nocc*nocc*3+2));
        double complex *t1Thalf = malloc(sizeof(double complex) * nvir*nocc * 2);
        double complex *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double complex e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                               nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                               fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                               2.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}


/*****************************************************************************
 *
 * mpi4pyscf
 *
 *****************************************************************************/
static void MPICCget_wv(double *w, double *v, double *cache,
                        double *fvohalf, double *vooo,
                        double *vv_op, double *t1Thalf,
                        double *t2T_a, double *t2T_c,
                        int nocc, int nvir, int a, int b, int c,
                        int a0, int b0, int c0, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 = -1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T_c+(c-c0)*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T_c+(c-c0)*nvoo+b*noo, &nocc, vooo+(a-a0)*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T_a + (a-a0) * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static double MPICCcontract6(int nocc, int nvir, int a, int b, int c,
                             double *mo_energy, double *t1T, double *fvo,
                             int *slices, double **data_ptrs, double *cache1,
                             int *permute_idx, double fac)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        const int da = a1 - a0;
        const int db = b1 - b0;
        const int dc = c1 - c0;
        const int nooo = nocc * nocc * nocc;
        const int nmo = nocc + nvir;
        const size_t nop = nocc * nmo;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *vvop_ab = data_ptrs[0] + ((a-a0)*db+b-b0) * nop;
        double *vvop_ac = data_ptrs[1] + ((a-a0)*dc+c-c0) * nop;
        double *vvop_ba = data_ptrs[2] + ((b-b0)*da+a-a0) * nop;
        double *vvop_bc = data_ptrs[3] + ((b-b0)*dc+c-c0) * nop;
        double *vvop_ca = data_ptrs[4] + ((c-c0)*da+a-a0) * nop;
        double *vvop_cb = data_ptrs[5] + ((c-c0)*db+b-b0) * nop;
        double *vooo_a = data_ptrs[6];
        double *vooo_b = data_ptrs[7];
        double *vooo_c = data_ptrs[8];
        double *t2T_a = data_ptrs[9 ];
        double *t2T_b = data_ptrs[10];
        double *t2T_c = data_ptrs[11];

        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        MPICCget_wv(w0, v0, wtmp, fvo, vooo_a, vvop_ab, t1T, t2T_a, t2T_c, nocc, nvir, a, b, c, a0, b0, c0, idx0);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_a, vvop_ac, t1T, t2T_a, t2T_b, nocc, nvir, a, c, b, a0, c0, b0, idx1);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_b, vvop_ba, t1T, t2T_b, t2T_c, nocc, nvir, b, a, c, b0, a0, c0, idx2);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_b, vvop_bc, t1T, t2T_b, t2T_a, nocc, nvir, b, c, a, b0, c0, a0, idx3);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_c, vvop_ca, t1T, t2T_c, t2T_b, nocc, nvir, c, a, b, c0, a0, b0, idx4);
        MPICCget_wv(w0, v0, wtmp, fvo, vooo_c, vvop_cb, t1T, t2T_c, t2T_a, nocc, nvir, c, b, a, c0, b0, a0, idx5);
        add_and_permute(z0, w0, v0, nocc, fac);

        double et;
        if (a == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy(w0, z0, mo_energy, nocc, a, b, c, 1.);
        }
        return et;
}

size_t _MPICCsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                           int *slices, double **data_ptrs)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        size_t m, a, b, c;

        m = 0;
        for (a = a0; a < a1; a++) {
        for (b = b0; b < MIN(b1, a+1); b++) {
        for (c = c0; c < MIN(c1, b+1); c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
        } } }
        return m;
}

void MPICCsd_t_contract(double *e_tot, double *mo_energy, double *t1T,
                        double *fvo, int nocc, int nvir,
                        int *slices, double **data_ptrs)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        int da = a1 - a0;
        int db = b1 - b0;
        int dc = c1 - c0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*dc);
        size_t njobs = _MPICCsd_t_gen_jobs(jobs, nocc, nvir, slices, data_ptrs);

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, fvo, jobs, e_tot, slices, \
               data_ptrs, permute_idx)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2));
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
                e += MPICCcontract6(nocc, nvir, a, b, c, mo_energy, t1Thalf,
                                    fvohalf, slices, data_ptrs, cache1,
                                    permute_idx, 1.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}

/*****************************************************************************
 *
 * pyscf periodic ccsd(t) with k-points
 *
 *****************************************************************************/

size_t _CCsd_t_gen_jobs_full(CacheJob *jobs, int nocc, int nvir,
                             int *slices)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        size_t m, a, b, c;

        m = 0;
        for (a = a0; a < a1; a++) {
        for (b = b0; b < b1; b++) {
        for (c = c0; c < c1; c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
        } } }
        return m;
}

static void CCzget_wv(double complex *w, double complex *v, double complex *cache,
                      double complex *fvohalf, double complex *vooo,
                      double complex *vv_op, double complex *vv_op2,
                      double complex *t1Thalf, double complex *t2T_c1,
                      double complex *t2T_c2, double complex *t2T_c3,
                      int nocc, int nvir, int a, int b, int c,
                      int a0, int b0, int c0, int *idx, int bool_add_v)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 = -1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;


        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T_c1+(c-c0)*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T_c2+(c-c0)*nvoo+b*noo, &nocc, vooo+(a-a0)*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T_c3 + (b-b0)*nvoo + a*noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                if(bool_add_v == 1){
                    v[idx[n]] += (vv_op2[j*nmo+i] * t1Thalf[c*nocc+k]
                                 + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
                }
        } } }
}

static void zcontract6_t3T(int nocc, int nvir, int a, int b, int c,
                           int *mo_offset, double complex *t3Tw,
                           double complex *t3Tv, double *mo_energy,
                           double complex *t1T, double complex *fvo, int *slices,
                           double complex **data_ptrs, double complex *cache1,
                           int *permute_idx)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        const int da = a1 - a0;
        const int db = b1 - b0;
        const int dc = c1 - c0;
        const int nooo = nocc * nocc * nocc;
        const int nmo = nocc + nvir;
        const int nop = nocc * nmo;
        const int nov = nocc * nvir;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        int ki = mo_offset[0];
        int kj = mo_offset[1];
        int kk = mo_offset[2];
        int ka = mo_offset[3];
        int kb = mo_offset[4];
        int kc = mo_offset[5];
        double complex *t1T_a = t1T + ka * nov;
        double complex *t1T_b = t1T + kb * nov;
        double complex *t1T_c = t1T + kc * nov;
        double complex *fvo_a = fvo + ka * nov;
        double complex *fvo_b = fvo + kb * nov;
        double complex *fvo_c = fvo + kc * nov;
        double complex *vvop_ab = data_ptrs[0] + ((a-a0)*db+b-b0) * nop;
        double complex *vvop_ac = data_ptrs[1] + ((a-a0)*dc+c-c0) * nop;
        double complex *vvop_ba = data_ptrs[2] + ((b-b0)*da+a-a0) * nop;
        double complex *vvop_bc = data_ptrs[3] + ((b-b0)*dc+c-c0) * nop;
        double complex *vvop_ca = data_ptrs[4] + ((c-c0)*da+a-a0) * nop;
        double complex *vvop_cb = data_ptrs[5] + ((c-c0)*db+b-b0) * nop;
        double complex *vooo_aj = data_ptrs[6];
        double complex *vooo_ak = data_ptrs[7];
        double complex *vooo_bi = data_ptrs[8];
        double complex *vooo_bk = data_ptrs[9];
        double complex *vooo_ci = data_ptrs[10];
        double complex *vooo_cj = data_ptrs[11];
        double complex *t2T_cj = data_ptrs[12];
        double complex *t2T_cb = data_ptrs[13];
        double complex *t2T_bk = data_ptrs[14];
        double complex *t2T_bc = data_ptrs[15];
        double complex *t2T_ci = data_ptrs[16];
        double complex *t2T_ca = data_ptrs[17];
        double complex *t2T_ak = data_ptrs[18];
        double complex *t2T_ac = data_ptrs[19];
        double complex *t2T_bi = data_ptrs[20];
        double complex *t2T_ba = data_ptrs[21];
        double complex *t2T_aj = data_ptrs[22];
        double complex *t2T_ab = data_ptrs[23];

        double complex *v0 = cache1;
        double complex *w0 = v0 + nooo;
        double complex *z0 = w0 + nooo;
        double complex *wtmp = z0;
        int i, j, k, n;
        int offset;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
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

        CCzget_wv(w0, v0, wtmp, fvo_c, vooo_aj, vvop_ab, vvop_ba, t1T_c, t2T_cj, t2T_cb, t2T_ba,
                  nocc, nvir, a, b, c, a0, b0, c0, idx0, (kk==kc));
        CCzget_wv(w0, v0, wtmp, fvo_b, vooo_ak, vvop_ac, vvop_ca, t1T_b, t2T_bk, t2T_bc, t2T_ca,
                  nocc, nvir, a, c, b, a0, c0, b0, idx1, (kj==kb));
        CCzget_wv(w0, v0, wtmp, fvo_c, vooo_bi, vvop_ba, vvop_ab, t1T_c, t2T_ci, t2T_ca, t2T_ab,
                  nocc, nvir, b, a, c, b0, a0, c0, idx2, (kk==kc));
        CCzget_wv(w0, v0, wtmp, fvo_a, vooo_bk, vvop_bc, vvop_cb, t1T_a, t2T_ak, t2T_ac, t2T_cb,
                  nocc, nvir, b, c, a, b0, c0, a0, idx3, (ka==ki));
        CCzget_wv(w0, v0, wtmp, fvo_b, vooo_ci, vvop_ca, vvop_ac, t1T_b, t2T_bi, t2T_ba, t2T_ac,
                  nocc, nvir, c, a, b, c0, a0, b0, idx4, (kb==kj));
        CCzget_wv(w0, v0, wtmp, fvo_a, vooo_cj, vvop_cb, vvop_bc, t1T_a, t2T_aj, t2T_ab, t2T_bc,
                  nocc, nvir, c, b, a, c0, b0, a0, idx5, (ka==ki));

        offset = (((a-a0)*db + b-b0)*dc + c-c0)*nooo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
            //div = 1. / (mo_energy[i+ki*nmo] + mo_energy[j+kj*nmo] + mo_energy[k+kk*nmo] - abc);
            t3Tw[offset + n] = w0[n];
            t3Tv[offset + n] = v0[n];
        } } }

}

void CCsd_zcontract_t3T(double complex *t3Tw, double complex *t3Tv, double *mo_energy,
                        double complex *t1T, double complex *fvo, int nocc, int nvir, int nkpts,
                        int *mo_offset, int *slices, double complex **data_ptrs)
{
        const int a0 = slices[0];
        const int a1 = slices[1];
        const int b0 = slices[2];
        const int b1 = slices[3];
        const int c0 = slices[4];
        const int c1 = slices[5];
        int da = a1 - a0;
        int db = b1 - b0;
        int dc = c1 - c0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*dc);
        size_t njobs = _CCsd_t_gen_jobs_full(jobs, nocc, nvir, slices);

        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, nkpts, t3Tw, t3Tv, mo_offset, mo_energy, t1T, fvo, jobs, slices, \
               data_ptrs, permute_idx)
{
        int a, b, c;
        size_t k;
        complex double *cache1 = malloc(sizeof(double complex) * (nocc*nocc*nocc*3+2));
        complex double *t1Thalf = malloc(sizeof(double complex) * nkpts*nvir*nocc*2);
        complex double *fvohalf = t1Thalf + nkpts*nvir*nocc;
        for (k = 0; k < nkpts*nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                zcontract6_t3T(nocc, nvir, a, b, c, mo_offset, t3Tw, t3Tv, mo_energy, t1Thalf,
                               fvohalf, slices, data_ptrs, cache1,
                               permute_idx);
        }
        free(t1Thalf);
        free(cache1);
}
        free(jobs);
        free(permute_idx);
}

