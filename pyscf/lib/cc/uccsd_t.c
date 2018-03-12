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
#include <string.h>
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

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride);

double _ccsd_t_permute_contract(double *z0, double *z1, double *z2, double *z3,
                                double *z4, double *z5, double *w, int n);

void _ccsd_t_get_denorm(double *d3, double *mo_energy, int nocc,
                        int a, int b, int c);

double complex
_ccsd_t_zpermute_contract(double complex *z0, double complex *z1,
                          double complex *z2, double complex *z3,
                          double complex *z4, double complex *z5,
                          double complex *w, int n);

/*
 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
 * - w.transpose(2,1,0) - w.transpose(0,2,1) - w.transpose(1,0,2)
 */
static void permute(double *out, double *w, int n)
{
        int nn = n * n;
        int i, j, k;

        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = w[i*nn+j*n+k] + w[j*nn+k*n+i] + w[k*nn+i*n+j]
                                - w[k*nn+j*n+i] - w[i*nn+k*n+j] - w[j*nn+i*n+k];
        } } }
}

/*
 * t2T = t2.transpose(2,3,0,1)
 * ov = vv_op[:,nocc:]
 * oo = vv_op[:,:nocc]
 * w = numpy.einsum('if,fjk->ijk', -ov, t2T[c])
 * w-= numpy.einsum('ijm,mk->ijk', vooo[a], t2T[b,c])
 * v = numpy.einsum('ij,k->ijk', oo, t1T[c])
 * v+= w
 */
static void get_wv(double *w, double *v, double *fvohalf, double *vooo,
                   double *vv_op, double *t1T, double *t2T,
                   int nocc, int nvir, int a, int b, int c)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, w, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
               &D1, w, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[n] = w[n] + vv_op[i*nmo+j] * t1T[c*nocc+k];
                v[n]+= pt2T[i*nocc+j] * fvohalf[c*nocc+k];
        } } }
}

static void sym_wv(double *w, double *v, double *fvohalf, double *vooo,
                   double *vv_op, double *t1T, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int nirrep,
                   int *o_ir_loc, int *v_ir_loc, int *oo_ir_loc, int *orbsym)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const int nooo = nocc * noo;
        const int nvoo = nvir * noo;
        int a_irrep = orbsym[nocc+a];
        int b_irrep = orbsym[nocc+b];
        int c_irrep = orbsym[nocc+c];
        int ab_irrep = a_irrep ^ b_irrep;
        int bc_irrep = c_irrep ^ b_irrep;
        int i, j, k, n;
        int fr, f0, f1, df, mr, m0, m1, dm, mk0;
        int ir, i0, i1, di, kr, k0, k1, dk, jr;
        int ijr, ij0, ij1, dij, jkr, jk0, jk1, djk;
        double *buf = v;
        double *pt2T;

        memset(w, 0, sizeof(double)*nooo);
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
               &D0, buf, &djk);
        for (n = 0, i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
        for (jr = 0; jr < nirrep; jr++) {
                kr = jkr ^ jr;
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[i*noo+j*nocc+k] -= buf[n];
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
               &D0, buf, &dk);
        for (n = 0, ir = 0; ir < nirrep; ir++) {
                jr = ijr ^ ir;
                for (i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[i*noo+j*nocc+k] -= buf[n];
                } }
        } }
                                }
                                mk0 += dm * dk;
                        }
                }
        }

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[n] = w[n] + vv_op[i*nmo+j] * t1T[c*nocc+k];
                v[n]+= pt2T[i*nocc+j] * fvohalf[c*nocc+k];
        } } }
}

static double contract6_aaa(int nocc, int nvir, int a, int b, int c,
                            double *mo_energy, double *t1T, double *t2T,
                            int nirrep, int *o_ir_loc, int *v_ir_loc,
                            int *oo_ir_loc, int *orbsym, double *fvo,
                            double *vooo, double *cache1, void **cache)
{
        int nooo = nocc * nocc * nocc;
        double *denorm = cache1;
        double *v0 = denorm + nooo;
        double *v1 = v0 + nooo;
        double *v2 = v1 + nooo;
        double *v3 = v2 + nooo;
        double *v4 = v3 + nooo;
        double *v5 = v4 + nooo;
        double *w0 = v5 + nooo;
        double *w1 = w0 + nooo;
        double *w2 = w1 + nooo;
        double *w3 = w2 + nooo;
        double *w4 = w3 + nooo;
        double *w5 = w4 + nooo;
        double *z0 = w5 + nooo;
        double *z1 = z0 + nooo;
        double *z2 = z1 + nooo;
        double *z3 = z2 + nooo;
        double *z4 = z3 + nooo;
        double *z5 = z4 + nooo;
        int i;

        if (nirrep == 1) {
                get_wv(w0, v0, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c);
                get_wv(w1, v1, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b);
                get_wv(w2, v2, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c);
                get_wv(w3, v3, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a);
                get_wv(w4, v4, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b);
                get_wv(w5, v5, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a);
        } else {
                sym_wv(w0, v0, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym);
                sym_wv(w1, v1, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym);
                sym_wv(w2, v2, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym);
                sym_wv(w3, v3, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym);
                sym_wv(w4, v4, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym);
                sym_wv(w5, v5, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym);
        }
        permute(z0, v0, nocc);
        permute(z1, v1, nocc);
        permute(z2, v2, nocc);
        permute(z3, v3, nocc);
        permute(z4, v4, nocc);
        permute(z5, v5, nocc);

        _ccsd_t_get_denorm(denorm, mo_energy, nocc, a, b, c);
        if (a == c) {
                for (i = 0; i < nooo; i++) {
                        denorm[i] *= 1./6;
                }
        } else if (a == b || b == c) {
                for (i = 0; i < nooo; i++) {
                        denorm[i] *= .5;
                }
        }
        for (i = 0; i < nooo; i++) {
                z0[i] *= denorm[i];
                z1[i] *= denorm[i];
                z2[i] *= denorm[i];
                z3[i] *= denorm[i];
                z4[i] *= denorm[i];
                z5[i] *= denorm[i];
        }

        double et = 0;
        et += _ccsd_t_permute_contract(z0, z1, z2, z3, z4, z5, w0, nocc);
        et += _ccsd_t_permute_contract(z1, z0, z4, z5, z2, z3, w1, nocc);
        et += _ccsd_t_permute_contract(z2, z3, z0, z1, z5, z4, w2, nocc);
        et += _ccsd_t_permute_contract(z3, z2, z5, z4, z0, z1, w3, nocc);
        et += _ccsd_t_permute_contract(z4, z5, z1, z0, z3, z2, w4, nocc);
        et += _ccsd_t_permute_contract(z5, z4, z3, z2, z1, z0, w5, nocc);
        return et;
}

void CCuccsd_t_aaa(double complex *e_tot,
                   double *mo_energy, double *t1T, double *t2T,
                   double *vooo, double *fvo,
                   int nocc, int nvir, int a0, int a1, int b0, int b1,
                   int nirrep, int *o_ir_loc, int *v_ir_loc,
                   int *oo_ir_loc, int *orbsym,
                   double *cache_row_a, double *cache_col_a,
                   double *cache_row_b, double *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        double fvohalf[nvir*nocc];
        int i;
        for (i = 0; i < nvir*nocc; i++) {
                fvohalf[i] = fvo[i] * .5;
        }

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvohalf, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*19+2));
        double e = 0;
#pragma omp for schedule (dynamic, 32)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6_aaa(nocc, nvir, a, b, c, mo_energy, t1T, t2T,
                                   nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                   fvohalf, vooo, cache1, jobs[k].cache);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}


/*************************************************
 *
 * UCCSD(T) beta-alpha-alpha
 *
 *************************************************/
static void get_wv_baa(double *w, double *v, double **vs_ts, double **cache,
                       int nocca, int noccb, int nvira, int nvirb,
                       int a, int b, int c)
{
        double *fvo = vs_ts[2];
        double *fVO = vs_ts[3];
        double *vooo = vs_ts[4];
        double *vOoO = vs_ts[5];
        double *VoOo = vs_ts[6];
        double *t1aT = vs_ts[7];
        double *t1bT = vs_ts[8];
        double *t2aaT = vs_ts[9];
        double *t2abT = vs_ts[10];
        double *vvop = cache[0];
        double *vVoP = cache[1];
        double *VvOp = cache[2];
        const double D0 = 0;
        const double D1 = 1;
        const double D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

/*
 * t2aaT = t2aa.transpose(2,3,0,1)
 * w  = numpy.einsum('ejI,ke->Ijk', t2abT[:,a], vvov) * 2
 * w += numpy.einsum('EjI,kE->Ijk', t2abT[b,:], vVoV) * 2
 * w += numpy.einsum('mj,mIk->Ijk', t2aaT[b,c], VoOo[a,:])
 * w += numpy.einsum('kM,MjI->Ijk', t2abT[b,a], vOoO[c,:]) * 2
 * w += numpy.einsum('ejk,Ie->Ijk', t2aaT[b,:], VvOv)
 * w += numpy.einsum('mI,mjk->Ijk', t2abT[b,a], vooo[c,:]) * 2
 * v  = numpy.einsum('kj,I->Ijk', vvoo, t1bT[a])
 * v += numpy.einsum('Ik,j->Ijk', VvOo, t1aT[b]) * 2
 * v += numpy.einsum('jk,I->Ijk', t2aaT[b,c], fVO[a]) * .5
 * v += numpy.einsum('kI,j->Ijk', t2abT[c,a], fvo[b]) * 2
 * v += w
 */
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira,
               &D2, vvop+nocca, &nmoa, t2abT+a*nOo, &nVoO,
               &D0, v, &nocca);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb,
               &D2, vVoP+noccb, &nmob, t2abT+b*(size_t)nVoO, &nOo,
               &D1, v, &nocca);
        dgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca,
               &D1, VoOo+a*noOo, &nOo, t2aaT+b*nvoo+c*noo, &nocca,
               &D1, v, &nOo);
        dgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb,
               &D2, t2abT+b*(size_t)nVoO+a*nOo, &noccb, vOoO+c*nOoO, &nOo,
               &D1, v, &nocca);
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                w[n] = v[j*nOo+i*nocca+k];
        } } }
        dgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira,
               &D1, t2aaT+b*nvoo, &noo, VvOp+nocca, &nmoa,
               &D1, w, &noo);
        dgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca,
               &D2, vooo+c*nooo, &noo, t2abT+b*(size_t)nVoO+a*nOo, &noccb,
               &D1, w, &noo);

        double t1aT2[nocca];
        double fvo2[nocca];
        double fVOhalf[noccb];
        for (i = 0; i < nocca; i++) {
                t1aT2[i] = t1aT[b*nocca+i] * 2;
                fvo2[i] = fvo[b*nocca+i] * 2;
        }
        for (i = 0; i < noccb; i++) {
                fVOhalf[i] = fVO[a*noccb+i] * .5;
        }
        double *pt2aaT = t2aaT + b * nvoo + c * noo;
        double *pt2abT = t2abT + (c*nvirb+a) * nOo;
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                v[n] = (w[n] + vvop[k*nmoa+j] * t1bT[a*noccb+i]
                        + VvOp[i*nmoa+k] * t1aT2[j]
                        + pt2aaT[j*nocca+k] * fVOhalf[i]
                        + pt2abT[k*noccb+i] * fvo2[j]);
        } } }
}

/*
 * w - w.transpose(0,2,1)
 */
static void permute_baa(double *out, double *w, int nocca, int noccb)
{
        int noo = nocca * nocca;
        int n;
        int i, j, k;

        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                out[n] = w[i*noo+j*nocca+k] - w[i*noo+k*nocca+j];
        } } }
}
static void get_denorm_baa(double *d3, double *mo_ea, double *mo_eb,
                           int nocca, int noccb, int a, int b, int c)
{
        int i, j, k, n;
        double abc = mo_eb[noccb+a] + mo_ea[nocca+b] + mo_ea[nocca+c];

        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                d3[n] = 1./(mo_eb[i] + mo_ea[j] + mo_ea[k] - abc);
        } } }
}

static double permute_contract_baa(double *z0, double *z1, double *w,
                                   int nocca, int noccb)
{
        int noo = nocca * nocca;
        int i, j, k;
        double et = 0;

        for (i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++) {
                et += z0[i*noo+j*nocca+k] * w[i*noo+j*nocca+k];
                et += z1[i*noo+k*nocca+j] * w[i*noo+j*nocca+k];
        } } }
        return et;
}

static double contract6_baa(int nocca, int noccb, int nvira, int nvirb,
                            int a, int b, int c,
                            double **vs_ts, void **cache, double *cache1)
{
        int nOoo = noccb * nocca * nocca;
        double *denorm = cache1;
        double *v0 = denorm + nOoo;
        double *v1 = v0 + nOoo;
        double *w0 = v1 + nOoo;
        double *w1 = w0 + nOoo;
        double *z0 = w1 + nOoo;
        double *z1 = z0 + nOoo;
        int i;

        get_wv_baa(w0, v0, vs_ts, ((double **)cache)  , nocca, noccb, nvira, nvirb, a, b, c);
        get_wv_baa(w1, v1, vs_ts, ((double **)cache)+3, nocca, noccb, nvira, nvirb, a, c, b);
        permute_baa(z0, v0, nocca, noccb);
        permute_baa(z1, v1, nocca, noccb);

        double *mo_ea = vs_ts[0];
        double *mo_eb = vs_ts[1];
        get_denorm_baa(denorm, mo_ea, mo_eb, nocca, noccb, a, b, c);
        if (b == c) {
                for (i = 0; i < nOoo; i++) {
                        denorm[i] *= .5;
                }
        }
        for (i = 0; i < nOoo; i++) {
                z0[i] *= denorm[i];
                z1[i] *= denorm[i];
        }

        double et = 0;
        et += permute_contract_baa(z0, z1, w0, nocca, noccb);
        et += permute_contract_baa(z1, z0, w1, nocca, noccb);
        return et;
}


static size_t gen_baa_jobs(CacheJob *jobs,
                           int nocca, int noccb, int nvira, int nvirb,
                           int a0, int a1, int b0, int b1,
                           void *cache_row_a, void *cache_col_a,
                           void *cache_row_b, void *cache_col_b, size_t stride)
{
        size_t nov = nocca * (nocca+nvira) * stride;
        size_t noV = nocca * (noccb+nvirb) * stride;
        size_t nOv = noccb * (nocca+nvira) * stride;
        int da = a1 - a0;
        int db = b1 - b0;
        int a, b, c;

        size_t m = 0;
        for (a = a0; a < a1; a++) {
        for (b = b0; b < b1; b++) {
        for (c = 0; c <= b; c++, m++) {
                jobs[m].a = a;
                jobs[m].b = b;
                jobs[m].c = c;
                if (c < b0) {
                        jobs[m].cache[0] = cache_col_b + nov*(db*(c   )+b-b0);
                } else {
                        jobs[m].cache[0] = cache_row_b + nov*(b1*(c-b0)+b   );
                }
                jobs[m].cache[1] = cache_col_a + noV*(da   *(c   )+a-a0);
                jobs[m].cache[2] = cache_row_a + nOv*(nvira*(a-a0)+c   );
                jobs[m].cache[3] = cache_row_b + nov*(b1   *(b-b0)+c   );
                jobs[m].cache[4] = cache_col_a + noV*(da   *(b   )+a-a0);
                jobs[m].cache[5] = cache_row_a + nOv*(nvira*(a-a0)+b   );
        } } }
        return m;
}

void CCuccsd_t_baa(double complex *e_tot,
                   double *mo_ea, double *mo_eb,
                   double *t1aT, double *t1bT, double *t2aaT, double *t2abT,
                   double *vooo, double *vOoO, double *VoOo,
                   double *fvo, double *fVO,
                   int nocca, int noccb, int nvira, int nvirb,
                   int a0, int a1, int b0, int b1,
                   void *cache_row_a, void *cache_col_a,
                   void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb,
                                    a0, a1, b0, b1,
                                    cache_row_a, cache_col_a,
                                    cache_row_b, cache_col_b, sizeof(double));
        double *vs_ts[] = {mo_ea, mo_eb, fvo, fVO, vooo, vOoO, VoOo,
                t1aT, t1bT, t2aaT, t2abT};

#pragma omp parallel default(none) \
        shared(njobs, nocca, noccb, nvira, nvirb, vs_ts, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double *cache1 = malloc(sizeof(double) * (noccb*nocca*nocca*7+2));
        double e = 0;
#pragma omp for schedule (dynamic, 32)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6_baa(nocca, noccb, nvira, nvirb, a, b, c, vs_ts,
                                   jobs[k].cache, cache1);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}



/*
 * Complex version of all functions
 */
static void zpermute(double complex *out, double complex *w, int n)
{
        int nn = n * n;
        int i, j, k;

        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = w[i*nn+j*n+k] + w[j*nn+k*n+i] + w[k*nn+i*n+j]
                                - w[k*nn+j*n+i] - w[i*nn+k*n+j] - w[j*nn+i*n+k];
        } } }
}

static void zget_wv(double complex *w, double complex *v,
                    double complex *fvohalf, double complex *vooo,
                    double complex *vv_op, double complex *t1T, double complex *t2T,
                    int nocc, int nvir, int a, int b, int c)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 =-1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;

        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &DN1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, w, &noo);
        zgemm_(&TRANS_N, &TRANS_T, &nocc, &noo, &nocc,
               &DN1, t2T+b*nvoo+c*noo, &nocc, vooo+a*nooo, &noo,
               &D1, w, &nocc);

        pt2T = t2T + a * nvoo + b * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[n] = w[n] + vv_op[i*nmo+j] * t1T[c*nocc+k];
                v[n]+= pt2T[i*nocc+j] * fvohalf[c*nocc+k];
        } } }
}

static double complex
zcontract6_aaa(int nocc, int nvir, int a, int b, int c,
               double *mo_energy, double complex *t1T, double complex *t2T,
               int nirrep, int *o_ir_loc, int *v_ir_loc,
               int *oo_ir_loc, int *orbsym, double complex *fvo,
               double complex *vooo, double complex *cache1, void **cache)
{
        int nooo = nocc * nocc * nocc;
        double complex *v0 = cache1;
        double complex *v1 = v0 + nooo;
        double complex *v2 = v1 + nooo;
        double complex *v3 = v2 + nooo;
        double complex *v4 = v3 + nooo;
        double complex *v5 = v4 + nooo;
        double complex *w0 = v5 + nooo;
        double complex *w1 = w0 + nooo;
        double complex *w2 = w1 + nooo;
        double complex *w3 = w2 + nooo;
        double complex *w4 = w3 + nooo;
        double complex *w5 = w4 + nooo;
        double complex *z0 = w5 + nooo;
        double complex *z1 = z0 + nooo;
        double complex *z2 = z1 + nooo;
        double complex *z3 = z2 + nooo;
        double complex *z4 = z3 + nooo;
        double complex *z5 = z4 + nooo;
        double *denorm = (double *)(z5 + nooo);
        int i;

        zget_wv(w0, v0, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c);
        zget_wv(w1, v1, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b);
        zget_wv(w2, v2, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c);
        zget_wv(w3, v3, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a);
        zget_wv(w4, v4, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b);
        zget_wv(w5, v5, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a);
        zpermute(z0, v0, nocc);
        zpermute(z1, v1, nocc);
        zpermute(z2, v2, nocc);
        zpermute(z3, v3, nocc);
        zpermute(z4, v4, nocc);
        zpermute(z5, v5, nocc);

        _ccsd_t_get_denorm(denorm, mo_energy, nocc, a, b, c);
        if (a == c) {
                for (i = 0; i < nooo; i++) {
                        denorm[i] *= 1./6;
                }
        } else if (a == b || b == c) {
                for (i = 0; i < nooo; i++) {
                        denorm[i] *= .5;
                }
        }
        for (i = 0; i < nooo; i++) {
                z0[i] = conj(z0[i]) * denorm[i];
                z1[i] = conj(z1[i]) * denorm[i];
                z2[i] = conj(z2[i]) * denorm[i];
                z3[i] = conj(z3[i]) * denorm[i];
                z4[i] = conj(z4[i]) * denorm[i];
                z5[i] = conj(z5[i]) * denorm[i];
        }

        double complex et = 0;
        et += _ccsd_t_zpermute_contract(z0, z1, z2, z3, z4, z5, w0, nocc);
        et += _ccsd_t_zpermute_contract(z1, z0, z4, z5, z2, z3, w1, nocc);
        et += _ccsd_t_zpermute_contract(z2, z3, z0, z1, z5, z4, w2, nocc);
        et += _ccsd_t_zpermute_contract(z3, z2, z5, z4, z0, z1, w3, nocc);
        et += _ccsd_t_zpermute_contract(z4, z5, z1, z0, z3, z2, w4, nocc);
        et += _ccsd_t_zpermute_contract(z5, z4, z3, z2, z1, z0, w5, nocc);
        return et;
}

void CCuccsd_t_zaaa(double complex *e_tot,
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
        double complex fvohalf[nvir*nocc];
        int i;
        for (i = 0; i < nvir*nocc; i++) {
                fvohalf[i] = fvo[i] * .5;
        }

#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvohalf, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) *
                                        (nocc*nocc*nocc*19+2));
        double complex e = 0;
#pragma omp for schedule (dynamic, 32)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6_aaa(nocc, nvir, a, b, c, mo_energy, t1T, t2T,
                                    nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                    fvohalf, vooo, cache1, jobs[k].cache);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}


/*************************************************
 *
 * UCCSD(T) beta-alpha-alpha
 *
 *************************************************/
static void zget_wv_baa(double complex *w, double complex *v,
                        double complex **vs_ts, double complex **cache,
                        int nocca, int noccb, int nvira, int nvirb,
                        int a, int b, int c)
{
        double complex *fvo = vs_ts[2];
        double complex *fVO = vs_ts[3];
        double complex *vooo = vs_ts[4];
        double complex *vOoO = vs_ts[5];
        double complex *VoOo = vs_ts[6];
        double complex *t1aT = vs_ts[7];
        double complex *t1bT = vs_ts[8];
        double complex *t2aaT = vs_ts[9];
        double complex *t2abT = vs_ts[10];
        double complex *vvop = cache[0];
        double complex *vVoP = cache[1];
        double complex *VvOp = cache[2];
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex D2 = 2;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const int nmoa = nocca + nvira;
        const int nmob = noccb + nvirb;
        const int noo = nocca * nocca;
        const int nOo = noccb * nocca;
        const size_t nooo = nocca * noo;
        const size_t noOo = nocca * nOo;
        const size_t nOoO = noccb * nOo;
        const size_t nvoo = nvira * noo;
        const int nVoO = nvirb * nOo;
        int i, j, k, n;

        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvira,
               &D2, vvop+nocca, &nmoa, t2abT+a*nOo, &nVoO,
               &D0, v, &nocca);
        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &nvirb,
               &D2, vVoP+noccb, &nmob, t2abT+b*(size_t)nVoO, &nOo,
               &D1, v, &nocca);
        zgemm_(&TRANS_N, &TRANS_T, &nOo, &nocca, &nocca,
               &D1, VoOo+a*noOo, &nOo, t2aaT+b*nvoo+c*noo, &nocca,
               &D1, v, &nOo);
        zgemm_(&TRANS_T, &TRANS_T, &nocca, &nOo, &noccb,
               &D2, t2abT+b*(size_t)nVoO+a*nOo, &noccb, vOoO+c*nOoO, &nOo,
               &D1, v, &nocca);
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                w[n] = v[j*nOo+i*nocca+k];
        } } }
        zgemm_(&TRANS_N, &TRANS_N, &noo, &noccb, &nvira,
               &D1, t2aaT+b*nvoo, &noo, VvOp+nocca, &nmoa,
               &D1, w, &noo);
        zgemm_(&TRANS_N, &TRANS_T, &noo, &noccb, &nocca,
               &D2, vooo+c*nooo, &noo, t2abT+b*(size_t)nVoO+a*nOo, &noccb,
               &D1, w, &noo);

        double complex t1aT2[nocca];
        double complex fvo2[nocca];
        double complex fVOhalf[noccb];
        for (i = 0; i < nocca; i++) {
                t1aT2[i] = t1aT[b*nocca+i] * 2;
                fvo2[i] = fvo[b*nocca+i] * 2;
        }
        for (i = 0; i < noccb; i++) {
                fVOhalf[i] = fVO[a*noccb+i] * .5;
        }
        double complex *pt2aaT = t2aaT + b * nvoo + c * noo;
        double complex *pt2abT = t2abT + (c*nvirb+a) * nOo;
        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                v[n] = (w[n] + vvop[k*nmoa+j] * t1bT[a*noccb+i]
                        + VvOp[i*nmoa+k] * t1aT2[j]
                        + pt2aaT[j*nocca+k] * fVOhalf[i]
                        + pt2abT[k*noccb+i] * fvo2[j]);
        } } }
}

/*
 * w - w.transpose(0,2,1)
 */
static void zpermute_baa(double complex *out, double complex *w, int nocca, int noccb)
{
        int noo = nocca * nocca;
        int n;
        int i, j, k;

        for (n = 0, i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++, n++) {
                out[n] = w[i*noo+j*nocca+k] - w[i*noo+k*nocca+j];
        } } }
}

static double complex
zpermute_contract_baa(double complex *z0, double complex *z1, double complex *w,
                      int nocca, int noccb)
{
        int noo = nocca * nocca;
        int i, j, k;
        double complex et = 0;

        for (i = 0; i < noccb; i++) {
        for (j = 0; j < nocca; j++) {
        for (k = 0; k < nocca; k++) {
                et += z0[i*noo+j*nocca+k] * w[i*noo+j*nocca+k];
                et += z1[i*noo+k*nocca+j] * w[i*noo+j*nocca+k];
        } } }
        return et;
}

static double complex
zcontract6_baa(int nocca, int noccb, int nvira, int nvirb,
               int a, int b, int c,
               double complex **vs_ts, void **cache, double complex *cache1)
{
        int nOoo = noccb * nocca * nocca;
        double complex *v0 = cache1;
        double complex *v1 = v0 + nOoo;
        double complex *w0 = v1 + nOoo;
        double complex *w1 = w0 + nOoo;
        double complex *z0 = w1 + nOoo;
        double complex *z1 = z0 + nOoo;
        double *denorm = (double *)(z1 + nOoo);
        int i;

        zget_wv_baa(w0, v0, vs_ts, ((double complex **)cache)  , nocca, noccb, nvira, nvirb, a, b, c);
        zget_wv_baa(w1, v1, vs_ts, ((double complex **)cache)+3, nocca, noccb, nvira, nvirb, a, c, b);
        zpermute_baa(z0, v0, nocca, noccb);
        zpermute_baa(z1, v1, nocca, noccb);

        double *mo_ea = (double *)vs_ts[0];
        double *mo_eb = (double *)vs_ts[1];
        get_denorm_baa(denorm, mo_ea, mo_eb, nocca, noccb, a, b, c);
        if (b == c) {
                for (i = 0; i < nOoo; i++) {
                        denorm[i] *= .5;
                }
        }
        for (i = 0; i < nOoo; i++) {
                z0[i] = conj(z0[i]) * denorm[i];
                z1[i] = conj(z1[i]) * denorm[i];
        }

        double complex et = 0;
        et += zpermute_contract_baa(z0, z1, w0, nocca, noccb);
        et += zpermute_contract_baa(z1, z0, w1, nocca, noccb);
        return et;
}



void CCuccsd_t_zbaa(double complex *e_tot,
                    double *mo_ea, double *mo_eb,
                    double complex *t1aT, double complex *t1bT,
                    double complex *t2aaT, double complex *t2abT,
                    double complex *vooo, double complex *vOoO, double complex *VoOo,
                    double complex *fvo, double complex *fVO,
                    int nocca, int noccb, int nvira, int nvirb,
                    int a0, int a1, int b0, int b1,
                    void *cache_row_a, void *cache_col_a,
                    void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = gen_baa_jobs(jobs, nocca, noccb, nvira, nvirb,
                                    a0, a1, b0, b1,
                                    cache_row_a, cache_col_a,
                                    cache_row_b, cache_col_b,
                                    sizeof(double complex));
        double complex *vs_ts[] = {(double complex *)mo_ea,
                (double complex *)mo_eb, fvo, fVO, vooo, vOoO, VoOo,
                t1aT, t1bT, t2aaT, t2abT};

#pragma omp parallel default(none) \
        shared(njobs, nocca, noccb, nvira, nvirb, vs_ts, jobs, e_tot)
{
        int a, b, c;
        size_t k;
        double complex *cache1 = malloc(sizeof(double complex) *
                                        (noccb*nocca*nocca*7+2));
        double complex e = 0;
#pragma omp for schedule (dynamic, 32)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6_baa(nocca, noccb, nvira, nvirb, a, b, c, vs_ts,
                                   jobs[k].cache, cache1);
        }
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
}

