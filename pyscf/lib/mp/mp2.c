/* Copyright 2014-2021 The PySCF Developers. All Rights Reserved.

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
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "mp/mp2.h"


/*  Get an array of pointers for each row of a 2D array of size n-by-m
*/
const double **_gen_ptr_arr(const double *p0, const size_t n, const size_t m)
{
    size_t i;
    const double *p;
    const double **parr = malloc(sizeof(double *) * n);
    for (i = 0, p = p0; i < n; ++i) {
        parr[i] = p;
        p += m;
    }
    return parr;
}

/*  Generate MP2 jobs for AO range (i0,i0+nocci,j0,j0+noccj) with/without s2 symmetry.
*/
size_t _MP2_gen_jobs(CacheJob *jobs, const int s2symm,
                     const size_t i0, const size_t j0, const size_t nocci, const size_t noccj)
{
    size_t i, j, ii, jj, m;
    if (s2symm) {
        for (m = 0, i = 0, ii = i0; i < nocci; ++i, ++ii) {
            for (j = 0, jj = j0; j < noccj; ++j, ++jj) {
                if (jj > ii) {
                    continue;
                }
                jobs[m].i = i;
                jobs[m].j = j;
                jobs[m].fac = (ii==jj)?(1):(2);
                ++m;
            }
        }
    } else {
        for (m = 0, i = 0, ii = i0; i < nocci; ++i, ++ii) {
            for (j = 0, jj = j0; j < noccj; ++j, ++jj) {
                jobs[m].i = i;
                jobs[m].j = j;
                jobs[m].fac = 1;
                ++m;
            }
        }
    }
    return m;
}

/*  Calculate DF-RMP2 energy for AO range (i0,i0+nocci,j0,j0+noccj) with real integrals

    Math:
        for i in range(i0,i0+nocci):
            for j in range(j0,j0+noccj):
                if s2symm:
                    if i > j: continue
                    fac = 1 if i == j else 2
                else:
                    fac = 1
                vab = einsum('aL,bL->ab', iaL[i-i0], jbL[j-j0])
                vabT = vab.T
                tab = vab / ∆eab
                ed_out += dot(vab, tab) * fac
                ex_out -= dot(vabT, tab) * fac
*/
void MP2_contract_d(double *ed_out, double *ex_out, const int s2symm,
                    const double *batch_iaL, const double *batch_jbL,
                    const int i0, const int j0, const int nocci, const int noccj,
                    const int nocc, const int nvir, const int naux,
                    const double *moeoo, const double *moevv,
                    double *t2_out, const int t2_ex)
{

    const int I1 = 1;
    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_Y = 'T';
    const char TRANS_N = 'N';

    const int nvv = nvir*nvir;
    const int nvx = nvir*naux;

    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci*noccj);
    size_t njob = _MP2_gen_jobs(jobs, s2symm, i0, j0, nocci, noccj);

    const double **parr_iaL = _gen_ptr_arr(batch_iaL, nocci, nvx);
    const double **parr_jbL = _gen_ptr_arr(batch_jbL, noccj, nvx);
    const double **parr_t2 = NULL;
    if (t2_out) {
        parr_t2 = _gen_ptr_arr(t2_out, nocc*nocc, nvv);
    }

#pragma omp parallel default(none) \
        shared(njob, jobs, batch_iaL, batch_jbL, parr_iaL, parr_jbL, moeoo, moevv, naux, i0, j0, nocc, nvir, nvv, noccj, D0, D1, I1, TRANS_N, TRANS_Y, ed_out, ex_out, parr_t2, t2_ex)
{
    double *cache = malloc(sizeof(double) * nvv*3);
    double *vab = cache;
    double *vabT = vab + nvv;
    double *tab = vabT + nvv;
    double eij;

    const double *iaL, *jbL;
    size_t i,j,a,m, ii, jj;
    double ed=0, ex=0, fac;

#pragma omp for schedule (dynamic, 4)

    for (m = 0; m < njob; ++m) {
        i = jobs[m].i;
        j = jobs[m].j;
        fac = jobs[m].fac;

        iaL = parr_iaL[i];
        jbL = parr_jbL[j];
        eij = moeoo[i*noccj+j];

        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbL, &naux, iaL, &naux,
               &D0, vab, &nvir);
        NPdtranspose(nvir, nvir, vab, vabT);
        // tab = vab / eijab
        for (a = 0; a < nvv; ++a) {
            tab[a] = vab[a] / (eij - moevv[a]);
        }
        // vab, tab -> ed
        ed += ddot_(&nvv, vab, &I1, tab, &I1) * fac;
        // vab, tba -> ex
        ex -= ddot_(&nvv, vabT, &I1, tab, &I1) * fac;

        // save t2
        if (parr_t2) {
            ii = i + i0;
            jj = j + j0;
            if (t2_ex) {
                NPdtranspose(nvir, nvir, tab, vabT);
                for (a = 0; a < nvv; ++a) {
                    tab[a] -= vabT[a];
                }
            }
            NPdcopy(parr_t2[ii*nocc+jj], tab, nvv);
            if (ii != jj) {
                NPdtranspose(nvir, nvir, tab, parr_t2[jj*nocc+ii]);
            }
        }
    }
    free(cache);

#pragma omp critical
{
    *ed_out += ed;
    *ex_out += ex;
}

} // parallel

    free(jobs);
    free(parr_iaL);
    free(parr_jbL);

}

/*  Calculate DF-RMP2 OS energy for AO range (i0,i0+nocci,j0,j0+noccj) with real integrals

    Math:
        for i in range(i0,i0+nocci):
            for j in range(j0,j0+noccj):
                vab = einsum('aL,bL->ab', iaL[i-i0], jbL[j-j0])
                tab = vab / ∆eab
                ed_out += dot(vab, tab) * fac
*/
void MP2_OS_contract_d(double *ed_out,
                       const double *batch_iaL, const double *batch_jbL,
                       const int i0, const int j0, const int nocci, const int noccj,
                       const int nocca, const int noccb,
                       const int nvira, const int nvirb, const int naux,
                       const double *moeoo, const double *moevv,
                       double *t2_out)
{

    const int I1 = 1;
    const double D0 = 0;
    const double D1 = 1;
    const char TRANS_Y = 'T';
    const char TRANS_N = 'N';

    const int nvv = nvira*nvirb;
    const int nvax = nvira*naux;
    const int nvbx = nvirb*naux;

    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci*noccj);
    size_t njob = _MP2_gen_jobs(jobs, 0, i0, j0, nocci, noccj);

    const double **parr_iaL = _gen_ptr_arr(batch_iaL, nocci, nvax);
    const double **parr_jbL = _gen_ptr_arr(batch_jbL, noccj, nvbx);
    const double **parr_t2 = NULL;
    if (t2_out) {
        parr_t2 = _gen_ptr_arr(t2_out, nocca*noccb, nvv);
    }

#pragma omp parallel default(none) \
        shared(njob, jobs, batch_iaL, batch_jbL, parr_iaL, parr_jbL, moeoo, moevv, naux, i0, j0, nocca, noccb, nvira, nvirb, nvv, noccj, D0, D1, I1, TRANS_N, TRANS_Y, ed_out, parr_t2)
{
    double *cache = malloc(sizeof(double) * nvv*2);
    double *vab = cache;
    double *tab = vab + nvv;
    double eij;

    const double *iaL, *jbL;
    size_t i,j,a,m, ii, jj;
    double ed=0, fac;

#pragma omp for schedule (dynamic, 4)

    for (m = 0; m < njob; ++m) {
        i = jobs[m].i;
        j = jobs[m].j;
        fac = jobs[m].fac;

        iaL = parr_iaL[i];
        jbL = parr_jbL[j];
        eij = moeoo[i*noccj+j];

        dgemm_(&TRANS_Y, &TRANS_N, &nvirb, &nvira, &naux,
               &D1, jbL, &naux, iaL, &naux,
               &D0, vab, &nvirb);
        // tab = vab / eijab
        for (a = 0; a < nvv; ++a) {
            tab[a] = vab[a] / (eij - moevv[a]);
        }
        // vab, tab -> ed
        ed += ddot_(&nvv, vab, &I1, tab, &I1) * fac;

        if (parr_t2) {
            ii = i + i0;
            jj = j + j0;
            NPdcopy(parr_t2[ii*noccb+jj], tab, nvv);
        }
    }
    free(cache);

#pragma omp critical
{
    *ed_out += ed;
}

} // parallel

    free(jobs);
    free(parr_iaL);
    free(parr_jbL);

}


/*  Calculate DF-RMP2 energy for AO range (i0,i0+nocci,j0,j0+noccj) with complex integrals

    Math:
        for i in range(i0,i0+nocci):
            for j in range(j0,j0+noccj):
                if s2symm:
                    if i > j: continue
                    fac = 1 if i == j else 2
                else:
                    fac = 1
                vab = einsum('aL,bL->ab', iaL[i-i0], jbL[j-j0])
                vabT = vab.T
                tab = conj(vab) / ∆eab
                ed_out += dot(vab, tab) * fac
                ex_out -= dot(vabT, tab) * fac
*/
void MP2_contract_c(double *ed_out, double *ex_out, const int s2symm,
                    const double *batch_iaLR, const double *batch_iaLI,
                    const double *batch_jbLR, const double *batch_jbLI,
                    const int i0, const int j0, const int nocci, const int noccj,
                    const int nvir, const int naux,
                    const double *moeoo, const double *moevv)
{

    const int I1 = 1;
    const double D0 = 0;
    const double D1 = 1;
    const double Dm1 = -1;
    const char TRANS_Y = 'T';
    const char TRANS_N = 'N';

    const int nvv = nvir*nvir;
    const int nvx = nvir*naux;

    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci*noccj);
    size_t njob = _MP2_gen_jobs(jobs, s2symm, i0, j0, nocci, noccj);

    const double **parr_iaLR = _gen_ptr_arr(batch_iaLR, nocci, nvx);
    const double **parr_iaLI = _gen_ptr_arr(batch_iaLI, nocci, nvx);
    const double **parr_jbLR = _gen_ptr_arr(batch_jbLR, noccj, nvx);
    const double **parr_jbLI = _gen_ptr_arr(batch_jbLI, noccj, nvx);

#pragma omp parallel default(none) \
        shared(njob, jobs, batch_iaLR, batch_iaLI, batch_jbLR, batch_jbLI, parr_iaLR, parr_iaLI, parr_jbLR, parr_jbLI, moeoo, moevv, naux, nvir, nvv, noccj, D0, D1, Dm1, I1, TRANS_N, TRANS_Y, ed_out, ex_out)
{
    double *cache = malloc(sizeof(double) * nvv*6);
    double *vabR = cache;
    double *vabI = vabR + nvv;
    double *vabTR = vabI + nvv;
    double *vabTI = vabTR + nvv;
    double *tabR = vabTI + nvv;
    double *tabI = tabR + nvv;
    double eij;

    const double *iaLR, *iaLI, *jbLR, *jbLI;
    size_t i,j,a,m;
    double ed=0, ex=0, fac;

#pragma omp for schedule (dynamic, 4)

    for (m = 0; m < njob; ++m) {
        i = jobs[m].i;
        j = jobs[m].j;
        fac = jobs[m].fac;

        iaLR = parr_iaLR[i]; iaLI = parr_iaLI[i];
        jbLR = parr_jbLR[j]; jbLI = parr_jbLI[j];
        eij = moeoo[i*noccj+j];

        // einsum([i]aL,[j]bL) -> [i][j]ab
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbLR, &naux, iaLR, &naux,
               &D0, vabR, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &Dm1, jbLI, &naux, iaLI, &naux,
               &D1, vabR, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbLR, &naux, iaLI, &naux,
               &D0, vabI, &nvir);
        dgemm_(&TRANS_Y, &TRANS_N, &nvir, &nvir, &naux,
               &D1, jbLI, &naux, iaLR, &naux,
               &D1, vabI, &nvir);
        NPdtranspose(nvir, nvir, vabR, vabTR);
        NPdtranspose(nvir, nvir, vabI, vabTI);
        // tab = vab / eijab
        for (a = 0; a < nvv; ++a) {
            tabR[a] =  vabR[a] / (eij - moevv[a]);
            tabI[a] = -vabI[a] / (eij - moevv[a]);
        }
        // vab, tab -> ed
        ed += ddot_(&nvv, vabR, &I1, tabR, &I1) * fac;
        ed -= ddot_(&nvv, vabI, &I1, tabI, &I1) * fac;
        // vab_ex, tab -> ex
        ex -= ddot_(&nvv, vabTR, &I1, tabR, &I1) * fac;
        ex += ddot_(&nvv, vabTI, &I1, tabI, &I1) * fac;
    }
    free(cache);

#pragma omp critical
{
    *ed_out += ed;
    *ex_out += ex;
}

} // parallel

    free(jobs);
    free(parr_iaLR); free(parr_iaLI);
    free(parr_jbLR); free(parr_jbLI);

}

/*  Calculate DF-RMP2 OS energy for AO range (i0,i0+nocci,j0,j0+noccj) with complex integrals

    Math:
        for i in range(i0,i0+nocci):
            for j in range(j0,j0+noccj):
                vab = einsum('aL,bL->ab', iaL[i-i0], jbL[j-j0])
                tab = conj(vab) / ∆eab
                ed_out += dot(vab, tab)
*/
void MP2_OS_contract_c(double *ed_out,
                       const double *batch_iaLR, const double *batch_iaLI,
                       const double *batch_jbLR, const double *batch_jbLI,
                       const int i0, const int j0, const int nocci, const int noccj,
                       const int nvira, const int nvirb, const int naux,
                       const double *moeoo, const double *moevv)
{

    const int I1 = 1;
    const double D0 = 0;
    const double D1 = 1;
    const double Dm1 = -1;
    const char TRANS_Y = 'T';
    const char TRANS_N = 'N';

    const int nvv = nvira*nvirb;
    const int nvax = nvira*naux;
    const int nvbx = nvirb*naux;

    CacheJob *jobs = malloc(sizeof(CacheJob) * nocci*noccj);
    size_t njob = _MP2_gen_jobs(jobs, 0, i0, j0, nocci, noccj);

    const double **parr_iaLR = _gen_ptr_arr(batch_iaLR, nocci, nvax);
    const double **parr_iaLI = _gen_ptr_arr(batch_iaLI, nocci, nvax);
    const double **parr_jbLR = _gen_ptr_arr(batch_jbLR, noccj, nvbx);
    const double **parr_jbLI = _gen_ptr_arr(batch_jbLI, noccj, nvbx);

#pragma omp parallel default(none) \
        shared(njob, jobs, batch_iaLR, batch_iaLI, batch_jbLR, batch_jbLI, parr_iaLR, parr_iaLI, parr_jbLR, parr_jbLI, moeoo, moevv, naux, nvira, nvirb, nvv, noccj, D0, D1, Dm1, I1, TRANS_N, TRANS_Y, ed_out)
{
    double *cache = malloc(sizeof(double) * nvv*4);
    double *vabR = cache;
    double *vabI = vabR + nvv;
    double *tabR = vabI + nvv;
    double *tabI = tabR + nvv;
    double eij;

    const double *iaLR, *iaLI, *jbLR, *jbLI;
    size_t i,j,a,m;
    double ed=0, fac;

#pragma omp for schedule (dynamic, 4)

    for (m = 0; m < njob; ++m) {
        i = jobs[m].i;
        j = jobs[m].j;
        fac = jobs[m].fac;

        iaLR = parr_iaLR[i]; iaLI = parr_iaLI[i];
        jbLR = parr_jbLR[j]; jbLI = parr_jbLI[j];
        eij = moeoo[i*noccj+j];

        // einsum([i]aL,[j]bL) -> [i][j]ab
        dgemm_(&TRANS_Y, &TRANS_N, &nvirb, &nvira, &naux,
               &D1, jbLR, &naux, iaLR, &naux,
               &D0, vabR, &nvirb);
        dgemm_(&TRANS_Y, &TRANS_N, &nvirb, &nvira, &naux,
               &Dm1, jbLI, &naux, iaLI, &naux,
               &D1, vabR, &nvirb);
        dgemm_(&TRANS_Y, &TRANS_N, &nvirb, &nvira, &naux,
               &D1, jbLR, &naux, iaLI, &naux,
               &D0, vabI, &nvirb);
        dgemm_(&TRANS_Y, &TRANS_N, &nvirb, &nvira, &naux,
               &D1, jbLI, &naux, iaLR, &naux,
               &D1, vabI, &nvirb);
        // tab = vab / eijab
        for (a = 0; a < nvv; ++a) {
            tabR[a] =  vabR[a] / (eij - moevv[a]);
            tabI[a] = -vabI[a] / (eij - moevv[a]);
        }
        // vab, tab -> ed
        ed += ddot_(&nvv, vabR, &I1, tabR, &I1) * fac;
        ed -= ddot_(&nvv, vabI, &I1, tabI, &I1) * fac;
    }
    free(cache);

#pragma omp critical
{
    *ed_out += ed;
}

} // parallel

    free(jobs);
    free(parr_iaLR); free(parr_iaLI);
    free(parr_jbLR); free(parr_jbLI);

}

#ifdef _OPENMP
void trisolve_parallel_grp(double *low, double *b, const int n, const int nrhs, const int grpfac)
{
    int ntmax = omp_get_max_threads();
    int ngrp = grpfac*ntmax;
    int mgrp = floor((double) (nrhs + ngrp - 1) / ngrp);
    ngrp = (int) floor( (double) (nrhs + mgrp - 1) / mgrp);
    const double **parr_b = _gen_ptr_arr(b, nrhs, n);
#pragma omp parallel default(none) shared(low, b, n, nrhs, ngrp, mgrp, parr_b)
{
    const char SIDE = 'L';
    const char UPLO = 'L';
    const char TRANS = 'N';
    const char DIAG = 'N';
    const double D1 = 1;

    int i;
    int info;
    double * bi;

    int igrp, i0, di;

#pragma omp for schedule (dynamic, 4)
    for (igrp = 0; igrp < ngrp; ++igrp) {
        i0 = igrp * mgrp;
        di = (i0+mgrp<=nrhs) ? (mgrp) : (nrhs-i0);
        dtrsm_(&SIDE, &UPLO, &TRANS, &DIAG, &n, &di, &D1, low, &n, parr_b[i0], &n);
    }
} // parallel
}
#endif
