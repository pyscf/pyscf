#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define USE_BLAS

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <cblas.h>

int64_t j3c_k2gamma(
        /* In */
        int64_t nk,                 // Number of k-points
        int64_t nao,                // Number of AOs
        int64_t naux,               // Size of auxiliary basis
        int64_t *kconserv,          // Momentum conserving k-point for each ki,kj
        double complex *phase,      // As returned from k2gamma.get_phase
        double complex *j3c_kpts,   // k-point sampled 3c-integrals (naux, nao, nao, nk, nk)
        bool compact,               // compact
        /* Out */
        double *j3c,                // (nk*naux, (nk*nao)*(nk*nao+1)/2)
        double *max_imag)           // Max encountered imaginary element
{
    int64_t ierr = 0;
    const size_t N = nao;
    const size_t K = nk;
    const size_t N2 = N*N;
    const size_t K2 = K*K;
    //const size_t KN = K*N;
    const size_t KN2 = K*N2;
    const size_t K2N = K2*N;
    const size_t K2N2 = K2*N2;
    const size_t K3N2 = K*K2*N2;
    const size_t NCOMP = (N*K)*(N*K+1)/2;

    const double complex Z0 = 0.0;
    const double complex Z1 = 1.0;

    // Precompute phase.conj() (avoid using CblasConjNoTrans, not BLAS standard, OpenBLAS specific)
    size_t i;
    double complex *phase_cc = malloc(K2 * sizeof(double complex));
    for (i = 0; i < K2; i++) {
        phase_cc[i] = conj(phase[i]);
    }

#pragma omp parallel private(i)
    {
    size_t l, a, b, ab;
    size_t ki, kj, kk;
    size_t ri, rj, rk;
    size_t idx;
    double rtmp = 0.0;
    double complex *work1 = malloc(K3N2 * sizeof(double complex));
    double complex *work2 = malloc(K3N2 * sizeof(double complex));

#pragma omp for reduction(+:ierr)
    for (l = 0; l < naux; l++) {

        memset(work1, 0, K3N2 * sizeof(double complex));

        // Copy j3c_kpts(L|a,b,ki,kj) -> work1(kk,L,a,b,ki,kj)
        for (ki = 0; ki < nk; ki++) {
        for (kj = 0; kj < nk; kj++) {
            kk = kconserv[ki*K + kj];
        for (ab = 0; ab < N2; ab++) {
            work1[kk*K2N2 + ab*K2 + ki*K+kj] = j3c_kpts[l*K2N2 + ab*K2 + ki*K + kj];
        }}}


        // FT kj -> rj
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K2N2, K, K, &Z1, work1, K, phase_cc, K, &Z0, work2, K);

        // Reorder work2(kk,a,b,ki,rj) -> work1(kk,a,b,rj,ki)
        for (i = 0; i < KN2; i++) {
        for (ki = 0; ki < K; ki++) {
        for (rj = 0; rj < K; rj++) {
            work1[i*K2 + rj*K + ki] = work2[i*K2 + ki*K + rj];
        }}}

        // FT ki -> ri
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K2N2, K, K, &Z1, work1, K, phase, K, &Z0, work2, K);

        // FT kk -> rk
        cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K2N2, K, &Z1, phase, K, work2, K2N2, &Z0, work1, K2N2);

        // Reorder work1(rk,a,b,rj,ri) -> j3c(rk,l,ri,a,rj,b)
        for (rk = 0; rk < K; rk++) {
            idx = 0;
        for (ri = 0; ri < K; ri++) {
        for (a = 0; a < N; a++) {
        for (rj = 0; rj < K; rj++) {
        for (b = 0; b < N; b++) {
            i = rk*K2N2 + a*K2N + b*K2 + rj*K + ri;
            rtmp = fabs(cimag(work1[i]));
            //printf("imaginary part = %.2e\n", rtmp);
            if (rtmp > *max_imag) {
                *max_imag = rtmp;
            //if (rtmp > max_imag) {
                //if (ierr == 0) {
                 //   printf("ERROR: signficant imaginary part= %.2e !\n", rtmp);
                //    ierr = 1;
                //}
            }

            // Fill all
            if (!compact) {
                j3c[rk*naux*K2N2 + l*K2N2 + idx++] = creal(work1[i]);
            }
            //Only fill lower triangular
            else if (ri*N+a >= rj*N+b) {
                j3c[rk*naux*NCOMP + l*NCOMP + idx++] = creal(work1[i]);
            }

        }}}}}
    }

    free(work1);
    free(work2);
    }

    free(phase_cc);

    //printf("C: Max imaginary element in j3c= %.2e\n", *max_imag);
    return ierr;
}

/* Transform three-center integrals from k-AOs to Gamma-MOs */
int64_t j3c_kao2gmo(
        /*** In ***/
        int64_t nk,                 // Number of k-points
        int64_t nao,                // Number of atomic orbitals in primitive cells
        int64_t nocc,               // Number of occupied orbitals in supercell
        int64_t nvir,               // Number of virtual orbitals in supercell
        int64_t naux,               // Number of auxiliary basis functions in primitive cell
        int64_t *kconserv,          // (nk, nk) Momentum conserving k-point for each ki,kj
        int64_t *kuniqmap,          // (nk, nk) mapping from (ki,kj) -> unique(ki-kj); if NULL, ki*nk+kj is used
        double complex *cocc,       // (nk, nao, nocc) Occupied MO coefficients in k-space representation
        double complex *cvir,       // (nk, nao, nvir) Virtual MO coefficients in k-space representation
        double complex *j3c,        // (nkuniq, naux, nao, nao) k-point sampled 3c-integrals. nkuniq is nk**2 if kuniqmap==NULL, or nk*(nk+1)/2 else
        //double complex *phase,    // (nk, nk) exp^{iR.K} // TODO
        /*** Inout ***/
        double complex *j3c_ov,     // (nk,naux,nocc,nvir) Three-center integrals (kL|ia)
        double complex *j3c_oo,     // (nk,naux,nocc,nocc) Three-center integrals (kL|ij) (Will not be calculated if NULL on entry)
        double complex *j3c_vv      // (nk,naux,nvir,nvir) Three-center integrals (kL|ab) (Will not be calculated if NULL on entry)
        )
{
    int64_t ierr = 0;
    const int64_t nmax = MAX(nocc, nvir);

#pragma omp parallel
    {
    /* Dummy indices */
    size_t l;           // DF basis
    size_t ki, kj, kk;  // k-points
    int64_t kij;        // ki,kj index
    size_t a, b;        // atomic orbitals
#ifdef USE_BLAS
    const double complex Z1 = 1.0;
#else
    /* Dummy indices */
    size_t i, j;        // occupied molecular orbitals
    size_t p, q;        // virtual molecular orbitals
#endif
    double complex *j3c_pt = NULL;
    double complex *work = malloc(nmax*nao * sizeof(double complex));
    double complex *work2 = malloc(nao*nao * sizeof(double complex));
    if (!(work && work2)) {
        printf("Error allocating temporary memory in j3c_kao2gamma. Exiting.\n");
        ierr = 1;
    }
// Do not perform calculation if any threads did not get memory
#pragma omp barrier
    if (ierr != 0) goto EXIT;
/* Parallelize over auxiliary index guarantees thread-safety (not atomic pragmas needed)
 * However Parallelizing over final k-point kk would also be thread-safe and possibly more efficient
 * (L can be included in the ZGEMMs) */
#pragma omp for
    for (l = 0; l < naux; l++) {
        //printf("l= %ld on thread %d / %d...\n", l, omp_get_thread_num(), omp_get_num_threads());
        for (ki = 0; ki < nk; ki++) {
        for (kj = 0; kj < nk; kj++) {

            kij = ki*nk + kj;
            kk = kconserv[kij];
            if (kuniqmap) {
                kij = kuniqmap[kij];
            }
            j3c_pt = &(j3c[(labs(kij)*naux + l)*nao*nao]);
            // Tranpose AOs if kij negative
            if (kij < 0) {
                for (a = 0; a < nao; a++) {
                for (b = 0; b < nao; b++) {
                    work2[b*nao+a] = conj(j3c_pt[a*nao+b]);
                }}
                j3c_pt = work2;
            }
            /* (L|occ,vir)
             * For occ-vir three-center integrals, the contraction order can make a big difference in FLOPs!
             * Choose best contraction order automatically */

            /* Contract occupieds, then virtuals */
            if (nocc <= nvir || j3c_oo) {
                memset(work, 0, nocc*nao * sizeof(double complex));
#ifdef USE_BLAS
                // (L|ao,ao) -> (L|occ,ao)
                cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, nocc, nao, nao,
                        &Z1, &(cocc[ki*nao*nocc]), nocc, j3c_pt, nao,
                        &Z1, work, nao);
                // (L|occ,ao) -> (L|occ,vir)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nocc, nvir, nao,
                        &Z1, work, nao, &cvir[kj*nao*nvir], nvir,
                        &Z1, &(j3c_ov[kk*nvir*nocc*naux + l*nvir*nocc]), nvir);
#else
                // (L|ao,ao) -> (L|occ,ao)
                for (a = 0; a < nao; a++) {
                for (b = 0; b < nao; b++) {
                for (i = 0; i < nocc; i++) {
                    work[i*nao + b] += j3c_pt[a*nao + b] * conj(cocc[ki*nao*nocc + a*nocc + i]);
                }}}
                // (L|occ,ao) -> (L|occ,vir)
                for (b = 0; b < nao; b++) {
                for (i = 0; i < nocc; i++) {
                for (p = 0; p < nvir; p++) {
                    j3c_ov[kk*nvir*nocc*naux + l*nvir*nocc + i*nvir + p] += work[i*nao + b] * cvir[kj*nao*nvir + b*nvir + p];
                }}}
#endif
            /* Contract virtuals, then occupieds */
            } else {
                memset(work, 0, nao*nvir * sizeof(double complex));
#ifdef USE_BLAS
                // (L|ao,ao) -> (L|ao,vir)
                cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nao, nvir, nao,
                        &Z1, j3c_pt, nao, &cvir[kj*nao*nvir], nvir,
                        &Z1, work, nvir);
                // (L|ao,vir) -> (L|occ,vir)
                cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, nocc, nvir, nao,
                        &Z1, &(cocc[ki*nao*nocc]), nocc, work, nvir,
                        &Z1, &(j3c_ov[kk*nvir*nocc*naux + l*nvir*nocc]), nvir);
#else
                // (L|ao,ao) -> (L|ao,vir)
                for (a = 0; a < nao; a++) {
                for (b = 0; b < nao; b++) {
                for (p = 0; p < nvir; p++) {
                    work[a*nvir + p] += j3c_pt[a*nao + b] * cvir[kj*nao*nvir + b*nvir + p];
                }}}
                // (L|ao,vir) -> (L|occ,vir)
                for (a = 0; a < nao; a++) {
                for (i = 0; i < nocc; i++) {
                for (p = 0; p < nvir; p++) {
                    j3c_ov[kk*nvir*nocc*naux + l*nvir*nocc + i*nvir + p] += work[a*nvir + p] * conj(cocc[ki*nao*nocc + a*nocc + i]);
                }}}
#endif
            }

            /* (L|occ,occ) */
            if (j3c_oo) {
#ifdef USE_BLAS
            // (L|occ,ao) -> (L|occ,occ)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nocc, nocc, nao,
                    &Z1, work, nao, &cocc[kj*nao*nocc], nocc,
                    &Z1, &(j3c_oo[kk*nocc*nocc*naux + l*nocc*nocc]), nocc);
#else
            // (L|occ,ao) -> (L|occ,occ)
            for (b = 0; b < nao; b++) {
            for (i = 0; i < nocc; i++) {
            for (j = 0; j < nocc; j++) {
            j3c_oo[kk*nocc*nocc*naux + l*nocc*nocc + i*nocc + j] += work[i*nao + b] * cocc[kj*nao*nocc + b*nocc + j];
            }}}
#endif
            }

            /* (L|vir,vir) */
            if (j3c_vv) {
            memset(work, 0, nvir*nao * sizeof(double complex));
#ifdef USE_BLAS
            // (L|ao,ao) -> (L|vir,ao)
            cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, nvir, nao, nao,
                    &Z1, &(cvir[ki*nao*nvir]), nvir, j3c_pt, nao,
                    &Z1, work, nao);
            // (L|vir,ao) -> (L|vir,vir)
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nvir, nvir, nao,
                    &Z1, work, nao, &cvir[kj*nao*nvir], nvir,
                    &Z1, &(j3c_vv[kk*nvir*nvir*naux + l*nvir*nvir]), nvir);
#else
            // (L|ao,ao) -> (L|vir,ao)
            for (a = 0; a < nao; a++) {
            for (b = 0; b < nao; b++) {
            for (p = 0; p < nvir; p++) {
                work[p*nao + b] += j3c_pt[a*nao + b] * conj(cvir[ki*nao*nvir + a*nvir + p]);
            }}}
            // (L|vir,ao) -> (L|vir,vir)
            for (b = 0; b < nao; b++) {
            for (p = 0; p < nvir; p++) {
            for (q = 0; q < nvir; q++) {
            j3c_vv[kk*nvir*nvir*naux + l*nvir*nvir + p*nvir + q] += work[p*nao + b] * cvir[kj*nao*nvir + b*nvir + q];
            }}}
#endif
            }
        }}
    } // loop over l
    //} // end if(work)

    //TODO
    ///* Rotate to real values */
    //if (phase) {
    //    *work = realloc(nk*nmax*nmax * sizeof(double complex));
    //    for (ki = 0; ki < nk; ki++) {
    //        memcpy(&(work[ki*nocc*nvir]), &(j3c_ov[(ki*naux + l)*nocc*nvir]), nocc*nvir * sizeof(double complex));
    //    }
    //    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nk, nocc*nvir, nk,
    //            &Z1, phase, nk, work, nocc*nvir,
    //            &Z1, j3c_oc, nao);
    //}

EXIT:
    free(work);
    free(work2);
    } // end of parallel region

    //printf("Returning to python...\n");
    return ierr;
}
