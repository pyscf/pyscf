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

void make_tril_indices(size_t n, size_t *tril_indices)
{

    size_t i, j, ij;
    size_t idx = 0;
    for (ij = 0, i = 0; i < n; i++) {
        for (j = 0; j < n; j++, ij++) {
            if (i >= j) {
                tril_indices[ij] = idx++;
            // No valid tril index
            } else {
                tril_indices[ij] = -1;
            }
        }
    }
    assert (idx == n*(n+1)/2);
}


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
    const size_t KN = K*N;
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

        // Copy j3c_kpts(L,a,b,ki,kj) -> work1(kk,L,a,b,ki,kj)
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

    printf("C: Max imaginary element in j3c= %.2e\n", *max_imag);

    return ierr;
}
