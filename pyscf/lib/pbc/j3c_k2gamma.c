#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <assert.h>


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
            //printf("i=%ld j=%ld ij=%ld idx=%ld\n", i, j, ij, idx);
        }
    }
    assert (idx == n*(n+1)/2);
}

void j3c_k2gamma(
        /* In */
        int64_t nk,                 // Number of k-points
        int64_t nao,                // Number of AOs
        int64_t naux,               // Size of auxiliary basis
        int64_t *kconserv,          // Momentum conserving k-point for each ki,kj
        double complex *phase,      // As returned from k2gamma.get_phase
        double complex *j3c_k,      // k-point sampled 3c-integrals (naux, nk, nao, nk, nao)
        /* Out */
        double complex *j3c)        // (nk*naux, (nk*nao)*(nk*nao+1)/2)
{
    int64_t ierr = 0;
    int64_t ki, kj, kij, kl, kl2;
    int64_t ri, rj, l, a, b, ab;
    int64_t Ria, Rjb;
    int64_t idx = 0;
    const int64_t nao2 = nao*nao;
    const int64_t nk2 = nk*nk;
    const int64_t ncomp = nk*nao * (nk*nao+1) / 2;
    const size_t tmpsize = naux*nk2*nao2;
    //const size_t tmpsize = naux*nk2*nao2;
    //double complex *tmp = calloc(tmpsize, sizeof(double complex));
    double complex tmp = 0.0;
    size_t *tril_indices = malloc(nk2*nao2 * sizeof(size_t));
    make_tril_indices(nk*nao, tril_indices);

    printf("Starting C...\n");
    printf("nk= %ld naux= %ld nao= %ld ncomp= %ld\n", nk, naux, nao, ncomp);
    printf("kconserv= %ld %ld ... %ld\n", kconserv[0], kconserv[1], kconserv[nk2-1]);
    printf("First/last element of j3c_k.real C= %e %e %e \n", creal(j3c_k[0]), creal(j3c_k[1]), creal(j3c_k[tmpsize-1]));
    printf("First/last element of j3c_k.imag C= %e %e %e \n", cimag(j3c_k[0]), cimag(j3c_k[1]), cimag(j3c_k[tmpsize-1]));

    for (kij = 0, ki = 0; ki < nk; ki++) {
    //for (kj = 0; kj <= ki; kj++, kij++) {
    for (kj = 0; kj < nk; kj++, kij++) {
        //memset(tmp, 0, tmpsize * sizeof(double complex));
        kl = kconserv[ki*nk + kj];
        printf("ki= %ld kj= %ld -> kl= %ld\n", ki, kj, kl);
        for (ri = 0; ri < nk; ri++) {
        for (rj = 0; rj < nk; rj++) {
            //printf("ri= %ld rj= %ld\n", ri, rj);
            for (ab = 0, a = 0; a < nao; a++) {
            for (b = 0; b < nao; b++, ab++) {
            //printf("a= %ld b= %ld\n", a, b);
                for (l = 0; l < naux; l++) {
                    //printf("l= %ld\n", l);
                    //tmp[l*nk2*nao2 + ri*nk*nao2 + rj*nao2 + ab] +=
                    //printf("MARK 1\n");
                    assert(l*nk2*nao2 + ki*nk*nao2 + a*nk*nao + kj*nao + b < tmpsize);
                    tmp = j3c_k[l*nk2*nao2 + ki*nk*nao2 + a*nk*nao + kj*nao + b] * phase[ri*nk + ki] * conj(phase[rj*nk + kj]);
                    //printf("MARK 2\n");

                    Ria = ri*nao + a;
                    Rjb = rj*nao + b;
                    idx = tril_indices[Ria*nk*nao + Rjb];
                    assert (idx < ncomp);
                    //printf("MARK 3\n");
                    //if (a <= b) {
                    if (Ria <= Rjb) {
                        assert (idx != -1);
                        //j3c[kl*tmpsize + l*nk2*nao2 + ri*nk*nao2 + a*nk*nao + rj*nao + b] += tmp;
                        assert (kl*naux*ncomp + l*ncomp + idx < nk*naux*ncomp);
                        //j3c[kl*naux*ncomp + l*ncomp + idx] += tmp;
                        //idx++;
                    } else {
                        assert (idx == -1);
                    }
                    //printf("MARK 4\n");
                    //printf("MARK 3\n");
                    //if (ki < kj) {
                    //    kl2 = kconserv[kj*nk + ki];
                    //    j3c[kl2*tmpsize + l*nk2*nao2 + rj*nk*nao2 + b*nk*nao + ri*nao + a] += tmp;
                    //}
                    //printf("MARK 4\n");
                }
            }}
        }}
    }}

    //free(tmp);
    printf("First/last element of j3c.real in C= %e %e %e \n", creal(j3c[0]), creal(j3c[1]), creal(j3c[nk*naux*ncomp-1]));
    printf("First/last element of j3c.imag in C= %e %e %e \n", cimag(j3c[0]), cimag(j3c[1]), cimag(j3c[nk*naux*ncomp-1]));
    printf("Freeing...\n");
    free(tril_indices);
    printf("Return to Python...\n");
    //return ierr;
}
