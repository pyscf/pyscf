#include "fft.h"
#include <omp.h>
#include <string.h>
#include <complex.h>
#include "vhf/fblas.h"
#include <math.h>

int get_omp_threads();
int omp_get_thread_num();

void _pack_aoR_to_aoPairR_diff(
    double *aoR_i,
    double *aoR_j,
    double *aoPairR,
    int nao_i,
    int nao_j,
    int ngrid)
{
    int nPair = nao_i * nao_j;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nPair; i++)
    {
        int i1 = i / nao_j;
        int j1 = i % nao_j;
        for (int k = 0; k < ngrid; k++)
        {
            aoPairR[i * ngrid + k] = aoR_i[i1 * ngrid + k] * aoR_j[j1 * ngrid + k];
        }
    }
}

void _pack_aoR_to_aoPairR_same(
    double *aoR,
    double *aoPairR,
    int nao,
    int ngrid)
{
    // int nPair = nao * (nao + 1) / 2;

#pragma omp parallel for schedule(static)
    for (int i1 = 0; i1 < nao; ++i1)
    {
        for (int j1 = 0; j1 <= i1; ++j1)
        {
            int i = i1 * (i1 + 1) / 2 + j1;
            for (int k = 0; k < ngrid; ++k)
            {
                aoPairR[i * ngrid + k] = aoR[i1 * ngrid + k] * aoR[j1 * ngrid + k];
            }
        }
    }
}

#define COMBINE2(i, j) ((i) < (j) ? (j) * (j + 1) / 2 + i : i * (i + 1) / 2 + j)

void _unpack_suberi_to_eri(
    double *eri,
    const int nao,
    double *suberi,
    const int nao_bra,
    const int *ao_loc_bra,
    const int nao_ket,
    const int *ao_loc_ket)
{
    int nPair = nao * (nao + 1) / 2;

    int nPair_ket = nao_ket * (nao_ket + 1) / 2;

    // printf("nao_bra  : %d\n", nao_bra);
    // printf("nao_ket  : %d\n", nao_ket);
    // printf("nPair    : %d\n", nPair);
    // printf("nPair_ket: %d\n", nPair_ket);

#pragma omp parallel for schedule(static)
    for (int i1 = 0; i1 < nao_bra; ++i1)
    {
        for (int j1 = 0; j1 <= i1; ++j1)
        {
            int i = ao_loc_bra[i1];
            int j = ao_loc_bra[j1];
            int ij = COMBINE2(i, j);
            int i1j1 = COMBINE2(i1, j1);
            // printf("i1: %d, j1: %d, i: %d, j: %d, ij: %d, i1j1: %d\n", i1, j1, i, j, ij, i1j1);
            for (int k1 = 0; k1 < nao_ket; ++k1)
            {
                for (int l1 = 0; l1 <= k1; ++l1)
                {
                    int k = ao_loc_ket[k1];
                    int l = ao_loc_ket[l1];
                    int kl = COMBINE2(k, l);
                    int k1l1 = COMBINE2(k1, l1);
                    eri[ij * nPair + kl] += suberi[i1j1 * nPair_ket + k1l1];
                }
            }
        }
    }
}

#undef COMBINE2