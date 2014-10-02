/*
 * ah in rah4_ ... means anti-hermitian for e1, hermitian for e2
 * ha in rha4_ ... means hermitian for e1, anti-hermitian for e2
 * aa in raa4_ ... means anti-hermitian for e1, anti-hermitian for e2
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include "time_rev.h"
#include "r_direct_dot.h"

#define LOCIJKL \
const int ish = shls[0]; \
const int jsh = shls[1]; \
const int ksh = shls[2]; \
const int lsh = shls[3]; \
const int istart = ao_loc[ish]; \
const int jstart = ao_loc[jsh]; \
const int kstart = ao_loc[ksh]; \
const int lstart = ao_loc[lsh]; \
const int iend = ao_loc[ish+1]; \
const int jend = ao_loc[jsh+1]; \
const int kend = ao_loc[ksh+1]; \
const int lend = ao_loc[lsh+1]; \
const int di = iend - istart; \
const int dj = jend - jstart; \
const int dk = kend - kstart; \
const int dl = lend - lstart;


void CVHFrha1_ji_s1kl(double complex *eri,
                      double complex *dm, double complex *vj,
                      int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs1_ji_s1kl(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}

void CVHFrha1_lk_s1ij(double complex *eri,
                      double complex *dm, double complex *vj,
                      int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs1_lk_s1ij(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}

void CVHFrha1_jk_s1il(double complex *eri,
                      double complex *dm, double complex *vk,
                      int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs1_jk_s1il(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
}
void CVHFrha1_li_s1kj(double complex *eri,
                      double complex *dm, double complex *vk,
                      int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs1_li_s1kj(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
}

void CVHFrha2ij_ji_s1kl(double complex *eri,
                        double complex *dm, double complex *vj,
                        int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs2ij_ji_s1kl(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}
void CVHFrha2ij_lk_s2ij(double complex *eri,
                        double complex *dm, double complex *vj,
                        int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs2ij_lk_s2ij(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}

void CVHFrha2ij_jk_s1il(double complex *eri,
                        double complex *dm, double complex *vk,
                        int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs2ij_jk_s1il(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
}
void CVHFrha2ij_li_s1kj(double complex *eri,
                        double complex *dm, double complex *vk,
                        int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        CVHFrs2ij_li_s1kj(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
}

void CVHFrha2kl_ji_s2kl(double complex *eri,
                        double complex *dm, double complex *vj,
                        int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[2] >= shls[3]);
        CVHFrs1_ji_s1kl(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}
void CVHFrha2kl_lk_s1ij(double complex *eri,
                        double complex *dm, double complex *vj,
                        int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[2] >= shls[3]);
        if (shls[2] == shls[3]) {
                CVHFrs1_lk_s1ij(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
                return;
        }
        LOCIJKL;
        int i, j, k, l, ij, ic;
        double complex dm_lk;
        double complex sdm[dl*dk];
        double complex *pvj;

        CVHFtimerev_ijminus(sdm, dm, tao, lstart, lend, kstart, kend, nao);

        for (ic = 0; ic < ncomp; ic++) {
                pvj = vj + istart*nao+jstart;
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        dm_lk = sdm[l*dk+k];
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                pvj[i*nao+j] += eri[ij] * dm_lk;
                        } }
                        eri += di * dj;
                } }
                vj += nao*nao;
        }
}

void CVHFrha2kl_jk_s1il(double complex *eri,
                       double complex *dm, double complex *vk,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[2] >= shls[3]);

        CVHFrs1_jk_s1il(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
        if (shls[2] == shls[3]) {
                return;
        }

        LOCIJKL;
        int i, j, k, l, ij, ic;
        double complex sdm[dj*dl];
        double complex svk[di*dk];

        CVHFtimerev_j(sdm, dm, tao, jstart, jend, lstart, lend, nao);

        for (ic = 0; ic < ncomp; ic++) {
                memset(svk, 0, sizeof(double complex)*dk*di);
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                svk[i*dk+k] -= eri[ij] * sdm[j*dl+l];
                        } }
                        eri += di * dj;
                } }
                CVHFtimerev_adbak_j(svk, vk, tao, istart, iend, kstart, kend, nao);
                vk += nao*nao;
        }
}
void CVHFrha2kl_li_s1kj(double complex *eri,
                       double complex *dm, double complex *vk,
                       int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[2] >= shls[3]);

        CVHFrs1_li_s1kj(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
        if (shls[2] == shls[3]) {
                return;
        }

        LOCIJKL;
        int i, j, k, l, ij, ic;
        double complex sdm[di*dk];
        double complex svk[dj*dl];

        CVHFtimerev_i(sdm, dm, tao, kstart, kend, istart, iend, nao);

        for (ic = 0; ic < ncomp; ic++) {
                memset(svk, 0, sizeof(double complex)*dl*dj);
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                svk[l*dj+j] -= eri[ij] * sdm[k*di+i];
                        } }
                        eri += di * dj;
                } }
                CVHFtimerev_adbak_i(svk, vk, tao, lstart, lend, jstart, jend, nao);
                vk += nao*nao;
        }
}

void CVHFrha4_ji_s2kl(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[0] >= shls[1]);
        assert(shls[2] >= shls[3]);
        CVHFrs2ij_ji_s1kl(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}
void CVHFrha4_lk_s2ij(double complex *eri,
                     double complex *dm, double complex *vj,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[0] >= shls[1]);
        assert(shls[2] >= shls[3]);
        CVHFrha2kl_lk_s1ij(eri, dm, vj, nao, ncomp, shls, ao_loc, tao);
}

void CVHFrha4_jk_s1il(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[0] >= shls[1]);
        assert(shls[2] >= shls[3]);

        CVHFrha2kl_jk_s1il(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
        if (shls[0] == shls[1]) {
                return;
        }

        LOCIJKL;
        int i, j, k, l, ij, ic;
        int n = (di+dj)*(dk+dl);
        double complex sdm[n];
        double complex svk[n];
        double complex *peri = eri;
        double complex *pvk = vk;

        // tjtikl
        CVHFtimerev_i(sdm, dm, tao, istart, iend, kstart, kend, nao);
        for (ic = 0; ic < ncomp; ic++) {
                memset(svk, 0, sizeof(double complex)*dl*dj);
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                svk[j*dl+l] += peri[ij] * sdm[i*dk+k];
                        } }
                        peri += di * dj;
                } }
                CVHFtimerev_adbak_i(svk, pvk, tao, jstart, jend, lstart, lend, nao);
                pvk += nao*nao;
        }
        if (shls[2] == shls[3]) {
                return;
        }

        // tjtitltk
        CVHFtimerev_block(sdm, dm, tao, istart, iend, lstart, lend, nao);
        for (ic = 0; ic < ncomp; ic++) {
                memset(svk, 0, sizeof(double complex)*dk*dj);
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                svk[j*dk+k] -= eri[ij] * sdm[i*dl+l];
                        } }
                        eri += di * dj;
                } }
                CVHFtimerev_adbak_block(svk, vk, tao, jstart, jend, kstart, kend, nao);
                vk += nao*nao;
        }
}
// should be identical to CVHFrs4_jk_s1il
void CVHFrha4_li_s1kj(double complex *eri,
                     double complex *dm, double complex *vk,
                     int nao, int ncomp, int *shls, int *ao_loc, int *tao)
{
        assert(shls[0] >= shls[1]);
        assert(shls[2] >= shls[3]);

        CVHFrha2kl_li_s1kj(eri, dm, vk, nao, ncomp, shls, ao_loc, tao);
        if (shls[0] == shls[1]) {
                return;
        }

        LOCIJKL;
        int i, j, k, l, ij, ic;
        int n = (di+dj)*(dk+dl);
        double complex sdm[n];
        double complex svk[n];
        double complex *peri = eri;
        double complex *pvk = vk;

        // tjtikl
        CVHFtimerev_j(sdm, dm, tao, lstart, lend, jstart, jend, nao);
        for (ic = 0; ic < ncomp; ic++) {
                memset(svk, 0, sizeof(double complex)*dk*di);
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                svk[k*di+i] += peri[ij] * sdm[l*dj+j];
                        } }
                        peri += di * dj;
                } }
                CVHFtimerev_adbak_j(svk, pvk, tao, kstart, kend, istart, iend, nao);
                pvk += nao*nao;
        }
        if (shls[2] == shls[3]) {
                return;
        }

        // tjtitltk
        CVHFtimerev_block(sdm, dm, tao, kstart, kend, jstart, jend, nao);
        for (ic = 0; ic < ncomp; ic++) {
                memset(svk, 0, sizeof(double complex)*dl*di);
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (j = 0, ij = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                svk[l*di+i] -= eri[ij] * sdm[k*dj+j];
                        } }
                        eri += di * dj;
                } }
                CVHFtimerev_adbak_block(svk, vk, tao, lstart, lend, istart, iend, nao);
                vk += nao*nao;
        }
}

