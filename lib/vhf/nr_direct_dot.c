/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAXCGTO 64

// eri in Fortran order; dm, vjk in C order

void CVHFnrs1_ji_s1kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pv, *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k * nao + l;
                for (j = j0; j < j1; j++) {
                        pdm = dm + nao*j;
                        for (i = i0; i < i1; i++) {
                                *pv += eri[ijkl] * pdm[i];
                                ijkl++;
                        }
                }
        } }
}
void CVHFnrs1_ji_s2kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        if (k0 >= l0) {
                CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}


void CVHFnrs1_lk_s1ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + l*nao+k;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+j] += eri[ijkl] * *pdm;
                                ijkl++;
                        }
                }
        } }
}
void CVHFnrs1_lk_s2ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        if (i0 >= j0) {
                CVHFnrs1_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}


void CVHFnrs1_jk_s1il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pdm = dm + j * nao + k;
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+l] += eri[ijkl] * *pdm;
                                ijkl++;
                        }
                }
        } }
}
void CVHFnrs1_jk_s2il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        if (i0 >= l0) {
                CVHFnrs1_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs1_li_s1kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm, *pv;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
                pdm = dm + l * nao;
                for (k = k0; k < k1; k++) {
                        for (j = j0; j < j1; j++) {
                                pv = vjk + k * nao + j;
                                for (i = i0; i < i1; i++) {
                                        *pv += eri[ijkl] * pdm[i];
                                        ijkl++;
                                }
                        }
                }
        }
}
void CVHFnrs1_li_s2kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        if (k0 >= j0) {
                CVHFnrs1_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

static void _jl_s1ik(double *eri, double *dm, double *vjk,
                     int i0, int i1, int j0, int j1,
                     int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pdm = dm + j * nao + l;
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+k] += eri[ijkl] * *pdm;
                                ijkl++;
                        }
                }
        } }
}
static void _lj_s1ki(double *eri, double *dm, double *vjk,
                     int i0, int i1, int j0, int j1,
                     int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pv, *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k *nao;
                for (j = j0; j < j1; j++) {
                        pdm = dm + l * nao + j;
                        for (i = i0; i < i1; i++) {
                                pv[i] += eri[ijkl] * *pdm;
                                ijkl++;
                        }
                }
        } }
}
static void _ik_s1jl(double *eri, double *dm, double *vjk,
                     int i0, int i1, int j0, int j1,
                     int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pv;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pv = vjk + j *nao;
                        for (i = i0; i < i1; i++) {
                                pv[l] += eri[ijkl] * dm[i*nao+k];
                                ijkl++;
                        }
                }
        } }
}
static void _ki_s1lj(double *eri, double *dm, double *vjk,
                     int i0, int i1, int j0, int j1,
                     int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm, *pv;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + k * nao;
                for (j = j0; j < j1; j++) {
                        pv = vjk + l * nao + j;
                        for (i = i0; i < i1; i++) {
                                *pv += eri[ijkl] * pdm[i];
                                ijkl++;
                        }
                }
        } }
}

void CVHFnrs2ij_ji_s1kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        if (i0 == j0) {
                return CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl, ij;
        double tdm[MAXCGTO*MAXCGTO];
        double *pv;

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[i*nao+j] + dm[j*nao+i];
                        ij++;
                }
        }

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k * nao + l;
                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                *pv += eri[ijkl] * tdm[ij];
                                ijkl++;
                                ij++;
                        }
                }
        } }
}
void CVHFnrs2ij_ji_s2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        if (k0 >= l0) {
                CVHFnrs2ij_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs2ij_lk_s1ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        if (i0 == j0) {
                return CVHFnrs1_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl, ij;
        double *pdm;
        double tjk[MAXCGTO*MAXCGTO];

        memset(tjk, 0, sizeof(double)*(i1-i0)*(j1-j0));
        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + l*nao+k;
                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                tjk[ij] += eri[ijkl] * *pdm;
                                ijkl++;
                                ij++;
                        }
                }
        } }

        ij = 0;
        for (j = j0; j < j1; j++) {
        for (i = i0; i < i1; i++) {
                vjk[i*nao+j] += tjk[ij];
                vjk[j*nao+i] += tjk[ij];
                ij++;
        } }
}
void CVHFnrs2ij_lk_s2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs2ij_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnrs2ij_jk_s1il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0) {
                _ik_s1jl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs2ij_jk_s2il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0 && j0 >= l0) {
                _ik_s1jl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs2ij_li_s1kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0) {
                _lj_s1ki(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs2ij_li_s2kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0 && k0 >= i0) {
                _lj_s1ki(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs2kl_ji_s1kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        if (k0 == l0) {
                return CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl;
        double *pdm;
        double tjk;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tjk = 0;
                for (j = j0; j < j1; j++) {
                        pdm = dm + nao*j;
                        for (i = i0; i < i1; i++) {
                                tjk += eri[ijkl] * pdm[i];
                                ijkl++;
                        }
                }
                vjk[k*nao+l] += tjk;
                vjk[l*nao+k] += tjk;
        } }
}
void CVHFnrs2kl_ji_s2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnrs2kl_lk_s1ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        if (k0 == l0) {
                return CVHFnrs1_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl;
        double tdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tdm = dm[k*nao+l] + dm[l*nao+k];
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+j] += eri[ijkl] * tdm;
                                ijkl++;
                        }
                }
        } }
}
void CVHFnrs2kl_lk_s2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        if (i0 >= j0) {
                CVHFnrs2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs2kl_jk_s1il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                _jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs2kl_jk_s2il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0 && i0 >= k0) {
                _jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs2kl_li_s1kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                _ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs2kl_li_s2kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0 && l0 >= j0) {
                _ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs4_ji_s1kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 == j0) {
                return;
        }
// swap ij
        int i, j, k, l, ijkl;
        double *pv;
        double tjk;

        if (k0 == l0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        *pv += eri[ijkl] * dm[i*nao+j];
                                        ijkl++;
                                }
                        }
                } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk = 0;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        tjk += eri[ijkl] * dm[i*nao+j];
                                        ijkl++;
                                }
                        }
                        vjk[k*nao+l] += tjk;
                        vjk[l*nao+k] += tjk;
                } }
        }
}

void CVHFnrs4_ji_s2kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnrs4_lk_s1ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 == j0) {
                return;
        }
// swap ij
        int i, j, k, l, ijkl;
        double *pv, *pdm;
        double tdm;

        if (k0 == l0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pdm = dm + l*nao+k;
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[i] += eri[ijkl] * *pdm;
                                        ijkl++;
                                }
                        }
                } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tdm = dm[l*nao+k] + dm[k*nao+l];
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[i] += eri[ijkl] * tdm;
                                        ijkl++;
                                }
                        }
                } }
        }
}

void CVHFnrs4_lk_s2ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnrs4_jk_s1il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                _jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);

                if (i0 > j0) {
                        int i, j, k, l, ijkl;
                        double *pv;
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        pv = vjk + j * nao + k;
                                        for (i = i0; i < i1; i++) {
                                                *pv += eri[ijkl] * dm[i*nao+l];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}
void CVHFnrs4_jk_s2il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                if (i0 >= k0) {
                        _jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
                if (i0 > j0 && j0 >= k0) {
                        int i, j, k, l, ijkl;
                        double *pv;
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        pv = vjk + j * nao + k;
                                        for (i = i0; i < i1; i++) {
                                                *pv += eri[ijkl] * dm[i*nao+l];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnrs4_li_s1kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                _ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                if (i0 > j0) {
                        int i, j, k, l, ijkl;
                        double *pdm, *pv;

                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pv = vjk + l * nao;
                                for (j = j0; j < j1; j++) {
                                        pdm = dm + k * nao + j;
                                        for (i = i0; i < i1; i++) {
                                                pv[i] += eri[ijkl] * *pdm;
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnrs4_li_s2kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                if (l0 >= j0) {
                        _ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
                if (i0 > j0 && l0 >= i0) {
                        int i, j, k, l, ijkl;
                        double *pdm, *pv;

                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pv = vjk + l * nao;
                                for (j = j0; j < j1; j++) {
                                        pdm = dm + k * nao + j;
                                        for (i = i0; i < i1; i++) {
                                                pv[i] += eri[ijkl] * *pdm;
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}


void CVHFnrs8_ji_s1kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs8_ji_s2kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_lk_s2ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs8_lk_s1ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs8_lk_s2ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_lk_s2ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs8_jk_s1il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs8_jk_s2il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnrs8_li_s1kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnrs8_li_s2kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs4_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 != k0 || j0 != l0) {
                CVHFnrs4_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}



/*************************************************
 * For anti symmetrized integrals
 *************************************************/
static void a_jl_s1ik(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pdm = dm + j * nao + l;
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+k] -= eri[ijkl] * *pdm;
                                ijkl++;
                        }
                }
        } }
}
static void a_lj_s1ki(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pv, *pdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k *nao;
                for (j = j0; j < j1; j++) {
                        pdm = dm + l * nao + j;
                        for (i = i0; i < i1; i++) {
                                pv[i] -= eri[ijkl] * *pdm;
                                ijkl++;
                        }
                }
        } }
}
static void a_ik_s1jl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pv;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pv = vjk + j *nao;
                        for (i = i0; i < i1; i++) {
                                pv[l] -= eri[ijkl] * dm[i*nao+k];
                                ijkl++;
                        }
                }
        } }
}
static void a_ki_s1lj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        int i, j, k, l, ijkl;
        double *pdm, *pv;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + k * nao;
                for (j = j0; j < j1; j++) {
                        pv = vjk + l * nao + j;
                        for (i = i0; i < i1; i++) {
                                *pv -= eri[ijkl] * pdm[i];
                                ijkl++;
                        }
                }
        } }
}

void CVHFnra2ij_ji_s1kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        if (i0 == j0) {
                return CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl, ij;
        double tdm[MAXCGTO*MAXCGTO];
        double *pv;

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[j*nao+i] - dm[i*nao+j];
                        ij++;
                }
        }

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k * nao + l;
                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                *pv += eri[ijkl] * tdm[ij];
                                ijkl++;
                                ij++;
                        }
                }
        } }
}
void CVHFnra2ij_ji_s2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        if (k0 >= l0) {
                CVHFnra2ij_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnra2ij_lk_s1ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        if (i0 == j0) {
                return CVHFnrs1_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl, ij;
        double *pdm;
        double tjk[MAXCGTO*MAXCGTO];

        memset(tjk, 0, sizeof(double)*(i1-i0)*(j1-j0));
        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + l*nao+k;
                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                tjk[ij] += eri[ijkl] * *pdm;
                                ijkl++;
                                ij++;
                        }
                }
        } }

        ij = 0;
        for (j = j0; j < j1; j++) {
        for (i = i0; i < i1; i++) {
                vjk[i*nao+j] += tjk[ij];
                vjk[j*nao+i] -= tjk[ij];
                ij++;
        } }
}
void CVHFnra2ij_lk_s2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnra2ij_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}
void CVHFnra2ij_lk_a2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnra2ij_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnra2ij_jk_s1il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0) {
                a_ik_s1jl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnra2ij_jk_s2il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0 && j0 >= l0) {
                a_ik_s1jl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnra2ij_li_s1kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0) {
                a_lj_s1ki(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnra2ij_li_s2kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        CVHFnrs1_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 > j0 && k0 >= i0) {
                a_lj_s1ki(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnra2kl_ji_s1kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        if (k0 == l0) {
                return CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl;
        double *pdm;
        double tjk;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tjk = 0;
                for (j = j0; j < j1; j++) {
                        pdm = dm + nao*j;
                        for (i = i0; i < i1; i++) {
                                tjk += eri[ijkl] * pdm[i];
                                ijkl++;
                        }
                }
                vjk[k*nao+l] += tjk;
                vjk[l*nao+k] -= tjk;
        } }
}
void CVHFnra2kl_ji_s2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}
void CVHFnra2kl_ji_a2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnra2kl_lk_s1ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        if (k0 == l0) {
                return CVHFnrs1_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }

        int i, j, k, l, ijkl;
        double tdm;

        ijkl = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tdm = dm[l*nao+k] - dm[k*nao+l];
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+j] += eri[ijkl] * tdm;
                                ijkl++;
                        }
                }
        } }
}
void CVHFnra2kl_lk_s2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        if (i0 >= j0) {
                CVHFnra2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnra2kl_jk_s1il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                a_jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnra2kl_jk_s2il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0 && i0 >= k0) {
                a_jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnra2kl_li_s1kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                a_ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}
void CVHFnra2kl_li_s2kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(k0 >= l0);
        CVHFnrs1_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0 && l0 >= j0) {
                a_ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        }
}

void CVHFnra4ij_ji_s1kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 == j0) {
                return;
        }
// swap ij
        int i, j, k, l, ijkl;
        double *pv;
        double tjk;

        if (k0 == l0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        *pv -= eri[ijkl] * dm[i*nao+j];
                                        ijkl++;
                                }
                        }
                } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk = 0;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        tjk -= eri[ijkl] * dm[i*nao+j];
                                        ijkl++;
                                }
                        }
                        vjk[k*nao+l] += tjk;
                        vjk[l*nao+k] += tjk;
                } }
        }
}

void CVHFnra4ij_ji_s2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2ij_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnra4ij_lk_s1ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 == j0) {
                return;
        }
// swap ij
        int i, j, k, l, ijkl;
        double *pv, *pdm;
        double tdm;

        if (k0 == l0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pdm = dm + l*nao+k;
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[i] -= eri[ijkl] * *pdm;
                                        ijkl++;
                                }
                        }
                } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tdm = dm[l*nao+k] + dm[k*nao+l];
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[i] -= eri[ijkl] * tdm;
                                        ijkl++;
                                }
                        }
                } }
        }
}
void CVHFnra4ij_lk_s2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}
void CVHFnra4ij_lk_a2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnra4ij_jk_s1il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2ij_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                _jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);

                if (i0 > j0) {
                        int i, j, k, l, ijkl;
                        double *pv;
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        pv = vjk + j * nao + k;
                                        for (i = i0; i < i1; i++) {
                                                *pv -= eri[ijkl] * dm[i*nao+l];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}
void CVHFnra4ij_jk_s2il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2ij_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                if (i0 >= k0) {
                        _jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
                if (i0 > j0 && j0 >= k0) {
                        int i, j, k, l, ijkl;
                        double *pv;
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        pv = vjk + j * nao + k;
                                        for (i = i0; i < i1; i++) {
                                                *pv -= eri[ijkl] * dm[i*nao+l];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnra4ij_li_s1kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                _ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                if (i0 > j0) {
                        int i, j, k, l, ijkl;
                        double *pdm, *pv;

                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pv = vjk + l * nao;
                                for (j = j0; j < j1; j++) {
                                        pdm = dm + k * nao + j;
                                        for (i = i0; i < i1; i++) {
                                                pv[i] -= eri[ijkl] * *pdm;
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnra4ij_li_s2kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                if (l0 >= j0) {
                        _ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
                if (i0 > j0 && l0 >= i0) {
                        int i, j, k, l, ijkl;
                        double *pdm, *pv;

                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pv = vjk + l * nao;
                                for (j = j0; j < j1; j++) {
                                        pdm = dm + k * nao + j;
                                        for (i = i0; i < i1; i++) {
                                                pv[i] -= eri[ijkl] * *pdm;
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnra4kl_ji_s1kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2kl_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 == j0) {
                return;
        }
// swap ij
        int i, j, k, l, ijkl;
        double *pv;
        double tjk;

        if (k0 == l0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        *pv += eri[ijkl] * dm[i*nao+j];
                                        ijkl++;
                                }
                        }
                } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk = 0;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        tjk += eri[ijkl] * dm[i*nao+j];
                                        ijkl++;
                                }
                        }
                        vjk[k*nao+l] += tjk;
                        vjk[l*nao+k] -= tjk;
                } }
        }
}
void CVHFnra4kl_ji_s2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2kl_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}
void CVHFnra4kl_ji_a2kl(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2kl_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnra4kl_lk_s1ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (i0 == j0) {
                return;
        }
// swap ij
        int i, j, k, l, ijkl;
        double *pv, *pdm;
        double tdm;

        if (k0 == l0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pdm = dm + l*nao+k;
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[i] += eri[ijkl] * *pdm;
                                        ijkl++;
                                }
                        }
                } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tdm = dm[l*nao+k] - dm[k*nao+l];
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[i] += eri[ijkl] * tdm;
                                        ijkl++;
                                }
                        }
                } }
        }
}

void CVHFnrs4kl_lk_s2ij(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnra2kl_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnra4kl_jk_s1il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                a_jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);

                if (i0 > j0) {
                        int i, j, k, l, ijkl;
                        double *pv;
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        pv = vjk + j * nao + k;
                                        for (i = i0; i < i1; i++) {
                                                *pv -= eri[ijkl] * dm[i*nao+l];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}
void CVHFnra4kl_jk_s2il(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                if (i0 >= k0) {
                        a_jl_s1ik(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
                if (i0 > j0 && j0 >= k0) {
                        int i, j, k, l, ijkl;
                        double *pv;
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                        pv = vjk + j * nao + k;
                                        for (i = i0; i < i1; i++) {
                                                *pv -= eri[ijkl] * dm[i*nao+l];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnra4kl_li_s1kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                a_ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                if (i0 > j0) {
                        int i, j, k, l, ijkl;
                        double *pdm, *pv;

                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pv = vjk + l * nao;
                                for (j = j0; j < j1; j++) {
                                        pdm = dm + k * nao + j;
                                        for (i = i0; i < i1; i++) {
                                                pv[i] -= eri[ijkl] * *pdm;
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnra4kl_li_s2kj(double *eri, double *dm, double *vjk,
                        int i0, int i1, int j0, int j1,
                        int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        CVHFnrs2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        if (k0 > l0) {
                if (l0 >= j0) {
                        a_ki_s1lj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
                if (i0 > j0 && l0 >= i0) {
                        int i, j, k, l, ijkl;
                        double *pdm, *pv;

                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pv = vjk + l * nao;
                                for (j = j0; j < j1; j++) {
                                        pdm = dm + k * nao + j;
                                        for (i = i0; i < i1; i++) {
                                                pv[i] -= eri[ijkl] * *pdm;
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

