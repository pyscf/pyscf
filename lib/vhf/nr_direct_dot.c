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
        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double *pdm;
        double tjk[MAXCGTO*MAXCGTO];
        memset(tjk, 0, sizeof(double)*dij);

        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + l*nao+k;
                for (ij = 0; ij < dij; ij++) {
                        tjk[ij] += eri[ij] * *pdm;
                }
                eri += dij;
        } }
        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        vjk[i*nao+j] += tjk[ij];
                        ij++;
                }
        }
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double tdm[MAXCGTO*MAXCGTO];
        double *pv;

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[i*nao+j] + dm[j*nao+i];
                        ij++;
                }
        }

        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k * nao + l;
                for (ij = 0; ij < dij; ij++) {
                        *pv += eri[ij] * tdm[ij];
                }
                eri += dij;
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double *pdm;
        double tjk[MAXCGTO*MAXCGTO];

        memset(tjk, 0, sizeof(double)*dij);
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + l*nao+k;
                for (ij = 0; ij < dij; ij++) {
                        tjk[ij] += eri[ij] * *pdm;
                }
                eri += dij;
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
        int i, j, k, l, ijkl;
        double *pdm, *pv;
        if (i0 > j0) {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pdm = dm + j * nao + k;
                        pv = vjk + j * nao + l;
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+l] += eri[ijkl] * *pdm;
                                *pv += eri[ijkl] * dm[i*nao+k];
                                ijkl++;
                        }
                } } }
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pdm = dm + j * nao;
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                ijkl++;
                        }
                } } }
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double tjk;
        double tdm[MAXCGTO*MAXCGTO];

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[j*nao+i];
                        ij++;
                }
        }

        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tjk = 0;
                for (ij = 0; ij < dij; ij++) {
                        tjk += eri[ij] * tdm[ij];
                }
                eri += dij;
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double tdm;
        double tjk[MAXCGTO*MAXCGTO];
        memset(tjk, 0, sizeof(double)*dij);

        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tdm = dm[k*nao+l] + dm[l*nao+k];
                for (ij = 0; ij < dij; ij++) {
                        tjk[ij] += eri[ij] * tdm;
                }
                eri += dij;
        } }
        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        vjk[i*nao+j] += tjk[ij];
                        ij++;
                }
        }
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
        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double *pv;
        double tjk;
        double tdm[MAXCGTO*MAXCGTO];

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[i*nao+j];
                        ij++;
                }
        }

        if (k0 == l0) {
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        for (ij = 0; ij < dij; ij++) {
                                *pv += eri[ij] * tdm[ij];
                        }
                        eri += dij;
                } }
        } else {
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk = 0;
                        for (ij = 0; ij < dij; ij++) {
                                tjk += eri[ij] * tdm[ij];
                        }
                        eri += dij;
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
        int i, j, k, l, ijkl;
        double *pdm, *pv;

        if (i0 == j0) {
                CVHFnrs2kl_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (k0 == l0) {
                CVHFnrs2ij_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else {
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pv = vjk + j * nao;
                        pdm = dm + j * nao;
                        for (i = i0; i < i1; i++) {
                                pv[k] += eri[ijkl] * dm[i*nao+l];
                                pv[l] += eri[ijkl] * dm[i*nao+k];
                                vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                vjk[i*nao+k] += eri[ijkl] * pdm[l];
                                ijkl++;
                        }
                } } }
        }
}
void CVHFnrs4_jk_s2il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        int i, j, k, l, ijkl;
        double *pdm, *pv;

        if (i0 == j0) {
                CVHFnrs2kl_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (k0 == l0) {
                CVHFnrs2ij_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                        for (j = j0; j < j1; j++) {
                                pdm = dm + j * nao;
                                for (i = i0; i < i1; i++) {
                                        vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                        ijkl++;
                                }
                        } } }
                } else { // l <= j < i < k
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                pdm = dm + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[l] += eri[ijkl] * dm[i*nao+k];
                                        vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                        ijkl++;
                                }
                        } } }
                }
        } else if (j0 < l0) { // j < l < k <= i
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pv = vjk + j * nao;
                        pdm = dm + j * nao;
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                vjk[i*nao+k] += eri[ijkl] * pdm[l];
                                ijkl++;
                        }
                } } }
        } else if (j0 < k0) { // l <= j < k <= i
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        for (j = j0; j < j1; j++) {
                                pv = vjk + j * nao;
                                pdm = dm + j * nao;
                                for (i = i0; i < i1; i++) {
                                        pv[l] += eri[ijkl] * dm[i*nao+k];
                                        vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                        vjk[i*nao+k] += eri[ijkl] * pdm[l];
                                        ijkl++;
                                }
                        }
                } }
        } else { // l < k <= j < i
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                        pv = vjk + j * nao;
                        pdm = dm + j * nao;
                        for (i = i0; i < i1; i++) {
                                pv[k] += eri[ijkl] * dm[i*nao+l];
                                pv[l] += eri[ijkl] * dm[i*nao+k];
                                vjk[i*nao+l] += eri[ijkl] * pdm[k];
                                vjk[i*nao+k] += eri[ijkl] * pdm[l];
                                ijkl++;
                        }
                } } }
        }
}

void CVHFnrs4_li_s1kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        if (i0 == j0) {
                CVHFnrs2kl_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (k0 == l0) {
                CVHFnrs2ij_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else {
                int i, j, k, l, ijkl;
                double *dmk, *dml, *pvk, *pvl;

                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pvk = vjk + k * nao;
                        pvl = vjk + l * nao;
                        dmk = dm + k * nao;
                        dml = dm + l * nao;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        pvk[j] += eri[ijkl] * dml[i];
                                        pvk[i] += eri[ijkl] * dml[j];
                                        pvl[j] += eri[ijkl] * dmk[i];
                                        pvl[i] += eri[ijkl] * dmk[j];
                                        ijkl++;
                                }
                        }
                } }
        }
}

void CVHFnrs4_li_s2kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        int i, j, k, l, ijkl;
        double *dmk, *dml, *pvk, *pvl;

        if (i0 == j0) {
                CVHFnrs2kl_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (k0 == l0) {
                CVHFnrs2ij_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pvk = vjk + k * nao;
                                dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                pvk[j] += eri[ijkl] * dml[i];
                                                ijkl++;
                                        }
                                }
                        } }
                } else { // j <= l < k < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pvk = vjk + k * nao;
                                pvl = vjk + l * nao;
                                dmk = dm + k * nao;
                                dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                pvk[j] += eri[ijkl] * dml[i];
                                                pvl[j] += eri[ijkl] * dmk[i];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        } else if (l0 < j0) { // l < j < i <= k
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pvk = vjk + k * nao;
                        pvl = vjk + l * nao;
                        dmk = dm + k * nao;
                        dml = dm + l * nao;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        pvk[j] += eri[ijkl] * dml[i];
                                        pvk[i] += eri[ijkl] * dml[j];
                                        ijkl++;
                                }
                        }
                } }
        } else if (l0 < i0) { // j <= l < i <= k
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pvk = vjk + k * nao;
                        pvl = vjk + l * nao;
                        dmk = dm + k * nao;
                        dml = dm + l * nao;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        pvk[j] += eri[ijkl] * dml[i];
                                        pvk[i] += eri[ijkl] * dml[j];
                                        pvl[j] += eri[ijkl] * dmk[i];
                                        ijkl++;
                                }
                        }
                } }
        } else { // j < i <= l < k
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pvk = vjk + k * nao;
                        pvl = vjk + l * nao;
                        dmk = dm + k * nao;
                        dml = dm + l * nao;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        pvk[j] += eri[ijkl] * dml[i];
                                        pvk[i] += eri[ijkl] * dml[j];
                                        pvl[j] += eri[ijkl] * dmk[i];
                                        pvl[i] += eri[ijkl] * dmk[j];
                                        ijkl++;
                                }
                        }
                } }
        }
}


void CVHFnrs8_ji_s1kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        if (i0 == k0 && j0 == l0) {
                CVHFnrs4_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (i0 == j0 || k0 == l0) {
                CVHFnrs4_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                CVHFnrs4_lk_s1ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else {
                int i, j, k, l, ij;
                int dij = (i1-i0) * (j1-j0);
                double tdm2, tjk2;
                double tdm1[MAXCGTO*MAXCGTO];
                double tjk1[MAXCGTO*MAXCGTO];
                memset(tjk1, 0, sizeof(double)*dij);

                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                tdm1[ij] = dm[i*nao+j] + dm[j*nao+i];
                                ij++;
                        }
                }

                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk2 = 0;
                        tdm2 = dm[k*nao+l] + dm[l*nao+k];
                        for (ij = 0; ij < dij; ij++) {
                                tjk2 += eri[ij] * tdm1[ij];
                                tjk1[ij] += eri[ij] * tdm2;
                        }
                        vjk[k*nao+l] += tjk2;
                        vjk[l*nao+k] += tjk2;
                        eri += dij;
                } }

                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+j] += tjk1[ij];
                                vjk[j*nao+i] += tjk1[ij];
                                ij++;
                        }
                }
        }
}
void CVHFnrs8_ji_s2kl(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        if (i0 == k0 && j0 == l0) {
                CVHFnrs4_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (i0 == j0 || k0 == l0) {
                CVHFnrs4_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                CVHFnrs4_lk_s2ij(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else {
                int i, j, k, l, ij;
                int dij = (i1-i0) * (j1-j0);
                double tdm2;
                double tdm1[MAXCGTO*MAXCGTO];
                double tjk[MAXCGTO*MAXCGTO];
                double *pv;
                memset(tjk, 0, sizeof(double)*dij);

                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                tdm1[ij] = dm[i*nao+j] + dm[j*nao+i];
                                ij++;
                        }
                }

                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        tdm2 = dm[k*nao+l] + dm[l*nao+k];
                        for (ij = 0; ij < dij; ij++) {
                                *pv += eri[ij] * tdm1[ij];
                                tjk[ij] += eri[ij] * tdm2;
                        }
                        eri += dij;
                } }

                ij = 0;
                for (j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++) {
                                vjk[i*nao+j] += tjk[ij];
                                ij++;
                        }
                }
        }
}

void CVHFnrs8_lk_s1ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs8_ji_s1kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}
void CVHFnrs8_lk_s2ij(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs8_ji_s2kl(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnrs8_li_s1kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        if (i0 == k0 && j0 == l0) {
                CVHFnrs4_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else if (i0 == j0 || k0 == l0) { // i0==l0 => i0==k0==l0
                CVHFnrs4_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                CVHFnrs4_jk_s1il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else {
                int i, j, k, l, ijkl;
                double *pvk, *pvl, *dmk, *dml;
                ijkl = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pvk = vjk + k * nao;
                        pvl = vjk + l * nao;
                        dmk = dm + k * nao;
                        dml = dm + l * nao;
                        for (j = j0; j < j1; j++) {
                                for (i = i0; i < i1; i++) {
                                        pvk[j] += eri[ijkl] * dml[i];
                                        pvk[i] += eri[ijkl] * dml[j];
                                        pvl[j] += eri[ijkl] * dmk[i];
                                        pvl[i] += eri[ijkl] * dmk[j];
                                        vjk[i*nao+k] += eri[ijkl] * dm[j*nao+l];
                                        vjk[i*nao+l] += eri[ijkl] * dm[j*nao+k];
                                        vjk[j*nao+k] += eri[ijkl] * dm[i*nao+l];
                                        vjk[j*nao+l] += eri[ijkl] * dm[i*nao+k];
                                        ijkl++;
                                }
                        }
                } }
        }
}
void CVHFnrs8_li_s2kj(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        assert(i0 >= j0);
        assert(k0 >= l0);
        if (i0 == k0) {
                CVHFnrs4_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                if (j0 != l0) {
                        CVHFnrs4_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                }
        } else if (i0 == j0 || k0 == l0) { // i0==l0 => i0==k0==l0
                CVHFnrs4_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
                CVHFnrs4_jk_s2il(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
        } else {
                int i, j, k, l, ijkl;
                double *pvk, *pvl, *dmk, *dml;

                if (j0 < l0) { // j <= l < k < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pvk = vjk + k * nao;
                                pvl = vjk + l * nao;
                                dmk = dm + k * nao;
                                dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                pvk[j] += eri[ijkl] * dml[i];
                                                pvl[j] += eri[ijkl] * dmk[i];
                                                vjk[i*nao+k] += eri[ijkl] * dm[j*nao+l];
                                                vjk[i*nao+l] += eri[ijkl] * dm[j*nao+k];
                                                ijkl++;
                                        }
                                }
                        } }
                } else if (j0 == l0) { // j == l < k < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pvk = vjk + k * nao;
                                pvl = vjk + l * nao;
                                dmk = dm + k * nao;
                                dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                pvk[j] += eri[ijkl] * dml[i];
                                                pvl[j] += eri[ijkl] * dmk[i];
                                                vjk[i*nao+k] += eri[ijkl] * dm[j*nao+l];
                                                vjk[i*nao+l] += eri[ijkl] * dm[j*nao+k];
                                                vjk[j*nao+l] += eri[ijkl] * dm[i*nao+k];
                                                ijkl++;
                                        }
                                }
                        } }
                } else if (j0 < k0) { // l < j < k < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pvk = vjk + k * nao;
                                dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                pvk[j] += eri[ijkl] * dml[i];
                                                vjk[i*nao+k] += eri[ijkl] * dm[j*nao+l];
                                                vjk[i*nao+l] += eri[ijkl] * dm[j*nao+k];
                                                vjk[j*nao+l] += eri[ijkl] * dm[i*nao+k];
                                                ijkl++;
                                        }
                                }
                        } }
                } else if (j0 == k0) { // l < j == k < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                pvk = vjk + k * nao;
                                dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                pvk[j] += eri[ijkl] * dml[i];
                                                vjk[j*nao+k] += eri[ijkl] * dm[i*nao+l];
                                                vjk[i*nao+k] += eri[ijkl] * dm[j*nao+l];
                                                vjk[i*nao+l] += eri[ijkl] * dm[j*nao+k];
                                                vjk[j*nao+l] += eri[ijkl] * dm[i*nao+k];
                                                ijkl++;
                                        }
                                }
                        } }
                } else { // l < k < j < i
                        ijkl = 0;
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                //pvk = vjk + k * nao;
                                //pvl = vjk + l * nao;
                                //dmk = dm + k * nao;
                                //dml = dm + l * nao;
                                for (j = j0; j < j1; j++) {
                                        for (i = i0; i < i1; i++) {
                                                //pvk[j] += eri[ijkl] * dml[i];
                                                //pvk[i] += eri[ijkl] * dml[j];
                                                //pvl[j] += eri[ijkl] * dmk[i];
                                                //pvl[i] += eri[ijkl] * dmk[j];
                                                vjk[i*nao+k] += eri[ijkl] * dm[j*nao+l];
                                                vjk[i*nao+l] += eri[ijkl] * dm[j*nao+k];
                                                vjk[j*nao+k] += eri[ijkl] * dm[i*nao+l];
                                                vjk[j*nao+l] += eri[ijkl] * dm[i*nao+k];
                                                ijkl++;
                                        }
                                }
                        } }
                }
        }
}

void CVHFnrs8_jk_s1il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs8_li_s1kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
}

void CVHFnrs8_jk_s2il(double *eri, double *dm, double *vjk,
                      int i0, int i1, int j0, int j1,
                      int k0, int k1, int l0, int l1, int nao)
{
        CVHFnrs8_li_s2kj(eri, dm, vjk, i0, i1, j0, j1, k0, k1, l0, l1, nao);
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double tdm[MAXCGTO*MAXCGTO];
        double *pv;

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[j*nao+i] - dm[i*nao+j];
                        ij++;
                }
        }

        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pv = vjk + k * nao + l;
                for (ij = 0; ij < dij; ij++) {
                        *pv += eri[ij] * tdm[ij];
                }
                eri += dij;
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double *pdm;
        double tjk[MAXCGTO*MAXCGTO];

        memset(tjk, 0, sizeof(double)*dij);
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                pdm = dm + l*nao+k;
                for (ij = 0; ij < dij; ij++) {
                        tjk[ij] += eri[ij] * *pdm;
                }
                eri += dij;
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

        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double tdm;
        double tjk[MAXCGTO*MAXCGTO];
        memset(tjk, 0, sizeof(double)*dij);

        ij = 0;
        for (l = l0; l < l1; l++) {
        for (k = k0; k < k1; k++) {
                tdm = dm[l*nao+k] - dm[k*nao+l];
                for (ij = 0; ij < dij; ij++) {
                        tjk[ij] += eri[ij] * tdm;
                }
                eri += dij;
        } }
        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        vjk[i*nao+j] += tjk[ij];
                        ij++;
                }
        }
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
        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double *pv;
        double tjk;
        double tdm[MAXCGTO*MAXCGTO];

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[i*nao+j];
                        ij++;
                }
        }

        if (k0 == l0) {
                ij = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        for (ij = 0; ij < dij; ij++) {
                                *pv -= eri[ij] * tdm[ij];
                        }
                        eri += dij;
                } }
        } else {
                ij = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk = 0;
                        for (ij = 0; ij < dij; ij++) {
                                tjk -= eri[ij] * tdm[ij];
                        }
                        eri += dij;
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
        int i, j, k, l, ij;
        int dij = (i1-i0) * (j1-j0);
        double *pv;
        double tjk;
        double tdm[MAXCGTO*MAXCGTO];

        ij = 0;
        for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++) {
                        tdm[ij] = dm[i*nao+j];
                        ij++;
                }
        }

        if (k0 == l0) {
                ij = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        pv = vjk + k * nao + l;
                        for (ij = 0; ij < dij; ij++) {
                                *pv += eri[ij] * tdm[ij];
                        }
                        eri += dij;
                } }
        } else {
                ij = 0;
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        tjk = 0;
                        for (ij = 0; ij < dij; ij++) {
                                tjk += eri[ij] * tdm[ij];
                        }
                        eri += dij;
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

