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
 * JKoperator
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "nr_direct.h"

#define ASSERT(expr, msg) \
        if (!(expr)) { fprintf(stderr, "Fail at %s\n", msg); exit(1); }

#define MAXCGTO 64

#define ISH0    0
#define ISH1    1
#define JSH0    2
#define JSH1    3
#define KSH0    4
#define KSH1    5
#define LSH0    6
#define LSH1    7

#define JKOP_ALLOCATE(ibra, iket, obra, oket) \
        static JKArray *JKOperator_allocate_##ibra##iket##obra##oket(int *shls_slice, int *ao_loc, int ncomp) \
{ \
        JKArray *jkarray = malloc(sizeof(JKArray)); \
        jkarray->dm_bra_sh0 = shls_slice[ibra##SH0]; \
        jkarray->dm_bra_sh1 = shls_slice[ibra##SH1]; \
        jkarray->dm_ket_sh0 = shls_slice[iket##SH0]; \
        jkarray->dm_ket_sh1 = shls_slice[iket##SH1]; \
        jkarray->v_bra_sh0  = shls_slice[obra##SH0]; \
        jkarray->v_bra_sh1  = shls_slice[obra##SH1]; \
        jkarray->v_ket_sh0  = shls_slice[oket##SH0]; \
        jkarray->v_ket_sh1  = shls_slice[oket##SH1]; \
        jkarray->v_ket_nsh  = shls_slice[oket##SH1] - shls_slice[oket##SH0]; \
        jkarray->offset0_outptr = jkarray->v_bra_sh0 * jkarray->v_ket_nsh + jkarray->v_ket_sh0; \
        jkarray->dm_dims[0] = ao_loc[shls_slice[ibra##SH1]] - ao_loc[shls_slice[ibra##SH0]]; \
        jkarray->dm_dims[1] = ao_loc[shls_slice[iket##SH1]] - ao_loc[shls_slice[iket##SH0]]; \
        jkarray->v_dims[0]  = ao_loc[shls_slice[obra##SH1]] - ao_loc[shls_slice[obra##SH0]]; \
        jkarray->v_dims[1]  = ao_loc[shls_slice[oket##SH1]] - ao_loc[shls_slice[oket##SH0]]; \
        int outptr_size =((shls_slice[obra##SH1] - shls_slice[obra##SH0]) * \
                          (shls_slice[oket##SH1] - shls_slice[oket##SH0])); \
        jkarray->outptr = malloc(sizeof(int) * outptr_size); \
        memset(jkarray->outptr, NOVALUE, sizeof(int) * outptr_size); \
        jkarray->stack_size = 0; \
        int data_size = jkarray->v_dims[0] * jkarray->v_dims[1] * ncomp; \
        jkarray->data = malloc(sizeof(double) * data_size); \
        jkarray->ncomp = ncomp; \
        return jkarray; \
}

#define JKOP_DATA_SIZE(obra, oket) \
        static size_t JKOperator_data_size_##obra##oket(int *shls_slice, int *ao_loc) \
{ \
        int nbra = ao_loc[shls_slice[obra##SH1]] - ao_loc[shls_slice[obra##SH0]]; \
        int nket = ao_loc[shls_slice[oket##SH1]] - ao_loc[shls_slice[oket##SH0]]; \
        return nbra * nket; \
}

#define ADD_JKOP(fname, ibra, iket, obra, oket, type) \
JKOperator CVHF##fname = {JKOperator_allocate_##ibra##iket##obra##oket, \
        JKOperator_deallocate, fname, JKOperator_data_size_##obra##oket, \
        JKOperator_sanity_check_##type}

JKOP_ALLOCATE(J, I, K, L)
JKOP_ALLOCATE(L, K, I, J)
JKOP_ALLOCATE(L, I, K, J)
JKOP_ALLOCATE(J, K, I, L)
JKOP_ALLOCATE(J, L, I, K)
JKOP_ALLOCATE(L, J, K, I)
JKOP_ALLOCATE(I, K, J, L)
JKOP_ALLOCATE(K, I, L, J)
JKOP_DATA_SIZE(K, L)
JKOP_DATA_SIZE(I, J)
JKOP_DATA_SIZE(K, J)
JKOP_DATA_SIZE(I, L)
JKOP_DATA_SIZE(K, I)
JKOP_DATA_SIZE(I, K)
JKOP_DATA_SIZE(J, L)
JKOP_DATA_SIZE(L, J)

static void JKOperator_deallocate(JKArray *jkarray)
{
        free(jkarray->outptr);
        free(jkarray->data);
        free(jkarray);
}

static void JKOperator_sanity_check_s1(int *shls_slice)
{
}
static void JKOperator_sanity_check_s2ij(int *shls_slice)
{
        ASSERT(((shls_slice[0] == shls_slice[2]) &&
                (shls_slice[1] == shls_slice[3])), "s2ij");
}
static void JKOperator_sanity_check_s2kl(int *shls_slice)
{
        ASSERT(((shls_slice[4] == shls_slice[6]) &&
                (shls_slice[5] == shls_slice[7])), "s2kl");
}
static void JKOperator_sanity_check_s4(int *shls_slice)
{
        ASSERT(((shls_slice[0] == shls_slice[2]) &&
                (shls_slice[1] == shls_slice[3])), "s4 ij");
        ASSERT(((shls_slice[4] == shls_slice[6]) &&
                (shls_slice[5] == shls_slice[7])), "s4 kl");
}
static void JKOperator_sanity_check_s8(int *shls_slice)
{
        ASSERT(((shls_slice[0] == shls_slice[2]) &&
                (shls_slice[1] == shls_slice[3])), "s8 ij");
        ASSERT(((shls_slice[4] == shls_slice[6]) &&
                (shls_slice[5] == shls_slice[7])), "s8 kl");
        ASSERT(((shls_slice[0] == shls_slice[4]) &&
                (shls_slice[1] == shls_slice[5])), "s8 ik");
}

#define iSH     0
#define jSH     1
#define kSH     2
#define lSH     3
#define LOCATE(v, i, j) \
        int d##i##j = d##i * d##j; \
        _poutptr = out->outptr + shls[i##SH]*out->v_ket_nsh+shls[j##SH] - out->offset0_outptr; \
        if (*_poutptr == NOVALUE) { \
                *_poutptr = out->stack_size; \
                out->stack_size += d##i##j * ncomp; \
                memset(out->data+*_poutptr, 0, sizeof(double)*d##i##j*ncomp); \
        } \
        double *v = out->data + *_poutptr;
#define DECLARE(v, i, j) \
        int ncomp = out->ncomp; \
        int ncol = out->dm_dims[1]; \
        int d##i = i##1 - i##0; \
        int d##j = j##1 - j##0; \
        int *_poutptr; \
        LOCATE(v, i, j)

/* eri in Fortran order; dm, out in C order */

static void nrs1_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, k, l);
        int i, j, k, l, ij, icomp;
        double tdm[MAXCGTO*MAXCGTO];

        for (ij = 0, j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        tdm[ij] = dm[j*ncol+i];
                }
        }
        int dij = ij;

        for (icomp = 0; icomp < ncomp; icomp++) {
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        for (ij = 0; ij < dij; ij++) {
                                v[k*dl+l] += eri[ij] * tdm[ij];
                        }
                        eri += dij;
                } }
                v += dkl;
        }
}
ADD_JKOP(nrs1_ji_s1kl, J, I, K, L, s1);

static void nrs1_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (k0 >= l0) {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_ji_s2kl, J, I, K, L, s1);


static void nrs1_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, i, j);
        int i, j, k, l, ij, icomp;
        double buf[MAXCGTO*MAXCGTO];

        for (icomp = 0; icomp < ncomp; icomp++) {

                for (i = 0; i < dij; i++) { buf[i] = 0; }
                for (l = l0; l < l1; l++) {
                for (k = k0; k < k1; k++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[ij] += eri[ij] * dm[l*ncol+k];
                        }
                        eri += dij;
                } }

                for (ij = 0, j = 0; j < dj; j++) {
                for (i = 0; i < di; i++, ij++) {
                        v[i*dj+j] += buf[ij];
                } }
                v += dij;
        }
}
ADD_JKOP(nrs1_lk_s1ij, L, K, I, J, s1);

static void nrs1_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 >= j0) {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_lk_s2ij, L, K, I, J, s1);


static void nrs1_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, i, l);
        int i, j, k, l, ijkl, icomp;

        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                for (l = 0; l < dl; l++) {
                for (k = k0; k < k1; k++) {
                for (j = j0; j < j1; j++) {
                for (i = 0; i < di; i++, ijkl++) {
                        v[i*dl+l] += eri[ijkl] * dm[j*ncol+k];
                } } } }
                v += dil;
        }
}
ADD_JKOP(nrs1_jk_s1il, J, K, I, L, s1);

static void nrs1_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 >= l0) {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_jk_s2il, J, K, I, L, s1);

static void nrs1_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, k, j);
        int i, j, k, l, ijkl, icomp;

        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                for (l = l0; l < l1; l++) {
                for (k = 0; k < dk; k++) {
                for (j = 0; j < dj; j++) {
                for (i = i0; i < i1; i++, ijkl++) {
                        v[k*dj+j] += eri[ijkl] * dm[l*ncol+i];
                } } } }
                v += dkj;
        }
}
ADD_JKOP(nrs1_li_s1kj, L, I, K, J, s1);

static void nrs1_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (k0 >= j0) {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_li_s2kj, L, I, K, J, s1);


static void nrs1_jl_s1ik(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, i, k);
        int i, j, k, l, ijkl, icomp;

        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                for (l = l0; l < l1; l++) {
                for (k = 0; k < dk; k++) {
                for (j = j0; j < j1; j++) {
                for (i = 0; i < di; i++, ijkl++) {
                        v[i*dk+k] += eri[ijkl] * dm[j*ncol+l];
                } } } }
                v += dik;
        }
}
ADD_JKOP(nrs1_jl_s1ik, J, L, I, K, s1);

static void nrs1_lj_s1ki(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, k, i);
        int i, j, k, l, ijkl, icomp;

        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                for (l = l0; l < l1; l++) {
                for (k = 0; k < dk; k++) {
                for (j = j0; j < j1; j++) {
                for (i = 0; i < di; i++, ijkl++) {
                        v[k*di+i] += eri[ijkl] * dm[l*ncol+j];
                } } } }
                v += dki;
        }
}
ADD_JKOP(nrs1_lj_s1ki, L, J, K, I, s1);

static void nrs1_ik_s1jl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, j, l);
        int i, j, k, l, ijkl, icomp;

        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                for (l = 0; l < dl; l++) {
                for (k = k0; k < k1; k++) {
                for (j = 0; j < dj; j++) {
                for (i = i0; i < i1; i++, ijkl++) {
                        v[j*dl+l] += eri[ijkl] * dm[i*ncol+k];
                } } } }
                v += djl;
        }
}
ADD_JKOP(nrs1_ik_s1jl, I, K, J, L, s1);

static void nrs1_ki_s1lj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, l, j);
        int i, j, k, l, ijkl, icomp;

        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                for (l = 0; l < dl; l++) {
                for (k = k0; k < k1; k++) {
                for (j = 0; j < dj; j++) {
                for (i = i0; i < i1; i++, ijkl++) {
                        v[l*dj+j] += eri[ijkl] * dm[k*ncol+i];
                } } } }
                v += dlj;
        }
}
ADD_JKOP(nrs1_ki_s1lj, K, I, L, J, s1);

static void nrs2ij_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(v, k, l);
                int i, j, k, l, ij, icomp;
                double tdm[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        tdm[ij] = dm[i*ncol+j] + dm[j*ncol+i];
                } }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                for (ij = 0; ij < dij; ij++) {
                                        v[k*dl+l] += eri[ij] * tdm[ij];
                                }
                                eri += dij;
                        } }
                        v += dkl;
                }
        } else {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_ji_s1kl, J, I, K, L, s2ij);

static void nrs2ij_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= l0) {
                nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_ji_s2kl, J, I, K, L, s2ij);


static void nrs2ij_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * dm[l*ncol+k];
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_lk_s1ij, L, K, I, J, s2ij);

static void nrs2ij_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs2ij_lk_s2ij, L, K, I, J, s2ij);


static void nrs2ij_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vil, i, l);
                int dj = j1 - j0;
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = k0; k < k1; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vil[i*dl+l] += eri[ijkl]*dm[jp*ncol+k];
                                vjl[j*dl+l] += eri[ijkl]*dm[ip*ncol+k];
                        } } } }
                        vil += dil;
                        vjl += djl;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_jk_s1il, J, K, I, L, s2ij);

static void nrs2ij_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (j0 >= l0) {
                nrs2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_jk_s2il, J, K, I, L, s2ij);


static void nrs2ij_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vkj, k, j);
                int di = i1 - i0;
                LOCATE(vki, k, i);
                int i, j, k, l, ip, jp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = l0; l < l1; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[l*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[l*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_li_s1kj, L, I, K, J, s2ij);

static void nrs2ij_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= i0) {
                nrs2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_li_s2kj, L, I, K, J, s2ij);


static void nrs2kl_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, ij, kl, icomp;
                double tdm[MAXCGTO*MAXCGTO];
                double buf[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        tdm[ij] = dm[j*ncol+i];
                } }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dkl; i++) { buf[i] = 0; }
                        for (kl = 0; kl < dkl; kl++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[kl] += eri[kl*dij+ij] * tdm[ij];
                        } }

                        for (kl = 0, l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++, kl++) {
                                vkl[k*dl+l] += buf[kl];
                                vlk[kl] += buf[kl];
                        } }
                        eri += dij * dkl;
                        vkl += dkl;
                        vlk += dkl;
                }
        } else {
                return nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_ji_s1kl, J, I, K, L, s2kl);

static void nrs2kl_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs2kl_ji_s2kl, J, I, K, L, s2kl);


static void nrs2kl_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(v, i, j);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                tdm = dm[k*ncol+l] + dm[l*ncol+k];
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                v[i*dj+j] += buf[ij];
                        } }
                        v += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_lk_s1ij, L, K, I, J, s2kl);

static void nrs2kl_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= j0) {
                nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_lk_s2ij, L, K, I, J, s2kl);


static void nrs2kl_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vil, i, l);
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                int i, j, k, l, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = j0; j < j1; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dm[j*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[j*ncol+lp];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_jk_s1il, J, K, I, L, s2kl);

static void nrs2kl_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= k0) {
                nrs2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_jk_s2il, J, K, I, L, s2kl);


static void nrs2kl_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkj, k, j);
                int dl = l1 - l0;
                LOCATE(vlj, l, j);
                int i, j, k, l, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) {
                        for (i = i0; i < i1; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+i];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+i];
                        } } } }
                        vkj += dkj;
                        vlj += dlj;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_li_s1kj, L, I, K, J, s2kl);

static void nrs2kl_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (l0 >= j0) {
                nrs2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_li_s2kj, L, I, K, J, s2kl);


static void nrs4_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, ij, kl, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, ij++) {
                                tdm[ij] = dm[i*ncol+j] + dm[j*ncol+i];
                        }
                }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = 0; i < dkl; i++) { buf[i] = 0; }
                        for (kl = 0; kl < dkl; kl++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[kl] += eri[kl*dij+ij] * tdm[ij];
                        } }

                        for (kl = 0, l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++, kl++) {
                                vkl[k*dl+l] += buf[kl];
                                vlk[kl] += buf[kl];
                        } }
                        eri += dij * dkl;
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nrs4_ji_s1kl, J, I, K, L, s4);

static void nrs4_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs4_ji_s2kl, J, I, K, L, s4);


static void nrs4_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                tdm = dm[l*ncol+k] + dm[k*ncol+l];
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nrs4_lk_s1ij, L, K, I, J, s4);

static void nrs4_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs4_lk_s2ij, L, K, I, J, s4);

static void nrs4_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nrs4_jk_s1il, J, K, I, L, s4);

static void nrs4_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dm[j*ncol+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        int dj = j1 - j0;
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vjl[j*dl+l] += eri[ijkl] *dm[ip*ncol+k];
                                        vil[i*dl+l] += eri[ijkl] *dm[jp*ncol+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                int i, j, k, l, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = j0; j < j1; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dm[j*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[j*ncol+lp];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nrs4_jk_s2il, J, K, I, L, s4);


static void nrs4_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] += eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nrs4_li_s1kj, L, I, K, J, s4);

static void nrs4_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = l0; l < l1; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dm[l*ncol+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        int dl = l1 - l0;
                        LOCATE(vlj, l, j);
                        int i, j, k, l, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] *dm[lp*ncol+i];
                                        vlj[l*dj+j] += eri[ijkl] *dm[kp*ncol+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                LOCATE(vkj, k, j);
                int i, j, k, l, ip, jp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = l0; l < l1; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[l*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[l*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] += eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nrs4_li_s2kj, L, I, K, J, s4);


static void nrs8_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0 && j0 == l0) {
                nrs4_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 == j0 || k0 == l0) {
                nrs4_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                int dk = k1 - k0;
                int dl = l1 - l0;
                LOCATE(vji, j, i);
                LOCATE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, kp, lp, ij, icomp;
                double tdm[MAXCGTO*MAXCGTO];
                double buf[MAXCGTO*MAXCGTO];
                double tdm2, tmp;

                for (ij = 0, j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        tdm[ij] = dm[i*ncol+j] + dm[j*ncol+i];
                } }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                                tmp = 0;
                                tdm2 = dm[kp*ncol+lp] + dm[lp*ncol+kp];
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                        buf[ij] += eri[ij] * tdm2;
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] += tmp;
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dji;
                        vkl += dkl;
                        vlk += dlk;
                }
        }
}
ADD_JKOP(nrs8_ji_s1kl, J, I, K, L, s8);

static void nrs8_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0 && j0 == l0) {
                nrs4_ji_s2kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 == j0 || k0 == l0) {
                nrs4_ji_s2kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_lk_s2ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                int dk = k1 - k0;
                int dl = l1 - l0;
                LOCATE(vkl, k, l);
                int i, j, k, l, kp, lp, ij, icomp;
                double tdm[MAXCGTO*MAXCGTO];
                double buf[MAXCGTO*MAXCGTO];
                double tdm2;

                for (ij = 0, j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        tdm[ij] = dm[i*ncol+j] + dm[j*ncol+i];
                } }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                                tdm2 = dm[kp*ncol+lp] + dm[lp*ncol+kp];
                                for (ij = 0; ij < dij; ij++) {
                                        vkl[k*dl+l] += eri[ij] * tdm[ij];
                                        buf[ij] += eri[ij] * tdm2;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                        } }
                        vij += dij;
                        vkl += dkl;
                }
        }
}
ADD_JKOP(nrs8_ji_s2kl, J, I, K, L, s8);


static void nrs8_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_lk_s1ij, L, K, I, J, s8);

static void nrs8_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_lk_s2ij, L, K, I, J, s8);


static void nrs8_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0 && j0 == l0) {
                nrs4_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 == j0 || k0 == l0) { // i0==l0 => i0==k0==l0
                nrs4_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkj, k, j);
                int di = i1 - i0;
                int dl = l1 - l0;
                LOCATE(vki, k, i);
                LOCATE(vlj, l, j);
                LOCATE(vli, l, i);
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] += eri[ijkl] * dm[kp*ncol+jp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                        vik += dik;
                        vil += dil;
                        vjk += djk;
                        vjl += djl;
                }
        }
}
ADD_JKOP(nrs8_li_s1kj, L, I, K, J, s8);

static void nrs8_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0) {
                nrs4_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                if (j0 != l0) {
                        nrs4_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                }
        } else if (i0 == j0 || k0 == l0) { // i0==l0 => i0==k0==l0
                nrs4_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                if (j0 < l0) { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        int di = i1 - i0;
                        int dl = l1 - l0;
                        LOCATE(vlj, l, j);
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                        vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                        vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                        vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                                vik += dik;
                                vil += dil;
                        }
                } else if (j0 == l0) { // j == l < k < i
                        DECLARE(vkj, k, j);
                        int di = i1 - i0;
                        int dl = l1 - l0;
                        LOCATE(vlj, l, j);
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                        vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                        vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                        vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                        vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                                vik += dik;
                                vil += dil;
                                vjl += djl;
                        }
                } else if (j0 < k0) { // l < j < k < i
                        DECLARE(vkj, k, j);
                        int di = i1 - i0;
                        int dl = l1 - l0;
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                        vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                        vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                        vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                } } } }
                                vkj += dkj;
                                vik += dik;
                                vil += dil;
                                vjl += djl;
                        }
                } else if (j0 == k0) { // l < j == k < i
                        DECLARE(vkj, k, j);
                        int di = i1 - i0;
                        int dl = l1 - l0;
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjk, j, k);
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                        vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                        vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                        vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                        vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                } } } }
                                vkj += dkj;
                                vjk += djk;
                                vik += dik;
                                vil += dil;
                                vjl += djl;
                        }
                } else { // l < k < j < i
                        DECLARE(vik, i, k);
                        int dj = j1 - j0;
                        int dl = l1 - l0;
                        LOCATE(vil, i, l);
                        LOCATE(vjk, j, k);
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                        vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                        vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                        vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                } } } }
                                vik += dik;
                                vil += dil;
                                vjk += djk;
                                vjl += djl;
                        }
                }
        }
}
ADD_JKOP(nrs8_li_s2kj, L, I, K, J, s8);


static void nrs8_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_jk_s1il, J, K, I, L, s8);

static void nrs8_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_jk_s2il, J, K, I, L, s8);


/*************************************************
 * For anti symmetrized integrals
 *************************************************/
static void nra2ij_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(v, k, l);
                int i, j, k, l, ij, icomp;
                double tdm[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, ij++) {
                                tdm[ij] = dm[j*ncol+i] - dm[i*ncol+j];
                        }
                }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                for (ij = 0; ij < dij; ij++) {
                                        v[k*dl+l] += eri[ij] * tdm[ij];
                                }
                                eri += dij;
                        } }
                        v += dkl;
                }
        } else {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_ji_s1kl, J, I, K, L, s2ij);

static void nra2ij_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= l0) {
                nra2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_ji_s2kl, J, I, K, L, s2ij);

static void nra2ij_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * dm[l*ncol+k];
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] -= buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_lk_s1ij, L, K, I, J, s2ij);

static void nra2ij_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2ij_lk_s2ij, L, K, I, J, s2ij);

static void nra2ij_lk_a2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2ij_lk_a2ij, L, K, I, J, s2ij);

static void nra2ij_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vil, i, l);
                int dj = j1 - j0;
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = k0; k < k1; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+k];
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+k];
                        } } } }
                        vil += dil;
                        vjl += djl;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_jk_s1il, J, K, I, L, s2ij);

static void nra2ij_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (j0 >= l0) {
                nra2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_jk_s2il, J, K, I, L, s2ij);

static void nra2ij_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vkj, k, j);
                int di = i1 - i0;
                LOCATE(vki, k, i);
                int i, j, k, l, ip, jp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = l0; l < l1; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[l*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[l*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_li_s1kj, L, I, K, J, s2ij);

static void nra2ij_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= i0) {
                nra2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_li_s2kj, L, I, K, J, s2ij);

static void nra2kl_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, ij, kl, icomp;
                double tdm[MAXCGTO*MAXCGTO];
                double buf[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, ij++) {
                                tdm[ij] = dm[j*ncol+i];
                        }
                }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dkl; i++) { buf[i] = 0; }
                        for (kl = 0; kl < dkl; kl++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[kl] += eri[kl*dij+ij] * tdm[ij];
                        } }

                        for (kl = 0, l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++, kl++) {
                                vkl[k*dl+l] += buf[kl];
                                vlk[kl] -= buf[kl];
                        } }
                        eri += dij * dkl;
                        vkl += dkl;
                        vlk += dkl;
                }
        } else {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_ji_s1kl, J, I, K, L, s2kl);

static void nra2kl_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2kl_ji_s2kl, J, I, K, L, s2kl);

static void nra2kl_ji_a2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2kl_ji_a2kl, J, I, K, L, s2kl);

static void nra2kl_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(v, i, j);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                tdm = dm[l*ncol+k] - dm[k*ncol+l];
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                v[i*dj+j] += buf[ij];
                        } }
                        v += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_lk_s1ij, L, K, I, J, s2kl);

static void nra2kl_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= j0) {
                nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_lk_s2ij, L, K, I, J, s2kl);

static void nra2kl_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vil, i, l);
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                int i, j, k, l, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = j0; j < j1; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dm[j*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[j*ncol+lp];
                        } } } }
                        vik += dik;
                        vil += dil;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_jk_s1il, J, K, I, L, s2kl);

static void nra2kl_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= k0) {
                nra2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_jk_s2il, J, K, I, L, s2kl);

static void nra2kl_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkj, k, j);
                int dl = l1 - l0;
                LOCATE(vlj, l, j);
                int i, j, k, l, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) {
                        for (i = i0; i < i1; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+i];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+i];
                        } } } }
                        vkj += dkj;
                        vlj += dlj;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_li_s1kj, L, I, K, J, s2kl);

static void nra2kl_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (l0 >= j0) {
                nra2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_li_s2kj, L, I, K, J, s2kl);

static void nra4ij_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, ij, kl, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, ij++) {
                                tdm[ij] = dm[j*ncol+i] - dm[i*ncol+j];
                        }
                }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dkl; i++) { buf[i] = 0; }
                        for (kl = 0; kl < dkl; kl++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[kl] += eri[kl*dij+ij] * tdm[ij];
                        } }

                        for (kl = 0, l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++, kl++) {
                                vkl[k*dl+l] += buf[kl];
                                vlk[kl] += buf[kl];
                        } }
                        eri += dij * dkl;
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nra4ij_ji_s1kl, J, I, K, L, s4);

static void nra4ij_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4ij_ji_s2kl, J, I, K, L, s4);

static void nra4ij_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                tdm = dm[l*ncol+k] + dm[k*ncol+l];
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] -= buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nra4ij_lk_s1ij, L, K, I, J, s4);

static void nra4ij_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4ij_lk_s2ij, L, K, I, J, s4);

static void nra4ij_lk_a2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4ij_lk_a2ij, L, K, I, J, s4);

static void nra4ij_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] -= eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4ij_jk_s1il, J, K, I, L, s4);

static void nra4ij_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dm[j*ncol+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        int dj = j1 - j0;
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vjl[j*dl+l] -= eri[ijkl] *dm[ip*ncol+k];
                                        vil[i*dl+l] += eri[ijkl] *dm[jp*ncol+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                int i, j, k, l, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = j0; j < j1; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dm[j*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[j*ncol+lp];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+kp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] -= eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] += eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4ij_jk_s2il, J, K, I, L, s4);

static void nra4ij_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] -= eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4ij_li_s1kj, L, I, K, J, s4);

static void nra4ij_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = l0; l < l1; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dm[l*ncol+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        int dl = l1 - l0;
                        LOCATE(vlj, l, j);
                        int i, j, k, l, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] *dm[lp*ncol+i];
                                        vlj[l*dj+j] += eri[ijkl] *dm[kp*ncol+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                LOCATE(vkj, k, j);
                int i, j, k, l, ip, jp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = l0; l < l1; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[l*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[l*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] += eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] -= eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4ij_li_s2kj, L, I, K, J, s4);

static void nra4kl_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, ij, kl, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, ij++) {
                                tdm[ij] = dm[i*ncol+j] + dm[j*ncol+i];
                        }
                }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dkl; i++) { buf[i] = 0; }
                        for (kl = 0; kl < dkl; kl++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[kl] += eri[kl*dij+ij] * tdm[ij];
                        } }

                        for (kl = 0, l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++, kl++) {
                                vkl[k*dl+l] += buf[kl];
                                vlk[kl] -= buf[kl];
                        } }
                        eri += dij * dkl;
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nra4kl_ji_s1kl, J, I, K, L, s4);

static void nra4kl_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4kl_ji_s2kl, J, I, K, L, s4);

static void nra4kl_ji_a2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4kl_ji_a2kl, J, I, K, L, s4);

static void nra4kl_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                tdm = dm[l*ncol+k] - dm[k*ncol+l];
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nra4kl_lk_s1ij, L, K, I, J, s4);

static void nra4kl_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4kl_lk_s2ij, L, K, I, J, s4);

static void nra4kl_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] -= eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4kl_jk_s1il, J, K, I, L, s4);

static void nra4kl_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dm[j*ncol+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        int dj = j1 - j0;
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vjl[j*dl+l] += eri[ijkl] *dm[ip*ncol+k];
                                        vil[i*dl+l] += eri[ijkl] *dm[jp*ncol+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                int i, j, k, l, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = j0; j < j1; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dm[j*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[j*ncol+lp];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[jp*ncol+lp];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] -= eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] += eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4kl_jk_s2il, J, K, I, L, s4);

static void nra4kl_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] -= eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4kl_li_s1kj, L, I, K, J, s4);

static void nra4kl_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = l0; l < l1; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dm[l*ncol+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        int dl = l1 - l0;
                        LOCATE(vlj, l, j);
                        int i, j, k, l, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] *dm[lp*ncol+i];
                                        vlj[l*dj+j] -= eri[ijkl] *dm[kp*ncol+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                LOCATE(vkj, k, j);
                int i, j, k, l, ip, jp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = l0; l < l1; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[l*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[l*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+ip];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] += eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] -= eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4kl_li_s2kj, L, I, K, J, s4);

/*
 * aa4: 4-fold permutation symmetry with anti-symm for ij and anti-symm for kl
 */
static void nraa4_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int i, j, k, l, ij, kl, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm[MAXCGTO*MAXCGTO];

                for (ij = 0, j = j0; j < j1; j++) {
                        for (i = i0; i < i1; i++, ij++) {
                                tdm[ij] = dm[j*ncol+i] - dm[i*ncol+j];
                        }
                }
                int dij = ij;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dkl; i++) { buf[i] = 0; }
                        for (kl = 0; kl < dkl; kl++) {
                        for (ij = 0; ij < dij; ij++) {
                                buf[kl] += eri[kl*dij+ij] * tdm[ij];
                        } }

                        for (kl = 0, l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++, kl++) {
                                vkl[k*dl+l] += buf[kl];
                                vlk[kl] -= buf[kl];
                        } }
                        eri += dij * dkl;
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nraa4_ji_s1kl, J, I, K, L, s4);

static void nraa4_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nraa4_ji_s2kl, J, I, K, L, s4);

static void nraa4_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                int i, j, k, l, ij, icomp;
                double buf[MAXCGTO*MAXCGTO];
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = l0; l < l1; l++) {
                        for (k = k0; k < k1; k++) {
                                tdm = dm[l*ncol+k] - dm[k*ncol+l];
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] -= buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nraa4_lk_s1ij, L, K, I, J, s4);

static void nraa4_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nraa4_lk_s2ij, L, K, I, J, s4);

static void nraa4_lk_a2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nraa4_lk_a2ij, L, K, I, J, s4);

static void nraa4_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nraa4_jk_s1il, J, K, I, L, s4);

static void nraa4_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = j0; j < j1; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dm[j*ncol+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        int dj = j1 - j0;
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ip, jp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = k0; k < k1; k++) {
                                for (j = 0; j < dj; j++) { jp = j0 + j;
                                for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                        vjl[j*dl+l] -= eri[ijkl] *dm[ip*ncol+k];
                                        vil[i*dl+l] += eri[ijkl] *dm[jp*ncol+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                int i, j, k, l, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = j0; j < j1; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dm[j*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[j*ncol+lp];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+kp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[jp*ncol+lp];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                int di = i1 - i0;
                int dk = k1 - k0;
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vjk[j*dk+k] += eri[ijkl] * dm[ip*ncol+lp];
                                vjl[j*dl+l] -= eri[ijkl] * dm[ip*ncol+kp];
                                vik[i*dk+k] -= eri[ijkl] * dm[jp*ncol+lp];
                                vil[i*dl+l] += eri[ijkl] * dm[jp*ncol+kp];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nraa4_jk_s2il, J, K, I, L, s4);

static void nraa4_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] += eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nraa4_li_s1kj, L, I, K, J, s4);

static void nraa4_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = l0; l < l1; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dm[l*ncol+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        int dl = l1 - l0;
                        LOCATE(vlj, l, j);
                        int i, j, k, l, kp, lp, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) { lp = l0 + l;
                                for (k = 0; k < dk; k++) { kp = k0 + k;
                                for (j = 0; j < dj; j++) {
                                for (i = i0; i < i1; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] *dm[lp*ncol+i];
                                        vlj[l*dj+j] -= eri[ijkl] *dm[kp*ncol+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                LOCATE(vkj, k, j);
                int i, j, k, l, ip, jp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = l0; l < l1; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[l*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[l*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+ip];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                int dj = j1 - j0;
                int dl = l1 - l0;
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ip, jp, kp, lp, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) { lp = l0 + l;
                        for (k = 0; k < dk; k++) { kp = k0 + k;
                        for (j = 0; j < dj; j++) { jp = j0 + j;
                        for (i = 0; i < di; i++, ijkl++) { ip = i0 + i;
                                vkj[k*dj+j] += eri[ijkl] * dm[lp*ncol+ip];
                                vki[k*di+i] -= eri[ijkl] * dm[lp*ncol+jp];
                                vlj[l*dj+j] -= eri[ijkl] * dm[kp*ncol+ip];
                                vli[l*di+i] += eri[ijkl] * dm[kp*ncol+jp];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nraa4_li_s2kj, L, I, K, J, s4);

