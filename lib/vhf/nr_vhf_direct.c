/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "cint.h"
#include "cvhf.h"
#include "misc.h"
#include "optimizer.h"
#include "nr_vhf_incore.h"

#define LOWERTRI_INDEX(I,J)     ((I) > (J) ? ((I)*((I)+1)/2+(J)) : ((J)*((J)+1)/2+(I)))
#define MAX(I,J)        ((I) > (J) ? (I) : (J))

/*
 * reorder the blocks, to a lower triangle sequence
 * index of [ [.] [..] [...] ] =>
 * [.
 *  ..
 *  ...]
 */
void CVHFindex_blocks2tri(int *idx, int *ao_loc,
                          const int *bas, const int nbas)
{
        int ish, jsh, i, j, i0, j0, ij;
        int di, dj;
        int off = 0;
        for (ish = 0; ish < nbas; ish++)
        for (jsh = 0; jsh <= ish; jsh++) {
                di = CINTcgto_spheric(ish, bas);
                dj = CINTcgto_spheric(jsh, bas);
                for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++)
                for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                        if (i0 >= j0) {
                                ij = LOWERTRI_INDEX(i0, j0);
                                idx[ij] = off+di*j+i;
                        }
                }
                off += di*dj;
        }
}

int CVHFnr_vhf_prescreen(int *shls, CVHFOpt *opt)
{
        if (!opt) {
                return 1; // no screen
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int l = shls[3];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < n);
        assert(l < n);
        double qijkl = opt->q_cond[i*n+j] * opt->q_cond[k*n+l];
        double dm_max =     4*opt->dm_cond[j*n+i];
        dm_max = MAX(dm_max,4*opt->dm_cond[l*n+k]);
        dm_max = MAX(dm_max,  opt->dm_cond[j*n+k]);
        dm_max = MAX(dm_max,  opt->dm_cond[j*n+l]);
        dm_max = MAX(dm_max,  opt->dm_cond[i*n+k]);
        dm_max = MAX(dm_max,  opt->dm_cond[i*n+l]);
        return dm_max*qijkl > opt->direct_scf_cutoff;
}

int CVHFfill_nr_eri_o2(double *eri, int ish, int jsh, int ksh_lim,
                       const int *atm, const int natm,
                       const int *bas, const int nbas, const double *env,
                       CINTOpt *opt, CVHFOpt *vhfopt)
{
        const int di = CINTcgto_spheric(ish, bas);
        const int dj = CINTcgto_spheric(jsh, bas);
        int ksh, lsh, dk, dl;
        int shls[4];
        double *buf = eri;
        int empty = 1;
        for (ksh = 0; ksh < ksh_lim; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                dk = CINTcgto_spheric(ksh, bas);
                dl = CINTcgto_spheric(lsh, bas);
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (!vhfopt ||
                    (*vhfopt->fprescreen)(shls, vhfopt)) {
                        empty = !cint2e_sph(buf, shls, atm, natm, bas, nbas, env, opt)
                                && empty;
                } else {
                        memset(buf, 0, sizeof(double)*di*dj*dk*dl);
                }
                buf += di*dj*dk*dl;
        } }
        return !empty;
}



/*************************************************
 * dm has nset components
 *************************************************/
static void CVHFnr8fold_jk_o0(int nset, double *vj, double *vk, double *tri_dm, double *dm,
                              double *eri, int ish, int jsh, int *ao_loc,
                              int *idx_tri, const int *bas, const int nbas)
{
        const int nao = ao_loc[nbas-1] + CINTcgto_spheric(nbas-1,bas);
        const int nao2 = nao*nao;
        const int npair = nao*(nao+1)/2;
        const int di = CINTcgto_spheric(ish, bas);
        const int dj = CINTcgto_spheric(jsh, bas);
        double *eri1 = malloc(sizeof(double)*nao*nao);
        int i, j, i0, j0, ij, kl, ij0;
        int last_ij;
        int k, iset;

        k = ao_loc[ish] + CINTcgto_spheric(ish, bas);
        last_ij = k*(k+1)/2;

        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++)
        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                if (i0 >= j0) {
                        ij = j * di + i;
                        for (kl = 0; kl < last_ij; kl++) {
                                eri1[kl] = eri[idx_tri[kl]*di*dj+ij];
                        }
                        for (iset = 0; iset < nset; iset++) {
                                ij0 = LOWERTRI_INDEX(i0, j0);
                                CVHFnr_eri8fold_vj_o2(vj+npair*iset, ij0, eri1,
                                                      tri_dm+npair*iset);
                                CVHFnr_eri8fold_vk_o0(vk+nao2*iset, i0, j0, nao,
                                                      eri1, dm+nao2*iset);
                        }
                }
        }
        free(eri1);
}

void CVHFnr8fold_jk_o3(int nset, double *vj, double *vk, double *tri_dm, double *dm,
                       double *eri, int ish, int jsh, int *ao_loc,
                       int *idx_tri, const int *bas, const int nbas)
{
        const int nao = ao_loc[nbas-1] + CINTcgto_spheric(nbas-1,bas);
        const int nao2 = nao*nao;
        const int npair = nao*(nao+1)/2;
        const int di = CINTcgto_spheric(ish, bas);
        const int dj = CINTcgto_spheric(jsh, bas);
        double *eri1 = malloc(sizeof(double)*nao*nao*4);
        double *eri2 = eri1 + nao*nao;
        double *eri3 = eri2 + nao*nao;
        double *eri4 = eri3 + nao*nao;
        double *peri;
        int *idxij = malloc(sizeof(int)*di*dj);
        int *idxi0 = malloc(sizeof(int)*di*dj);
        int *idxj0 = malloc(sizeof(int)*di*dj);
        int i, j, i0, j0, ij, kl, ij0, ij1, ij2, ij3, ij4;
        int off, last_ij;
        int k, lenij, iset;

        lenij = 0;
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++)
        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
                if (i0 >= j0) {
                        idxi0[lenij] = i0;
                        idxj0[lenij] = j0;
                        idxij[lenij] = j * di + i;
                        lenij++;
                }
        }

        k = ao_loc[ish] + CINTcgto_spheric(ish, bas);
        last_ij = k*(k+1)/2;

        for (k = 0; k < lenij-3; k += 4) {
                ij1 = idxij[k  ];
                ij2 = idxij[k+1];
                ij3 = idxij[k+2];
                ij4 = idxij[k+3];
                for (kl = 0; kl < last_ij; kl++) {
                        off = idx_tri[kl]*di*dj;
                        eri1[kl] = eri[off+ij1];
                        eri2[kl] = eri[off+ij2];
                        eri3[kl] = eri[off+ij3];
                        eri4[kl] = eri[off+ij4];
                }
                for (iset = 0; iset < nset; iset++) {
                for (i = 0, peri = eri1; i < 4; i++, peri+=nao*nao) {
                        i0 = idxi0[k+i];
                        j0 = idxj0[k+i];
                        ij0 = LOWERTRI_INDEX(i0, j0);
                        CVHFnr_eri8fold_vj_o2(vj+npair*iset, ij0, peri,
                                              tri_dm+npair*iset);
                        CVHFnr_eri8fold_vk_o4(vk+nao2*iset, i0, j0, nao, peri,
                                              dm+nao2*iset);
                } }
        }
        for (; k < lenij; k++) {
                ij = idxij[k];
                for (kl = 0; kl < last_ij; kl++) {
                        eri1[kl] = eri[idx_tri[kl]*di*dj+ij];
                }
                for (iset = 0; iset < nset; iset++) {
                        i0 = idxi0[k];
                        j0 = idxj0[k];
                        ij0 = LOWERTRI_INDEX(i0, j0);
                        CVHFnr_eri8fold_vj_o2(vj+npair*iset, ij0, eri1,
                                              tri_dm+npair*iset);
                        CVHFnr_eri8fold_vk_o4(vk+nao2*iset, i0, j0, nao, eri1,
                                              dm+nao2*iset);
                }
        }
        free(idxi0);
        free(idxj0);
        free(idxij);
        free(eri1);
}

/**************************************************
 * for symmetric density matrix
 **************************************************/
void CVHFnr_direct_o4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        const int npair = nao*(nao+1)/2;
        double *tri_dm = malloc(sizeof(double)*npair*nset);
        double *tri_vj = malloc(sizeof(double)*npair*nset);
        double *vj_priv, *vk_priv;
        int i, j, ij;
        int *ij2i = malloc(sizeof(int)*nbas*nbas);
        int *idx_tri = malloc(sizeof(int)*nao*nao);
        int *ao_loc = malloc(sizeof(int)*nbas);
        int di, dj;
        double *eribuf;

        for (i = 0; i < nset; i++) {
                CVHFcompress_nr_dm(tri_dm+npair*i, dm+nao*nao*i, nao);
        }
        CVHFset_ij2i(ij2i, nbas);
        memset(tri_vj, 0, sizeof(double)*npair*nset);
        memset(vk, 0, sizeof(double)*nao*nao*nset);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        CVHFindex_blocks2tri(idx_tri, ao_loc, bas, nbas);

        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        if (vhfopt) {
                CVHFset_direct_scf_dm(vhfopt, dm, nset, atm, natm, bas, nbas, env);
        }

#pragma omp parallel default(none) \
        shared(tri_dm, dm, tri_vj, vk, ij2i, ao_loc, idx_tri, \
               atm, bas, env, opt, vhfopt) \
        private(ij, i, j, di, dj, vj_priv, vk_priv, eribuf)
        {
                vj_priv = malloc(sizeof(double)*npair*nset);
                vk_priv = malloc(sizeof(double)*nao*nao*nset);
                memset(vj_priv, 0, sizeof(double)*npair*nset);
                memset(vk_priv, 0, sizeof(double)*nao*nao*nset);
#pragma omp for nowait schedule(guided)
                for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                        i = ij2i[ij];
                        j = ij - (i*(i+1)/2);
                        di = CINTcgto_spheric(i, bas);
                        dj = CINTcgto_spheric(j, bas);
                        eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);
                        if (CVHFfill_nr_eri_o2(eribuf, i, j, i+1,
                                               atm, natm, bas, nbas, env,
                                               opt, vhfopt)) {
                                CVHFnr8fold_jk_o3(nset, vj_priv, vk_priv,
                                                  tri_dm, dm, eribuf,
                                                  i, j, ao_loc, idx_tri, bas, nbas);
                        }
                        free(eribuf);
                }
#pragma omp critical
                {
                        for (i = 0; i < npair*nset; i++) {
                                tri_vj[i] += vj_priv[i];
                        }
                        for (i = 0; i < nao*nao*nset; i++) {
                                vk[i] += vk_priv[i];
                        }
                }
                free(vj_priv);
                free(vk_priv);
        }

        int iset;
        double *pj;
        for (iset = 0; iset < nset; iset++) {
                pj = tri_vj + npair * iset;
                for (i = 0, ij = 0; i < nao; i++) {
                        for (j = 0; j <= i; j++, ij++) {
                                vj[i*nao+j] = pj[ij];
                                vj[j*nao+i] = pj[ij];
                                vk[j*nao+i] = vk[i*nao+j];
                        }
                }
                vj += nao*nao;
                vk += nao*nao;
        }
        CINTdel_optimizer(&opt);
        free(ij2i);
        free(idx_tri);
        free(ao_loc);
        free(tri_dm);
        free(tri_vj);
}


void CVHFnr_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env)
{
        CVHFinit_optimizer(vhfopt, atm, natm, bas, nbas, env);
        (*vhfopt)->fprescreen = &CVHFnr_vhf_prescreen;
        CVHFset_direct_scf(*vhfopt, atm, natm, bas, nbas, env);
}



/**************************************************
 * for general density matrix
 **************************************************/
static void CVHFnr_direct_sub(double *dm, double *vj, double *vk, const int nset,
                              CVHFOpt *vhfopt, void (*const fvk)(),
                              const int *atm, const int natm,
                              const int *bas, const int nbas,
                              const double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        const int npair = nao*(nao+1)/2;
        double *tri_dm = malloc(sizeof(double)*npair*nset);
        double *tri_vj = malloc(sizeof(double)*npair*nset);
        double *vj_priv, *vk_priv;
        int i, j, ij;
        int *ij2i = malloc(sizeof(int)*nbas*nbas);
        int *idx_tri = malloc(sizeof(int)*nao*nao);
        int *ao_loc = malloc(sizeof(int)*nbas);
        int di, dj;
        double *eribuf;

        for (i = 0; i < nset; i++) {
                CVHFcompress_nr_dm(tri_dm+npair*i, dm+nao*nao*i, nao);
        }
        CVHFset_ij2i(ij2i, nbas);
        memset(tri_vj, 0, sizeof(double)*npair*nset);
        memset(vk, 0, sizeof(double)*nao*nao*nset);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        CVHFindex_blocks2tri(idx_tri, ao_loc, bas, nbas);

        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        if (vhfopt) {
                CVHFset_direct_scf_dm(vhfopt, dm, nset, atm, natm, bas, nbas, env);
        }

#pragma omp parallel default(none) \
        shared(tri_dm, dm, tri_vj, vk, ij2i, ao_loc, idx_tri, \
               atm, bas, env, opt, vhfopt) \
        private(ij, i, j, di, dj, vj_priv, vk_priv, eribuf)
        {
                vj_priv = malloc(sizeof(double)*npair*nset);
                vk_priv = malloc(sizeof(double)*nao*nao*nset);
                memset(vj_priv, 0, sizeof(double)*npair*nset);
                memset(vk_priv, 0, sizeof(double)*nao*nao*nset);
#pragma omp for nowait schedule(guided)
                for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                        i = ij2i[ij];
                        j = ij - (i*(i+1)/2);
                        di = CINTcgto_spheric(i, bas);
                        dj = CINTcgto_spheric(j, bas);
                        eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);
                        if (CVHFfill_nr_eri_o2(eribuf, i, j, i+1,
                                               atm, natm, bas, nbas, env,
                                               opt, vhfopt)) {
                                (*fvk)(nset, vj_priv, vk_priv,
                                       tri_dm, dm, eribuf,
                                       i, j, ao_loc, idx_tri, bas, nbas);
                        }
                        free(eribuf);
                }
#pragma omp critical
                {
                        for (i = 0; i < npair*nset; i++) {
                                tri_vj[i] += vj_priv[i];
                        }
                        for (i = 0; i < nao*nao*nset; i++) {
                                vk[i] += vk_priv[i];
                        }
                }
                free(vj_priv);
                free(vk_priv);
        }

        int iset;
        double *pj;
        for (iset = 0; iset < nset; iset++) {
                pj = tri_vj + npair * iset;
                for (i = 0, ij = 0; i < nao; i++) {
                        for (j = 0; j <= i; j++, ij++) {
                                vj[i*nao+j] = pj[ij];
                                vj[j*nao+i] = pj[ij];
                        }
                }
                vj += nao*nao;
                vk += nao*nao;
        }
        CINTdel_optimizer(&opt);
        free(ij2i);
        free(idx_tri);
        free(ao_loc);
        free(tri_dm);
        free(tri_vj);
}

void CVHFnr_direct(double *dm, double *vj, double *vk, const int nset,
                   CVHFOpt *vhfopt, int hermi,
                   const int *atm, const int natm,
                   const int *bas, const int nbas, const double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        int i, j, iset;
        if (hermi == DM_HERMITIAN) {
                CVHFnr_direct_sub(dm, vj, vk, nset, vhfopt, &CVHFnr8fold_jk_o3,
                                  atm, natm, bas, nbas, env);
                for (iset = 0; iset < nset; iset++) {
                        for (i = 0; i < nao; i++) {
                                for (j = 0; j <= i; j++) {
                                        vk[j*nao+i] = vk[i*nao+j];
                                }
                        }
                        vk += nao*nao;
                }
        } else if (hermi == DM_ANTI) {
                CVHFnr_direct_sub(dm, vj, vk, nset, vhfopt, &CVHFnr8fold_jk_o3,
                                  atm, natm, bas, nbas, env);
                for (iset = 0; iset < nset; iset++) {
                        for (i = 0; i < nao; i++) {
                                for (j = 0; j <= i; j++) {
                                        vk[j*nao+i] = -vk[i*nao+j];
                                }
                        }
                        vk += nao*nao;
                }
        } else { // plain
                CVHFnr_direct_sub(dm, vj, vk, nset, vhfopt, &CVHFnr8fold_jk_o0,
                                  atm, natm, bas, nbas, env);
        }
}
