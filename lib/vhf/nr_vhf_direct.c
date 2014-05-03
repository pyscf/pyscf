/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "cint.h"
#include "misc.h"
#include "optimizer.h"
#include "nr_vhf_incore.h"

#define LOWERTRI_INDEX(I,J)     ((I) > (J) ? ((I)*((I)+1)/2+(J)) : ((J)*((J)+1)/2+(I)))
#define MAX(I,J)        ((I) > (J) ? (I) : (J))

/*
 * reorder the blocks, to a lower triangle sequence
 */
void index_blocks2tri(unsigned int *idx, int *ao_loc,
                      const int *bas, const int nbas)
{
        unsigned int ish, jsh, i, j, i0, j0, ij;
        unsigned int di, dj;
        unsigned int off = 0;
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

int CVHFnr_vhf_prescreen(unsigned int *shls, CVHFOpt *opt)
{
        if (!opt) {
                return 1; // not screen
        }
        unsigned int i = shls[0];
        unsigned int j = shls[1];
        unsigned int k = shls[2];
        unsigned int l = shls[3];
        unsigned int n = opt->nbas;
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

int nr8fold_eri_o1(double *eri, unsigned int ish, unsigned int jsh,
                   CVHFOpt *vhfopt, const int *atm, const int natm,
                   const int *bas, const int nbas, const double *env,
                   CINTOpt *opt)
{
        const unsigned int di = CINTcgto_spheric(ish, bas);
        const unsigned int dj = CINTcgto_spheric(jsh, bas);
        unsigned int ksh, lsh, dk, dl;
        unsigned int shls[4];
        double *buf = eri;
        int empty = 1;
        for (ksh = 0; ksh <= ish; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                dk = CINTcgto_spheric(ksh, bas);
                dl = CINTcgto_spheric(lsh, bas);
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                empty = !cint2e_sph(buf, shls, atm, natm, bas, nbas, env, opt)
                        && empty;
                buf += di*dj*dk*dl;
        } }
        return !empty;
}

int nr8fold_eri_o2(double *eri, unsigned int ish, unsigned int jsh,
                   CVHFOpt *vhfopt, const int *atm, const int natm,
                   const int *bas, const int nbas, const double *env,
                   CINTOpt *opt)
{
        const unsigned int di = CINTcgto_spheric(ish, bas);
        const unsigned int dj = CINTcgto_spheric(jsh, bas);
        unsigned int ksh, lsh, dk, dl;
        unsigned int shls[4];
        double *buf = eri;
        int empty = 1;
        for (ksh = 0; ksh <= ish; ksh++) {
        for (lsh = 0; lsh <= ksh; lsh++) {
                dk = CINTcgto_spheric(ksh, bas);
                dl = CINTcgto_spheric(lsh, bas);
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (CVHFnr_vhf_prescreen(shls, vhfopt)) {
                        empty = !cint2e_sph(buf, shls, atm, natm, bas, nbas, env, opt)
                                && empty;
                } else {
                        memset(buf, 0, sizeof(double)*di*dj*dk*dl);
                }
                buf += di*dj*dk*dl;
        } }
        return !empty;
}

void nr8fold_jk_o3(double *vj, double *vk, double *tri_dm, double *dm,
                   double *eri, unsigned int ish, unsigned int jsh, int *ao_loc,
                   unsigned int *idx_tri, const int *bas, const int nbas)
{
        const unsigned int nao = ao_loc[nbas-1] + CINTcgto_spheric(nbas-1,bas);
        const unsigned int di = CINTcgto_spheric(ish, bas);
        const unsigned int dj = CINTcgto_spheric(jsh, bas);
        double *eri1 = malloc(sizeof(double)*nao*nao*4);
        double *eri2 = eri1 + nao*nao;
        double *eri3 = eri2 + nao*nao;
        double *eri4 = eri3 + nao*nao;
        unsigned int *idxij = malloc(sizeof(unsigned int)*di*dj);
        unsigned int *idxi0 = malloc(sizeof(unsigned int)*di*dj);
        unsigned int *idxj0 = malloc(sizeof(unsigned int)*di*dj);
        unsigned int i, j, i0, j0, ij, kl, ij0, ij1, ij2, ij3, ij4;
        unsigned int off, last_kl;
        unsigned int k, lenij;

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
        last_kl = k*(k+1)/2; 

        for (k = 0; k+3 < lenij; k += 4) {
                ij1 = idxij[k  ];
                ij2 = idxij[k+1];
                ij3 = idxij[k+2];
                ij4 = idxij[k+3];
                for (kl = 0; kl < last_kl; kl++) {
                        off = idx_tri[kl]*di*dj;
                        eri1[kl] = eri[off+ij1];
                        eri2[kl] = eri[off+ij2];
                        eri3[kl] = eri[off+ij3];
                        eri4[kl] = eri[off+ij4];
                }
                i0 = idxi0[k];
                j0 = idxj0[k];
                ij0 = LOWERTRI_INDEX(i0, j0);
                nr_eri8fold_vj_o2(vj, ij0, eri1, tri_dm);
                nr_eri8fold_vk_o4(vk, i0, j0, nao, eri1, dm);
                i0 = idxi0[k+1];
                j0 = idxj0[k+1];
                ij0 = LOWERTRI_INDEX(i0, j0);
                nr_eri8fold_vj_o2(vj, ij0, eri2, tri_dm);
                nr_eri8fold_vk_o4(vk, i0, j0, nao, eri2, dm);
                i0 = idxi0[k+2];
                j0 = idxj0[k+2];
                ij0 = LOWERTRI_INDEX(i0, j0);
                nr_eri8fold_vj_o2(vj, ij0, eri3, tri_dm);
                nr_eri8fold_vk_o4(vk, i0, j0, nao, eri3, dm);
                i0 = idxi0[k+3];
                j0 = idxj0[k+3];
                ij0 = LOWERTRI_INDEX(i0, j0);
                nr_eri8fold_vj_o2(vj, ij0, eri4, tri_dm);
                nr_eri8fold_vk_o4(vk, i0, j0, nao, eri4, dm);
        }
        for (; k < lenij; k++) {
                ij = idxij[k];
                for (kl = 0; kl < last_kl; kl++) {
                        eri1[kl] = eri[idx_tri[kl]*di*dj+ij];
                }
                i0 = idxi0[k];
                j0 = idxj0[k];
                ij0 = LOWERTRI_INDEX(i0, j0);
                nr_eri8fold_vj_o2(vj, ij0, eri1, tri_dm);
                nr_eri8fold_vk_o4(vk, i0, j0, nao, eri1, dm);
        }
        free(idxi0);
        free(idxj0);
        free(idxij);
        free(eri1);
}

void nr_vhf_direct_o4(double *dm, double *vj, double *vk, CVHFOpt *vhfopt,
                      const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env)
{
        unsigned int nao = CINTtot_cgto_spheric(bas, nbas);
        unsigned int npair = nao*(nao+1)/2;
        double *tri_dm = malloc(sizeof(double)*npair);
        double *tri_vj = malloc(sizeof(double)*npair);
        double *vj_priv, *vk_priv;
        unsigned int i, j, ij;
        unsigned int *ij2i = malloc(sizeof(unsigned int)*nbas*nbas);
        unsigned int *idx_tri = malloc(sizeof(unsigned int)*nao*nao);
        int *ao_loc = malloc(sizeof(unsigned int)*nbas);
        unsigned int di, dj;
        double *eribuf;

        compress_dm(tri_dm, dm, nao);
        set_ij2i(ij2i, nbas);
        memset(tri_vj, 0, sizeof(double)*npair);
        memset(vk, 0, sizeof(double)*nao*nao);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        index_blocks2tri(idx_tri, ao_loc, bas, nbas);

        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        if (vhfopt) {
                CVHFset_direct_scf_dm(vhfopt, dm, atm, natm, bas, nbas, env);
        }

#pragma omp parallel default(none) \
        shared(tri_dm, dm, tri_vj, vk, ij2i, nao, npair, ao_loc, idx_tri, \
               atm, bas, env, opt, vhfopt) \
        private(ij, i, j, di, dj, vj_priv, vk_priv, eribuf)
        {
                vj_priv = malloc(sizeof(double)*npair);
                vk_priv = malloc(sizeof(double)*nao*nao);
                memset(vj_priv, 0, sizeof(double)*npair);
                memset(vk_priv, 0, sizeof(double)*nao*nao);
#pragma omp for nowait schedule(guided, 2)
                for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                        i = ij2i[ij];
                        j = ij - (i*(i+1)/2);
                        di = CINTcgto_spheric(i, bas);
                        dj = CINTcgto_spheric(j, bas);
                        eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);
                        //nr8fold_eri_o1(eribuf, i, j, vhfopt, atm, natm, bas, nbas, env, opt);
                        if (nr8fold_eri_o2(eribuf, i, j, vhfopt,
                                           atm, natm, bas, nbas, env, opt)) {
                                nr8fold_jk_o3(vj_priv, vk_priv, tri_dm, dm, eribuf,
                                              i, j, ao_loc, idx_tri, bas, nbas);
                        }
                        free(eribuf);
                }
#pragma omp critical
                {
                        for (i = 0; i < npair; i++) {
                                tri_vj[i] += vj_priv[i];
                        }
                        for (i = 0; i < nao*nao; i++) {
                                vk[i] += vk_priv[i];
                        }
                }
                free(vj_priv);
                free(vk_priv);
        }

        for (i = 0, ij = 0; i < nao; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        vj[i*nao+j] = tri_vj[ij];
                        vj[j*nao+i] = tri_vj[ij];
                        vk[j*nao+i] = vk[i*nao+j];
                }
        }
        CINTdel_2e_optimizer(&opt);
        free(ij2i);
        free(idx_tri);
        free(ao_loc);
        free(tri_dm);
        free(tri_vj);
}

void nr_vhf_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env)
{
        CVHFinit_optimizer(vhfopt, atm, natm, bas, nbas, env);
        CVHFset_direct_scf(*vhfopt, atm, natm, bas, nbas, env);
}

