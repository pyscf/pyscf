/*
 * File: vhf_drv.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * HF potential (in AO representation)
 */

#include <stdlib.h>
#include <omp.h>
//#define NDEBUG //assert(X) if (!(X)) goto fail;
#include <assert.h>
#include "cint.h"
#include "vhf_drv.h"
#include "vhf/fblas.h"

#define MAX_OPTS 8 // should decide length of list opt according to filter

int pass(){ return 1; }

/*
 * reduce vj/vk via a set of filters
 */
static void vhf_reduce(const FPtr filter[], CINTOpt *opt_lst[],
                       const double *dm, double *vj, double *vk,
                       int ndim, int nset, int nset_dm, int idx, int *ao_loc,
                       const int *atm, const int natm,
                       const int *bas, const int nbas, const double *env)
{
        FPtr intor, fscreen;
        for (; *filter != NULL; filter+=4, opt_lst++) {
                intor = *(filter+1);
                fscreen = *(filter+2);
                assert(NULL != intor);
                (*filter)(intor, fscreen, dm, vj, vk,
                          &ndim, &nset, &nset_dm, &idx, ao_loc,
                          atm, &natm, bas, &nbas, env,
                          opt_lst); // pass a pointer of CINTOpt* to Fortran
        }
}

static void init_optimizers(CINTOpt *opt_lst[], const FPtr filter[], 
                            const int *atm, const int natm,
                            const int *bas, const int nbas, const double *env)
{
        void (*optimizer)(CINTOpt **, const int *, const int,
                          const int *, const int, const double *);
        int i = 0;
        for (i = 0; i < MAX_OPTS; i++) {
                opt_lst[i] = NULL;
        }
        for (i = 0; filter[i*4] != NULL; i++) {
                optimizer = filter[i*4+3];
                (*optimizer)(&opt_lst[i], atm, natm, bas, nbas, env);
        }
}
static void del_optimizers(CINTOpt *opt_lst[])
{
        int i = 0;
        for (i = 0; i < MAX_OPTS; i++) {
                if (opt_lst[i]) {
                        CINTdel_2e_optimizer(opt_lst+i);
                }
        }
}


/*
 *************************************************
 *
 * dm_{ij} = C_i C_j^*
 * ndim = shape(vj,0) = shape(vk,0) = shape(dm,0)
 */
static int nr_vhf_drv_sph(const FPtr* filter,
                          const double *dm, double *vj, double *vk,
                          const int ndim, const int nset, const int nset_dm,
                          const int *atm, const int natm,
                          const int *bas, const int nbas, const double *env)
{
        const int INC1 = 1;
        const double D1 = 1;
        const int len = ndim * ndim * nset * nset_dm;
        int idx;
        int *const ao_loc = malloc(sizeof(int) * nbas);
        double *vj_priv, *vk_priv;
        FPtr vhf_pre = filter[0];
        FPtr vhf_after = filter[1];
        filter += 2;

        CINTOpt *opt_lst[MAX_OPTS];
        init_optimizers(opt_lst, filter, atm, natm, bas, nbas, env);

        CINTdset0(len, vj);
        CINTdset0(len, vk);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        (*vhf_pre)(dm, &ndim, &nset, &nset_dm,
                   atm, &natm, bas, &nbas, env);

#pragma omp parallel default(none) \
        shared(filter, dm, vj, vk, atm, bas, env, opt_lst) \
        private(vj_priv, vk_priv, idx)
        {
                vj_priv = malloc(sizeof(double) * len);
                vk_priv = malloc(sizeof(double) * len);
                CINTdset0(len, vj_priv);
                CINTdset0(len, vk_priv);
#pragma omp for nowait schedule(guided, 2)
                for (idx = 0; idx < nbas * nbas; idx++) {
                        //l = idx / nbas;
                        //k = idx - l * nbas;
                        vhf_reduce(filter, opt_lst, dm, vj_priv, vk_priv,
                                   ndim, nset, nset_dm, idx, ao_loc,
                                   atm, natm, bas, nbas, env);
                }
#pragma omp critical
                {
                        daxpy_(&len, &D1, vj_priv, &INC1, vj, &INC1);
                        daxpy_(&len, &D1, vk_priv, &INC1, vk, &INC1);
                }
                free(vj_priv);
                free(vk_priv);
        }
        free(ao_loc);

        // post process to reform vj and vk
        (*vhf_after)(vj, vk, &ndim, &nset, &nset_dm,
                     atm, &natm, bas, &nbas, env);
        del_optimizers(opt_lst);
        return 0;
}

int nr_vhf_drv(const FPtr* filter,
               const double *dm, double *vj, double *vk,
               const int ndim, const int nset, const int nset_dm,
               const int *atm, const int natm,
               const int *bas, const int nbas, const double *env)
{
        return nr_vhf_drv_sph(filter, dm, vj, vk, ndim, nset, nset_dm,
                              atm, natm, bas, nbas, env);
}

int nr_vhf_drv_(const FPtr* filter,
                const double *dm, double *vj, double *vk,
                const int *ndim, const int *nset, const int *nset_dm,
                const int *atm, const int *natm,
                const int *bas, const int *nbas, const double *env)
{
        return nr_vhf_drv(filter, dm, vj, vk, *ndim, *nset, *nset_dm,
                          atm, *natm, bas, *nbas, env);
}

int r_vhf_drv(const FPtr* filter,
              const double *dm, double *vj, double *vk,
              const int ndim, const int nset, const int nset_dm,
              const int *atm, const int natm,
              const int *bas, const int nbas, const double *env)
{
        const int INC1 = 1;
        const double D1 = 1;
        const int len = ndim * ndim * nset * nset_dm * OF_CMPLX;
        int idx;
        int *const ao_loc = malloc(sizeof(int) * (nbas + ndim));
        int *tao = ao_loc + nbas;
        double *const tdm = malloc(sizeof(double) * ndim * ndim * nset_dm * OF_CMPLX);
        double *vj_priv, *vk_priv;
        FPtr vhf_pre = filter[0];
        FPtr vhf_after = filter[1];
        filter += 2;

        CINTOpt *opt_lst[MAX_OPTS];
        init_optimizers(opt_lst, filter, atm, natm, bas, nbas, env);

        CINTdset0(len, vj);
        CINTdset0(len, vk);
        CINTshells_spinor_offset(ao_loc, bas, nbas);
        time_reversal_spinor(tao, bas, nbas);
        (*vhf_pre)(tdm, dm, &ndim, &nset, &nset_dm,
                   atm, &natm, bas, &nbas, env);

#pragma omp parallel default(none) \
        shared(filter, vj, vk, atm, bas, env, opt_lst) \
        private(vj_priv, vk_priv, idx)
        {
                vj_priv = malloc(sizeof(double) * len);
                vk_priv = malloc(sizeof(double) * len);
                CINTdset0(len, vj_priv);
                CINTdset0(len, vk_priv);
#pragma omp for nowait schedule(guided, 2)
                for (idx = 0; idx < nbas * nbas; idx++) {
                        //l = idx / nbas;
                        //k = idx - l * nbas;
                        vhf_reduce(filter, opt_lst, tdm, vj_priv, vk_priv,
                                   ndim, nset, nset_dm, idx, ao_loc,
                                   atm, natm, bas, nbas, env);
                }
#pragma omp critical
                {
                        daxpy_(&len, &D1, vj_priv, &INC1, vj, &INC1);
                        daxpy_(&len, &D1, vk_priv, &INC1, vk, &INC1);
                }
                free(vj_priv);
                free(vk_priv);
        }
        free(ao_loc);
        free(tdm);

        // post process to reform vj and vk
        (*vhf_after)(vj, vk, &ndim, &nset, &nset_dm,
                     atm, &natm, bas, &nbas, env);
        del_optimizers(opt_lst);
        return 0;
}
int r_vhf_drv_(const FPtr* filter,
               const double *dm, double *vj, double *vk,
               const int *ndim, const int *nset, const int *nset_dm,
               const int *atm, const int *natm,
               const int *bas, const int *nbas, const double *env)
{
        return r_vhf_drv(filter, dm, vj, vk, *ndim, *nset, *nset_dm,
                         atm, *natm, bas, *nbas, env);
}
