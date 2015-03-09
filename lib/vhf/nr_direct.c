/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "nr_direct.h"

#define LOWERTRI_INDEX(I,J)     ((I) > (J) ? ((I)*((I)+1)/2+(J)) : ((J)*((J)+1)/2+(I)))
#define MAX(I,J)        ((I) > (J) ? (I) : (J))
// 9f or 7g or 5h functions should be enough
#define NCTRMAX         64


/*
 * distribute a 4-shell-eris' block to 2d-array (C-contiguous), no symmetry
 */
void CVHFunpack_nrblock2rect(double *buf, double *eri,
                             int ish, int jsh, int dkl, int nao, int *ao_loc)
{
        size_t nao2 = nao * nao;
        int iloc = ao_loc[ish];
        int jloc = ao_loc[jsh];
        int di = ao_loc[ish+1] - iloc;
        int dj = ao_loc[jsh+1] - jloc;
        int i, j, kl;
        eri += iloc * nao + jloc;

        for (kl = 0; kl < dkl; kl++) {
                for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                        eri[i*nao+j] = buf[j*di+i];
                } }
                eri += nao2;
                buf += di*dj;
        }
}
/*
 * distribute a 4-shell-eris' block to lower triangle array (C-contiguous, i>=j),
 * hermitian is assumed
 */
void CVHFunpack_nrblock2tril(double *buf, double *eri,
                             int ish, int jsh, int dkl, int nao, int *ao_loc)
{
        size_t nao2 = nao * nao;
        int iloc = ao_loc[ish];
        int jloc = ao_loc[jsh];
        int di = ao_loc[ish+1] - iloc;
        int dj = ao_loc[jsh+1] - jloc;
        int i, j, kl;
        eri += iloc*(iloc+1)/2 + jloc;
        double *eri0 = eri;

        if (ish > jsh) {
                for (kl = 0; kl < dkl; kl++) {
                        eri0 = eri + nao2 * kl;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        eri0[j] = buf[j*di+i];
                                }
                                eri0 += ao_loc[ish] + i + 1;
                        }
                        buf += di*dj;
                }
        } else { // ish == jsh
                for (kl = 0; kl < dkl; kl++) {
                        eri0 = eri + nao2 * kl;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        eri0[j] = buf[j*di+i];
                                }
                                // row ao_loc[ish]+i has ao_loc[ish]+i+1 elements
                                eri0 += ao_loc[ish] + i + 1;
                        }
                        buf += di*dj;
                }
        }
}
/*
 * distribute a 4-shell-eris' block to 2d-array (C-contiguous),
 * hermitian is assumed
 */
void CVHFunpack_nrblock2trilu(double *buf, double *eri,
                              int ish, int jsh, int dkl, int nao, int *ao_loc)
{
        size_t nao2 = nao * nao;
        int iloc = ao_loc[ish];
        int jloc = ao_loc[jsh];
        int di = ao_loc[ish+1] - iloc;
        int dj = ao_loc[jsh+1] - jloc;
        int i, j, kl;
        double *eril = eri + iloc * nao + jloc;
        double *eriu = eri + jloc * nao + iloc;

        for (kl = 0; kl < dkl; kl++) {
                for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++) {
                        eril[i*nao+j] = buf[j*di+i];
                        eriu[j*nao+i] = buf[j*di+i];
                } }
                eril += nao2;
                eriu += nao2;
                buf += di*dj;
        }
}
/*
 * like CVHFunpack_nrblock2trilu, but anti-hermitian between ij is assumed
 */
void CVHFunpack_nrblock2trilu_anti(double *buf, double *eri,
                                   int ish, int jsh, int dkl, int nao, int *ao_loc)
{
        size_t nao2 = nao * nao;
        int iloc = ao_loc[ish];
        int jloc = ao_loc[jsh];
        int di = ao_loc[ish+1] - iloc;
        int dj = ao_loc[jsh+1] - jloc;
        int i, j, kl;
        double *eril = eri + iloc * nao + jloc;
        double *eriu = eri + jloc * nao + iloc;

        if (ish > jsh) {
                for (kl = 0; kl < dkl; kl++) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                eril[i*nao+j] = buf[j*di+i];
                                eriu[j*nao+i] =-buf[j*di+i];
                        } }
                        eril += nao2;
                        eriu += nao2;
                        buf += di*dj;
                }
        } else {
                CVHFunpack_nrblock2rect(buf, eri, ish, jsh, dkl, nao, ao_loc);
        }
}



/*
 * for given ksh, lsh, loop all ish, jsh
 */
int CVHFfill_nr_s1(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ncomp, int ksh, int lsh,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int *ao_loc = envs->ao_loc;
        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int ish, jsh, di, dj;
        int empty = 1;
        int shls[4];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*dk*dl*ncomp);

        shls[2] = ksh;
        shls[3] = lsh;

        for (ish = 0; ish < envs->nbas; ish++) {
        for (jsh = 0; jsh < envs->nbas; jsh++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)) {
                        empty = !(*intor)(buf, shls, envs->atm, envs->natm,
                                          envs->bas, envs->nbas, envs->env,
                                          cintopt)
                                && empty;
                } else {
                        memset(buf, 0, sizeof(double)*di*dj*dk*dl*ncomp);
                }
                (*funpack)(buf, eri, ish, jsh, dk*dl*ncomp, nao, ao_loc);
        } }

        free(buf);
        return !empty;
}
/*
 * for given ksh, lsh, loop all ish > jsh
 */
static int fill_s2(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ncomp, int ksh, int lsh, int ish_count,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int *ao_loc = envs->ao_loc;
        const int dk = ao_loc[ksh+1] - ao_loc[ksh];
        const int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int ish, jsh, di, dj;
        int empty = 1;
        int shls[4];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*dk*dl*ncomp);

        shls[2] = ksh;
        shls[3] = lsh;

        for (ish = 0; ish < ish_count; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt,
                                  envs->atm, envs->bas, envs->env)) {
                        empty = !(*intor)(buf, shls, envs->atm, envs->natm,
                                          envs->bas, envs->nbas, envs->env,
                                          cintopt)
                                && empty;
                } else {
                        memset(buf, 0, sizeof(double)*di*dj*dk*dl*ncomp);
                }
                (*funpack)(buf, eri, ish, jsh, dk*dl*ncomp, nao, ao_loc);
        } }

        free(buf);
        return !empty;
}
int CVHFfill_nr_s2ij(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ksh, int lsh,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        return fill_s2(intor, funpack, fprescreen, eri, ncomp,
                       ksh, lsh, envs->nbas, cintopt, vhfopt, envs);
}
int CVHFfill_nr_s2kl(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                     double *eri, int ncomp, int ksh, int lsh,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ksh >= lsh) {
                return CVHFfill_nr_s1(intor, funpack, fprescreen, eri, ncomp,
                                      ksh, lsh, cintopt, vhfopt, envs);
        } else {
                return 0;
        }
}
int CVHFfill_nr_s4(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ncomp, int ksh, int lsh,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ksh >= lsh) {
                return fill_s2(intor, funpack, fprescreen, eri, ncomp,
                               ksh, lsh, envs->nbas, cintopt, vhfopt, envs);
        } else {
                return 0;
        }
}
int CVHFfill_nr_s8(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ncomp, int ksh, int lsh,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs)
{
        if (ksh >= lsh) {
                // 8-fold symmetry, k>=l, k>=i>=j, 
                return fill_s2(intor, funpack, fprescreen, eri, ncomp,
                               ksh, lsh, ksh+1, cintopt, vhfopt, envs);
        } else {
                return 0;
        }
}


/* Note funpack order |kl) in Fortran-contiguous, (ij| in C-contiguous */
static void filldot_kgtl(int (*intor)(), void (*funpack)(), int (*fill)(),
                         void (**fjk)(), double **dms, double *vjk,
                         int n_dm, int ncomp, int ksh, int lsh,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int nao2 = nao * nao;
        const int kloc = envs->ao_loc[ksh];
        const int lloc = envs->ao_loc[lsh];
        const int dk = envs->ao_loc[ksh+1] - kloc;
        const int dl = envs->ao_loc[lsh+1] - lloc;
        int k, l, k0, l0, ieri, idm;
        double *eri = malloc(sizeof(double)*dk*dl*nao2*ncomp);
        double *peri;
        double *dm;
        void (*pf)();
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

        if ((*fill)(intor, funpack, fprescreen,
                    eri, ncomp, ksh, lsh, cintopt, vhfopt, envs)) {
                for (idm = 0; idm < n_dm; idm++) {
                        pf = fjk[idm];
                        dm = dms[idm];
                        for (ieri = 0; ieri < ncomp; ieri++) {
                                peri = eri + nao2*dk*dl * ieri;
                                for (l0 = lloc, l = 0; l < dl; l++, l0++) {
                                for (k0 = kloc, k = 0; k < dk; k++, k0++) {
                                        (*pf)(peri, dm, vjk, nao, k0, l0);
                                        peri += nao2;
                                } }
                                vjk += nao2;
                        }
                }
        }
        free(eri);
}
static void filldot_keql(int (*intor)(), void (*funpack)(), int (*fill)(),
                         void (**fjk)(), double **dms, double *vjk,
                         int n_dm, int ncomp, int ksh, int lsh,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         struct _VHFEnvs *envs)
{
        const int nao = envs->nao;
        const int nao2 = nao * nao;
        const int kloc = envs->ao_loc[ksh];
        const int lloc = envs->ao_loc[lsh];
        const int dk = envs->ao_loc[ksh+1] - kloc;
        const int dl = envs->ao_loc[lsh+1] - lloc;
        int k, l, k0, l0, ieri, idm, off;
        double *eri = malloc(sizeof(double)*dk*dl*nao2*ncomp);
        double *peri;
        double *dm;
        void (*pf)();
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

        if ((*fill)(intor, funpack, fprescreen,
                    eri, ncomp, ksh, lsh, cintopt, vhfopt, envs)) {
                for (idm = 0; idm < n_dm; idm++) {
                        pf = fjk[idm];
                        dm = dms[idm];
                        for (ieri = 0; ieri < ncomp; ieri++) {
                                peri = eri + nao2*dk*dl * ieri;
                                for (k0 = kloc, k = 0; k < dk; k++, k0++) {
                                for (l0 = lloc, l = 0; l0 <= k0; l++, l0++) {
                                        off = nao2 * (l*dk+k);
                                        (*pf)(peri+off, dm, vjk, nao, k0, l0);
                                } }
                                vjk += nao2;
                        }
                }
        }
        free(eri);
}

/*
 * generate eris and dot against dm to get J,K matrix
 */
/*
 * call fill_s8 to fill eri tril which has (kloc+dk)*(kloc+dk+1)/2 elements
 * shells (ij|kl) has 8-fold symmetry k>=l, k>=i>=j.  note the implicit
 * transposing of fill functions and the consistence between funpack and fjk.
 *
 * dot_s8, dot_s4 assumes the lower triangle eri saved continuously.
 * unpack_tril save lower triangle eris continuously, bug unpack_rect
 * and unpack_trilu save the entire 2d matrix
 */
void CVHFfill_dot_nrs8(int (*intor)(), void (*funpack)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _VHFEnvs *envs)
{
        if (ksh > lsh) {
                filldot_kgtl(intor, funpack, CVHFfill_nr_s8, fjk,
                             dms, vjk, n_dm, ncomp,
                             ksh, lsh, cintopt, vhfopt, envs);
        } else if (ksh == lsh) {
                filldot_keql(intor, funpack, CVHFfill_nr_s8, fjk,
                             dms, vjk, n_dm, ncomp,
                             ksh, lsh, cintopt, vhfopt, envs);
        } else { // ksh < lsh
                return;
        }
}
/*
 * call fill_s4 to fill eri tril which has nao*(nao+1)/2 elements
 * shells (ij|kl) has 4-fold symmetry k>=l, i>=j.  note the implicit
 * transposing of fill functions and the consistence between funpack and fjk.
 *
 * dot_s8, dot_s4 assumes the lower triangle eri saved continuously.
 * unpack_tril save lower triangle eris continuously, bug unpack_rect
 * and unpack_trilu save the entire 2d matrix
 */
void CVHFfill_dot_nrs4(int (*intor)(), void (*funpack)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _VHFEnvs *envs)
{
        if (ksh > lsh) {
                filldot_kgtl(intor, funpack, CVHFfill_nr_s4, fjk,
                             dms, vjk, n_dm, ncomp,
                             ksh, lsh, cintopt, vhfopt, envs);
        } else if (ksh == lsh) {
                filldot_keql(intor, funpack, CVHFfill_nr_s4, fjk,
                             dms, vjk, n_dm, ncomp,
                             ksh, lsh, cintopt, vhfopt, envs);
        } else { // ksh < lsh
                return;
        }
}
/*
 * call fill_s2kl to fill eri which has nao*nao elements
 * kl in (ij|kl) has k>=l, ij is regular rectangle matrix.  note the
 * implicit transposing of fill functions and the consistence between
 * funpack and fjk.
 */
void CVHFfill_dot_nrs2kl(int (*intor)(), void (*funpack)(), void (**fjk)(),
                         double **dms, double *vjk,
                         int n_dm, int ncomp, int ksh, int lsh,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         struct _VHFEnvs *envs)
{
        if (ksh > lsh) {
                filldot_kgtl(intor, funpack, CVHFfill_nr_s2kl, fjk,
                             dms, vjk, n_dm, ncomp,
                             ksh, lsh, cintopt, vhfopt, envs);
        } else if (ksh == lsh) {
                filldot_keql(intor, funpack, CVHFfill_nr_s2kl, fjk,
                             dms, vjk, n_dm, ncomp,
                             ksh, lsh, cintopt, vhfopt, envs);
        } else { // ksh < lsh
                return;
        }
}
/*
 * call fill_s2ij to fill eri which has nao*(nao+1)/2 elements
 * kl in (ij|kl) loop over all kl, and ij has i>=j.  note the implicit
 * transposing of fill functions and the consistence between funpack and fjk.
 */
void CVHFfill_dot_nrs2ij(int (*intor)(), void (*funpack)(), void (**fjk)(),
                         double **dms, double *vjk,
                         int n_dm, int ncomp, int ksh, int lsh,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         struct _VHFEnvs *envs)
{
        filldot_kgtl(intor, funpack, CVHFfill_nr_s2ij, fjk,
                     dms, vjk, n_dm, ncomp,
                     ksh, lsh, cintopt, vhfopt, envs);
}

void CVHFfill_dot_nrs1(int (*intor)(), void (*funpack)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _VHFEnvs *envs)
{
        filldot_kgtl(intor, funpack, CVHFfill_nr_s1, fjk,
                     dms, vjk, n_dm, ncomp,
                     ksh, lsh, cintopt, vhfopt, envs);
}

/*
 * drv loop over kl, generate eris of ij for given kl, call fjk to
 * calculate vj, vk. Note the implicit TRANSPOSING between ij,kl in
 * funpack.  So fdot of s2kl should call fjk of s2ij and fdot of
 * s2ij should call fjk of s2kl
 * 
 * n_dm is the number of dms for one [array(ij|kl)],
 * ncomp is the number of components that produced by intor
 */
void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), void (*funpack)(),
                       void (**fjk)(), double **dms, double *vjk,
                       int n_dm, int ncomp, CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        double *v_priv;
        int k, l, kl;
        int *ao_loc = malloc(sizeof(int)*(nbas+1));
        struct _VHFEnvs envs = {natm, nbas, atm, bas, env, nao, ao_loc};

        memset(vjk, 0, sizeof(double)*nao*nao*n_dm*ncomp);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        ao_loc[nbas] = nao;

#pragma omp parallel default(none) \
        shared(intor, funpack, fdot, fjk, \
               dms, vjk, n_dm, ncomp, nbas, cintopt, vhfopt, envs) \
        private(kl, k, l, v_priv)
        {
                v_priv = malloc(sizeof(double)*nao*nao*n_dm*ncomp);
                memset(v_priv, 0, sizeof(double)*nao*nao*n_dm*ncomp);
#pragma omp for nowait schedule(dynamic, 2)
                for (kl = 0; kl < nbas*nbas; kl++) {
                        k = kl / nbas;
                        l = kl - k * nbas;
                        (*fdot)(intor, funpack, fjk,
                                dms, v_priv, n_dm, ncomp, k, l,
                                cintopt, vhfopt, &envs);
                }
#pragma omp critical
                {
                        for (k = 0; k < nao*nao*n_dm*ncomp; k++) {
                                vjk[k] += v_priv[k];
                        }
                }
                free(v_priv);
        }

        free(ao_loc);
}

