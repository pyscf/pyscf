/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
//#define NDEBUG

//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))

/*
 * driver
 */
void AO2MOnr_e1_drv(int (*intor)(), void (*ftranse1)(), int (*fmmm)(),
                    double *eri, double *mo_coeff,
                    int ksh_start, int ksh_count,
                    int i_start, int i_count, int j_start, int j_count,
                    int ncomp, CINTOpt *cintopt, CVHFOpt *vhfopt,
                    int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e2_drv(void (*ftranse2)(), int (*fmmm)(),
                    double *vout, double *vin, double *mo_coeff,
                    int nijcount, int nao,
                    int i_start, int i_count, int j_start, int j_count);
/*
 * Denoting 2e integrals (ij|kl),
 * AO2MOnr_e1_drv transforms ij for ksh_start <= k shell < ksh_end.
 * The transformation C_pi C_qj (pq|k*) coefficients are stored in
 * mo_coeff, C_pi and C_qj are offset by i_start and i_count, j_start and j_count.
 * The output eri is an 2D array, ordered as (kl-AO-pair,ij-MO-pair) in
 * C-order.  Transposing is needed before calling AO2MOnr_e2_drv.
 *
 * AO2MOnr_e2_drv transforms kl for nijcount of ij pairs.
 * vin is assumed to be an C-array of (ij-MO-pair, kl-AO-pair)
 * vout is an C-array of (ij-MO-pair, kl-MO-pair)
 *
 * ftranse1 and ftranse2
 * ---------------------
 * AO2MOtranse1_nr_s4, AO2MOtranse1_nr_s2ij, AO2MOtranse1_nr_s2kl AO2MOtranse1_nr_s1
 * AO2MOtranse2_nr_s4, AO2MOtranse2_nr_s2ij, AO2MOtranse2_nr_s2kl AO2MOtranse2_nr_s1
 * Labels s4, s2, s1 are used to label the AO integral symmetry.  The
 * symmetry of transformed integrals are controled by function fmmm
 *
 * fmmm
 * ----
 * fmmm dim requirements:
 *                      | vout                          | eri
 * ---------------------+-------------------------------+-------------------
 *  AO2MOmmm_nr_s2_s2   | [:,bra_count*(bra_count+1)/2] | [:,nao*(nao+1)/2]
 *                      |    and bra_count==ket_count   |
 *  AO2MOmmm_nr_s2_iltj | [:,bra_count*ket_count]       | [:,nao*nao]
 *  AO2MOmmm_nr_s2_igtj | [:,bra_count*ket_count]       | [:,nao*nao]
 *  AO2MOmmm_nr_s1_iltj | [:,bra_count*ket_count]       | [:,nao*nao]
 *  AO2MOmmm_nr_s1_igtj | [:,bra_count*ket_count]       | [:,nao*nao]
 *
 * AO2MOmmm_nr_s1_iltj, AO2MOmmm_nr_s1_igtj, AO2MOmmm_nr_s2_s2,
 * AO2MOmmm_nr_s2_iltj, AO2MOmmm_nr_s2_igtj
 * Pick a proper function from the 5 kinds of AO2MO transformation.
 * 1. AO integral I_ij != I_ji, use
 *    AO2MOmmm_nr_s1_iltj or AO2MOmmm_nr_s1_igtj
 * 2. AO integral I_ij == I_ji, but the MO coefficients for bra and ket
 *    are different, use
 *    AO2MOmmm_nr_s2_iltj or AO2MOmmm_nr_s2_igtj
 * 3. AO integral I_ij == I_ji, and the MO coefficients are the same for
 *    bra and ket, use
 *    AO2MOmmm_nr_s2_s2
 *
 *      ftrans           |     allowed fmmm
 * ----------------------+---------------------
 *  AO2MOtranse1_nr_s4   |  AO2MOmmm_nr_s2_s2
 *  AO2MOtranse1_nr_s2ij |  AO2MOmmm_nr_s2_iltj
 *  AO2MOtranse2_nr_s4   |  AO2MOmmm_nr_s2_igtj                                                                                   
 *  AO2MOtranse2_nr_s2kl |                               
 * ----------------------+---------------------
 *  AO2MOtranse1_nr_s2kl |  AO2MOmmm_nr_s2_s2
 *  AO2MOtranse1_nr_s1   |  AO2MOmmm_nr_s2_iltj
 *  AO2MOtranse2_nr_s2ij |  AO2MOmmm_nr_s2_igtj
 *  AO2MOtranse2_nr_s1   |  AO2MOmmm_nr_s1_iltj
 *                       |  AO2MOmmm_nr_s1_igtj
 *
 */

struct _AO2MOEnvs {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        int nao;
        int ksh_start;
        int ksh_count;
        int bra_start;
        int bra_count;
        int ket_start;
        int ket_count;
        int ncomp;
        int *ao_loc;
        double *mo_coeff;
};


/* for m > n
 * calculate the upper triangle part (of Fortran order matrix)
 *   _        |------- n -------| _
 *   diag_off [ . . . . . . . . ] |
 *   _        [ . . . . . . . . ] m
 *            [   . . . . . . . ] |
 *            [     . . . . . . ] _
 */
void AO2MOdtriumm_o1(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int BLK = 48;
        int mstart = m - MAX(0, (m-diag_off)/BLK)*BLK;
        int nstart = mstart - diag_off;
        int nleft;

        dgemm_(&TRANS_T, &TRANS_N, &mstart, &n, &k,
               &D1, a, &k, b, &k, &D0, c, &m);

        for (; mstart < m; mstart+=BLK, nstart+=BLK) {
                nleft = n - nstart;

                dgemm_(&TRANS_T, &TRANS_N, &BLK, &nleft, &k,
                       &D1, a+mstart*k, &k, b+nstart*k, &k,
                       &D0, c+nstart*m+mstart, &m);
        }
}

/* for m < n
 * calculate the upper triangle part (of Fortran order matrix)
 *   _        |------- n -------| _
 *   diag_off [ . . . . . . . . ] |
 *   _        [ . . . . . . . . ] m
 *            [   . . . . . . . ] |
 *            [     . . . . . . ] _
 */
void AO2MOdtriumm_o2(int m, int n, int k, int diag_off,
                     double *a, double *b, double *c)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int BLK = 48;
        int nstart, nleft;
        int mend = diag_off;

        for (nstart = 0; nstart+BLK < m-diag_off; nstart+=BLK) {
                mend += BLK;
                dgemm_(&TRANS_T, &TRANS_N, &mend, &BLK, &k,
                       &D1, a, &k, b+nstart*k, &k,
                       &D0, c+nstart*m, &m);
        }
        nleft = n - nstart;
        dgemm_(&TRANS_T, &TRANS_N, &m, &nleft, &k,
               &D1, a, &k, b+nstart*k, &k,
               &D0, c+nstart*m, &m);
}


/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 * s1, s2 here to label the AO symmetry
 */
int AO2MOmmm_nr_s1_iltj(double *vout, double *eri, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*i_count);

        // C_pi (pq| = (iq|, where (pq| is in C-order
        dgemm_(&TRANS_N, &TRANS_N, &nao, &i_count, &nao,
               &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
               &D0, buf, &nao);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &nao,
               &D1, mo_coeff+j_start*nao, &nao, buf, &nao,
               &D0, vout, &j_count);
        free(buf);
        return 0;
}
/*
 * s1-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*nao]
 */
int AO2MOmmm_nr_s1_igtj(double *vout, double *eri, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*j_count);

        // C_qj (pq| = (pj|, where (pq| is in C-order
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &nao, &nao,
               &D1, mo_coeff+j_start*nao, &nao, eri, &nao,
               &D0, buf, &j_count);
        dgemm_(&TRANS_N, &TRANS_N, &j_count, &i_count, &nao,
               &D1, buf, &j_count, mo_coeff+i_start*nao, &nao,
               &D0, vout, &j_count);
        free(buf);
        return 0;
}

/*
 * s2-AO integrals to s2-MO integrals
 * shape requirements:
 *      vout[:,bra_count*(bra_count+1)/2] and bra_count==ket_count,
 *      eri[:,nao*(nao+1)/2]
 * first s2 is the AO symmetry, second s2 is the MO symmetry
 */
int AO2MOmmm_nr_s2_s2(double *vout, double *eri, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: assert(envs->bra_count == envs->ket_count);
                        return envs->bra_count * (envs->bra_count+1) / 2;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*(nao*i_count+i_count*j_count));
        double *buf1 = buf + nao*i_count;
        int i, j, ij;

        // C_pi (pq| = (iq|, where (pq| is in C-order
        dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
               &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
               &D0, buf, &nao);
        AO2MOdtriumm_o1(j_count, i_count, nao, 0,
                        mo_coeff+j_start*nao, buf, buf1);

        for (i = 0, ij = 0; i < i_count; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        vout[ij] = buf1[j];
                }
                buf1 += j_count;
        }
        free(buf);
        return 0;
}

/*
 * s2-AO integrals to s1-MO integrals, efficient for i_count < j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*(nao+1)/2]
 */
int AO2MOmmm_nr_s2_iltj(double *vout, double *eri, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*i_count);

        // C_pi (pq| = (iq|, where (pq| is in C-order
        dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
               &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
               &D0, buf, &nao);
        // C_qj (iq| = (ij|
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &nao,
               &D1, mo_coeff+j_start*nao, &nao, buf, &nao,
               &D0, vout, &j_count);
        free(buf);
        return 0;
}

/*
 * s2-AO integrals to s1-MO integrals, efficient for i_count > j_count
 * shape requirements:
 *      vout[:,bra_count*ket_count], eri[:,nao*(nao+1)/2]
 */
int AO2MOmmm_nr_s2_igtj(double *vout, double *eri, struct _AO2MOEnvs *envs,
                        int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int nao = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*j_count);

        // C_qj (pq| = (pj|, where (pq| is in C-order
        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, eri, &nao, mo_coeff+j_start*nao, &nao,
               &D0, buf, &nao);
        // C_pi (pj| = (ij|
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &nao,
               &D1, buf, &nao, mo_coeff+i_start*nao, &nao,
               &D0, vout, &j_count);
        free(buf);
        return 0;
}


/*
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
int AO2MOfill_nr_s1(int (*intor)(), int (*fprescreen)(),
                    double *eri, int ncomp, int ksh, int lsh,
                    CINTOpt *cintopt, CVHFOpt *vhfopt, struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        int dk = ao_loc[ksh+1] - ao_loc[ksh];
        int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int ish, jsh, di, dj;
        int empty = 1;
        int shls[4];
        double *buf = malloc(sizeof(double)*nao*nao*dk*dl*ncomp);

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
                CVHFunpack_nrblock2rect(buf, eri, ish, jsh, dk*dl*ncomp,
                                        nao, ao_loc);
        } }

        free(buf);
        return !empty;
}
/*
 * for given ksh, lsh, loop all ish > jsh
 */
static int fill_s2(int (*intor)(), int (*fprescreen)(),
                   double *eri, int ncomp, int ksh, int lsh,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int *ao_loc = envs->ao_loc;
        int dk = ao_loc[ksh+1] - ao_loc[ksh];
        int dl = ao_loc[lsh+1] - ao_loc[lsh];
        int ish, jsh, di, dj;
        int empty = 1;
        int shls[4];
        double *buf = malloc(sizeof(double)*nao*nao*dk*dl*ncomp);

        shls[2] = ksh;
        shls[3] = lsh;

        for (ish = 0; ish < envs->nbas; ish++) {
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
                CVHFunpack_nrblock2rect(buf, eri, ish, jsh, dk*dl*ncomp,
                                        nao, ao_loc);
        } }

        free(buf);
        return !empty;
}

int AO2MOfill_nr_s2ij(int (*intor)(), int (*fprescreen)(),
                      double *eri, int ncomp, int ksh, int lsh,
                      CINTOpt *cintopt, CVHFOpt *vhfopt, struct _AO2MOEnvs *envs)
{
        return fill_s2(intor, fprescreen, eri, ncomp,
                       ksh, lsh, cintopt, vhfopt, envs);
}
int AO2MOfill_nr_s2kl(int (*intor)(), int (*fprescreen)(),
                      double *eri, int ncomp, int ksh, int lsh,
                      CINTOpt *cintopt, CVHFOpt *vhfopt, struct _AO2MOEnvs *envs)
{
        if (ksh >= lsh) {
                return AO2MOfill_nr_s1(intor, fprescreen, eri, ncomp,
                                       ksh, lsh, cintopt, vhfopt, envs);
        } else {
                return 0;
        }
}
int AO2MOfill_nr_s4(int (*intor)(), int (*fprescreen)(),
                    double *eri, int ncomp, int ksh, int lsh,
                    CINTOpt *cintopt, CVHFOpt *vhfopt, struct _AO2MOEnvs *envs)
{
        if (ksh >= lsh) {
                return fill_s2(intor, fprescreen, eri, ncomp,
                               ksh, lsh, cintopt, vhfopt, envs);
        } else {
                return 0;
        }
}


static int step_rect(i, j, nao, ioff)
{
        return (i-ioff) * nao + j;
}
static int step_tril(i, j, nao, ioff)
{
        return (i*(i+1)/2 + j) - (ioff*(ioff+1)/2);
}

static void trans_kgtl(int (*intor)(), int (*fmmm)(), int (*fill)(), int (*fstep)(),
                       double *eri_mo, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int nao2 = nao * nao;
        int kloc = envs->ao_loc[ksh];
        int lloc = envs->ao_loc[lsh];
        int dk = envs->ao_loc[ksh+1] - kloc;
        int dl = envs->ao_loc[lsh+1] - lloc;
        int ncomp = envs->ncomp;
        int kstart = envs->ao_loc[envs->ksh_start];
        int kend = envs->ao_loc[envs->ksh_start+envs->ksh_count];
        int k, l, k0, l0, icomp;
        unsigned long ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        unsigned long neri_mo = (*fstep)(kend, 0, nao, kstart) * ij_pair; // size of one component
        unsigned long off;
        double *eri = malloc(sizeof(double)*dk*dl*nao2*ncomp);
        double *peri;
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

        if ((*fill)(intor, fprescreen,
                    eri, ncomp, ksh, lsh, cintopt, vhfopt, envs)) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        peri = eri + nao2*dk*dl * icomp;
                        for (l0 = lloc, l = 0; l < dl; l++, l0++) {
                        for (k0 = kloc, k = 0; k < dk; k++, k0++) {
                                off = ij_pair * (*fstep)(k0, l0, nao, kstart);
                                (*fmmm)(eri_mo+off, peri, envs, 0);
                                peri += nao2;
                        } }
                        eri_mo += neri_mo;
                }
        }
        free(eri);
}

static void trans_keql(int (*intor)(), int (*fmmm)(), int (*fill)(), int (*fstep)(),
                       double *eri_mo, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        int nao2 = nao * nao;
        int kloc = envs->ao_loc[ksh];
        int lloc = envs->ao_loc[lsh];
        int dk = envs->ao_loc[ksh+1] - kloc;
        int dl = envs->ao_loc[lsh+1] - lloc;
        int ncomp = envs->ncomp;
        int kstart = envs->ao_loc[envs->ksh_start];
        int kend = envs->ao_loc[envs->ksh_start+envs->ksh_count];
        int k, l, k0, l0, icomp;
        unsigned long ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        unsigned long neri_mo = (*fstep)(kend, 0, nao, kstart) * ij_pair;
        unsigned long off;
        double *eri = malloc(sizeof(double)*dk*dl*nao2*ncomp);
        double *peri;
        int (*fprescreen)();

        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

        if ((*fill)(intor, fprescreen,
                    eri, ncomp, ksh, lsh, cintopt, vhfopt, envs)) {
                for (icomp = 0; icomp < ncomp; icomp++) {
                        peri = eri + nao2*dk*dl * icomp;
                        for (k0 = kloc, k = 0; k < dk; k++, k0++) {
                        for (l0 = lloc, l = 0; l0 <= k0; l++, l0++) {
                                off = ij_pair * (*fstep)(k0, l0, nao, kstart);
                                (*fmmm)(eri_mo+off, peri+nao2*(l*dk+k), envs,0);
                        } }
                        eri_mo += neri_mo;
                }
        }
        free(eri);
}

/*
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
// fmmm can be AO2MOmmm_nr_s2_s2, AO2MOmmm_nr_s2_iltj, AO2MOmmm_nr_s2_igtj
void AO2MOtranse1_nr_s4(int (*intor)(), int (*fmmm)(),
                        double *eri_mo, int ksh, int lsh,
                        CINTOpt *cintopt, CVHFOpt *vhfopt,
                        struct _AO2MOEnvs *envs)
{
        if (ksh > lsh) {
                trans_kgtl(intor, fmmm, AO2MOfill_nr_s4, step_tril,
                           eri_mo, ksh, lsh, cintopt, vhfopt, envs);
        } else if (ksh == lsh) {
                trans_keql(intor, fmmm, AO2MOfill_nr_s4, step_tril,
                           eri_mo, ksh, lsh, cintopt, vhfopt, envs);
        } else { // ksh < lsh
                return;
        }
}
// fmmm can be AO2MOmmm_nr_s2_s2, AO2MOmmm_nr_s2_iltj, AO2MOmmm_nr_s2_igtj
void AO2MOtranse1_nr_s2ij(int (*intor)(), int (*fmmm)(),
                          double *eri_mo, int ksh, int lsh,
                          CINTOpt *cintopt, CVHFOpt *vhfopt,
                          struct _AO2MOEnvs *envs)
{
        trans_kgtl(intor, fmmm, AO2MOfill_nr_s2ij, step_rect,
                   eri_mo, ksh, lsh, cintopt, vhfopt, envs);
}
// AO2MOmmm_nr_s1_iltj, AO2MOmmm_nr_s1_igtj
// fmmm can be AO2MOmmm_nr_s2_s2, AO2MOmmm_nr_s2_iltj, AO2MOmmm_nr_s2_igtj
// However, for the last three, AO2MOtranse2_nr_s4 is more efficient
void AO2MOtranse1_nr_s2kl(int (*intor)(), int (*fmmm)(),
                          double *eri_mo, int ksh, int lsh,
                          CINTOpt *cintopt, CVHFOpt *vhfopt,
                          struct _AO2MOEnvs *envs)
{
        if (ksh > lsh) {
                trans_kgtl(intor, fmmm, AO2MOfill_nr_s2kl, step_tril,
                           eri_mo, ksh, lsh, cintopt, vhfopt, envs);
        } else if (ksh == lsh) {
                trans_keql(intor, fmmm, AO2MOfill_nr_s2kl, step_tril,
                           eri_mo, ksh, lsh, cintopt, vhfopt, envs);
        } else { // ksh < lsh
                return;
        }
}
// AO2MOmmm_nr_s1_iltj, AO2MOmmm_nr_s1_igtj
// fmmm can be AO2MOmmm_nr_s2_s2, AO2MOmmm_nr_s2_iltj, AO2MOmmm_nr_s2_igtj
// However, for the last three, AO2MOtranse2_nr_s2ij is more efficient
void AO2MOtranse1_nr_s1(int (*intor)(), int (*fmmm)(),
                        double *eri_mo, int ksh, int lsh,
                        CINTOpt *cintopt, CVHFOpt *vhfopt,
                        struct _AO2MOEnvs *envs)
{
        trans_kgtl(intor, fmmm, AO2MOfill_nr_s1, step_rect,
                   eri_mo, ksh, lsh, cintopt, vhfopt, envs);
}


/*
 * ************************************************
 * s1, s2ij, s2kl, s4 here to label the AO symmetry
 */
void AO2MOtranse2_nr_s1(int (*fmmm)(),
                        double *vout, double *vin, int row_id,
                        struct _AO2MOEnvs *envs)
{
        unsigned long ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        unsigned long nao2 = envs->nao * envs->nao;
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, envs, 0);
}

void AO2MOtranse2_nr_s2ij(int (*fmmm)(),
                          double *vout, double *vin, int row_id,
                          struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s1(fmmm, vout, vin, row_id, envs);
}

void AO2MOtranse2_nr_s2kl(int (*fmmm)(),
                          double *vout, double *vin, int row_id,
                          struct _AO2MOEnvs *envs)
{
        int nao = envs->nao;
        unsigned long ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        unsigned long nao2 = nao*(nao+1)/2;
        double *buf = malloc(sizeof(double) * nao*nao);
        NPdunpack_tril(nao, vin+nao2*row_id, buf, 0);
        (*fmmm)(vout+ij_pair*row_id, buf, envs, 0);
        free(buf);
}

void AO2MOtranse2_nr_s4(int (*fmmm)(),
                        double *vout, double *vin, int row_id,
                        struct _AO2MOEnvs *envs)
{
        AO2MOtranse2_nr_s2kl(fmmm, vout, vin, row_id, envs);
}


/*
 * ************************************************
 * Denoting 2e integrals (ij|kl),
 * transform ij for ksh_start <= k shell < ksh_end.
 * The transformation C_pi C_qj (pq|k*) coefficients are stored in
 * mo_coeff, C_pi and C_qj are offset by i_start and i_count, j_start and j_count
 *
 * The output eri is an 2D array, ordered as (kl-AO-pair,ij-MO-pair) in
 * C-order.  Transposing is needed before calling AO2MOnr_e2_drv.
 */
void AO2MOnr_e1_drv(int (*intor)(), void (*ftranse1)(), int (*fmmm)(),
                    double *eri, double *mo_coeff,
                    int ksh_start, int ksh_count,
                    int i_start, int i_count, int j_start, int j_count,
                    int ncomp, CINTOpt *cintopt, CVHFOpt *vhfopt,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        int nao = CINTtot_cgto_spheric(bas, nbas);
        int *ao_loc = malloc(sizeof(int)*(nbas+1));
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        ao_loc[nbas] = nao;

        struct _AO2MOEnvs envs = {natm, nbas, atm, bas, env, nao,
                                  ksh_start, ksh_count,
                                  i_start, i_count, j_start, j_count,
                                  ncomp, ao_loc, mo_coeff};

        int ksh, lsh, kl;
        const int klsh_start = ksh_start*nbas;
        const int klsh_end = (ksh_start+ksh_count) * nbas;
#pragma omp parallel default(none) \
        shared(eri, envs, cintopt, vhfopt, \
               ftranse1, fmmm, intor, nbas) \
        private(kl, ksh, lsh)
#pragma omp for nowait schedule(dynamic)
        for (kl = klsh_start; kl < klsh_end; kl++) {
                ksh = kl / nbas;
                lsh = kl - ksh*nbas;
                (*ftranse1)(intor, fmmm,
                            eri, ksh, lsh, cintopt, vhfopt, &envs);
        }

        free(ao_loc);
}

void AO2MOnr_e2_drv(void (*ftranse2)(), int (*fmmm)(),
                    double *vout, double *vin, double *mo_coeff,
                    int nijcount, int nao,
                    int i_start, int i_count, int j_start, int j_count)
{
        struct _AO2MOEnvs envs;
        envs.bra_start = i_start;
        envs.bra_count = i_count;
        envs.ket_start = j_start;
        envs.ket_count = j_count;
        envs.nao = nao;
        envs.mo_coeff = mo_coeff;

        int i;
#pragma omp parallel default(none) \
        shared(ftranse2, fmmm, vout, vin, nijcount, envs) \
        private(i)
#pragma omp for nowait schedule(static)
        for (i = 0; i < nijcount; i++) {
                (*ftranse2)(fmmm, vout, vin, i, &envs);
        }
}

