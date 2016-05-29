/*
 *
 */

#include <stdlib.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define SYMMETRIC       3
#define NCTRMAX         72

static void dcopy(double *out, double *in, int comp, int ni, int nj, int di, int dj)
{
        int i, j, ic;
        for (ic = 0; ic < comp; ic++) {
                for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                out[j*ni+i] = in[j*di+i];
                        }
                }
                out += ni * nj;
                in  += di * dj;
        }
}

/*
 * mat(naoi,naoj,comp) in F-order
 */
void GTOint2c(int (*intor)(), double *mat, int comp, int hermi,
              int *shls_slice, int *ao_loc, CINTOpt *opt,
              int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];

#pragma omp parallel default(none) \
        shared(intor, mat, comp, hermi, ao_loc, opt, atm, natm, bas, nbas, env)
{
        int ish, jsh, ij, di, dj, i0, j0;
        int shls[2];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*comp);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        // fill up only upper triangle of F-array
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                (*intor)(buf, shls, atm, natm, bas, nbas, env);

                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                dcopy(mat+j0*naoi+i0, buf, comp, naoi, naoj, di, dj);
        }
        free(buf);
}
        if (hermi != PLAIN) { // lower triangle of F-array
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPdsymm_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}

static void zcopy(double complex *out, double complex *in,
                  int comp, int ni, int nj, int di, int dj)
{
        int i, j, ic;
        for (ic = 0; ic < comp; ic++) {
                for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++) {
                                out[j*ni+i] = in[j*di+i];
                        }
                }
                out += ni * nj;
                in  += di * dj;
        }
}


void GTOint2c_spinor(int (*intor)(), double complex *mat, int comp, int hermi,
                     int *shls_slice, int *ao_loc, CINTOpt *opt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];

#pragma omp parallel default(none) \
        shared(intor, mat, comp, hermi, ao_loc, opt, atm, natm, bas, nbas, env)
{
        int ish, jsh, ij, di, dj, i0, j0;
        int shls[2];
        double complex *buf = malloc(sizeof(double complex)*NCTRMAX*NCTRMAX*comp);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                (*intor)(buf, shls, atm, natm, bas, nbas, env);

                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                zcopy(mat+j0*naoi+i0, buf, comp, naoi, naoj, di, dj);
        }
        free(buf);
}
        if (hermi != PLAIN) {
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPzhermi_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}




void GTOint2c2e(int (*intor)(), double *mat, int comp, int hermi,
              int *shls_slice, int *ao_loc, CINTOpt *opt,
              int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];

#pragma omp parallel default(none) \
        shared(intor, mat, comp, hermi, ao_loc, opt, atm, natm, bas, nbas, env)
{
        int ish, jsh, ij, di, dj, i0, j0;
        int shls[2];
        double *buf = malloc(sizeof(double)*NCTRMAX*NCTRMAX*comp);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                (*intor)(buf, shls, atm, natm, bas, nbas, env, opt);

                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                dcopy(mat+j0*naoi+i0, buf, comp, naoi, naoj, di, dj);
        }
        free(buf);
}
        if (hermi != PLAIN) {
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPdsymm_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}

void GTOint2c2e_spinor(int (*intor)(), double complex *mat, int comp, int hermi,
                     int *shls_slice, int *ao_loc, CINTOpt *opt,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];

#pragma omp parallel default(none) \
        shared(intor, mat, comp, hermi, ao_loc, opt, atm, natm, bas, nbas, env)
{
        int ish, jsh, ij, di, dj, i0, j0;
        int shls[2];
        double complex *buf = malloc(sizeof(double complex)*NCTRMAX*NCTRMAX*comp);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                (*intor)(buf, shls, atm, natm, bas, nbas, env, opt);

                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                di = ao_loc[ish+1] - ao_loc[ish];
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                zcopy(mat+j0*naoi+i0, buf, comp, naoi, naoj, di, dj);
        }
        free(buf);
}
        if (hermi != PLAIN) {
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPzhermi_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}

