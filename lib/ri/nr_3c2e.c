/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * auxe2: (ij|P) where P is the auxiliary basis
 */

#include <stdlib.h>
#include <assert.h>
#include "cint.h"
#include "vhf/fblas.h"
#include "ao2mo/nr_ao2mo.h"

#define NCTRMAX         72
#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define OUTPUTIJ        1
#define INPUT_IJ        2

/*
 * (ij| are stored in C-order
 */
static void dcopy_s1(double *out, double *in, int comp,
                     int nj, int nij, int nijk, int di, int dj, int dk)
{
        const size_t dij = di * dj;
        int i, j, k, ic;
        double *pout, *pin;
        for (ic = 0; ic < comp; ic++) {
                for (k = 0; k < dk; k++) {
                        pout = out + k * nij;
                        pin  = in  + k * dij;
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pout[i*nj+j] = pin[j*di+i];
                        } }
                }
                out += nijk;
                in  += dij * dk;
        }
}
void RInr3c_fill_s1(int (*intor)(), double *out, double *buf,
                    int comp, int ish, int jsh,
                    int *shls_slice, int *ao_loc, CINTOpt *cintopt,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t naok = ao_loc[ksh1] - ao_loc[ksh0];
        const size_t nij = naoi * naoj;
        const size_t nijk = nij * naok;

        ish += ish0;
        jsh += jsh0;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        out += ip * naoj + jp;

        int ksh, dk, k0;
        int shls[3];

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
                shls[2] = ksh;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                k0 = ao_loc[ksh  ] - ao_loc[ksh0];
                (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, cintopt, NULL);
                dcopy_s1(out+k0*nij, buf, comp, naoj, nij, nijk, di, dj, dk);
        }
}

