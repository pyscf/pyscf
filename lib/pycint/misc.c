/*
 * File: misc.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include "cint.h"
/*
#define DEBUG_ON(i, dj, l, k, s) printf("%d : %d, l = %d, kappa = %d, %s\n", \
                                 (i), (dj), (l), (k), (s));
*/
#define DEBUG_ON(...)


/*
 * tao(i) = -j  means  T(f_i) = -f_j
 * tao(i) =  j  means  T(f_i) =  f_j
 * Note: abs(tao(i)) is 1-based. C-array should indexed with abs(tao(i))-1
 */
void time_reversal_spinor(int tao[], const int *bas, const int nbas)
{
        unsigned int i, ibas, l, n, m, dj;

        i = 0;
        for (ibas = 0; ibas < nbas; ibas++) {
                l = bas(ANG_OF, ibas);
                if (l % 2 == 0) {
                        for (n = 0; n < bas(NCTR_OF, ibas); n++) {
                                if (bas(KAPPA_OF, ibas) >= 0) {
                                        // j = l - 1/2: d3/2, g7/2,...
                                        dj = l * 2;
                                        for (m = 0; m < dj; m += 2) {
                                                tao[i+m  ] = -(i + dj - m);
                                                tao[i+m+1] =   i + dj - (m+1);
                                        }
                                        DEBUG_ON(i, dj, l, bas(KAPPA_OF, ibas), "t(|-m>)=-|m>");
                                        i += dj;
                                }
                                if (bas(KAPPA_OF, ibas) <= 0) {
                                        // j = l + 1/2: s1/2, d5/2,...
                                        dj = l * 2 + 2;
                                        for (m = 0; m < dj; m += 2) {
                                                tao[i+m  ] = -(i + dj - m);
                                                tao[i+m+1] =   i + dj - (m+1);
                                        }
                                        DEBUG_ON(i, dj, l, bas(KAPPA_OF, ibas), "t(|-m>)=-|m>");
                                        i += dj;
                                }
                        }
                } else {
                        for (n = 0; n < bas(NCTR_OF, ibas); n++) {
                                if (bas(KAPPA_OF, ibas) >= 0) {
                                        // j = l - 1/2: p1/2, f5/2,...
                                        dj = l * 2;
                                        for (m = 0; m < dj; m += 2) {
                                                tao[i+m  ] =   i + dj - m;
                                                tao[i+m+1] = -(i + dj - (m+1));
                                        }
                                        DEBUG_ON(i, dj, l, bas(KAPPA_OF, ibas), "t(|-m>)=|m>");
                                        i += dj;
                                }
                                if (bas(KAPPA_OF, ibas) <= 0) {
                                        // j = l + 1/2: p3/2, f7/2,...
                                        dj = l * 2 + 2;
                                        for (m = 0; m < dj; m += 2) {
                                                tao[i+m  ] =   i + dj - m;
                                                tao[i+m+1] = -(i + dj - (m+1));
                                        }
                                        DEBUG_ON(i, dj, l, bas(KAPPA_OF, ibas), "t(|-m>)=|m>");
                                        i += dj;
                                }
                        }
                }
        }
}

void time_reversal_spinor_(int tao[], const int *bas, const int *nbas)
{
        time_reversal_spinor(tao, bas, *nbas);
}

