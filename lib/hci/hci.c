/*
 * Slater-Condon rule implementation for Heat-Bath CI
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "hci.h"
//#include <omp.h>
#include <limits.h>


void contract_h_c(double *h1, double *eri, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, double *hdiag, int ndet, double *ci1) {

    size_t ip, jp, p;
    int nset = norb / 64 + 1;

    printf("Number of orbitals:         %d\n", norb);
    printf("Number of determinants:     %d\n", ndet);
    printf("Number of string sets:      %d\n", nset);
    printf("Number of alpha electrons:  %d\n", neleca);
    printf("Number of beta electrons:   %d\n", nelecb);

    // Loop over pairs of determinants
    for (ip = 0; ip < ndet; ++ip) {
        uint64_t *stria = strs + ip * 2 * nset;
        uint64_t *strib = strs + ip * 2 * nset + nset;
        for (jp = 0; jp < ip; ++jp) {
            uint64_t *strja = strs + jp * 2 * nset;
            uint64_t *strjb = strs + jp * 2 * nset + nset;
            int n_excit_a = n_excitations(stria, strja, nset);
            int n_excit_b = n_excitations(strib, strjb, nset);
//            printf("%d %d %d %d %d %d\n", stria[0], strib[0], strja[0], strjb[0], n_excit_a, n_excit_b);
//            printf("%s %s %s %s\n", int2bin(stria[0]), int2bin(strib[0]), int2bin(strja[0]), int2bin(strjb[0]));
            // Single excitation
            if ((n_excit_a + n_excit_b) == 1) {
                int *ia;
                // alpha->alpha
                if (n_excit_b == 0) {
                    ia = get_single_excitation(stria, strja, nset);
                    int i = ia[0];
                    int a = ia[1];
                    double sign = compute_cre_des_sign(a, i, stria, nset);
                    int *occsa = compute_occ_list(stria, nset, norb, neleca);
                    int *occsb = compute_occ_list(strib, nset, norb, nelecb);
 
 //                   // Test
 //                   printf("i: %d -> a: %d (%f)\n", ia[0], ia[1], sign);
 //                   for (p = 0; p < neleca; ++p) printf("%d ", occsa[p]);
 //                   printf("\n");
 //                   // Test

                    double fai = h1[a * norb + i];
                    for (p = 0; p < neleca; ++p) {
                        int k = occsa[p];
                        int kkai = k * norb * norb * norb + k * norb * norb + a * norb + i;
                        int kiak = k * norb * norb * norb + i * norb * norb + a * norb + k;
                        fai += eri[kkai] - eri[kiak];
                    }
                    for (p = 0; p < nelecb; ++p) {
                        int k = occsb[p];
                        int kkai = k * norb * norb * norb + k * norb * norb + a * norb + i;
                        fai += eri[kkai];
                    }

                    ci1[jp] += sign * fai * civec[ip];
                    ci1[ip] += sign * fai * civec[jp];
                }
                // beta->beta
                else if (n_excit_a == 0) {
                    ia = get_single_excitation(strib, strjb, nset);
                    int i = ia[0];
                    int a = ia[1];
                    double sign = compute_cre_des_sign(a, i, strib, nset);
                    int *occsa = compute_occ_list(stria, nset, norb, neleca);
                    int *occsb = compute_occ_list(strib, nset, norb, nelecb);

 //                   // Test
 //                   printf("i: %d -> a: %d (%f)\n", ia[0], ia[1], sign);
 //                   for (p = 0; p < nelecb; ++p) printf("%d ", occsb[p]);
 //                   printf("\n");
 //                   // Test
                    
                    double fai = h1[a * norb + i];
                    for (p = 0; p < nelecb; ++p) {
                        int k = occsb[p];
                        int kkai = k * norb * norb * norb + k * norb * norb + a * norb + i;
                        int kiak = k * norb * norb * norb + i * norb * norb + a * norb + k;
                        fai += eri[kkai] - eri[kiak];
                    }
                    for (p = 0; p < neleca; ++p) {
                        int k = occsa[p];
                        int kkai = k * norb * norb * norb + k * norb * norb + a * norb + i;
                        fai += eri[kkai];
                    }

                    ci1[jp] += sign * fai * civec[ip];
                    ci1[ip] += sign * fai * civec[jp];
                }
            }
            // Double excitation
            else if ((n_excit_a + n_excit_b) == 2) {
                int i, j, a, b;
                // alpha,alpha->alpha,alpha
                if (n_excit_b == 0) {
	            int *ijab = get_double_excitation(stria, strja, nset);
                    i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
//                    printf("(Case 1) i: %d j: %d -> a: %d b: %d\n", i, j, a, b);

                    double v, sign;
                    int ajbi = a * norb * norb * norb + j * norb * norb + b * norb + i;
                    int aibj = a * norb * norb * norb + i * norb * norb + b * norb + j;
                    if (a > j || i > b) {
                        v = eri[ajbi] - eri[aibj];
                        sign = compute_cre_des_sign(b, i, stria, nset);
                        sign *= compute_cre_des_sign(a, j, stria, nset);
                    } 
                    else {
                        v = eri[aibj] - eri[ajbi];
                        sign = compute_cre_des_sign(b, j, stria, nset);
                        sign *= compute_cre_des_sign(a, i, stria, nset);
                    }

                    ci1[jp] += sign * v * civec[ip];
                    ci1[ip] += sign * v * civec[jp];
//                    printf("(Case 1) v: %f\n", v);
                }
                // beta,beta->beta,beta
                else if (n_excit_a == 0) {
	            int *ijab = get_double_excitation(strib, strjb, nset);
                    i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
//                    printf("(Case 2) i: %d j: %d -> a: %d b: %d\n", i, j, a, b);

                    double v, sign;
                    int ajbi = a * norb * norb * norb + j * norb * norb + b * norb + i;
                    int aibj = a * norb * norb * norb + i * norb * norb + b * norb + j;
                    if (a > j || i > b) {
                        v = eri[ajbi] - eri[aibj];
                        sign = compute_cre_des_sign(b, i, strib, nset);
                        sign *= compute_cre_des_sign(a, j, strib, nset);
                    } 
                    else {
                        v = eri[aibj] - eri[ajbi];
                        sign = compute_cre_des_sign(b, j, strib, nset);
                        sign *= compute_cre_des_sign(a, i, strib, nset);
                    }

                    ci1[jp] += sign * v * civec[ip];
                    ci1[ip] += sign * v * civec[jp];
                }
                // alpha,beta->alpha,beta
                else {
                    int *ia = get_single_excitation(stria, strja, nset);
                    int *jb = get_single_excitation(strib, strjb, nset);
                    i = ia[0]; a = ia[1]; j = jb[0]; b = jb[1];

                    double v = eri[a * norb * norb * norb + i * norb * norb + b * norb + j];
                    double sign = compute_cre_des_sign(a, i, stria, nset);
                    sign *= compute_cre_des_sign(b, j, strib, nset);

                    ci1[jp] += sign * v * civec[ip];
                    ci1[ip] += sign * v * civec[jp];
//                    printf("(Case 3) i: %d j: %d -> a: %d b: %d\n", i, j, a, b);
                }
            }
//            printf("%20.10f %20.10f\n", ci1[ip], ci1[jp]);
        } // end loop over jp
        // Add diagonal elements
        ci1[ip] += hdiag[ip] * civec[ip];
    }

}


int n_excitations(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int d = 0;

    for (p = 0; p < nset; ++p) {
        d += popcount(str1[p] ^ str2[p]);
    }

    return d / 2;

}


int popcount(uint64_t x) {

    const uint64_t m1  = 0x5555555555555555; //binary: 0101...
    const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    const uint64_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
    const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
    const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
//    const uint64_t hff = 0xffffffffffffffff; //binary: all ones
//    const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
    x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
    x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
    x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
    x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
    x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
    x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 

    return x;

}


int *get_single_excitation(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int *ia = malloc(sizeof(int) * 2);

    for (p = 0; p < nset; ++p) {
        uint64_t str_tmp = str1[p] ^ str2[p];
        uint64_t str_particle = str_tmp & str2[p];
        uint64_t str_hole = str_tmp & str1[p];

        if (popcount(str_particle) == 1) {
            ia[1] = trailz(str_particle) + 64 * p;
        }
       
        if (popcount(str_hole) == 1) {
            ia[0] = trailz(str_hole) + 64 * p;
        }
    }

    return ia;

}


int *get_double_excitation(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int *ijab = malloc(sizeof(int) * 4);
    int particle_ind = 2;
    int hole_ind = 0;

    for (p = 0; p < nset; ++p) {
        uint64_t str_tmp = str1[p] ^ str2[p];
        uint64_t str_particle = str_tmp & str2[p];
        uint64_t str_hole = str_tmp & str1[p];
        int n_particle = popcount(str_particle);
        int n_hole = popcount(str_hole);

        if (n_particle == 1) {
            ijab[particle_ind] = trailz(str_particle) + 64 * p;
            particle_ind++;
        }
        else if (n_particle == 2) {
            int a = trailz(str_particle);
            ijab[2] = a + 64 * p;
            str_particle &= ~(1 << a);
            int b = trailz(str_particle);
            ijab[3] = b + 64 * p;
        }
       
        if (n_hole == 1) {
            ijab[hole_ind] = trailz(str_hole) + 64 * p;
            hole_ind++;
        }
        else if (n_hole == 2) {
            int i = trailz(str_hole);
            ijab[0] = i + 64 * p;
            str_hole &= ~(1 << i);
            int j = trailz(str_hole);
            ijab[1] = j + 64 * p;
        }
    }

    return ijab;
}


int trailz(uint64_t v) {

    int c = 64;

    v &= -(signed) v;
    if (v) c--;
    if (v & 0x00000000ffffffff) c -= 32;
    if (v & 0x0000ffff0000ffff) c -= 16;
    if (v & 0x00ff00ff00ff00ff) c -= 8;
    if (v & 0x0f0f0f0f0f0f0f0f) c -= 4;
    if (v & 0x3333333333333333) c -= 2;
    if (v & 0x5555555555555555) c -= 1;

    return c;
}


// Function to print int as a char for debug purposes
char *int2bin(int i) {
    size_t bits = sizeof(int) * CHAR_BIT;

    char * str = malloc(bits + 1);
    if(!str) return NULL;
    str[bits] = 0;

    // type punning because signed shift is implementation-defined
    unsigned u = *(unsigned *)&i;
    for(; bits--; u >>= 1)
        str[bits] = u & 1 ? '1' : '0';

    return str;
}


double compute_cre_des_sign(int p, int q, uint64_t *str, int nset) {

    double sign;
    int nperm;
    size_t i;

    int pg = p / 64;
    int qg = q / 64;
    int pb = p % 64;
    int qb = q % 64;

    if (pg > qg) {
        nperm = 0;
        for (i = nset-pg; i < nset-qg-1; ++i) {
            nperm += popcount(str[i]);
        }
        nperm += popcount(str[-1 - pg] & ((1 << pb) - 1));
        nperm += str[-1 - qg] >> (qb + 1);
    }
    else if (pg < qg) {
        nperm = 0;
        for (i = nset-qg; i < nset-pg-1; ++i) {
            nperm += popcount(str[i]);
        }
        nperm += popcount(str[-1 - qg] & ((1 << qb) - 1));
        nperm += str[-1 - pg] >> (pb + 1);
    }
    else {
        uint64_t mask;
        if (p > q) mask = (1 << pb) - (1 << (qb + 1));
        else       mask = (1 << qb) - (1 << (pb + 1));
        nperm = popcount(str[pg] & mask);
    }

    if (nperm % 2) sign = -1.0;
    else           sign = 1.0;

    return sign;

}


int *compute_occ_list(uint64_t *string, int nset, int norb, int nelec) {

    size_t k, i;

    int *occ = malloc(sizeof(int) * nelec);
    int off = 0;
    int occ_ind = 0;

    for (k = nset; k > 0; --k) {
        int i_max = ((norb - off) < 64 ? (norb - off) : 64);
        for (i = 0; i < i_max; ++i) {
            int i_occ = (string[k-1] >> i) & 1;
            if (i_occ) {
                occ[occ_ind] = i + off;
                occ_ind++;
            }
        }
        off += 64;
    }

    return occ;
    
}
