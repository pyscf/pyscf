/*
 * File: fci_string.c
 *
 */

#include <stdlib.h>
#include <string.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"


/*
 * Hamming weight popcount
 */

int FCIpopcount_1(size_t x)
{
        const size_t m1  = 0x5555555555555555; //binary: 0101...
        const size_t m2  = 0x3333333333333333; //binary: 00110011..
        const size_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
        const size_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
        const size_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
        const size_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
//        const size_t hff = 0xffffffffffffffff; //binary: all ones
//        const size_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
        x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
        x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
        x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
        x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
        x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
        x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 
        return x;
}

int FCIpopcount_4(size_t x)
{
        int count;
        for (count = 0; x; count++) {
                x &= x-1;
        }
        return count;
}


int FCIparity(size_t string0, size_t string1)
{
        size_t ss;
        if (string1 > string0) {
                ss = string1 - string0;
                // string1&ss gives the number of 1s between two strings
                if (FCIpopcount_1(string1&ss) % 2) {
                        return -1;
                } else {
                        return 1;
                }
        } else if (string1 == string0) {
                return 1;
        } else {
                ss = string0 - string1;
                if (FCIpopcount_1(string0&ss) % 2) {
                        return -1;
                } else {
                        return 1;
                }
        }
}

static int binomial(int n, int m)
{
        int i;
        size_t num = 1;
        size_t div = 1;
        double dnum = 1;
        double ddiv = 1;
        if (n < 28) {
                if (m+m >= n) {
                        for (i = 0; i < n-m; i++) {
                                num *= m+i+1;
                                div *= i+1;
                        }
                } else {
                        for (i = 0; i < m; i++) {
                                num *= (n-m)+i+1;
                                div *= i+1;
                        }
                }
                return num / div;
        } else {
                if (m+m >= n) {
                        for (i = 0; i < n-m; i++) {
                                dnum *= m+i+1;
                                ddiv *= i+1;
                        }
                } else {
                        for (i = 0; i < m; i++) {
                                dnum *= (n-m)+i+1;
                                ddiv *= i+1;
                        }
                }
                return (int)(dnum / ddiv);
        }
}

int FCIstr2addr(int norb, int nelec, size_t string)
{
        int addr = 0;
        int nelec_left = nelec;
        int norb_left;
        for (norb_left = norb - 1; norb_left >= 0; norb_left--) {
                if (nelec_left == 0 || norb_left < nelec_left) {
                        break;
                } else if ((1UL<<norb_left) & string) {
                        addr += binomial(norb_left, nelec_left);
                        nelec_left--;
                }
        }
        return addr;
}


void FCIlinkstr_index(int *link_index, int norb, int na, int nocc,
                      size_t *strs, int store_trilidx)
{
        int occ[norb];
        int vir[norb];
        int nvir = norb - nocc;
        int nlink = nocc * nvir + nocc;
        int str_id, io, iv, i, a, k, ia;
        size_t str0, str1;
        int *tab;

        for (str_id = 0; str_id < na; str_id++) {
                str0 = strs[str_id];
                for (i = 0, io = 0, iv = 0; i < norb; i++) {
                        if (str0 & (1UL<<i)) {
                                occ[io] = i;
                                io += 1;
                        } else {
                                vir[iv] = i;
                                iv += 1;
                        }
                }

                tab = link_index + str_id * nlink * 4;
                if (store_trilidx) {
                        for (k = 0; k < nocc; k++) {
                                tab[k*4+0] = occ[k]*(occ[k]+1)/2+occ[k];
                                tab[k*4+2] = str_id;
                                tab[k*4+3] = 1;
                        }
                        for (i = 0; i < nocc; i++) {
                                for (a = 0; a < nvir; a++, k++) {
                                        str1 = (str0^(1UL<<occ[i])) |
                                                (1UL<<vir[a]);
                                        if (vir[a] > occ[i]) {
                                                ia = vir[a]*(vir[a]+1)/2+occ[i];
                                        } else {
                                                ia = occ[i]*(occ[i]+1)/2+vir[a];
                                        }
                                        tab[k*4+0] = ia;
                                        tab[k*4+2] = FCIstr2addr(norb, nocc, str1);
                                        tab[k*4+3] = FCIparity(str1, str0);
                                }
                        }

                } else {

                        for (k = 0; k < nocc; k++) {
                                tab[k*4+0] = occ[k];
                                tab[k*4+1] = occ[k];
                                tab[k*4+2] = str_id;
                                tab[k*4+3] = 1;
                        }

                        for (i = 0; i < nocc; i++) {
                                for (a = 0; a < nvir; a++, k++) {
                                        str1 = (str0^(1UL<<occ[i])) |
                                                (1UL<<vir[a]);
                                        tab[k*4+0] = vir[a];
                                        tab[k*4+1] = occ[i];
                                        tab[k*4+2] = FCIstr2addr(norb, nocc, str1);
                                        tab[k*4+3] = FCIparity(str1, str0);
                                }
                        }
                }
        }
}
