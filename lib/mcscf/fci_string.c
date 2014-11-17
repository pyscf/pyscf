/*
 * File: fci_string.c
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))


/*
 * Hamming weight popcount
 */

int FCIpopcount_1(unsigned long x)
{
        const unsigned long m1  = 0x5555555555555555; //binary: 0101...
        const unsigned long m2  = 0x3333333333333333; //binary: 00110011..
        const unsigned long m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
        const unsigned long m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
        const unsigned long m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
        const unsigned long m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
//        const unsigned long hff = 0xffffffffffffffff; //binary: all ones
//        const unsigned long h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
        x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
        x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
        x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
        x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
        x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
        x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 
        return x;
}

int FCIpopcount_4(unsigned long x)
{
        int count;
        for (count = 0; x; count++) {
                x &= x-1;
        }
        return count;
}


int FCIparity(unsigned long string0, unsigned long string1)
{
        unsigned long ss;
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
        unsigned long num = 1;
        unsigned long div = 1;
        for (i = 0; i < n-m; i++) {
                num *= m+i+1;
        }
        for (i = 0; i < n-m; i++) {
                div *= i+1;
        }
        return num / div;
}

int FCIstr2addr(int norb, int nelec, unsigned long string)
{
        int addr = 0;
        int nelec_left = nelec;
        int norb_left;
        for (norb_left = norb - 1; norb_left >= 0; norb_left--) {
                if (nelec_left == 0 || norb_left < nelec_left) {
                        break;
                } else if ((1<<norb_left) & string) {
                        addr += binomial(norb_left, nelec_left);
                        nelec_left--;
                }
        }
        return addr;
}


void FCIlinkstr_index(int *link_index, int norb, int na, int nocc,
                      unsigned long *strs, int store_trilidx)
{
        int occ[norb];
        int vir[norb];
        int nvir = norb - nocc;
        int nlink = nocc * nvir + nocc;
        int str_id, io, iv, i, a, k, ia;
        unsigned long str0, str1;
        int *tab;

        for (str_id = 0; str_id < na; str_id++) {
                str0 = strs[str_id];
                for (i = 0, io = 0, iv = 0; i < norb; i++) {
                        if (str0 & (1<<i)) {
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
                                        str1 = (str0^(1<<occ[i])) | (1<<vir[a]);
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
                                        str1 = (str0^(1<<occ[i])) | (1<<vir[a]);
                                        tab[k*4+0] = vir[a];
                                        tab[k*4+1] = occ[i];
                                        tab[k*4+2] = FCIstr2addr(norb, nocc, str1);
                                        tab[k*4+3] = FCIparity(str1, str0);
                                }
                        }
                }
        }
}
