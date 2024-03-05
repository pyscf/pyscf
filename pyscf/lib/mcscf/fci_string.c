/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "fci.h"

/*
 * Hamming weight popcount
 */

int FCIpopcount_1(uint64_t x)
{
        const uint64_t m1  = 0x5555555555555555; //binary: 0101...
        const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
        const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
        const uint64_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
        const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
        const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
//        const uint64_t hff = 0xffffffffffffffff; //binary: all ones
//        const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
        x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
        x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
        x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
        x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
        x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
        x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 
        return x;
}

int FCIpopcount_4(uint64_t x)
{
        int count;
        for (count = 0; x; count++) {
                x &= x-1;
        }
        return count;
}


/*
 * sign of  a^+ a |string>
 */
int FCIcre_des_sign(int p, int q, uint64_t string0)
{
        uint64_t mask;
        if (p > q) {
                mask = (1ULL << p) - (1ULL << (q+1));
        } else {
                mask = (1ULL << q) - (1ULL << (p+1));
        }
        if (FCIpopcount_1(string0 & mask) % 2) {
                return -1;
        } else {
                return 1;
        }
}

int FCIcre_sign(int p, uint64_t string0)
{
        if (string0 & (1ULL<<p)) {
                return 0;
        } else if (FCIpopcount_1(string0 >> (p+1)) % 2) {
                return -1;
        } else {
                return 1;
        }
}

int FCIdes_sign(int p, uint64_t string0)
{
        if (!(string0 & (1ULL<<p))) {
                return 0;
        } else if (FCIpopcount_1(string0 >> (p+1)) % 2) {
                return -1;
        } else {
                return 1;
        }
}

// [math.comb(n, m) for n in range(1, 21) for m in range(n)]
static int _binomial_cache[] = {
        1,
        1, 2,
        1, 3, 3,
        1, 4, 6, 4,
        1, 5, 10, 10, 5,
        1, 6, 15, 20, 15, 6,
        1, 7, 21, 35, 35, 21, 7,
        1, 8, 28, 56, 70, 56, 28, 8,
        1, 9, 36, 84, 126, 126, 84, 36, 9,
        1, 10, 45, 120, 210, 252, 210, 120, 45, 10,
        1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11,
        1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12,
        1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13,
        1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14,
        1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15,
        1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16,
        1, 17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376, 6188, 2380, 680, 136, 17,
        1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, 31824, 18564, 8568, 3060, 816, 153, 18,
        1, 19, 171, 969, 3876, 11628, 27132, 50388, 75582, 92378, 92378, 75582, 50388, 27132, 11628, 3876, 969, 171, 19,
        1, 20, 190, 1140, 4845, 15504, 38760, 77520, 125970, 167960, 184756, 167960, 125970, 77520, 38760, 15504, 4845, 1140, 190, 20,
};
static int binomial(int n, int m)
{
        if (m >= n) {
                return 1;
        } else if (n <= 20) {
                return _binomial_cache[n*(n-1)/2+m];
        } else {
                if (m*2 <= n) {
                        m = n - m;
                }
		int val = binomial(n-1,m-1) + binomial(n-1,m);
                return val;
        }
}

int FCIstr2addr(int norb, int nelec, uint64_t string)
{
        size_t addr = 0;
        int nelec_left = nelec;
        int norb_left;
        for (norb_left = norb - 1; norb_left >= 0; norb_left--) {
                if (nelec_left == 0 || norb_left < nelec_left) {
                        break;
                } else if ((1ULL<<norb_left) & string) {
                        addr += binomial(norb_left, nelec_left);
                        nelec_left--;
                }
        }
        return addr;
}
void FCIstrs2addr(int *addrs, uint64_t *strings, int count, int norb, int nelec)
{
        int i, norb_left, nelec_left;
        size_t addr;
        uint64_t nextaddr0 = binomial(norb-1, nelec);
        uint64_t nextaddr, str;
        for (i = 0; i < count; i++) {
                str = strings[i];
                addr = 0;
                nelec_left = nelec;
                nextaddr = nextaddr0;
                for (norb_left = norb - 1; norb_left >= 0; norb_left--) {
                        if (nelec_left == 0 || norb_left < nelec_left) {
                                break;
                        } else if ((1ULL<<norb_left) & str) {
                                assert(nextaddr == binomial(norb_left, nelec_left));
                                addr += nextaddr;
                                nextaddr *= nelec_left;
                                nextaddr /= norb_left;
                                nelec_left--;
                        } else {
                                nextaddr *= norb_left - nelec_left;
                                nextaddr /= norb_left;
                        }
                }
                addrs[i] = addr;
        }
}

void FCIaddrs2str(uint64_t *strings, int *addrs, int count, int norb, int nelec)
{
        int i, nelec_left, norb_left, addr;
        uint64_t nextaddr0 = binomial(norb-1, nelec);
        uint64_t nextaddr, str1;
        for (i = 0; i < count; i++) {
                addr = addrs[i];
                if (addr == 0 || nelec == norb || nelec == 0) {
                        strings[i] = (1UL << nelec) - 1UL;
                        continue;
                }

                str1 = 0;
                nelec_left = nelec;
                nextaddr = nextaddr0;
                for (norb_left = norb-1; norb_left >= 0; norb_left--) {
                        assert(nextaddr == binomial(norb_left, nelec_left));
                        if (nelec_left == 0) {
                                break;
                        } else if (addr == 0) {
                                str1 |= (1UL << nelec_left) - 1UL;
                                break;
                        } else if (nextaddr <= addr) {
                                str1 |= 1UL << norb_left;
                                addr -= nextaddr;
                                nextaddr *= nelec_left;
                                nextaddr /= norb_left;
                                nelec_left--;
                        } else {
                                nextaddr *= norb_left - nelec_left;
                                nextaddr /= norb_left;
                        }
                }
                strings[i] = str1;
        }
}

// [cre, des, target_address, parity]
void FCIlinkstr_index(int *link_index, int norb, int na, int nocc,
                      uint64_t *strs, int store_trilidx)
{
        int occ[norb];
        int vir[norb];
        int nvir = norb - nocc;
        int nlink = nocc * nvir + nocc;
        int str_id, io, iv, i, a, k, ia;
        uint64_t str0, str1;
        uint64_t str1s[nocc*nvir];
        int addrbuf[nocc*nvir];
        int *tab;

        for (str_id = 0; str_id < na; str_id++) {
                str0 = strs[str_id];
                for (i = 0, io = 0, iv = 0; i < norb; i++) {
                        if (str0 & (1ULL<<i)) {
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
                                str1 = (str0^(1ULL<<occ[i])) | (1ULL<<vir[a]);
                                str1s[k-nocc] = str1;
                                if (vir[a] > occ[i]) {
                                        ia = vir[a]*(vir[a]+1)/2+occ[i];
                                } else {
                                        ia = occ[i]*(occ[i]+1)/2+vir[a];
                                }
                                tab[k*4+0] = ia;
                                //tab[k*4+2] = FCIstr2addr(norb, nocc, str1);
                                tab[k*4+3] = FCIcre_des_sign(vir[a], occ[i], str0);
                        } }
                        FCIstrs2addr(addrbuf, str1s, nocc*nvir, norb, nocc);
                        for (k = 0; k < nocc*nvir; k++) {
                                tab[(k+nocc)*4+2] = addrbuf[k];
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
                                str1 = (str0^(1ULL<<occ[i])) | (1ULL<<vir[a]);
                                str1s[k-nocc] = str1;
                                tab[k*4+0] = vir[a];
                                tab[k*4+1] = occ[i];
                                //tab[k*4+2] = FCIstr2addr(norb, nocc, str1);
                                tab[k*4+3] = FCIcre_des_sign(vir[a], occ[i], str0);
                        } }
                        FCIstrs2addr(addrbuf, str1s, nocc*nvir, norb, nocc);
                        for (k = 0; k < nocc*nvir; k++) {
                                tab[(k+nocc)*4+2] = addrbuf[k];
                        }
                }
        }
}

// [cre, des, target_address, parity]
void FCIcre_str_index(int *link_index, int norb, int na, int nocc,
                      uint64_t *strs)
{
        int nvir = norb - nocc;
        int str_id, i, k;
        uint64_t str0, str1;
        int *tab = link_index;

        for (str_id = 0; str_id < na; str_id++) {
                str0 = strs[str_id];
                k = 0;
                for (i = 0; i < norb; i++) {
                        if (!(str0 & (1ULL<<i))) {
                                str1 = str0 | (1ULL<<i);
                                tab[k*4+0] = i;
                                tab[k*4+1] = 0;
                                tab[k*4+2] = FCIstr2addr(norb, nocc+1, str1);
                                if (FCIpopcount_1(str0>>(i+1)) % 2) {
                                        tab[k*4+3] = -1;
                                } else {
                                        tab[k*4+3] = 1;
                                }
                                k++;
                        }
                }
                tab += nvir * 4;
        }
}

// [cre, des, target_address, parity]
void FCIdes_str_index(int *link_index, int norb, int na, int nocc,
                      uint64_t *strs)
{
        int str_id, i, k;
        uint64_t str0, str1;
        int *tab = link_index;

        for (str_id = 0; str_id < na; str_id++) {
                str0 = strs[str_id];
                k = 0;
                for (i = 0; i < norb; i++) {
                        if (str0 & (1ULL<<i)) {
                                str1 = str0 ^ (1ULL<<i);
                                tab[k*4+0] = 0;
                                tab[k*4+1] = i;
                                tab[k*4+2] = FCIstr2addr(norb, nocc-1, str1);
                                if (FCIpopcount_1(str0>>(i+1)) % 2) {
                                        tab[k*4+3] = -1;
                                } else {
                                        tab[k*4+3] = 1;
                                }
                                k++;
                        }
                }
                tab += nocc * 4;
        }
}

/*
 ***********************************************************
 */

void FCIcompress_link(_LinkT *clink, int *link_index,
                      int norb, int nstr, int nlink)
{
        int j, k;
        for (k = 0; k < nstr; k++) {
                for (j = 0; j < nlink; j++) {
                        clink[j].a    = link_index[j*4+0];
                        clink[j].i    = link_index[j*4+1];
                        clink[j].addr = link_index[j*4+2];
                        clink[j].sign = link_index[j*4+3];
                }
                clink += nlink;
                link_index += nlink * 4;
        }
}

void FCIcompress_link_tril(_LinkTrilT *clink, int *link_index,
                           int nstr, int nlink)
{
        int i, j;
        for (i = 0; i < nstr; i++) {
                for (j = 0; j < nlink; j++) {
                        clink[j].ia   = link_index[j*4+0];
                        clink[j].addr = link_index[j*4+2];
                        clink[j].sign = link_index[j*4+3];
                }
                clink += nlink;
                link_index += nlink * 4;
        }
}
