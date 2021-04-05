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

static int binomial(int n, int m)
{
        int i;
        if (m >= n) {
                return 1;
        } else if (n < 28) {
                uint64_t num = 1;
                uint64_t div = 1;
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
                double dnum = 1;
                double ddiv = 1;
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
