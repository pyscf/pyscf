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
 * Author: Alexander Sokolov <alexander.y.sokolov@gmail.com>
 *
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

// Computes C' = H * C in the selected CI basis
void contract_h_c(double *h1, double *eri, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, double *hdiag, uint64_t ndet, double *ci1) {

    int *ts = malloc(sizeof(int) * ndet);

    #pragma omp parallel
    {

    size_t ip, jp, p;
    int nset = (norb + 63) / 64;
 
    // Calculate excitation level for prescreening
    ts[0] = 0;
    uint64_t *str1a = strs;
    uint64_t *str1b = strs + nset;
    #pragma omp for schedule(static)
    for (ip = 1; ip < ndet; ++ip) {
        uint64_t *stria = strs + ip * 2 * nset;
        uint64_t *strib = strs + ip * 2 * nset + nset;
        ts[ip] = (n_excitations(stria, str1a, nset) + n_excitations(strib, str1b, nset));
    }

    // Loop over pairs of determinants
    #pragma omp for schedule(static)
    for (ip = 0; ip < ndet; ++ip) {
        for (jp = 0; jp < ndet; ++jp) {
            if (abs(ts[ip] - ts[jp]) < 3) {
                uint64_t *stria = strs + ip * 2 * nset;
                uint64_t *strib = strs + ip * 2 * nset + nset;
                uint64_t *strja = strs + jp * 2 * nset;
                uint64_t *strjb = strs + jp * 2 * nset + nset;
                int n_excit_a = n_excitations(stria, strja, nset);
                int n_excit_b = n_excitations(strib, strjb, nset);
                // Diagonal term
                if (ip == jp) {
                    ci1[ip] += hdiag[ip] * civec[ip];
                }
                // Single excitation
                else if ((n_excit_a + n_excit_b) == 1) {
                    int *ia;
                    // alpha->alpha
                    if (n_excit_b == 0) {
                        ia = get_single_excitation(stria, strja, nset);
                        int i = ia[0];
                        int a = ia[1];
                        double sign = compute_cre_des_sign(a, i, stria, nset);
                        int *occsa = compute_occ_list(stria, nset, norb, neleca);
                        int *occsb = compute_occ_list(strib, nset, norb, nelecb);
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
                        if (fabs(fai) > 1.0E-14) ci1[ip] += sign * fai * civec[jp];
                        free(occsa);
                        free(occsb);
                    }
                    // beta->beta
                    else if (n_excit_a == 0) {
                        ia = get_single_excitation(strib, strjb, nset);
                        int i = ia[0];
                        int a = ia[1];
                        double sign = compute_cre_des_sign(a, i, strib, nset);
                        int *occsa = compute_occ_list(stria, nset, norb, neleca);
                        int *occsb = compute_occ_list(strib, nset, norb, nelecb);
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
                        if (fabs(fai) > 1.0E-14) ci1[ip] += sign * fai * civec[jp];
                        free(occsa);
                        free(occsb);
                    }
                   free(ia);
                }
                // Double excitation
                else if ((n_excit_a + n_excit_b) == 2) {
                    int i, j, a, b;
                    // alpha,alpha->alpha,alpha
                    if (n_excit_b == 0) {
	                int *ijab = get_double_excitation(stria, strja, nset);
                        i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
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
                        if (fabs(v) > 1.0E-14) ci1[ip] += sign * v * civec[jp];
                        free(ijab);
                    }
                    // beta,beta->beta,beta
                    else if (n_excit_a == 0) {
	                int *ijab = get_double_excitation(strib, strjb, nset);
                        i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
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
                        if (fabs(v) > 1.0E-14) ci1[ip] += sign * v * civec[jp];
                        free(ijab);
                    }
                    // alpha,beta->alpha,beta
                    else {
                        int *ia = get_single_excitation(stria, strja, nset);
                        int *jb = get_single_excitation(strib, strjb, nset);
                        i = ia[0]; a = ia[1]; j = jb[0]; b = jb[1];
                        double v = eri[a * norb * norb * norb + i * norb * norb + b * norb + j];
                        double sign = compute_cre_des_sign(a, i, stria, nset);
                        sign *= compute_cre_des_sign(b, j, strib, nset);
                        if (fabs(v) > 1.0E-14) ci1[ip] += sign * v * civec[jp];
                        free(ia);
                        free(jb);
                   }
                }
            } // end if over ts
        } // end loop over jp
    } // end loop over ip

    } // end omp
  
    free(ts);

}

// Compare two strings and compute excitation level
int n_excitations(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int d = 0;

    for (p = 0; p < nset; ++p) {
        d += popcount(str1[p] ^ str2[p]);
    }

    return d / 2;

}

// Compute number of set bits in a string
int popcount(uint64_t x) {

    const uint64_t m1  = 0x5555555555555555; //binary: 0101...
    const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    const uint64_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
    const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
    const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
    x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
    x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
    x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
    x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
    x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
    x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 

    return x;

}

// Compute orbital indices for a single excitation 
int *get_single_excitation(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int *ia = malloc(sizeof(int) * 2);

    for (p = 0; p < nset; ++p) {
        size_t pp = nset - p - 1;
        uint64_t str_tmp = str1[pp] ^ str2[pp];
        uint64_t str_particle = str_tmp & str2[pp];
        uint64_t str_hole = str_tmp & str1[pp];

        if (popcount(str_particle) == 1) {
            ia[1] = trailz(str_particle) + 64 * p;
        }
       
        if (popcount(str_hole) == 1) {
            ia[0] = trailz(str_hole) + 64 * p;
        }
    }

    return ia;

}

// Compute orbital indices for a double excitation 
int *get_double_excitation(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int *ijab = malloc(sizeof(int) * 4);
    int particle_ind = 2;
    int hole_ind = 0;

    for (p = 0; p < nset; ++p) {
        size_t pp = nset - p - 1;
        uint64_t str_tmp = str1[pp] ^ str2[pp];
        uint64_t str_particle = str_tmp & str2[pp];
        uint64_t str_hole = str_tmp & str1[pp];
        int n_particle = popcount(str_particle);
        int n_hole = popcount(str_hole);

        if (n_particle == 1) {
            ijab[particle_ind] = trailz(str_particle) + 64 * p;
            particle_ind++;
        }
        else if (n_particle == 2) {
            int a = trailz(str_particle);
            ijab[2] = a + 64 * p;
            str_particle &= ~(1ULL << a);
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
            str_hole &= ~(1ULL << i);
            int j = trailz(str_hole);
            ijab[1] = j + 64 * p;
        }
    }

    return ijab;

}

// Compute number of trailing zeros in a bit string
int trailz(uint64_t v) {

    int c = 64;

    // Trick to unset all bits but the first one
    v &= -(int64_t) v;
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
char *int2bin(uint64_t i) {

    size_t bits = sizeof(uint64_t) * CHAR_BIT;

    char * str = malloc(bits + 1);
    if(!str) return NULL;
    str[bits] = 0;

    // type punning because signed shift is implementation-defined
    uint64_t u = *(uint64_t *)&i;
    for(; bits--; u >>= 1)
        str[bits] = u & 1 ? '1' : '0';

    return str;

}

// Compute sign for a pair of creation and desctruction operators
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
        nperm += popcount(str[nset -1 - pg] & ((1ULL << pb) - 1));
        nperm += str[nset -1 - qg] >> (qb + 1);
    }
    else if (pg < qg) {
        nperm = 0;
        for (i = nset-qg; i < nset-pg-1; ++i) {
            nperm += popcount(str[i]);
        }
        nperm += popcount(str[nset -1 - qg] & ((1ULL << qb) - 1));
        nperm += str[nset -1 - pg] >> (pb + 1);
    }
    else {
        uint64_t mask;
        if (p > q) mask = (1ULL << pb) - (1ULL << (qb + 1));
        else       mask = (1ULL << qb) - (1ULL << (pb + 1));
        nperm = popcount(str[nset -1 - pg] & mask);
    }

    if (nperm % 2) sign = -1.0;
    else           sign = 1.0;

    return sign;

}

// Compute a list of occupied orbitals for a given string
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


// Compute a list of occupied orbitals for a given string
int *compute_vir_list(uint64_t *string, int nset, int norb, int nelec) {

    size_t k, i;

    int *vir = malloc(sizeof(int) * (norb-nelec));
    int off = 0;
    int vir_ind = 0;

    for (k = nset; k > 0; --k) {
        int i_max = ((norb - off) < 64 ? (norb - off) : 64);
        for (i = 0; i < i_max; ++i) {
            int i_occ = (string[k-1] >> i) & 1;
            if (!i_occ) {
                vir[vir_ind] = i + off;
                vir_ind++;
            }
        }
        off += 64;
    }

    return vir;
    
}

// Select determinants to include in the CI space
void select_strs(double *h1, double *eri, double *jk, uint64_t *eri_sorted, uint64_t *jk_sorted, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, uint64_t ndet_start, uint64_t ndet_finish, double select_cutoff, uint64_t *strs_add, uint64_t* strs_add_size) {

    size_t p, q, r, i, k, a, ip, jp, kp, lp, ij, iset, idet;

    uint64_t max_strs_add = strs_add_size[0];
    int nset = (norb + 63) / 64;

    // Compute Fock intermediates
    double *focka = malloc(sizeof(double) * norb * norb);
    double *fockb = malloc(sizeof(double) * norb * norb);
    for (p = 0; p < norb; ++p) {
        for (q = 0; q < norb; ++q) {
            double vja = 0.0;
            double vka = 0.0;
            for (i = 0; i < neleca; ++i) {
                size_t iipq = i * norb * norb * norb + i * norb * norb + p * norb + q;
                size_t piiq = p * norb * norb * norb + i * norb * norb + i * norb + q;
                vja += eri[iipq];
                vka += eri[piiq];
            }
            double vjb = 0.0;
            double vkb = 0.0;
            for (i = 0; i < nelecb; ++i) {
                size_t iipq = i * norb * norb * norb + i * norb * norb + p * norb + q;
                size_t piiq = p * norb * norb * norb + i * norb * norb + i * norb + q;
                vjb += eri[iipq];
                vkb += eri[piiq];
            }
            focka[p * norb + q] = h1[p * norb + q] + vja + vjb - vka;
            fockb[p * norb + q] = h1[p * norb + q] + vja + vjb - vkb;
        }
    }

    int *holes_a = malloc(sizeof(int) * norb);
    int *holes_b = malloc(sizeof(int) * norb);
    int *particles_a = malloc(sizeof(int) * norb);
    int *particles_b = malloc(sizeof(int) * norb);
    uint64_t strs_added = 0;

    // Loop over determinants
    for (idet = ndet_start; idet < ndet_finish; ++idet) {
        uint64_t *stra = strs + idet * 2 * nset;
        uint64_t *strb = strs + idet * 2 * nset + nset;
        int *occsa = compute_occ_list(stra, nset, norb, neleca);
        int *occsb = compute_occ_list(strb, nset, norb, nelecb);
        int *virsa = compute_vir_list(stra, nset, norb, neleca);
        int *virsb = compute_vir_list(strb, nset, norb, nelecb);
        double tol = select_cutoff / fabs(civec[idet]);

        // Single excitations
        int n_holes_a = 0;
        int n_holes_b = 0;
	int n_particles_a = 0;
	int n_particles_b = 0;
        for (p = 0; p < (norb - neleca); ++p) {
            i = virsa[p];
            if (i < neleca) {
                holes_a[n_holes_a] = i;
                n_holes_a++;
            }
        }
        for (p = 0; p < neleca; ++p) {
            i = occsa[p];
            if (i >= neleca) {
                particles_a[n_particles_a] = i;
                n_particles_a++;
            }
        }
        for (p = 0; p < (norb - nelecb); ++p) {
            i = virsb[p];
            if (i < nelecb) {
                holes_b[n_holes_b] = i;
                n_holes_b++;
            }
        }
        for (p = 0; p < nelecb; ++p) {
            i = occsb[p];
            if (i >= nelecb) {
                particles_b[n_particles_b] = i;
                n_particles_b++;
            }
        }

        // TODO: recompute Fock for each |Phi_I> and make sure it matches Fock in the code below
        // alpha->alpha
        for (p = 0; p < neleca; ++p) {
            i = occsa[p];
            for (q = 0; q < (norb - neleca); ++q) {
                a = virsa[q];
                double fai = focka[a * norb + i];
                for (r = 0; r < n_particles_a; ++r) {
                    k = particles_a[r];
                    fai += jk[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                for (r = 0; r < n_holes_a; ++r) {
                    k = holes_a[r];
                    fai -= jk[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                for (r = 0; r < n_particles_b; ++r) {
                    k = particles_b[r];
                    fai += eri[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                for (r = 0; r < n_holes_b; ++r) {
                    k = holes_b[r];
                    fai -= eri[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                if (fabs(fai) > tol) {
                    uint64_t *tmp = toggle_bit(stra, nset, a);
                    uint64_t *new_str = toggle_bit(tmp, nset, i);
                    for (iset = 0; iset < nset; ++iset) {
                        // new alpha string
                        strs_add[strs_added * 2 * nset + iset] = new_str[iset];
                        // old beta string
                        strs_add[strs_added * 2 * nset + nset + iset] = strb[iset];
                    }
                    free(tmp);
                    free(new_str);
                    strs_added++;
                }
            }
        }

        // beta->beta
        for (p = 0; p < nelecb; ++p) {
            i = occsb[p];
            for (q = 0; q < (norb - nelecb); ++q) {
                a = virsb[q];
                double fai = fockb[a * norb + i];
                for (r = 0; r < n_particles_b; ++r) {
                    k = particles_b[r];
                    fai += jk[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                for (r = 0; r < n_holes_b; ++r) {
                    k = holes_b[r];
                    fai -= jk[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                for (r = 0; r < n_particles_a; ++r) {
                    k = particles_a[r];
                    fai += eri[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                for (r = 0; r < n_holes_a; ++r) {
                    k = holes_a[r];
                    fai -= eri[k * norb * norb * norb + k * norb * norb + a * norb + i];
                }
                if (fabs(fai) > tol) {
                    uint64_t *tmp = toggle_bit(strb, nset, a);
                    uint64_t *new_str = toggle_bit(tmp, nset, i);
                    for (iset = 0; iset < nset; ++iset) {
                        // old alpha string
                        strs_add[strs_added * 2 * nset + iset] = stra[iset];
                        // new beta string
                        strs_add[strs_added * 2 * nset + nset + iset] = new_str[iset];
                    }
                    free(tmp);
                    free(new_str);
                    strs_added++;
                }
            }
        }

        size_t ip_occ, jp_occ, kp_occ, lp_occ, ih;
        // Double excitations
        for (p = 0; p < norb * norb * norb * norb; ++p) {
            ih = jk_sorted[p];
            int aaaa_bbbb_done = (fabs(jk[ih]) < tol);
            if (!aaaa_bbbb_done) {
                lp = ih % norb;
                ij = ih / norb;
                kp = ij % norb;
                ij = ij / norb;
                jp = ij % norb;
                ip = ij / norb;
                // alpha,alpha->alpha,alpha
                ip_occ = 0;
                jp_occ = 0;
                kp_occ = 0;
                lp_occ = 0;
                for (r = 0; r < neleca; ++r) {
                    int occ_index = occsa[r];
                    if (ip == occ_index) ip_occ = 1;
                    if (jp == occ_index) jp_occ = 1;
                    if (kp == occ_index) kp_occ = 1;
                    if (lp == occ_index) lp_occ = 1;
                }
                if (jp_occ && lp_occ && !ip_occ && !kp_occ) {
                    uint64_t *tmp = toggle_bit(stra, nset, jp);
                    uint64_t *new_str = toggle_bit(tmp, nset, ip);
                    tmp = toggle_bit(new_str, nset, lp);
                    new_str = toggle_bit(tmp, nset, kp);
                    for (iset = 0; iset < nset; ++iset) {
                        strs_add[strs_added * 2 * nset + iset] = new_str[iset];
                        strs_add[strs_added * 2 * nset + nset + iset] = strb[iset];
                    }
                    free(tmp);
                    free(new_str);
                    strs_added++;
                }
                // beta,beta->beta,beta
                ip_occ = 0;
                jp_occ = 0;
                kp_occ = 0;
                lp_occ = 0;
                for (r = 0; r < nelecb; ++r) {
                    int occ_index = occsb[r];
                    if (ip == occ_index) ip_occ = 1;
                    if (jp == occ_index) jp_occ = 1;
                    if (kp == occ_index) kp_occ = 1;
                    if (lp == occ_index) lp_occ = 1;
                }
                if (jp_occ && lp_occ && !ip_occ && !kp_occ) {
                    uint64_t *tmp = toggle_bit(strb, nset, jp);
                    uint64_t *new_str = toggle_bit(tmp, nset, ip);
                    tmp = toggle_bit(new_str, nset, lp);
                    new_str = toggle_bit(tmp, nset, kp);
                    for (iset = 0; iset < nset; ++iset) {
                        strs_add[strs_added * 2 * nset + iset] = stra[iset];
                        strs_add[strs_added * 2 * nset + nset + iset] = new_str[iset];
                    }
                    free(tmp);
                    free(new_str);
                    strs_added++;
                }
            }
            // alpha,beta->alpha,beta
            ih = eri_sorted[p];
            int aabb_done = (fabs(eri[ih]) < tol);
            if (!aabb_done) {
                lp = ih % norb;
                ij = ih / norb;
                kp = ij % norb;
                ij = ij / norb;
                jp = ij % norb;
                ip = ij / norb;
                ip_occ = 0;
                jp_occ = 0;
                kp_occ = 0;
                lp_occ = 0;
                for (r = 0; r < neleca; ++r) {
                    int occ_index = occsa[r];
                    if (ip == occ_index) ip_occ = 1;
                    if (jp == occ_index) jp_occ = 1;
                }
                for (r = 0; r < nelecb; ++r) {
                    int occ_index = occsb[r];
                    if (kp == occ_index) kp_occ = 1;
                    if (lp == occ_index) lp_occ = 1;
                }
                if (jp_occ && lp_occ && !ip_occ && !kp_occ) {
                    uint64_t *tmp = toggle_bit(stra, nset, jp);
                    uint64_t *new_str_a = toggle_bit(tmp, nset, ip);
                    tmp = toggle_bit(strb, nset, lp);
                    uint64_t *new_str_b = toggle_bit(tmp, nset, kp);
                    for (iset = 0; iset < nset; ++iset) {
                        strs_add[strs_added * 2 * nset + iset] = new_str_a[iset];
                        strs_add[strs_added * 2 * nset + nset + iset] = new_str_b[iset];
                    }
                    free(tmp);
                    free(new_str_a);
                    free(new_str_b);
                    strs_added++;
                }
            }
            // Break statement
            if (aaaa_bbbb_done && aabb_done) {
                break;
            }
        } 
        free(occsa);
        free(occsb);
        free(virsa);
        free(virsb);
        if (strs_added > max_strs_add) {
            printf("\nError: Number of selected strings is greater than the size of the buffer array (%ld vs %ld).\n", strs_added, max_strs_add);
            exit(EXIT_FAILURE);
        }
    } // end loop over determinants

    free(focka);
    free(fockb);
    free(holes_a);
    free(holes_b);
    free(particles_a);
    free(particles_b);

    strs_add_size[0] = strs_added;

}

// Toggle bit at a specified position
uint64_t *toggle_bit(uint64_t *str, int nset, int p) {

    size_t i;

    uint64_t *new_str = malloc(sizeof(uint64_t) * nset);

    for (i = 0; i < nset; ++i) {
        new_str[i] = str[i];
    }
    
    int p_set = p / 64;
    int p_rel = p % 64;

    new_str[nset - p_set - 1] ^= 1ULL << p_rel; 

    return new_str;    

}

// Compares two string indices and determines the order
int order(uint64_t *strs_i, uint64_t *strs_j, int nset) {

    size_t i;

    for (i = 0; i < nset; ++i) {
        if (strs_i[i] > strs_j[i]) return 1;
        else if (strs_j[i] > strs_i[i]) return -1;
    }
 
    return 0;

}

// Recursive quick sort of string array indices
void qsort_idx(uint64_t *strs, uint64_t *idx, uint64_t *nstrs_, int nset, uint64_t *new_idx) {

    size_t p;

    uint64_t nstrs = nstrs_[0];

    if (nstrs <= 1) {
        for (p = 0; p < nstrs; ++p) new_idx[p] = idx[p];
    } 
    else {
        uint64_t ref = idx[nstrs - 1];
        uint64_t *group_lt = malloc(sizeof(uint64_t) * nstrs);
        uint64_t *group_gt = malloc(sizeof(uint64_t) * nstrs);
        uint64_t group_lt_nstrs = 0;
        uint64_t group_gt_nstrs = 0;
        for (p = 0; p < (nstrs - 1); ++p) {
            uint64_t i = idx[p];
            uint64_t *stri = strs + i * nset;
            uint64_t *strj = strs + ref * nset;
            int c = order(stri, strj, nset);
            if (c == -1) {
                group_lt[group_lt_nstrs] = i;
                group_lt_nstrs++;
            }
            else if (c == 1) {
                group_gt[group_gt_nstrs] = i;
                group_gt_nstrs++;
            }
        }
        uint64_t *new_idx_lt = malloc(sizeof(uint64_t) * group_lt_nstrs);
        uint64_t *new_idx_gt = malloc(sizeof(uint64_t) * group_gt_nstrs);
        qsort_idx(strs, group_lt, &group_lt_nstrs, nset, new_idx_lt);
        qsort_idx(strs, group_gt, &group_gt_nstrs, nset, new_idx_gt);
        nstrs = group_lt_nstrs + group_gt_nstrs + 1;
        nstrs_[0] = nstrs;
        for (p = 0; p < nstrs; ++p) {
            if (p < group_lt_nstrs)       new_idx[p] = new_idx_lt[p];
            else if (p == group_lt_nstrs) new_idx[p] = ref;
            else                          new_idx[p] = new_idx_gt[p - group_lt_nstrs - 1];
        }
        free(new_idx_lt); 
        free(new_idx_gt);
        free(group_lt);
        free(group_gt);
    }

}

// Helper function to perform recursive sort (nset is a total number of strings)
void argunique(uint64_t *strs, uint64_t *sort_idx, uint64_t *nstrs_, int nset) {

    size_t p;

    uint64_t *init_idx = malloc(sizeof(uint64_t) * nstrs_[0]);

    for (p = 0; p < nstrs_[0]; ++p) init_idx[p] = p;

    qsort_idx(strs, init_idx, nstrs_, nset, sort_idx);

    free(init_idx);

}

// Computes C' = S2 * C in the selected CI basis
void contract_ss_c(int norb, int neleca, int nelecb, uint64_t *strs, double *civec, uint64_t ndet, double *ci1) {

    int *ts = malloc(sizeof(int) * ndet);

    #pragma omp parallel
    {

    size_t ip, jp, p, q;
    int nset = (norb + 63) / 64;
 
    // Calculate excitation level for prescreening
    ts[0] = 0;
    uint64_t *str1a = strs;
    uint64_t *str1b = strs + nset;
    #pragma omp for schedule(static)
    for (ip = 1; ip < ndet; ++ip) {
        uint64_t *stria = strs + ip * 2 * nset;
        uint64_t *strib = strs + ip * 2 * nset + nset;
        ts[ip] = (n_excitations(stria, str1a, nset) + n_excitations(strib, str1b, nset));
    }

    // Loop over pairs of determinants
    #pragma omp for schedule(static)
    for (ip = 0; ip < ndet; ++ip) {
        for (jp = 0; jp < ndet; ++jp) {
            if (abs(ts[ip] - ts[jp]) < 3) {
                uint64_t *stria = strs + ip * 2 * nset;
                uint64_t *strib = strs + ip * 2 * nset + nset;
                uint64_t *strja = strs + jp * 2 * nset;
                uint64_t *strjb = strs + jp * 2 * nset + nset;
                int n_excit_a = n_excitations(stria, strja, nset);
                int n_excit_b = n_excitations(strib, strjb, nset);
                // Diagonal term
                if (ip == jp) {
                    double apb = (double) (neleca + nelecb);
                    double amb = (double) (neleca - nelecb);
                    double prefactor = apb / 2.0 + amb * amb / 4.0;
                    int *occsa = compute_occ_list(stria, nset, norb, neleca);
                    int *occsb = compute_occ_list(strib, nset, norb, nelecb);
                    for (p = 0; p < neleca; ++p) {
                        int pa = occsa[p];
                        for (q = 0; q < nelecb; ++q) {
                            int qb = occsb[q];
                            if (pa == qb) prefactor -= 1.0;
                        }
                    }
                    ci1[ip] += prefactor * civec[ip];
                    free(occsa);
                    free(occsb);
                }
                // Double excitation
                else if ((n_excit_a + n_excit_b) == 2) {
                    int i, j, a, b;
                    // alpha,beta->alpha,beta
                    if (n_excit_a == n_excit_b) {
                        int *ia = get_single_excitation(stria, strja, nset);
                        int *jb = get_single_excitation(strib, strjb, nset);
                        i = ia[0]; a = ia[1]; j = jb[0]; b = jb[1];
                        if (i == b && j == a) {
                            double sign = compute_cre_des_sign(a, i, stria, nset);
                            sign *= compute_cre_des_sign(b, j, strib, nset);
                            ci1[ip] -= sign * civec[jp];
                        }
                        free(ia);
                        free(jb);
                    }
                }
            } // end if over ts
        } // end loop over jp
    } // end loop over ip

    } // end omp
  
    free(ts);

}

// Computes C' = H * C and C'' = S2 * C simultaneously in the selected CI basis
void contract_h_c_ss_c(double *h1, double *eri, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, double *hdiag, uint64_t ndet, double *ci1, double *ci2) {

    int *ts = malloc(sizeof(int) * ndet);

    #pragma omp parallel
    {

    size_t ip, jp, p, q;
    int nset = (norb + 63) / 64;
 
    // Calculate excitation level for prescreening
    ts[0] = 0;
    uint64_t *str1a = strs;
    uint64_t *str1b = strs + nset;
    #pragma omp for schedule(static)
    for (ip = 1; ip < ndet; ++ip) {
        uint64_t *stria = strs + ip * 2 * nset;
        uint64_t *strib = strs + ip * 2 * nset + nset;
        ts[ip] = (n_excitations(stria, str1a, nset) + n_excitations(strib, str1b, nset));
    }

    // Loop over pairs of determinants
    #pragma omp for schedule(static)
    for (ip = 0; ip < ndet; ++ip) {
        for (jp = 0; jp < ndet; ++jp) {
            if (abs(ts[ip] - ts[jp]) < 3) {
                uint64_t *stria = strs + ip * 2 * nset;
                uint64_t *strib = strs + ip * 2 * nset + nset;
                uint64_t *strja = strs + jp * 2 * nset;
                uint64_t *strjb = strs + jp * 2 * nset + nset;
                int n_excit_a = n_excitations(stria, strja, nset);
                int n_excit_b = n_excitations(strib, strjb, nset);
                // Diagonal term
                if (ip == jp) {
                    ci1[ip] += hdiag[ip] * civec[ip];
                    // S^2
                    double apb = (double) (neleca + nelecb);
                    double amb = (double) (neleca - nelecb);
                    double prefactor = apb / 2.0 + amb * amb / 4.0;
                    int *occsa = compute_occ_list(stria, nset, norb, neleca);
                    int *occsb = compute_occ_list(strib, nset, norb, nelecb);
                    for (p = 0; p < neleca; ++p) {
                        int pa = occsa[p];
                        for (q = 0; q < nelecb; ++q) {
                            int qb = occsb[q];
                            if (pa == qb) prefactor -= 1.0;
                        }
                    }
                    ci2[ip] += prefactor * civec[ip];
                    free(occsa);
                    free(occsb);
                }
                // Single excitation
                else if ((n_excit_a + n_excit_b) == 1) {
                    int *ia;
                    // alpha->alpha
                    if (n_excit_b == 0) {
                        ia = get_single_excitation(stria, strja, nset);
                        int i = ia[0];
                        int a = ia[1];
                        double sign = compute_cre_des_sign(a, i, stria, nset);
                        int *occsa = compute_occ_list(stria, nset, norb, neleca);
                        int *occsb = compute_occ_list(strib, nset, norb, nelecb);
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
                        if (fabs(fai) > 1.0E-14) ci1[ip] += sign * fai * civec[jp];
                        free(occsa);
                        free(occsb);
                    }
                    // beta->beta
                    else if (n_excit_a == 0) {
                        ia = get_single_excitation(strib, strjb, nset);
                        int i = ia[0];
                        int a = ia[1];
                        double sign = compute_cre_des_sign(a, i, strib, nset);
                        int *occsa = compute_occ_list(stria, nset, norb, neleca);
                        int *occsb = compute_occ_list(strib, nset, norb, nelecb);
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
                        if (fabs(fai) > 1.0E-14) ci1[ip] += sign * fai * civec[jp];
                        free(occsa);
                        free(occsb);
                    }
                   free(ia);
                }
                // Double excitation
                else if ((n_excit_a + n_excit_b) == 2) {
                    int i, j, a, b;
                    // alpha,alpha->alpha,alpha
                    if (n_excit_b == 0) {
	                int *ijab = get_double_excitation(stria, strja, nset);
                        i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
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
                        if (fabs(v) > 1.0E-14) ci1[ip] += sign * v * civec[jp];
                        free(ijab);
                    }
                    // beta,beta->beta,beta
                    else if (n_excit_a == 0) {
	                int *ijab = get_double_excitation(strib, strjb, nset);
                        i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
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
                        if (fabs(v) > 1.0E-14) ci1[ip] += sign * v * civec[jp];
                        free(ijab);
                    }
                    // alpha,beta->alpha,beta
                    else {
                        int *ia = get_single_excitation(stria, strja, nset);
                        int *jb = get_single_excitation(strib, strjb, nset);
                        i = ia[0]; a = ia[1]; j = jb[0]; b = jb[1];
                        double v = eri[a * norb * norb * norb + i * norb * norb + b * norb + j];
                        double sign = compute_cre_des_sign(a, i, stria, nset);
                        sign *= compute_cre_des_sign(b, j, strib, nset);
                        if (fabs(v) > 1.0E-14) ci1[ip] += sign * v * civec[jp];
                        // S^2
                        if (i == b && j == a) {
                            ci2[ip] -= sign * civec[jp];
                        }
                        free(ia);
                        free(jb);
                   }
                }
            } // end if over ts
        } // end loop over jp
    } // end loop over ip

    } // end omp
  
    free(ts);

}

// 2-RDM is sorted in physicists notation: gamma_pqsr=<\Phi|a_p^dag a_q^dag a_r a_s|\Phi>
void compute_rdm12s(int norb, int neleca, int nelecb, uint64_t *strs, double *civec, uint64_t ndet, double *rdm1a, double *rdm1b, double *rdm2aa, double *rdm2ab, double *rdm2bb) {

    #pragma omp parallel
    {

    size_t ip, jp, p, q, r, s;
    int nset = (norb + 63) / 64;
    double ci_sq = 0.0;
    double *rdm1a_private = malloc(sizeof(double) * norb * norb);
    double *rdm1b_private = malloc(sizeof(double) * norb * norb);
    double *rdm2aa_private = malloc(sizeof(double) * norb * norb * norb * norb);
    double *rdm2ab_private = malloc(sizeof(double) * norb * norb * norb * norb);
    double *rdm2bb_private = malloc(sizeof(double) * norb * norb * norb * norb);

    for (p = 0; p < norb * norb; ++p) {
        rdm1a_private[p] = 0.0;
        rdm1b_private[p] = 0.0;
    }
    for (p = 0; p < norb * norb * norb * norb; ++p) {
        rdm2aa_private[p] = 0.0;
        rdm2ab_private[p] = 0.0;
        rdm2bb_private[p] = 0.0;
    }
 
    // Loop over pairs of determinants
    #pragma omp for schedule(static) 
    for (ip = 0; ip < ndet; ++ip) {
        for (jp = 0; jp < ndet; ++jp) {
            uint64_t *stria = strs + ip * 2 * nset;
            uint64_t *strib = strs + ip * 2 * nset + nset;
            uint64_t *strja = strs + jp * 2 * nset;
            uint64_t *strjb = strs + jp * 2 * nset + nset;
            int n_excit_a = n_excitations(stria, strja, nset);
            int n_excit_b = n_excitations(strib, strjb, nset);
            // Diagonal term
            if (ip == jp) {
                int *occsa = compute_occ_list(stria, nset, norb, neleca);
                int *occsb = compute_occ_list(strib, nset, norb, nelecb);
                ci_sq = civec[ip] * civec[ip];
                // Diagonal rdm1_aa
                for (p = 0; p < neleca; ++p) {
                    int k = occsa[p];
                    int kk = k * norb + k;
                    rdm1a_private[kk] += ci_sq;
                }
                // Diagonal rdm1_bb
                for (p = 0; p < nelecb; ++p) {
                    int k = occsb[p];
                    int kk = k * norb + k;
                    rdm1b_private[kk] += ci_sq;
                }
                // Diagonal rdm2_aaaa
                for (p = 0; p < neleca; ++p) {
                    int k = occsa[p];
                    for (q = 0; q < neleca; ++q) {
                        int j = occsa[q];
                        int kjkj = k * norb * norb * norb + j * norb * norb + k * norb + j;
                        int kjjk = k * norb * norb * norb + j * norb * norb + j * norb + k;
                        rdm2aa_private[kjkj] += ci_sq;
                        rdm2aa_private[kjjk] -= ci_sq;
                    }
                    // Diagonal rdm2_abab
                    for (q = 0; q < nelecb; ++q) {
                        int j = occsb[q];
                        int kjkj = k * norb * norb * norb + j * norb * norb + k * norb + j;
                        rdm2ab_private[kjkj] += ci_sq;
                    }
                }
                // Diagonal rdm2_bbbb
                for (p = 0; p < nelecb; ++p) {
                    int k = occsb[p];
                    for (q = 0; q < nelecb; ++q) {
                        int j = occsb[q];
                        int kjkj = k * norb * norb * norb + j * norb * norb + k * norb + j;
                        int kjjk = k * norb * norb * norb + j * norb * norb + j * norb + k;
                        rdm2bb_private[kjkj] += ci_sq;
                        rdm2bb_private[kjjk] -= ci_sq;
                    }
                }
                free(occsa);
                free(occsb);
            }
            // Single excitation
            else if ((n_excit_a + n_excit_b) == 1) {
                int *ia;
                // alpha->alpha
                if (n_excit_b == 0) {
                    ia = get_single_excitation(stria, strja, nset);
                    int i = ia[0];
                    int a = ia[1];
                    double sign = compute_cre_des_sign(a, i, stria, nset);
                    int *occsa = compute_occ_list(stria, nset, norb, neleca);
                    int *occsb = compute_occ_list(strib, nset, norb, nelecb);
                    ci_sq = sign * civec[ip] * civec[jp];
                    // rdm1_aa
                    rdm1a_private[a * norb + i] += ci_sq;
                    // rdm2_aaaa
                    for (p = 0; p < neleca; ++p) {
                        int k = occsa[p];
                        int akik = a * norb * norb * norb + k * norb * norb + i * norb + k;
                        int akki = a * norb * norb * norb + k * norb * norb + k * norb + i;
                        int kaki = k * norb * norb * norb + a * norb * norb + k * norb + i;
                        int kaik = k * norb * norb * norb + a * norb * norb + i * norb + k;
                        rdm2aa_private[akik] += ci_sq;
                        rdm2aa_private[akki] -= ci_sq;
                        rdm2aa_private[kaik] -= ci_sq;
                        rdm2aa_private[kaki] += ci_sq;
                    }
                    // rdm2_abab
                    for (p = 0; p < nelecb; ++p) {
                        int k = occsb[p];
                        int akik = a * norb * norb * norb + k * norb * norb + i * norb + k;
                        rdm2ab_private[akik] += ci_sq;
                    }
                    free(occsa);
                    free(occsb);
                }
                // beta->beta
                else if (n_excit_a == 0) {
                    ia = get_single_excitation(strib, strjb, nset);
                    int i = ia[0];
                    int a = ia[1];
                    double sign = compute_cre_des_sign(a, i, strib, nset);
                    int *occsa = compute_occ_list(stria, nset, norb, neleca);
                    int *occsb = compute_occ_list(strib, nset, norb, nelecb);
                    ci_sq = sign * civec[ip] * civec[jp];
                    // rdm1_bb
                    rdm1b_private[a * norb + i] += ci_sq;
                    // rdm2_bbbb
                    for (p = 0; p < nelecb; ++p) {
                        int k = occsb[p];
                        int akik = a * norb * norb * norb + k * norb * norb + i * norb + k;
                        int akki = a * norb * norb * norb + k * norb * norb + k * norb + i;
                        int kaki = k * norb * norb * norb + a * norb * norb + k * norb + i;
                        int kaik = k * norb * norb * norb + a * norb * norb + i * norb + k;
                        rdm2bb_private[akik] += ci_sq;
                        rdm2bb_private[akki] -= ci_sq;
                        rdm2bb_private[kaik] -= ci_sq;
                        rdm2bb_private[kaki] += ci_sq;
                    }
                    // rdm2_abab
                    for (p = 0; p < neleca; ++p) {
                        int k = occsa[p];
                        int kaki = k * norb * norb * norb + a * norb * norb + k * norb + i;
                        rdm2ab_private[kaki] += ci_sq;
                    }
                    free(occsa);
                    free(occsb);
                }
               free(ia);
            }
            // Double excitation
            else if ((n_excit_a + n_excit_b) == 2) {
                int i, j, a, b;
                // rdm2_aaaa
                if (n_excit_b == 0) {
	            int *ijab = get_double_excitation(stria, strja, nset);
                    i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
                    double sign;
                    int baij = b * norb * norb * norb + a * norb * norb + i * norb + j;
                    int baji = b * norb * norb * norb + a * norb * norb + j * norb + i;
                    int abij = a * norb * norb * norb + b * norb * norb + i * norb + j;
                    int abji = a * norb * norb * norb + b * norb * norb + j * norb + i;
                    if (a > j || i > b) {
                        sign = compute_cre_des_sign(b, i, stria, nset);
                        sign *= compute_cre_des_sign(a, j, stria, nset);
                        ci_sq = sign * civec[ip] * civec[jp];
                        rdm2aa_private[baij] += ci_sq;
                        rdm2aa_private[baji] -= ci_sq;
                        rdm2aa_private[abij] -= ci_sq;
                        rdm2aa_private[abji] += ci_sq;
                    } 
                    else {
                        sign = compute_cre_des_sign(b, j, stria, nset);
                        sign *= compute_cre_des_sign(a, i, stria, nset);
                        ci_sq = sign * civec[ip] * civec[jp];
                        rdm2aa_private[baij] -= ci_sq;
                        rdm2aa_private[baji] += ci_sq;
                        rdm2aa_private[abij] += ci_sq;
                        rdm2aa_private[abji] -= ci_sq;
                    }
                    free(ijab);
                }
                // rdm2_bbbb
                else if (n_excit_a == 0) {
	            int *ijab = get_double_excitation(strib, strjb, nset);
                    i = ijab[0]; j = ijab[1]; a = ijab[2]; b = ijab[3];
                    double v, sign;
                    int baij = b * norb * norb * norb + a * norb * norb + i * norb + j;
                    int baji = b * norb * norb * norb + a * norb * norb + j * norb + i;
                    int abij = a * norb * norb * norb + b * norb * norb + i * norb + j;
                    int abji = a * norb * norb * norb + b * norb * norb + j * norb + i;
                    if (a > j || i > b) {
                        sign = compute_cre_des_sign(b, i, strib, nset);
                        sign *= compute_cre_des_sign(a, j, strib, nset);
                        ci_sq = sign * civec[ip] * civec[jp];
                        rdm2bb_private[baij] += ci_sq;
                        rdm2bb_private[baji] -= ci_sq;
                        rdm2bb_private[abij] -= ci_sq;
                        rdm2bb_private[abji] += ci_sq;
                    } 
                    else {
                        sign = compute_cre_des_sign(b, j, strib, nset);
                        sign *= compute_cre_des_sign(a, i, strib, nset);
                        ci_sq = sign * civec[ip] * civec[jp];
                        rdm2bb_private[baij] -= ci_sq;
                        rdm2bb_private[baji] += ci_sq;
                        rdm2bb_private[abij] += ci_sq;
                        rdm2bb_private[abji] -= ci_sq;
                    }
                    free(ijab);
                }
                // rdm2_abab
                else {
                    int *ia = get_single_excitation(stria, strja, nset);
                    int *jb = get_single_excitation(strib, strjb, nset);
                    i = ia[0]; a = ia[1]; j = jb[0]; b = jb[1];
                    double sign = compute_cre_des_sign(a, i, stria, nset);
                    sign *= compute_cre_des_sign(b, j, strib, nset);
                    ci_sq = sign * civec[ip] * civec[jp];
                    int abij = a * norb * norb * norb + b * norb * norb + i * norb + j;
                    rdm2ab_private[abij] += ci_sq;
                    free(ia);
                    free(jb);
               }
            }
        } // end loop over jp
    } // end loop over ip

    #pragma omp critical
    {
    for (p = 0; p < norb * norb; ++p) {
        rdm1a[p] += rdm1a_private[p];
        rdm1b[p] += rdm1b_private[p];
    }
    for (p = 0; p < norb * norb * norb * norb; ++p) {
        rdm2aa[p] += rdm2aa_private[p];
        rdm2ab[p] += rdm2ab_private[p];
        rdm2bb[p] += rdm2bb_private[p];
    }
    }
 
    free(rdm1a_private);
    free(rdm1b_private);
    free(rdm2aa_private);
    free(rdm2ab_private);
    free(rdm2bb_private);

    } // end omp
  
}
