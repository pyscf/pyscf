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
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <assert.h>
#include "config.h"
#include "cint.h"

#define INTBUFMAX       1000
#define INTBUFMAX10     8000
#define IMGBLK          80
#define OF_CMPLX        2

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define ABS(X)          ((X>0)?(X):(-X))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static int shloc_partition(int *kshloc, int *ao_loc, int ksh0, int ksh1, int dkmax)
{
        int ksh;
        int nloc = 0;
        int loclast = ao_loc[ksh0];
        kshloc[0] = ksh0;
        for (ksh = ksh0+1; ksh < ksh1; ksh++) {
                assert(ao_loc[ksh+1] - ao_loc[ksh] < dkmax);
                if (ao_loc[ksh+1] - loclast > dkmax) {
                        nloc += 1;
                        kshloc[nloc] = ksh;
                        loclast = ao_loc[ksh];
                }
        }
        nloc += 1;
        kshloc[nloc] = ksh1;
        return nloc;
}

static int shloc_partition_by_atom(int *kshloc, int *katmloc, int *ao_loc,
                                   int *shl_loc, int katm0, int katm1,
                                   int dkmax)
{
    int *kshloc_ = kshloc;
    int nkshloc, nkshloc_;
    int katm;

    nkshloc = 0;
    katmloc[0] = 0;
    for(katm=katm0; katm < katm1; ++katm) {
        nkshloc_ = shloc_partition(kshloc_, ao_loc,
                                   shl_loc[katm], shl_loc[katm+1], dkmax);
        nkshloc += nkshloc_;
        katmloc[katm-katm0+1] = nkshloc;
        // printf("katm= %d\n", katm);
        // int i;
        // for(i=0; i<nkshloc_+1; ++i) {
        //     printf("%d %d\n", i, kshloc_[i]);
        // }
        kshloc_ += nkshloc_;
    }
    // printf("\n");
    // for(katm=0;katm<katm1-katm0+1;++katm) {
    //     printf("katm=%d  %d\n", katm, katmloc[katm]);
    // }
    // printf("\n");
    return nkshloc;
}


double get_dsqure(double *ri, double *rj)
{
    double dx = ri[0]-rj[0];
    double dy = ri[1]-rj[1];
    double dz = ri[2]-rj[2];
    return dx*dx+dy*dy+dz*dz;
}
void get_rc(double *rc, double *ri, double *rj, double ei, double ej) {
    double eij = ei+ej;
    rc[0] = (ri[0]*ei + rj[0]*ej) / eij;
    rc[1] = (ri[1]*ei + rj[1]*ej) / eij;
    rc[2] = (ri[2]*ei + rj[2]*ej) / eij;
}
size_t max_shlsize(int *ao_loc, int nbas)
{
    int i, dimax=0;
    for(i=0; i<nbas; ++i) {
        dimax = MAX(dimax,ao_loc[i+1]-ao_loc[i]);
    }
    return dimax;
}

void fill_sr2c2e_g(int (*intor)(), double *out,
                   int comp, CINTOpt *cintopt,
                   int *ao_loc, int *ao_locsup,
                   double *uniq_Rcuts, int *refuniqshl_map,
                   int *refsupshl_loc, int *refsupshl_map,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
{
    size_t IJstart, IJSH, ISH, JSH, Ish, Jsh, I0, I1, J0, J1;
    size_t jsh, jsh_, i, j, i0, jmax, iptrxyz, jptrxyz;
    size_t di, dj, dic, dijc;
    int shls[2];
    double Rcut2, Rij2;
    double *ri, *rj;
    const int dimax = max_shlsize(ao_loc, nbas);
    int shls_slice[4];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);
    double *buf = malloc(sizeof(double)*(dimax*dimax+cache_size));
    double *buf_L, *pbuf, *cache;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

    for(Ish=0; Ish<nbas; ++Ish) {
        ISH = refuniqshl_map[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        IJstart = I0*(I0+1)/2;
        di = I1 - I0;
        dic = di * comp;
        shls[1] = Ish;
        iptrxyz = atm[PTR_COORD+bas[ATOM_OF+Ish*BAS_SLOTS]*ATM_SLOTS];
        ri = env+iptrxyz;
        for(Jsh=0; Jsh<=Ish; ++Jsh) {
            JSH = refuniqshl_map[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dijc = dic * dj;
            buf_L = buf;
            pbuf = buf + dijc;
            cache = pbuf + dijc;
            for(j=0; j<dijc; ++j) {
                buf_L[j] = 0.;
            }
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            Rcut2 = uniq_Rcuts[IJSH]*uniq_Rcuts[IJSH];

            for(jsh_=refsupshl_loc[Jsh]; jsh_<refsupshl_loc[Jsh+1]; ++jsh_) {
                jsh = refsupshl_map[jsh_];
                shls[0] = jsh;
                jptrxyz = atmsup[PTR_COORD+bassup[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
                rj = envsup+jptrxyz;
                Rij2 = get_dsqure(ri,rj);
                if(Rij2 < Rcut2) {
                    if(safe) {
                        envsup[PTR_RANGE_OMEGA] = 0.;
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            for(j=0; j<dijc; ++j) {
                                buf_L[j] += pbuf[j];
                            }
                        }
                        envsup[PTR_RANGE_OMEGA] = omega;
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            for(j=0; j<dijc; ++j) {
                                buf_L[j] -= pbuf[j];
                            }
                        }
                    } else {
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            for(j=0; j<dijc; ++j) {
                                buf_L[j] += pbuf[j];
                            }
                        }
                    }
                }
            }

            i0 = IJstart;
            for(i=0; i<di; ++i) {
                jmax = (Ish==Jsh)?(i+1):(dj);
                for(j=0; j<jmax; ++j) {
                    out[i0+j] = buf_L[i*dj+j];
                }
                i0 += I0+i+1;
            }

            IJstart += dj;
        }
    }

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(buf);
}

void fill_sr2c2e_k(int (*intor)(), double complex *out,
                   int comp, CINTOpt *cintopt,
                   double complex *expLk, int nkpts,
                   int *ao_loc, int *ao_locsup,
                   double *uniq_Rcuts, int *refuniqshl_map,
                   int *refsupshl_loc, int *refsupshl_map,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
{
    double *expLk_r = malloc(sizeof(double) * natmsup*nkpts * OF_CMPLX);
    double *expLk_i = expLk_r + natmsup*nkpts;
    double *expLk_r_, *expLk_i_;
    double phi_r, phi_i, tmp;
    double complex *outk;
    int i;
    for (i = 0; i < natmsup*nkpts; i++) {
            expLk_r[i] = creal(expLk[i]);
            expLk_i[i] = cimag(expLk[i]);
    }

    size_t IJstart, IJSH, ISH, JSH, Ish, Jsh, I0, I1, J0, J1;
    size_t jsh, jsh_, j, i0, kk, jmax, iptrxyz, jptrxyz, jatm;
    size_t di, dj, dic, dijc;
    int shls[2];
    int nao2 = ao_loc[nbas]*(ao_loc[nbas]+1)/2;
    double Rcut2, Rij2;
    double *ri, *rj;
    const int dimax = max_shlsize(ao_loc, nbas);
    int shls_slice[4];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);
    double *buf = malloc(sizeof(double)*(dimax*dimax+cache_size));
    double *buf_L, *pbuf, *cache;
    char skip;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

    for(Ish=0; Ish<nbas; ++Ish) {
        ISH = refuniqshl_map[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        IJstart = I0*(I0+1)/2;
        di = I1 - I0;
        dic = di * comp;
        shls[1] = Ish;
        iptrxyz = atm[PTR_COORD+bas[ATOM_OF+Ish*BAS_SLOTS]*ATM_SLOTS];
        ri = env+iptrxyz;
        for(Jsh=0; Jsh<=Ish; ++Jsh) {
            JSH = refuniqshl_map[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dijc = dic * dj;
            buf_L = buf;
            pbuf = buf + dijc;
            cache = pbuf + dijc;
            for(j=0; j<dijc; ++j) {
                buf_L[j] = 0.;
            }
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            Rcut2 = uniq_Rcuts[IJSH]*uniq_Rcuts[IJSH];

            for(jsh_=refsupshl_loc[Jsh]; jsh_<refsupshl_loc[Jsh+1]; ++jsh_) {
                jsh = refsupshl_map[jsh_];
                shls[0] = jsh;
                jatm = bassup[ATOM_OF+jsh*BAS_SLOTS];
                expLk_r_ = expLk_r + jatm * nkpts;
                expLk_i_ = expLk_i + jatm * nkpts;
                jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                rj = envsup+jptrxyz;
                Rij2 = get_dsqure(ri,rj);
                if(Rij2 < Rcut2) {
                    skip = 1;
                    if(safe) {
                        envsup[PTR_RANGE_OMEGA] = 0.;
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            skip = 0;
                            envsup[PTR_RANGE_OMEGA] = omega;
                            (*intor)(buf_L, NULL, shls, atmsup, natmsup, bassup,
                                         nbassup, envsup, cintopt, cache);
                            for(j=0; j<dijc; ++j) {
                                pbuf[j] -= buf_L[j];
                            }
                        }
                    } else {
                        if ((*intor)(pbuf, NULL, shls, atmsup, natmsup, bassup,
                                     nbassup, envsup, cintopt, cache)) {
                            skip = 0;
                        }
                    }
                    if(!skip) {
                        for(kk=0; kk<nkpts; ++kk) {
                            phi_r = expLk_r_[kk];
                            phi_i = expLk_i_[kk];
                            outk = out + kk * nao2;
                            i0 = IJstart;
                            for(i=0; i<di; ++i) {
                                jmax = (Ish==Jsh)?(i+1):(dj);
                                for(j=0; j<jmax; ++j) {
                                    tmp = pbuf[i*dj+j];
                                    outk[i0+j] += tmp * phi_r +
                                                  tmp * phi_i * _Complex_I;
                                }
                                i0 += I0+i+1;
                            }
                        }
                    }
                }
            }

            IJstart += dj;
        }
    }

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(expLk_r);
    free(buf);
}

void fill_sr3c2e_g(int (*intor)(), double *out,
                   int comp, CINTOpt *cintopt,
                   int *ao_loc, int *ao_locsup, int *shl_loc,
                   int *refuniqshl_map,
                   int *auxuniqshl_map, int nbasauxuniq,
                   double *uniq_Rcut2s, double *refexp,
                   int *refshlstart_by_atm, int *supshlstart_by_atm,
                   int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                   int *refatmprd_loc,
                   int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                   int *atm, int natm, int *bas, int nbas, int nbasaux,
                   double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
/*
    ao_loc = concatenate([ao_loc, ao_locaux])
    ao_locsup = concatenate([ao_locsup, ao_locaux])
*/
{
    const int *supatmpr_i_lst = supatmpr_lst;
    const int *supatmpr_j_lst = supatmpr_lst + nsupatmpr;
    const int kshshift = nbassup - nbas;
    const int nbassupaux = nbassup + nbasaux;
    const int natmsupaux = natmsup + natm;

    int Ish, Jsh, IJsh, ISH, JSH, IJSH, Ishshift, Jshshift, ijsh, ijsh0, ijsh1, ish, jsh, I0, I1, J0, J1, IJstart;
    int Iatm, Jatm, IJatm, iatm, jatm, ijatm, ijatm0, ijatm1;
    int Katm, Ksh, Ksh0, Ksh1, ksh, K0, K1, KSH;
    int iptrxyz, jptrxyz, kptrxyz;
    int idij, idij0, idij1, Idij, Idij0;
    int di, dj, dk, dij, dijk, dijktot, dijkmax;
    int dimax = max_shlsize(ao_loc, nbas);
    int dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    int dktot = ao_loc[nbas+nbasaux] - ao_loc[nbas];
    int i,j,jmax,k,i0;
    int nao = ao_loc[nbas]-ao_loc[0];
    int nao2 = nao*(nao+1)/2;
    char skip;
    double ei, ej, Rijk2, Rcut2;
    double *uniq_Rcut2s_K, *ri, *rj, *rk;
    double rc[3];

    int shls[3];
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas+nbasaux, env);
// TODO: batch Ksh, which could be HUGE for big supercell.
    const int buf_size = dimax*dimax*dktot;
    const int tmp_size = dimax*dimax*dkmax*2;
    double *buf = malloc(sizeof(double)*(buf_size+tmp_size+cache_size));
    double *buf_L, *buf_Lk, *pbuf, *pbuf2, *cache, *outk;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

// >>>>>>>>
    unsigned int count = 0;
// <<<<<<<<

    for(Ish=0, IJsh=0; Ish<nbas; ++Ish) {
        Iatm = bassup[ATOM_OF+Ish*BAS_SLOTS];
        Ishshift = Ish - refshlstart_by_atm[Iatm];
        ISH = refuniqshl_map[Ish];
        ei = refexp[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        IJstart = I0*(I0+1)/2;
        di = I1 - I0;
        for(Jsh=0; Jsh<=Ish; ++Jsh, ++IJsh) {
            Jatm = bassup[ATOM_OF+Jsh*BAS_SLOTS];
            IJatm = (Iatm>=Jatm)?(Iatm*(Iatm+1)/2+Jatm):(Jatm*(Jatm+1)/2+Iatm);
            Jshshift = Jsh - refshlstart_by_atm[Jatm];
            JSH = refuniqshl_map[Jsh];
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            ej = refexp[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dij = di * dj;
            dijktot = dij * dktot;
            dijkmax = dij * dkmax;
            buf_L = buf;
            pbuf = buf_L + dijktot;
            pbuf2 = pbuf + dijkmax;
            cache = pbuf2 + dijkmax;
            for(i=0; i<dijktot; ++i) {
                buf_L[i] = 0;
            }

            idij0 = refatmprd_loc[IJatm];
            Idij0 = uniqshlpr_dij_loc[IJSH];
            idij1 = idij0 + uniqshlpr_dij_loc[IJSH+1] - Idij0;
            for(idij=idij0; idij<idij1; ++idij) {
                // get cutoff for IJ
                Idij = Idij0 + idij-idij0;
                uniq_Rcut2s_K = uniq_Rcut2s + Idij * nbasauxuniq;

                ijatm0 = supatmpr_loc[idij];
                ijatm1 = supatmpr_loc[idij+1];
                for(ijatm=ijatm0; ijatm<ijatm1; ++ijatm) {
                    iatm = supatmpr_i_lst[ijatm];
                    jatm = supatmpr_j_lst[ijatm];
                    iptrxyz = atmsup[PTR_COORD+iatm*ATM_SLOTS];
                    ri = envsup+iptrxyz;
                    jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                    rj = envsup+jptrxyz;
                    get_rc(rc, ri, rj, ei, ej);

                    // supmol atm index to supmol shl index
                    ish = supshlstart_by_atm[iatm] + Ishshift;
                    jsh = supshlstart_by_atm[jatm] + Jshshift;
                    shls[1] = ish;
                    shls[0] = jsh;

                    buf_Lk = buf_L;

                    for(Katm=natm; Katm<2*natm; ++Katm) { // first natm is ref
                        kptrxyz = atm[PTR_COORD+Katm*ATM_SLOTS];
                        rk = env+kptrxyz;
                        Rijk2 = get_dsqure(rc, rk);
                        Ksh0 = shl_loc[Katm];
                        Ksh1 = shl_loc[Katm+1];
                        for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                            KSH = auxuniqshl_map[Ksh-nbas];
                            Rcut2 = uniq_Rcut2s_K[KSH];
                            K0 = ao_loc[Ksh];
                            K1 = ao_loc[Ksh+1];
                            dk = K1 - K0;
                            dijk = dij * dk;

                            if(Rijk2<=Rcut2) {
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                ++count;
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                ksh = Ksh + kshshift;
                                shls[2] = ksh;

                                skip = 1;
                                if(safe) {
                                    envsup[PTR_RANGE_OMEGA] = 0.;
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                        envsup[PTR_RANGE_OMEGA] = omega;
                                        (*intor)(pbuf2, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache);
                                        for(i=0; i<dijk; ++i) {
                                            pbuf[i] -= pbuf2[i];
                                        }
                                    }
                                } else {
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                    }
                                }

                                if(!skip) {
                                    for(i=0; i<dijk; ++i) {
                                        buf_Lk[i] += pbuf[i];
                                    }
                                }
                            }

                            buf_Lk += dijk;
                        } // Ksh
                    } // Katm
                } // ijatm
            } // idij

            buf_Lk = buf_L;
            outk = out;
            Ksh0 = shl_loc[natm];
            Ksh1 = shl_loc[2*natm];
            for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                dk = ao_loc[Ksh+1] - ao_loc[Ksh];
                for(k=0; k<dk; ++k) {
                    i0 = IJstart;
                    for(i=0; i<di; ++i) {
                        jmax = (Ish==Jsh)?(i+1):(dj);
                        for(j=0; j<jmax; ++j) {
                            outk[i0+j] = buf_Lk[i*dj+j];
                        }
                        i0 += I0+i+1;
                    }
                    outk += nao2;
                    buf_Lk += dij;
                }
            }

            IJstart += dj;

        } // Jsh
    } // Ish

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(buf);

// >>>>>>>>
    printf("num significant shlpr %d\n", count);
// <<<<<<<<
}

void fill_sr3c2e_g_ijsh(int (*intor)(), double *out, double *buf,
                   int comp, CINTOpt *cintopt,
                   int Ish, int Jsh, int IJstart,
                   int *ao_loc, int *ao_locsup, int *shl_loc,
                   int *refuniqshl_map,
                   int *auxuniqshl_map, int nbasauxuniq,
                   double *uniq_Rcut2s, double *refexp,
                   int *refshlstart_by_atm, int *supshlstart_by_atm,
                   int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                   int *refatmprd_loc,
                   int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                   int *atm, int natm, int *bas, int nbas, int nbasaux,
                   double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
/*
    ao_loc = concatenate([ao_loc, ao_locaux])
    ao_locsup = concatenate([ao_locsup, ao_locaux])
*/
{
// >>>>>>>>
    // unsigned int count = 0;
// <<<<<<<<

    int shls[3];
    const int *supatmpr_i_lst = supatmpr_lst;
    const int *supatmpr_j_lst = supatmpr_lst + nsupatmpr;
    const int kshshift = nbassup - nbas;
    const int nbassupaux = nbassup + nbasaux;
    const int natmsupaux = natmsup + natm;

    int i, j, jmax, k, i0;
    int I0 = ao_loc[Ish];
    int nao = ao_loc[nbas]-ao_loc[0];
    int nao2 = nao*(nao+1)/2;
    char skip;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

// atm/shl info
    int Iatm, Jatm, IJatm, Katm, katm, ijatm0, ijatm1, ijatm, iatm, jatm, iptrxyz, jptrxyz, kptrxyz;
    double *ri, *rj, *rk, rc[3], Rijk2, Rcut2;
    Iatm = bassup[ATOM_OF+Ish*BAS_SLOTS];
    Jatm = bassup[ATOM_OF+Jsh*BAS_SLOTS];
    IJatm = (Iatm>=Jatm)?(Iatm*(Iatm+1)/2+Jatm):(Jatm*(Jatm+1)/2+Iatm);

    int ish, jsh, Ishshift, Jshshift, ISH, JSH, IJSH, Ksh, Ksh0, Ksh1, KSH, ksh;
    Ishshift = Ish - refshlstart_by_atm[Iatm];
    Jshshift = Jsh - refshlstart_by_atm[Jatm];
    ISH = refuniqshl_map[Ish];
    JSH = refuniqshl_map[Jsh];
    IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);

    double ei, ej;
    ei = refexp[Ish];
    ej = refexp[Jsh];

// partition ksh_loc by atom and buf size
    int di, dj, dk, dkmax, dkbuf, dij, dijk, dijkmax, dijkbuf, nkshloc;
    int kshloc[nbasaux+1], katmloc[natm+1], kloc, kloc0, kloc1;
    di = ao_loc[Ish+1] - ao_loc[Ish];
    dj = ao_loc[Jsh+1] - ao_loc[Jsh];
    dij = di * dj;
    dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    dkbuf = MAX(INTBUFMAX/dij, dkmax);
    dijkbuf = dij * dkbuf;
    dijkmax = dij * dkmax;
    nkshloc = shloc_partition_by_atom(kshloc, katmloc, ao_loc, shl_loc,
                                      natm, 2*natm, dkbuf);

    double *buf_L, *pbuf, *pbuf2, *cache, *buf_Lk, *outk;
    buf_L = buf;
    pbuf = buf_L + dijkbuf;
    pbuf2 = pbuf + dijkmax;
    cache = pbuf2 + dijkmax;

    int idij0, idij1, idij, Idij0, Idij;
    double * uniq_Rcut2s_K;
    idij0 = refatmprd_loc[IJatm];
    Idij0 = uniqshlpr_dij_loc[IJSH];
    idij1 = idij0 + uniqshlpr_dij_loc[IJSH+1] - Idij0;

    outk = out;
    for(Katm=natm; Katm<2*natm; ++Katm) { // first natm is ref
        kptrxyz = atm[PTR_COORD+Katm*ATM_SLOTS];
        rk = env+kptrxyz;

        katm = Katm - natm;
        kloc0 = katmloc[katm];
        kloc1 = katmloc[katm+1];

        for(kloc=kloc0; kloc<kloc1; ++kloc) {

            for(i=0; i<dijkbuf; ++i) {
                buf_L[i] = 0;
            }

            Ksh0 = kshloc[kloc];
            Ksh1 = kshloc[kloc+1];

            for(idij=idij0; idij<idij1; ++idij) {
                // get cutoff for IJ
                Idij = Idij0 + idij-idij0;
                uniq_Rcut2s_K = uniq_Rcut2s + Idij * nbasauxuniq;

                ijatm0 = supatmpr_loc[idij];
                ijatm1 = supatmpr_loc[idij+1];
                for(ijatm=ijatm0; ijatm<ijatm1; ++ijatm) {
                    iatm = supatmpr_i_lst[ijatm];
                    jatm = supatmpr_j_lst[ijatm];
                    iptrxyz = atmsup[PTR_COORD+iatm*ATM_SLOTS];
                    ri = envsup+iptrxyz;
                    jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                    rj = envsup+jptrxyz;
                    get_rc(rc, ri, rj, ei, ej);
                    Rijk2 = get_dsqure(rc, rk);

                    // supmol atm index to supmol shl index
                    ish = supshlstart_by_atm[iatm] + Ishshift;
                    jsh = supshlstart_by_atm[jatm] + Jshshift;
                    shls[1] = ish;
                    shls[0] = jsh;

                    buf_Lk = buf_L;
                    for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                        KSH = auxuniqshl_map[Ksh-nbas];
                        Rcut2 = uniq_Rcut2s_K[KSH];
                        dk = ao_loc[Ksh+1] - ao_loc[Ksh];
                        dijk = dij * dk;

                        if(Rijk2<=Rcut2) {
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ++count;
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                            ksh = Ksh + kshshift;
                            shls[2] = ksh;

                            skip = 1;
                            if(safe) {
                                envsup[PTR_RANGE_OMEGA] = 0.;
                                if ((*intor)(pbuf, NULL, shls, atmsup,
                                             natmsupaux, bassup, nbassupaux,
                                             envsup, cintopt, cache)) {
                                    skip = 0;
                                    envsup[PTR_RANGE_OMEGA] = omega;
                                    (*intor)(pbuf2, NULL, shls, atmsup,
                                             natmsupaux, bassup, nbassupaux,
                                             envsup, cintopt, cache);
                                    for(i=0; i<dijk; ++i) {
                                        pbuf[i] -= pbuf2[i];
                                    }
                                }
                            } else {
                                if ((*intor)(pbuf, NULL, shls, atmsup,
                                             natmsupaux, bassup, nbassupaux,
                                             envsup, cintopt, cache)) {
                                    skip = 0;
                                }
                            }

                            if(!skip) {
                                for(i=0; i<dijk; ++i) {
                                    buf_Lk[i] += pbuf[i];
                                }
                            }
                        }

                        buf_Lk += dijk;
                    } // Ksh

                } // ijatm
            } // idij

            buf_Lk = buf_L;
            for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                dk = ao_loc[Ksh+1] - ao_loc[Ksh];
                for(k=0; k<dk; ++k) {
                    i0 = IJstart;
                    for(i=0; i<di; ++i) {
                        jmax = (Ish==Jsh)?(i+1):(dj);
                        for(j=0; j<jmax; ++j) {
                            outk[i0+j] += buf_Lk[i*dj+j];
                        }
                        i0 += I0+i+1;
                    }
                    outk += nao2;
                    buf_Lk += dij;
                }
            }

        } // kloc
    } // Katm

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

// >>>>>>>>
    // printf("num significant shlpr %d\n", count);
// <<<<<<<<
}

void PBCnr_sr3c2e_g_drv(int (*intor)(), double *out,
                        int comp, CINTOpt *cintopt,
                        int *ao_loc, int *ao_locsup, int *shl_loc,
                        int *refuniqshl_map,
                        int *auxuniqshl_map, int nbasauxuniq,
                        double *uniq_Rcut2s, double *refexp,
                        int *refshlstart_by_atm, int *supshlstart_by_atm,
                        int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                        int *refatmprd_loc,
                        int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                        int *atm, int natm, int *bas, int nbas, int nbasaux,
                        double *env,
                        int *atmsup, int natmsup, int *bassup,
                        int nbassup, double *envsup, int nenvsup, char safe)
{
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas+nbasaux, env);
// initialize IJstart_lst and count_lst (i.e., buffer size)
    int Ish, Jsh, IJsh, IJstart, di, dj, dijkmax;
    int dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int nbas2 = nbas*(nbas+1)/2;
    int IJstart_lst[nbas2];
    size_t count_lst[nbas2];
    for(Ish=0, IJsh=0; Ish<nbas; ++Ish) {
        di = ao_loc[Ish+1] - ao_loc[Ish];
        IJstart = ao_loc[Ish]*(ao_loc[Ish]+1)/2;
        for(Jsh=0; Jsh<=Ish; ++Jsh, ++IJsh) {
            dj = ao_loc[Jsh+1] - ao_loc[Jsh];
            IJstart_lst[IJsh] = IJstart;
            IJstart += dj;

            dijkmax = di*dj * dkmax;
// MAX(INTBUFMAX, dijk) to ensure buffer is enough for at least one (i,j,k) shell
// *2 for safe mode
            count_lst[IJsh] = (2*dijkmax + MAX(INTBUFMAX, dijkmax)) * comp;
        }
    }

#pragma omp parallel
{
    int ish, jsh, ijsh, ij;
    double *env_loc = malloc(sizeof(double)*nenvsup);
    NPdcopy(env_loc, envsup, nenvsup);
#pragma omp for schedule(dynamic)
    for (ij = 0; ij < nbas*nbas; ij++) {
        ish = ij / nbas;
        jsh = ij % nbas;
        if (ish < jsh) {
            continue;
        }
        ijsh = ish*(ish+1)/2 + jsh;
        double *buf = malloc(sizeof(double)*(count_lst[ijsh]+cache_size));

        fill_sr3c2e_g_ijsh(intor, out, buf,
                           comp, cintopt,
                           ish, jsh, IJstart_lst[ijsh],
                           ao_loc, ao_locsup, shl_loc,
                           refuniqshl_map,
                           auxuniqshl_map, nbasauxuniq,
                           uniq_Rcut2s, refexp,
                           refshlstart_by_atm, supshlstart_by_atm,
                           uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                           refatmprd_loc,
                           supatmpr_loc, supatmpr_lst, nsupatmpr,
                           atm, natm, bas, nbas, nbasaux, env,
                           atmsup, natmsup, bassup, nbassup, env_loc, safe);

        free(buf);
    }
    free(env_loc);
} // omp parallel
}


// >>>>>>>> debug block
void fill_sr3c2e_g_nosave(int (*intor)(), double *out,
                   int comp, CINTOpt *cintopt,
                   int *ao_loc, int *ao_locsup, int *shl_loc,
                   int *refuniqshl_map,
                   int *auxuniqshl_map, int nbasauxuniq,
                   double *uniq_Rcut2s, double *refexp,
                   int *refshlstart_by_atm, int *supshlstart_by_atm,
                   int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                   int *refatmprd_loc,
                   int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                   int *atm, int natm, int *bas, int nbas, int nbasaux,
                   double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
/*
    ao_loc = concatenate([ao_loc, ao_locaux])
    ao_locsup = concatenate([ao_locsup, ao_locaux])
*/
{
    const int *supatmpr_i_lst = supatmpr_lst;
    const int *supatmpr_j_lst = supatmpr_lst + nsupatmpr;
    const int kshshift = nbassup - nbas;
    const int nbassupaux = nbassup + nbasaux;
    const int natmsupaux = natmsup + natm;

    int Ish, Jsh, IJsh, ISH, JSH, IJSH, Ishshift, Jshshift, ijsh, ijsh0, ijsh1, ish, jsh, I0, I1, J0, J1, IJstart;
    int Iatm, Jatm, IJatm, iatm, jatm, ijatm, ijatm0, ijatm1;
    int Katm, Ksh, Ksh0, Ksh1, ksh, K0, K1, KSH;
    int iptrxyz, jptrxyz, kptrxyz;
    int idij, idij0, idij1, Idij, Idij0;
    int di, dj, dk, dij, dijk, dijktot, dijkmax;
    int dimax = max_shlsize(ao_loc, nbas);
    int dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    int dktot = ao_loc[nbas+nbasaux] - ao_loc[nbas];
    int i,j,jmax,k,i0;
    int nao = ao_loc[nbas]-ao_loc[0];
    int nao2 = nao*(nao+1)/2;
    char skip;
    double ei, ej, Rijk2, Rcut2;
    double *uniq_Rcut2s_K, *ri, *rj, *rk;
    double rc[3];

    int shls[3];
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas+nbasaux, env);
// TODO: batch Ksh, which could be HUGE for big supercell.
    const int buf_size = dimax*dimax*dktot;
    const int tmp_size = dimax*dimax*dkmax*2;
    double *buf = malloc(sizeof(double)*(buf_size+tmp_size+cache_size));
    double *buf_L, *buf_Lk, *pbuf, *pbuf2, *cache, *outk;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

// >>>>>>>>
    unsigned int count = 0;
// <<<<<<<<

    for(Ish=0, IJsh=0; Ish<nbas; ++Ish) {
        Iatm = bassup[ATOM_OF+Ish*BAS_SLOTS];
        Ishshift = Ish - refshlstart_by_atm[Iatm];
        ISH = refuniqshl_map[Ish];
        ei = refexp[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        IJstart = I0*(I0+1)/2;
        di = I1 - I0;
        for(Jsh=0; Jsh<=Ish; ++Jsh, ++IJsh) {
            Jatm = bassup[ATOM_OF+Jsh*BAS_SLOTS];
            IJatm = (Iatm>=Jatm)?(Iatm*(Iatm+1)/2+Jatm):(Jatm*(Jatm+1)/2+Iatm);
            Jshshift = Jsh - refshlstart_by_atm[Jatm];
            JSH = refuniqshl_map[Jsh];
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            ej = refexp[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dij = di * dj;
            dijktot = dij * dktot;
            dijkmax = dij * dkmax;
            buf_L = buf;
            pbuf = buf_L + dijktot;
            pbuf2 = pbuf + dijkmax;
            cache = pbuf2 + dijkmax;
            for(i=0; i<dijktot; ++i) {
                buf_L[i] = 0;
            }

            idij0 = refatmprd_loc[IJatm];
            Idij0 = uniqshlpr_dij_loc[IJSH];
            idij1 = idij0 + uniqshlpr_dij_loc[IJSH+1] - Idij0;
            for(idij=idij0; idij<idij1; ++idij) {
                // get cutoff for IJ
                Idij = Idij0 + idij-idij0;
                uniq_Rcut2s_K = uniq_Rcut2s + Idij * nbasauxuniq;

                ijatm0 = supatmpr_loc[idij];
                ijatm1 = supatmpr_loc[idij+1];
                for(ijatm=ijatm0; ijatm<ijatm1; ++ijatm) {
                    iatm = supatmpr_i_lst[ijatm];
                    jatm = supatmpr_j_lst[ijatm];
                    iptrxyz = atmsup[PTR_COORD+iatm*ATM_SLOTS];
                    ri = envsup+iptrxyz;
                    jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                    rj = envsup+jptrxyz;
                    get_rc(rc, ri, rj, ei, ej);

                    // supmol atm index to supmol shl index
                    ish = supshlstart_by_atm[iatm] + Ishshift;
                    jsh = supshlstart_by_atm[jatm] + Jshshift;
                    shls[1] = ish;
                    shls[0] = jsh;

                    buf_Lk = buf_L;

                    for(Katm=natm; Katm<2*natm; ++Katm) { // first natm is ref
                        kptrxyz = atm[PTR_COORD+Katm*ATM_SLOTS];
                        rk = env+kptrxyz;
                        Rijk2 = get_dsqure(rc, rk);
                        Ksh0 = shl_loc[Katm];
                        Ksh1 = shl_loc[Katm+1];
                        for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                            KSH = auxuniqshl_map[Ksh-nbas];
                            Rcut2 = uniq_Rcut2s_K[KSH];
                            K0 = ao_loc[Ksh];
                            K1 = ao_loc[Ksh+1];
                            dk = K1 - K0;
                            dijk = dij * dk;

                            if(Rijk2<=Rcut2) {
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                ++count;
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                ksh = Ksh + kshshift;
                                shls[2] = ksh;

                                skip = 1;
                                if(safe) {
                                    envsup[PTR_RANGE_OMEGA] = 0.;
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                        envsup[PTR_RANGE_OMEGA] = omega;
                                        (*intor)(pbuf2, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache);
                                        for(i=0; i<dijk; ++i) {
                                            pbuf[i] -= pbuf2[i];
                                        }
                                    }
                                } else {
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                    }
                                }

                                if(!skip) {
                                    for(i=0; i<dijk; ++i) {
                                        buf_Lk[i] += pbuf[i];
                                    }
                                }
                            }

                            buf_Lk += dijk;
                        } // Ksh
                    } // Katm
                } // ijatm
            } // idij

            // commented for speed test
            /*
            buf_Lk = buf_L;
            outk = out;
            Ksh0 = shl_loc[natm];
            Ksh1 = shl_loc[2*natm];
            for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                dk = ao_loc[Ksh+1] - ao_loc[Ksh];
                for(k=0; k<dk; ++k) {
                    i0 = IJstart;
                    for(i=0; i<di; ++i) {
                        jmax = (Ish==Jsh)?(i+1):(dj);
                        for(j=0; j<jmax; ++j) {
                            outk[i0+j] = buf_Lk[i*dj+j];
                        }
                        i0 += I0+i+1;
                    }
                    outk += nao2;
                    buf_Lk += dij;
                }
            }
            */

            IJstart += dj;

        } // Jsh
    } // Ish

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(buf);

// >>>>>>>>
    printf("num significant shlpr %d\n", count);
// <<<<<<<<
}
// <<<<<<<<

void fill_sr3c2e_kk(int (*intor)(), double complex *out,
                   int comp, CINTOpt *cintopt,
                   double complex *expLk, int nkpts,
                   int *kptij_idx, int nkptijs,
                   int *ao_loc, int *ao_locsup, int *shl_loc,
                   int *refuniqshl_map,
                   int *auxuniqshl_map, int nbasauxuniq,
                   double *uniq_Rcut2s, double *refexp,
                   int *refshlstart_by_atm, int *supshlstart_by_atm,
                   int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                   int *refatmprd_loc,
                   int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                   int *atm, int natm, int *bas, int nbas, int nbasaux,
                   double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
/*
    ao_loc = concatenate([ao_loc, ao_locaux])
    ao_locsup = concatenate([ao_locsup, ao_locaux])
*/
{
    double *expLk_r = malloc(sizeof(double) * natmsup*nkpts * OF_CMPLX);
    double *expLk_i = expLk_r + natmsup*nkpts;
    double *expLk_r_iatm, *expLk_i_iatm, *expLk_r_jatm, *expLk_i_jatm;
    double *phi_r = malloc(sizeof(double) * 2*nkptijs * OF_CMPLX);
    double *phi_i = phi_r + nkptijs;
    double *psi_r = phi_i + nkptijs;
    double *psi_i = psi_r + nkptijs;
    double u_r, u_i, v_r, v_i, tmp;
    int ikij, ki, kj;
    const int *kptij_i_idx = kptij_idx;
    const int *kptij_j_idx = kptij_idx + nkptijs;
    double complex *outk, *outkk, *outkkij, *outkkji;
    int i;
    for (i = 0; i < natmsup*nkpts; i++) {
            expLk_r[i] = creal(expLk[i]);
            expLk_i[i] = cimag(expLk[i]);
    }

    const int *supatmpr_i_lst = supatmpr_lst;
    const int *supatmpr_j_lst = supatmpr_lst + nsupatmpr;
    const int kshshift = nbassup - nbas;
    const int nbassupaux = nbassup + nbasaux;
    const int natmsupaux = natmsup + natm;

    int Ish, Jsh, IJsh, ISH, JSH, IJSH, Ishshift, Jshshift, ijsh, ijsh0, ijsh1, ish, jsh, I0, I1, J0, J1, IJstart;
    int Iatm, Jatm, IJatm, iatm, jatm, ijatm, ijatm0, ijatm1;
    int Katm, Ksh, Ksh0, Ksh1, ksh, K0, K1, KSH;
    int iptrxyz, jptrxyz, kptrxyz;
    int idij, idij0, idij1, Idij, Idij0;
    int di, dj, dk, dij, dijk, dijktot, dijkmax;
    int dimax = max_shlsize(ao_loc, nbas);
    int dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    int dktot = ao_loc[nbas+nbasaux] - ao_loc[nbas];
    int j,k,kij;
    int nao = ao_loc[nbas]-ao_loc[0];
    int nao2 = nao*nao;
    int naux = ao_loc[nbas+nbasaux]-ao_loc[nbas];
    int nao2naux = nao*nao*naux;
    char skip;
    double ei, ej, Rijk2, Rcut2;
    double *uniq_Rcut2s_K, *ri, *rj, *rk;
    double rc[3];

    int shls[3];
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas+nbasaux, env);
// TODO: batch Ksh, which could be HUGE for big supercell.
    const int tmp_size = dimax*dimax*dkmax*2;
    double *buf = malloc(sizeof(double)*(tmp_size+cache_size));
    double *pbuf, *pbuf2, *cache;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

// >>>>>>>>
    unsigned int count = 0;
// <<<<<<<<

    for(Ish=0, IJsh=0; Ish<nbas; ++Ish) {
        Iatm = bassup[ATOM_OF+Ish*BAS_SLOTS];
        Ishshift = Ish - refshlstart_by_atm[Iatm];
        ISH = refuniqshl_map[Ish];
        ei = refexp[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        di = I1 - I0;
        for(Jsh=0; Jsh<=Ish; ++Jsh, ++IJsh) {
            Jatm = bassup[ATOM_OF+Jsh*BAS_SLOTS];
            IJatm = (Iatm>=Jatm)?(Iatm*(Iatm+1)/2+Jatm):(Jatm*(Jatm+1)/2+Iatm);
            Jshshift = Jsh - refshlstart_by_atm[Jatm];
            JSH = refuniqshl_map[Jsh];
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            ej = refexp[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dij = di * dj;
            dijktot = dij * dktot;
            dijkmax = dij * dkmax;
            pbuf = buf;
            pbuf2 = pbuf + dijkmax;
            cache = pbuf2 + dijkmax;

            idij0 = refatmprd_loc[IJatm];
            Idij0 = uniqshlpr_dij_loc[IJSH];
            idij1 = idij0 + uniqshlpr_dij_loc[IJSH+1] - Idij0;
            for(idij=idij0; idij<idij1; ++idij) {
                // get cutoff for IJ
                Idij = Idij0 + idij-idij0;
                uniq_Rcut2s_K = uniq_Rcut2s + Idij * nbasauxuniq;

                ijatm0 = supatmpr_loc[idij];
                ijatm1 = supatmpr_loc[idij+1];
                for(ijatm=ijatm0; ijatm<ijatm1; ++ijatm) {
                    iatm = supatmpr_i_lst[ijatm];
                    jatm = supatmpr_j_lst[ijatm];
                    iptrxyz = atmsup[PTR_COORD+iatm*ATM_SLOTS];
                    ri = envsup+iptrxyz;
                    jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                    rj = envsup+jptrxyz;
                    get_rc(rc, ri, rj, ei, ej);

                    // supmol atm index to supmol shl index
                    ish = supshlstart_by_atm[iatm] + Ishshift;
                    jsh = supshlstart_by_atm[jatm] + Jshshift;
                    shls[1] = ish;
                    shls[0] = jsh;

                    // pre-compute phase product for iatm/jatm pair
                    expLk_r_iatm = expLk_r + iatm * nkpts;
                    expLk_i_iatm = expLk_i + iatm * nkpts;
                    expLk_r_jatm = expLk_r + jatm * nkpts;
                    expLk_i_jatm = expLk_i + jatm * nkpts;
                    for(ikij=0; ikij<nkptijs; ++ikij) {
                        ki = kptij_i_idx[ikij];
                        kj = kptij_j_idx[ikij];
                        u_r = expLk_r_iatm[ki];
                        u_i = expLk_i_iatm[ki];
                        v_r = expLk_r_jatm[kj];
                        v_i = expLk_i_jatm[kj];
                        phi_r[ikij] = u_r*v_r+u_i*v_i;
                        phi_i[ikij] = u_r*v_i-u_i*v_r;
                        if(Ish != Jsh) {
                            u_r = expLk_r_jatm[ki];
                            u_i = expLk_i_jatm[ki];
                            v_r = expLk_r_iatm[kj];
                            v_i = expLk_i_iatm[kj];
                            psi_r[ikij] = u_r*v_r+u_i*v_i;
                            psi_i[ikij] = u_r*v_i-u_i*v_r;
                        }
                    }

                    for(Katm=natm; Katm<2*natm; ++Katm) { // first natm is ref
                        kptrxyz = atm[PTR_COORD+Katm*ATM_SLOTS];
                        rk = env+kptrxyz;
                        Rijk2 = get_dsqure(rc, rk);
                        Ksh0 = shl_loc[Katm];
                        Ksh1 = shl_loc[Katm+1];
                        for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                            KSH = auxuniqshl_map[Ksh-nbas];
                            Rcut2 = uniq_Rcut2s_K[KSH];
                            K0 = ao_loc[Ksh]-nao;
                            K1 = ao_loc[Ksh+1]-nao;
                            dk = K1 - K0;
                            dijk = dij * dk;

                            if(Rijk2<=Rcut2) {
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                ++count;
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                ksh = Ksh + kshshift;
                                shls[2] = ksh;

                                skip = 1;
                                if(safe) {
                                    envsup[PTR_RANGE_OMEGA] = 0.;
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                        envsup[PTR_RANGE_OMEGA] = omega;
                                        (*intor)(pbuf2, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache);
                                        for(i=0; i<dijk; ++i) {
                                            pbuf[i] -= pbuf2[i];
                                        }
                                    }
                                } else {
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                    }
                                }

                                if(!skip) {
                                    outk = out;
                                    for(ikij=0; ikij<nkptijs; ++ikij) {
                                        u_r = phi_r[ikij];
                                        u_i = phi_i[ikij];
                                        if(Ish != Jsh) {
                                            v_r = psi_r[ikij];
                                            v_i = psi_i[ikij];
                                        }
                                        outkk = outk + K0*nao2;
                                        for(k=0, kij=0; k<dk; ++k) {
                                            outkkij = outkk + I0*nao + J0;
                                            for(i=0; i<di; ++i) {
                                                for(j=0; j<dj; ++j, ++kij) {
                                                    tmp = pbuf[kij];
                                                    outkkij[j] += tmp*u_r +
                                                        tmp*u_i*_Complex_I;
                                                }
                                                outkkij += nao;
                                            }
                                            outkk += nao2;
                                        }
                                        if(Ish != Jsh) {
                                            outkk = outk + K0*nao2;
                                            for(k=0, kij=0; k<dk; ++k) {
                                                outkkji = outkk + J0*nao + I0;
                                                for(i=0; i<di; ++i) {
                                                    for(j=0; j<dj; ++j, ++kij) {
                                                        tmp = pbuf[kij];
                                                        outkkji[j*nao] += tmp*v_r +
                                                            tmp*v_i*_Complex_I;
                                                    }
                                                    outkkji += 1;
                                                }
                                                outkk += nao2;
                                            }
                                        }
                                        outk += nao2naux;

                                    } // ikij
                                }
                            }
                        } // Ksh
                    } // Katm
                } // ijsh
            } // idij
        } // Jsh
    } // Ish

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(phi_r);
    free(expLk_r);
    free(buf);

// >>>>>>>>
    printf("num significant shlpr %d\n", count);
// <<<<<<<<
}

void fill_sr3c2e_kk_ijsh(int (*intor)(), double complex *out, double *buf,
                         int comp, CINTOpt *cintopt,
                         int Ish, int Jsh,
                         double *expLk_r, double *expLk_i, int nkpts,
                         int *kptij_idx, int nkptijs,
                         int *ao_loc, int *ao_locsup, int *shl_loc,
                         int *refuniqshl_map,
                         int *auxuniqshl_map, int nbasauxuniq,
                         double *uniq_Rcut2s, double *refexp,
                         int *refshlstart_by_atm, int *supshlstart_by_atm,
                         int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                         int *refatmprd_loc,
                         int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                         int *atm, int natm, int *bas, int nbas, int nbasaux,
                         double *env,
                         int *atmsup, int natmsup, int *bassup,
                         int nbassup, double *envsup, char safe)
/*
    ao_loc = concatenate([ao_loc, ao_locaux])
    ao_locsup = concatenate([ao_locsup, ao_locaux])
*/
{
    int shls[3];
    const int *supatmpr_i_lst = supatmpr_lst;
    const int *supatmpr_j_lst = supatmpr_lst + nsupatmpr;
    const int kshshift = nbassup - nbas;
    const int nbassupaux = nbassup + nbasaux;
    const int natmsupaux = natmsup + natm;

    const int *kptij_i_idx = kptij_idx;
    const int *kptij_j_idx = kptij_idx + nkptijs;
    double *phi_r = malloc(sizeof(double) * 2*nkptijs * OF_CMPLX);
    double *phi_i = phi_r + nkptijs;
    double *psi_r = phi_i + nkptijs;
    double *psi_i = psi_r + nkptijs;

    int i, j, k, kij, ikij, ki, kj;
    int I0 = ao_loc[Ish], J0 = ao_loc[Jsh];
    int nao = ao_loc[nbas]-ao_loc[0];
    int nao2 = nao*nao;
    int naux = ao_loc[nbas+nbasaux]-ao_loc[nbas];
    int nao2naux = nao*nao*naux;
    char skip;
    char eq_IJsh = Ish == Jsh;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

// atm/shl info
    int Iatm, Jatm, IJatm, Katm, katm, ijatm0, ijatm1, ijatm, iatm, jatm, iptrxyz, jptrxyz, kptrxyz;
    double *ri, *rj, *rk, rc[3], Rijk2, Rcut2;
    Iatm = bassup[ATOM_OF+Ish*BAS_SLOTS];
    Jatm = bassup[ATOM_OF+Jsh*BAS_SLOTS];
    IJatm = (Iatm>=Jatm)?(Iatm*(Iatm+1)/2+Jatm):(Jatm*(Jatm+1)/2+Iatm);

    int ish, jsh, Ishshift, Jshshift, ISH, JSH, IJSH, Ksh, Ksh0, Ksh1, KSH, ksh;
    Ishshift = Ish - refshlstart_by_atm[Iatm];
    Jshshift = Jsh - refshlstart_by_atm[Jatm];
    ISH = refuniqshl_map[Ish];
    JSH = refuniqshl_map[Jsh];
    IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);

    double ei, ej;
    ei = refexp[Ish];
    ej = refexp[Jsh];

// partition ksh_loc by atom and buf size
    int di, dj, dk, dkmax, dkbuf, dktot, dij, dijk, dijkmax, dijkbuf, dijktot, nkshloc;
    int kshloc[nbasaux+1], katmloc[natm+1], kloc, kloc0, kloc1;
    di = ao_loc[Ish+1] - ao_loc[Ish];
    dj = ao_loc[Jsh+1] - ao_loc[Jsh];
    dij = di * dj;
    dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    dkbuf = MAX(INTBUFMAX/dij, dkmax);
    dijkbuf = dij * dkbuf;
    dijkmax = dij * dkmax;
    nkshloc = shloc_partition_by_atom(kshloc, katmloc, ao_loc, shl_loc,
                                      natm, 2*natm, dkbuf);

    double *buf_L1, *buf_L2, *pbuf, *pbuf2, *cache;
    buf_L1 = buf;
    buf_L2 = buf_L1 + nkptijs * dijkbuf * OF_CMPLX;
    if(eq_IJsh) {
        pbuf = buf_L2;
    } else {
        pbuf = buf_L2 + nkptijs * dijkbuf * OF_CMPLX;
    }
    pbuf2 = pbuf + dijkmax;
    cache = pbuf2 + dijkmax;

    double complex *pout1, *out1_k, *outk1_k, *buf_Lk1, *buf_Lk1_k;
    double complex *pout2, *out2_k, *outk2_k, *buf_Lk2, *buf_Lk2_k;
    double *expLk_r_iatm, *expLk_i_iatm, *expLk_r_jatm, *expLk_i_jatm;
    double u_r, u_i, v_r, v_i, tmp;

    int idij0, idij1, idij, Idij0, Idij;
    double * uniq_Rcut2s_K;
    idij0 = refatmprd_loc[IJatm];
    Idij0 = uniqshlpr_dij_loc[IJSH];
    idij1 = idij0 + uniqshlpr_dij_loc[IJSH+1] - Idij0;

    pout1 = out + I0 * nao + J0;
    if(!eq_IJsh) {
        pout2 = out + J0 * nao + I0;
    }
    for(Katm=natm; Katm<2*natm; ++Katm) { // first natm is ref
        kptrxyz = atm[PTR_COORD+Katm*ATM_SLOTS];
        rk = env+kptrxyz;

        katm = Katm - natm;
        kloc0 = katmloc[katm];
        kloc1 = katmloc[katm+1];

        for(kloc=kloc0; kloc<kloc1; ++kloc) {

            Ksh0 = kshloc[kloc];
            Ksh1 = kshloc[kloc+1];
            dktot = ao_loc[Ksh1] - ao_loc[Ksh0];
            dijktot = dij * dktot;

            for(i=0; i<nkptijs*dijktot*OF_CMPLX; ++i) {
                buf_L1[i] = 0;
            }
            if(!eq_IJsh) {
                for(i=0; i<nkptijs*dijktot*OF_CMPLX; ++i) {
                    buf_L2[i] = 0;
                }
            }

            for(idij=idij0; idij<idij1; ++idij) {
                // get cutoff for IJ
                Idij = Idij0 + idij-idij0;
                uniq_Rcut2s_K = uniq_Rcut2s + Idij * nbasauxuniq;

                ijatm0 = supatmpr_loc[idij];
                ijatm1 = supatmpr_loc[idij+1];
                for(ijatm=ijatm0; ijatm<ijatm1; ++ijatm) {
                    iatm = supatmpr_i_lst[ijatm];
                    jatm = supatmpr_j_lst[ijatm];
                    iptrxyz = atmsup[PTR_COORD+iatm*ATM_SLOTS];
                    ri = envsup+iptrxyz;
                    jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                    rj = envsup+jptrxyz;
                    get_rc(rc, ri, rj, ei, ej);
                    Rijk2 = get_dsqure(rc, rk);

                    // pre-compute phase product for iatm/jatm pair
                    expLk_r_iatm = expLk_r + iatm * nkpts;
                    expLk_i_iatm = expLk_i + iatm * nkpts;
                    expLk_r_jatm = expLk_r + jatm * nkpts;
                    expLk_i_jatm = expLk_i + jatm * nkpts;
                    for(ikij=0; ikij<nkptijs; ++ikij) {
                        ki = kptij_i_idx[ikij];
                        kj = kptij_j_idx[ikij];
                        u_r = expLk_r_iatm[ki];
                        u_i = expLk_i_iatm[ki];
                        v_r = expLk_r_jatm[kj];
                        v_i = expLk_i_jatm[kj];
                        phi_r[ikij] = u_r*v_r+u_i*v_i;
                        phi_i[ikij] = u_r*v_i-u_i*v_r;
                        if(!eq_IJsh) {
                            u_r = expLk_r_jatm[ki];
                            u_i = expLk_i_jatm[ki];
                            v_r = expLk_r_iatm[kj];
                            v_i = expLk_i_iatm[kj];
                            psi_r[ikij] = u_r*v_r+u_i*v_i;
                            psi_i[ikij] = u_r*v_i-u_i*v_r;
                        }
                    }

                    // supmol atm index to supmol shl index
                    ish = supshlstart_by_atm[iatm] + Ishshift;
                    jsh = supshlstart_by_atm[jatm] + Jshshift;
                    shls[1] = ish;
                    shls[0] = jsh;

                    buf_Lk1 = buf_L1;   // shape: nkptij,dktot,dij
                    if(!eq_IJsh) {
                        buf_Lk2 = buf_L2;
                    }
                    for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                        KSH = auxuniqshl_map[Ksh-nbas];
                        Rcut2 = uniq_Rcut2s_K[KSH];
                        dk = ao_loc[Ksh+1] - ao_loc[Ksh];
                        dijk = dij * dk;

                        if(Rijk2<=Rcut2) {
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                            // ++count;
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                            ksh = Ksh + kshshift;
                            shls[2] = ksh;

                            skip = 1;
                            if(safe) {
                                envsup[PTR_RANGE_OMEGA] = 0.;
                                if ((*intor)(pbuf, NULL, shls, atmsup,
                                             natmsupaux, bassup, nbassupaux,
                                             envsup, cintopt, cache)) {
                                    skip = 0;
                                    envsup[PTR_RANGE_OMEGA] = omega;
                                    (*intor)(pbuf2, NULL, shls, atmsup,
                                             natmsupaux, bassup, nbassupaux,
                                             envsup, cintopt, cache);
                                    for(i=0; i<dijk; ++i) {
                                        pbuf[i] -= pbuf2[i];
                                    }
                                }
                            } else {
                                if ((*intor)(pbuf, NULL, shls, atmsup,
                                             natmsupaux, bassup, nbassupaux,
                                             envsup, cintopt, cache)) {
                                    skip = 0;
                                }
                            }

                            if(!skip) {

                                buf_Lk1_k = buf_Lk1;
                                if(!eq_IJsh) {
                                    buf_Lk2_k = buf_Lk2;
                                }
                                for(ikij=0; ikij<nkptijs; ++ikij) {
                                    u_r = phi_r[ikij];
                                    u_i = phi_i[ikij];
                                    if(Ish != Jsh) {
                                        v_r = psi_r[ikij];
                                        v_i = psi_i[ikij];
                                    }

                                    for(kij=0; kij<dijk; kij++) {
                                        tmp = pbuf[kij];
                                        buf_Lk1_k[kij] += tmp*u_r +
                                                        tmp*u_i*_Complex_I;
                                    }
                                    buf_Lk1_k += dijktot;   // shift kpt idx

                                    if(!eq_IJsh) {
                                        for(kij=0; kij<dijk; kij++) {
                                            tmp = pbuf[kij];
                                            buf_Lk2_k[kij] += tmp*v_r +
                                                            tmp*v_i*_Complex_I;
                                        }
                                        buf_Lk2_k += dijktot;   // shift kpt idx
                                    }

                                } // ikij
                            } // if skip
                        } // if cutoff

                        buf_Lk1 += dijk;
                        if(!eq_IJsh) {
                            buf_Lk2 += dijk;
                        }
                    } // Ksh

                } // ijatm
            } // idij

            out1_k = pout1; // shape: nkptij, naux, nao2
            buf_Lk1_k = buf_L1; // shape: nkptij, dktot, dij
            for(ikij=0; ikij<nkptijs; ++ikij) {

                outk1_k = out1_k;
                for(k=0, kij=0; k<dktot; ++k) {
                    for(i=0; i<di; ++i) {
                        for(j=0; j<dj; ++j, ++kij) {
                            outk1_k[i*nao+j] += buf_Lk1_k[kij];
                        }
                    }
                    outk1_k += nao2;
                }
                out1_k += nao2naux;
                buf_Lk1_k += dijktot;
            }
            pout1 += dktot * nao2;

            if(!eq_IJsh) {

                out2_k = pout2;
                buf_Lk2_k = buf_L2;
                for(ikij=0; ikij<nkptijs; ++ikij) {
                    outk2_k = out2_k;
                    for(k=0, kij=0; k<dktot; ++k) {
                        for(i=0; i<di; ++i) {
                            for(j=0; j<dj; ++j, ++kij) {
                                outk2_k[j*nao+i] += buf_Lk2_k[kij];
                            }
                        }
                        outk2_k += nao2;
                    }
                    out2_k += nao2naux;
                    buf_Lk2_k += dijktot;
                }
                pout2 += dktot * nao2;
            } // if eq_IJsh

        } // kloc
    } // Katm

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(phi_r);

// >>>>>>>>
    // printf("num significant shlpr %d\n", count);
// <<<<<<<<
}

void PBCnr_sr3c2e_kk_drv(int (*intor)(), double complex *out,
                         int comp, CINTOpt *cintopt,
                         double complex *expLk, int nkpts,
                         int *kptij_idx, int nkptijs,
                         int *ao_loc, int *ao_locsup, int *shl_loc,
                         int *refuniqshl_map,
                         int *auxuniqshl_map, int nbasauxuniq,
                         double *uniq_Rcut2s, double *refexp,
                         int *refshlstart_by_atm, int *supshlstart_by_atm,
                         int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                         int *refatmprd_loc,
                         int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                         int *atm, int natm, int *bas, int nbas, int nbasaux,
                         double *env,
                         int *atmsup, int natmsup, int *bassup,
                         int nbassup, double *envsup, int nenvsup, char safe)
{
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas+nbasaux, env);
// initialize count_lst (i.e., buffer size)
    int Ish, Jsh, IJsh, di, dj, dijkmax;
    int dkmax = GTOmax_shell_dim(ao_loc, shls_slice+4, 1);
    int nbas2 = nbas*(nbas+1)/2;
    size_t count_lst[nbas2], count;
    for(Ish=0, IJsh=0; Ish<nbas; ++Ish) {
        di = ao_loc[Ish+1] - ao_loc[Ish];
        for(Jsh=0; Jsh<=Ish; ++Jsh, ++IJsh) {
            dj = ao_loc[Jsh+1] - ao_loc[Jsh];
            dijkmax = di*dj * dkmax;
// 2*dijk for safe mode
            count = dijkmax*2;
// MAX(INTBUFMAX, dijk) to ensure buffer is enough for at least one (i,j,k) shell
            if(Ish == Jsh) {
                count += MAX(INTBUFMAX, dijkmax)*OF_CMPLX;
            } else {
// *2 when Ish!=Jsh for (Jsh,Ish)
                count += MAX(INTBUFMAX, dijkmax)*OF_CMPLX*2;
            }
            count_lst[IJsh] = count * nkptijs * comp;
        }
    }

// buffer for expLk
    double *expLk_r = malloc(sizeof(double) * natmsup*nkpts * OF_CMPLX);
    double *expLk_i = expLk_r + natmsup*nkpts;
    int i;
    for (i = 0; i < natmsup*nkpts; i++) {
        expLk_r[i] = creal(expLk[i]);
        expLk_i[i] = cimag(expLk[i]);
    }

#pragma omp parallel
{
    int ish, jsh, ijsh, ij;
    double *env_loc = malloc(sizeof(double)*nenvsup);
    NPdcopy(env_loc, envsup, nenvsup);
#pragma omp for schedule(dynamic)
    for (ij = 0; ij < nbas*nbas; ij++) {
        ish = ij / nbas;
        jsh = ij % nbas;
        if (ish < jsh) {
            continue;
        }
        ijsh = ish*(ish+1)/2 + jsh;
        double *buf = malloc(sizeof(double)*(count_lst[ijsh]+cache_size));

        fill_sr3c2e_kk_ijsh(intor, out, buf,
                            comp, cintopt,
                            ish, jsh,
                            expLk_r, expLk_i, nkpts, kptij_idx, nkptijs,
                            ao_loc, ao_locsup, shl_loc,
                            refuniqshl_map,
                            auxuniqshl_map, nbasauxuniq,
                            uniq_Rcut2s, refexp,
                            refshlstart_by_atm, supshlstart_by_atm,
                            uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                            refatmprd_loc,
                            supatmpr_loc, supatmpr_lst, nsupatmpr,
                            atm, natm, bas, nbas, nbasaux, env,
                            atmsup, natmsup, bassup, nbassup, env_loc, safe);

        free(buf);
    }
    free(env_loc);
} // omp parallel
    free(expLk_r);
}

void fill_sr3c2e_kk_bvk(int (*intor)(), double complex *out,
                   int comp, CINTOpt *cintopt,
                   double complex *expLk, int nkpts,
                   int *kptij_idx, int nkptijs,
                   int *ao_loc, int *ao_locsup, int *shl_loc,
                   int *refuniqshl_map,
                   int *auxuniqshl_map, int nbasauxuniq,
                   double *uniq_Rcut2s, double *refexp,
                   int *refshlstart_by_atm, int *supshlstart_by_atm,
                   int *uniqshlpr_dij_loc, // IJSH,IJSH+1 --> idij0, idij1
                   int *refatmprd_loc,
                   int *supatmpr_loc, int *supatmpr_lst, int nsupatmpr,
                   int *atm, int natm, int *bas, int nbas, int nbasaux,
                   double *env,
                   int *atmsup, int natmsup, int *bassup,
                   int nbassup, double *envsup, char safe)
/*
    ao_loc = concatenate([ao_loc, ao_locaux])
    ao_locsup = concatenate([ao_locsup, ao_locaux])
*/
{
    double *expLk_r = malloc(sizeof(double) * natmsup*nkpts * OF_CMPLX);
    double *expLk_i = expLk_r + natmsup*nkpts;
    double *expLk_r_iatm, *expLk_i_iatm, *expLk_r_jatm, *expLk_i_jatm;
    double *phi_r = malloc(sizeof(double) * 2*nkptijs * OF_CMPLX);
    double *phi_i = phi_r + nkptijs;
    double *psi_r = phi_i + nkptijs;
    double *psi_i = psi_r + nkptijs;
    double u_r, u_i, v_r, v_i, tmp;
    int ikij, ki, kj;
    const int *kptij_i_idx = kptij_idx;
    const int *kptij_j_idx = kptij_idx + nkptijs;
    double complex *outk, *outkk, *outkkij, *outkkji;
    int i;
    for (i = 0; i < natmsup*nkpts; i++) {
            expLk_r[i] = creal(expLk[i]);
            expLk_i[i] = cimag(expLk[i]);
    }

    const int *supatmpr_i_lst = supatmpr_lst;
    const int *supatmpr_j_lst = supatmpr_lst + nsupatmpr;
    const int kshshift = nbassup - nbas;
    const int nbassupaux = nbassup + nbasaux;
    const int natmsupaux = natmsup + natm;

    int Ish, Jsh, IJsh, ISH, JSH, IJSH, Ishshift, Jshshift, ijsh, ijsh0, ijsh1, ish, jsh, I0, I1, J0, J1, IJstart;
    int Iatm, Jatm, IJatm, iatm, jatm, ijatm, ijatm0, ijatm1;
    int Katm, Ksh, Ksh0, Ksh1, ksh, K0, K1, KSH;
    int iptrxyz, jptrxyz, kptrxyz;
    int idij, idij0, idij1, Idij, Idij0;
    int di, dj, dk, dij, dijk, dijktot, dijkmax;
    int dimax = max_shlsize(ao_loc, nbas);
    int dijmax = dimax * dimax;
    // int dkmax = max_shlsize(ao_loc+nbas, nbasaux);
    int dkmax = INTBUFMAX / dijmax;
    int kshloc[nbasaux+1];
    int *kshloc_ = kshloc;
    int nkshloc = 0;
    for(Katm=natm; Katm < 2*natm; ++Katm) {
        Ksh0 = shl_loc[Katm];
        Ksh1 = shl_loc[Katm+1];
        int nkshloc_ = shloc_partition(kshloc_, ao_loc, Ksh0, Ksh1, dkmax);
        nkshloc += nkshloc_;
        printf("Katm= %d\n", Katm);
        for(i=0; i<nkshloc_+1; ++i) {
            printf("%d %d\n", i, kshloc_[i]);
        }
        kshloc_ += nkshloc_;
    }
    printf("nkshloc= %d\n", nkshloc);
    for(i=0; i<nkshloc+1; ++i) {
        printf("%d %d\n", i, kshloc[i]);
    }
    exit(1);
    int dktot = ao_loc[nbas+nbasaux] - ao_loc[nbas];
    int j,k,kij;
    int nao = ao_loc[nbas]-ao_loc[0];
    int nao2 = nao*nao;
    int naux = ao_loc[nbas+nbasaux]-ao_loc[nbas];
    int nao2naux = nao*nao*naux;
    char skip;
    double ei, ej, Rijk2, Rcut2;
    double *uniq_Rcut2s_K, *ri, *rj, *rk;
    double rc[3];

    int shls[3];
    int shls_slice[6];
    shls_slice[0] = 0;
    shls_slice[1] = nbas;
    shls_slice[2] = 0;
    shls_slice[3] = nbas;
    shls_slice[4] = nbas;
    shls_slice[5] = nbas+nbasaux;
    const int cache_size = GTOmax_cache_size(intor, shls_slice, 3,
                                             atm, natm, bas, nbas+nbasaux, env);
// TODO: batch Ksh, which could be HUGE for big supercell.
    const int tmp_size = dimax*dimax*dkmax*2;
    double *buf = malloc(sizeof(double)*(tmp_size+cache_size));
    double *pbuf, *pbuf2, *cache;
    double omega = ABS(envsup[PTR_RANGE_OMEGA]);

// >>>>>>>>
    unsigned int count = 0;
// <<<<<<<<

    for(Ish=0, IJsh=0; Ish<nbas; ++Ish) {
        Iatm = bassup[ATOM_OF+Ish*BAS_SLOTS];
        Ishshift = Ish - refshlstart_by_atm[Iatm];
        ISH = refuniqshl_map[Ish];
        ei = refexp[Ish];
        I0 = ao_loc[Ish];
        I1 = ao_loc[Ish+1];
        di = I1 - I0;
        for(Jsh=0; Jsh<=Ish; ++Jsh, ++IJsh) {
            Jatm = bassup[ATOM_OF+Jsh*BAS_SLOTS];
            IJatm = (Iatm>=Jatm)?(Iatm*(Iatm+1)/2+Jatm):(Jatm*(Jatm+1)/2+Iatm);
            Jshshift = Jsh - refshlstart_by_atm[Jatm];
            JSH = refuniqshl_map[Jsh];
            IJSH = (ISH>=JSH)?(ISH*(ISH+1)/2+JSH):(JSH*(JSH+1)/2+ISH);
            ej = refexp[Jsh];
            J0 = ao_loc[Jsh];
            J1 = ao_loc[Jsh+1];
            dj = J1 - J0;
            dij = di * dj;
            dijktot = dij * dktot;
            dijkmax = dij * dkmax;
            pbuf = buf;
            pbuf2 = pbuf + dijkmax;
            cache = pbuf2 + dijkmax;

            idij0 = refatmprd_loc[IJatm];
            Idij0 = uniqshlpr_dij_loc[IJSH];
            idij1 = idij0 + uniqshlpr_dij_loc[IJSH+1] - Idij0;

            for(Katm=natm; Katm<2*natm; ++Katm) { // first natm is ref
                kptrxyz = atm[PTR_COORD+Katm*ATM_SLOTS];
                rk = env+kptrxyz;
                Ksh0 = shl_loc[Katm];
                Ksh1 = shl_loc[Katm+1];

                for(idij=idij0; idij<idij1; ++idij) {
                    // get cutoff for IJ
                    Idij = Idij0 + idij-idij0;
                    uniq_Rcut2s_K = uniq_Rcut2s + Idij * nbasauxuniq;

                    ijatm0 = supatmpr_loc[idij];
                    ijatm1 = supatmpr_loc[idij+1];
                    for(ijatm=ijatm0; ijatm<ijatm1; ++ijatm) {
                        iatm = supatmpr_i_lst[ijatm];
                        jatm = supatmpr_j_lst[ijatm];
                        iptrxyz = atmsup[PTR_COORD+iatm*ATM_SLOTS];
                        ri = envsup+iptrxyz;
                        jptrxyz = atmsup[PTR_COORD+jatm*ATM_SLOTS];
                        rj = envsup+jptrxyz;
                        get_rc(rc, ri, rj, ei, ej);

                        Rijk2 = get_dsqure(rc, rk);

                        // supmol atm index to supmol shl index
                        ish = supshlstart_by_atm[iatm] + Ishshift;
                        jsh = supshlstart_by_atm[jatm] + Jshshift;
                        shls[1] = ish;
                        shls[0] = jsh;

                        // pre-compute phase product for iatm/jatm pair
                        expLk_r_iatm = expLk_r + iatm * nkpts;
                        expLk_i_iatm = expLk_i + iatm * nkpts;
                        expLk_r_jatm = expLk_r + jatm * nkpts;
                        expLk_i_jatm = expLk_i + jatm * nkpts;
                        for(ikij=0; ikij<nkptijs; ++ikij) {
                            ki = kptij_i_idx[ikij];
                            kj = kptij_j_idx[ikij];
                            u_r = expLk_r_iatm[ki];
                            u_i = expLk_i_iatm[ki];
                            v_r = expLk_r_jatm[kj];
                            v_i = expLk_i_jatm[kj];
                            phi_r[ikij] = u_r*v_r+u_i*v_i;
                            phi_i[ikij] = u_r*v_i-u_i*v_r;
                            if(Ish != Jsh) {
                                u_r = expLk_r_jatm[ki];
                                u_i = expLk_i_jatm[ki];
                                v_r = expLk_r_iatm[kj];
                                v_i = expLk_i_iatm[kj];
                                psi_r[ikij] = u_r*v_r+u_i*v_i;
                                psi_i[ikij] = u_r*v_i-u_i*v_r;
                            }
                        }

                        for(Ksh=Ksh0; Ksh<Ksh1; ++Ksh) {
                            KSH = auxuniqshl_map[Ksh-nbas];
                            Rcut2 = uniq_Rcut2s_K[KSH];
                            K0 = ao_loc[Ksh]-nao;
                            K1 = ao_loc[Ksh+1]-nao;
                            dk = K1 - K0;
                            dijk = dij * dk;

                            if(Rijk2<=Rcut2) {
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                ++count;
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                ksh = Ksh + kshshift;
                                shls[2] = ksh;

                                skip = 1;
                                if(safe) {
                                    envsup[PTR_RANGE_OMEGA] = 0.;
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                        envsup[PTR_RANGE_OMEGA] = omega;
                                        (*intor)(pbuf2, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache);
                                        for(i=0; i<dijk; ++i) {
                                            pbuf[i] -= pbuf2[i];
                                        }
                                    }
                                } else {
                                    if ((*intor)(pbuf, NULL, shls, atmsup,
                                                 natmsupaux, bassup, nbassupaux,
                                                 envsup, cintopt, cache)) {
                                        skip = 0;
                                    }
                                }

                                if(!skip) {
                                    outk = out;
                                    for(ikij=0; ikij<nkptijs; ++ikij) {
                                        u_r = phi_r[ikij];
                                        u_i = phi_i[ikij];
                                        if(Ish != Jsh) {
                                            v_r = psi_r[ikij];
                                            v_i = psi_i[ikij];
                                        }
                                        outkk = outk + K0*nao2;
                                        for(k=0, kij=0; k<dk; ++k) {
                                            outkkij = outkk + I0*nao + J0;
                                            for(i=0; i<di; ++i) {
                                                for(j=0; j<dj; ++j, ++kij) {
                                                    tmp = pbuf[kij];
                                                    outkkij[j] += tmp*u_r +
                                                        tmp*u_i*_Complex_I;
                                                }
                                                outkkij += nao;
                                            }
                                            outkk += nao2;
                                        }
                                        if(Ish != Jsh) {
                                            outkk = outk + K0*nao2;
                                            for(k=0, kij=0; k<dk; ++k) {
                                                outkkji = outkk + J0*nao + I0;
                                                for(i=0; i<di; ++i) {
                                                    for(j=0; j<dj; ++j, ++kij) {
                                                        tmp = pbuf[kij];
                                                        outkkji[j*nao] += tmp*v_r +
                                                            tmp*v_i*_Complex_I;
                                                    }
                                                    outkkji += 1;
                                                }
                                                outkk += nao2;
                                            }
                                        }
                                        outk += nao2naux;

                                    } // ikij
                                }
                            }
                        } // Ksh
                    } // Katm
                } // ijsh
            } // idij
        } // Jsh
    } // Ish

    if(safe) {
        envsup[PTR_RANGE_OMEGA] = -omega;
    }

    free(phi_r);
    free(expLk_r);
    free(buf);

// >>>>>>>>
    printf("num significant shlpr %d\n", count);
// <<<<<<<<
}
