#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

"""
Hong-Zhou Ye and Timothy C. Berkelbach, to be published.
"""

import h5py
import ctypes
import numpy as np
from scipy.special import gamma, comb

from pyscf import gto as mol_gto
from pyscf.scf import _vhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import BOHR
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.supmol import (get_refuniq_map, binary_search, get_norm,
                                 get_dist_mat)
from pyscf.pbc.df.intor_j2c import (Gamma, get_multipole, make_supmol_j2c,
                                    get_atom_Rcuts_2c)


def fintor_sreri(mol, intor, shls_slice, omega, safe):
    if safe:
        I = mol.intor(intor, shls_slice=shls_slice)
        with mol.with_range_coulomb(abs(omega)):
            I -= mol.intor(intor, shls_slice=shls_slice)
    else:
        with mol.with_range_coulomb(-abs(omega)):
            I = mol.intor(intor, shls_slice=shls_slice)
    return I
def get_schwartz_data(bas_lst, omega, dijs_lst=None, keep1ctr=True, safe=True):
    """
        if dijs_lst is None:
            "2c"-mode:  Q = 2-norm[(a|a)]^(1/2)
        else:
            "4c"-mode:  Q = 2-norm[(ab|ab)]^(1/2)
    """
    def get1ctr(bas_lst):
        """ For a shell consists of multiple contracted GTOs, keep only the one with the greatest weight on the most diffuse primitive GTOs (since others are likely core orbitals).
        """
        bas_lst_new = []
        for bas in bas_lst:
            nprim = len(bas) - 1
            nctr = len(bas[1]) - 1
            if nprim == 1 or nctr == 1:  # prim shell or ctr shell with 1 cGTO
                bas_new = bas
            else:
                ecs = np.array(bas[1:])
                es = ecs[:,0]
                imin = es.argmin()
                jmax = abs(ecs[imin,1:]).argmax()
                cs = ecs[:,jmax+1]
                bas_new = [bas[0]] + [(e,c) for e,c in zip(es,cs)]
            bas_lst_new.append(bas_new)
        return bas_lst_new
    if keep1ctr:
        bas_lst = get1ctr(bas_lst)
    if dijs_lst is None:
        mol = mol_gto.M(atom="H 0 0 0", basis=bas_lst, spin=None)
        nbas = mol.nbas
        intor = "int2c2e"
        Qs = np.zeros(nbas)
        for k in range(nbas):
            shls_slice = (k,k+1,k,k+1)
            I = fintor_sreri(mol, intor, shls_slice, omega, safe)
            Qs[k] = get_norm( I )**0.5
    else:
        def compute1_(mol, dij, intor, shls_slice, omega, safe):
            mol._env[mol._atm[1,mol_gto.PTR_COORD]] = dij
            return get_norm(
                        fintor_sreri(mol, intor, shls_slice, omega, safe)
                    )**0.5
        mol = mol_gto.M(atom="H 0 0 0; H 0 0 0", basis=bas_lst, spin=None)
        nbas = mol.nbas//2
        n2 = nbas*(nbas+1)//2
        if len(dijs_lst) != n2:
            raise RuntimeError("dijs_lst has wrong len (expecting %d; got %d)" % (n2, len(dijs_lst)))
        intor = "int2e"
        Qs = [None] * n2
        ij = 0
        for i in range(nbas):
            for j in range(i+1):
                j_ = j + nbas
                shls_slice = (i,i+1,j_,j_+1,i,i+1,j_,j_+1)
                dijs = dijs_lst[ij]
                Qs[ij] = [compute1_(mol, dij, intor, shls_slice, omega, safe)
                          for dij in dijs]
                ij += 1

    return Qs

def get_ovlp_dcut(bas_lst, precision, r0=None):
    """ Given a list of basis, determine cutoff radius for the ovlp between each unique shell pair to drop below "precision".

    Return:
        1d array of length nbas*(nbas+1)//2 with nbas=len(bas_lst).
    """
    mol = mol_gto.M(atom="H 0 0 0; H 0 0 0", basis=bas_lst)
    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2

    es = np.array([mol.bas_exp(i).min() for i in range(nbas)])
    etas = 1/(1/es[:,None] + 1/es)

    def estimate1(ish,jsh,R0,R1):
        shls_slice = (ish,ish+1,nbas+jsh,nbas+jsh+1)
        prec0 = precision * min(etas[ish,jsh],1.)
        def fcheck(R):
            mol._env[mol._atm[1,mol_gto.PTR_COORD]] = R
            I = get_norm( mol.intor("int1e_ovlp", shls_slice=shls_slice) )
            prec = prec0 * min(1./R,1.)
            return I < prec
        return binary_search(R0, R1, 1, True, fcheck)

    if r0 is None: r0 = 30
    R0 = r0 * 0.3
    R1 = r0
    dcuts = np.zeros(n2)
    ij = 0
    for i in range(nbas):
        for j in range(i+1):
            dcuts[ij] = estimate1(i,j,R0, R1)
            ij += 1
    return dcuts
def get_schwartz_dcut(bas_lst, cellvol, omega, precision, r0=None, safe=True,
                      vol_correct=False):
    """ Given a list of basis, determine cutoff radius for the Schwartz Q between each unique shell pair to drop below "precision". The Schwartz Q is define:
        Q = 2-norm[ (ab|ab) ]^(1/2)

    Return:
        1d array of length nbas*(nbas+1)//2 with nbas=len(bas_lst).
    """
    mol = mol_gto.M(atom="H 0 0 0; H 0 0 0", basis=bas_lst)
    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2

    es = np.array([mol.bas_exp(i).min() for i in range(nbas)])
    etas = 1/(1/es[:,None] + 1/es)
# >>>>>>> debug block
    if vol_correct:
        fac = 2*np.pi/cellvol
    else:
        fac = 1.
# <<<<<<<

    intor = "int2e"
    def estimate1(ish,jsh,R0,R1):
        shls_slice = (ish,ish+1,nbas+jsh,nbas+jsh+1,
                      ish,ish+1,nbas+jsh,nbas+jsh+1)
        prec0 = precision * min(etas[ish,jsh],1.)
        def fcheck(R):
            mol._env[mol._atm[1,mol_gto.PTR_COORD]] = R
            I = get_norm(
                    fintor_sreri(mol, intor, shls_slice, omega, safe)
                )**0.5
# >>>>>>> debug block
            I *= fac
# <<<<<<<
            prec = prec0 * min(1./R,1.)
            return I < prec
        return binary_search(R0, R1, 1, True, fcheck)

    if r0 is None: r0 = 30
    R0 = r0 * 0.3
    R1 = r0
    dcuts = np.zeros(n2)
    ij = 0
    for i in range(nbas):
        for j in range(i+1):
            dcuts[ij] = estimate1(i,j,R0, R1)
            ij += 1
    return dcuts

def make_dijs_lst(dcuts, dstep):
    return [np.arange(0,dcut,dstep) for dcut in dcuts]

def get_bincoeff(d,e1,e2,l1,l2):
    d1 = -e2/(e1+e2) * d
    d2 = e1/(e1+e2) * d
    lmax = l1+l2
    cbins = np.zeros(lmax+1)
    for l in range(0,lmax+1):
        cl = 0.
        lpmin = max(-l,l-2*l2)
        lpmax = min(l,2*l1-l)
        for lp in range(lpmin,lpmax+1,2):
            l1p = (l+lp) // 2
            l2p = (l-lp) // 2
            cl += d1**(l1-l1p)*d2**(l2-l2p) * comb(l1,l1p) * comb(l2,l2p)
        cbins[l] = cl
    return cbins
def get_3c2e_Rcuts_for_d(mol, auxmol, ish, jsh, dij, cellvol, omega, precision,
                         fac_type, Qij, Rprec=1,
                         vol_correct=False, eta_correct=True, R_correct=True):
    """ Determine for AO shlpr (ish,jsh) separated by dij, the cutoff radius for
            2-norm( (ksh|v_SR(omega)|ish,jsh) ) < precision
        The estimator used here is
            ~ 0.5/pi * exp(-etaij*dij^2) * O_{k,lk} *
                \sum_{l=lmin}^{lmax} L_{li,lj}^{l} O_{ij,l} *
                Gamma(lk+l+1/2, eta2*R^2) / R^(lk+l+1)
        where
            eij = ei + ej
            lij = li + lj
            etaij = 1/(1/ei+1/ej)
            O_{k,lk} = 0.5*pi * (2*lk+1)^0.5 / ek^(lk+3/2)
            O_{ij,l} = 0.5*pi * (2*l+1)^0.5 / eij^(l+3/2)
            lmax = lij
            if d == 0:
                lmin = |li-lj|
                L_{li,lj}^{l} = eij^((l-lij)/2) * ((lij-1)!/(l-1)!)^0.5
            else:
                lmin = 0
                L_{li,lj}^{l} = \sum'_{m=-l}^{l} comb(li,mi) * comb(lj,mj) * di^(li-mi) * dj^(lj-mj)
                where
                    mi = (l+m)/2
                    mj = (l-m)/2
                    di = -ej/eij * (dij + extij)
                    dj = ei/eij * (dij + extij)
                where "extij" is the extent of orbital pair ij.

        Similar to :func:`get_2c2e_Rcut`, the estimator is multiplied by factor of eta and/or 1/R if "eta_correct" and/or "R_correct" are set to True.

    Args:
        mol/auxmol (Mole object):
            Provide AO/aux basis info.
        ish/jsh (int):
            AO shl index.
        dij (float):
            Separation between ish and jsh; in BOHR
        omega (float):
            erfc(omega * r12) / r12
        precision (float):
            target precision.
    """
# sanity check for estimators
    FAC_TYPE = fac_type.upper()
    if not FAC_TYPE in [
            "ISF0",  # (ss|s)
            "ISF",   # (ss|X)
            "ISFQ0", # (Q_ss|X)
            "ISFQL", # (Q_lmax|X)
            "ME"     # \sum_l (l|X)
        ]:
        raise RuntimeError("Unknown estimator requested {}".format(fac_type))

# get bas info
    nbasaux = auxmol.nbas
    eks = [auxmol.bas_exp(ksh)[0] for ksh in range(nbasaux)]
    lks = [int(auxmol.bas_angular(ksh)) for ksh in range(nbasaux)]
    cks = [auxmol._libcint_ctr_coeff(ksh)[0,0] for ksh in range(nbasaux)]

    def get_lec(mol, i):
        l = int(mol.bas_angular(i))
        es = mol.bas_exp(i)
        imin = es.argmin()
        e = es[imin]
        c = abs(mol._libcint_ctr_coeff(i)[imin]).max()
        return l,e,c
    l1,e1,c1 = get_lec(mol, ish)
    l2,e2,c2 = get_lec(mol, jsh)

# local helper funcs
    def init_feval(e1,e2,e3,l1,l2,l3,c1,c2,c3, d, Q, FAC_TYPE):
        e12 = e1+e2
        l12 = l1+l2

        eta1 = 1/(1/e12+1/e3)
        eta2 = 1/(1/eta1+1/omega**2.)
        eta12 = 1/(1/e1+1/e2)

        fac = c1*c2*c3 * 0.5/np.pi
# >>>>>>>> debug block
        if vol_correct:
            fac *= 2*np.pi/cellvol
# <<<<<<<<
        if FAC_TYPE == "ME":

            O3 = get_multipole(l3, e3)

            if d < 1e-3:    # concentric
                ls = np.arange(abs(l1-l2),l12+1)
                O12s = get_multipole(ls, e12)
                l_facs = O12s * O3 * e12**(0.5*(ls-l12)) * (
                                gamma(max(l12,1))/gamma(np.maximum(ls,1)))**0.5
            else:
                fac *= np.exp(-eta12*d**2.)
                ls = np.arange(0,l12+1)
                O12s = get_multipole(ls, e12)
                l_facs = O12s * O3 * abs(get_bincoeff(d,e1,e2,l1,l2))

            def feval(R):
                I = 0.
                for l_fac,l in zip(l_facs,ls):
                    I += l_fac * Gamma(l+l3+0.5,eta2*R**2.) / R**(l+l3+1)
                return I * fac

        elif FAC_TYPE == "ISF0":

            O12 = get_multipole(0, e12)
            O3 = get_multipole(0, e3)
            fac *= np.exp(-eta12*d**2.)

            def feval(R):
                return fac * O12 * O3 * Gamma(0.5, eta2*R**2) / R

        elif FAC_TYPE == "ISF":

            O12 = get_multipole(0, e12)
            O3 = get_multipole(l3, e3)
            fac *= np.exp(-eta12*d**2.)

            def feval(R):
                return fac * O12 * O3 * Gamma(l3+0.5, eta2*R**2) / R**(l3+1)

        elif FAC_TYPE in ["ISFQ0","ISFQL"]:

            eta1212 = 0.5 * e12
            eta1212w = 1/(1/eta1212+1/omega**2.)

            O3 = get_multipole(l3, e3)

            def feval(R):

                if FAC_TYPE == "ISFQ0":
                    L12 = 0
                    Q2S = 2*np.pi**0.75/(2*(eta1212**0.5-eta1212w**0.5))**0.5/(c1*c2)
                    O12 = Q * Q2S
                    veff = Gamma(L12+l3+0.5, eta2*R**2) / R**(L12+l3+1)
                else:
                    l12min = abs(l1-l2) if d<1e-3 else 0
                    ls = np.arange(l12min,l12+1)
                    l_facs = (eta1212**(ls+0.5) - eta1212w**(ls+0.5))**-0.5
                    veffs = Gamma(ls+l3+0.5, eta2*R**2.) / R**(ls+l3+1)
                    ilmax = (l_facs*veffs).argmax()
                    l_fac = l_facs[ilmax]
                    veff = veffs[ilmax]
                    Q2S = 2**0.5*np.pi**0.75/(c1*c2) * l_fac
                    O12 = Q * Q2S

                return fac * O12 * O3 * veff

        else:
            raise RuntimeError

        return feval

    def estimate1(ksh, R0, R1):
        l3 = lks[ksh]
        e3 = eks[ksh]
        c3 = cks[ksh]
        feval = init_feval(e1,e2,e3,l1,l2,l3,c1,c2,c3, dij, Qij, FAC_TYPE)

        eta2 = 1/(1/(e1+e2)+1/e2+1/omega**2.)
        prec0 = precision * (min(eta2,1.) if eta_correct else 1.)
        def fcheck(R):
            prec = prec0 * (min(1./R,1.) if R_correct else 1.)
            I = feval(R)
            return I < prec
        return binary_search(R0, R1, Rprec, True, fcheck)

# estimating Rcuts
    Rcuts = np.zeros(nbasaux)
    R0 = 5
    R1 = 20
    for ksh in range(nbasaux):
        Rcuts[ksh] = estimate1(ksh, R0, R1)

    return Rcuts
def get_3c2e_Rcuts(bas_lst, auxbas_lst, dijs_lst, cellvol, omega, precision,
                   fac_type, Qijs_lst, Rprec=1,
                   eta_correct=True, R_correct=True, vol_correct=False):
    """ Given a list of basis ("bas_lst") and auxiliary basis ("auxbas_lst"), determine the cutoff radius for
        2-norm( (k|v_SR(omega)|ij) ) < precision
    where i and j shls are separated by d specified by "dijs_lst".
    """

    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2
    nbasaux = len(auxbas_lst)

    mol = mol_gto.M(atom="H 0 0 0", basis=bas_lst, spin=None)
    auxmol = mol_gto.M(atom="H 0 0 0", basis=auxbas_lst, spin=None)

    ij = 0
    Rcuts = []
    for i in range(nbas):
        for j in range(i+1):
            dijs = dijs_lst[ij]
            Qijs = Qijs_lst[ij]
            for dij,Qij in zip(dijs,Qijs):
                Rcuts_dij = get_3c2e_Rcuts_for_d(mol, auxmol, i, j, dij,
                                                 cellvol, omega, precision,
                                                 fac_type, Qij,
                                                 Rprec=Rprec,
                                                 eta_correct=eta_correct,
                                                 R_correct=R_correct,
                                                 vol_correct=vol_correct)
                Rcuts.append(Rcuts_dij)
            ij += 1
    Rcuts = np.asarray(Rcuts).reshape(-1)
    return Rcuts

def get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, bas_loc, auxbas_loc):
    natm = len(bas_loc) - 1
    assert(len(auxbas_loc) == natm+1)
    bas_loc_inv = np.concatenate([[i]*(bas_loc[i+1]-bas_loc[i])
                                  for i in range(natm)])
    nbas = bas_loc[-1]
    nbas2 = nbas*(nbas+1)//2
    nbasaux = auxbas_loc[-1]
    Rcuts_ = Rcuts.reshape(-1,nbasaux)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst])
    betas = np.maximum(bas_exps[:,None],bas_exps) / (bas_exps[:,None]+bas_exps)

    atom_Rcuts = np.zeros((natm,natm))
    for katm in range(natm):    # aux atm
        k0, k1 = auxbas_loc[katm:katm+2]
        Rcuts_katm = np.max(Rcuts_[:,k0:k1], axis=1)

        rcuts_katm = np.zeros(natm)
        for ij in range(nbas2):
            i = int(np.floor((-1+(1+8*ij)**0.5)*0.5))
            j = ij - i*(i+1)//2
            ei = bas_exps[i]
            ej = bas_exps[j]
            bi = ej/(ei+ej)
            bj = ei/(ei+ej)
            dijs = dijs_lst[ij]
            idij0,idij1 = dijs_loc[ij:ij+2]
            rimax = (Rcuts_katm[idij0:idij1] + dijs*bi).max()
            rjmax = (Rcuts_katm[idij0:idij1] + dijs*bj).max()
            iatm = bas_loc_inv[i]
            jatm = bas_loc_inv[j]
            rcuts_katm[iatm] = max(rcuts_katm[iatm],rimax)
            rcuts_katm[jatm] = max(rcuts_katm[jatm],rjmax)

        atom_Rcuts[katm] = rcuts_katm

    return atom_Rcuts

def make_supmol_j3c(cell, atom_Rcuts, uniq_atms):
    return make_supmol_j2c(cell, atom_Rcuts, uniq_atms)

def get_cellpr_data(cell, supmol, uniq_atm_dcuts, uniq_atm_dijs_lst, uniq_atms,
                    dtype=np.int32):
    """
    Return:
        refatmprd_loc [size : natm2+1]
        supatmpr_loc [size : natm2d+1]
        supatmpr_lst [size : nsupshlpr]
    """
    def tril_idx(i,j):
        return i*(i+1)//2+j if i>=j else j*(j+1)//2+i

    dcuts = uniq_atm_dcuts
    dijs_lst = uniq_atm_dijs_lst
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst])

    natm = cell.natm
    natm2 = natm*(natm+1)//2
    refuniqatm_map = [uniq_atms.index(cell.atom_symbol(i)) for i in range(natm)]
    nbas = cell.nbas
    Rssup = supmol.atom_coords()

    supatmpr_loc = [None] * natm2
    supatmpr_lst = [None] * natm2 * 2
    for Iatm in range(natm):
        IATM = refuniqatm_map[Iatm]
        iatms = supmol._refsupatm_map[supmol._refsupatm_loc[Iatm]:
                                      supmol._refsupatm_loc[Iatm+1]]
        Ris = Rssup[iatms]
        for Jatm in range(Iatm+1):
            IJatm = tril_idx(Iatm,Jatm)
            JATM = refuniqatm_map[Jatm]
            IJATM = tril_idx(IATM,JATM)
            dcut = dcuts[IJATM]
            if Iatm == Jatm:
                jatms = iatms
                Rjs = Ris
            else:
                jatms = supmol._refsupatm_map[supmol._refsupatm_loc[Jatm]:
                                              supmol._refsupatm_loc[Jatm+1]]
                Rjs = Rssup[jatms]
            njatm = len(jatms)
            dijs = get_dist_mat(Ris, Rjs)
            dij_atmprids = np.arange(dijs.size)
            dijs = dijs.reshape(-1)

            ids_keep = np.where(dijs <= dcut)[0]
            if ids_keep.size == 0:
                supatmpr_loc[IJatm] = np.array([], dtype=dtype)
                supatmpr_lst[IJatm] = np.array([], dtype=dtype)
                supatmpr_lst[IJatm+natm2] = np.array([], dtype=dtype)
                continue

            idij0 = dijs_loc[IJATM]
            idij1 = dijs_loc[IJATM+1]
            dijs_bins = dijs_lst[IJATM]
            dijs_keep = dijs[ids_keep]
            dijs_inverse = np.digitize(dijs_keep, bins=dijs_bins)-1
            atmpr_keep_IJatm = []
            for idij,dij in enumerate(dijs_bins):
                mask = dijs_inverse==idij
                if np.any(mask):
                    atmpr_keep_IJatm.append(dij_atmprids[ids_keep[mask]])
                else:
                    atmpr_keep_IJatm.append(np.array([], dtype=int))

            atmpr_len_IJatm = np.asarray([len(x)
                                        for x in atmpr_keep_IJatm],
                                        dtype=dtype)
            atmpr_keep_IJatm = np.concatenate(atmpr_keep_IJatm)
            iatms_keep = iatms[atmpr_keep_IJatm//njatm]
            jatms_keep = jatms[atmpr_keep_IJatm%njatm]
            # write
            supatmpr_loc[IJatm] = atmpr_len_IJatm
            supatmpr_lst[IJatm] = iatms_keep.astype(dtype)
            supatmpr_lst[IJatm+natm2] = jatms_keep.astype(dtype)
            # clear
            atmpr_keep_IJatm = atmpr_len_IJatm = iatms_keep = jatms_keep = None

    refatmprd_loc = np.cumsum([0]+[len(x) for x in supatmpr_loc]).astype(dtype)
    supatmpr_lst = np.concatenate(supatmpr_lst)
    supatmpr_loc = np.cumsum([0]+np.concatenate(
                             supatmpr_loc).tolist()).astype(dtype)

    return refatmprd_loc, supatmpr_lst, supatmpr_loc

def intor_j3c(cell, auxcell, omega, kptij_lst=np.zeros((1,2,3)), out=None,
              precision=None, use_cintopt=True, safe=True, fac_type="ME",
              reshapek=True,
# +++++++ Use the default for the following unless you know what you are doing
              eta_correct=True, R_correct=True,
              vol_correct_d=False, vol_correct_R=False,
              dstep=1,  # unit: Angstrom
# -------
# +++++++ debug options
              ret_timing=False,
              force_kcode=False,
              discard_integrals=False,  # compute j3c and discard (no store)
              no_screening=False,   # set Rcuts to effectively infinity
# -------
              ):

    cput0 = np.asarray([logger.process_clock(), logger.perf_counter()])

    if precision is None: precision = cell.precision

    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
                                                        get_refuniq_map(auxcell)
    nbasauxuniq = len(uniq_basaux)

    # dcuts = get_ovlp_dcut(uniq_bas, precision, r0=cell.rcut)
    Qauxs = get_schwartz_data(uniq_basaux, omega, keep1ctr=False, safe=True)
    dcuts = get_schwartz_dcut(uniq_bas, cell.vol, omega, precision/Qauxs.max(),
                              r0=cell.rcut, vol_correct=vol_correct_d)
    uniq_atm_dcuts = lib.pack_tril(get_atom_Rcuts_2c(dcuts, uniq_bas_loc))
    # print(dcuts)
    # print(uniq_atm_dcuts)
    uniq_atm_dijs_lst = make_dijs_lst(uniq_atm_dcuts, dstep/BOHR)
    # print(uniq_atm_dijs_lst)
    dijs_lst = make_dijs_lst(dcuts, dstep/BOHR)
    # print(dijs_lst)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst]).astype(np.int32)
    uniq_atm_dijs_loc = np.cumsum([0]+[len(dijs) for dijs in uniq_atm_dijs_lst]).astype(np.int32)
    # print(dijs_loc)
    # print(uniq_atm_dijs_loc)
    if fac_type.upper() in ["ISFQ0","ISFQL"]:
        Qs_lst = get_schwartz_data(uniq_bas, omega, dijs_lst, keep1ctr=True,
                                   safe=True)
    else:
        Qs_lst = [np.zeros_like(dijs) for dijs in dijs_lst]
    Rcuts = get_3c2e_Rcuts(uniq_bas, uniq_basaux, dijs_lst, cell.vol, omega,
                           precision, fac_type, Qs_lst,
                           eta_correct=eta_correct, R_correct=R_correct,
                           vol_correct=vol_correct_R)
    Rcut2s = Rcuts**2.
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    atom_Rcuts = get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, uniq_bas_loc,
                                   uniq_basaux_loc)
    supmol = make_supmol_j3c(cell, atom_Rcuts, uniq_atms)
    refatmprd_loc, supatmpr_lst, supatmpr_loc = \
            get_cellpr_data(cell, supmol, uniq_atm_dcuts, uniq_atm_dijs_lst,
                            uniq_atms)
    # print(refatmprd_loc)
    refexp = np.asarray([cell.bas_exp(i).min() for i in range(cell.nbas)])

# concatenate atm/bas/env
    atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                     auxcell._atm, auxcell._bas, auxcell._env)
    atmsup, bassup, envsup = mol_gto.conc_env(
                                    supmol._atm, supmol._bas, supmol._env,
                                    auxcell._atm, auxcell._bas, auxcell._env)
    env[mol_gto.PTR_RANGE_OMEGA] = envsup[mol_gto.PTR_RANGE_OMEGA] = -abs(omega)

    dtype_idx = np.int32
    natm = cell.natm
    nbas = cell.nbas
    nbasaux = auxcell.nbas
    nbassup = supmol.nbas
    ao_loc = cell.ao_loc_nr()
    ao_locaux = auxcell.ao_loc_nr()
    ao_locsup = supmol.ao_loc_nr()
    ao_loc = np.concatenate([ao_loc[:-1], ao_locaux+ao_loc[-1]])
    ao_locsup = np.concatenate([ao_locsup[:-1], ao_locaux+ao_locsup[-1]])
    shl_loc = np.concatenate([cell.aoslice_nr_by_atom()[:,0],
                              auxcell.aoslice_nr_by_atom()[:,0]+nbas,
                              [nbas+nbasaux]]).astype(dtype_idx, copy=False)
    nsupatmpr = len(supatmpr_lst)//2
    nao = cell.nao
    naoaux = auxcell.nao

    refshlstart_by_atm = cell.aoslice_nr_by_atom()[:,0].astype(np.int32)
    supshlstart_by_atm = supmol.aoslice_nr_by_atom()[:,0].astype(np.int32)

    nsupatmpr_tot = supmol.natm*(supmol.natm+1)//2
    logger.debug1(cell, "nsupatmpr_tot= %d  nsupatmpr_keep= %d  ( %.2f %% )",
                  nsupatmpr_tot, nsupatmpr, nsupatmpr/nsupatmpr_tot*100)
    memsp = nsupatmpr*8/1024**2.
    logger.debug1(cell, "mem use by atmpr data %.2f MB", memsp)

    intor = "int3c2e"
    intor, comp = mol_gto.moleintor._get_intor_and_comp(
                                            cell._add_suffix(intor), None)
    assert(comp == 1)
    if use_cintopt:
        cintopt = _vhf.make_cintopt(atmsup, bassup, envsup, intor)
    else:
        cintopt = lib.c_null_ptr()

    cput1 = np.asarray( logger.timer(cell, 'j3c precompute', *cput0) )
    dt0 = cput1 - cput0

    kptij_lst = np.asarray(kptij_lst).reshape(-1,2,3)
    if gamma_point(kptij_lst) and not force_kcode:
        # drv = libpbc.fill_sr3c2e_g
# >>>>> debug block
        if discard_integrals:
            drv = libpbc.fill_sr3c2e_g_nosave
        else:
            # drv = libpbc.fill_sr3c2e_g
            drv = libpbc.PBCnr_sr3c2e_g_drv
        if no_screening:
            Rcut2s = np.clip(Rcut2s, 1e20, None)
# <<<<<
        def fill_j3c(out):
            drv(getattr(libpbc, intor),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), cintopt,
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ao_locsup.ctypes.data_as(ctypes.c_void_p),
                shl_loc.ctypes.data_as(ctypes.c_void_p),
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                refexp.ctypes.data_as(ctypes.c_void_p),
                refshlstart_by_atm.ctypes.data_as(ctypes.c_void_p),
                supshlstart_by_atm.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                refatmprd_loc.ctypes.data_as(ctypes.c_void_p),
                supatmpr_loc.ctypes.data_as(ctypes.c_void_p),
                supatmpr_lst.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nsupatmpr),
                atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),
                ctypes.c_int(nbasaux),
                env.ctypes.data_as(ctypes.c_void_p),
                atmsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                bassup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                envsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(envsup.size),
                ctypes.c_char(safe))

        nao2 = nao*(nao+1)//2
        memj3c = naoaux*nao2*8/1024**2.
        logger.debug1(cell, "estimated mem for 3c2e %.2f MB", memj3c)
        # out = np.zeros((naoaux,nao2), dtype=np.float64)
# >>>>> debug block
        if discard_integrals:
            out = np.array([0], dtype=np.float64)
        else:
            if out is None:
                out = np.zeros((naoaux,nao2), dtype=np.float64)
            else:
                assert(out.dtype == np.float64 and out.shape == (1,naoaux,nao2))
                out.fill(0.)
# <<<<<
        with supmol.with_range_coulomb(-abs(omega)):
            fill_j3c(out)
        out = out.reshape(1,naoaux,nao2)
    else:
        nkptijs = len(kptij_lst)
        kpti = kptij_lst[:,0]
        kptj = kptij_lst[:,1]
        kpts = unique(np.vstack([kpti,kptj]))[0]
        expLk = np.exp(1j * lib.dot(supmol._Ls, kpts.T))
        wherei = np.where(abs(kpti[:,None,:]-kpts).sum(axis=2) <
                             KPT_DIFF_TOL)[1].astype(dtype_idx)
        wherej = np.where(abs(kptj[:,None,:]-kpts).sum(axis=2) <
                             KPT_DIFF_TOL)[1].astype(dtype_idx)
        nkpts = len(kpts)
        kptij_idx = np.concatenate([wherei,wherej])

        drv = libpbc.fill_sr3c2e_kk
# *******
        # drv = libpbc.fill_sr3c2e_kk_bvk
# *******
        def fill_j3c(out):
            drv(getattr(libpbc, intor),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), cintopt,
                expLk.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkpts),
                kptij_idx.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkptijs),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ao_locsup.ctypes.data_as(ctypes.c_void_p),
                shl_loc.ctypes.data_as(ctypes.c_void_p),
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                refexp.ctypes.data_as(ctypes.c_void_p),
                refshlstart_by_atm.ctypes.data_as(ctypes.c_void_p),
                supshlstart_by_atm.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                refatmprd_loc.ctypes.data_as(ctypes.c_void_p),
                supatmpr_loc.ctypes.data_as(ctypes.c_void_p),
                supatmpr_lst.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nsupatmpr),
                atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),
                ctypes.c_int(nbasaux),
                env.ctypes.data_as(ctypes.c_void_p),
                atmsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                bassup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                envsup.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_char(safe))

        nao2 = nao*nao
        memj3c = nkptijs*naoaux*nao2*16/1024**2.
        logger.debug1(cell, "estimated mem for 3c2e %.2f MB", memj3c)
        if out is None:
            out_ = np.zeros((nkptijs,naoaux,nao2), dtype=np.complex128)
        else:
            assert(out.dtype == np.complex128 and
                   out.shape == (nkptijs,naoaux,nao2))
            out.fill(0.+0.j)
            out_ = out
        with supmol.with_range_coulomb(-abs(omega)):
            fill_j3c(out_)

        if reshapek:
            aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
            tril_idx = np.tril_indices(nao)
            tril_idx = tril_idx[0] * nao + tril_idx[1]
            out = [None] * nkptijs
            for kij in range(nkptijs):
                v = out_[kij]
                if gamma_point(kptij_lst[kij]):
                    v = v.real
                if aosym_ks2[kij]:
                    v = v[:,tril_idx]
                out[kij] = v
            out_ = None
        else:
            out = out_

    cput0 = np.asarray( logger.timer(cell, 'j3c compute', *cput1) )
    dt1 = cput0 - cput1

    if ret_timing:
        return out, dt0, dt1
    else:
        return out


def intor_j3c_outcore(cell, auxcell, omega, erifile, dataname, comp=1,
                      max_memory=4000, kptij_lst=None, precision=None,
                      use_cintopt=True, safe=True, fac_type="ME",
# +++++++ Use the default for the following unless you know what you are doing
                      eta_correct=True, R_correct=True,
                      vol_correct_d=False, vol_correct_R=False,
                      dstep=1,  # unit: Angstrom
# -------
# +++++++ debug options
                      force_kcode=False,
                      discard_integrals=False,  # compute and discard (no store)
                      no_screening=False,   # set Rcuts to effectively infinity
# -------
              ):

    if isinstance(erifile, h5py.Group):
        feri = erifile
    elif h5py.is_hdf5(erifile):
        feri = h5py.File(erifile, 'a')
    else:
        feri = h5py.File(erifile, 'w')
    if dataname in feri:
        del(feri[dataname])
    if dataname+'-kptij' in feri:
        del(feri[dataname+'-kptij'])

    if kptij_lst is None:
        kptij_lst = np.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    nkptijs = len(kptij_lst)
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    sorted_ij_idx = np.hstack([np.where(uniq_inverse == k)[0]
                               for k, kpt in enumerate(uniq_kpts)])

    nao = cell.nao
    naoaux = auxcell.nao
    if gamma_point(kptij_lst):
        assert(nkptijs == 1)
        dtype = np.float64
        dsize = 8
        nao2 = nao*(nao+1)//2
    else:
        dtype = np.complex128
        dsize = 16
        nao2 = nao*nao
    mem_perk = nao2*naoaux * dsize/1024**2.
    nkptij_max = int(np.floor(max_memory*0.47 / mem_perk))
    assert(nkptij_max > 0)
    nkseg = nkptijs//nkptij_max + 1
    kptijranges = [(i*nkptij_max,(i+1)*nkptij_max) for i in range(nkseg-1)] + \
                    [((nkseg-1)*nkptij_max,nkptijs)]

    aosym_ks2 = abs(kptis-kptjs).sum(axis=1) < KPT_DIFF_TOL
    tril_idx = np.tril_indices(nao)
    tril_idx = tril_idx[0] * nao + tril_idx[1]

    buf = np.empty((nkptij_max*nao2*naoaux),dtype=dtype)
    bufs = [buf, np.empty_like(buf)]
    def process(kptijrange):
        kp0, kp1 = kptijrange
        kptij_lst_p = kptij_lst[sorted_ij_idx[kp0:kp1]]
        nkptij_p = kp1 - kp0
        mat = np.ndarray((nkptij_p,naoaux,nao2), dtype=dtype, buffer=bufs[0])
        bufs[:] = bufs[1], bufs[0]
        intor_j3c(cell, auxcell, omega, out=mat, kptij_lst=kptij_lst_p,
                  precision=precision,
                  use_cintopt=use_cintopt,
                  safe=safe,
                  fac_type=fac_type,
                  reshapek=False,   # avoid reshaping j3c
            # settings for lat sum
                  eta_correct=eta_correct, R_correct=R_correct,
                  vol_correct_d=vol_correct_d, vol_correct_R=vol_correct_R,
                  dstep=dstep,
            # debugging options
                  force_kcode=force_kcode,
                  discard_integrals=discard_integrals,
                  no_screening=no_screening
            )
        mat = mat.reshape(nkptij_p,comp,naoaux,nao2)
        return mat

    kp0, kp1 = kptijranges[0]
    logger.debug(cell, 'kptij_lst seg %d/%d  range %d ~ %d',
                 0,nkseg,kp0,kp1)
    for istep, mat in enumerate(lib.map_with_prefetch(process, kptijranges)):
        kp0, kp1 = kptijranges[istep]
        for ik,k in enumerate(sorted_ij_idx[kp0:kp1]):
            v = mat[ik]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao2 == nao*nao:
                v = v[:,:,tril_idx]
            feri['%s/%d/0' % (dataname,k)] = v
        mat = None

        if istep < nkseg-1:
            kp0, kp1 = kptijranges[istep+1]
            logger.debug(cell, 'kptij_lst seg %d/%d  range %d ~ %d',
                         istep+1,nkseg,kp0,kp1)

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


if __name__ == "__main__":
    from pyscf.pbc import gto, tools
    from pyscf import df
    from utils import get_lattice_sc40

    fml = "c"
    atom, a = get_lattice_sc40(fml)
    basis = "gth-dzvp"
    # basis = "cc-pvdz"
#     basis = """
# C    P
#       0.1517000              1.0000000
# C    D
#       0.5500000              1.0000000
# """
    # atom = "C 0 0 0"
    # a = np.eye(3) * 2.8
    # basis = [[0,[0.8,1.]], [1,[1,1.]]]

    cell = gto.Cell(atom=atom, a=a, basis=basis, spin=None)
    cell.verbose = 0
    cell.build()
    cell.verbose = 6

    # cell = tools.super_cell(cell, [2,2,2])

    auxcell = df.make_auxmol(cell)

    kptij_lst = np.zeros((1,2,3))
    # kptij_lst = np.asarray([np.random.rand(3)]*2).reshape(1,2,3)
    # kptij_lst = np.random.rand(1,2,3)
    nkpts = len(kptij_lst)

    omega = 0.8

    mesh = [31]*3
    from aft_j3c import j3c_aft
    # j3 = j3c_aft(cell, auxcell, omega, mesh, kptij_lst=kptij_lst)
    j3 = None

    prec_lst = [1e-8,1e-10]

    js = []
    j2s = []
    for prec in prec_lst:
        j, dt0, dt1 = intor_j3c(cell, auxcell, omega, kptij_lst=kptij_lst,
                                precision=prec)
        js.append(j)
        print("init time CPU %7.3f  wall %7.3f" % (dt0[0], dt0[1]))
        print("calc time CPU %7.3f  wall %7.3f" % (dt1[0], dt1[1]))

        if not j3 is None:
            for k in range(nkpts):

                # j[k] = lib.unpack_tril(j[k]).reshape(auxcell.nao,-1)

                err = abs(j[k] - j3[k])
                print("kpt %d "%k, err.max(), err.mean())
                err_r = abs(j[k].T.real-j3[k].T.real)
                print("real ", err_r.max(), err_r.mean())
                err_i = abs(j[k].T.imag-j3[k].T.imag)
                print("imag ", err_i.max(), err_i.mean())
                if j[k].shape[0] <= 10:
                    from frankenstein.tools.io_utils import dumpMat
                    # dumpMat(j[k].T.real)
                    # dumpMat(j3[k].T.real)
                    dumpMat(err_r, fmt="%.1e")
                    # dumpMat(j[k].T.imag*1e4)
                    # dumpMat(j3[k].T.imag*1e4)
                    dumpMat(err_i, fmt="%.1e")

        # from _3_intor import intor_j3c as intor_j3c_
        # j2, dt20, dt21 = intor_j3c_(cell, auxcell, omega, kpts=kpts, precision=prec,
        #               ret_timing=True)
        # j2s.append(j2)
        # print("init time CPU %7.3f  wall %7.3f" % (dt20[0], dt20[1]))
        # print("calc time CPU %7.3f  wall %7.3f" % (dt21[0], dt21[1]))
        #
        # if j[0].ndim == 2 and j[0].shape[1] <= 10:
        #     from frankenstein.tools.io_utils import dumpMat
        #     dumpMat(j[0])
        #     dumpMat(j2[0])
        # for k in range(nkpts):
        #     e = abs(j2[k]-j[k])
        #     print("kpt %d "%k, e.max(), e.mean())

    for k in range(nkpts):
        print("kpt %d" % k)
        for j in js[:-1]:
            errmat = abs(j[k]-js[-1][k])
            maxerr = errmat.max()
            meanerr = errmat.mean()
            print(" ", maxerr, meanerr)

        if len(j2s) > 0:
            for j2 in j2s[:-1]:
                errmat = abs(j2[k]-j2s[-1][k])
                maxerr = errmat.max()
                meanerr = errmat.mean()
                print(" ", maxerr, meanerr)
