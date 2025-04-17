#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import ctypes
import h5py
import numpy as np
from scipy.special import gamma, gammaincc, comb

from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import BOHR
from pyscf import gto as mol_gto
from pyscf.pbc.df.rsdf_builder import _round_off_to_odd_mesh
from pyscf.pbc.df.incore import libpbc, make_auxcell
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL
from pyscf.pbc.tools import pbc as pbctools
from pyscf.scf import _vhf
from pyscf.pbc.tools import k2gamma


""" General helper functions
"""
class MoleNoBasSort(mol_gto.mole.Mole):
    def build(self, **kwargs):
        self.atom = kwargs.pop('atom')
        self.basis = kwargs.pop('basis')
        self._atom = mol_gto.format_atom(self.atom)
        if isinstance(self.basis, dict):
            self._basis = self.basis
        else:
            self._basis = {a[0]: self.basis for a in self._atom}

        env = np.zeros(mol_gto.PTR_ENV_START)
        # _bas should be constructed as it is in the input (see issue #1942).
        self._atm, self._bas, self._env = self.make_env(
            self._atom, self._basis, env)
        self._built = True
        return self

def _remove_exp_basis_(bold, amin, amax):
    bnew = []
    for b in bold:
        l = b[0]
        ecs = np.array(b[1:])
        if ecs.ndim == 1:
            ecs = np.array([ecs])
        nprim, nctr = ecs.shape
        if nprim == 1:  # pGTO
            e = ecs[0,0]
            if e >= amin and e <= amax:
                bnew.append(b)
        else:  # cGTO
            es = ecs[:,0]
            ecsnew = ecs[(es>=amin) & (es<=amax)]
            nprimnew = ecsnew.shape[0]
            if nprimnew == 0:   # no prims left
                continue
            elif nprimnew == 1: # cGTO reduces to pGTO
                bnew.append([l, [ecsnew[0,0], 1.]])
            else:
                csnew = ecsnew[:,1:]
                # cGTOs having zero coeff after expn discard are removed
                csnew = csnew[:,abs(csnew).sum(axis=0) > 1e-16]
                esnew = ecsnew[:,:1]
                ecsnew = np.hstack([esnew,csnew])
                bnew.append([l] + [ec.tolist() for ec in ecsnew])
    return bnew
def remove_exp_basis(basis, amin=None, amax=None):
    r""" Removing primitives with exponents not in (amin,amax).

    Args:
        basis (list or dict):
            PySCF GTO basis format (as in mol/cell._basis), e.g.,:
                basis = [[0, (1.5, 1.)], [1, (0.7, 1.)]]
                basis = {"C": [[0, (1.5, 1.)], [1, (0.7, 1.)]],
                         "H": [[0, (0.3, 1.)]]}
        amin/amax (float; default: None):
            Primitives with exponents <= amin or >= amax will be removed.
            If either is not provided (default), no primitives will be removed
            by that criterion.
    """
    if amin is None and amax is None:
        return basis
    if amin is None: amin = 0.
    if amax is None: amax = float('inf')
    if isinstance(basis, dict):
        basisnew = {atm:_remove_exp_basis_(blist,amin,amax)
                    for atm,blist in basis.items()}
    else:
        basisnew = _remove_exp_basis_(basis,amin,amax)
    return basisnew
def _binary_search(xlo, xhi, xtol, ret_bigger, fcheck, args=None,
                   MAX_RESCALE=5, MAX_CYCLE=20, early_exit=True):
    if args is None: args = ()
# rescale xlo/xhi if necessary
    first_time = True
    count = 0
    while True:
        ylo = fcheck(xlo, *args)
        if not ylo:
            xlo_rescaled = count > 0
            break
        if ylo and first_time and early_exit:
            return xlo
        if first_time: first_time = False
        xlo *= 0.5
        if count > MAX_RESCALE:
            return xlo
        count += 1
    if xlo_rescaled and xlo*2 < xhi:
        xhi = xlo * 2
    else:
        count = 0
        while True:
            yhi = fcheck(xhi, *args)
            if yhi:
                xhi_rescaled = count > 0
                break
            xhi *= 1.5
            if count > MAX_RESCALE:
                raise RuntimeError("Failed to find a rhs bound.")
            count += 1
        if xhi_rescaled and xhi/1.5 > xlo:
            xlo = xhi / 1.5
# search
    cycle = 0
    while xhi-xlo > xtol:
        if cycle > MAX_CYCLE:
            raise RuntimeError
        cycle += 1
        xmi = 0.5*(xlo + xhi)
        fmi = fcheck(xmi, *args)
        if fmi:
            xhi = xmi
        else:
            xlo = xmi
    xret = xhi if ret_bigger else xlo
    return xret
def _get_refuniq_map(cell):
    """
    Return:
        refuniqshl_map[Ish]: the uniq shl "ISH" that corresponds to ref
        shl "Ish".
        uniq_atms: a list of unique atom symbols.
        uniq_bas: concatenate basis for all uniq atomsm, i.e.,
                    ``[*cell._basis[atm] for atm in uniq_atms]``
        uniq_bas_loc: uniq bas loc by uniq atoms (similar to cell.ao_loc)
    """
# get uniq atoms that respect the order it appears in cell
    n = len(cell._basis.keys())
    uniq_atms = []
    for i in range(cell.natm):
        atm = cell.atom_symbol(i)
        if atm not in uniq_atms:
            uniq_atms.append(atm)
        if len(uniq_atms) == n:
            break
    natm_uniq = len(uniq_atms)
# get uniq basis
    uniq_bas = [bas for ATM in uniq_atms for bas in cell._basis[ATM]]
    uniq_bas_loc = np.cumsum([0]+[len(cell._basis[ATM]) for ATM in uniq_atms])
    atms = np.array([cell.atom_symbol(i) for i in range(cell.natm)])
    shlstart = np.concatenate([cell.aoslice_nr_by_atom()[:,0], [cell.nbas]])
    refuniqshl_map = np.empty(cell.nbas, dtype=int)
    for IATM in range(natm_uniq):
        Iatms = np.where(atms==uniq_atms[IATM])[0]
        for ISHL in range(*uniq_bas_loc[IATM:IATM+2]):
            Ishlshift = ISHL - uniq_bas_loc[IATM]
            refuniqshl_map[shlstart[Iatms]+Ishlshift] = ISHL
# format to int32 (for interfacing C code)
    refuniqshl_map = np.asarray(refuniqshl_map, dtype=np.int32)
    return refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc
def _get_norm(a, axis=None):
    return np.linalg.norm(a, axis=axis)
def _get_cell_id_in_cellplusimag(cell, nimgs):
    Nimgs = np.array(nimgs)*2+1
    natm = cell.natm
    i0 = Nimgs[0]//2 * np.prod(Nimgs[1:]) + \
            Nimgs[1]//2 * Nimgs[2] + Nimgs[2]//2
    return i0 * natm
def _get_dist_mat(rs1, rs2, dmin=1e-16):
    d = (np.linalg.norm(rs1,axis=1)**2.)[:,None] + \
         np.linalg.norm(rs2,axis=1)**2. - 2.*np.dot(rs1,rs2.T)
    np.clip(d, dmin**2., None, out=d)
    return d**0.5
def _get_Lsmin(cell, Rcuts, uniq_atms, dimension=None):
    """ Given atom-pairwise cutoff, determine the needed lattice translational
    vectors, Ls.
    """
    natm_uniq = len(uniq_atms)
    Rcut = Rcuts.max()
# build cell plus imgs
    b = cell.reciprocal_vectors(norm_to=1)
    heights_inv = np.linalg.norm(b, axis=1)
    # see issue #1017 and PR #1044 for details of boundary_penalty
    scaled_atom_coords = cell.atom_coords().dot(b.T)
    boundary_penalty = np.max([abs(scaled_atom_coords).max(axis=0),
                               abs(1 - scaled_atom_coords).max(axis=0)], axis=0)
    nimgs = np.ceil(Rcut * heights_inv + boundary_penalty).astype(int)
    if dimension is None:
        dimension = cell.dimension
    if dimension == 0:
        nimgs = [0, 0, 0]
    elif dimension == 1:
        nimgs = [nimgs[0], 0, 0]
    elif dimension == 2:
        nimgs = [nimgs[0], nimgs[1], 0]
    cell_all = pbctools.cell_plus_imgs(cell, nimgs)
    Rs_all = cell_all.atom_coords()
    natm_all = cell_all.natm
    atms_all = np.asarray([cell_all.atom_symbol(ia) for ia in range(natm_all)])
# find atoms from the ref cell
    iatm0_ref = _get_cell_id_in_cellplusimag(cell, nimgs)
    natm_ref = cell.natm
    atms_ref = np.asarray([cell.atom_symbol(ia) for ia in range(natm_ref)])
    Rs_ref = Rs_all[iatm0_ref:iatm0_ref+natm_ref]
    mask_ref = np.zeros(natm_all, dtype=bool)
    mask_ref[iatm0_ref:iatm0_ref+natm_ref] = True
# find all atoms that (1) outside ref cell and (2) within Rcut
    uniq_atm_ids_ref = [np.where(atms_ref==atm)[0] for atm in uniq_atms]
    atm_ref_uniq_ids = np.empty(natm_ref, dtype=int)
    for iatm in range(natm_uniq):
        atm_ref_uniq_ids[uniq_atm_ids_ref[iatm]] = iatm
    atm_to_keep = []
    for iatm in range(natm_uniq):
        atm1 = uniq_atms[iatm]
        atm1_ids = np.where(atms_all==atm1)[0]
        d_atm1_ref = _get_dist_mat(Rs_all[atm1_ids], Rs_ref)
        d_atm1_ref[mask_ref[atm1_ids]] = Rcut*100 # exclude atms from ref cell
        mask_ = np.zeros(d_atm1_ref.shape[0], dtype=bool)
        for jatm_ref in range(natm_ref):
            jatm_uniq = atm_ref_uniq_ids[jatm_ref]
            np.logical_or(mask_, d_atm1_ref[:,jatm_ref]<Rcuts[iatm,jatm_uniq],
                          out=mask_)
        atm_to_keep.append(atm1_ids[mask_])
    atm_to_keep = np.sort(np.concatenate(atm_to_keep))

    Rs_sup = np.vstack([Rs_ref, Rs_all[atm_to_keep]])
# atom posvec to (uniq) cell posvec
    latvec = cell.lattice_vectors()
    rs = cell.atom_coords()
    ds = (Rs_sup[:,None,:] - rs).reshape(-1,3)
    ts = np.round( np.linalg.solve(latvec.T, ds.T).T, 4 )
    ids_keep = np.where(abs(np.rint(ts)-ts).sum(axis=1) < 1e-6)[0]
    Ts = ts[ids_keep]
    Ls = lib.dot(Ts, latvec)
    ls = np.dot(Ls, np.random.rand(3))
    uniq_ls, uniq_idx = np.unique(ls, return_index=True)
    return np.asarray(Ls[uniq_idx], order="C")

""" Prescreening 2c
"""
def _Gamma(s, x):
    return gammaincc(s,x) * gamma(s)
def _get_multipole(l, alp):
    return 0.5*np.pi * (2*l+1)**0.5 / alp**(l+1.5)
def _get_2c2e_Rcut(bas_lst, omega, precision, Rprec=1., lmp=True, lasympt=True,
                   eta_correct=True, R_correct=False):
    r""" Given a list of pgto by "bas_lst", determine the cutoff radii for
    j2c lat sum s.t. the truncation error drops below "precision".

    Ref:
        Ye, and Berkelbach J. Chem. Phys. 155, 124106 (2021)
        DOI: 10.1063/5.0064151

    Args:
        bas_lst (list):
            A list of basis where the cutoff is estimated for all pairs.
            Example: [[0,(0.5,1.)], [1,(10.2,-0.02),(1.7,0.5),(0.3,0.37)]]
        omega (float):
            Range-separation parameter (only the absolute value matters).
        precision (float):
            Desired precision, e.g., 1e-8.
        Rprec (float):
            The precision for which the cutoff eqn is solved (default: 1 BOHR).
        lmp & lasympt (bool, default: both True):
            These two args together determine the estimator used for the 2c2e
            SR integrals. We *recommend* using the default.
            A full list of option (all eqns are referred to the ref above):
            ---------------------------------
            lmp    lasympt  estimator
            ---------------------------------
            True   True     Olvl, eqn (19)
            True   False    Olv0, eqn (22)
            False  True     O0vl, eqn (23)
            False  False    O0v0, eqn (10)
            ---------------------------------
        eta_correct & R_correct (bool, default: True & False):
            See the discussion in Sec. II E 3 and particularly eqn (60) in the
            ref above. We *recommend* using the default.

    Returns:
        A 1d array of cutoff distances for each AO shl pair (compressed in the usual upper triangular way, ish<jsh).
    """
    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2
    ls = np.array([bas_lst[i][0] for i in range(nbas)])
    es = np.array([bas_lst[i][1][0] for i in range(nbas)])
    cs = np.zeros_like(es)
    lmax = ls.max()
    for l in range(lmax+1):
        idx = np.where(ls==l)[0]
        cs[idx] = mol_gto.gto_norm(l, es[idx])
    etas = lib.pack_tril( 1/((1/es)[:,None]+1/es+1/omega**2.) )
    if lmp: # use real multipoles
        Os = _get_multipole(ls, es)
    else:   # use charges
        Os = _get_multipole(np.zeros_like(ls), es)
    if lasympt: # invoke angl dependence in R
        Ls = lib.pack_tril( ls[:,None]+ls )
    else:       # use (s|s) formula
        Ls = np.zeros_like(etas).astype(int)
    Os *= cs
    facs = lib.pack_tril(Os[:,None] * Os) / np.pi**0.5

    def estimate1(ij, R0,R1):
        l = Ls[ij]
        fac = facs[ij]
        eta = etas[ij]
        prec0 = precision * (min(eta,1.) if eta_correct else 1.)
        def fcheck(R):
            prec = prec0 * (min(1./R,1.) if R_correct else 1.)
            I = fac * _Gamma(l+0.5, eta*R**2.) / R**(l+1)
            return I < prec
        return _binary_search(R0, R1, Rprec, True, fcheck)

    R0 = 5
    R1 = 20
    Rcuts = np.zeros(n2)
    ij = 0
    for i in range(nbas):
        for j in range(i+1):
            Rcuts[ij] = estimate1(ij, R0,R1)
            ij += 1
    return Rcuts
def _get_atom_Rcuts_2c(Rcuts, bas_loc):
    natm = len(bas_loc) - 1
    atom_Rcuts = np.zeros((natm,natm))
    Rcuts_ = lib.unpack_tril(Rcuts)
    for iatm in range(natm):
        i0,i1 = bas_loc[iatm:iatm+2]
        for jatm in range(iatm+1):
            j0,j1 = bas_loc[jatm:jatm+2]
            Rcut = Rcuts_[i0:i1,j0:j1].max()
            atom_Rcuts[iatm,jatm] = atom_Rcuts[jatm,iatm] = Rcut
    return atom_Rcuts

""" Prescreening 3c
"""
def _fintor_sreri(mol, intor, shls_slice, omega, safe):
    if safe:
        I = mol.intor(intor, shls_slice=shls_slice)
        with mol.with_range_coulomb(abs(omega)):
            I -= mol.intor(intor, shls_slice=shls_slice)
    else:
        with mol.with_range_coulomb(-abs(omega)):
            I = mol.intor(intor, shls_slice=shls_slice)
    return I
def _get_schwartz_data(bas_lst, omega, dijs_lst=None, keep1ctr=True, safe=True):
    """
        if dijs_lst is None:
            "2c"-mode:  Q = 2-norm[(a|a)]^(1/2)
        else:
            "4c"-mode:  Q = 2-norm[(ab|ab)]^(1/2)
    """
    def get1ctr(bas_lst):
        """ For a shell consists of multiple contracted GTOs, keep only the
        one with the greatest weight on the most diffuse primitive GTOs
        (since others are likely core orbitals).
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
                bas_new = [bas[0]] + list(zip(es,cs))
            bas_lst_new.append(bas_new)
        return bas_lst_new
    if keep1ctr:
        bas_lst = get1ctr(bas_lst)
    if dijs_lst is None:
        mol = MoleNoBasSort()
        mol.build(dump_input=False, parse_arg=False, atom="H 0 0 0", basis=bas_lst, spin=None)
        nbas = mol.nbas
        intor = "int2c2e"
        Qs = np.zeros(nbas)
        for k in range(nbas):
            shls_slice = (k,k+1,k,k+1)
            I = _fintor_sreri(mol, intor, shls_slice, omega, safe)
            Qs[k] = _get_norm( I )**0.5
    else:
        def compute1_(mol, dij, intor, shls_slice, omega, safe):
            mol._env[mol._atm[1,mol_gto.PTR_COORD]] = dij
            return _get_norm(
                        _fintor_sreri(mol, intor, shls_slice, omega, safe)
                    )**0.5
        mol = MoleNoBasSort()
        mol.build(dump_input=False, parse_arg=False, atom="H 0 0 0", basis=bas_lst, spin=None)
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
def _get_schwartz_dcut(bas_lst, omega, precision, r0=None, safe=True):
    """ Given a list of basis, determine cutoff radius for the Schwartz Q
    between each unique shell pair to drop below "precision". The Schwartz
    Q is define:
        Q = 2-norm[ (ab|ab) ]^(1/2)

    Return:
        1d array of length nbas*(nbas+1)//2 with nbas=len(bas_lst).
    """
    mol = MoleNoBasSort()
    mol.build(dump_input=False, parse_arg=False, atom="H 0 0 0; H 0 0 0", basis=bas_lst)
    nbas = len(bas_lst)
    n2 = nbas*(nbas+1)//2

    es = np.array([mol.bas_exp(i).min() for i in range(nbas)])
    etas = 1/(1/es[:,None] + 1/es)

    intor = "int2e"
    def estimate1(ish,jsh,R0,R1):
        shls_slice = (ish,ish+1,nbas+jsh,nbas+jsh+1,
                      ish,ish+1,nbas+jsh,nbas+jsh+1)
        prec0 = precision * min(etas[ish,jsh],1.)
        def fcheck(R):
            mol._env[mol._atm[1,mol_gto.PTR_COORD]] = R
            I = _get_norm(_fintor_sreri(mol, intor, shls_slice, omega, safe))**0.5
            prec = prec0 * min(1./R,1.)
            return I < prec
        return _binary_search(R0, R1, 1, True, fcheck)

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
def _make_dijs_lst(dcuts, dstep):
    return [np.arange(0,dcut,dstep) for dcut in dcuts]
def _get_bincoeff(d,e1,e2,l1,l2):
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
def _get_3c2e_Rcuts_for_d(mol, auxmol, ish, jsh, dij, omega, precision,
                          estimator, Qij, Rprec=1,
                          eta_correct=True, R_correct=True):
    r""" Determine for AO shlpr (ish,jsh) separated by dij, the cutoff radius
    for 2-norm( (ksh|v_SR(omega)|ish,jsh) ) < precision.

    Ref:
        Ye, and Berkelbach J. Chem. Phys. 155, 124106 (2021)
        DOI: 10.1063/5.0064151

    Args:
        mol/auxmol (Mole object):
            Provide AO/aux basis info.
        ish/jsh (int):
            AO shl index.
        dij (float):
            Separation between ish and jsh; in BOHR
        omega (float):
            Define the SR Coulomb potential: erfc(omega * r12) / r12
        precision (float):
            The target precision.
        estimator (str):
            Estimator for the SR integrals. We *recommend* using "ME".
            A full list of options (all eqns are referred to the ref above):
            - "ISF": eqn (28)
                Assuming (ss|s) type integrals.
            - "ISF0":
                Assuming (ss|X), i.e., the angular momentum of ket (auxiliary)
                is considered.
            - "ISFQ0": eqn (35)
                Assuming (Q_ss|X), i.e., the AO pair density in the bra is
                approximated by the Schwartz Q integrals of two s orbitals.
            - "ISFQL": eqn (36)
                Assuming (Q_lmax|X). Similar to "ISFQ0" but the angular
                dependence of the AO pair is considered with lmax=l1+l2.
                This is found to improve upon "ISFQ0" when omega is not small.
            - "ME": eqn (41)
                An approximate multiple expansion.
        Qij (float):
            The Schwarz Q integrals between ish and jsh; see eqn (31) in the
            ref above.
        Rprec (float; default: 1):
            A binary search will be used to find the appropriate cutoff Rcut.
            This arg controls the precision to which the binary search is
            performed. The default is 1 [Bohr].
        eta_correct/R_correct (bool; default: both True):
            See the discussion in Sec. II E 3 and particularly eqn (59) in the
            ref above. We *recommend* using the default.

    Returns:
        An array of cutoff distances by ksh (i.e,. the auxiliary shl index).
    """
# sanity check for estimators
    ESTIMATOR = estimator.upper()
    if ESTIMATOR not in ["ISF0", "ISF", "ISFQ0", "ISFQL", "ME"]:
        raise RuntimeError("Unknown estimator requested {}".format(estimator))

# get bas info
    nbasaux = auxmol.nbas
    # some auxbasis sets like def2-ri/jkfit have cGTOs --> use most diffuse pGTO
    eks = [auxmol.bas_exp(ksh)[-1] for ksh in range(nbasaux)]
    lks = [int(auxmol.bas_angular(ksh)) for ksh in range(nbasaux)]
    cks = [abs(auxmol._libcint_ctr_coeff(ksh)[-1]).max()
           for ksh in range(nbasaux)]

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
    def init_feval(e1,e2,e3,l1,l2,l3,c1,c2,c3, d, Q, ESTIMATOR):
        e12 = e1+e2
        l12 = l1+l2

        eta1 = 1/(1/e12+1/e3)
        eta2 = 1/(1/eta1+1/omega**2.)
        eta12 = 1/(1/e1+1/e2)

        fac = c1*c2*c3 * 0.5/np.pi
        if ESTIMATOR == "ME":

            O3 = _get_multipole(l3, e3)

            if d < 1e-3:    # concentric
                ls = np.arange(abs(l1-l2),l12+1)
                O12s = _get_multipole(ls, e12)
                l_facs = O12s * O3 * e12**(0.5*(ls-l12)) * (
                                gamma(max(l12,1))/gamma(np.maximum(ls,1)))**0.5
            else:
                fac *= np.exp(-eta12*d**2.)
                ls = np.arange(0,l12+1)
                O12s = _get_multipole(ls, e12)
                l_facs = O12s * O3 * abs(_get_bincoeff(d,e1,e2,l1,l2))

            def feval(R):
                I = (l_facs * _Gamma(ls+l3+0.5,eta2*R**2.) / R**(ls+l3+1)).sum()
                return I * fac

        elif ESTIMATOR == "ISF0":

            O12 = _get_multipole(0, e12)
            O3 = _get_multipole(0, e3)
            fac *= np.exp(-eta12*d**2.)

            def feval(R):
                return fac * O12 * O3 * _Gamma(0.5, eta2*R**2) / R

        elif ESTIMATOR == "ISF":

            O12 = _get_multipole(0, e12)
            O3 = _get_multipole(l3, e3)
            fac *= np.exp(-eta12*d**2.)

            def feval(R):
                return fac * O12 * O3 * _Gamma(l3+0.5, eta2*R**2) / R**(l3+1)

        elif ESTIMATOR in ["ISFQ0","ISFQL"]:

            eta1212 = 0.5 * e12
            eta1212w = 1/(1/eta1212+1/omega**2.)

            O3 = _get_multipole(l3, e3)

            def feval(R):

                if ESTIMATOR == "ISFQ0":
                    L12 = 0
                    Q2S = 2*np.pi**0.75/(2*(eta1212**0.5-eta1212w**0.5))**0.5/(c1*c2)
                    O12 = Q * Q2S
                    veff = _Gamma(L12+l3+0.5, eta2*R**2) / R**(L12+l3+1)
                else:
                    l12min = abs(l1-l2) if d<1e-3 else 0
                    ls = np.arange(l12min,l12+1)
                    l_facs = (eta1212**(ls+0.5) - eta1212w**(ls+0.5))**-0.5
                    veffs = _Gamma(ls+l3+0.5, eta2*R**2.) / R**(ls+l3+1)
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
        feval = init_feval(e1,e2,e3,l1,l2,l3,c1,c2,c3, dij, Qij, ESTIMATOR)

        eta2 = 1/(1/(e1+e2)+1/e2+1/omega**2.)
        prec0 = precision * (min(eta2,1.) if eta_correct else 1.)
        def fcheck(R):
            prec = prec0 * (min(1./R,1.) if R_correct else 1.)
            I = feval(R)
            return I < prec
        return _binary_search(R0, R1, Rprec, True, fcheck)

# estimating Rcuts
    Rcuts = np.zeros(nbasaux)
    R0 = 5
    R1 = 20
    for ksh in range(nbasaux):
        Rcuts[ksh] = estimate1(ksh, R0, R1)

    return Rcuts
def _get_3c2e_Rcuts(bas_lst_or_mol, auxbas_lst_or_auxmol, dijs_lst, omega,
                    precision, estimator, Qijs_lst, Rprec=1,
                    eta_correct=True, R_correct=True):
    """ Given a list of basis ("bas_lst") and auxiliary basis ("auxbas_lst"),
    determine the cutoff radius for
        2-norm( (k|v_SR(omega)|ij) ) < precision
    where i and j shls are separated by d specified by "dijs_lst".
    """

    if isinstance(bas_lst_or_mol, mol_gto.mole.MoleBase):
        mol = bas_lst_or_mol
    else:
        bas_lst = bas_lst_or_mol
        mol = MoleNoBasSort()
        mol.build(dump_input=False, parse_arg=False, atom="H 0 0 0", basis=bas_lst, spin=None)

    if isinstance(auxbas_lst_or_auxmol, mol_gto.mole.MoleBase):
        auxmol = auxbas_lst_or_auxmol
    else:
        auxbas_lst = auxbas_lst_or_auxmol
        auxmol = MoleNoBasSort()
        auxmol.build(dump_input=False, parse_arg=False, atom="H 0 0 0", basis=auxbas_lst, spin=None)

    nbas = mol.nbas

    ij = 0
    Rcuts = []
    for i in range(nbas):
        for j in range(i+1):
            dijs = dijs_lst[ij]
            Qijs = Qijs_lst[ij]
            for dij,Qij in zip(dijs,Qijs):
                Rcuts_dij = _get_3c2e_Rcuts_for_d(mol, auxmol, i, j, dij,
                                                  omega, precision,
                                                  estimator, Qij,
                                                  Rprec=Rprec,
                                                  eta_correct=eta_correct,
                                                  R_correct=R_correct)
                Rcuts.append(Rcuts_dij)
            ij += 1
    Rcuts = np.asarray(Rcuts).reshape(-1)
    return Rcuts
def _get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, bas_loc, auxbas_loc):
    natm = len(bas_loc) - 1
    assert (len(auxbas_loc) == natm+1)
    bas_loc_inv = np.concatenate([[i]*(bas_loc[i+1]-bas_loc[i])
                                  for i in range(natm)])
    nbas = bas_loc[-1]
    nbas2 = nbas*(nbas+1)//2
    nbasaux = auxbas_loc[-1]
    Rcuts_ = Rcuts.reshape(-1,nbasaux)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst])

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

""" bvk
"""
def _get_bvk_data(cell, Ls, bvk_kmesh):
    ### [START] Hongzhou's style of bvk
    # Using Ls = translations.dot(a)
    translations = np.linalg.solve(cell.lattice_vectors().T, Ls.T)
    # t_mod is the translations inside the BvK cell
    t_mod = translations.round(3).astype(int) % np.asarray(bvk_kmesh)[:,None]
    cell_loc_bvk = np.ravel_multi_index(t_mod, bvk_kmesh).astype(np.int32)

    nimgs = Ls.shape[0]
    bvk_nimgs = np.prod(bvk_kmesh)
    iL_by_bvk = np.zeros(nimgs, dtype=int)
    cell_loc = np.zeros(bvk_nimgs+1, dtype=int)
    shift = 0
    for i in range(bvk_nimgs):
        x = np.where(cell_loc_bvk == i)[0]
        nx = x.size
        cell_loc[i+1] = nx
        iL_by_bvk[shift:shift+nx] = x
        shift += nx

    cell_loc[1:] = np.cumsum(cell_loc[1:])
    cell_loc_bvk = np.asarray(cell_loc, dtype=np.int32, order="C")

    Ls_sorted = np.array(Ls[iL_by_bvk], order="C")
    ### [END] Hongzhou's style of bvk
    bvkmesh_Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_kmesh, True)

    return Ls_sorted, bvkmesh_Ls, cell_loc_bvk

""" determining omega/mesh and basis splitting
"""
def _estimate_ke_cutoff_for_omega_kpt_corrected(cell, omega, precision, kmax):
    fac = 32*np.pi**2    # Qiming
    # fac = 4 * cell.vol / np.pi  # Hongzhou
    ke_cutoff = -2*omega**2 * np.log(precision / (fac*omega**2))
    ke_cutoff = ((2*ke_cutoff)**0.5 + kmax)**2. * 0.5
    return ke_cutoff

def estimate_omega_for_npw(cell, npw_max, precision=None, kmax=0,
                           round2odd=True):
    """ Find the biggest omega where the appropriate PW mesh for achieving a
    given precision in the LR integrals computed using AFT does not exceed
    npw_max in size.

    Args:
        cell (PBC Cell object)
        npw_max (int):
            The returned mesh will stasify
                mesh[0]*mesh[1]*mesh[2] <= npw_max
        precision (float; default: None):
            Target precision for the integrals. If not provided, cell.precision
            is used.
        kmax (float; default: 0):
            'Gmin' in eqn (28) of https://arxiv.org/pdf/2012.07929.pdf
            See :func:'RSDF._rs_build' for how to determine it for given a kpt
            mesh; note the use of a minimum kmax there to improve accuracy for
            Gamma point.
        round2odd (bool; default: True):
            If True, the mesh is required to be odd in each dimension.

    Returns:
        omega (float), ke_cutoff (float), mesh (array of size 3).
    """

    if precision is None: precision = cell.precision
    # TODO: add extra precision for small omega ~ 2*omega / np.pi**0.5
    latvecs = cell.lattice_vectors()

    def omega2all(omega):
        ke_cutoff = _estimate_ke_cutoff_for_omega_kpt_corrected(cell, omega,
                                                                precision, kmax)
        mesh = pbctools.cutoff_to_mesh(latvecs, ke_cutoff)
        if round2odd:
            mesh = _round_off_to_odd_mesh(mesh)
        return ke_cutoff, mesh
    def fcheck(omega):
        return np.prod(omega2all(omega)[1]) > npw_max
    omega_rg = np.asarray([0.05,2])
    omega = _binary_search(*omega_rg, 0.02, False, fcheck)
    ke_cutoff, mesh = omega2all(omega)

    return omega, ke_cutoff, mesh

def estimate_mesh_for_omega(cell, omega, precision=None, kmax=0,
                            round2odd=True):
    """ Find the appropriate PW mesh size s.t. the LR integrals computed via AFT
    achieves a given precision.

    Args:
        See Args for :func:'estimate_omega_for_npw'.
    """

    if precision is None: precision = cell.precision
    ke_cutoff = _estimate_ke_cutoff_for_omega_kpt_corrected(cell, omega,
                                                            precision, kmax)
    mesh = pbctools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)
    if round2odd:
        mesh = _round_off_to_odd_mesh(mesh)

    return ke_cutoff, mesh

""" short-range j2c via screened lattice sum
"""
def intor_j2c(cell, omega, precision=None, kpts=None, hermi=1, shls_slice=None,
              no_screening=False):
    """ Calculate the SR 2c integrals (i| erfc(omega*r12)/r12 |j) via a real-
    space lattice sum.

    Args:
        cell (PBC Cell object)
        omega (float):
            The omega in the SR Coulomb potential (only its absolute value will
            be used).
        precision (float; default: None):
            Target precision for the integrals. If not provided, cell.precision
            is used.
        kpts (array; default: None):
            The kpoint mesh used to sample the Brillouin zone. If not provided,
            Gamma point is used.
        hermi (int; default: 1):
            Hermitian symmetry: 0 for s1, and 1 for s2.
        shls_slice (array of size 4; default: None):
            Range of shl indices organized as (ish0, ish1, jsh0, jsh1). If not
            provided, i/jsh0 = 0 and i/jsh1 = cell.nbas.
        no_screening (bool; default: False):
            If set to True, the integral bound is only used in determining a
            global rcut for the lattice sum and no further (i.e., shlpair-wise)
            screening is performed. This arg is for debug purpose.

    Returns:
        If a single kpt, return the matrix of integrals in the GTO basis.
        Otherwise, return a list of matrices corresponding to different kpts.
    """

    log = logger.Logger(cell.stdout, cell.verbose)

    t1 = np.asarray([logger.process_clock(), logger.perf_counter()])

    intor = "int2c2e"
    intor, comp = mol_gto.moleintor._get_intor_and_comp(
                                            cell._add_suffix(intor), None)
    assert (comp == 1)

# prescreening data
    if precision is None: precision = cell.precision

    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = _get_refuniq_map(cell)
    Rcuts = _get_2c2e_Rcut(uniq_bas, omega, precision, lmp=True, lasympt=True,
                           eta_correct=True, R_correct=False)
    Rcut2s = np.ones(Rcuts)*1e20 if no_screening else Rcuts**2.
    atom_Rcuts = _get_atom_Rcuts_2c(Rcuts, uniq_bas_loc)
    cell_rcut = atom_Rcuts.max()
    Ls = _get_Lsmin(cell, atom_Rcuts, uniq_atms)
    log.debug1("j2c prescreening: cell rcut %.2f Bohr  keep %d imgs",
               cell_rcut, Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
# end prescreening data

    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    if hermi == 0:
        aosym = 's1'
    else:
        aosym = 's2'
    fill = getattr(libpbc, 'PBCsr2c_fill_k'+aosym)
    fintor = getattr(mol_gto.moleintor.libcgto, intor)
    cintopt = lib.c_null_ptr()

    pcell = cell.copy(deep=False)
    pcell.precision = min(cell.precision, cell.precision)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                             cell._atm, cell._bas, cell._env)
    env[mol_gto.PTR_RANGE_OMEGA] = -abs(omega)
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    j0 += cell.nbas
    j1 += cell.nbas
    ao_loc = mol_gto.moleintor.make_loc(bas, intor)
    ni = ao_loc[i1] - ao_loc[i0]
    nj = ao_loc[j1] - ao_loc[j0]

    out = np.empty((nkpts,comp,ni,nj), dtype=np.complex128)

    expkL = np.asarray(np.exp(1j*np.dot(kpts_lst, Ls.T)), order='C')
    drv = libpbc.PBCsr2c_k_drv

    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(i0, i1, j0, j1),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
        Rcut2s.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pcell.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

    mat = []
    for k, kpt in enumerate(kpts_lst):
        v = out[k]
        if hermi != 0:
            for ic in range(comp):
                lib.hermi_triu(v[ic], hermi=hermi, inplace=True)
        if comp == 1:
            v = v[0]
        if abs(kpt).sum() < 1e-9:  # gamma_point
            v = v.real
        mat.append(v)

    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        mat = mat[0]

    t1 = log.timer_debug1('j2c latsum', *t1)

    return mat

""" short-range j3c via real space lattice sum
    Modified from pyscf.pbc.df.outcore/incore
"""
def _aux_e2_nospltbas(cell, auxcell_or_auxbasis, omega, erifile,
                      intor='int3c2e', aosym='s2ij', comp=None,
                      kptij_lst=None, dataname='eri_mo', shls_slice=None,
                      max_memory=2000, bvk_kmesh=None, precision=None,
                      estimator="ME", verbose=0):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, nao_pair, naux) or four-index
    integral tensor (kptij_idx, comp, nao_pair, naux) are stored on disk.

    **This function should be only used by RSGDF initialization function
    _make_j3c**

    Args:
        kptij_lst : ``(*,2,3)`` array
            A list of (kpti, kptj)
        estimator (str; default: "ME"):
            The integral estimator used for screening. Options are
            - "ME": multipole expansion-based estimator
            - "ISFQ0": Izmaylov-Scuseris-Frisch (ISF)-style estimator adapted to
                       3c2e integrals.
            - "ISFQL": Modified ISF-style estimator to account for angular
                       momentum (hence "L" in the name).
            "ME" (the default) is slightly more accurate than the other two,
            especially for orbitals of high angular momentum. For details, see
            discussion in Ye, and Berkelbach J. Chem. Phys. 155, 124106 (2021)
            DOI: 10.1063/5.0064151

    '''
    log = logger.Logger(cell.stdout, cell.verbose)

    if isinstance(auxcell_or_auxbasis, mol_gto.MoleBase):
        auxcell = auxcell_or_auxbasis
    else:
        auxcell = make_auxcell(cell, auxcell_or_auxbasis)

# prescreening data
    t1 = (logger.process_clock(), logger.perf_counter())
    if precision is None: precision = cell.precision
    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = _get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = \
                                                        _get_refuniq_map(auxcell)

    dstep = 1 # 1 Ang bin size for shl pair
    dstep_BOHR = dstep / BOHR
    Qauxs = _get_schwartz_data(uniq_basaux, omega, keep1ctr=False, safe=True)
    dcuts = _get_schwartz_dcut(uniq_bas, omega, precision/Qauxs.max(),
                               r0=cell.rcut)
    dijs_lst = _make_dijs_lst(dcuts, dstep_BOHR)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst]).astype(np.int32)
    if estimator.upper() in ["ISFQ0","ISFQL"]:
        Qs_lst = _get_schwartz_data(uniq_bas, omega, dijs_lst, keep1ctr=True,
                                    safe=True)
    else:
        Qs_lst = [np.zeros_like(dijs) for dijs in dijs_lst]
    Rcuts = _get_3c2e_Rcuts(uniq_bas, uniq_basaux, dijs_lst, omega, precision,
                            estimator, Qs_lst)
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    atom_Rcuts = _get_atom_Rcuts_3c(Rcuts, dijs_lst, bas_exps, uniq_bas_loc,
                                    uniq_basaux_loc)
    cell_rcut = atom_Rcuts.max()
    uniqexp = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    nbasauxuniq = len(uniq_basaux)
    dcut2s = dcuts**2.
    Rcut2s = Rcuts**2.
    Ls = _get_Lsmin(cell, atom_Rcuts, uniq_atms)
    prescreening_data = (refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp,
                         dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls)
    log.debug("j3c prescreening: cell rcut %.2f Bohr  keep %d imgs",
              cell_rcut, Ls.shape[0])
    t1 = log.timer_debug1('prescrn warmup', *t1)
# prescreening data ends here

    intor, comp = mol_gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if isinstance(erifile, h5py.Group):
        feri = erifile
    elif h5py.is_hdf5(erifile):
        feri = lib.H5FileWrap(erifile, 'a')
    else:
        feri = lib.H5FileWrap(erifile, 'w')
    if dataname in feri:
        del (feri[dataname])
    if dataname+'-kptij' in feri:
        del (feri[dataname+'-kptij'])

    if kptij_lst is None:
        kptij_lst = np.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    shlpr_mask = np.ones((shls_slice[1]-shls_slice[0],
                          shls_slice[3]-shls_slice[2]),
                          dtype=np.int8, order="C")

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = int(ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]])
    nj = int(ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]])
    nkptij = len(kptij_lst)

    nii = int(ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = np.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert (shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = np.double
    else:
        dtype = np.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    from pyscf.ao2mo.outcore import balance_segs
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = np.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    bufs = [buf, np.empty_like(buf)]
    bufmem = buf.size*16/1024**2.
    if bufmem > max_memory * 0.5:
        raise RuntimeError("""
Computing 3c2e integrals requires %.2f MB memory, which exceeds the given
maximum memory %.2f MB. Try giving PySCF more memory."""
                           % (bufmem*2., max_memory))

    int3c = wrap_int3c_nospltbas(cell, auxcell, omega, shlpr_mask,
                                 prescreening_data, intor, aosym, comp,
                                 kptij_lst,
                                 bvk_kmesh=bvk_kmesh)

    tspans = np.zeros((2,2))     # cmpt, cmpt+save
    tspannames = ["cmpt", "cmpt+save"]
    def process(aux_range):
        sh0, sh1, nrow = aux_range
        sub_slice = (shls_slice[0], shls_slice[1],
                     shls_slice[2], shls_slice[3],
                     shls_slice[4]+sh0, shls_slice[4]+sh1)
        mat = np.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype,
                         buffer=bufs[0])
        bufs[:] = bufs[1], bufs[0]
        tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
        int3c(sub_slice, mat)
        tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
        tspans[0] += tock_ - tick_
        return mat

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = np.hstack([np.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])

    tril_idx = np.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]

    tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
    for istep, mat in enumerate(lib.map_with_prefetch(process, auxranges)):
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v
        mat = None
    tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
    tspans[1] += tock_ - tick_

    for tspan, tspanname in zip(tspans, tspannames):
        log.debug1("    CPU time for %10s %9.2f sec, wall time %9.2f sec",
                   "%10s"%tspanname, *tspan)
    log.debug1("%s", "")

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile

def wrap_int3c_nospltbas(cell, auxcell, omega, shlpr_mask, prescreening_data,
                         intor='int3c2e', aosym='s1',
                         comp=1, kptij_lst=np.zeros((1,2,3)),
                         cintopt=None, bvk_kmesh=None):
    log = logger.Logger(cell.stdout, cell.verbose)

    refuniqshl_map, auxuniqshl_map, nbasauxuniq, uniqexp, dcut2s, dstep_BOHR, Rcut2s, dijs_loc, Ls = prescreening_data

# GTO data
    intor = cell._add_suffix(intor)
    pcell = cell.copy(deep=False)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = mol_gto.conc_env(cell._atm, cell._bas, cell._env,
                                             cell._atm, cell._bas, cell._env)
    ao_loc = mol_gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    ao_loc = np.asarray(np.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                        dtype=np.int32)
    atm, bas, env = mol_gto.conc_env(atm, bas, env,
                                     auxcell._atm, auxcell._bas, auxcell._env)
    env[mol_gto.PTR_RANGE_OMEGA] = -abs(omega)
    nimgs = len(Ls)
    nbas = cell.nbas

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]

    if bvk_kmesh is None:
        Ls_ = Ls
    else:
        Ls, Ls_, cell_loc_bvk = _get_bvk_data(cell, Ls, bvk_kmesh)
        bvk_nimgs = Ls_.shape[0]

    if gamma_point(kptij_lst):
        assert (aosym[:2] == "s2")
        kk_type = 'g'
        dtype = np.double
        nkpts = nkptij = 1
        kptij_idx = np.array([0], dtype=np.int32)
        expkL = np.ones(1)
    elif is_zero(kpti-kptj):  # j_only
        kk_type = 'k'
        dtype = np.complex128
        kpts = kptij_idx = np.asarray(kpti, order='C')
        expkL = np.exp(1j * np.dot(kpts, Ls_.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = np.complex128
        kpts = unique(np.vstack([kpti,kptj]))[0]
        expkL = np.exp(1j * np.dot(kpts, Ls_.T))
        wherei = np.where(abs(kpti.reshape(-1,1,3)-kpts).sum(axis=2)
                          < KPT_DIFF_TOL)[1]
        wherej = np.where(abs(kptj.reshape(-1,1,3)-kpts).sum(axis=2)
                          < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = np.asarray(wherei*nkpts+wherej, dtype=np.int32)
        nkptij = len(kptij_lst)

    if cintopt is None:
        if nbas > 0:
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
# Remove the precomputed pair data because the pair data corresponds to the
# integral of cell #0 while the lattice sum moves shls to all repeated images.
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)

    cfunc_prefix = "PBCsr3c"
    if not (gamma_point(kptij_lst) or bvk_kmesh is None):
        cfunc_prefix += "_bvk"
    fill = "%s_%s%s" % (cfunc_prefix, kk_type, aosym[:2])
    drv = getattr(libpbc, "%s_%s_drv"%(cfunc_prefix,kk_type))

    log.debug("Using %s to evaluate SR integrals", fill)

    if gamma_point(kptij_lst):
        def int3c(shls_slice, out):
            assert out.dtype == dtype
            shls_slice = (shls_slice[0], shls_slice[1],
                          nbas+shls_slice[2], nbas+shls_slice[3],
                          nbas*2+shls_slice[4], nbas*2+shls_slice[5])
            drv(getattr(libpbc, intor), getattr(libpbc, fill),
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(comp), ctypes.c_int(nimgs),
                Ls.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int*6)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                cintopt,
                shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbasauxuniq),
                uniqexp.ctypes.data_as(ctypes.c_void_p),
                dcut2s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(dstep_BOHR),
                Rcut2s.ctypes.data_as(ctypes.c_void_p),
                dijs_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),
                env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size)
                )
            return out

    else:

        if bvk_kmesh is None:
            def int3c(shls_slice, out):
                assert out.dtype == dtype
                shls_slice = (shls_slice[0], shls_slice[1],
                              nbas+shls_slice[2], nbas+shls_slice[3],
                              nbas*2+shls_slice[4], nbas*2+shls_slice[5])
                drv(getattr(libpbc, intor), getattr(libpbc, fill),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nkptij), ctypes.c_int(nkpts),
                    ctypes.c_int(comp), ctypes.c_int(nimgs),
                    Ls.ctypes.data_as(ctypes.c_void_p),
                    expkL.ctypes.data_as(ctypes.c_void_p),
                    kptij_idx.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int*6)(*shls_slice),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    cintopt,
                    shlpr_mask.ctypes.data_as(ctypes.c_void_p),  # shlpr_mask
                    refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbasauxuniq),
                    uniqexp.ctypes.data_as(ctypes.c_void_p),
                    dcut2s.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(dstep_BOHR),
                    Rcut2s.ctypes.data_as(ctypes.c_void_p),
                    dijs_loc.ctypes.data_as(ctypes.c_void_p),
                    atm.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(cell.natm),
                    bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCsr3c_drv
                    env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size)
                    )
                return out
        else:
            def int3c(shls_slice, out):
                assert out.dtype == dtype
                shls_slice = (shls_slice[0], shls_slice[1],
                              nbas+shls_slice[2], nbas+shls_slice[3],
                              nbas*2+shls_slice[4], nbas*2+shls_slice[5])
                drv(getattr(libpbc, intor), getattr(libpbc, fill),
                    out.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nkptij), ctypes.c_int(nkpts),
                    ctypes.c_int(comp), ctypes.c_int(nimgs),
                    ctypes.c_int(bvk_nimgs),
                    Ls.ctypes.data_as(ctypes.c_void_p),
                    expkL.ctypes.data_as(ctypes.c_void_p),
                    kptij_idx.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int*6)(*shls_slice),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    cintopt,
                    cell_loc_bvk.ctypes.data_as(ctypes.c_void_p), # cell_loc_bvk
                    shlpr_mask.ctypes.data_as(ctypes.c_void_p),   # shlpr_mask
                    refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    auxuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbasauxuniq),
                    uniqexp.ctypes.data_as(ctypes.c_void_p),
                    dcut2s.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(dstep_BOHR),
                    Rcut2s.ctypes.data_as(ctypes.c_void_p),
                    dijs_loc.ctypes.data_as(ctypes.c_void_p),
                    atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                    bas.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbas),
                    env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size)
                    )
                return out

    return int3c
