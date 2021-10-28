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

import ctypes
import numpy as np

from pyscf import gto as mol_gto
from pyscf.scf import _vhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
from pyscf.lib import logger
libpbc = lib.load_library('libpbc')

from pyscf.pbc.df.supmol import (suplat_by_Rcut, _build_supmol_,
                                 get_refuniq_map, binary_search)


from scipy.special import gamma, gammaincc
def Gamma(s, x):
    return gammaincc(s,x) * gamma(s)
def get_multipole(l, alp):
    return 0.5*np.pi * (2*l+1)**0.5 / alp**(l+1.5)
def get_2c2e_Rcut(bas_lst, cellvol, omega, precision, Rprec=1.,
                  lmp=True, lasympt=True,
                  eta_correct=True, R_correct=False, vol_correct=False):
    """ Given a list of pgto by "bas_lst", determine the cutoff radii for j2c lat sum s.t. the truncation error drops below "precision". j2c is estimated as

        j12(R) ~ C1*C2 * O1 * O2 * Gamma(l12+0.5, eta*Rc^2) /
                    (pi^0.5 * Rc^(l12+1))

    where l12 = l1+l2, eta = 1/(1/a1+1/a2+1/omega^2). The error introduced by truncating at Rc is

        err ~ \int_Rc^{\infty} dR R^2 j2c(R)
            ~ \int_Rc^{\infty} dR R^2 exp(-eta * R^2)
            ~ j2c(Rc) * Rc / eta

    Arguments "eta_correct" and "R_correct" control whether the corresponding correction is applied.

    Args:
        bas_lst:
            A list of basis where the cutoff is estimated for all pairs.
            Example: [[0,(0.5,1.)], [1,(10.2,-0.02),(1.7,0.5),(0.3,0.37)]]
        cellvol:
            Cell volume (Bohr^3).
        omega (float):
            Range-separation parameter (only the absolute value matters).
        precision (float):
            Desired precision, e.g., 1e-8.
        Rprec (float):
            The precision for which the cutoff eqn is solved (default: 1 BOHR).
        lmp & lasympt (bool, default: both True):
            The final estimator is
                j2c ~ fmp * fasympt
            where
                =========================================================
                | lmp     | fmp                                         |
                ---------------------------------------------------------
                | True    | O1(l1) * O2(l2)                             |
                | False   | O1(0) * O2(0)                               |
                ---------------------------------------------------------
                | lasympt | fasympt                                     |
                | True    | Gamma(l12+0.5,eta*R^2) / (pi^0.5*R^(l12+1)) |
                | False   | erfc(eta^0.5*R) / R                         |
                =========================================================
        eta_correct & R_correct (bool, default: True & False):
            if eta_correct : j2c *= max(1,1/eta)
            if R_correct   : j2c *= min(1,R)
            Apparently, both "corrections" will make the estimate larger, which is equivalent to using a smaller "precision" (i.e., tighter thresh).
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
        Os = get_multipole(ls, es)
    else:   # use charges
        Os = get_multipole(np.zeros_like(ls), es)
    if lasympt: # invoke angl dependence in R
        Ls = lib.pack_tril( ls[:,None]+ls )
    else:       # use (s|s) formula
        Ls = np.zeros_like(etas).astype(int)
    Os *= cs
    facs = lib.pack_tril(Os[:,None] * Os) / np.pi**0.5
# >>>>>>> debug block
    if vol_correct:
        facs *= 2*np.pi / cellvol
# <<<<<<<

    def estimate1(ij, R0,R1):
        l = Ls[ij]
        fac = facs[ij]
        eta = etas[ij]
        prec0 = precision * (min(eta,1.) if eta_correct else 1.)
        def fcheck(R):
            prec = prec0 * (min(1./R,1.) if R_correct else 1.)
            I = fac * Gamma(l+0.5, eta*R**2.) / R**(l+1)
            return I < prec
        return binary_search(R0, R1, Rprec, True, fcheck)

    R0 = 5
    R1 = 20
    Rcuts = np.zeros(n2)
    ij = 0
    for i in range(nbas):
        for j in range(i+1):
            Rcuts[ij] = estimate1(ij, R0,R1)
            ij += 1
    return Rcuts

def get_atom_Rcuts_2c(Rcuts, bas_loc):
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

def make_supmol_j2c(cell, atom_Rcuts, uniq_atms):
    atms_sup, Rs_sup = suplat_by_Rcut(cell, uniq_atms, atom_Rcuts)
    return _build_supmol_(cell, atms_sup, Rs_sup)

def intor_j2c(cell, omega, kpts=np.zeros((1,3)), precision=None,
              use_cintopt=True, safe=True,
# +++++++ Use the default for the following unless you know what you are doing
              lmp=True, lasympt=True,
              eta_correct=True, R_correct=False, vol_correct=False,
# -------
# +++++++ debug options
              ret_timing=False,
              no_screening=False,   # set Rcuts to effectively infinity
# -------
              ):

    cput0 = np.asarray([logger.process_clock(), logger.perf_counter()])

    if precision is None: precision = cell.precision

    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = get_refuniq_map(cell)
    Rcuts = get_2c2e_Rcut(uniq_bas, cell.vol, omega, precision,
                          lmp=lmp, lasympt=lasympt,
                          eta_correct=eta_correct, R_correct=R_correct,
                          vol_correct=vol_correct)
    atom_Rcuts = get_atom_Rcuts_2c(Rcuts, uniq_bas_loc)
    supmol = make_supmol_j2c(cell, atom_Rcuts, uniq_atms)

    logger.debug1(cell, "supmol for j2c: natm= %d, nbas= %d",
                  supmol.natm, supmol.nbas)

    refsupshl_loc = supmol._refsupshl_loc
    refsupshl_map = supmol._refsupshl_map
    ao_loc = cell.ao_loc_nr()
    ao_locsup = supmol.ao_loc_nr()

    nao = cell.nao
    nao2 = nao*(nao+1)//2

    intor = "int2c2e"
    intor, comp = mol_gto.moleintor._get_intor_and_comp(
                                            cell._add_suffix(intor), None)
    assert(comp == 1)

    if use_cintopt:
        cintopt = _vhf.make_cintopt(cell._atm, cell._bas, cell._env, intor)
    else:
        cintopt = lib.c_null_ptr()

    cput1 = np.asarray([logger.process_clock(), logger.perf_counter()])
    dt0 = cput1 - cput0

    kpts = np.asarray(kpts).reshape(-1,3)
    if gamma_point(kpts):
        drv = libpbc.fill_sr2c2e_g
# >>>>> debug block
        if no_screening:
            Rcuts = np.clip(Rcuts, 1e20, None)
# <<<<<
        def fill_j2c(out):
            drv(getattr(libpbc, intor),
                out.ctypes.data_as(ctypes.c_void_p),
                comp, cintopt,
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ao_locsup.ctypes.data_as(ctypes.c_void_p),
                Rcuts.ctypes.data_as(ctypes.c_void_p),
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                refsupshl_loc.ctypes.data_as(ctypes.c_void_p),
                refsupshl_map.ctypes.data_as(ctypes.c_void_p),
                cell._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(cell.natm),
                cell._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(cell.nbas),
                cell._env.ctypes.data_as(ctypes.c_void_p),
                supmol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                supmol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                supmol._env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_char(safe))

        out = np.empty(nao2, dtype=np.double)
        with supmol.with_range_coulomb(-abs(omega)):
            fill_j2c(out)
        out = [lib.unpack_tril(out)]
    else:
        kpts = kpts.reshape(-1,3)
        nkpts = len(kpts)
        expLk = np.exp(1j * lib.dot(supmol._Ls, kpts.T))
        drv = libpbc.fill_sr2c2e_k
# >>>>> debug block
        if no_screening:
            Rcuts = np.clip(Rcuts, 1e20, None)
# <<<<<
        def fill_j2c(out):
            drv(getattr(libpbc, intor),
                out.ctypes.data_as(ctypes.c_void_p),
                comp, cintopt,
                expLk.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkpts),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ao_locsup.ctypes.data_as(ctypes.c_void_p),
                Rcuts.ctypes.data_as(ctypes.c_void_p),
                refuniqshl_map.ctypes.data_as(ctypes.c_void_p),
                refsupshl_loc.ctypes.data_as(ctypes.c_void_p),
                refsupshl_map.ctypes.data_as(ctypes.c_void_p),
                cell._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(cell.natm),
                cell._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(cell.nbas),
                cell._env.ctypes.data_as(ctypes.c_void_p),
                supmol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                supmol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                supmol._env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_char(safe))

        out = np.zeros(nkpts*nao2, dtype=np.complex128)
        with supmol.with_range_coulomb(-abs(omega)):
            fill_j2c(out)
        out = out.reshape(nkpts,nao2)
        outr = lib.unpack_tril(out.real)
        outi = lib.unpack_tril(out.imag, filltriu=2)
        for outi_ in outi: np.fill_diagonal(outi_, 0.)
        out = outr + outi*1j

        out_ = [None] * nkpts
        for k in range(nkpts):
            if gamma_point(kpts[k]):
                out_[k] = out[k].real
            else:
                out_[k] = out[k]
        out = out_
        out_ = None
    cput0 = np.asarray([logger.process_clock(), logger.perf_counter()])
    dt1 = cput0 - cput1

    if ret_timing:
        return out, dt0, dt1
    else:
        return out


if __name__ == "__main__":
    from pyscf.pbc import gto, tools
    from pyscf import df
    from utils import get_lattice_sc40

    fml = "zns"
    atom, a = get_lattice_sc40(fml)
    basis = "ccecp-cc-pvdz"

    cell = gto.Cell(atom=atom, a=a, basis=basis, spin=None)
    cell.verbose = 0
    cell.build()
    cell.verbose = 6

    # cell = tools.super_cell(cell, [2,2,2])

    auxcell = df.make_auxmol(cell)

    kpts = np.zeros((1,3))
    # kpts = np.random.rand(1,3)
    nkpts = len(kpts)

    omega = 0.1

    prec_lst = [1e-8,1e-12]

    js = []
    for prec in prec_lst:
        j = intor_j2c(auxcell, omega, kpts=kpts, precision=prec,
                      ret_timing=False,
                      eta_correct=True,
                      R_correct=False)
        js.append(j)

    for k in range(nkpts):
        print(k, kpts[k])
        for j in js[:-1]:
            errmat = abs(j[k]-js[-1][k])
            maxerr = errmat.max()
            meanerr = errmat.mean()
            print(" ", maxerr, meanerr)
