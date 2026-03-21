#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import ctypes
import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as mol_rhf
from pyscf.grad.rhf import _write
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.dft.multigrid import MultiGridNumInt2

SCREEN_VHF_DM_CONTRA = getattr(__config__, 'pbc_rhf_grad_screen_vhf_dm_contract', True)
libpbc = lib.load_library('libpbc')

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None, kpt=np.zeros(3)):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    if hasattr(mf, '_numint'):
        ni = mf._numint
        assert isinstance(ni, MultiGridNumInt2)
    else:
        ni = mf.with_df
        raise NotImplementedError

    s1 = mf_grad.get_ovlp(mol, kpt)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0, kpt)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    if gamma_point(kpt):
        h1ao = -mol.pbc_intor('int1e_ipkin', kpt=kpt)
        if mol._pseudo:
            de  = ni.vpploc_part1_nuc_grad(dm0, kpts=kpt.reshape(-1,3))
            de += pp_int.vpploc_part2_nuc_grad(mol, dm0)
            de += pp_int.vppnl_nuc_grad(mol, dm0)
            if hasattr(ni, 'vpplocG_part1'):
                if ni.vpplocG_part1 is None:
                    h1ao -= ni.get_vpploc_part1_ip1(kpts=kpt.reshape(-1,3))
        else:
            de = ni.get_nuc_nuc_grad(dm0, kpts=kpt)
            h1ao -= ni.get_nuc_ip1(kpts=kpt)
        de += _contract_vhf_dm(mf_grad, np.add(h1ao, vhf), dm0) * 2
        de += _contract_vhf_dm(mf_grad, s1, dme0) * -2
        h1ao = s1 = vhf = dm0 = dme0 = None
        de = de[atmlst]
    else:
        raise NotImplementedError

    for k, ia in enumerate(atmlst):
        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de, atmlst)
    return de


def _contract_vhf_dm(mf_grad, vhf, dm, comp=3, atmlst=None,
                     screen=SCREEN_VHF_DM_CONTRA):
    from pyscf.gto.mole import ao_loc_nr, ATOM_OF
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list

    t0 = (logger.process_clock(), logger.perf_counter())

    mol = mf_grad.mol
    natm = mol.natm
    nbas = mol.nbas
    shls_slice = np.asarray([0,nbas,0,nbas], order="C", dtype=np.int32)
    ao_loc = np.asarray(ao_loc_nr(mol), order="C", dtype=np.int32)
    shls_atm = np.asarray(mol._bas[:,ATOM_OF].copy(), order="C", dtype=np.int32)

    de = np.zeros((natm,comp), order="C")
    vhf = np.asarray(vhf, order="C")
    dm = np.asarray(dm, order="C")

    if screen:
        neighbor_list = build_neighbor_list_for_shlpairs(mol)
    else:
        neighbor_list = lib.c_null_ptr()
    func = getattr(libpbc, "contract_vhf_dm", None)
    try:
        func(de.ctypes.data_as(ctypes.c_void_p),
             vhf.ctypes.data_as(ctypes.c_void_p),
             dm.ctypes.data_as(ctypes.c_void_p),
             ctypes.byref(neighbor_list),
             shls_slice.ctypes.data_as(ctypes.c_void_p),
             ao_loc.ctypes.data_as(ctypes.c_void_p),
             shls_atm.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(comp), ctypes.c_int(natm),
             ctypes.c_int(nbas))
    except RuntimeError:
        raise
    free_neighbor_list(neighbor_list)

    if atmlst is not None:
        de = de[atmlst]

    logger.timer(mf_grad, '_contract_vhf_dm', *t0)
    return de


def get_ovlp(cell, kpt=np.zeros(3)):
    return -cell.pbc_intor('int1e_ipovlp', kpt=kpt)


def get_veff(mf_grad, mol, dm, kpt=np.zeros(3)):
    mf = mf_grad.base
    xc_code = getattr(mf, 'xc', None)
    kpts = kpt.reshape(-1,3)
    return -mf._numint.get_veff_ip1(dm, xc_code=xc_code, kpts=kpts)


def grad_nuc(cell, atmlst=None, ew_eta=None, ew_cut=None):
    from pyscf.pbc.gto import ewald_methods

    t0 = (logger.process_clock(), logger.perf_counter())

    grad = ewald_methods.ewald_nuc_grad(cell, ew_eta, ew_cut)
    if atmlst is not None:
        grad = grad[atmlst]

    logger.timer(cell, 'nuclear gradient', *t0)
    return grad


class GradientsBase(mol_rhf.GradientsBase):
    '''Base class for Gamma-point nuclear gradient'''
    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        return grad_nuc(mol, atmlst)

    def get_ovlp(self, mol=None, kpt=np.zeros(3)):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol, kpt)

    def optimizer(self, solver='ase'):
        '''Geometry optimization solver
        '''
        solver = solver.lower()
        if solver == 'ase':
            from pyscf.geomopt import ase_solver
            return ase_solver.GeometryOptimizer(self.base)
        else:
            raise RuntimeError(f'Optimization solver {solver} not supported')


class Gradients(GradientsBase):
    '''Non-relativistic Gamma-point restricted Hartree-Fock gradients'''
    def get_veff(self, mol=None, dm=None, kpt=np.zeros(3)):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm, kpt)

    make_rdm1e = mol_rhf.Gradients.make_rdm1e
    grad_elec = grad_elec
