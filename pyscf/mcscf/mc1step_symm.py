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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc1step
from pyscf.mcscf import mc2step
from pyscf.mcscf import casci_symm
from pyscf.mcscf import addons
from pyscf import fci
from pyscf.soscf.newton_ah import _force_Ex_Ey_degeneracy_


class SymAdaptedCASSCF(mc1step.CASSCF):
    __doc__ = mc1step.CASSCF.__doc__
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        mc1step.CASSCF.__init__(self, mf_or_mol, ncas, nelecas, ncore, frozen)

        assert(self.mol.symmetry)
        fcisolver = self.fcisolver
        if isinstance(fcisolver, fci.direct_spin0.FCISolver):
            self.fcisolver = fci.direct_spin0_symm.FCISolver(self.mol)
        else:
            self.fcisolver = fci.direct_spin1_symm.FCISolver(self.mol)
        self.fcisolver.__dict__.update(fcisolver.__dict__)

    @property
    def wfnsym(self):
        return self.fcisolver.wfnsym
    @wfnsym.setter
    def wfnsym(self, wfnsym):
        self.fcisolver.wfnsym = wfnsym

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback, mc1step.kernel)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback, mc2step.kernel)

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if callback is None: callback = self.callback
        if _kern is None: _kern = mc1step.kernel

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        log = logger.Logger(self.stdout, self.verbose)

        # Initialize/overwrite self.fcisolver.orbsym and self.fcisolver.wfnsym
        mo_coeff = self.mo_coeff = casci_symm.label_symmetry_(self, mo_coeff, ci0)

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        log.note('CASSCF energy = %.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        mask = mc1step.CASSCF.uniq_var_indices(self, nmo, ncore, ncas, frozen)
# Call _symmetrize function to remove the symmetry forbidden matrix elements
# (by setting their mask value to 0 in _symmetrize).  Then pack_uniq_var and
# unpack_uniq_var function only operates on those symmetry allowed matrix
# elements.
        # self.mo_coeff.orbsym is initialized in kernel function
        return _symmetrize(mask, self.mo_coeff.orbsym, self.mol.groupname)

    def _eig(self, mat, b0, b1, orbsym=None):
        # self.mo_coeff.orbsym is initialized in kernel function
        if orbsym is None:
            orbsym = self.mo_coeff.orbsym[b0:b1]
        return casci_symm.eig(mat, orbsym)

    def rotate_mo(self, mo, u, log=None):
        '''Rotate orbitals with the given unitary matrix'''
        mo = mc1step.CASSCF.rotate_mo(self, mo, u, log)
        mo = lib.tag_array(mo, orbsym=self.mo_coeff.orbsym)
        return mo

    def sort_mo_by_irrep(self, cas_irrep_nocc,
                         cas_irrep_ncore=None, mo_coeff=None, s=None):
        '''Select active space based on symmetry information.
        See also :func:`pyscf.mcscf.addons.sort_mo_by_irrep`
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return addons.sort_mo_by_irrep(self, mo_coeff, cas_irrep_nocc,
                                       cas_irrep_ncore, s)

    def newton(self):
        from pyscf.mcscf import newton_casscf_symm
        mc1 = newton_casscf_symm.CASSCF(self._scf, self.ncas, self.nelecas)
        mc1.__dict__.update(self.__dict__)
        mc1.max_cycle_micro = 10
        # MRH, 04/08/2019: enable state-average CASSCF second-order algorithm
        from pyscf.mcscf.addons import StateAverageMCSCFSolver
        if isinstance (self, StateAverageMCSCFSolver):
            mc1 = mc1.state_average_(self.weights)
        return mc1

CASSCF = SymAdaptedCASSCF

def _symmetrize(mat, orbsym, groupname):
    mat1 = numpy.zeros_like(mat)
    orbsym = numpy.asarray(orbsym)
    allowed = orbsym.reshape(-1,1) == orbsym
    mat1[allowed] = mat[allowed]

    if groupname in ('Dooh', 'Coov'):
        _force_Ex_Ey_degeneracy_(mat1, orbsym)
    return mat1

from pyscf import scf
scf.hf_symm.RHF.CASSCF = scf.hf_symm.ROHF.CASSCF = lib.class_as_method(SymAdaptedCASSCF)
scf.uhf_symm.UHF.CASSCF = None


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.symmetry = 1
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASSCF(m, 6, 4)
    mc.fcisolver = pyscf.fci.solver(mol)
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(m, 6, (3,1))
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    print(emc - -75.7155632535814)

    mc = CASSCF(m, 6, (3,1))
    mc.fcisolver.wfnsym = 'B1'
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    print(emc - -75.6406597705231)
