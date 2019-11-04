#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
from pyscf import symm
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc1step
from pyscf.mcscf import newton_casscf
from pyscf.mcscf import casci_symm
from pyscf import fci


class CASSCF(newton_casscf.CASSCF):
    __doc__ = newton_casscf.CASSCF.__doc__
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        newton_casscf.CASSCF.__init__(self, mf_or_mol, ncas, nelecas, ncore, frozen)
        assert(self.mol.symmetry)
        self.fcisolver = fci.solver(self.mol, False, True)
        self.fcisolver.max_cycle = 25
        #self.fcisolver.max_space = 25

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if callback is None: callback = self.callback
        if _kern is None: _kern = newton_casscf.kernel

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        log = logger.Logger(self.stdout, self.verbose)

        mo_coeff = self.mo_coeff = casci_symm.label_symmetry_(self, mo_coeff)
#
#        if (getattr(self.fcisolver, 'wfnsym', None) and
#            self.fcisolver.wfnsym is None and
#            getattr(self.fcisolver, 'guess_wfnsym', None)):
#            wfnsym = self.fcisolver.guess_wfnsym(self.ncas, self.nelecas, ci0,
#                                                 verbose=log)
#            wfnsym = symm.irrep_id2name(self.mol.groupname, wfnsym)
#            log.info('Active space CI wfn symmetry = %s', wfnsym)

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
        mo = newton_casscf.CASSCF.rotate_mo(self, mo, u, log)
        mo = lib.tag_array(mo, orbsym=self.mo_coeff.orbsym)
        return mo

def _symmetrize(mat, orbsym, groupname):
    mat1 = numpy.zeros_like(mat)
    orbsym = numpy.asarray(orbsym)
    allowed = orbsym.reshape(-1,1) == orbsym
    mat1[allowed] = mat[allowed]
    return mat1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    import pyscf.fci
    from pyscf.mcscf import addons

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
    emc = mc.kernel(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(m, 6, (3,1))
    #mc.fcisolver = pyscf.fci.direct_spin1
    mc.fcisolver = pyscf.fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.kernel(mo)[0]
    print(emc - -75.7155632535814)

    mc = CASSCF(m, 6, (3,1))
    mc.fcisolver.wfnsym = 'B1'
    mc.verbose = 4
    emc = mc.kernel(mo)[0]
    print(emc - -75.6406597705231)
