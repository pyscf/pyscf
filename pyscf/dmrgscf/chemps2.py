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

import os, sys
import imp
import numpy
import pyscf.ao2mo

try:
    from pyscf.dmrgscf import settings
    PyCheMPS2 = None
except ImportError:
    try:
        import PyCheMPS2
    except ImportError:
        msg = ('settings.py not found.  Please create %s\n' %
               os.path.join(os.path.dirname(__file__), 'settings.py'))
        sys.stderr.write(msg)

# point group ID defined in CheMPS2, see
# http://sebwouters.github.io/CheMPS2/classCheMPS2_1_1Irreps.html
GROUPNAME_ID = {
    'C1' : 0,
    'Ci' : 1,
    'C2' : 2,
    'Cs' : 3,
    'D2' : 4,
    'C2v': 5,
    'C2h': 6,
    'D2h': 7,
}

class CheMPS2(object):
    def __init__(self, mol, **kwargs):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        if self.mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None
        self.orbsym = []

# ref.
# https://github.com/SebWouters/CheMPS2/blob/master/psi4plugins/dmrgci.cc
        self.wfn_irrep = 0
        #self.spin_2s = 0  # spin = 2*s, 0 means singlet
        self.dmrg_states = [ 200 , 500 , 1000 , 1000 ]
        self.dmrg_noise = [ 1 , 1 , 1 , 0 ]
        self.dmrg_e_convergence = 1e-8
        self.dmrg_noise_factor = 0.03
        self.dmrg_maxiter_noise = 5
        self.dmrg_maxiter_silent = 100

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = pyscf.lib.logger.Logger(self.stdout, verbose)
        log.info('******** CheMPS2 flags ********')
        log.info('dmrg_states = %s', str(self.dmrg_states))
        log.info('dmrg_noise = %s', str(self.dmrg_noise))
        log.info('dmrg_e_convergence = %g', self.dmrg_e_convergence)
        log.info('dmrg_noise_factor = %g', self.dmrg_noise_factor)
        log.info('dmrg_maxiter_noise = %d', self.dmrg_maxiter_noise)
        log.info('dmrg_maxiter_silent = %d', self.dmrg_maxiter_silent)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        global PyCheMPS2
        if PyCheMPS2 is None:
            PyCheMPS2 = imp.load_dynamic('PyCheMPS2', settings.PYCHEMPS2BIN)

        Initializer = PyCheMPS2.PyInitialize()
        Initializer.Init()

        if self.groupname is not None:
            groupNumber = GROUPNAME_ID[self.groupname]
        else:
            groupNumber = 0
            self.orbsym = numpy.zeros(norb, numpy.int32)
        Ham = PyCheMPS2.PyHamiltonian(norb, groupNumber,
                                      numpy.asarray(self.orbsym, dtype=numpy.int32))
        eri = pyscf.ao2mo.restore(1, eri, norb)
        for i in range(norb):
            for j in range(norb):
                totsym = self.orbsym[i] ^ self.orbsym[j]
                if 0 == totsym:
                    Ham.setTmat(i, j, h1e[i,j])
                for k in range(norb):
                    for l in range(norb):
                        totsym = self.orbsym[i] \
                               ^ self.orbsym[j] \
                               ^ self.orbsym[k] \
                               ^ self.orbsym[l]
                        if 0 == totsym:
                            Ham.setVmat(i, k, j, l, eri[i,j,k,l])
        Ham.setEconst(0)

        if isinstance(nelec, (int, numpy.integer)):
            spin2 = 0
        else:
            spin2 = (nelec[0]-nelec[1]) * 2
            nelec = sum(nelec)

        Prob = PyCheMPS2.PyProblem(Ham, spin2, nelec, self.wfn_irrep)
        Prob.SetupReorderD2h()

        OptScheme = PyCheMPS2.PyConvergenceScheme(len(self.dmrg_states))
        for cnt,m in enumerate(self.dmrg_states):
            if self.dmrg_noise[cnt]:
                OptScheme.setInstruction(cnt, m, self.dmrg_e_convergence,
                                         self.dmrg_maxiter_noise,
                                         self.dmrg_noise_factor)
            else:
                OptScheme.setInstruction(cnt, m, self.dmrg_e_convergence,
                                         self.dmrg_maxiter_silent, 0.0)

        with pyscf.lib.capture_stdout() as stdout:
            theDMRG = PyCheMPS2.PyDMRG(Prob, OptScheme)
            Energy = theDMRG.Solve() + ecore
            theDMRG.calc2DMandCorrelations()
            pyscf.lib.logger.debug1(self.mol, stdout.read())

        rdm2 = numpy.empty((norb,)*4)
        for i in range(norb):
            for j in range(norb):
                for k in range(norb):
                    for l in range(norb):
                        rdm2[i,j,k,l] = theDMRG.get2DMA(i, j, k, l)

        with pyscf.lib.capture_stdout() as stdout:
            theDMRG.deleteStoredOperators()
            pyscf.lib.logger.debug1(self.mol, stdout.read())

# The order of deallocation matters!
        del(theDMRG)
        del(OptScheme)
        del(Prob)
        del(Ham)
        del(Initializer)

        fakewfn_by_rdm2 = rdm2
        return Energy, fakewfn_by_rdm2

    def make_rdm12(self, fakewfn_by_rdm2, ncas, nelec, **kwargs):
        if not isinstance(nelec, (int, numpy.integer)):
            nelec = sum(nelec)
# CheMPS2 uses physics notation
        rdm2 = fakewfn_by_rdm2.transpose(0,2,1,3)
        rdm1 = numpy.einsum('ijkk->ij', rdm2) / (nelec-1)
        return rdm1, rdm2

    def make_rdm1(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return self.make_rdm12(fcivec, norb, nelec, **kwargs)[0]



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'out-chemps2',
        atom = [['H', (0.,0.,i)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True,
        symmetry_subgroup = 'D2h',
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    mc.fcisolver = CheMPS2(mol)
    mc.fcisolver.dmrg_e_convergence = 1e-8
    emc_1 = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = CheMPS2(mol)
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'out-casscf',
        atom = [['H', (0.,0.,i)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True,
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    emc_1ref = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    emc_0ref = mc.casci()[0]

    print('CheMPS2-CI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('CheMPS2-SCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))
