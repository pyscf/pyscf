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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Different FCI solvers are implemented to support different type of symmetry.
                    Symmetry
File                Point group   Spin singlet   Real hermitian*    Alpha/beta degeneracy
direct_spin0_symm   Yes           Yes            Yes                Yes
direct_spin1_symm   Yes           No             Yes                Yes
direct_spin0        No            Yes            Yes                Yes
direct_spin1        No            No             Yes                Yes
direct_uhf          No            No             Yes                No
direct_nosym        No            No             No**               Yes

*  Real hermitian Hamiltonian implies (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
** Hamiltonian is real but not hermitian, (ij|kl) != (ji|kl) ...
'''

import sys
import ctypes
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import symm
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf import __config__

libfci = lib.load_library('libfci')

TOTIRREPS = 8

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=None):
    return direct_spin0.contract_1e(f1e, fcivec, norb, nelec, link_index)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
# the input fcivec should be symmetrized
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    if orbsym is None:
        return direct_spin0.contract_2e(eri, fcivec, norb, nelec, link_index)

    eri = ao2mo.restore(4, eri, norb)
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    assert(neleca == nelecb)
    link_indexa = direct_spin0._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    eri_irs, rank_eri, irrep_eri = direct_spin1_symm.reorder_eri(eri, norb, orbsym)

    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    aidx, link_indexa = direct_spin1_symm.gen_str_irrep(strsa, orbsym, link_indexa,
                                                        rank_eri, irrep_eri)

    Tirrep = ctypes.c_void_p*TOTIRREPS
    linka_ptr = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in link_indexa])
    eri_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in eri_irs])
    dimirrep = (ctypes.c_int*TOTIRREPS)(*[x.shape[0] for x in eri_irs])
    fcivec_shape = fcivec.shape
    fcivec = fcivec.reshape((na,na), order='C')
    ci1new = numpy.zeros_like(fcivec)
    nas = (ctypes.c_int*TOTIRREPS)(*[x.size for x in aidx])

    ci0 = []
    ci1 = []
    for ir in range(TOTIRREPS):
        ma, mb = aidx[ir].size, aidx[wfnsym^ir].size
        ci0.append(numpy.zeros((ma,mb)))
        ci1.append(numpy.zeros((ma,mb)))
        if ma > 0 and mb > 0:
            lib.take_2d(fcivec, aidx[ir], aidx[wfnsym^ir], out=ci0[ir])
    ci0_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci0])
    ci1_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci1])
    libfci.FCIcontract_2e_symm1(eri_ptrs, ci0_ptrs, ci1_ptrs,
                                ctypes.c_int(norb), nas, nas,
                                ctypes.c_int(nlinka), ctypes.c_int(nlinka),
                                linka_ptr, linka_ptr, dimirrep,
                                ctypes.c_int(wfnsym))
    for ir in range(TOTIRREPS):
        if ci0[ir].size > 0:
            lib.takebak_2d(ci1new, ci1[ir], aidx[ir], aidx[wfnsym^ir])
    return lib.transpose_sum(ci1new, inplace=True).reshape(fcivec_shape)


def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    assert(len(orbsym) == norb)
    cis = FCISolver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.max_space = max_space
    cis.nroots = nroots
    cis.davidson_only = davidson_only
    cis.pspace_size = pspace_size
    cis.orbsym = orbsym
    cis.wfnsym = wfnsym

    unknown = {}
    for k, v in kwargs.items():
        if not hasattr(cis, k):
            unknown[k] = v
        setattr(cis, k, v)
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))

    wfnsym = direct_spin1_symm._id_wfnsym(cis, norb, nelec, cis.orbsym,
                                          cis.wfnsym)
    if cis.wfnsym is not None and ci0 is None:
        ci0 = addons.symm_initguess(norb, nelec, orbsym, wfnsym)

    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, ecore=ecore, **unknown)
    return e, c

make_rdm1 = direct_spin0.make_rdm1
make_rdm1s = direct_spin0.make_rdm1s
make_rdm12 = direct_spin0.make_rdm12

trans_rdm1s = direct_spin0.trans_rdm1s
trans_rdm1 = direct_spin0.trans_rdm1
trans_rdm12 = direct_spin0.trans_rdm12

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym, wfnsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    assert(neleca == nelecb)
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)
    na = nb = len(airreps)

    init_strs = []
    iroot = 0
    for addr in numpy.argsort(hdiag):
        addra = addr // nb
        addrb = addr % nb
        if airreps[addra] ^ airreps[addrb] == wfnsym:
            if (addrb,addra) not in init_strs:
                init_strs.append((addra,addrb))
                iroot += 1
                if iroot >= nroots:
                    break
    ci0 = []
    for addra,addrb in init_strs:
        x = numpy.zeros((na,nb))
        if addra == addrb == 0:
            x[addra,addrb] = 1
        else:
            x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
        ci0.append(x.ravel())

    # Add noise
    #ci0[0][0 ] += 1e-5
    #ci0[0][-1] -= 1e-5
    if len(ci0) == 0:
        raise RuntimeError('No determinant matches the target symmetry %s' %
                           wfnsym)
    return ci0


class FCISolver(direct_spin0.FCISolver):

    davidson_only = getattr(__config__, 'fci_direct_spin1_symm_FCI_davidson_only', True)

    # pspace may break point group symmetry
    pspace_size = getattr(__config__, 'fci_direct_spin1_symm_FCI_pspace_size', 0)

    def __init__(self, mol=None, **kwargs):
        direct_spin0.FCISolver.__init__(self, mol, **kwargs)
        # wfnsym will be guessed based on initial guess if it is None
        self.wfnsym = None

    def dump_flags(self, verbose=None):
        direct_spin0.FCISolver.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        if isinstance(self.wfnsym, str):
            log.info('specified CI wfn symmetry = %s', self.wfnsym)
        elif isinstance(self.wfnsym, (int, numpy.number)):
            log.info('specified CI wfn symmetry = %s',
                     symm.irrep_id2name(self.mol.groupname, self.wfnsym))

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return direct_spin0.make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return direct_spin0.pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym,
                                              wfnsym)
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, wfnsym, **kwargs)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.orbsym,
                                              self.wfnsym)
        return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

    def guess_wfnsym(self, norb, nelec, fcivec=None, orbsym=None, wfnsym=None,
                     **kwargs):
        if orbsym is None:
            orbsym = self.orbsym
        if fcivec is None:
            wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym,
                                                  wfnsym)
        else:
            wfnsym = addons.guess_wfnsym(fcivec, norb, nelec, orbsym)

        verbose = kwargs.get('verbose', None)
        log = logger.new_logger(self, verbose)
        log.debug('Guessing CI wfn symmetry = %s', wfnsym)
        return wfnsym

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec

        wfnsym = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)
        with lib.temporary_env(self, orbsym=orbsym, wfnsym=wfnsym):
            e, c = direct_spin0.kernel_ms0(self, h1e, eri, norb, nelec, ci0, None,
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c

FCI = FCISolver


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g',}
    mol.symmetry = 1
    mol.build()
    m = scf.RHF(mol)
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    numpy.random.seed(1)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,na))
    fcivec = fcivec + fcivec.T

    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    print(numpy.allclose(orbsym, [0, 0, 2, 0, 3, 0, 2]))
    cis = FCISolver(mol)
    cis.orbsym = orbsym
    fcivec = addons.symmetrize_wfn(fcivec, norb, nelec, cis.orbsym, wfnsym=0)
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin0.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
    e = cis.kernel(h1e, eri, norb, nelec, ecore=m.energy_nuc(), davidson_only=True)[0]
    print(e, e - -75.012647118991595)

    mol.atom = [['H', (0, 0, i)] for i in range(8)]
    mol.basis = {'H': 'sto-3g'}
    mol.symmetry = True
    mol.build()
    m = scf.RHF(mol)
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,na))
    fcivec = fcivec + fcivec.T
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = orbsym
    fcivec = addons.symmetrize_wfn(fcivec, norb, nelec, cis.orbsym, wfnsym=0)
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin0.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
