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
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import symm
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf.fci.addons import _unpack_nelec
from pyscf import __config__

libfci = lib.load_library('libfci')

TOTIRREPS = 8

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=None):
    return direct_spin1.contract_1e(f1e, fcivec, norb, nelec, link_index)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    if orbsym is None:
        return direct_spin1.contract_2e(eri, fcivec, norb, nelec, link_index)

    eri = ao2mo.restore(4, eri, norb)
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa, link_indexb = direct_spin1._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    eri_irs, rank_eri, irrep_eri = reorder_eri(eri, norb, orbsym)

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    aidx, link_indexa = gen_str_irrep(strsa, orbsym, link_indexa, rank_eri, irrep_eri)
    if neleca == nelecb:
        bidx, link_indexb = aidx, link_indexa
    else:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        bidx, link_indexb = gen_str_irrep(strsb, orbsym, link_indexb, rank_eri, irrep_eri)

    Tirrep = ctypes.c_void_p*TOTIRREPS
    linka_ptr = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in link_indexa])
    linkb_ptr = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in link_indexb])
    eri_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in eri_irs])
    dimirrep = (ctypes.c_int*TOTIRREPS)(*[x.shape[0] for x in eri_irs])
    fcivec_shape = fcivec.shape
    fcivec = fcivec.reshape((na,nb), order='C')
    ci1new = numpy.zeros_like(fcivec)
    nas = (ctypes.c_int*TOTIRREPS)(*[x.size for x in aidx])
    nbs = (ctypes.c_int*TOTIRREPS)(*[x.size for x in bidx])

# aa, ab
    ci0 = []
    ci1 = []
    for ir in range(TOTIRREPS):
        ma, mb = aidx[ir].size, bidx[wfnsym^ir].size
        ci0.append(numpy.zeros((ma,mb)))
        ci1.append(numpy.zeros((ma,mb)))
        if ma > 0 and mb > 0:
            lib.take_2d(fcivec, aidx[ir], bidx[wfnsym^ir], out=ci0[ir])
    ci0_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci0])
    ci1_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci1])
    libfci.FCIcontract_2e_symm1(eri_ptrs, ci0_ptrs, ci1_ptrs,
                                ctypes.c_int(norb), nas, nbs,
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                linka_ptr, linkb_ptr, dimirrep,
                                ctypes.c_int(wfnsym))
    for ir in range(TOTIRREPS):
        if ci0[ir].size > 0:
            lib.takebak_2d(ci1new, ci1[ir], aidx[ir], bidx[wfnsym^ir])

# bb, ba
    ci0T = []
    for ir in range(TOTIRREPS):
        mb, ma = bidx[ir].size, aidx[wfnsym^ir].size
        ci0T.append(numpy.zeros((mb,ma)))
        if ma > 0 and mb > 0:
            lib.transpose(ci0[wfnsym^ir], out=ci0T[ir])
    ci0, ci0T = ci0T, None
    ci1 = [numpy.zeros_like(x) for x in ci0]
    ci0_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci0])
    ci1_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci1])
    libfci.FCIcontract_2e_symm1(eri_ptrs, ci0_ptrs, ci1_ptrs,
                                ctypes.c_int(norb), nbs, nas,
                                ctypes.c_int(nlinkb), ctypes.c_int(nlinka),
                                linkb_ptr, linka_ptr, dimirrep,
                                ctypes.c_int(wfnsym))
    for ir in range(TOTIRREPS):
        if ci0[ir].size > 0:
            lib.takebak_2d(ci1new, lib.transpose(ci1[ir]), aidx[wfnsym^ir], bidx[ir])
    return ci1new.reshape(fcivec_shape)


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

    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, ecore=ecore, **unknown)
    return e, c

make_rdm1 = direct_spin1.make_rdm1
make_rdm1s = direct_spin1.make_rdm1s
make_rdm12 = direct_spin1.make_rdm12

trans_rdm1s = direct_spin1.trans_rdm1s
trans_rdm1 = direct_spin1.trans_rdm1
trans_rdm12 = direct_spin1.trans_rdm12

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym, wfnsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())

def _id_wfnsym(cis, norb, nelec, orbsym, wfnsym):
    if wfnsym is None:
        neleca, nelecb = _unpack_nelec(nelec)
        wfnsym = 0  # Ag, A1 or A
        for i in orbsym[nelecb:neleca]:
            wfnsym ^= i
    elif isinstance(wfnsym, str):
        wfnsym = symm.irrep_name2id(cis.mol.groupname, wfnsym)
    return wfnsym % 10

def _gen_strs_irrep(strs, orbsym):
    orbsym = numpy.asarray(orbsym) % 10
    irreps = numpy.zeros(len(strs), dtype=numpy.int32)
    if isinstance(strs, cistring.OIndexList):
        nocc = strs.shape[1]
        for i in range(nocc):
            irreps ^= orbsym[strs[:,i]]
    else:
        for i, ir in enumerate(orbsym):
            irreps[numpy.bitwise_and(strs, 1<<i) > 0] ^= ir
    return irreps

def _get_init_guess(airreps, birreps, nroots, hdiag, orbsym, wfnsym=0):
    na = len(airreps)
    nb = len(birreps)
    ci0 = []
    iroot = 0
    for addr in numpy.argsort(hdiag):
        x = numpy.zeros((na*nb))
        addra = addr // nb
        addrb = addr % nb
        if airreps[addra] ^ birreps[addrb] == wfnsym:
            x[addr] = 1
            ci0.append(x)
            iroot += 1
            if iroot >= nroots:
                break
    try:
        # Add noise
        ci0[0][0 ] += 1e-5
        ci0[0][-1] -= 1e-5
    except IndexError:
        raise IndexError('Configuration of required symmetry (wfnsym=%d) not found' % wfnsym)
    return ci0
def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps = birreps = _gen_strs_irrep(strsa, orbsym)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        birreps = _gen_strs_irrep(strsb, orbsym)
    return _get_init_guess(airreps, birreps, nroots, hdiag, orbsym, wfnsym)


def reorder_eri(eri, norb, orbsym):
    if orbsym is None:
        return [eri], numpy.arange(norb), numpy.zeros(norb,dtype=numpy.int32)
# map irrep IDs of Dooh or Coov to D2h, C2v
# see symm.basis.linearmole_symm_descent
    orbsym = numpy.asarray(orbsym) % 10
# irrep of (ij| pair
    trilirrep = (orbsym[:,None]^orbsym)[numpy.tril_indices(norb)]
# and the number of occurence for each irrep
    dimirrep = numpy.asarray(numpy.bincount(trilirrep), dtype=numpy.int32)
# we sort the irreps of (ij| pair, to group the pairs which have same irreps
# "order" is irrep-id-sorted index. The (ij| paired is ordered that the
# pair-id given by order[0] comes first in the sorted pair
# "rank" is a sorted "order". Given nth (ij| pair, it returns the place(rank)
# of the sorted pair
    old_eri_irrep = numpy.asarray(trilirrep, dtype=numpy.int32)
    rank_in_irrep = numpy.empty_like(old_eri_irrep)
    p0 = 0
    eri_irs = [numpy.zeros((0,0))] * TOTIRREPS
    for ir, nnorb in enumerate(dimirrep):
        idx = numpy.asarray(numpy.where(trilirrep == ir)[0], dtype=numpy.int32)
        rank_in_irrep[idx] = numpy.arange(nnorb, dtype=numpy.int32)
        eri_irs[ir] = lib.take_2d(eri, idx, idx)
        p0 += nnorb
    return eri_irs, rank_in_irrep, old_eri_irrep

def gen_str_irrep(strs, orbsym, link_index, rank_eri, irrep_eri):
    airreps = _gen_strs_irrep(strs, orbsym)
    na = len(airreps)
    rank = numpy.zeros(na, dtype=numpy.int32)
    aidx = [numpy.zeros(0,dtype=numpy.int32)] * TOTIRREPS
    for ir in range(TOTIRREPS):
        aidx[ir] = numpy.where(airreps == ir)[0]
        ma = len(aidx[ir])
        if ma > 0:
            rank[aidx[ir]] = numpy.arange(ma, dtype=numpy.int32)
    link_index = link_index.copy()
    link_index[:,:,1] = irrep_eri[link_index[:,:,0]]
    link_index[:,:,0] = rank_eri[link_index[:,:,0]]
    link_index[:,:,2] = rank[link_index[:,:,2]]
    link_index = [link_index.take(aidx[ir], axis=0) for ir in range(TOTIRREPS)]
    return aidx, link_index


class FCISolver(direct_spin1.FCISolver):

    davidson_only = getattr(__config__, 'fci_direct_spin1_symm_FCI_davidson_only', True)
    # pspace may break point group symmetry
    pspace_size = getattr(__config__, 'fci_direct_spin1_symm_FCI_pspace_size', 0)

    def __init__(self, mol=None, **kwargs):
        direct_spin1.FCISolver.__init__(self, mol, **kwargs)
        # wfnsym will be guessed based on initial guess if it is None
        self.wfnsym = None

    def dump_flags(self, verbose=None):
        direct_spin1.FCISolver.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        if isinstance(self.wfnsym, str):
            log.info('Input CI wfn symmetry = %s', self.wfnsym)
        elif isinstance(self.wfnsym, (int, numpy.number)):
            log.info('Input CI wfn symmetry = %s',
                     symm.irrep_id2name(self.mol.groupname, self.wfnsym))
        else:
            log.info('CI wfn symmetry = %s', self.wfnsym)
        return self

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        nelec = _unpack_nelec(nelec, self.spin)
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        nelec = _unpack_nelec(nelec, self.spin)
        return direct_spin1.make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        nelec = _unpack_nelec(nelec, self.spin)
        return direct_spin1.pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        wfnsym = _id_wfnsym(self, norb, nelec, orbsym, wfnsym)
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, wfnsym, **kwargs)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = _id_wfnsym(self, norb, nelec, self.orbsym, self.wfnsym)
        nelec = _unpack_nelec(nelec, self.spin)
        return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

    def guess_wfnsym(self, norb, nelec, fcivec=None, orbsym=None, wfnsym=None,
                     **kwargs):
        '''
        Guess point group symmetry of the FCI wavefunction.  If fcivec is
        given, the symmetry of fcivec is used.  Otherwise the symmetry is
        based on the HF determinant.
        '''
        if orbsym is None:
            orbsym = self.orbsym
        nelec = _unpack_nelec(nelec, self.spin)
        if fcivec is None:
            wfnsym = _id_wfnsym(self, norb, nelec, orbsym, wfnsym)
        else:
            # TODO: if wfnsym is given in the input, check whether the
            # symmetry of fcivec is consistent with given wfnsym.
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
            e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
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
    nelec = mol.nelectron-1
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    numpy.random.seed(1)
    na = cistring.num_strings(norb, nelec//2+1)
    nb = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,nb))

    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = orbsym
    fcivec = addons.symmetrize_wfn(fcivec, norb, nelec, cis.orbsym, wfnsym=0)

    ci1 = cis.contract_2e(eri, fcivec, norb, nelec, orbsym=cis.orbsym, wfnsym=0)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))

    ci1 = contract_2e(eri, fcivec, norb, nelec, orbsym=orbsym)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
    cis.wfnsym = 3
    e = cis.kernel(h1e, eri, norb, nelec, ecore=m.energy_nuc(), davidson_only=True)[0]
    print(e, e - -74.695029029452357)

    mol.atom = [['H', (0, 0, i)] for i in range(8)]
    mol.basis = {'H': 'sto-3g'}
    mol.symmetry = True
    mol.build()
    m = scf.RHF(mol)
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron + 1
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    na = cistring.num_strings(norb, nelec//2+1)
    nb = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,nb))
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    orbsym = orbsym % 10
    fcivec = addons.symmetrize_wfn(fcivec, norb, nelec, orbsym, wfnsym=5)
    cis = FCISolver(mol)
    cis.orbsym = orbsym
    cis.wfnsym = 5
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
