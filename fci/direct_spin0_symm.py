#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for Singlet state
#
# Other files in the directory
# direct_ms0   MS=0, same number of alpha and beta nelectrons
# direct_spin0 singlet
# direct_spin1 arbitary number of alpha and beta electrons, based on RHF/ROHF
#              MO integrals
# direct_uhf   arbitary number of alpha and beta electrons, based on UHF
#              MO integrals
#

import os
import ctypes
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.symm
import pyscf.ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf.fci import direct_ms0
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm

libfci = pyscf.lib.load_library('libmcscf')

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=[]):
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
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=[]):
    if not orbsym:
        return direct_spin0.contract_2e(eri, fcivec, norb, nelec, link_index)

    eri = pyscf.ao2mo.restore(4, eri, norb)
    if link_index is None:
        if isinstance(nelec, int):
            neleca = nelec//2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na,nlink,_ = link_index.shape
    ci1 = numpy.empty((na,na))

    eri, link_index, dimirrep = \
            direct_spin1_symm.reorder4irrep(eri, norb, link_index, orbsym)
    dimirrep = numpy.array(dimirrep, dtype=numpy.int32)

    libfci.FCIcontract_2e_spin0_symm(eri.ctypes.data_as(ctypes.c_void_p),
                                     fcivec.ctypes.data_as(ctypes.c_void_p),
                                     ci1.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(norb), ctypes.c_int(na),
                                     ctypes.c_int(nlink),
                                     link_index.ctypes.data_as(ctypes.c_void_p),
                                     dimirrep.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(len(dimirrep)))
    return pyscf.lib.transpose_sum(ci1, inplace=True)


def kernel(h1e, eri, norb, nelec, ci0=None, eshift=.1, tol=1e-8, orbsym=[],
           *kwargs):
    if isinstance(nelec, int):
        neleca = nelec//2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    h1e = numpy.ascontiguousarray(h1e)
    eri = numpy.ascontiguousarray(eri)
    link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na = link_index.shape[0]
    hdiag = direct_ms0.make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = direct_ms0.pspace(h1e, eri, norb, nelec, hdiag)
    pw, pv = scipy.linalg.eigh(h0)
    if len(addr) == na*na:
        ci0 = numpy.empty((na*na))
        ci0[addr] = pv[:,0]
        if abs(pw[0]-pw[1]) > 1e-12:
            return pw[0], pyscf.lib.transpose_sum(ci0.reshape(na,na),True)*.5

    precond = direct_spin1.make_pspace_precond(hdiag, pw, pv, addr, eshift)
    #precond = lambda x, e, *args: x/(hdiag-(e-eshift))

    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec, link_index, orbsym)
        return hc.ravel()

    if ci0 is None:
        ci0 = numpy.zeros(na*na)
        ci0[0] = 1
    else:
        ci0 = ci0.ravel()

    e, c = pyscf.lib.davidson(hop, ci0, precond, tol=tol, lindep=1e-8)
    return e, pyscf.lib.transpose_sum(c.reshape(na,na)) * .5

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    return direct_spin0.make_rdm1(fcivec, norb, nelec, link_index)

# alpha and beta 1pdm
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    return direct_spin0.make_rdm1s(fcivec, norb, nelec, link_index)

# dm_pq,rs = <|p^+ q r^+ s|>
# dm_pq,rs = dm_sr,qp;  dm_qp,rs = dm_rs,qp
# need call reorder_rdm for this rdm2 to get standard 2pdm

def make_rdm12(fcivec, norb, nelec, link_index=None):
    return direct_spin0.make_rdm12(fcivec, norb, nelec, link_index)

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin0.trans_rdm1s(cibra, ciket, norb, nelec, link_index)

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin0.trans_rdm1(cibra, ciket, norb, nelec, link_index)

# dm_pq,rs = <I|p^+ q r^+ s|J>
def trans_rdm12(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin0.trans_rdm12(cibra, ciket, norb, nelec, link_index)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=[]):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())


class FCISolver(direct_spin0.FCISolver):
    def __init__(self, mol, **kwargs):
        self.orbsym = []
        direct_spin0.FCISolver.__init__(self, mol, **kwargs)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return direct_ms0.make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return direct_ms0.pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=[], **kwargs):
        if not orbsym:
            orbsym = self.orbsym
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, **kwargs)

    def eig(self, op, x0, precond):
        return pyscf.lib.davidson(op, x0, precond, self.conv_threshold,
                                  self.max_cycle, self.max_space, self.lindep,
                                  self.max_memory, log=self.verbose)

    def make_precond(self, hdiag, pspaceig, pspaceci, addr):
        return direct_spin1.make_pspace_precond(hdiag, pspaceig, pspaceci, addr,
                                                self.level_shift)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.mol.check_sanity(self)
        e, ci = direct_ms0.kernel_ms0(self, h1e, eri, norb, nelec, ci0)
# when norb is small, ci is obtained by exactly diagonalization. It can happen
# that the ground state is triplet (ci = -ci.T), symmetrize the coefficients
# will lead to ci = 0
#        ci = pyscf.lib.transpose_sum(ci, inplace=True) * .5
        return e, ci

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return direct_spin0.make_rdm1s(fcivec, norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return direct_spin0.make_rdm1(fcivec, norb, nelec, link_index)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return direct_spin0.make_rdm12(fcivec, norb, nelec, link_index)

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return direct_spin0.trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return direct_spin0.trans_rdm1(cibra, ciket, norb, nelec, link_index)

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return direct_spin0.trans_rdm12(cibra, ciket, norb, nelec, link_index)



if __name__ == '__main__':
    import time
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

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
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.RHF.get_hcore(mol), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    numpy.random.seed(1)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,na))
    fcivec = fcivec + fcivec.T

    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    print(numpy.allclose(orbsym, [0, 0, 2, 0, 3, 0, 2]))
    cis = FCISolver(mol)
    cis.orbsym = cis.orbsym
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin0.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))

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
    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = cis.orbsym
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin0.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
