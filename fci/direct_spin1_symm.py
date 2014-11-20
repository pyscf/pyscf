#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for Singlet state
#
# Other files in the directory
# direct_spin1   MS=0, same number of alpha and beta nelectrons
# direct_spin1 singlet
# direct_spin1 arbitary number of alpha and beta electrons, based on RHF/ROHF
#              MO integrals
# direct_uhf   arbitary number of alpha and beta electrons, based on UHF
#              MO integrals
#

import os
import ctypes
import numpy
import pyscf.lib
import pyscf.symm
import pyscf.ao2mo
import davidson
import cistring
import direct_spin1

_loaderpath = os.path.dirname(pyscf.lib.__file__)
libfci = numpy.ctypeslib.load_library('libmcscf', _loaderpath)

def reorder4irrep(eri, norb, link_index, orbsym):
    if not orbsym:
        return eri, link_index, numpy.array(norb, dtype=numpy.int32)
    orbsym = numpy.array(orbsym)
# irrep of (ij| pair
    trilirrep = (orbsym[:,None]^orbsym)[numpy.tril_indices(norb)]
# and the number of occurence for each irrep
    dimirrep = numpy.array(numpy.bincount(trilirrep), dtype=numpy.int32)
# we sort the irreps of (ij| pair, to group the pairs which have same irreps
# "order" is irrep-id-sorted index. The (ij| paired is ordered that the
# pair-id given by order[0] comes first in the sorted pair
# "rank" is a sorted "order". Given nth (ij| pair, it returns the place(rank)
# of the sorted pair
    order = numpy.argsort(trilirrep)
    rank = order.argsort()
    eri = eri[order][:,order]
    link_index_irrep = link_index.copy()
    link_index_irrep[:,:,0] = rank[link_index[:,:,0]]
    return eri, link_index_irrep, dimirrep

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=[]):
    return direct_spin1.contract_1e(f1e, fcivec, norb, nelec, link_index)

# the input fcivec should be symmetrized
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=[],
                bufsize=1024):
    if not orbsym:
        return direct_spin1.contract_2e(eri, fcivec, norb, nelec, link_index,
                                      bufsize)

    eri = pyscf.ao2mo.restore(4, eri, norb)
    if link_index is None:
        if isinstance(nelec, int):
            nelecb = nelec/2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    fcivec = fcivec.reshape(na,nb)
    ci1 = numpy.empty_like(fcivec)

    eri, link_indexa, dimirrep = reorder4irrep(eri, norb, link_indexa, orbsym)
    link_indexb = reorder4irrep(eri, norb, link_indexb, orbsym)[1]
    dimirrep = numpy.array(dimirrep, dtype=numpy.int32)

    libfci.FCIcontract_rhf2e_spin1_symm(eri.ctypes.data_as(ctypes.c_void_p),
                                        fcivec.ctypes.data_as(ctypes.c_void_p),
                                        ci1.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(norb),
                                        ctypes.c_int(na), ctypes.c_int(nb),
                                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                                        dimirrep.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(len(dimirrep)),
                                        ctypes.c_int(bufsize))
    return ci1


def kernel(h1e, eri, norb, nelec, ci0=None, eshift=.1, tol=1e-8, orbsym=[],
           **kwargs):
    if isinstance(nelec, int):
        nelecb = nelec/2
        neleca = nelec - nelecb
        nelec = (neleca, nelecb)
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    hdiag = direct_spin1.make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = direct_spin1.pspace(h1e, eri, norb, nelec, hdiag)
    pw, pv = numpy.linalg.eigh(h0)
    if len(addr) == na*nb:
        ci0 = numpy.empty((na*nb))
        ci0[addr] = pv[:,0]
        return pw[0], ci0.reshape(na,nb)

    def precond(r, e0, x0, *args):
        #h0e0 = h0 - numpy.eye(len(addr))*(e0-eshift)
        h0e0inv = numpy.dot(pv/(pw-(e0-eshift)), pv.T)
        hdiaginv = 1/(hdiag - (e0-eshift))
        h0x0 = x0 * hdiaginv
        #h0x0[addr] = numpy.linalg.solve(h0e0, x0[addr])
        h0x0[addr] = numpy.dot(h0e0inv, x0[addr])
        h0r = r * hdiaginv
        #h0r[addr] = numpy.linalg.solve(h0e0, r[addr])
        h0r[addr] = numpy.dot(h0e0inv, r[addr])
        e1 = numpy.dot(x0, h0r) / numpy.dot(x0, h0x0)
        x1 = r - e1*x0
        #pspace_x1 = x1[addr].copy()
        x1 *= hdiaginv
# pspace (h0-e0)^{-1} cause diverging?
        #x1[addr] = numpy.linalg.solve(h0e0, pspace_x1)
        return x1

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec, link_index, orbsym)
        return hc.ravel()

    if ci0 is None:
        ci0 = numpy.zeros(na*nb)
        ci0[0] = 1
    else:
        ci0 = ci0.ravel()

    e, c = davidson.dsyev(hop, ci0, precond, tol=tol, lindep=1e-8)
    return e, c.reshape(na,nb)

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    return direct_spin1.make_rdm1(fcivec, norb, nelec, link_index)

# alpha and beta 1pdm
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    return direct_spin1.make_rdm1s(fcivec, norb, nelec, link_index)

# dm_pq,rs = <|p^+ q r^+ s|>
# dm_pq,rs = dm_sr,qp;  dm_qp,rs = dm_rs,qp
# need call reorder_rdm for this rdm2 to get standard 2pdm

def make_rdm12(fcivec, norb, nelec, link_index=None):
    return direct_spin1.make_rdm12(fcivec, norb, nelec, link_index)

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin1.trans_rdm1s(cibra, ciket, norb, nelec, link_index)

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin1.trans_rdm1(cibra, ciket, norb, nelec, link_index)

# dm_pq,rs = <I|p^+ q r^+ s|J>
def trans_rdm12(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin1.trans_rdm12(cibra, ciket, norb, nelec, link_index)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=[]):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())


class FCISolver(direct_spin1.FCISolver):
    def __init__(self, mol, **kwargs):
        self.orbsym = []
        direct_spin1.FCISolver.__init__(self, mol, **kwargs)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=[], **kwargs):
        if orbsym is []:
            orbsym = self.orbsym
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, **kwargs)


if __name__ == '__main__':
    import time
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
    nelec = mol.nelectron-1
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.RHF.get_hcore(mol), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    numpy.random.seed(1)
    na = cistring.num_strings(norb, nelec/2+1)
    nb = cistring.num_strings(norb, nelec/2)
    fcivec = numpy.random.random((na,nb))

    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = orbsym

    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))

    link_index = cistring.gen_linkstr_index(range(norb), nelec/2+1)
    eri1 = ao2mo.restore(4, eri, norb)
    eri1,_,dimirrep = reorder4irrep(eri1, norb, link_index, orbsym)
    p0 = 0
    for i in dimirrep:
        eri1[p0:p0+i,p0:p0+i] = 0
        p0 += i
    print(numpy.allclose(eri1, 0))

    ci1 = contract_2e(eri, fcivec, norb, nelec, orbsym=orbsym)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))


    mol.atom = [['H', (0, 0, i)] for i in range(8)]
    mol.basis = {'H': 'sto-3g'}
    mol.symmetry = True
    mol.build()
    m = scf.RHF(mol)
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron + 1
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    na = cistring.num_strings(norb, nelec/2+1)
    nb = cistring.num_strings(norb, nelec/2)
    fcivec = numpy.random.random((na,nb))
    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = orbsym
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
