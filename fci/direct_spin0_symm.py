#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for Singlet state
#
# Other files in the directory
# direct_spin0 singlet
# direct_spin1 arbitary number of alpha and beta electrons, based on RHF/ROHF
#              MO integrals
# direct_uhf   arbitary number of alpha and beta electrons, based on UHF
#              MO integrals
#

import sys
import ctypes
import numpy
import pyscf.lib
import pyscf.gto
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf import symm
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm
from pyscf.fci import addons

libfci = pyscf.lib.load_library('libfci')

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
        if isinstance(nelec, (int, numpy.integer)):
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


def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=.001, tol=1e-10,
           lindep=1e-14, max_cycle=50, nroots=1, orbsym=[], wfnsym=None,
           **kwargs):
    cis = FCISolver(None)
    cis.level_shift = level_shift
    cis.orbsym = orbsym
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.wfnsym = wfnsym
    cis.nroots = nroots

    unknown = []
    for k, v in kwargs.items():
        setattr(cis, k, v)
        if not hasattr(cis, k):
            unknown.append(k)
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown), __name__))

    #wfnsym = direct_spin1_symm._id_wfnsym(cis, norb, nelec, cis.wfnsym)
    wfnsym = 0
    e, c = direct_spin0.kernel_ms0(cis, h1e, eri, norb, nelec, ci0=ci0)
    if cis.wfnsym is not None:
        if cis.nroots > 1:
            c = [addons.symmetrize_wfn(ci, norb, nelec, orbsym, wfnsym)
                 for ci in c]
        else:
            c = addons.symmetrize_wfn(c, norb, nelec, orbsym, wfnsym)
    return e, c

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    return direct_spin0.make_rdm1(fcivec, norb, nelec, link_index)

# alpha and beta 1pdm
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    return direct_spin0.make_rdm1s(fcivec, norb, nelec, link_index)

# dm_pq,rs = <|p^+ q r^+ s|>
# dm_pq,rs = dm_sr,qp;  dm_qp,rs = dm_rs,qp
# need call reorder_rdm for this rdm2 to get standard 2pdm

def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    return direct_spin0.make_rdm12(fcivec, norb, nelec, link_index, reorder)

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin0.trans_rdm1s(cibra, ciket, norb, nelec, link_index)

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin0.trans_rdm1(cibra, ciket, norb, nelec, link_index)

# dm_pq,rs = <I|p^+ q r^+ s|J>
def trans_rdm12(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    return direct_spin0.trans_rdm12(cibra, ciket, norb, nelec, link_index, reorder)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=[]):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())


class FCISolver(direct_spin0.FCISolver):
    def __init__(self, mol, **kwargs):
        self.orbsym = []
        self.wfnsym = None
        direct_spin0.FCISolver.__init__(self, mol, **kwargs)
        self.davidson_only = True

    def dump_flags(self, verbose=None):
        direct_spin0.FCISolver.dump_flags(self, verbose)
        if isinstance(self.wfnsym, str):
            logger.info(self, 'specified total symmetry = %s', self.wfnsym)
        elif isinstance(self.wfnsym, (int, numpy.integer)):
            logger.info(self, 'specified total symmetry = %s',
                        symm.irrep_id2name(self.mol.groupname, self.wfnsym))

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=[], **kwargs):
        if not orbsym:
            orbsym = self.orbsym
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, **kwargs)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.wfnsym)
        logger.debug(self, 'total symmetry = %s',
                     symm.irrep_id2name(self.mol.groupname, wfnsym))
        if self.wfnsym is not None and ci0 is None:
            ci0 = addons.symm_initguess(norb, nelec, self.orbsym, wfnsym)
        e, c = direct_spin0.kernel_ms0(self, h1e, eri, norb, nelec, ci0,
                                       **kwargs)
        if self.wfnsym is not None:
            if self.nroots > 1:
                c = [addons.symmetrize_wfn(ci, norb, nelec, self.orbsym, wfnsym)
                     for ci in c]
            else:
                c = addons.symmetrize_wfn(c, norb, nelec, self.orbsym, wfnsym)
        return e, c


if __name__ == '__main__':
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
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    numpy.random.seed(1)
    na = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,na))
    fcivec = fcivec + fcivec.T

    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
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
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = cis.orbsym
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin0.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
