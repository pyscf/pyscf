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

import sys
import ctypes
import numpy
import pyscf.lib
import pyscf.gto
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf import symm
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import addons

libfci = pyscf.lib.load_library('libfci')

def reorder4irrep(eri, norb, link_index, orbsym):
    if not list(orbsym):
        return eri, link_index, numpy.array(norb, dtype=numpy.int32)
    orbsym = numpy.array(orbsym)
# map irrep IDs of Dooh or Coov to D2h, C2v
# see symm.basis.linearmole_symm_descent
    orbsym = orbsym % 10
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
    eri = eri.take(order,axis=0).take(order,axis=1)
    link_index_irrep = link_index.copy()
    link_index_irrep[:,:,0] = rank[link_index[:,:,0]]
    return numpy.asarray(eri, order='C'), link_index_irrep, dimirrep

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=[]):
    return direct_spin1.contract_1e(f1e, fcivec, norb, nelec, link_index)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=[]):
    fcivec = numpy.asarray(fcivec, order='C')
    if not list(orbsym):
        return direct_spin1.contract_2e(eri, fcivec, norb, nelec, link_index)

    eri = pyscf.ao2mo.restore(4, eri, norb)
    if link_index is None:
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert(fcivec.size == na*nb)
    ci1 = numpy.empty_like(fcivec)

    eri, link_indexa, dimirrep = reorder4irrep(eri, norb, link_indexa, orbsym)
    link_indexb = reorder4irrep(eri, norb, link_indexb, orbsym)[1]
    dimirrep = numpy.array(dimirrep, dtype=numpy.int32)

    libfci.FCIcontract_2e_spin1_symm(eri.ctypes.data_as(ctypes.c_void_p),
                                     fcivec.ctypes.data_as(ctypes.c_void_p),
                                     ci1.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(norb),
                                     ctypes.c_int(na), ctypes.c_int(nb),
                                     ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                     link_indexa.ctypes.data_as(ctypes.c_void_p),
                                     link_indexb.ctypes.data_as(ctypes.c_void_p),
                                     dimirrep.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(len(dimirrep)))
    return ci1


def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=[], wfnsym=None,
           **kwargs):
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
        setattr(cis, k, v)
        if not hasattr(cis, k):
            unknown[k] = v
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))

    wfnsym = _id_wfnsym(cis, norb, nelec, cis.wfnsym)
    if cis.wfnsym is not None and ci0 is None:
        ci0 = addons.symm_initguess(norb, nelec, orbsym, wfnsym)

    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, **unknown)
    if cis.wfnsym is not None:
        if cis.nroots > 1:
            c = [addons.symmetrize_wfn(ci, norb, nelec, orbsym, wfnsym)
                 for ci in c]
        else:
            c = addons.symmetrize_wfn(c, norb, nelec, orbsym, wfnsym)
    return e, c

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    return direct_spin1.make_rdm1(fcivec, norb, nelec, link_index)

# alpha and beta 1pdm
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    return direct_spin1.make_rdm1s(fcivec, norb, nelec, link_index)

# dm_pq,rs = <|p^+ q r^+ s|>
# dm_pq,rs = dm_sr,qp;  dm_qp,rs = dm_rs,qp
# need call reorder_rdm for this rdm2 to get standard 2pdm

def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    return direct_spin1.make_rdm12(fcivec, norb, nelec, link_index, reorder)

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin1.trans_rdm1s(cibra, ciket, norb, nelec, link_index)

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    return direct_spin1.trans_rdm1(cibra, ciket, norb, nelec, link_index)

# dm_pq,rs = <I|p^+ q r^+ s|J>
def trans_rdm12(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    return direct_spin1.trans_rdm12(cibra, ciket, norb, nelec, link_index, reorder)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=[]):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())

def _id_wfnsym(cis, norb, nelec, wfnsym):
    if wfnsym is None:
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        wfnsym = 0  # Ag, A1 or A
        for i in cis.orbsym[nelecb:neleca]:
            wfnsym ^= i
    elif isinstance(wfnsym, str):
        wfnsym = symm.irrep_name2id(cis.mol.groupname, wfnsym) % 10
    return wfnsym

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(norb), nelecb))
    airreps = numpy.zeros(strsa.size, dtype=numpy.int32)
    birreps = numpy.zeros(strsb.size, dtype=numpy.int32)
    for i in range(norb):
        airreps[numpy.bitwise_and(strsa, 1<<i) > 0] ^= orbsym[i]
        birreps[numpy.bitwise_and(strsb, 1<<i) > 0] ^= orbsym[i]
    na = len(strsa)
    nb = len(strsb)

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
    # Add noise
    ci0[0][0 ] += 1e-5
    ci0[0][-1] -= 1e-5
    return ci0


class FCISolver(direct_spin1.FCISolver):
    def __init__(self, mol=None, **kwargs):
        self.orbsym = []
        self.wfnsym = None
        direct_spin1.FCISolver.__init__(self, mol, **kwargs)
        self.davidson_only = True
        self.pspace_size = 0  # Improper pspace size may break symmetry

    def dump_flags(self, verbose=None):
        if verbose is None: verbose = self.verbose
        direct_spin1.FCISolver.dump_flags(self, verbose)
        log = pyscf.lib.logger.Logger(self.stdout, verbose)
        if isinstance(self.wfnsym, str):
            log.info('specified CI wfn symmetry = %s', self.wfnsym)
        elif isinstance(self.wfnsym, (int, numpy.integer)):
            log.info('specified CI wfn symmetry = %s',
                     symm.irrep_id2name(self.mol.groupname, self.wfnsym))
        return self

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return direct_spin1.make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return direct_spin1.pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=[], **kwargs):
        if not list(orbsym):
            orbsym = self.orbsym
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, **kwargs)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = _id_wfnsym(self, norb, nelec, self.wfnsym)
        return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

    def guess_wfnsym(self, norb, nelec, fcivec=None, wfnsym=None, **kwargs):
        if fcivec is None:
            wfnsym = _id_wfnsym(self, norb, nelec, wfnsym)
        else:
            wfnsym = addons.guess_wfnsym(fcivec, norb, nelec, self.orbsym)
        if 'verbose' in kwargs:
            if isinstance(kwargs['verbose'], logger.Logger):
                log = kwargs['verbose']
            else:
                log = logger.Logger(self.stdout, kwargs['verbose'])
            log.debug('Guessing CI wfn symmetry = %s', wfnsym)
        else:
            logger.debug(self, 'Guessing CI wfn symmetry = %s', wfnsym)
        return wfnsym

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is not None:
            self.orbsym, orbsym_bak = orbsym, self.orbsym
        if wfnsym is not None:
            self.wfnsym, wfnsym_bak = wfnsym, self.wfnsym
        else:
            wfnsym_bak = None
        if self.verbose >= logger.WARN:
            self.check_sanity()

        wfnsym = self.guess_wfnsym(norb, nelec, ci0, self.wfnsym, **kwargs)
        e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                                       tol, lindep, max_cycle, max_space, nroots,
                                       davidson_only, pspace_size, **kwargs)
        if self.wfnsym is not None:
            # should I remove the non-symmetric contributions in each
            # call of contract_2e?
            if nroots > 1:
                c = [addons.symmetrize_wfn(ci, norb, nelec, self.orbsym, wfnsym)
                     for ci in c]
            else:
                c = addons.symmetrize_wfn(c, norb, nelec, self.orbsym, wfnsym)
        if orbsym is not None:
            self.orbsym = orbsym_bak
        if wfnsym_bak is not None:
            self.wfnsym = wfnsym_bak
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

    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))

    link_index = cistring.gen_linkstr_index(range(norb), nelec//2+1)
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
    na = cistring.num_strings(norb, nelec//2+1)
    nb = cistring.num_strings(norb, nelec//2)
    fcivec = numpy.random.random((na,nb))
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
    cis = FCISolver(mol)
    cis.orbsym = orbsym
    ci1 = cis.contract_2e(eri, fcivec, norb, nelec)
    ci1ref = direct_spin1.contract_2e(eri, fcivec, norb, nelec)
    print(numpy.allclose(ci1ref, ci1))
