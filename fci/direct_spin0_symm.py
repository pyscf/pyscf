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
    assert(fcivec.flags.c_contiguous)
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

    wfnsym = direct_spin1_symm._id_wfnsym(cis, norb, nelec, cis.wfnsym)
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

    init_strs = []
    iroot = 0
    for addr in numpy.argsort(hdiag):
        addra = addr // nb
        addrb = addr % nb
        if airreps[addra] ^ birreps[addrb] == wfnsym:
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
    return ci0


class FCISolver(direct_spin0.FCISolver):
    def __init__(self, mol=None, **kwargs):
        self.orbsym = []
        self.wfnsym = None
        direct_spin0.FCISolver.__init__(self, mol, **kwargs)
        self.davidson_only = True
        self.pspace_size = 0  # Improper pspace size may break symmetry

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

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.wfnsym)
        return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

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
        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.wfnsym)
        if 'verbose' in kwargs:
            if isinstance(kwargs['verbose'], logger.Logger):
                log = kwargs['verbose']
            else:
                log = logger.Logger(self.stdout, kwargs['verbose'])
            log.debug('total symmetry = %s', wfnsym)
        else:
            logger.debug(self, 'total symmetry = %s', wfnsym)
        e, c = direct_spin0.kernel_ms0(self, h1e, eri, norb, nelec, ci0,
                                       tol, lindep, max_cycle, max_space, nroots,
                                       davidson_only, pspace_size, **kwargs)
        if self.wfnsym is not None:
            if self.nroots > 1:
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
