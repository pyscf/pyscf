#!/usr/bin/env python

import os
import ctypes
import _ctypes
import numpy
import pyscf.lib
import cistring
import direct_spin1
import rdm

_loaderpath = os.path.dirname(pyscf.lib.__file__)
librdm = numpy.ctypeslib.load_library('libmcscf', _loaderpath)

######################################################
# Spin squared operator
######################################################
# S^2 = (S+ * S- + S- * S+)/2 + Sz * Sz
# S+ = \sum_i S_i+ ~ effective for all beta occupied orbitals.
# S- = \sum_i S_i- ~ effective for all alpha occupied orbitals.
# There are two cases for S+*S-
# 1) same electron \sum_i s_i+*s_i-, <CI|s_i+*s_i-|CI> gives
#       <p|s+s-|q> \gammalpha_qp = trace(\gammalpha) = neleca
# 2) different electrons for \sum s_i+*s_j- (i\neq j, n*(n-1) terms)
# As a two-particle operator S+*S-
#       = <ij|s+s-|kl>Gamma_{ki,lj} = <iajb|s+s-|kbla>Gamma_{kbia,lajb}
#       = <ia|s+|kb><jb|s-|la>Gamma_{kbia,lajb}
# <CI|S+*S-|CI> = neleca + <ia|s+|kb><jb|s-|la>Gamma_{kbia,lajb}
#
# There are two cases for S-*S+
# 1) same electron \sum_i s_i-*s_i+
#       <p|s+s-|q> \gammabeta_qp = trace(\gammabeta) = nelecb
# 2) different electrons
#       = <ij|s-s+|kl>Gamma_{ki,lj} = <ibja|s-s+|kalb>Gamma_{kaib,lbja}
#       = <ib|s+|ka><ja|s-|lb>Gamma_{kaib,lbja}
# <CI|S-*S+|CI> = nelecb + <ib|s+|ka><ja|s-|lb>Gamma_{kaib,lbja}
#
# Sz*Sz = Msz^2 = (neleca-nelecb)^2
# 1) same electron
#       <p|ss|q>\gamma_qp = (neleca+nelecb)/4
# 2) different electrons
#       <ij|2s1s2|kl>Gamma_{ki,lj}/2
#       =(<ia|ka><ja|la>Gamma_{kaia,laja} - <ia|ka><jb|lb>Gamma_{kaia,lbjb}
#       - <ib|kb><ja|la>Gamma_{kbib,laja} + <ib|kb><jb|lb>Gamma_{kbib,lbjb})/4

def spin_square(ci, norb, nelec):
# <CI|S+*S-|CI> = neleca + \delta_{ik}\delta_{jl}Gamma_{kbia,lajb}
# <CI|S-*S+|CI> = nelecb + \delta_{ik}\delta_{jl}Gamma_{kaib,lbja}
# <CI|Sz*Sz|CI> = \delta_{ik}\delta_{jl}(Gamma_{kaia,laja} - Gamma_{kaia,lbjb}
#                                       -Gamma_{kbib,laja} + Gamma_{kbib,lbjb})
#               + (neleca+nelecb)/4
    if isinstance(nelec, int):
        neleca = nelecb = nelec / 2
    else:
        neleca, nelecb = nelec
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
            direct_spin1.make_rdm12s(ci, norb, nelec)
    ssz =(_bi_trace(dm2aa) - _bi_trace(dm2ab) \
        + _bi_trace(dm2bb) - _bi_trace(dm2ab)) * .25 \
        + (neleca + nelecb) *.25

    dm2baab = _make_rdm2_baab(ci, norb, nelec)
    dm2abba = _make_rdm2_abba(ci, norb, nelec)
    dm2baab = rdm.reorder_rdm(dm1a, dm2baab, inplace=True)[1]
    dm2abba = rdm.reorder_rdm(dm1b, dm2abba, inplace=True)[1]
    ssxy = (_bi_trace(dm2baab) + _bi_trace(dm2abba) + neleca + nelecb) * .5
    ss = ssxy + ssz

    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

# sum_ij A(i,i,j,j)
def _bi_trace(a):
    atmp = numpy.einsum('iipq->pq', a)
    return numpy.einsum('ii', atmp)

# for S+*S-
# dm(pq,rs) * [q(beta)^+ p(alpha) s(alpha)^+ r(beta)]
# size of intermediate determinants (norb,neleca+1;norb,nelecb-1)
def _make_rdm2_abba(ci, norb, nelec):
    if isinstance(nelec, int):
        neleca = nelecb = nelec / 2
    else:
        neleca, nelecb = nelec
    if neleca == norb: # no intermediate determinants
        return numpy.zeros((norb,norb,norb,norb))
    ades_index = cistring.gen_des_str_index(range(norb), neleca+1)
    bcre_index = cistring.gen_cre_str_index(range(norb), nelecb-1)
    instra = cistring.num_strings(norb, neleca+1)
    nb = cistring.num_strings(norb, nelecb)
    dm1 = numpy.empty((norb,norb))
    dm2 = numpy.empty((norb,norb,norb,norb))
    fn = _ctypes.dlsym(librdm._handle, 'FCIdm2_abba_kern')
    librdm.FCIspindm12_drv(ctypes.c_void_p(fn),
                           dm1.ctypes.data_as(ctypes.c_void_p),
                           dm2.ctypes.data_as(ctypes.c_void_p),
                           ci.ctypes.data_as(ctypes.c_void_p),
                           ci.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(norb),
                           ctypes.c_int(instra), ctypes.c_int(nb),
                           ctypes.c_int(neleca), ctypes.c_int(nelecb),
                           ades_index.ctypes.data_as(ctypes.c_void_p),
                           bcre_index.ctypes.data_as(ctypes.c_void_p))
    return dm2.transpose(1,0,2,3).copy('C')
def make_rdm2_abba(ci, norb, nelec):
    dm2 = _make_rdm2_abba(ci, norb, nelec)
    dm1b = rdm.make_rdm1_spin1('FCImake_rdm1b', ci, ci, norb, nelec)
    dm1b, dm2 = rdm.reorder_rdm(dm1b, dm2, inplace=True)
    return dm2

# for S-*S+
# dm(pq,rs) * [q(alpha)^+ p(beta) s(beta)^+ r(alpha)]
# size of intermediate determinants (norb,neleca-1;norb,nelecb+1)
def _make_rdm2_baab(ci, norb, nelec):
    if isinstance(nelec, int):
        neleca = nelecb = nelec / 2
    else:
        neleca, nelecb = nelec
    if nelecb == norb: # no intermediate determinants
        return numpy.zeros((norb,norb,norb,norb))
    acre_index = cistring.gen_cre_str_index(range(norb), neleca-1)
    bdes_index = cistring.gen_des_str_index(range(norb), nelecb+1)
    instra = cistring.num_strings(norb, neleca-1)
    nb = cistring.num_strings(norb, nelecb)
    dm1 = numpy.empty((norb,norb))
    dm2 = numpy.empty((norb,norb,norb,norb))
    fn = _ctypes.dlsym(librdm._handle, 'FCIdm2_baab_kern')
    librdm.FCIspindm12_drv(ctypes.c_void_p(fn),
                           dm1.ctypes.data_as(ctypes.c_void_p),
                           dm2.ctypes.data_as(ctypes.c_void_p),
                           ci.ctypes.data_as(ctypes.c_void_p),
                           ci.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(norb),
                           ctypes.c_int(instra), ctypes.c_int(nb),
                           ctypes.c_int(neleca), ctypes.c_int(nelecb),
                           acre_index.ctypes.data_as(ctypes.c_void_p),
                           bdes_index.ctypes.data_as(ctypes.c_void_p))
    return dm2.transpose(1,0,2,3).copy('C')
def make_rdm2_baab(ci, norb, nelec):
    dm2 = _make_rdm2_baab(ci, norb, nelec)
    dm1a = rdm.make_rdm1_spin1('FCImake_rdm1a', ci, ci, norb, nelec)
    dm1a, dm2 = rdm.reorder_rdm(dm1a, dm2, inplace=True)
    return dm2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    import fci

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
        ['H', ( 0.,-1.    ,-2.   )],
        ['H', ( 1.,-1.5   , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    cis = fci.solver(mol)
    cis.verbose = 5
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    e, ci0 = cis.kernel(h1e, eri, norb, nelec)
    ss = spin_square(ci0, norb, nelec)
    print(ss)
    ci1 = numpy.zeros((4,4))
    ci1[0,0] = 1
    print(spin_square(ci1, 4, (3,1)))
