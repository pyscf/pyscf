#!/usr/bin/env python

import ctypes
import _ctypes
from functools import reduce
import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import rdm

librdm = pyscf.lib.load_library('libfci')

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
#       = <ij|s+s-|kl>Gamma_{ik,jl} = <iajb|s+s-|kbla>Gamma_{iakb,jbla}
#       = <ia|s+|kb><jb|s-|la>Gamma_{iakb,jbla}
# <CI|S+*S-|CI> = neleca + <ia|s+|kb><jb|s-|la>Gamma_{iakb,jbla}
#
# There are two cases for S-*S+
# 1) same electron \sum_i s_i-*s_i+
#       <p|s+s-|q> \gammabeta_qp = trace(\gammabeta) = nelecb
# 2) different electrons
#       = <ij|s-s+|kl>Gamma_{ik,jl} = <ibja|s-s+|kalb>Gamma_{ibka,jalb}
#       = <ib|s+|ka><ja|s-|lb>Gamma_{ibka,jalb}
# <CI|S-*S+|CI> = nelecb + <ib|s+|ka><ja|s-|lb>Gamma_{ibka,jalb}
#
# Sz*Sz = Msz^2 = (neleca-nelecb)^2
# 1) same electron
#       <p|ss|q>\gamma_qp = <p|q>\gamma_qp = (neleca+nelecb)/4
# 2) different electrons
#       <ij|2s1s2|kl>Gamma_{ik,jl}/2
#       =(<ia|ka><ja|la>Gamma_{iaka,jala} - <ia|ka><jb|lb>Gamma_{iaka,jblb}
#       - <ib|kb><ja|la>Gamma_{ibkb,jala} + <ib|kb><jb|lb>Gamma_{ibkb,jblb})/4

# set aolst for local spin expectation value, which is defined as
#       <CI|ao><ao|S^2|CI>
# For a complete list of AOs, I = \sum |ao><ao|, it becomes <CI|S^2|CI>
def spin_square(fcivec, norb, nelec, mo_coeff=None, ovlp=1):
    '''General spin square operator'''
# <CI|S+*S-|CI> = neleca + \delta_{ik}\delta_{jl}Gamma_{iakb,jbla}
# <CI|S-*S+|CI> = nelecb + \delta_{ik}\delta_{jl}Gamma_{ibka,jalb}
# <CI|Sz*Sz|CI> = \delta_{ik}\delta_{jl}(Gamma_{iaka,jala} - Gamma_{iaka,jblb}
#                                       -Gamma_{ibkb,jala} + Gamma_{ibkb,jblb})
#               + (neleca+nelecb)/4
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec

    if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
        mo_coeff = (mo_coeff, mo_coeff)
    elif mo_coeff is None:
        mo_coeff = (numpy.eye(norb),) * 2

# projected overlap matrix elements for partial trace
    if isinstance(ovlp, numpy.ndarray):
        ovlpaa = reduce(numpy.dot, (mo_coeff[0].T, ovlp, mo_coeff[0]))
        ovlpbb = reduce(numpy.dot, (mo_coeff[1].T, ovlp, mo_coeff[1]))
        ovlpab = reduce(numpy.dot, (mo_coeff[0].T, ovlp, mo_coeff[1]))
        ovlpba = reduce(numpy.dot, (mo_coeff[1].T, ovlp, mo_coeff[0]))
    else:
        ovlpaa = numpy.dot(mo_coeff[0].T, mo_coeff[0])
        ovlpbb = numpy.dot(mo_coeff[1].T, mo_coeff[1])
        ovlpab = numpy.dot(mo_coeff[0].T, mo_coeff[1])
        ovlpba = numpy.dot(mo_coeff[1].T, mo_coeff[0])

    # if ovlp=1, ssz = (neleca-nelecb)**2 * .25
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
            direct_spin1.make_rdm12s(fcivec, norb, nelec)
    ssz =(_bi_trace(dm2aa, ovlpaa, ovlpaa)
        - _bi_trace(dm2ab, ovlpaa, ovlpbb)
        + _bi_trace(dm2bb, ovlpbb, ovlpbb)
        - _bi_trace(dm2ab, ovlpaa, ovlpbb)) * .25 \
        +(_trace(dm1a, ovlpaa)
        + _trace(dm1b, ovlpbb)) *.25

    dm2baab = _make_rdm2_baab(fcivec, norb, nelec)
    dm2abba = _make_rdm2_abba(fcivec, norb, nelec)
    dm2baab = rdm.reorder_rdm(dm1b, dm2baab, inplace=True)[1]
    dm2abba = rdm.reorder_rdm(dm1a, dm2abba, inplace=True)[1]
    ssxy =(_bi_trace(dm2abba, ovlpab, ovlpba)
         + _bi_trace(dm2baab, ovlpba, ovlpab) \
         + _trace(dm1a, ovlpaa)
         + _trace(dm1b, ovlpbb)) * .5
    ss = ssxy + ssz

    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

def spin_square0(fcivec, norb, nelec):
    '''Spin square for CI wfn of spin-degenerated Hamiltonian'''
    ss = numpy.einsum('ij,ij->', fcivec, contract_ss(fcivec, norb, nelec))
    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip


def _trace(dm1, ovlp):
    return numpy.einsum('ij,ji->', dm1, ovlp)

def _bi_trace(dm2, ovlp1, ovlp2):
    return numpy.einsum('jilk,ij,kl->', dm2, ovlp1, ovlp2)

def local_spin(fcivec, norb, nelec, mo_coeff=None, ovlp=1, aolst=[]):
    if isinstance(ovlp, numpy.ndarray):
        nao = ovlp.shape[0]
        if len(aolst) == 0:
            lstnot = []
        else:
            lstnot = [i for i in range(nao) if i not in aolst]
        s = ovlp.copy()
        s[lstnot] = 0
        s[:,lstnot] = 0
    else:
        if len(aolst) == 0:
            aolst = range(norb)
        s = numpy.zeros((norb,norb))
        s[aolst,aolst] = 1
    return spin_square(fcivec, norb, nelec, mo_coeff, s)

# for S+*S-
# dm(pq,rs) * [p(beta)^+ q(alpha) r(alpha)^+ s(beta)]
# size of intermediate determinants (norb,neleca+1;norb,nelecb-1)
def _make_rdm2_baab(fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    if neleca == norb or nelecb == 0: # no intermediate determinants
        return numpy.zeros((norb,norb,norb,norb))
    ades_index = cistring.gen_des_str_index(range(norb), neleca+1)
    bcre_index = cistring.gen_cre_str_index(range(norb), nelecb-1)
    instra = cistring.num_strings(norb, neleca+1)
    nb = cistring.num_strings(norb, nelecb)
    dm1 = numpy.empty((norb,norb))
    dm2 = numpy.empty((norb,norb,norb,norb))
    fn = _ctypes.dlsym(librdm._handle, 'FCIdm2_baab_kern')
    librdm.FCIspindm12_drv(ctypes.c_void_p(fn),
                           dm1.ctypes.data_as(ctypes.c_void_p),
                           dm2.ctypes.data_as(ctypes.c_void_p),
                           fcivec.ctypes.data_as(ctypes.c_void_p),
                           fcivec.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(norb),
                           ctypes.c_int(instra), ctypes.c_int(nb),
                           ctypes.c_int(neleca), ctypes.c_int(nelecb),
                           ades_index.ctypes.data_as(ctypes.c_void_p),
                           bcre_index.ctypes.data_as(ctypes.c_void_p))
    return dm2
def make_rdm2_baab(fcivec, norb, nelec):
    dm2 = _make_rdm2_baab(fcivec, norb, nelec)
    dm1b = rdm.make_rdm1_spin1('FCImake_rdm1b', fcivec, fcivec, norb, nelec)
    dm1b, dm2 = rdm.reorder_rdm(dm1b, dm2, inplace=True)
    return dm2

# for S-*S+
# dm(pq,rs) * [q(alpha)^+ p(beta) s(beta)^+ r(alpha)]
# size of intermediate determinants (norb,neleca-1;norb,nelecb+1)
def _make_rdm2_abba(fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    if nelecb == norb or neleca == 0: # no intermediate determinants
        return numpy.zeros((norb,norb,norb,norb))
    acre_index = cistring.gen_cre_str_index(range(norb), neleca-1)
    bdes_index = cistring.gen_des_str_index(range(norb), nelecb+1)
    instra = cistring.num_strings(norb, neleca-1)
    nb = cistring.num_strings(norb, nelecb)
    dm1 = numpy.empty((norb,norb))
    dm2 = numpy.empty((norb,norb,norb,norb))
    fn = _ctypes.dlsym(librdm._handle, 'FCIdm2_abba_kern')
    librdm.FCIspindm12_drv(ctypes.c_void_p(fn),
                           dm1.ctypes.data_as(ctypes.c_void_p),
                           dm2.ctypes.data_as(ctypes.c_void_p),
                           fcivec.ctypes.data_as(ctypes.c_void_p),
                           fcivec.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(norb),
                           ctypes.c_int(instra), ctypes.c_int(nb),
                           ctypes.c_int(neleca), ctypes.c_int(nelecb),
                           acre_index.ctypes.data_as(ctypes.c_void_p),
                           bdes_index.ctypes.data_as(ctypes.c_void_p))
    return dm2
def make_rdm2_abba(fcivec, norb, nelec):
    dm2 = _make_rdm2_abba(fcivec, norb, nelec)
    dm1a = rdm.make_rdm1_spin1('FCImake_rdm1a', fcivec, fcivec, norb, nelec)
    dm1a, dm2 = rdm.reorder_rdm(dm1a, dm2, inplace=True)
    return dm2


def contract_ss(fcivec, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelecb = nelec // 2
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)

    def gen_map(fstr_index, nelec):
        a_index = fstr_index(range(norb), nelec)
        amap = numpy.zeros((a_index.shape[0],norb,2), dtype=numpy.int32)
        for k, tab in enumerate(a_index):
            amap[k,tab[:,0]] = tab[:,2:]
        return amap

    if neleca > 0:
        ades = gen_map(cistring.gen_des_str_index, neleca)
    else:
        ades = None

    if nelecb > 0:
        bdes = gen_map(cistring.gen_des_str_index, nelecb)
    else:
        bdes = None

    if neleca < norb:
        acre = gen_map(cistring.gen_cre_str_index, neleca)
    else:
        acre = None

    if nelecb < norb:
        bcre = gen_map(cistring.gen_cre_str_index, nelecb)
    else:
        bcre = None

    def trans(ci1, aindex, bindex, nea, neb):
        if aindex is None or bindex is None:
            return None

        t1 = numpy.zeros((cistring.num_strings(norb,nea),
                          cistring.num_strings(norb,neb)))
        for i in range(norb):
            signa = aindex[:,i,1]
            signb = bindex[:,i,1]
            maska = numpy.where(signa!=0)[0]
            maskb = numpy.where(signb!=0)[0]
            ida = aindex[maska,i,0]
            idb = bindex[maskb,i,0]
            citmp = pyscf.lib.take_2d(fcivec, maska, maskb)
            citmp = numpy.einsum('i,j,ij->ij', signa[maska], signb[maskb], citmp)
            #: t1[ida.reshape(-1,1),idb] += citmp
            pyscf.lib.takebak_2d(t1, citmp, ida, idb)
        for i in range(norb):
            signa = aindex[:,i,1]
            signb = bindex[:,i,1]
            maska = numpy.where(signa!=0)[0]
            maskb = numpy.where(signb!=0)[0]
            ida = aindex[maska,i,0]
            idb = bindex[maskb,i,0]
            citmp = pyscf.lib.take_2d(t1, ida, idb)
            citmp = numpy.einsum('i,j,ij->ij', signa[maska], signb[maskb], citmp)
            #: ci1[maska.reshape(-1,1), maskb] += citmp
            pyscf.lib.takebak_2d(ci1, citmp, maska, maskb)

    ci1 = numpy.zeros((na,nb))
    trans(ci1, ades, bcre, neleca-1, nelecb+1) # S+*S-
    trans(ci1, acre, bdes, neleca+1, nelecb-1) # S-*S+
    ci1 *= .5
    ci1 += (neleca-nelecb)**2*.25*fcivec
    return ci1


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import fci

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
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
    ss = spin_square0(ci0, norb, nelec)
    print(ss)
    ss = local_spin(ci0, norb, nelec, m.mo_coeff, m.get_ovlp(), range(5))
    print('local spin for H1..H5 = 0.998988389', ss[0])
    ci1 = numpy.zeros((4,4))
    ci1[0,0] = 1
    print(spin_square (ci1, 4, (3,1)))
    print(spin_square0(ci1, 4, (3,1)))

    print(numpy.einsum('ij,ij->', ci1, contract_ss(ci1, 4, (3,1))),
          spin_square(ci1, 4, (3,1)))

    numpy.random.seed(1)
    n = cistring.num_strings(6,3)
    ci0 = numpy.random.random((n,n))
    print(numpy.einsum('ij,ij->', ci0,contract_ss(ci0, 6, 6)),
          spin_square(ci0, 6, 6))

    na = cistring.num_strings(6,4)
    nb = cistring.num_strings(6,2)
    ci0 = numpy.random.random((na,nb))
    print(numpy.einsum('ij,ij->', ci0,contract_ss(ci0, 6, (4,2))),
          spin_square(ci0, 6, (4,2)))


    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 0 ,  0    , 0.   )],
        ['H', ( 0 ,  0    , 8.   )],
    ]

    mol.basis = {'H': 'cc-pvdz'}
    mol.spin = 0
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    cis = fci.direct_spin0.FCISolver(mol)
    cis.verbose = 5
    norb = m.mo_coeff.shape[1]
    nelec = (mol.nelectron, 0)
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    e, ci0 = cis.kernel(h1e, eri, norb, nelec)
    ss = spin_square(ci0, norb, nelec, m.mo_coeff, m.get_ovlp())
    print('local spin for H1+H2 = 0', ss[0])
    ss = local_spin(ci0, norb, nelec, m.mo_coeff, m.get_ovlp(), range(5))
    print('local spin for H1 = 0.75', ss[0])
    ss = local_spin(ci0, norb, nelec, m.mo_coeff, m.get_ovlp(), range(5,10))
    print('local spin for H2 = 0.75', ss[0])

    ss = spin_square0(ci0, norb, nelec)
    print('tot spin for HH = 0', ss[0])

    numpy.random.seed(1)
    n = cistring.num_strings(10,5)
    ci0 = numpy.random.random((n,n))
    ss1 = numpy.einsum('ij,ij->', ci0, contract_ss(ci0, 10, 10))
    print(ss1, spin_square(ci0, 10, 10))
