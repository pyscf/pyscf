#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
import numpy
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

librdm = cistring.libfci

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
#       = <ib|s-|ka><ja|s+|lb>Gamma_{ibka,jalb}
# <CI|S-*S+|CI> = nelecb + <ib|s-|ka><ja|s+|lb>Gamma_{ibka,jalb}
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
def spin_square_general(dm1a, dm1b, dm2aa, dm2ab, dm2bb, mo_coeff, ovlp=1):
    r'''General spin square operator.

    ... math::

        <CI|S_+*S_-|CI> &= n_\alpha + \delta_{ik}\delta_{jl}Gamma_{i\alpha k\beta ,j\beta l\alpha } \\
        <CI|S_-*S_+|CI> &= n_\beta + \delta_{ik}\delta_{jl}Gamma_{i\beta k\alpha ,j\alpha l\beta } \\
        <CI|S_z*S_z|CI> &= \delta_{ik}\delta_{jl}(Gamma_{i\alpha k\alpha ,j\alpha l\alpha }
                         - Gamma_{i\alpha k\alpha ,j\beta l\beta }
                         - Gamma_{i\beta k\beta ,j\alpha l\alpha}
                         + Gamma_{i\beta k\beta ,j\beta l\beta})
                         + (n_\alpha+n_\beta)/4

    Given the overlap betwen non-degenerate alpha and beta orbitals, this
    function can compute the expectation value spin square operator for
    UHF-FCI wavefunction
    '''

    if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
        mo_coeff = (mo_coeff, mo_coeff)

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
    ssz = (numpy.einsum('ijkl,ij,kl->', dm2aa, ovlpaa, ovlpaa) -
           numpy.einsum('ijkl,ij,kl->', dm2ab, ovlpaa, ovlpbb) +
           numpy.einsum('ijkl,ij,kl->', dm2bb, ovlpbb, ovlpbb) -
           numpy.einsum('ijkl,ij,kl->', dm2ab, ovlpaa, ovlpbb)) * .25
    ssz += (numpy.einsum('ji,ij->', dm1a, ovlpaa) +
            numpy.einsum('ji,ij->', dm1b, ovlpbb)) *.25

    dm2abba = -dm2ab.transpose(0,3,2,1)  # alpha^+ beta^+ alpha beta
    dm2baab = -dm2ab.transpose(2,1,0,3)  # beta^+ alpha^+ beta alpha
    ssxy =(numpy.einsum('ijkl,ij,kl->', dm2baab, ovlpba, ovlpab) +
           numpy.einsum('ijkl,ij,kl->', dm2abba, ovlpab, ovlpba) +
           numpy.einsum('ji,ij->', dm1a, ovlpaa) +
           numpy.einsum('ji,ij->', dm1b, ovlpbb)) * .5
    ss = ssxy + ssz

    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

@lib.with_doc(spin_square_general.__doc__)
def spin_square(fcivec, norb, nelec, mo_coeff=None, ovlp=1, fcisolver=None):
    if mo_coeff is None:
        mo_coeff = (numpy.eye(norb),) * 2

    if fcisolver is None:
        from pyscf.fci import direct_spin1
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
            direct_spin1.make_rdm12s(fcivec, norb, nelec)
    else:
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
            fcisolver.make_rdm12s(fcivec, norb, nelec)

    return spin_square_general(dm1a, dm1b, dm2aa, dm2ab, dm2bb, mo_coeff, ovlp)

def spin_square0(fcivec, norb, nelec):
    '''Spin square for RHF-FCI CI wfn only (obtained from spin-degenerated
    Hamiltonian)'''
    ci1 = contract_ss(fcivec, norb, nelec)
    ss = numpy.einsum('ij,ij->', fcivec.reshape(ci1.shape), ci1)
    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

def local_spin(fcivec, norb, nelec, mo_coeff=None, ovlp=1, aolst=[]):
    r'''Local spin expectation value, which is defined as

    <CI|(local S^2)|CI>

    The local S^2 operator only couples the orbitals specified in aolst. The
    cross term which involves the interaction between the local part (in aolst)
    and non-local part (not in aolst) is not included. As a result, the value
    of local_spin is not additive. In other words, if local_spin is computed
    twice with the complementary aolst in the two runs, the summation does not
    equal to the S^2 of the entire system.

    For a complete list of AOs, the value of local_spin is equivalent to <CI|S^2|CI>
    '''
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
            aolst = numpy.arange(norb)
        s = numpy.zeros((norb,norb))
        s[aolst,aolst] = 1
    return spin_square(fcivec, norb, nelec, mo_coeff, s)

# for S+*S-
# dm(pq,rs) * [p(beta)^+ q(alpha) r(alpha)^+ s(beta)]
# size of intermediate determinants (norb,neleca+1;norb,nelecb-1)
def make_rdm2_baab(fcivec, norb, nelec):
    from pyscf.fci import direct_spin1
    dm2aa, dm2ab, dm2bb = direct_spin1.make_rdm12s(fcivec, norb, nelec)[1]
    dm2baab = -dm2ab.transpose(2,1,0,3)
    return dm2baab

# for S-*S+
# dm(pq,rs) * [q(alpha)^+ p(beta) s(beta)^+ r(alpha)]
# size of intermediate determinants (norb,neleca-1;norb,nelecb+1)
def make_rdm2_abba(fcivec, norb, nelec):
    from pyscf.fci import direct_spin1
    dm2aa, dm2ab, dm2bb = direct_spin1.make_rdm12s(fcivec, norb, nelec)[1]
    dm2abba = -dm2ab.transpose(0,3,2,1)
    return dm2abba


def contract_ss(fcivec, norb, nelec):
    '''Contract spin square operator with FCI wavefunction :math:`S^2 |CI>`
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)

    def gen_map(fstr_index, nelec, des=True):
        a_index = fstr_index(range(norb), nelec)
        amap = numpy.zeros((a_index.shape[0],norb,2), dtype=numpy.int32)
        if des:
            for k, tab in enumerate(a_index):
                amap[k,tab[:,1]] = tab[:,2:]
        else:
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
        acre = gen_map(cistring.gen_cre_str_index, neleca, False)
    else:
        acre = None

    if nelecb < norb:
        bcre = gen_map(cistring.gen_cre_str_index, nelecb, False)
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
            addra = aindex[maska,i,0]
            addrb = bindex[maskb,i,0]
            citmp = lib.take_2d(fcivec, maska, maskb)
            citmp *= signa[maska].reshape(-1,1)
            citmp *= signb[maskb]
            #: t1[addra.reshape(-1,1),addrb] += citmp
            lib.takebak_2d(t1, citmp, addra, addrb)
        for i in range(norb):
            signa = aindex[:,i,1]
            signb = bindex[:,i,1]
            maska = numpy.where(signa!=0)[0]
            maskb = numpy.where(signb!=0)[0]
            addra = aindex[maska,i,0]
            addrb = bindex[maskb,i,0]
            citmp = lib.take_2d(t1, addra, addrb)
            citmp *= signa[maska].reshape(-1,1)
            citmp *= signb[maskb]
            #: ci1[maska.reshape(-1,1), maskb] += citmp
            lib.takebak_2d(ci1, citmp, maska, maskb)

    ci1 = numpy.zeros((na,nb))
    trans(ci1, ades, bcre, neleca-1, nelecb+1) # S+*S-
    trans(ci1, acre, bdes, neleca+1, nelecb-1) # S-*S+
    ci1 *= .5
    ci1 += (neleca-nelecb)**2*.25*fcivec
    return ci1


if __name__ == '__main__':
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
    print('local spin for H1..H5 = 0.999', ss[0])
    ci1 = numpy.zeros((4,4))
    ci1[0,0] = 1
    print(spin_square (ci1, 4, (3,1)))
    print(spin_square0(ci1, 4, (3,1)))

    print(numpy.einsum('ij,ij->', ci1, contract_ss(ci1, 4, (3,1))),
          spin_square(ci1, 4, (3,1))[0])
