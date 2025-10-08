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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Full CI solver for spin-free Hamiltonian.  This solver can be used to compute
doublet, triplet,...

The CI wfn are stored as a 2D array [alpha,beta], where each row corresponds
to an alpha string.  For each row (alpha string), there are
total-num-beta-strings of columns.  Each column corresponds to a beta string.

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
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import spin_op
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf.fci.addons import _unpack_nelec, civec_spinless_repr
from pyscf import __config__

libfci = cistring.libfci

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    '''Contract the 1-electron Hamiltonian with a FCI vector to get a new FCI
    vector.
    '''
    fcivec = numpy.asarray(fcivec, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    assert fcivec.dtype == f1e.dtype == numpy.float64
    f1e_tril = lib.pack_tril(f1e)
    ci1 = numpy.zeros_like(fcivec)
    libfci.FCIcontract_a_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    libfci.FCIcontract_b_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(FCIvector)

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    r'''Contract the 4-index tensor eri[pqrs] with a FCI vector

    .. math::

        |output\rangle = E_{pq} E_{rs} eri_{pq,rs} |CI\rangle \\

        E_{pq}E_{rs} = E_{pr,qs} + \delta_{qr} E_{ps} \\

        E_{pq} = p^+ q + \bar{p}^+ \bar{q}

        E_{pr,qs} = p^+ r^+ s q + \bar{p}^+ r^+ s \bar{q} + ...

    :math:`p,q,...` means spin-up orbitals and :math:`\bar{p}, \bar{q}` means
    spin-down orbitals.

    Note the input argument eri is NOT the 2e hamiltonian tensor. 2e hamiltonian is

    .. math::

        h2e &= (pq|rs) E_{pr,qs} \\
            &= (pq|rs) (E_{pq}E_{rs} - \delta_{qr} E_{ps}) \\
            &= eri_{pq,rs} E_{pq}E_{rs} \\

    So the relation between eri and hamiltonian (the 2e-integral tensor) is

    .. math::

        eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)

    to restore the symmetry between pq and rs,

    .. math::

        eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]

    See also :func:`direct_spin1.absorb_h1e`
    '''
    fcivec = numpy.asarray(fcivec, order='C')
    eri = numpy.asarray(ao2mo.restore(4, eri, norb), order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    assert fcivec.dtype == eri.dtype == numpy.float64
    ci1 = numpy.empty_like(fcivec)

    libfci.FCIcontract_2e_spin1(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(FCIvector)

def make_hdiag(h1e, eri, norb, nelec, compress=False):
    '''Diagonal Hamiltonian for Davidson preconditioner

    Kwargs:
        compress (bool) : whether to remove symmetry forbidden elements
    '''
    if not (h1e.dtype == eri.dtype == numpy.float64):
        raise NotImplementedError('Complex Hamiltonian')

    neleca, nelecb = _unpack_nelec(nelec)
    h1e = numpy.asarray(h1e, order='C')
    eri = numpy.asarray(ao2mo.restore(1, eri, norb), order='C')
    occslsta = occslstb = cistring.gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslstb = cistring.gen_occslst(range(norb), nelecb)
    na = len(occslsta)
    nb = len(occslstb)

    hdiag = numpy.empty(na*nb)
    jdiag = numpy.asarray(numpy.einsum('iijj->ij',eri), order='C')
    kdiag = numpy.asarray(numpy.einsum('ijji->ij',eri), order='C')
    c_h1e = h1e.ctypes.data_as(ctypes.c_void_p)
    c_jdiag = jdiag.ctypes.data_as(ctypes.c_void_p)
    c_kdiag = kdiag.ctypes.data_as(ctypes.c_void_p)
    libfci.FCImake_hdiag_uhf(hdiag.ctypes.data_as(ctypes.c_void_p),
                             c_h1e, c_h1e, c_jdiag, c_jdiag, c_jdiag, c_kdiag, c_kdiag,
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(neleca), ctypes.c_int(nelecb),
                             occslsta.ctypes.data_as(ctypes.c_void_p),
                             occslstb.ctypes.data_as(ctypes.c_void_p))
    return hdiag

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if h1e.dtype == numpy.complex128 or eri.dtype == numpy.complex128:
        raise NotImplementedError('Complex Hamiltonian')

    if not isinstance(nelec, (int, numpy.number)):
        nelec = sum(nelec)
    h2e = ao2mo.restore(1, eri.copy(), norb)
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return ao2mo.restore(4, h2e, norb) * fac

def pspace(h1e, eri, norb, nelec, hdiag=None, np=400):
    '''pspace Hamiltonian to improve Davidson preconditioner. See, CPL, 169, 463
    '''
    if norb >= 64:
        raise NotImplementedError('norb >= 64')

    if h1e.dtype == numpy.complex128 or eri.dtype == numpy.complex128:
        raise NotImplementedError('Complex Hamiltonian')

    neleca, nelecb = _unpack_nelec(nelec)
    h1e = numpy.ascontiguousarray(h1e)
    eri = ao2mo.restore(1, eri, norb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    if hdiag is None:
        hdiag = make_hdiag(h1e, eri, norb, nelec, compress=False)
    assert hdiag.size == na * nb
    if hdiag.size <= np:
        addr = numpy.arange(hdiag.size)
    else:
        try:
            addr = numpy.argpartition(hdiag, np-1)[:np].copy()
        except AttributeError:
            addr = numpy.argsort(hdiag)[:np].copy()
    addra, addrb = divmod(addr, nb)
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)
    np = len(addr)
    h0 = numpy.zeros((np,np))
    libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                            h1e.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p),
                            stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(np))

    HERMITIAN_THRESHOLD = 1e-10
    if (abs(h1e - h1e.T).max() < HERMITIAN_THRESHOLD and
        abs(eri - eri.transpose(1,0,3,2)).max() < HERMITIAN_THRESHOLD):
        # symmetric Hamiltonian
        h0 = lib.hermi_triu(h0)
    else:
        # Fill the upper triangular part
        h0 = numpy.asarray(h0, order='F')
        h1e = numpy.asarray(h1e.T, order='C')
        eri = numpy.asarray(eri.transpose(1,0,3,2), order='C')
        libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                                h1e.ctypes.data_as(ctypes.c_void_p),
                                eri.ctypes.data_as(ctypes.c_void_p),
                                stra.ctypes.data_as(ctypes.c_void_p),
                                strb.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(np))

    idx = numpy.arange(np)
    h0[idx,idx] = hdiag[addr]
    return addr, h0

# be careful with single determinant initial guess. It may diverge the
# preconditioner when the eigvalue of first davidson iter equals to hdiag
def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    return _kfactory(FCISolver, h1e, eri, norb, nelec, ci0, level_shift,
                     tol, lindep, max_cycle, max_space, nroots,
                     davidson_only, pspace_size, ecore=ecore, **kwargs)
def _kfactory(Solver, h1e, eri, norb, nelec, ci0=None, level_shift=1e-3,
              tol=1e-10, lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
              davidson_only=False, pspace_size=400, ecore=0, **kwargs):
    cis = Solver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.max_space = max_space
    cis.nroots = nroots
    cis.davidson_only = davidson_only
    cis.pspace_size = pspace_size

    unknown = {}
    for k in kwargs:
        if not hasattr(cis, k):
            unknown[k] = kwargs[k]
        setattr(cis, k, kwargs[k])
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))
    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, ecore=ecore, **unknown)
    return e, c

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    '''Compute the FCI electronic energy for given Hamiltonian and FCI vector.
    '''
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))


def make_rdm1s(fcivec, norb, nelec, link_index=None):
    r'''Spin separated 1-particle density matrices.
    The return values include two density matrices: (alpha,alpha), (beta,beta)

    dm1[p,q] = <q^\dagger p>

    The convention is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        link_index = (link_indexa, link_indexb)
    rdm1a = rdm.make_rdm1_spin1('FCImake_rdm1a', fcivec, fcivec,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCImake_rdm1b', fcivec, fcivec,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

def make_rdm1(fcivec, norb, nelec, link_index=None):
    r'''Spin-traced one-particle density matrix

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention is based on McWeeney's book, Eq (5.4.20)
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, link_index)
    return rdm1a + rdm1b

def make_rdm12s(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin separated 1- and 2-particle density matrices.
    The return values include two lists, a list of 1-particle density matrices
    and a list of 2-particle density matrices.  The density matrices are:
    (alpha,alpha), (beta,beta) for 1-particle density matrices;
    (alpha,alpha,alpha,alpha), (alpha,alpha,beta,beta),
    (beta,beta,beta,beta) for 2-particle density matrices.

    1pdm[p,q] = :math:`\langle q^\dagger p\rangle`;
    2pdm[p,q,r,s] = :math:`\langle p^\dagger r^\dagger s q\rangle`.

    Energy should be computed as
    E = einsum('pq,qp', h1, 1pdm) + 1/2 * einsum('pqrs,pqrs', eri, 2pdm)
    where h1[p,q] = <p|h|q> and eri[p,q,r,s] = (pq|rs)
    '''
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCIrdm12kern_a', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCIrdm12kern_b', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', fcivec, fcivec,
                                    norb, nelec, link_index, 0)
    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin traced 1- and 2-particle density matrices.

    1pdm[p,q] = :math:`\langle q_\alpha^\dagger p_\alpha \rangle +
                       \langle q_\beta^\dagger  p_\beta \rangle`;
    2pdm[p,q,r,s] = :math:`\langle p_\alpha^\dagger r_\alpha^\dagger s_\alpha q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\alpha^\dagger s_\alpha q_\beta\rangle +
                           \langle p_\alpha^\dagger r_\beta^\dagger  s_\beta  q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\beta^\dagger  s_\beta  q_\beta\rangle`.

    Energy should be computed as
    E = einsum('pq,qp', h1, 1pdm) + 1/2 * einsum('pqrs,pqrs', eri, 2pdm)
    where h1[p,q] = <p|h|q> and eri[p,q,r,s] = (pq|rs)
    '''
    #(dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
    #        make_rdm12s(fcivec, norb, nelec, link_index, reorder)
    #return dm1a+dm1b, dm2aa+dm2ab+dm2ab.transpose(2,3,0,1)+dm2bb
    dm1, dm2 = rdm.make_rdm12_spin1('FCIrdm12kern_sf', fcivec, fcivec,
                                    norb, nelec, link_index, 1)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2

def make_rdm123(fcivec, norb, nelec, link_index=None, reorder=True):
    '''Spin traced 1-, 2-, and 3-particle density matrices.'''
    dm1, dm2, dm3 = rdm.make_dm123('FCI3pdm_kern_sf', fcivec, fcivec, norb, nelec)
    if reorder:
        dm1, dm2, dm3 = rdm.reorder_dm123(dm1, dm2, dm3, inplace=True)
    return dm1, dm2, dm3

def make_rdm123s(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin separated 1-, 2-, and 3-particle density matrices.

    1pdm[p,q] = :math:`\langle q_\alpha^\dagger p_\alpha \rangle +
                       \langle q_\beta^\dagger  p_\beta \rangle`;
    2pdm[p,q,r,s] = :math:`\langle p_\alpha^\dagger r_\alpha^\dagger s_\alpha q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\alpha^\dagger s_\alpha q_\beta\rangle +
                           \langle p_\alpha^\dagger r_\beta^\dagger  s_\beta  q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\beta^\dagger  s_\beta  q_\beta\rangle`.
    '''
    if (not reorder):
        raise NotImplementedError('reorder=False not currently supported')
    ci_spinless = civec_spinless_repr([fcivec,], norb, [nelec,])
    rdm1, rdm2, rdm3 = make_rdm123(ci_spinless, norb*2, (nelec[0]+nelec[1],0))

    rdm1a = rdm1[:norb,:norb]
    rdm1b = rdm1[norb:,norb:]
    # assert np.allclose(rdm1a+rdm1b, rdm1)

    rdm2aa = rdm2[:norb,:norb,:norb,:norb]
    rdm2ab = rdm2[:norb,:norb,norb:,norb:]
    rdm2bb = rdm2[norb:,norb:,norb:,norb:]
    # assert np.allclose(rdm2aa+rdm2bb+rdm2ab+rdm2ab.transpose(2,3,0,1), rdm2)

    rdm3aaa = rdm3[:norb,:norb,:norb,:norb,:norb,:norb]
    rdm3aab = rdm3[:norb,:norb,:norb,:norb,norb:,norb:]
    rdm3abb = rdm3[:norb,:norb,norb:,norb:,norb:,norb:]
    rdm3bbb = rdm3[norb:,norb:,norb:,norb:,norb:,norb:]
    # assert np.allclose(rdm3aaa+rdm3bbb+rdm3aab+rdm3aab.transpose(0,1,4,5,2,3)+\
    # rdm3aab.transpose(4,5,0,1,2,3)+rdm3abb+rdm3abb.transpose(2,3,0,1,4,5)+rdm3abb.transpose(2,3,4,5,0,1), rdm3)
    return (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb), (rdm3aaa, rdm3aab, rdm3abb, rdm3bbb)

def make_rdm1234(fcivec, norb, nelec, link_index=None, reorder=True):
    '''Spin traced 1-, 2-, 3, 4-particle density matrices.'''
    dm1, dm2, dm3, dm4 = rdm.make_dm1234('FCI4pdm_kern_sf', fcivec, fcivec, norb, nelec)
    if reorder:
        dm1, dm2, dm3, dm4 = rdm.reorder_dm1234(dm1, dm2, dm3, dm4, inplace=True)
    return dm1, dm2, dm3, dm4

def make_rdm1234s(fcivec, norb, nelec, link_index=None, reorder=True):
    r'''Spin separated 1-, 2-, 3, 4-particle density matrices.

    1pdm[p,q] = :math:`\langle q_\alpha^\dagger p_\alpha \rangle +
                       \langle q_\beta^\dagger  p_\beta \rangle`;
    2pdm[p,q,r,s] = :math:`\langle p_\alpha^\dagger r_\alpha^\dagger s_\alpha q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\alpha^\dagger s_\alpha q_\beta\rangle +
                           \langle p_\alpha^\dagger r_\beta^\dagger  s_\beta  q_\alpha\rangle +
                           \langle p_\beta^\dagger  r_\beta^\dagger  s_\beta  q_\beta\rangle`.
    '''
    if (not reorder):
        raise NotImplementedError('reorder=False not currently supported')
    ci_spinless = civec_spinless_repr([fcivec,], norb, [nelec,])
    rdm1, rdm2, rdm3, rdm4 = make_rdm1234(ci_spinless, norb*2, (nelec[0]+nelec[1],0))

    # define slices
    alpha = slice(0, norb)
    beta = slice(norb, None)

    rdm1a = rdm1[alpha, alpha]
    rdm1b = rdm1[beta, beta]
    # assert np.allclose(rdm1a+rdm1b, rdm1)

    rdm2aa = rdm2[alpha, alpha, alpha, alpha]
    rdm2ab = rdm2[alpha, alpha, beta, beta]
    rdm2bb = rdm2[beta, beta, beta, beta]
    # assert np.allclose(rdm2aa+rdm2bb+rdm2ab+rdm2ab.transpose(2,3,0,1), rdm2)

    rdm3aaa = rdm3[alpha, alpha, alpha, alpha, alpha, alpha]
    rdm3aab = rdm3[alpha, alpha, alpha, alpha, beta, beta]
    rdm3abb = rdm3[alpha, alpha, beta, beta, beta, beta]
    rdm3bbb = rdm3[beta, beta, beta, beta, beta, beta]
    # assert np.allclose(rdm3aaa+rdm3bbb+rdm3aab+rdm3aab.transpose(0,1,4,5,2,3)+\
    # rdm3aab.transpose(4,5,0,1,2,3)+rdm3abb+rdm3abb.transpose(2,3,0,1,4,5)+rdm3abb.transpose(2,3,4,5,0,1), rdm3)

    rdm4aaaa = rdm4[alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha]
    rdm4aaab = rdm4[alpha, alpha, alpha, alpha, alpha, alpha, beta, beta]
    rdm4aabb = rdm4[alpha, alpha, alpha, alpha, beta, beta, beta, beta]
    rdm4abbb = rdm4[alpha, alpha, beta, beta, beta, beta, beta, beta]
    rdm4bbbb = rdm4[beta, beta, beta, beta, beta, beta, beta, beta]
    # assert numpy.allclose(
    #     rdm4aaaa
    #     + rdm4bbbb
    #     + rdm4aaab
    #     + rdm4aaab.transpose(0, 1, 2, 3, 6, 7, 4, 5)
    #     + rdm4aaab.transpose(0, 1, 6, 7, 2, 3, 4, 5)
    #     + rdm4aaab.transpose(6, 7, 0, 1, 2, 3, 4, 5)
    #     + rdm4aabb
    #     + rdm4aabb.transpose(0, 1, 4, 5, 2, 3, 6, 7)
    #     + rdm4aabb.transpose(0, 1, 4, 5, 6, 7, 2, 3)
    #     + rdm4aabb.transpose(4, 5, 2, 3, 0, 1, 6, 7)
    #     + rdm4aabb.transpose(4, 5, 2, 3, 6, 7, 0, 1)
    #     + rdm4abbb
    #     + rdm4abbb.transpose(2, 3, 0, 1, 4, 5, 6, 7)
    #     + rdm4abbb.transpose(2, 3, 4, 5, 0, 1, 6, 7)
    #     + rdm4abbb.transpose(2, 3, 4, 5, 6, 7, 0, 1),
    #     rdm4,
    # )
    return (
        (rdm1a, rdm1b),
        (rdm2aa, rdm2ab, rdm2bb),
        (rdm3aaa, rdm3aab, rdm3abb, rdm3bbb),
        (rdm4aaaa, rdm4aaab, rdm4aabb, rdm4abbb, rdm4bbbb),
    )

def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    r'''Spin separated transition 1-particle density matrices.
    The return values include two density matrices: (alpha,alpha), (beta,beta).
    See also function :func:`make_rdm1s`

    1pdm[p,q] = :math:`\langle q^\dagger p \rangle`
    '''
    rdm1a = rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    r'''Spin traced transition 1-particle transition density matrices.

    1pdm[p,q] = :math:`\langle q_\alpha^\dagger p_\alpha \rangle
                       + \langle q_\beta^\dagger p_\beta \rangle`
    '''
    rdm1a, rdm1b = trans_rdm1s(cibra, ciket, norb, nelec, link_index)
    return rdm1a + rdm1b

def trans_rdm12s(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    r'''Spin separated 1- and 2-particle transition density matrices.
    The return values include two lists, a list of 1-particle transition
    density matrices and a list of 2-particle transition density matrices.
    The density matrices are:
    (alpha,alpha), (beta,beta) for 1-particle transition density matrices;
    (alpha,alpha,alpha,alpha), (alpha,alpha,beta,beta),
    (beta,beta,alpha,alpha), (beta,beta,beta,beta) for 2-particle transition
    density matrices.

    1pdm[p,q] = :math:`\langle q^\dagger p\rangle`;
    2pdm[p,q,r,s] = :math:`\langle p^\dagger r^\dagger s q\rangle`.
    '''
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket,
                                       norb, nelec, link_index, 2)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket,
                                       norb, nelec, link_index, 2)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket,
                                    norb, nelec, link_index, 0)
    _, dm2ba = rdm.make_rdm12_spin1('FCItdm12kern_ab', ciket, cibra,
                                    norb, nelec, link_index, 0)
    dm2ba = dm2ba.transpose(3,2,1,0)
    if reorder:
        dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
        dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

def trans_rdm12(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    r'''Spin traced transition 1- and 2-particle transition density matrices.

    1pdm[p,q] = :math:`\langle q^\dagger p\rangle`;
    2pdm[p,q,r,s] = :math:`\langle p^\dagger r^\dagger s q\rangle`.
    '''
    #(dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = \
    #        trans_rdm12s(cibra, ciket, norb, nelec, link_index, reorder)
    #return dm1a+dm1b, dm2aa+dm2ab+dm2ba+dm2bb
    dm1, dm2 = rdm.make_rdm12_spin1('FCItdm12kern_sf', cibra, ciket,
                                    norb, nelec, link_index, 2)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2

def _get_init_guess(na, nb, nroots, hdiag, nelec):
    # The "nroots" lowest determinats based on energy expectation value.
    ci0 = []
    neleca, nelecb = _unpack_nelec(nelec)
    if neleca == nelecb and na == nb:
        hdiag = lib.pack_tril(hdiag.reshape(na, na))
        if hdiag.size <= nroots:
            addrs = numpy.arange(hdiag.size)
        else:
            addrs = numpy.argpartition(hdiag, nroots-1)[:nroots]
        for addr in addrs:
            addra = (int)((2*addr+.25)**.5 - .5 + 1e-7)
            addrb = addr - addra*(addra+1)//2
            x = numpy.zeros((na, na))
            x[addra,addrb] = 1
            ci0.append(x.ravel().view(FCIvector))
    else:
        if hdiag.size <= nroots:
            addrs = numpy.arange(hdiag.size)
        else:
            addrs = numpy.argpartition(hdiag, nroots-1)[:nroots]
        for addr in addrs:
            x = numpy.zeros((na*nb))
            x[addr] = 1
            ci0.append(x.view(FCIvector))

    # Add noise
    ci0[0][0 ] += 1e-5
    ci0[0][-1] -= 1e-5
    return ci0

def get_init_guess(norb, nelec, nroots, hdiag):
    '''Initial guess is the single Slater determinant
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    return _get_init_guess(na, nb, nroots, hdiag, nelec)


###############################################################
# direct-CI driver
###############################################################

def kernel_ms1(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None, hop=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
    '''
    Args:
        h1e: ndarray
            1-electron Hamiltonian
        eri: ndarray
            2-electron integrals in chemist's notation
        norb: int
            Number of orbitals
        nelec: int or (int, int)
            Number of electrons of the system

    Kwargs:
        ci0: ndarray
            Initial guess
        link_index: ndarray
            A lookup table to cache the addresses of CI determinants in
            wave-function vector
        tol: float
            Convergence tolerance
        lindep: float
            Linear dependence threshold
        max_cycle: int
            Max. iterations for diagonalization
        max_space: int
            Max. trial vectors to store for sub-space diagonalization method
        nroots: int
            Number of states to solve
        davidson_only: bool
            Whether to call subspace diagonalization (davidson solver) or do a
            full diagonalization (lapack eigh) for small systems
        pspace_size: int
            Number of determinants as the threshold of "small systems",
        hop: function(c) => array_like_c
            Function to use for the Hamiltonian multiplication with trial vector

    Note: davidson solver requires more arguments. For the parameters not
    dispatched, they can be passed to davidson solver via the extra keyword
    arguments **kwargs
    '''
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(fci, verbose)

    nelec = _unpack_nelec(nelec, fci.spin)
    assert (0 <= nelec[0] <= norb and 0 <= nelec[1] <= norb)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec, compress=False).ravel()
    num_dets = hdiag.size
    pspace_size = min(num_dets, pspace_size)
    addr = [0]
    pw = pv = None
    if pspace_size > 0 and norb < 64:
        addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, pspace_size)
        pw, pv = fci.eig(h0)
        pspace_size = len(addr)

    if getattr(fci, 'sym_allowed_idx', None):
        # Remove symmetry forbidden elements
        sym_idx = numpy.hstack(fci.sym_allowed_idx)
        civec_size = sym_idx.size
    else:
        sym_idx = None
        civec_size = num_dets

    if max_memory < civec_size*6*8e-6:
        log.warn('Not enough memory for FCI solver. '
                 'The minimal requirement is %.0f MB', civec_size*60e-6)

    if pspace_size >= civec_size and ci0 is None and not davidson_only:
        if nroots > 1:
            nroots = min(civec_size, nroots)
            civec = numpy.empty((nroots,civec_size))
            civec[:,addr] = pv[:,:nroots].T
            return pw[:nroots]+ecore, civec
        elif pspace_size == 1 or abs(pw[0]-pw[1]) > 1e-12:
            # Check degeneracy. Degenerated wfn may break point group symmetry.
            # Davidson iteration with a proper initial guess can avoid this problem.
            civec = numpy.empty(civec_size)
            civec[addr] = pv[:,0]
            return pw[0]+ecore, civec
    pw = pv = h0 = None

    if sym_idx is None:
        precond = fci.make_precond(hdiag)
    else:
        precond = fci.make_precond(hdiag[sym_idx])

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    if hop is None:
        cpu0 = [logger.process_clock(), logger.perf_counter()]
        def hop(c):
            hc = fci.contract_2e(h2e, c, norb, nelec, link_index)
            cpu0[:] = log.timer_debug1('contract_2e', *cpu0)
            return hc.ravel()

    def init_guess():
        if callable(getattr(fci, 'get_init_guess', None)):
            return fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            x0 = []
            for i in range(min(len(addr), nroots)):
                x = numpy.zeros(civec_size)
                x[addr[i]] = 1
                x0.append(x)
            return x0

    if ci0 is None:
        ci0 = init_guess  # lazy initialization to reduce memory footprint
    elif not callable(ci0):
        if isinstance(ci0, numpy.ndarray):
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]
        if sym_idx is not None and ci0[0].size != civec_size:
            ci0 = [x[sym_idx] for x in ci0]
        # If provided initial guess ci0 are accidentally the eigenvectors of the
        # system, Davidson solver may be failed to find enough roots as it is
        # unable to generate more subspace basis from ci0. Adding vectors so
        # initial guess to help Davidson solver generate enough basis.
        if len(ci0) < nroots:
            ci0.extend(init_guess()[len(ci0):])

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    with lib.with_omp_threads(fci.threads):
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    return e+ecore, c

def make_pspace_precond(hdiag, pspaceig, pspaceci, addr, level_shift=0):
    # precondition with pspace Hamiltonian, CPL, 169, 463
    def precond(r, e0, x0, *args):
        #h0e0 = h0 - numpy.eye(len(addr))*(e0-level_shift)
        h0e0inv = numpy.dot(pspaceci/(pspaceig-(e0-level_shift)), pspaceci.T)
        hdiaginv = 1/(hdiag - (e0-level_shift))
        hdiaginv[abs(hdiaginv)>1e8] = 1e8
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
    return precond

def make_diag_precond(hdiag, pspaceig, pspaceci, addr, level_shift=0):
    return lib.make_diag_precond(hdiag, level_shift)


class FCIBase(lib.StreamObject):
    '''Full CI solver

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_cycle : int
            Total number of iterations. Default is 100
        max_space : tuple of int
            Davidson iteration space size. Default is 14.
        conv_tol : float
            Energy convergence tolerance. Default is 1e-10.
        level_shift : float
            Level shift applied in the preconditioner to avoid singularity.
            Default is 1e-3
        davidson_only : bool
            By default, the entire Hamiltonian matrix will be constructed and
            diagonalized if the system is small (see attribute pspace_size).
            Setting this parameter to True will enforce the eigenvalue
            problems being solved by Davidson subspace algorithm.  This flag
            should be enabled when initial guess is given or particular spin
            symmetry or point-group symmetry is required because the initial
            guess or symmetry are completely ignored in the direct diagonalization.
        pspace_size : int
            The dimension of Hamiltonian matrix over which Davidson iteration
            algorithm will be used for the eigenvalue problem.  Default is 400.
            This is roughly corresponding to a (6e,6o) system.
        nroots : int
            Number of states to be solved.  Default is 1, the ground state.
        spin : int or None
            Spin (2S = nalpha-nbeta) of the system.  If this attribute is None,
            spin will be determined by the argument nelec (number of electrons)
            of the kernel function.
        wfnsym : str or int
            Symmetry of wavefunction.  It is used only in direct_spin1_symm
            and direct_spin0_symm solver.

    Saved results

        eci : float or a list of float
            FCI energy(ies)
        ci : nparray
            FCI wfn vector(s)
        converged : bool (or a list of bool for multiple roots)
            Whether davidson iteration is converged

    Examples:

    >>> from pyscf import gto, scf, ao2mo, fci
    >>> mol = gto.M(atom='Li 0 0 0; Li 0 0 1', basis='sto-3g')
    >>> mf = scf.RHF(mol).run()
    >>> h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    >>> eri = ao2mo.kernel(mol, mf.mo_coeff)
    >>> cisolver = fci.direct_spin1.FCI(mol)
    >>> e, ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc())
    >>> print(e)
    -14.4197890826
    '''

    max_cycle = getattr(__config__, 'fci_direct_spin1_FCI_max_cycle', 100)
    max_space = getattr(__config__, 'fci_direct_spin1_FCI_max_space', 12)
    conv_tol = getattr(__config__, 'fci_direct_spin1_FCI_conv_tol', 1e-10)
    conv_tol_residual = getattr(__config__, 'fci_direct_spin1_FCI_conv_tol_residual', None)
    lindep = getattr(__config__, 'fci_direct_spin1_FCI_lindep', 1e-14)

    # level shift in preconditioner is helpful to avoid singularity and linear
    # dependence basis in davidson diagonalization solver
    level_shift = getattr(__config__, 'fci_direct_spin1_FCI_level_shift', 1e-3)

    # force the diagonalization use davidson iteration.  When the CI space
    # is small, the solver exactly diagonalizes the Hamiltonian.  But this
    # solution will ignore the initial guess.  Setting davidson_only can
    # enforce the solution on the initial guess state
    davidson_only = getattr(__config__, 'fci_direct_spin1_FCI_davidson_only', False)

    pspace_size = getattr(__config__, 'fci_direct_spin1_FCI_pspace_size', 400)
    threads = getattr(__config__, 'fci_direct_spin1_FCI_threads', None)
    lessio = getattr(__config__, 'fci_direct_spin1_FCI_lessio', False)

    _keys = {
        'max_cycle', 'max_space', 'conv_tol', 'lindep',
        'level_shift', 'davidson_only', 'pspace_size', 'threads', 'lessio',
        'mol', 'nroots', 'spin', 'orbsym', 'wfnsym', 'converged', 'norb',
        'nelec', 'eci', 'ci',
    }

    def __init__(self, mol=None):
        if mol is None:
            self.verbose = logger.NOTE
            self.max_memory = lib.param.MAX_MEMORY
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
            self.max_memory = mol.max_memory
        self.mol = mol
        self.nroots = 1
        self.spin = None
        # Initialize symmetry attributes for the compatibility with direct_spin1_symm
        # solver.  They are not used by direct_spin1 solver.
        self.orbsym = None
        self.wfnsym = None

        self.converged = False
        self.norb = None
        self.nelec = None
        self.eci = None
        self.ci = None

    @property
    def e_tot(self):
        return self.eci

    @property
    def nstates(self):
        return self.nroots
    @nstates.setter
    def nstates(self, x):
        self.nroots = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('max. cycles = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('davidson only = %s', self.davidson_only)
        log.info('linear dependence = %g', self.lindep)
        log.info('level shift = %g', self.level_shift)
        log.info('max iter space = %d', self.max_space)
        log.info('max_memory %d MB', self.max_memory)
        log.info('nroots = %d', self.nroots)
        log.info('pspace_size = %d', self.pspace_size)
        log.info('spin = %s', self.spin)
        return self

    @lib.with_doc(absorb_h1e.__doc__)
    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        nelec = _unpack_nelec(nelec, self.spin)
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    @lib.with_doc(make_hdiag.__doc__)
    def make_hdiag(self, h1e, eri, norb, nelec, compress=False):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_hdiag(h1e, eri, norb, nelec, compress)

    @lib.with_doc(pspace.__doc__)
    def pspace(self, h1e, eri, norb, nelec, hdiag=None, np=400):
        nelec = _unpack_nelec(nelec, self.spin)
        return pspace(h1e, eri, norb, nelec, hdiag, np)

    @lib.with_doc(contract_1e.__doc__)
    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    @lib.with_doc(contract_2e.__doc__)
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, numpy.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)

        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=self.lessio, **kwargs)
        if kwargs['nroots'] == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci

    def make_precond(self, hdiag, pspaceig=None, pspaceci=None, addr=None):
        if pspaceig is None:
            return make_diag_precond(hdiag, pspaceig, pspaceci, addr,
                                     self.level_shift)
        else:
            # Note: H0 in pspace may break symmetry.
            return make_pspace_precond(hdiag, pspaceig, pspaceci, addr,
                                       self.level_shift)

    @lib.with_doc(get_init_guess.__doc__)
    def get_init_guess(self, norb, nelec, nroots, hdiag):
        return get_init_guess(norb, nelec, nroots, hdiag)

    @lib.with_doc(kernel_ms1.__doc__)
    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec = _unpack_nelec(nelec, self.spin)
        link_index = _unpack(norb, nelec, None)
        e, c = kernel_ms1(self, h1e, eri, norb, nelec, ci0, link_index,
                          tol, lindep, max_cycle, max_space, nroots,
                          davidson_only, pspace_size, ecore=ecore, **kwargs)
        self.eci = e

        na = link_index[0].shape[0]
        nb = link_index[1].shape[0]
        if nroots > 1:
            self.ci = [x.reshape(na,nb).view(FCIvector) for x in c]
        else:
            self.ci = c.reshape(na,nb).view(FCIvector)
        return self.eci, self.ci

    @lib.with_doc(energy.__doc__)
    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        nelec = _unpack_nelec(nelec, self.spin)
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

    def spin_square(self, fcivec, norb, nelec):
        nelec = _unpack_nelec(nelec, self.spin)
        return spin_op.spin_square0(fcivec, norb, nelec)
    spin_square.__doc__ = spin_op.spin_square0.__doc__

    @lib.with_doc(make_rdm1s.__doc__)
    def make_rdm1s(self, fcivec, norb, nelec, link_index=None):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm1s(fcivec, norb, nelec, link_index)

    @lib.with_doc(make_rdm1.__doc__)
    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm1(fcivec, norb, nelec, link_index)

    @lib.with_doc(make_rdm12s.__doc__)
    def make_rdm12s(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm12s(fcivec, norb, nelec, link_index, reorder)

    @lib.with_doc(make_rdm12.__doc__)
    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)

    @lib.with_doc(make_rdm123s.__doc__)
    def make_rdm123s(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm123s(fcivec, norb, nelec, link_index, reorder)

    @lib.with_doc(make_rdm123.__doc__)
    def make_rdm123(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm123(fcivec, norb, nelec, link_index, reorder)

    @lib.with_doc(make_rdm1234s.__doc__)
    def make_rdm1234s(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm1234s(fcivec, norb, nelec, link_index, reorder)

    @lib.with_doc(make_rdm1234.__doc__)
    def make_rdm1234(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm1234(fcivec, norb, nelec, link_index, reorder)

    def make_rdm2(self, fcivec, norb, nelec, link_index=None, reorder=True):
        r'''Spin traced 2-particle density matrice

        NOTE the 2pdm is :math:`\langle p^\dagger q^\dagger s r\rangle` but
        stored as [p,r,q,s]
        '''
        nelec = _unpack_nelec(nelec, self.spin)
        return self.make_rdm12(fcivec, norb, nelec, link_index, reorder)[1]

    @lib.with_doc(trans_rdm1s.__doc__)
    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None):
        nelec = _unpack_nelec(nelec, self.spin)
        return trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    @lib.with_doc(trans_rdm1.__doc__)
    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        nelec = _unpack_nelec(nelec, self.spin)
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    @lib.with_doc(trans_rdm12s.__doc__)
    def trans_rdm12s(self, cibra, ciket, norb, nelec, link_index=None,
                     reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return trans_rdm12s(cibra, ciket, norb, nelec, link_index, reorder)

    @lib.with_doc(trans_rdm12.__doc__)
    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        nelec = _unpack_nelec(nelec, self.spin)
        return trans_rdm12(cibra, ciket, norb, nelec, link_index, reorder)

    def large_ci(self, fcivec, norb, nelec,
                 tol=getattr(__config__, 'fci_addons_large_ci_tol', .1),
                 return_strs=getattr(__config__, 'fci_addons_large_ci_return_strs', True)):
        nelec = _unpack_nelec(nelec, self.spin)
        return addons.large_ci(fcivec, norb, nelec, tol, return_strs)

    def contract_ss(self, fcivec, norb, nelec):  # noqa: F811
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_ss(fcivec, norb, nelec)

    def gen_linkstr(self, norb, nelec, tril=True, spin=None):
        if spin is None:
            spin = self.spin
        neleca, nelecb = _unpack_nelec(nelec, spin)
        if tril:
            link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
            link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
        else:
            link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb


class FCISolver(FCIBase):
    # transform_ci_for_orbital_rotation only available for FCI wavefunctions.
    # Some approx FCI solver does not have this functionality.
    def transform_ci_for_orbital_rotation(self, fcivec, norb, nelec, u):
        nelec = _unpack_nelec(nelec, self.spin)
        return addons.transform_ci_for_orbital_rotation(fcivec, norb, nelec, u)

    to_gpu = lib.to_gpu

FCI = FCISolver

class FCIvector(numpy.ndarray):
    '''An 2D np array for FCI coefficients'''

    # Special cases for ndarray when the array was modified (through ufunc)
    def __array_wrap__(self, out, context=None, return_scalar=False):
        if out.shape == self.shape:
            return out
        elif out.shape == ():  # if ufunc returns a scalar
            return out[()]
        else:
            return out.view(numpy.ndarray)

def _unpack(norb, nelec, link_index, spin=None):
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        link_indexa = link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
        return link_indexa, link_indexb
    else:
        return link_index
