#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

'''
Non-relativistic generalized Hartree-Fock analytical nuclear gradients

The one-electron Hamiltonian is assumed to be spin-free, so spin-orbit ECPs
(GHF.with_soc) and the X2C Hamiltonian (GHF.x2c1e()) are rejected rather than
silently treated as if their extra terms had no geometry dependence.
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad


def _spin_block_dms(dm):
    '''Real density matrices that the derivative J and K integrals are
    contracted with.

    A GHF density matrix is Hermitian over the spin-blocked AO basis and is
    complex whenever the solution is not collinear,

        dm = [[dm_aa, dm_ab],
              [dm_ba, dm_bb]]

    The Coulomb term only sees the charge density dm_aa + dm_bb. Its
    imaginary part is antisymmetric and cancels against the symmetric
    two-electron integrals, so only the real part is kept.

    The exchange term sees each spin block on its own. Using the hermiticity
    of the full density matrix, dm_ts[nu,mu] = dm_st[mu,nu].conj(), the real
    exchange energy separates into real and imaginary parts,

        E_K = -1/2 sum_st sum_mn ( K[Re dm_st]_mn Re dm_st_mn
                                 + K[Im dm_st]_mn Im dm_st_mn )

    which lets the whole gradient run through the existing real integral
    drivers instead of a complex one.

    Returns:
        (dm_j, dm_k) where dm_j is the real charge density and dm_k holds the
        eight real matrices ordered Re/Im of aa, ab, ba, bb.
    '''
    nao = dm.shape[-1] // 2
    blocks = (dm[:nao,:nao], dm[:nao,nao:], dm[nao:,:nao], dm[nao:,nao:])
    dm_j = (blocks[0] + blocks[3]).real
    dm_k = numpy.asarray([numpy.asarray(part(block), dtype=numpy.float64)
                          for block in blocks
                          for part in (numpy.real, numpy.imag)])
    return numpy.ascontiguousarray(dm_j), numpy.ascontiguousarray(dm_k)


def _check_spin_free_hcore(mf):
    '''Refuse Hamiltonians whose one-electron part is not spin-free.

    grad_elec contracts the derivative of the spin-free core Hamiltonian with
    the charge density dm_aa + dm_bb. That is only the whole one-electron
    gradient when hcore is block diagonal with the same block in each spin
    channel. Spin-orbit ECPs and the X2C Hamiltonian both break that, and the
    derivatives of the extra terms are not available here, so the result would
    silently be missing a contribution rather than being wrong by a little.
    '''
    if getattr(mf, 'with_x2c', None) is not None:
        raise NotImplementedError(
            'Nuclear gradients for the X2C Hamiltonian. grad.ghf assumes a '
            'spin-free one-electron Hamiltonian, which X2C is not.')
    if getattr(mf, 'with_soc', None) and getattr(mf.mol, 'has_ecp_soc',
                                                 lambda: False)():
        raise NotImplementedError(
            'Nuclear gradients with spin-orbit ECPs (with_soc=True). The '
            'derivative of the ECP spin-orbit term is not implemented, so the '
            'gradient would be missing that contribution.')

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of GHF/GKS gradients

    Args:
        mf_grad : grad.ghf.Gradients or grad.gks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    _check_spin_free_hcore(mf)
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1(dm0, mo_coeff, mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    nao = dm0.shape[-1] // 2

    # hcore and the overlap are spin-free, so only the charge density couples
    # to them. Both matrices are Hermitian, so their antisymmetric imaginary
    # parts cancel against the symmetric hcore derivative and against the
    # bra-plus-ket sum of the overlap derivative.
    dm0_sf = (dm0[:nao,:nao] + dm0[nao:,nao:]).real
    dme0_sf = (dme0[:nao,:nao] + dme0[nao:,nao:]).real

    dm_j, dm_k = _spin_block_dms(dm0)

    if mol._pseudo:
        from pyscf.gto.pp_int import vpploc_nuc_grad, vppnl_nuc_grad
        de = vpploc_nuc_grad(mol, dm0_sf)
        de += vppnl_nuc_grad(mol, dm0_sf)
    else:
        de = numpy.zeros((mol.natm, 3))

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-GHF Coulomb repulsion')
    vj, vk, vxc = mf_grad.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    if atmlst is None:
        atmlst = range(mol.natm)
    else:
        de = de[atmlst]

    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0_sf)
# s1, vj and vk are \nabla <i|..|j>, *2 for the contributions of nabla|ket>
        de[k] += numpy.einsum('xij,ij->x', vj[:,p0:p1], dm_j[p0:p1]) * 2
        if vk is not None:
            de[k] -= numpy.einsum('sxij,sij->x', vk[:,:,p0:p1], dm_k[:,p0:p1]) * 2
        if vxc is not None:
            # The XC potential is spin-block diagonal: it is contracted with
            # the real diagonal blocks, in the order (aa, bb).
            de[k] += numpy.einsum('sxij,sij->x', vxc[:,:,p0:p1],
                                  dm_k[::6][:,p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2

        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
    return de


def get_veff(mf_grad, mol, dm):
    '''
    First order derivative of the GHF potential matrix (wrt electron
    coordinates)

    Args:
        mf_grad : grad.ghf.Gradients or grad.gks.Gradients object

    Returns:
        (vj, vk, vxc). vj is contracted with the charge density and vk with
        the eight real spin-block matrices, both from :func:`_spin_block_dms`.
        vk is None when there is no exact exchange, and vxc is None for
        Hartree-Fock; grad.gks fills in the exchange-correlation part.
    '''
    dm_j, dm_k = _spin_block_dms(dm)
    vj = mf_grad.get_j(mol, dm_j)
    vk = mf_grad.get_k(mol, dm_k)
    return vj, vk, None


def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    return rhf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)


class Gradients(rhf_grad.GradientsBase):
    '''Non-relativistic generalized Hartree-Fock gradients
    '''
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    grad_elec = grad_elec

Grad = Gradients

from pyscf import scf
scf.ghf.GHF.Gradients = lib.class_as_method(Gradients)
