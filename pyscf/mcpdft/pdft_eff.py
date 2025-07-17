#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hermes
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

import numpy as np
from pyscf import __config__, lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.mcpdft.otpd import _grid_ao2mo
import ctypes

SWITCH_SIZE = getattr(__config__, 'dft_numint_SWITCH_SIZE', 800)
libpdft = lib.load_library('libpdft')


# TODO: outcore implementation; can I use properties instead of copying?
class _ERIS:
    '''Stores two-body PDFT on-top effective integral arrays in a form
    compatible with existing MC-SCF kernel and derivative functions.
    Unlike actual eris, PDFT 2-electron effective integrals have 24-fold
    permutation symmetry, so j_pc = k_pc and ppaa = papa.transpose
    (0,2,1,3). The mcscf _ERIS is currently undocumented so I won't
    spend more time documenting this for now.
    '''

    def __init__(self, mol, mo_coeff, ncore, ncas, method='incore',
                 paaa_only=False, aaaa_only=False, jk_pc=False, verbose=0, stdout=None):
        self.mol = mol
        self.mo_coeff = mo_coeff
        self.nao, self.nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.vhf_c = np.zeros((self.nmo, self.nmo), dtype=mo_coeff.dtype)
        self.method = method
        self.paaa_only = paaa_only
        self.aaaa_only = aaaa_only
        self.jk_pc = jk_pc
        self.verbose = verbose
        self.stdout = stdout
        if method == 'incore':
            self.papa = np.zeros((self.nmo, ncas, self.nmo, ncas),
                                 dtype=mo_coeff.dtype)
            self.j_pc = np.zeros((self.nmo, ncore), dtype=mo_coeff.dtype)
        else:
            raise NotImplementedError("method={} for pdft_eff2".format(
                self.method))

    def _accumulate(self, ot, ao, weight, rho_c, rho_a, eff_Pi,
                    non0tab=None, shls_slice=None, ao_loc=None):
        args = [ot, ao, weight, rho_c, rho_a, eff_Pi, non0tab, shls_slice, ao_loc]
        self._accumulate_vhf_c(*args)
        if self.method.lower() == 'incore':
            self._accumulate_ppaa_incore(*args)
        else:
            raise NotImplementedError("method={} for pdft_eff2".format(
                self.method))
        self._accumulate_j_pc(*args)

    def _accumulate_vhf_c(self, ot, ao, weight, rho_c, rho_a, eff_Pi,
                          non0tab, shls_slice, ao_loc):
        mo_coeff = self.mo_coeff
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas

        vrho_c = _contract_eff_rho(eff_Pi, rho_c)
        self.vhf_c += mo_coeff.conjugate().T @ ot.get_eff_1body(ao,
                                                             weight, vrho_c, non0tab=non0tab, shls_slice=shls_slice,
                                                             ao_loc=ao_loc,
                                                             hermi=1) @ mo_coeff
        self.energy_core = np.trace(self.vhf_c[:ncore, :ncore]) #/ 2
        if self.paaa_only:
            # 1/2 v_aiuv D_ii D_uv = v^ai_uv D_uv -> F_ai, F_ia
            # needs to be in here since it would otherwise be calculated using
            # ppaa and papa. This is harmless to the CI problem because the
            # elements in the active space and core-core sector are ignored
            # below.
            eff_rho_a = _contract_eff_rho(eff_Pi, rho_a)
            vhf_a = get_eff_1body(ot, ao, weight, eff_rho_a, non0tab=non0tab,
                                  shls_slice=shls_slice, ao_loc=ao_loc, hermi=1)
            vhf_a = mo_coeff.conjugate().T @ vhf_a @ mo_coeff
            vhf_a[ncore:nocc, :] = vhf_a[:, ncore:nocc] = 0.0
            self.vhf_c += vhf_a

    def _ftpt_vhf_c(self):
        return self.nao + 1

    def _accumulate_ppaa_incore(self, ot, ao, weight, rho_c, rho_a,
                                eff_Pi, non0tab, shls_slice, ao_loc):
        # ao is here stored in row-major order = deriv,AOs,grids regardless of
        # what the ndarray object thinks
        mo_coeff = self.mo_coeff
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        nderiv = eff_Pi.shape[0]
        mo_cas = _grid_ao2mo(self.mol, ao[:nderiv], mo_coeff[:, ncore:nocc],
                             non0tab)
        if self.aaaa_only:
            aaaa = ot.get_eff_2body([mo_cas, mo_cas, mo_cas,
                                      mo_cas], weight, eff_Pi, aosym='s1')
            self.papa[ncore:nocc, :, ncore:nocc, :] += aaaa
        elif self.paaa_only:
            paaa = ot.get_eff_2body([ao, mo_cas, mo_cas, mo_cas],
                                 weight, eff_Pi, aosym='s1')
            paaa = np.tensordot(mo_coeff.T, paaa, axes=1)
            self.papa[:, :, ncore:nocc, :] += paaa
            self.papa[ncore:nocc, :, :, :] += paaa.transpose(2, 3, 0, 1)
            self.papa[ncore:nocc, :, ncore:nocc, :] -= paaa[ncore:nocc, :, :, :]
        else:
            papa = ot.get_eff_2body([ao, mo_cas, ao, mo_cas],
                                 weight, eff_Pi, aosym='s1')
            papa = np.tensordot(mo_coeff.T, papa, axes=1)
            self.papa += np.tensordot(mo_coeff.T, papa,
                                      axes=((1), (2))).transpose(1, 2, 0, 3)

    def _ftpt_ppaa_incore(self):
        nao, ncas = self.nao, self.ncas
        ncol = 1 + 2 * ncas
        ij_aa = int(self.aaaa_only)
        kl_aa = int(ij_aa or self.paaa_only)
        ncol += (ij_aa + kl_aa) * (nao - ncas)
        return ncol * ncas

    def _accumulate_j_pc(self, ot, ao, weight, rho_c, rho_a, vPi,
                         non0tab, shls_slice, ao_loc):
        mo_coeff = self.mo_coeff
        ncore = self.ncore
        nderiv = vPi.shape[0]
        if self.jk_pc:
            mo = _square_ao(_grid_ao2mo(self.mol, ao[:nderiv], mo_coeff,
                                        non0tab))
            mo_core = mo[:, :, :ncore]
            self.j_pc += ot.get_eff_1body([mo, mo_core], weight, vPi)

    def _ftpt_j_pc(self):
        return self.nao + self.ncore + self.ncas + 1

    def _accumulate_ftpt(self):
        """ memory footprint of _accumulate, divided by nderiv_Pi*ngrids """
        ftpt_fns = [self._ftpt_vhf_c]
        if self.method.lower() == 'incore':
            ftpt_fns.append(self._ftpt_ppaa_incore)
        else:
            raise NotImplementedError("method={} for pdft_eff2".format(
                self.method))
        if self.verbose > logger.DEBUG:
            ftpt_fns.append(self._ftpt_j_pc)
        ncol = 0
        for fn in ftpt_fns: ncol = max(ncol, fn())
        return ncol

    def _finalize(self):
        if self.method == 'incore':
            self.ppaa = np.ascontiguousarray(self.papa.transpose(0, 2, 1, 3))
            self.k_pc = self.j_pc.copy()
        else:
            raise NotImplementedError("method={} for pdft_eff2".format(
                self.method))
        self.k_pc = self.j_pc.copy()


def _contract_eff_rho(eff, rho, add_eff_rho=None):
    """ Make a jk-like eff_rho from eff and a density. k = j so it's just
        eff * eff_rho / 2 , but the product rule needs to be followed """
    if rho.ndim == 1:
        rho = rho[None, :]

    nderiv = eff.shape[0]
    eff_rho = eff * rho[0]

    if nderiv > 1:
        eff_rho[0] += (eff[1:4] * rho[1:4]).sum(0)

    eff_rho /= 2
    # eff involves lower derivatives than eff_rho in original translation
    # make sure vot * rho gets added to only the proper component(s)
    if add_eff_rho is not None:
        add_eff_rho[:nderiv] += eff_rho
        eff_rho = add_eff_rho
    return eff_rho


def _square_ao(ao):
    # On a grid, square each element of an AO or MO array, but preserve the
    # chain rule so that columns 1 to 4 are still the first derivative of
    # the squared AO value, etc.
    nderiv = ao.shape[0]
    ao_sq = ao * ao[0]
    if nderiv > 1:
        ao_sq[1:4] *= 2
    if nderiv > 4:
        ao_sq[4:10] += ao[1:4] ** 2
        ao_sq[4:10] *= 2
    return ao_sq

def get_eff_1body(otfnal, ao, weight, kern, non0tab=None,
                  shls_slice=None, ao_loc=None, hermi=0):
    r''' Contract the kern with d vrho/ dDpq.

    Args:
        ao : ndarray or 2 ndarrays of shape (*,ngrids,nao)
            contains values and derivatives of nao.
            2 different ndarrays can have different nao but not
            different ngrids
        weight : ndarray of shape (ngrids)
            containing numerical integration weights
        kern : ndarray of shape (*,ngrids)
            the derivative of the on-top potential with respect to
            density (vrho)/ If not provided, it is calculated.

    Kwargs:
        non0tab : ndarray of shape (nblk, nbas)
            Identifies blocks of grid points which are nonzero on
            each AO shell so as to exploit sparsity.
            If you want the "ao" array to be in the MO basis, just
            leave this as None. If hermi == 0, it only applies
            to the bra index ao array, even if the ket index ao
            array is the same (so probably always pass hermi = 1
            in that case)
        shls_slice : sequence of integers of len 2
            Identifies starting and stopping indices of AO shells
        ao_loc : ndarray of length nbas
            Offset to first AO of each shell
        hermi : integer or logical
            Toggle whether veff is supposed to be a Hermitian matrix
            You can still pass two different ao arrays for the bra and
            the ket indices, for instance if one of them is supposed to
            be a higher derivative. They just have to have the same nao
            in that case.

    Returns : ndarray of shape (nao[0],nao[1])
        The 1-body effective term corresponding to kernel times the AO's,
        in the atomic-orbital basis. In PDFT this functional is always
        spin-symmetric.
    '''
    if isinstance(ao, np.ndarray) and ao.ndim == 3:
        ao = [ao, ao]
    elif len(ao) != 2:
        raise NotImplementedError("uninterpretable aos!")
    elif ao[0].size < ao[1].size:
        # Life pro-tip: do more operations with smaller arrays and fewer
        # operations with bigger arrays
        ao = [ao[1], ao[0]]

    kern = kern.copy()
    kern *= weight[None, :]

    # Zeroth and first derivatives
    nderiv = kern.shape[0]

    first_pass = min(nderiv, 4)
    eff_ao = _contract_kern_ao(kern[:first_pass], ao[1][:first_pass])
    nterm = eff_ao.shape[0]
    eff = sum([_dot_ao_mo(otfnal.mol, a, v, non0tab=non0tab,
                          shls_slice=shls_slice, ao_loc=ao_loc, hermi=hermi)
               for a, v in zip(ao[0][0:nterm], eff_ao)])

    if nderiv > 4:
        # check if we have laplacian...
        vtau = kern[4]

        vtau *= 0.5
        eff += _tau_dot_ao_mo(
            otfnal.mol,
            ao[1],
            ao[0],
            vtau,
            non0tab=non0tab,
            shls_slice=shls_slice,
            ao_loc=ao_loc,
            hermi=hermi,
        )

    if nderiv > 5:
        raise NotImplementedError("laplacian translated meta-GGA functional")

    return eff

def get_eff_2body(otfnal, ao, weight, kern, aosym='s4', eff_ao=None):
    if isinstance(ao, np.ndarray) and ao.ndim == 3:
        ao = [ao, ao, ao, ao]
    elif len(ao) != 4:
        raise NotImplementedError('fancy orbital subsets and fast evaluation '
                                  'in get_eff_2body')

    if isinstance(aosym, int):
        aosym = str(aosym)
    ij_symm = "4" in aosym or "2ij" in aosym
    kl_symm = "4" in aosym or "2kl" in aosym

    if eff_ao is None:
        eff_ao = otfnal.get_eff_2body_kl(ao[2], ao[3], weight, kern, symm=kl_symm)

    nderiv = eff_ao.shape[0]
    ao2 = _contract_ao1_ao2(ao[0], ao[1], nderiv, symm=ij_symm)
    try:
        ao2 = ao2.transpose(0, 3, 2, 1)

    except ValueError as e:
        print(ao[0].shape, ao[1].shape, ao2.shape)
        raise(e)

    eff_ao = eff_ao.transpose(0, 3, 2, 1)
    ijkl_shape = list(ao2.shape[1:-1]) + list(eff_ao.shape[1:-1])
    ao2 = ao2.reshape(ao2.shape[0], -1, ao2.shape[-1]).transpose(0, 2, 1)
    eff_ao = eff_ao.reshape(eff_ao.shape[0], -1, eff_ao.shape[-1]).transpose(0, 2, 1)
    eff = sum([_dot_ao_mo(otfnal.mol, a, v) for a, v in zip(ao2, eff_ao)])
    eff = eff.reshape(*ijkl_shape)

    return eff


def get_eff_2body_kl(ot, ao_k, ao_l, weight, kern, symm=False):
    kern = kern.copy()
    kern *= weight[None, :]

    # Flatten deriv and grid so I can tensordot it all at once
    # Index symmetry can be built into _contract_ao1_ao2
    kern_ao = _contract_kern_ao(kern, ao_l)
    eff_ao = _contract_ao_eff_ao(ao_k, kern_ao, symm=symm)
    return eff_ao

def _dot_ao_mo(mol, ao, mo, non0tab=None, shls_slice=None, ao_loc=None,
               hermi=0):
    # f_ij = int g_i(r) * h_j(r) dr
    # Like numint._dot_ao_ao, but allows for two different bases on the
    # rows and columns
    if hermi:
        return numint._dot_ao_ao(mol, ao, mo, non0tab=non0tab,
                                 shls_slice=shls_slice, ao_loc=ao_loc, hermi=hermi)
    ngrids, nao = ao.shape
    nmo = mo.shape[-1]

    if nao < SWITCH_SIZE:
        return lib.dot(ao.T.conj(), mo)

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if not mo.flags.f_contiguous:
        mo = lib.transpose(mo)
    if ao.dtype == mo.dtype == np.double:
        fn = libpdft.VOTdot_ao_mo
    else:
        raise NotImplementedError("Complex-orbital PDFT")

    if non0tab is None or shls_slice is None or ao_loc is None:
        pnon0tab = pshls_slice = pao_loc = lib.c_null_ptr()
    else:
        pnon0tab = non0tab.ctypes.data_as(ctypes.c_void_p)
        pshls_slice = (ctypes.c_int * 2)(*shls_slice)
        pao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)

    vv = np.empty((nao, nmo), dtype=ao.dtype)
    fn(vv.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       mo.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(nmo),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       pnon0tab, pshls_slice, pao_loc)
    return vv


# TODO: unittest?
def _contract_kern_ao(vot, ao, out=None):
    # Evaluate v_i(r) = v(r) * AO_i(r) and its derivatives on a grid
    # Note that the chain rule means that v' * AO' -> v, v' * AO -> v'
    ''' REQUIRES array in shape = (nderiv,nao,ngrids) and data layout
        = (nderiv,ngrids,nao)/row-major '''
    nderiv = vot.shape[0]
    ao = np.ascontiguousarray(ao.transpose(0, 2, 1))
    nao, ngrids = ao.shape[1:]
    vao = np.ndarray((nderiv, nao, ngrids), dtype=ao.dtype,
                     buffer=out).transpose(0, 2, 1)
    ao = ao.transpose(0, 2, 1)
    vao[0] = numint._scale_ao(ao[:nderiv], vot, out=vao[0])
    if nderiv > 1:
        for i in range(1, 4):
            vao[i] = numint._scale_ao(ao[0:1, :, :], vot[i:i + 1, :], out=vao[i])

    return vao


def _contract_ao_eff_ao(ao, vao, symm=False):
    r''' Outer-product of ao grid and eff * ao grid
    Can be used with two-orb-dimensional vao if the last two dimensions
    are flattened into "nao"

    Args:
        ao : ndarray of shape (*,ngrids,nao1)
        vao : ndarray of shape (nderiv,ngrids,nao2)

    Kwargs:
        symm : logical
            If true, nao1 == nao2 must be true

    Returns: ndarray of shape (nderiv,ngrids,nao1,nao2)
        or (nderiv,ngrids,nao1*(nao1+1)//2)
    '''
    ao = ao.transpose(0, 2, 1)
    vao = vao.transpose(0, 2, 1)
    assert ao.flags.c_contiguous, 'shape = {} ; strides = {}'.format(
        ao.shape, ao.strides)
    assert vao.flags.c_contiguous, 'shape = {} ; strides = {}'.format(
        vao.shape, vao.strides)

    nderiv = vao.shape[0]
    if symm:
        ix_p, ix_q = np.tril_indices(ao.shape[1])
        ao = ao[:, ix_p]
        vao = vao[:, ix_q]
    else:
        ao = np.expand_dims(ao, -2)
        vao = np.expand_dims(vao, -3)
    prod = ao[0] * vao
    if nderiv > 1:
        prod[0] += (ao[1:4] * vao[1:4]).sum(0)
    if symm:
        prod = prod.transpose(0, 2, 1)
    else:
        prod = prod.transpose(0, 3, 2, 1)
    return prod


# TODO: unittest?
def _contract_ao1_ao2(ao1, ao2, nderiv, symm=False):
    # Evaluate P_ij(r) = AO_i(r) * AO_j(r) and its derivatives on a grid
    ao1 = ao1.transpose(0, 2, 1)
    ao2 = ao2.transpose(0, 2, 1)
    assert (ao1.flags.c_contiguous), 'shape = {} ; strides = {}'.format(
        ao1.shape, ao1.strides)
    assert (ao2.flags.c_contiguous), 'shape = {} ; strides = {}'.format(
        ao2.shape, ao2.strides)
    if symm:  # TODO: C implementation of this slow indexing
        ix_p, ix_q = np.tril_indices(ao1.shape[1])
        ao1 = ao1[:nderiv, ix_p]
        ao2 = ao2[:nderiv, ix_q]
    else:
        ao1 = np.expand_dims(ao1, -2)[:nderiv]
        ao2 = np.expand_dims(ao2, -3)[:nderiv]
    prod = ao1[:nderiv] * ao2[0]
    if nderiv > 1:
        prod[1:4] += ao1[0] * ao2[1:4]  # Product rule
    ao2 = None
    if symm:
        prod = prod.transpose(0, 2, 1)
    else:
        prod = prod.transpose(0, 3, 2, 1)
    return prod


def _tau_dot_ao_mo(
    mol, ao, mo, vtau, non0tab=None, shls_slice=None, ao_loc=None, hermi=0
):
    vao = numint._scale_ao(ao[1], vtau)
    eff = _dot_ao_mo(
        mol,
        mo[1],
        vao,
        non0tab=non0tab,
        shls_slice=shls_slice,
        ao_loc=ao_loc,
        hermi=hermi,
    )
    vao = numint._scale_ao(ao[2], vtau)
    eff += _dot_ao_mo(
        mol,
        mo[2],
        vao,
        non0tab=non0tab,
        shls_slice=shls_slice,
        ao_loc=ao_loc,
        hermi=hermi,
    )
    vao = numint._scale_ao(ao[3], vtau)
    eff += _dot_ao_mo(
        mol,
        mo[3],
        vao,
        non0tab=non0tab,
        shls_slice=shls_slice,
        ao_loc=ao_loc,
        hermi=hermi,
    )
    return eff
