#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Generate PBC (K)SCF response functions
'''

import numpy
from pyscf import lib
from pyscf.pbc.scf import khf, kuhf, krohf, kghf

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None, with_nlc=True):
    from pyscf.pbc.dft import numint, multigrid
    assert isinstance(mf, khf.KRHF)

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if isinstance(mf, khf.pbchf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if omega != 0:  # For range separated Coulomb
            raise NotImplementedError

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:  # for newton solver
            spin = 0
        else:
            spin = 1
        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc, mo_coeff,
                                            mo_occ, spin, kpts)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:  # Without specify singlet, general case
            def vind(dm1, kshift=0):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    assert kshift == 0
                    v1 = ni.nr_rks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, kpts, max_memory=max_memory)
                if hybrid:
                    if omega == 0:
                        vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                        vk *= hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=-omega)
                        vk *= hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega)
                        vk *= alpha
                    else: # SR and LR exchange with different ratios
                        vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                        vk *= hyb
                        vk += _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega) * (alpha-hyb)
                    if hermi != 2:
                        v1 += vj - .5 * vk
                    else:
                        v1 += -.5 * vk
                elif hermi != 2:
                    v1 += _get_j(mf, cell, dm1, hermi, kpts, kshift)
                return v1

        elif singlet:
            fxc *= .5
            def vind(dm1, kshift=0):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    assert kshift == 0
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                if hybrid:
                    if omega == 0:
                        vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                        vk *= hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=-omega)
                        vk *= hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega)
                        vk *= alpha
                    else: # SR and LR exchange with different ratios
                        vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                        vk *= hyb
                        vk += _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega) * (alpha-hyb)
                    if hermi != 2:
                        v1 += vj - .5 * vk
                    else:
                        v1 += -.5 * vk
                elif hermi != 2:
                    v1 += _get_j(mf, cell, dm1, hermi, kpts, kshift)
                return v1
        else:  # triplet
            def vind(dm1, kshift=0):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    assert kshift == 0
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if omega == 0:
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift) * hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=-omega) * hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega) * alpha
                    else: # SR and LR exchange with different ratios
                        vk = _get_k(mf, cell, dm1, hermi, kpts, kshift) * hyb
                        vk += _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1, kshift=0):
                vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                return vj - .5 * vk
        else:
            def vind(dm1, kshift=0):
                return -.5 * _get_k(mf, cell, dm1, hermi, kpts, kshift)

    return vind

def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True):
    from pyscf.pbc.dft import multigrid
    assert isinstance(mf, (kuhf.KUHF, krohf.KROHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if isinstance(mf, khf.pbchf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        if omega != 0:  # For range separated Coulomb
            raise NotImplementedError

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1, kpts)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1, kshift=0):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                assert kshift == 0
                v1 = ni.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, kpts, max_memory=max_memory)
            if not hybrid:
                if with_j:
                    vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                    v1 += vj[0] + vj[1]
            else:
                if omega == 0:
                    vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                    vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vj = _get_j(mf, cell, dm1, hermi, kpts, kshift)
                    vk = _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
                    vk *= hyb
                    vk += _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=omega) * (alpha-hyb)
                if with_j:
                    v1 += vj[0] + vj[1] - vk
                else:
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1, kshift=0):
            vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1, kshift=0):
            return -_get_k(mf, cell, dm1, hermi, kpts, kshift)

    return vind

def _gen_ghf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of KGHF response function and
    KGHF density matrices.
    '''
    raise NotImplementedError

def _get_jk_kshift(mf, dm_kpts, hermi, kpts, kshift, with_j=True, with_k=True,
                   omega=None):
    from pyscf.pbc.df.df_jk import get_j_kpts_kshift, get_k_kpts_kshift
    vj = vk = None
    if with_j:
        vj = get_j_kpts_kshift(mf.with_df, dm_kpts, kshift, hermi=hermi, kpts=kpts)
    if with_k:
        vk = get_k_kpts_kshift(mf.with_df, dm_kpts, kshift, hermi=hermi, kpts=kpts,
                               exxdiv=mf.exxdiv)
    return vj, vk
def _get_jk(mf, cell, dm1, hermi, kpts, kshift, with_j=True, with_k=True, omega=None):
    from pyscf.pbc import df
    if kshift == 0:
        return mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts,
                         with_j=with_j, with_k=with_k, omega=omega)
    elif omega is not None and omega != 0:
        raise NotImplementedError
    elif mf.rsjk is not None or not isinstance(mf.with_df, df.df.DF):
        lib.logger.error(mf, 'Non-zero kshift is only supported by GDF/RSDF.')
        raise NotImplementedError
    else:
        return _get_jk_kshift(mf, dm1, hermi, kpts, kshift,
                              with_j=with_j, with_k=with_k, omega=omega)
def _get_j(mf, cell, dm1, hermi, kpts, kshift, omega=None):
    return _get_jk(mf, cell, dm1, hermi, kpts, kshift, True, False, omega)[0]
def _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=None):
    return _get_jk(mf, cell, dm1, hermi, kpts, kshift, False, True, omega)[1]


khf.KRHF.gen_response = _gen_rhf_response
kuhf.KUHF.gen_response = _gen_uhf_response
kghf.KGHF.gen_response = _gen_ghf_response
krohf.KROHF.gen_response = _gen_uhf_response

from pyscf.pbc.scf import hf, uhf, rohf, ghf

def _gen_rhf_response_gam(mf, mo_coeff=None, mo_occ=None,
                          singlet=None, hermi=0, max_memory=None, with_nlc=True):
    from pyscf.pbc.dft import numint, multigrid
    assert isinstance(mf, hf.RHF)

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpt = mf.kpt
    if isinstance(mf, khf.pbchf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:  # for newton solver
            spin = 0
        else:
            spin = 1
        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc, mo_coeff,
                                            mo_occ, spin, kpt)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:  # Without specify singlet, general case
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, kpt, max_memory=max_memory)
                if hybrid:
                    if omega == 0:
                        vj, vk = mf.get_jk(cell, dm1, hermi, kpt=kpt)
                        vk *= hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vj = mf.get_j(cell, dm1, hermi, kpt=kpt)
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=-omega)
                        vk *= hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vj = mf.get_j(cell, dm1, hermi, kpt=kpt)
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega)
                        vk *= alpha
                    else: # SR and LR exchange with different ratios
                        vj, vk = mf.get_jk(cell, dm1, hermi, kpt=kpt)
                        vk *= hyb
                        vk += mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega) * (alpha-hyb)
                    if hermi != 2:
                        v1 += vj - .5 * vk
                    else:
                        v1 += -.5 * vk
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpt=kpt)
                return v1

        elif singlet:
            fxc *= .5
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc, kpt,
                                              max_memory=max_memory)
                if hybrid:
                    if omega == 0:
                        vj, vk = mf.get_jk(cell, dm1, hermi, kpt=kpt)
                        vk *= hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vj = mf.get_j(cell, dm1, hermi, kpt=kpt)
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=-omega)
                        vk *= hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vj = mf.get_j(cell, dm1, hermi, kpt=kpt)
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega)
                        vk *= alpha
                    else: # SR and LR exchange with different ratios
                        vj, vk = mf.get_jk(cell, dm1, hermi, kpt=kpt)
                        vk *= hyb
                        vk += mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega) * (alpha-hyb)
                    if hermi != 2:
                        v1 += vj - .5 * vk
                    else:
                        v1 += -.5 * vk
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpt=kpt)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc, kpt,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if omega == 0:
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt) * hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=-omega) * hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega) * alpha
                    else: # SR and LR exchange with different ratios
                        vk = mf.get_k(cell, dm1, hermi, kpt=kpt) * hyb
                        vk += mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpt=kpt)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(cell, dm1, hermi=hermi, kpt=kpt)

    return vind

def _gen_uhf_response_gam(mf, mo_coeff=None, mo_occ=None,
                          with_j=True, hermi=0, max_memory=None, with_nlc=True):
    from pyscf.pbc.dft import multigrid
    assert isinstance(mf, (uhf.UHF, rohf.ROHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpt = mf.kpt
    if isinstance(mf, khf.pbchf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1, kpt)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, kpt, max_memory=max_memory)
            if not hybrid:
                if with_j:
                    vj = mf.get_j(cell, dm1, hermi=hermi, kpt=kpt)
                    v1 += vj[0] + vj[1]
            else:
                if omega == 0:
                    vj, vk = mf.get_jk(cell, dm1, hermi, kpt=kpt)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vj = mf.get_j(cell, dm1, hermi, kpt=kpt)
                    vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vj = mf.get_j(cell, dm1, hermi, kpt=kpt)
                    vk = mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(cell, dm1, hermi, kpt=kpt)
                    vk *= hyb
                    vk += mf.get_k(cell, dm1, hermi, kpt=kpt, omega=omega) * (alpha-hyb)
                if with_j:
                    v1 += vj[0] + vj[1] - vk
                else:
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpt=kpt)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(cell, dm1, hermi=hermi, kpt=kpt)

    return vind

def _gen_ghf_response_gam(mf, mo_coeff=None, mo_occ=None,
                          with_j=True, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of KGHF response function and
    KGHF density matrices.
    '''
    raise NotImplementedError


hf.RHF.gen_response = _gen_rhf_response_gam
uhf.UHF.gen_response = _gen_uhf_response_gam
ghf.GHF.gen_response = _gen_ghf_response_gam
rohf.ROHF.gen_response = _gen_uhf_response_gam
