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
Generate SCF response functions
'''

import warnings
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, rohf, uhf, ghf, dhf

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
        with_nlc (boolean) : NLC contribution is typically very small. This flag
        allows to skip NLC contribution.
    '''
    assert isinstance(mf, hf.RHF) and not isinstance(mf, (uhf.UHF, rohf.ROHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if singlet is None: # for ground state orbital hessian
            spin = 0
        else:
            spin = 1
        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, spin)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:
            # Without specify singlet, used in ground state orbital hessian
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                    if with_nlc and mf.do_nlc():
                        from pyscf.hessian.rks import get_vnlc_resp # Cannot import at top due to circular dependency
                        v1 += get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1, max_memory)
                if hybrid:
                    if omega == 0:
                        vj, vk = mf.get_jk(mol, dm1, hermi)
                        vk *= hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vj = mf.get_j(mol, dm1, hermi)
                        vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                        vk *= hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vj = mf.get_j(mol, dm1, hermi)
                        vk = mf.get_k(mol, dm1, hermi, omega=omega)
                        vk *= alpha
                    else: # SR and LR exchange with different ratios
                        vj, vk = mf.get_jk(mol, dm1, hermi)
                        vk *= hyb
                        vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                    if hermi != 2:
                        v1 += vj - .5 * vk
                    else:
                        v1 += -.5 * vk
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            fxc *= .5
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, hermi, True,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    if with_nlc and mf.do_nlc():
                        from pyscf.hessian.rks import get_vnlc_resp # Cannot import at top due to circular dependency
                        v1 += get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1, max_memory)
                if hybrid:
                    if omega == 0:
                        vj, vk = mf.get_jk(mol, dm1, hermi)
                        vk *= hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vj = mf.get_j(mol, dm1, hermi)
                        vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                        vk *= hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vj = mf.get_j(mol, dm1, hermi)
                        vk = mf.get_k(mol, dm1, hermi, omega=omega)
                        vk *= alpha
                    else: # SR and LR exchange with different ratios
                        vj, vk = mf.get_jk(mol, dm1, hermi)
                        vk *= hyb
                        vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                    if hermi != 2:
                        v1 += vj - .5 * vk
                    else:
                        v1 += -.5 * vk
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:  # triplet
            fxc *= .5
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, mf.grids, mf.xc, dm0, dm1, hermi, False,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    if with_nlc and mf.do_nlc():
                        pass # fxc = 0, do nothing
                if hybrid:
                    if omega == 0:
                        vk = mf.get_k(mol, dm1, hermi) * hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=-omega) * hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=omega) * alpha
                    else: # SR and LR exchange with different ratios
                        vk = mf.get_k(mol, dm1, hermi) * hyb
                        vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    assert isinstance(mf, (uhf.UHF, rohf.ROHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)
                if with_nlc and mf.do_nlc():
                    from pyscf.hessian.rks import get_vnlc_resp # Cannot import at top due to circular dependency
                    v1 += get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1[0] + dm1[1], max_memory)
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj[0] + vj[1]
            else:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                if with_j:
                    v1 += vj[0] + vj[1] - vk
                else:
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_ghf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of GHF response function and
    GHF density matrices.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        from pyscf.dft import numint2c, r_numint
        ni = mf._numint
        assert isinstance(ni, (numint2c.NumInt2C, r_numint.RNumInt))
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.get_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, 0, hermi,
                                rho0, vxc, fxc, max_memory=max_memory)
                if with_nlc and mf.do_nlc():
                    from pyscf.hessian.rks import get_vnlc_resp
                    nao = mo_coeff.shape[0] // 2
                    dm1_sf = dm1[...,:nao,:nao] + dm1[...,nao:,nao:]
                    mo_uks = [mo_coeff[:nao], mo_coeff[nao:]]
                    mo_occ_uks = [mo_occ, mo_occ]
                    # The get_vnlc_resp function uses mf._numint and explicitly
                    # calls the NumInt functions for mf._numint
                    ni1c = mf._numint._to_numint1c()
                    with lib.temporary_env(mf, _numint=ni1c):
                        vxc_nlc = get_vnlc_resp(mf, mol, mo_uks, mo_occ_uks,
                                                dm1_sf.real, max_memory)
                        v1[...,:nao,:nao] += vxc_nlc
                        v1[...,nao:,nao:] += vxc_nlc
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj
            else:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                if with_j:
                    v1 += vj - vk
                else:
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj - vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_dhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of DHF response function and
    DHF density matrices.
    '''
    return _gen_ghf_response(mf, mo_coeff, mo_occ, with_j, hermi, max_memory)


hf.RHF.gen_response = _gen_rhf_response
uhf.UHF.gen_response = _gen_uhf_response
ghf.GHF.gen_response = _gen_ghf_response
# Use UHF response function for ROHF because in second order solver uhf
# response function is called to compute ROHF orbital hessian
rohf.ROHF.gen_response = _gen_uhf_response
dhf.DHF.gen_response = _gen_dhf_response
