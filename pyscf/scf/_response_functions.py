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
Generate SCF response functions
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, rohf, uhf, ghf

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
    '''
    assert(not isinstance(mf, (uhf.UHF, rohf.ROHF)))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        from pyscf.dft import numint
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10

        # mf can be pbc.dft.RKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:
            # for ground state orbital hessian
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                mo_coeff, mo_occ, 0)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
        dm0 = None  #mf.make_rdm1(mo_coeff, mo_occ)

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
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
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
                      with_j=True, hermi=0, max_memory=None):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10

        # mf can be pbc.dft.UKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        #dm0 =(numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].T.conj()),
        #      numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].T.conj()))
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
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += vj[0] + vj[1] - vk
                else:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
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
                      with_j=True, hermi=0, max_memory=None):
    '''Generate a function to compute the product of GHF response function and
    GHF density matrices.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        raise NotImplementedError

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj - vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _is_dft_object(mf):
    return getattr(mf, 'xc', None) is not None and hasattr(mf, '_numint')


hf.RHF.gen_response = _gen_rhf_response
uhf.UHF.gen_response = _gen_uhf_response
ghf.GHF.gen_response = _gen_ghf_response
# Use UHF response function for ROHF because in second order solver uhf
# response function is called to compute ROHF orbital hessian
rohf.ROHF.gen_response = _gen_uhf_response
