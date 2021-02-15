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
Generate KSCF response functions
'''

import numpy
from pyscf import lib
from pyscf.pbc.scf import khf, kuhf, krohf, kghf

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    from pyscf.pbc.dft import numint, multigrid
    assert(isinstance(mf, khf.KRHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = abs(hyb) > 1e-10
        if abs(omega) > 1e-10:  # For range separated Coulomb
            raise NotImplementedError

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:  # for newton solver
            rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc, mo_coeff,
                                                mo_occ, 0, kpts)
        else:
            if isinstance(mo_occ, numpy.ndarray):
                mo_occ = mo_occ*.5
            else:
                mo_occ = [x*.5 for x in mo_occ]
            rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ]*2,
                                                spin=1, kpts=kpts)
        dm0 = None #mf.make_rdm1(mo_coeff, mo_occ)

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
                                       rho0, vxc, fxc, kpts, max_memory=max_memory)
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                        v1 += vj - .5 * hyb * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                        v1 += vj - .5 * hyb * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    v1 += -.5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)

    return vind

def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    from pyscf.pbc.dft import multigrid
    assert(isinstance(mf, (kuhf.KUHF, krohf.KROHF)))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = abs(hyb) > 1e-10
        if abs(omega) > 1e-10:  # For range separated Coulomb
            raise NotImplementedError

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1, kpts)
        #dm0 =(numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].conj().T),
        #      numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].conj().T))
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, kpts, max_memory=max_memory)
            if not hybrid:
                if with_j:
                    vj = mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                    v1 += vj[0] + vj[1] - vk * hyb
                else:
                    v1 -= hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)

    return vind

def _gen_ghf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    '''Generate a function to compute the product of KGHF response function and
    KGHF density matrices.
    '''
    raise NotImplementedError


khf.KRHF.gen_response = _gen_rhf_response
kuhf.KUHF.gen_response = _gen_uhf_response
kghf.KGHF.gen_response = _gen_ghf_response
krohf.KROHF.gen_response = _gen_uhf_response

