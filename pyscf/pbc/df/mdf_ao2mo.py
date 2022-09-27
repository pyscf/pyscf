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

import numpy
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df import aft_ao2mo
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import gamma_point, unique
from pyscf import __config__


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):
    if mydf._cderi is None:
        mydf.build()

    kptijkl = _format_kpts(kpts)
    eri = aft_ao2mo.get_eri(mydf, kptijkl, compact=compact)
    eri += df_ao2mo.get_eri(mydf, kptijkl, compact=compact)
    return eri


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    if mydf._cderi is None:
        mydf.build()

    kptijkl = _format_kpts(kpts)
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    eri_mo = aft_ao2mo.general(mydf, mo_coeffs, kptijkl, compact=compact)
    eri_mo += df_ao2mo.general(mydf, mo_coeffs, kptijkl, compact=compact)
    return eri_mo

def ao2mo_7d(mydf, mo_coeff_kpts, kpts=None, factor=1, out=None):
    cell = mydf.cell
    if kpts is None:
        kpts = mydf.kpts
    nkpts = len(kpts)

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)

    # Shape of the orbitals can be different on different k-points. The
    # orbital coefficients must be formatted (padded by zeros) so that the
    # shape of the orbital coefficients are the same on all k-points. This can
    # be achieved by calling pbc.mp.kmp2.padded_mo_coeff function
    nmoi, nmoj, nmok, nmol = [x.shape[2] for x in mo_coeff_kpts]
    eri_shape = (nkpts, nkpts, nkpts, nmoi, nmoj, nmok, nmol)
    if gamma_point(kpts):
        dtype = numpy.result_type(*mo_coeff_kpts)
    else:
        dtype = numpy.complex128

    if out is None:
        out = numpy.empty(eri_shape, dtype=dtype)
    else:
        assert (out.shape == eri_shape)

    if mydf._cderi is None:
        mydf.build()

    kptij_lst = numpy.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:,0]
    kptjs_lst = kptij_lst[:,1]
    kpt_ji = kptjs_lst - kptis_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    ngrids = numpy.prod(mydf.mesh)
    nao = cell.nao_nr()
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0]-nao**4*16/1e6) * .5

    tao = []
    ao_loc = None
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    def process(uniq_id, kpt, fswap):
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_id)[0]

        kptjs = kptjs_lst[adapted_ji_idx]
        coulG = mydf.weighted_coulG(q, False, mydf.mesh)
        coulG *= factor

        moij_list = []
        ijslice_list = []
        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            moij, ijslice = _conc_mos(mo_coeff_kpts[0][ki], mo_coeff_kpts[1][kj])[2:]
            moij_list.append(moij)
            ijslice_list.append(ijslice)
            fswap.create_dataset('zij/'+str(ji), (ngrids,nmoi*nmoj), 'D')

        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, kptjs,
                                           max_memory=max_memory):
            for ji, aoao in enumerate(aoaoks):
                ki = adapted_ji_idx[ji] // nkpts
                kj = adapted_ji_idx[ji] %  nkpts
                buf = aoao.transpose(1,2,0).reshape(nao**2,p1-p0)
                zij = _ao2mo.r_e2(lib.transpose(buf), moij_list[ji],
                                  ijslice_list[ji], tao, ao_loc)
                zij *= coulG[p0:p1,None]
                fswap['zij/'+str(ji)][p0:p1] = zij

        mokl_list = []
        klslice_list = []
        for kk in range(nkpts):
            kl = kconserv[ki, kj, kk]
            mokl, klslice = _conc_mos(mo_coeff_kpts[2][kk], mo_coeff_kpts[3][kl])[2:]
            mokl_list.append(mokl)
            klslice_list.append(klslice)
            fswap.create_dataset('zkl/'+str(kk), (ngrids,nmok*nmol), 'D')

        ki = adapted_ji_idx[0] // nkpts
        kj = adapted_ji_idx[0] % nkpts
        kptls = kpts[kconserv[ki, kj, :]]
        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, -kptls,
                                           max_memory=max_memory):
            for kk, aoao in enumerate(aoaoks):
                buf = aoao.conj().transpose(1,2,0).reshape(nao**2,p1-p0)
                zkl = _ao2mo.r_e2(lib.transpose(buf), mokl_list[kk],
                                  klslice_list[kk], tao, ao_loc)
                fswap['zkl/'+str(kk)][p0:p1] = zkl

        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts

            moij, ijslice = _conc_mos(mo_coeff_kpts[0][ki], mo_coeff_kpts[1][kj])[2:]
            zij = []
            for LpqR, LpqI, sign in mydf.sr_loop(kpts[[ki,kj]], max_memory, False, mydf.blockdim):
                zij.append(_ao2mo.r_e2(LpqR+LpqI*1j, moij, ijslice, tao, ao_loc))

            for kk in range(nkpts):
                kl = kconserv[ki, kj, kk]
                eri_mo = lib.dot(numpy.asarray(fswap['zij/'+str(ji)]).T,
                                 numpy.asarray(fswap['zkl/'+str(kk)]))

                for i, (LrsR, LrsI, sign) in \
                        enumerate(mydf.sr_loop(kpts[[kk,kl]], max_memory, False, mydf.blockdim)):
                    zkl = _ao2mo.r_e2(LrsR+LrsI*1j, mokl_list[kk],
                                      klslice_list[kk], tao, ao_loc)
                    lib.dot(zij[i].T, zkl, sign*factor, eri_mo, 1)

                if dtype == numpy.double:
                    eri_mo = eri_mo.real
                out[ki,kj,kk] = eri_mo.reshape(eri_shape[3:])
        del (fswap['zij'])
        del (fswap['zkl'])

    with lib.H5TmpFile() as fswap:
        for uniq_id, kpt in enumerate(uniq_kpts):
            process(uniq_id, kpt, fswap)

    return out

