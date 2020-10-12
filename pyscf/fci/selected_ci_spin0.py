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

import ctypes
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import direct_spin1
from pyscf.fci import selected_ci

libfci = lib.load_library('libfci')

def contract_2e(eri, civec_strs, norb, nelec, link_index=None):
    ci_coeff, nelec, ci_strs = selected_ci._unpack(civec_strs, nelec)
    if link_index is None:
        link_index = selected_ci._all_linkstr_index(ci_strs, norb, nelec)
    cd_indexa, dd_indexa, cd_indexb, dd_indexb = link_index
    na, nlinka = nb, nlinkb = cd_indexa.shape[:2]

    eri = ao2mo.restore(1, eri, norb)
    eri1 = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    idx,idy = numpy.tril_indices(norb, -1)
    idx = idx * norb + idy
    eri1 = lib.take_2d(eri1.reshape(norb**2,-1), idx, idx) * 2
    lib.transpose_sum(eri1, inplace=True)
    eri1 *= .5
    fcivec = ci_coeff.reshape(na,nb)
    # (aa|aa)
    ci1 = numpy.zeros_like(fcivec)
    if nelec[0] > 1:
        ma, mlinka = mb, mlinkb = dd_indexa.shape[:2]
        libfci.SCIcontract_2e_aaaa(eri1.ctypes.data_as(ctypes.c_void_p),
                                   fcivec.ctypes.data_as(ctypes.c_void_p),
                                   ci1.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(norb),
                                   ctypes.c_int(na), ctypes.c_int(nb),
                                   ctypes.c_int(ma), ctypes.c_int(mlinka),
                                   dd_indexa.ctypes.data_as(ctypes.c_void_p))

    h_ps = numpy.einsum('pqqs->ps', eri) * (.5/nelec[0])
    eri1 = eri.copy()
    for k in range(norb):
        eri1[:,:,k,k] += h_ps
        eri1[k,k,:,:] += h_ps
    eri1 = ao2mo.restore(4, eri1, norb)
    lib.transpose_sum(eri1, inplace=True)
    eri1 *= .5
    # (bb|aa)
    libfci.SCIcontract_2e_bbaa(eri1.ctypes.data_as(ctypes.c_void_p),
                               fcivec.ctypes.data_as(ctypes.c_void_p),
                               ci1.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_int(norb),
                               ctypes.c_int(na), ctypes.c_int(nb),
                               ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                               cd_indexa.ctypes.data_as(ctypes.c_void_p),
                               cd_indexb.ctypes.data_as(ctypes.c_void_p))

    lib.transpose_sum(ci1, inplace=True)
    return selected_ci._as_SCIvector(ci1.reshape(ci_coeff.shape), ci_strs)

def enlarge_space(myci, civec_strs, eri, norb, nelec):
    if isinstance(civec_strs, (tuple, list)):
        nelec, (strsa, strsb) = selected_ci._unpack(civec_strs[0], nelec)[1:]
        ci_coeff = lib.asarray(civec_strs)
    else:
        ci_coeff, nelec, (strsa, strsb) = selected_ci._unpack(civec_strs, nelec)
    na = nb = len(strsa)
    ci0 = ci_coeff.reshape(-1,na,nb)
    abs_ci = abs(ci0).max(axis=0)

    eri = ao2mo.restore(1, eri, norb)
    eri_pq_max = abs(eri.reshape(norb**2,-1)).max(axis=1).reshape(norb,norb)

    civec_a_max = abs_ci.max(axis=1)
    ci_aidx = numpy.where(civec_a_max > myci.ci_coeff_cutoff)[0]
    civec_a_max = civec_a_max[ci_aidx]
    strsa = strsa[ci_aidx]
    strsa_add = selected_ci.select_strs(myci, eri, eri_pq_max, civec_a_max, strsa, norb, nelec[0])
    strsa = numpy.append(strsa, strsa_add)
    aidx = numpy.argsort(strsa)
    strsa = strsa[aidx]
    aidx = numpy.where(aidx < len(ci_aidx))[0]

    ci_bidx = ci_aidx
    strsb = strsa
    bidx = aidx

    ma = mb = len(strsa)
    cs = []
    for i in range(ci0.shape[0]):
        ci1 = numpy.zeros((ma,mb))
        tmp = lib.take_2d(ci0[i], ci_aidx, ci_bidx)
        lib.takebak_2d(ci1, tmp, aidx, bidx)
        cs.append(selected_ci._as_SCIvector(ci1, (strsa,strsb)))

    if ci_coeff[0].ndim == 0 or ci_coeff[0].shape[-1] != nb:
        cs = [c.ravel() for c in cs]

    if (isinstance(ci_coeff, numpy.ndarray) and
        ci_coeff.shape[0] == na or ci_coeff.shape[0] == na*nb):
        cs = cs[0]
    return cs


def make_hdiag(h1e, eri, ci_strs, norb, nelec):
    hdiag = selected_ci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
    na = len(ci_strs[0])
    lib.transpose_sum(hdiag.reshape(na,na), inplace=True)
    hdiag *= .5
    return hdiag

def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           select_cutoff=1e-3, ci_coeff_cutoff=1e-3, ecore=0, **kwargs):
    return direct_spin1._kfactory(SelectedCI, h1e, eri, norb, nelec, ci0,
                                  level_shift, tol, lindep, max_cycle,
                                  max_space, nroots, davidson_only,
                                  pspace_size, select_cutoff=select_cutoff,
                                  ci_coeff_cutoff=ci_coeff_cutoff, ecore=ecore,
                                  **kwargs)

make_rdm1s = selected_ci.make_rdm1s
make_rdm2s = selected_ci.make_rdm2s
make_rdm1 = selected_ci.make_rdm1
make_rdm2 = selected_ci.make_rdm2

trans_rdm1s = selected_ci.trans_rdm1s
trans_rdm1 = selected_ci.trans_rdm1


class SelectedCI(selected_ci.SelectedCI):
    def contract_2e(self, eri, civec_strs, norb, nelec, link_index=None, **kwargs):
# The argument civec_strs is a CI vector in function FCISolver.contract_2e.
# Save and patch self._strs to make this contract_2e function compatible to
# FCISolver.contract_2e.
        if getattr(civec_strs, '_strs', None) is not None:
            self._strs = civec_strs._strs
        else:
            assert(civec_strs.size == len(self._strs[0])*len(self._strs[1]))
            civec_strs = selected_ci._as_SCIvector(civec_strs, self._strs)
        return contract_2e(eri, civec_strs, norb, nelec, link_index)

    def make_hdiag(self, h1e, eri, ci_strs, norb, nelec):
        return make_hdiag(h1e, eri, ci_strs, norb, nelec)

    enlarge_space = enlarge_space

SCI = SelectedCI


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf.fci import direct_spin0

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
        ['H', ( 1., 2.    , 3.   )],
        ['H', ( 1., 2.    , 4.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    m = scf.RHF(mol)
    m.kernel()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1, c1 = kernel(h1e, eri, norb, nelec)
    e2, c2 = direct_spin0.kernel(h1e, eri, norb, nelec)
    print(e1, e1 - -11.894559902235565, 'diff to FCI', e1-e2)
