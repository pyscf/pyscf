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
from pyscf.fci import direct_spin1_symm
from pyscf.fci import selected_ci
from pyscf.fci import selected_ci_symm
from pyscf.fci import selected_ci_spin0

libfci = lib.load_library('libfci')

def contract_2e(eri, civec_strs, norb, nelec, link_index=None, orbsym=None):
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
    eri1, dd_indexa, dimirrep = selected_ci_symm.reorder4irrep(eri1, norb, dd_indexa, orbsym, -1)
    fcivec = ci_coeff.reshape(na,nb)
    ci1 = numpy.zeros_like(fcivec)
    # (aa|aa)
    if nelec[0] > 1:
        ma, mlinka = mb, mlinkb = dd_indexa.shape[:2]
        libfci.SCIcontract_2e_aaaa_symm(eri1.ctypes.data_as(ctypes.c_void_p),
                                        fcivec.ctypes.data_as(ctypes.c_void_p),
                                        ci1.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(norb),
                                        ctypes.c_int(na), ctypes.c_int(nb),
                                        ctypes.c_int(ma), ctypes.c_int(mlinka),
                                        dd_indexa.ctypes.data_as(ctypes.c_void_p),
                                        dimirrep.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(len(dimirrep)))

    h_ps = numpy.einsum('pqqs->ps', eri) * (.5/nelec[0])
    eri1 = eri.copy()
    for k in range(norb):
        eri1[:,:,k,k] += h_ps
        eri1[k,k,:,:] += h_ps
    eri1 = ao2mo.restore(4, eri1, norb)
    lib.transpose_sum(eri1, inplace=True)
    eri1 *= .5
    eri1, cd_indexa, dimirrep = selected_ci_symm.reorder4irrep(eri1, norb, cd_indexa, orbsym)
    # (bb|aa)
    libfci.SCIcontract_2e_bbaa_symm(eri1.ctypes.data_as(ctypes.c_void_p),
                                    fcivec.ctypes.data_as(ctypes.c_void_p),
                                    ci1.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(norb),
                                    ctypes.c_int(na), ctypes.c_int(nb),
                                    ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                    cd_indexa.ctypes.data_as(ctypes.c_void_p),
                                    cd_indexa.ctypes.data_as(ctypes.c_void_p),
                                    dimirrep.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(len(dimirrep)))

    lib.transpose_sum(ci1, inplace=True)
    return selected_ci._as_SCIvector(ci1.reshape(ci_coeff.shape), ci_strs)

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


class SelectedCI(selected_ci_symm.SelectedCI):
    def contract_2e(self, eri, civec_strs, norb, nelec, link_index=None,
                    orbsym=None, **kwargs):
        if orbsym is None:
            orbsym = self.orbsym
        if getattr(civec_strs, '_strs', None) is not None:
            self._strs = civec_strs._strs
        else:
            assert(civec_strs.size == len(self._strs[0])*len(self._strs[1]))
            civec_strs = selected_ci._as_SCIvector(civec_strs, self._strs)
        return contract_2e(eri, civec_strs, norb, nelec, link_index, orbsym)

    def make_hdiag(self, h1e, eri, ci_strs, norb, nelec):
        return selected_ci_spin0.make_hdiag(h1e, eri, ci_strs, norb, nelec)

    enlarge_space = selected_ci_spin0.enlarge_space

SCI = SelectedCI


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import symm

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = 'sto-3g'
    mol.symmetry = 1
    mol.build()
    m = scf.RHF(mol).run()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
    eri = ao2mo.incore.full(m._eri, m.mo_coeff)
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)

    myci = SelectedCI().set(orbsym=orbsym)
    e1, c1 = myci.kernel(h1e, eri, norb, nelec)
    myci = direct_spin1_symm.FCISolver().set(orbsym=orbsym)
    e2, c2 = myci.kernel(h1e, eri, norb, nelec)
    print(e1 - e2)

