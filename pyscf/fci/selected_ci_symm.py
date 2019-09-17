#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm
from pyscf.fci import selected_ci
from pyscf.fci import addons

libfci = lib.load_library('libfci')

def reorder4irrep(eri, norb, link_index, orbsym, offdiag=0):
    if orbsym is None:
        return eri, link_index, numpy.array(norb, dtype=numpy.int32)
    orbsym = numpy.asarray(orbsym)
# map irrep IDs of Dooh or Coov to D2h, C2v
# see symm.basis.linearmole_symm_descent
    orbsym = orbsym % 10
# irrep of (ij| pair
    trilirrep = (orbsym[:,None]^orbsym)[numpy.tril_indices(norb, offdiag)]
# and the number of occurence for each irrep
    dimirrep = numpy.array(numpy.bincount(trilirrep), dtype=numpy.int32)
# we sort the irreps of (ij| pair, to group the pairs which have same irreps
# "order" is irrep-id-sorted index. The (ij| paired is ordered that the
# pair-id given by order[0] comes first in the sorted pair
# "rank" is a sorted "order". Given nth (ij| pair, it returns the place(rank)
# of the sorted pair
    order = numpy.asarray(trilirrep.argsort(), dtype=numpy.int32)
    rank = numpy.asarray(order.argsort(), dtype=numpy.int32)
    eri = lib.take_2d(eri, order, order)
    link_index_irrep = link_index.copy()
    link_index_irrep[:,:,0] = rank[link_index[:,:,0]]
    return numpy.asarray(eri, order='C'), link_index_irrep, dimirrep

def contract_2e(eri, civec_strs, norb, nelec, link_index=None, orbsym=None):
    ci_coeff, nelec, ci_strs = selected_ci._unpack(civec_strs, nelec)
    if link_index is None:
        link_index = selected_ci._all_linkstr_index(ci_strs, norb, nelec)
    cd_indexa, dd_indexa, cd_indexb, dd_indexb = link_index
    na, nlinka = cd_indexa.shape[:2]
    nb, nlinkb = cd_indexb.shape[:2]

    eri = ao2mo.restore(1, eri, norb)
    eri1 = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    idx,idy = numpy.tril_indices(norb, -1)
    idx = idx * norb + idy
    eri1 = lib.take_2d(eri1.reshape(norb**2,-1), idx, idx) * 2
    eri1, dd_indexa, dimirrep = reorder4irrep(eri1, norb, dd_indexa, orbsym, -1)
    dd_indexb = reorder4irrep(eri1, norb, dd_indexb, orbsym, -1)[1]
    fcivec = ci_coeff.reshape(na,nb)
    # (bb|bb)
    if nelec[1] > 1:
        mb, mlinkb = dd_indexb.shape[:2]
        fcivecT = lib.transpose(fcivec)
        ci1T = numpy.zeros((nb,na))
        libfci.SCIcontract_2e_aaaa_symm(eri1.ctypes.data_as(ctypes.c_void_p),
                                        fcivecT.ctypes.data_as(ctypes.c_void_p),
                                        ci1T.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(norb),
                                        ctypes.c_int(nb), ctypes.c_int(na),
                                        ctypes.c_int(mb), ctypes.c_int(mlinkb),
                                        dd_indexb.ctypes.data_as(ctypes.c_void_p),
                                        dimirrep.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(len(dimirrep)))
        ci1 = lib.transpose(ci1T, out=fcivecT)
    else:
        ci1 = numpy.zeros_like(fcivec)
    # (aa|aa)
    if nelec[0] > 1:
        ma, mlinka = dd_indexa.shape[:2]
        libfci.SCIcontract_2e_aaaa_symm(eri1.ctypes.data_as(ctypes.c_void_p),
                                        fcivec.ctypes.data_as(ctypes.c_void_p),
                                        ci1.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(norb),
                                        ctypes.c_int(na), ctypes.c_int(nb),
                                        ctypes.c_int(ma), ctypes.c_int(mlinka),
                                        dd_indexa.ctypes.data_as(ctypes.c_void_p),
                                        dimirrep.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_int(len(dimirrep)))

    h_ps = numpy.einsum('pqqs->ps', eri)
    eri1 = eri * 2
    for k in range(norb):
        eri1[:,:,k,k] += h_ps/nelec[0]
        eri1[k,k,:,:] += h_ps/nelec[1]
    eri1 = ao2mo.restore(4, eri1, norb)
    eri1, cd_indexa, dimirrep = reorder4irrep(eri1, norb, cd_indexa, orbsym)
    cd_indexb = reorder4irrep(eri1, norb, cd_indexb, orbsym)[1]
    # (bb|aa)
    libfci.SCIcontract_2e_bbaa_symm(eri1.ctypes.data_as(ctypes.c_void_p),
                                    fcivec.ctypes.data_as(ctypes.c_void_p),
                                    ci1.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(norb),
                                    ctypes.c_int(na), ctypes.c_int(nb),
                                    ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                    cd_indexa.ctypes.data_as(ctypes.c_void_p),
                                    cd_indexb.ctypes.data_as(ctypes.c_void_p),
                                    dimirrep.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(len(dimirrep)))

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


class SelectedCI(selected_ci.SelectedCI):
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

    def get_init_guess(self, ci_strs, norb, nelec, nroots, hdiag):
        '''Initial guess is the single Slater determinant
        '''
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.orbsym,
                                              self.wfnsym)
        airreps = direct_spin1_symm._gen_strs_irrep(ci_strs[0], self.orbsym)
        birreps = direct_spin1_symm._gen_strs_irrep(ci_strs[1], self.orbsym)
        ci0 = direct_spin1_symm._get_init_guess(airreps, birreps,
                                                nroots, hdiag, self.orbsym, wfnsym)
        return [selected_ci._as_SCIvector(x, ci_strs) for x in ci0]

    def guess_wfnsym(self, norb, nelec, fcivec=None, orbsym=None, wfnsym=None,
                     **kwargs):
        if orbsym is None:
            orbsym = self.orbsym
        if fcivec is None:
            wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym,
                                                  wfnsym)
        else:
            strsa, strsb = getattr(fcivec, '_strs', self._strs)
            wfnsym = addons._guess_wfnsym(fcivec, strsa, strsb, orbsym)
        verbose = kwargs.get('verbose', None)
        log = logger.new_logger(self, verbose)
        log.debug('Guessing CI wfn symmetry = %s', wfnsym)
        return wfnsym

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()

        with lib.temporary_env(self, orbsym=orbsym, wfnsym=wfnsym):
            e, c = selected_ci.kernel_float_space(self, h1e, eri, norb, nelec, ci0,
                                                  tol, lindep, max_cycle, max_space,
                                                  nroots, davidson_only, ecore=ecore,
                                                  **kwargs)
            if wfnsym is not None:
                wfnsym0 = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)
                strsa, strsb = c._strs
                if nroots > 1:
                    for i, ci in enumerate(c):
                        ci = addons._symmetrize_wfn(ci, strsa, strsb, self.orbsym, wfnsym0)
                        c[i] = selected_ci._as_SCIvector(ci, c._strs)
                else:
                    ci = addons._symmetrize_wfn(c, strsa, strsb, self.orbsym, wfnsym0)
                    c = selected_ci._as_SCIvector(ci, c._strs)

        self.eci, self.ci = e, c
        return e, c

SCI = SelectedCI


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import symm
    from pyscf.fci import cistring

    norb, nelec = 7, (4,4)
    strs = cistring.gen_strings4orblist(range(norb), nelec[0])
    numpy.random.seed(11)
    mask = numpy.random.random(len(strs)) > .3
    strsa = strs[mask]
    mask = numpy.random.random(len(strs)) > .2
    strsb = strs[mask]
    ci_strs = (strsa, strsb)
    civec_strs = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
    orbsym = (numpy.random.random(norb) * 4).astype(int)
    nn = norb*(norb+1)//2
    eri = (numpy.random.random(nn*(nn+1)//2)-.2)**3

    ci0 = selected_ci.to_fci(civec_strs, norb, nelec)
    ci0 = addons.symmetrize_wfn(ci0, norb, nelec, orbsym)
    civec_strs = selected_ci.from_fci(ci0, civec_strs._strs, norb, nelec)
    e1 = numpy.dot(civec_strs.ravel(), contract_2e(eri, civec_strs, norb, nelec, orbsym=orbsym).ravel())
    e2 = numpy.dot(ci0.ravel(), direct_spin1_symm.contract_2e(eri, ci0, norb, nelec, orbsym=orbsym).ravel())
    print(e1-e2)

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

