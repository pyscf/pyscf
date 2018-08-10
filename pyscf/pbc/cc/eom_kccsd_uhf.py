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
# Authors: James D. McClain
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#

import time
from functools import reduce
import itertools
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd
from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from pyscf.pbc.cc import kintermediates_uhf

einsum = lib.einsum

def enforce_2p_spin_doublet(r2, orbspin, kconserv, kshift, excitation):
    '''Enforces condition that net spin can only change by +/- 1/2'''
    assert(excitation in ['ip', 'ea'])
    if excitation == 'ip':
        nkpts, nocc, nvir = np.array(r2.shape)[[1, 3, 4]]
    elif excitation == 'ea':
        nkpts, nocc, nvir = np.array(r2.shape)[[1, 2, 3]]
    else:
        raise NotImplementedError

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]

    if excitation == 'ip':
        for ki, kj in itertools.product(range(nkpts), repeat=2):
            if ki > kj:  # Avoid double-counting of anti-symmetrization
                continue
            ka = kconserv[ki, kshift, kj]
            idxoaa = idxoa[ki][:,None] * nocc + idxoa[kj]
            idxoab = idxoa[ki][:,None] * nocc + idxob[kj]
            idxoba = idxob[ki][:,None] * nocc + idxoa[kj]
            idxobb = idxob[ki][:,None] * nocc + idxob[kj]

            r2_tmp = 0.5 * (r2[ki, kj] - r2[kj, ki].transpose(1, 0, 2))
            r2_tmp = r2_tmp.reshape(nocc**2, nvir)
            # Zero out states with +/- 3 unpaired spins
            r2_tmp[idxobb.ravel()[:, None], idxva[kshift]] = 0.0
            r2_tmp[idxoaa.ravel()[:, None], idxvb[kshift]] = 0.0

            r2[ki, kj] = r2_tmp.reshape(nocc, nocc, nvir)
            r2[kj, ki] = -r2[ki, kj].transpose(1, 0, 2)

        # Check...
        #
        #for ki, kj in itertools.product(range(nkpts), repeat=2):
        #    tmp = r2[ki, kj]
        #    print np.linalg.norm(tmp.imag), np.linalg.norm(tmp.real), \
        #          np.linalg.norm(tmp + r2[kj, ki].transpose(1, 0, 2))
    else:
        raise NotImplementedError
        for kj, ka in itertools.product(range(nkpts), repeat=2):
            ki = kconserv[kshift, kj, ka]
            idxvaa = idxva[ka][:,None] * nvir + idxva[kshift]
            idxvab = idxva[ka][:,None] * nvir + idxvb[kshift]
            idxvba = idxvb[ka][:,None] * nvir + idxva[kshift]
            idxvbb = idxvb[ka][:,None] * nvir + idxvb[kshift]

            # TODO
    return r2

########################################
# EOM-IP-CCSD
########################################

class EOMIP(eom_kgccsd.EOMIP):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMIP.__init__(self, cc)

def enforce_2p_spin_ip_doublet(r2, orbspin, kconserv, kshift):
    return enforce_2p_spin_doublet(r2, orbspin, kconserv, kshift, 'ip')

def spin2spatial_ip_doublet(r1, r2, orbspin, kconserv, kshift):
    nkpts, nocc, nvir = np.array(r2.shape)[[1, 3, 4]]

    idxoa = [np.where(orbspin[k][:nocc] == 0)[0] for k in range(nkpts)]
    idxob = [np.where(orbspin[k][:nocc] == 1)[0] for k in range(nkpts)]
    idxva = [np.where(orbspin[k][nocc:] == 0)[0] for k in range(nkpts)]
    idxvb = [np.where(orbspin[k][nocc:] == 1)[0] for k in range(nkpts)]
    nocc_a = len(idxoa[0])  # Assume nocc/nvir same for each k-point
    nocc_b = len(idxob[0])
    nvir_a = len(idxva[0])
    nvir_b = len(idxvb[0])

    r1a = r1[idxoa[kshift]]
    r1b = r1[idxob[kshift]]

    r2aaa = np.zeros((nkpts,nkpts,nkpts,nocc_a,nocc_a,nvir_a), dtype=r2.dtype)
    r2baa = np.zeros((nkpts,nkpts,nkpts,nocc_b,nocc_a,nvir_a), dtype=r2.dtype)
    r2abb = np.zeros((nkpts,nkpts,nkpts,nocc_a,nocc_b,nvir_b), dtype=r2.dtype)
    r2bbb = np.zeros((nkpts,nkpts,nkpts,nocc_b,nocc_b,nvir_b), dtype=r2.dtype)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        idxoaa = idxoa[ki][:,None] * nocc + idxoa[kj]
        idxoab = idxoa[ki][:,None] * nocc + idxob[kj]
        idxoba = idxob[ki][:,None] * nocc + idxoa[kj]
        idxobb = idxob[ki][:,None] * nocc + idxob[kj]

        r2_tmp = r2[ki, kj].reshape(nocc**2, nvir)
        r2aaa_tmp = lib.take_2d(r2_tmp, idxoaa.ravel(), idxva[ka])
        r2baa_tmp = lib.take_2d(r2_tmp, idxoba.ravel(), idxva[ka])
        r2abb_tmp = lib.take_2d(r2_tmp, idxoab.ravel(), idxvb[ka])
        r2bbb_tmp = lib.take_2d(r2_tmp, idxobb.ravel(), idxvb[ka])

        r2aaa[ki, kj] = r2aaa_tmp.reshape(nocc_a, nocc_a, nvir_a)
        r2baa[ki, kj] = r2baa_tmp.reshape(nocc_b, nocc_a, nvir_a)
        r2abb[ki, kj] = r2abb_tmp.reshape(nocc_a, nocc_b, nvir_b)
        r2bbb[ki, kj] = r2bbb_tmp.reshape(nocc_b, nocc_b, nvir_b)
    return [r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb]

def spatial2spin_ip_doublet(r1, r2, orbspin=None):
    '''Convert R1/R2 of spatial orbital representation to R1/R2 of
    spin-orbital representation
    '''
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    nkpts, nocc_a, nvir_a = np.array(r2aaa.shape)[[0, 1, 2]]
    nkpts, nocc_b, nvir_b = np.array(r2bbb.shape)[[0, 1, 2]]

    if orbspin is None:
        orbspin = np.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b

    r1 = np.zeros((nocc), dtype=r1a.dtype)
    r1[idxoa] = r1a
    r1[idxob] = r1b

    #r2 = np.zeros((nocc**2, nvir), dtype=r2aaa.dtype)
    #idxoaa = idxoa[:,None] * nocc + idxoa
    #idxoab = idxoa[:,None] * nocc + idxob
    #idxoba = idxob[:,None] * nocc + idxoa
    #idxobb = idxob[:,None] * nocc + idxob
    #idxvaa = idxva[:,None] * nvir + idxva
    #idxvab = idxva[:,None] * nvir + idxvb
    #idxvba = idxvb[:,None] * nvir + idxva
    #idxvbb = idxvb[:,None] * nvir + idxvb
    #r2aaa = r2aaa.reshape(nocc_a*nocc_a, nvir_a)
    #r2baa = r2baa.reshape(nocc_b*nocc_a, nvir_a)
    #r2abb = r2abb.reshape(nocc_a*nocc_b, nvir_b)
    #r2bbb = r2bbb.reshape(nocc_b*nocc_b, nvir_b)
    #lib.takebak_2d(r2, r2aaa, idxoaa.ravel(), idxva.ravel())
    #lib.takebak_2d(r2, r2baa, idxoba.ravel(), idxva.ravel())
    #lib.takebak_2d(r2, r2abb, idxoab.ravel(), idxvb.ravel())
    #lib.takebak_2d(r2, r2bbb, idxobb.ravel(), idxvb.ravel())
    #r2aba = -r2baa
    #r2bab = -r2abb
    #lib.takebak_2d(r2, r2aba, idxoab.T.ravel(), idxva.ravel())
    #lib.takebak_2d(r2, r2bab, idxoba.T.ravel(), idxvb.ravel())
    return r1, r2.reshape(nocc, nocc, nvir)

#def amplitudes_to_vector_ip(r1, r2):
#    '''For spin orbitals'''
#    r1a, r1b = r1
#    r2aaa, r2baa, r2abb, r2bbb = r2
#    nocca, noccb, nvirb = r2abb.shape
#    idxa = np.tril_indices(nocca, -1)
#    idxb = np.tril_indices(noccb, -1)
#    return np.hstack((r1a, r1b,
#                      r2aaa[idxa].ravel(), r2baa.ravel(),
#                      r2abb.ravel(), r2bbb[idxb].ravel()))
#
#def vector_to_amplitudes_ip(vector, nmo, nocc):
#    '''For spin orbitals'''
#    nocca, noccb = nocc
#    nmoa, nmob = nmo
#    nvira, nvirb = nmoa-nocca, nmob-noccb
#
#    sizes = (nocca, noccb, nocca*(nocca-1)//2*nvira, noccb*nocca*nvira,
#             nocca*noccb*nvirb, noccb*(noccb-1)//2*nvirb)
#    sections = np.cumsum(sizes[:-1])
#    r1a, r1b, r2a, r2baa, r2abb, r2b = np.split(vector, sections)
#    r2a = r2a.reshape(nocca*(nocca-1)//2,nvira)
#    r2b = r2b.reshape(noccb*(noccb-1)//2,nvirb)
#    r2baa = r2baa.reshape(noccb,nocca,nvira).copy()
#    r2abb = r2abb.reshape(nocca,noccb,nvirb).copy()
#
#    idxa = np.tril_indices(nocca, -1)
#    idxb = np.tril_indices(noccb, -1)
#    r2aaa = np.zeros((nocca,nocca,nvira), vector.dtype)
#    r2bbb = np.zeros((noccb,noccb,nvirb), vector.dtype)
#    r2aaa[idxa[0],idxa[1]] = r2a
#    r2aaa[idxa[1],idxa[0]] =-r2a
#    r2bbb[idxb[0],idxb[1]] = r2b
#    r2bbb[idxb[1],idxb[0]] =-r2b
#
#    r1 = (r1a.copy(), r1b.copy())
#    r2 = (r2aaa, r2baa, r2abb, r2bbb)
#    return r1, r2

########################################
# EOM-EA-CCSD
########################################

class EOMEA(eom_kgccsd.EOMEA):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMEA.__init__(self, cc)

def enforce_2p_spin_ea_doublet(r2, orbspin, kconserv, kshift):
    return enforce_2p_spin_doublet(r2, orbspin, kconserv, kshift, 'ea')

def spin2spatial_ea(r1, r2, orbspin):
    nocc, nvir = r2.shape[:2]

    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]
    nocc_a = len(idxoa)
    nocc_b = len(idxob)
    nvir_a = len(idxva)
    nvir_b = len(idxvb)

    r1a = r1[idxva]
    r1b = r1[idxvb]

    idxoaa = idxoa[:,None] * nocc + idxoa
    idxoab = idxoa[:,None] * nocc + idxob
    idxoba = idxob[:,None] * nocc + idxoa
    idxobb = idxob[:,None] * nocc + idxob
    idxvaa = idxva[:,None] * nvir + idxva
    idxvab = idxva[:,None] * nvir + idxvb
    idxvba = idxvb[:,None] * nvir + idxva
    idxvbb = idxvb[:,None] * nvir + idxvb

    r2 = r2.reshape(nocc, nvir**2)
    r2aaa = lib.take_2d(r2, idxoa.ravel(), idxvaa.ravel())
    r2aba = lib.take_2d(r2, idxoa.ravel(), idxvba.ravel())
    r2bab = lib.take_2d(r2, idxob.ravel(), idxvab.ravel())
    r2bbb = lib.take_2d(r2, idxob.ravel(), idxvbb.ravel())

    r2aaa = r2aaa.reshape(nocc_a, nvir_a, nvir_a)
    r2aba = r2aba.reshape(nocc_a, nvir_b, nvir_a)
    r2bab = r2bab.reshape(nocc_b, nvir_a, nvir_b)
    r2bbb = r2bbb.reshape(nocc_b, nvir_b, nvir_b)
    return [r1a, r1b], [r2aaa, r2aba, r2bab, r2bbb]

def spatial2spin_ea(r1, r2, orbspin=None):
    '''Convert R1/R2 of spatial orbital representation to R1/R2 of
    spin-orbital representation
    '''
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    nocc_a, nvir_a = r2aaa.shape[:2]
    nocc_b, nvir_b = r2bbb.shape[:2]

    if orbspin is None:
        orbspin = np.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    idxoa = np.where(orbspin[:nocc] == 0)[0]
    idxob = np.where(orbspin[:nocc] == 1)[0]
    idxva = np.where(orbspin[nocc:] == 0)[0]
    idxvb = np.where(orbspin[nocc:] == 1)[0]

    r1 = np.zeros((nvir), dtype=r1a.dtype)
    r1[idxva] = r1a
    r1[idxvb] = r1b

    r2 = np.zeros((nocc, nvir**2), dtype=r2aaa.dtype)
    idxoaa = idxoa[:,None] * nocc + idxoa
    idxoab = idxoa[:,None] * nocc + idxob
    idxoba = idxob[:,None] * nocc + idxoa
    idxobb = idxob[:,None] * nocc + idxob
    idxvaa = idxva[:,None] * nvir + idxva
    idxvab = idxva[:,None] * nvir + idxvb
    idxvba = idxvb[:,None] * nvir + idxva
    idxvbb = idxvb[:,None] * nvir + idxvb

    r2aaa = r2aaa.reshape(nocc_a, nvir_a*nvir_a)
    r2aba = r2aba.reshape(nocc_a, nvir_b*nvir_a)
    r2bab = r2bab.reshape(nocc_b, nvir_a*nvir_b)
    r2bbb = r2bbb.reshape(nocc_b, nvir_b*nvir_b)

    lib.takebak_2d(r2, r2aaa, idxoa.ravel(), idxvaa.ravel())
    lib.takebak_2d(r2, r2aba, idxoa.ravel(), idxvba.ravel())
    lib.takebak_2d(r2, r2bab, idxob.ravel(), idxvab.ravel())
    lib.takebak_2d(r2, r2bbb, idxob.ravel(), idxvbb.ravel())
    r2aab = -r2aba
    r2bba = -r2bab
    lib.takebak_2d(r2, r2bba, idxob.ravel(), idxvba.T.ravel())
    lib.takebak_2d(r2, r2aab, idxoa.ravel(), idxvab.T.ravel())
    r2 = r2.reshape(nocc, nvir, nvir)
    return r1, r2

#def vector_to_amplitudes_ea(vector, nmo, nocc):
#    nocca, noccb = nocc
#    nmoa, nmob = nmo
#    nvira, nvirb = nmoa-nocca, nmob-noccb
#
#    sizes = (nvira, nvirb, nocca*nvira*(nvira-1)//2, nocca*nvirb*nvira,
#             noccb*nvira*nvirb, noccb*nvirb*(nvirb-1)//2)
#    sections = np.cumsum(sizes[:-1])
#    r1a, r1b, r2a, r2aba, r2bab, r2b = np.split(vector, sections)
#    r2a = r2a.reshape(nocca,nvira*(nvira-1)//2)
#    r2b = r2b.reshape(noccb,nvirb*(nvirb-1)//2)
#    r2aba = r2aba.reshape(nocca,nvirb,nvira).copy()
#    r2bab = r2bab.reshape(noccb,nvira,nvirb).copy()
#
#    idxa = np.tril_indices(nvira, -1)
#    idxb = np.tril_indices(nvirb, -1)
#    r2aaa = np.zeros((nocca,nvira,nvira), vector.dtype)
#    r2bbb = np.zeros((noccb,nvirb,nvirb), vector.dtype)
#    r2aaa[:,idxa[0],idxa[1]] = r2a
#    r2aaa[:,idxa[1],idxa[0]] =-r2a
#    r2bbb[:,idxb[0],idxb[1]] = r2b
#    r2bbb[:,idxb[1],idxb[0]] =-r2b
#
#    r1 = (r1a.copy(), r1b.copy())
#    r2 = (r2aaa, r2aba, r2bab, r2bbb)
#    return r1, r2
#
#def amplitudes_to_vector_ea(r1, r2):
#    r1a, r1b = r1
#    r2aaa, r2aba, r2bab, r2bbb = r2
#    nocca, nvirb, nvira = r2aba.shape
#    idxa = np.tril_indices(nvira, -1)
#    idxb = np.tril_indices(nvirb, -1)
#    return np.hstack((r1a, r1b,
#                      r2aaa[:,idxa[0],idxa[1]].ravel(),
#                      r2aba.ravel(), r2bab.ravel(),
#                      r2bbb[:,idxb[0],idxb[1]].ravel()))

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    from pyscf import lo

    cell = gto.Cell()
    cell.atom='''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    #cell.basis = [[0, (1., 1.)], [1, (.5, 1.)]]
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.build()

    np.random.seed(1)
    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((2,3,nmo))
    kmf.mo_occ[0,:,:3] = 1
    kmf.mo_occ[1,:,:1] = 1
    kmf.mo_energy = np.arange(nmo) + np.random.random((2,3,nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2

    mo = (np.random.random((2,3,nmo,nmo)) +
          np.random.random((2,3,nmo,nmo))*1j - .5-.5j)
    s = kmf.get_ovlp()
    kmf.mo_coeff = np.empty_like(mo)
    nkpts = len(kmf.kpts)
    for k in range(nkpts):
        kmf.mo_coeff[0,k] = lo.orth.vec_lowdin(mo[0,k], s[k])
        kmf.mo_coeff[1,k] = lo.orth.vec_lowdin(mo[1,k], s[k])

    def rand_t1_t2(mycc):
        nkpts = mycc.nkpts
        nocca, noccb = mycc.nocc
        nmoa, nmob = mycc.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        np.random.seed(1)
        t1a = (np.random.random((nkpts,nocca,nvira)) +
               np.random.random((nkpts,nocca,nvira))*1j - .5-.5j)
        t1b = (np.random.random((nkpts,noccb,nvirb)) +
               np.random.random((nkpts,noccb,nvirb))*1j - .5-.5j)
        t2aa = (np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira))*1j - .5-.5j)
        kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
        t2aa = t2aa - t2aa.transpose(1,0,2,4,3,5,6)
        tmp = t2aa.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
        t2ab = (np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
        t2bb = (np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
                np.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
        t2bb = t2bb - t2bb.transpose(1,0,2,4,3,5,6)
        tmp = t2bb.copy()
        for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
            kl = kconserv[ki, kk, kj]
            t2bb[ki,kj,kk] = t2bb[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)

        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        return t1, t2

    import kccsd_uhf
    mycc = kccsd_uhf.KUCCSD(kmf)
    eris = mycc.ao2mo()

    t1, t2 = rand_t1_t2(mycc)
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)

    from pyscf.pbc.cc import kccsd
    kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(kmf))
    kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
    spin_t1 = kccsd.spatial2spin(t1, kccsd_eris.orbspin, kconserv)
    spin_t2 = kccsd.spatial2spin(t2, kccsd_eris.orbspin, kconserv)

    # EOM-EA
    myeom = EOMIP(mycc)
    imds = myeom.make_imds(eris=kccsd_eris, t1=spin_t1, t2=spin_t2)
    orbspin = kccsd_eris.orbspin

    np.random.seed(0)

    nkpts = mycc.nkpts
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    kshift = 0  # excitation out of 0th k-point
    nmo = nmoa + nmob
    nocc = nocca + noccb
    nvir = nmo - nocc
    spin_r1_ip = (np.random.rand(nvir)*1j +
                  np.random.rand(nvir) - 0.5 - 0.5*1j)
    spin_r2_ip = (np.random.rand(nkpts**2 * nocc**2 * nvir) +
                  np.random.rand(nkpts**2 * nocc**2 * nvir)*1j - 0.5 - 0.5*1j)
    spin_r2_ip = spin_r2_ip.reshape(nkpts, nkpts, nocc, nocc, nvir)
    spin_r2_ip = enforce_2p_spin_ip_doublet(spin_r2_ip, orbspin, kconserv, kshift)
    [r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb] = \
        spin2spatial_ip_doublet(spin_r1_ip, spin_r2_ip, orbspin, kconserv, kshift)
    #r1, r2 = spin2spatial_ea(r1, r2, orbspin)
