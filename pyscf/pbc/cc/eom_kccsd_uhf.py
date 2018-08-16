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


########################################
# EOM-IP-CCSD
########################################

class EOMIP(eom_kgccsd.EOMIP):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMIP.__init__(self, cc)

def amplitudes_to_vector_ip(r1, r2):
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    return np.hstack((r1a, r1b,
                      r2aaa.ravel(), r2baa.ravel(),
                      r2abb.ravel(), r2bbb.ravel()))

def vector_to_amplitudes_ip(vector, nkpts, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nocca, noccb, nkpts**2*nocca*nocca*nvira, nkpts**2*noccb*nocca*nvira,
             nkpts**2*nocca*noccb*nvirb, nkpts**2*noccb*noccb*nvirb)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2aaa, r2baa, r2abb, r2bbb = np.split(vector, sections)

    r2aaa = r2aaa.reshape(nkpts,nkpts,nocca,nocca,nvira).copy()
    r2baa = r2baa.reshape(nkpts,nkpts,noccb,nocca,nvira).copy()
    r2abb = r2abb.reshape(nkpts,nkpts,nocca,noccb,nvirb).copy()
    r2bbb = r2bbb.reshape(nkpts,nkpts,noccb,noccb,nvirb).copy()

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2baa, r2abb, r2bbb)
    return r1, r2

########################################
# EOM-EA-CCSD
########################################

class EOMEA(eom_kgccsd.EOMEA):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMEA.__init__(self, cc)

def vector_to_amplitudes_ea(vector, nkpts, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    sizes = (nvira, nvirb, nkpts**2*nocca*nvira*nvira, nkpts**2*nocca*nvirb*nvira,
             nkpts**2*noccb*nvira*nvirb, nkpts**2*noccb*nvirb*nvirb)
    sections = np.cumsum(sizes[:-1])
    r1a, r1b, r2a, r2aba, r2bab, r2b = np.split(vector, sections)

    r2aaa = r2aaa.reshape(nocca,nvira,nvira).copy()
    r2aba = r2aba.reshape(nocca,nvirb,nvira).copy()
    r2bab = r2bab.reshape(noccb,nvira,nvirb).copy()
    r2bbb = r2bab.reshape(noccb,nvirb,nvirb).copy()

    r1 = (r1a.copy(), r1b.copy())
    r2 = (r2aaa, r2aba, r2bab, r2bbb)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    return np.hstack((r1a, r1b,
                      r2aaa.ravel(),
                      r2aba.ravel(), r2bab.ravel(),
                      r2bbb.ravel()))

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
    cell.mesh = [5, 5, 5]
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
    orbspin = kccsd_eris.orbspin

    nkpts = mycc.nkpts
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb

    kshift = 0  # excitation out of 0th k-point
    nmo = nmoa + nmob
    nocc = nocca + noccb
    nvir = nmo - nocc

    np.random.seed(0)
    # IP version
    myeom = EOMIP(mycc)
    imds = myeom.make_imds(eris=kccsd_eris, t1=spin_t1, t2=spin_t2)

    spin_r1_ip = (np.random.rand(nocc)*1j +
                  np.random.rand(nocc) - 0.5 - 0.5*1j)
    spin_r2_ip = (np.random.rand(nkpts**2 * nocc**2 * nvir) +
                  np.random.rand(nkpts**2 * nocc**2 * nvir)*1j - 0.5 - 0.5*1j)
    spin_r2_ip = spin_r2_ip.reshape(nkpts, nkpts, nocc, nocc, nvir)
    spin_r2_ip = eom_kgccsd.enforce_2p_spin_ip_doublet(spin_r2_ip, orbspin, kconserv, kshift)
    [r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb] = \
        spin2spatial_ip_doublet(spin_r1_ip, spin_r2_ip, orbspin, kconserv, kshift)

    r1, r2 = spatial2spin_ip_doublet([r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb], kconserv, kshift, orbspin=orbspin)

    # EA version
    myeom = EOMEA(mycc)
    imds = myeom.make_imds(eris=kccsd_eris, t1=spin_t1, t2=spin_t2)

    spin_r1_ea = (np.random.rand(nvir)*1j +
                  np.random.rand(nvir) - 0.5 - 0.5*1j)
    spin_r2_ea = (np.random.rand(nkpts**2 * nocc * nvir**2) +
                  np.random.rand(nkpts**2 * nocc * nvir**2)*1j - 0.5 - 0.5*1j)
    spin_r2_ea = spin_r2_ea.reshape(nkpts, nkpts, nocc, nvir, nvir)
    spin_r2_ea = eom_kgccsd.enforce_2p_spin_ea_doublet(spin_r2_ea, orbspin, kconserv, kshift)
    [r1a, r1b], [r2aaa, r2aba, r2bab, r2bbb] = \
        spin2spatial_ea_doublet(spin_r1_ea, spin_r2_ea, orbspin, kconserv, kshift)

    r1, r2 = spatial2spin_ea_doublet([r1a, r1b], [r2aaa, r2aba, r2bab, r2bbb], kconserv, kshift, orbspin=orbspin)
