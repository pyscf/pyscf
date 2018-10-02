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

########################################
# EOM-EA-CCSD
########################################

class EOMEA(eom_kgccsd.EOMEA):
    def __init__(self, cc):
        #if not isinstance(cc, kccsd.GCCSD):
        #    raise TypeError
        self.kpts = cc.kpts
        eom_kgccsd.EOMEA.__init__(self, cc)

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
    spin_t1 = kccsd.spatial2spin(t1, eris._kccsd_eris.orbspin, kconserv)
    spin_t2 = kccsd.spatial2spin(t2, eris._kccsd_eris.orbspin, kconserv)

    # EOM-EA
    myeom = EOMEA(mycc)

    imds = myeom.make_imds(eris=eris._kccsd_eris, t1=spin_t1, t2=spin_t2)
    #orbspin = imds.eris.orbspin
    #Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    #print(lib.finger(Ht1[0]) - (-1.2692088297292825-12.893074780897923j))
    #print(lib.finger(Ht1[1]) - (-11.831413366451148+19.95758532598137j ))
    #print(lib.finger(Ht2[0])*1e-2 - (0.97436765562779959 +0.16548728742427826j ))
    #print(lib.finger(Ht2[1])*1e-2 - (-1.7752605990115735 +4.2106261874056212j  ))
    #print(lib.finger(Ht2[2])*1e-3 - (-0.52223406190978494-0.91888685193234421j))

    #kmf.mo_occ[:] = 0
    #kmf.mo_occ[:,:,:2] = 1
    #mycc = KUCCSD(kmf)
    #eris = mycc.ao2mo()
    #t1, t2 = rand_t1_t2(mycc)
    #Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    #print(lib.finger(Ht1[0]) - (3.7571382837650931+3.6719235677672519j))
    #print(lib.finger(Ht1[1])*1e-2 - (-0.42270622344333642+0.65025799860663025j))
    #print(lib.finger(Ht2[0])*1e-2 - (2.5124103335695689  -1.3180553113575906j ))
    #print(lib.finger(Ht2[1])*1e-2 - (-2.4427382960124304 +0.15329780363467621j))
    #print(lib.finger(Ht2[2])*1e-2 - (3.0683780903085842  +2.580910132273615j  ))

    #from pyscf.pbc.cc import kccsd
    #kgcc = kccsd.GCCSD(scf.addons.convert_to_ghf(kmf))
    #kccsd_eris = kccsd._make_eris_incore(kgcc, kgcc._scf.mo_coeff)
    #r1 = kgcc.spatial2spin(t1)
    #r2 = kgcc.spatial2spin(t2)
    #ge = kccsd.energy(kgcc, r1, r2, kccsd_eris)
    #r1, r2 = kgcc.update_amps(r1, r2, kccsd_eris)
    #ue = energy(mycc, t1, t2, eris)
    #print(abs(ge - ue))
    #print(abs(r1 - kgcc.spatial2spin(Ht1)).max())
    #print(abs(r2 - kgcc.spatial2spin(Ht2)).max())

    #e2,t1,t2 = mycc.init_amps(eris)
    #ge2,gt1,gt2 = kgcc.init_amps(kccsd_eris)
    #r1 = kgcc.spatial2spin(t1)
    #r2 = kgcc.spatial2spin(t2)
    #print(abs(e2 - ge2))
    #print(abs(r1 - gt1).max())
    #print(abs(r2 - gt2).max())
    #exit()

    #kmf = kmf.density_fit(auxbasis=[[0, (1., 1.)], [0, (.5, 1.)]])
    #mycc = KUCCSD(kmf)
    #eris = _make_df_eris(mycc, mycc.mo_coeff)
    #t1, t2 = rand_t1_t2(mycc)
    #Ht1, Ht2 = mycc.update_amps(t1, t2, eris)
    #print(lib.finger(Ht1[0]) - (3.6569734813260473 +3.8092774902489754j))
    #print(lib.finger(Ht1[1]) - (-105.8651917884019 +219.86020519421155j))
    #print(lib.finger(Ht2[0]) - (-265.25767382882208+215.41888861285341j))
    #print(lib.finger(Ht2[1]) - (-115.13953446128346-49.303887916188629j))
    #print(lib.finger(Ht2[2]) - (122.51835547779413 +33.85757422327751j ))
