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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.scf import khf, khf_ksymm
from pyscf.pbc.dft import gen_grid, multigrid
from pyscf.pbc.dft import rks, krks

@lib.with_doc(krks.get_veff.__doc__)
def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    if isinstance(kpts, np.ndarray):
        return krks.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = kpts_band is None
    if kpts_band is None:
        kpts_band = kpts.kpts_ibz

    ni = ks._numint
    if isinstance(ni, multigrid.MultiGridNumInt):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        j_in_xc = ni.xc_with_j
    else:
        ks.initialize_grids(cell, dm, kpts)
        j_in_xc = False

    max_memory = ks.max_memory - lib.current_memory()[0]
    n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, dm, 0, hermi,
                            kpts=kpts, kpts_band=kpts_band,
                            max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    if ks.do_nlc():
        if ni.libxc.is_nlc(ks.xc):
            xc = ks.xc
        else:
            assert ni.libxc.is_nlc(ks.nlc)
            xc = ks.nlc
        n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm,
                                      0, hermi, kpts, max_memory=max_memory)
        exc += enlc
        vxc += vnlc
        logger.info(ks, 'nelec with nlc grids = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    weight = kpts.weights_ibz
    vj, vk = krks._get_jk(ks, cell, dm, hermi, kpts, kpts_band, with_j=not j_in_xc)
    if j_in_xc:
        ecoul = vxc.ecoul
    else:
        vxc += vj
        ecoul = None
        if ground_state:
            ecoul = np.einsum('K,Kij,Kji', weight, dm, vj) * .5
    if ni.libxc.is_hybrid_xc(ks.xc):
        vxc -= .5 * vk
        if ground_state:
            exc -= np.einsum('K,Kij,Kji', weight, dm, vk).real * .25
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    logger.timer(ks, 'veff', *t0)
    return vxc


class KsymAdaptedKRKS(krks.KRKS, khf_ksymm.KRHF):

    reset = khf_ksymm.KsymAdaptedKSCF.reset
    get_veff = get_veff

    kpts = khf_ksymm.KsymAdaptedKSCF.kpts
    kmesh = khf_ksymm.KsymAdaptedKSCF.kmesh
    get_ovlp = khf_ksymm.KsymAdaptedKSCF.get_ovlp
    get_hcore = khf_ksymm.KsymAdaptedKSCF.get_hcore
    get_jk = khf_ksymm.KsymAdaptedKSCF.get_jk
    get_occ = khf_ksymm.KsymAdaptedKSCF.get_occ
    init_guess_by_chkfile = khf_ksymm.KsymAdaptedKSCF.init_guess_by_chkfile
    dump_chk = khf_ksymm.KsymAdaptedKSCF.dump_chk
    eig = khf_ksymm.KsymAdaptedKSCF.eig
    get_orbsym = khf_ksymm.KsymAdaptedKSCF.get_orbsym
    orbsym = khf_ksymm.KsymAdaptedKSCF.orbsym
    _finalize = khf_ksymm.KsymAdaptedKSCF._finalize
    get_init_guess = khf_ksymm.KRHF.get_init_guess

    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 **kwargs):
        khf_ksymm.KRHF.__init__(self, cell, kpts, exxdiv=exxdiv, **kwargs)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        khf_ksymm.KRHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.cell, dm_kpts)

        weight = self.kpts.weights_ibz
        e1 = np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts)
        ecoul = vhf.ecoul
        exc = vhf.exc
        tot_e = e1 + ecoul + exc
        self.scf_summary['e1'] = e1.real
        self.scf_summary['coul'] = ecoul.real
        self.scf_summary['exc'] = exc.real
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
        if khf.CHECK_COULOMB_IMAG and abs(ecoul.imag) > self.cell.precision*10:
            logger.warn(self, "Coulomb energy has imaginary part %s. "
                        "Coulomb integrals (e-e, e-N) may not converge !",
                        ecoul.imag)
        return tot_e.real, ecoul.real + exc.real

    def to_hf(self):
        '''Convert to KRHF object.'''
        from pyscf.pbc.scf.khf_ksymm import KRHF
        return self._transfer_attrs_(KRHF(self.cell, self.kpts))

KRKS = KsymAdaptedKRKS
