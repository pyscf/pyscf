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
from pyscf.pbc.scf import khf, khf_ksymm, kuhf_ksymm
from pyscf.pbc.dft import gen_grid, multigrid
from pyscf.pbc.dft import rks, kuks

@lib.with_doc(kuks.get_veff.__doc__)
def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    if isinstance(kpts, np.ndarray):
        return kuks.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)

    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint

    # ndim = 4 : dm.shape = ([alpha,beta], nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and dm.shape[0] == 2 and kpts_band is None)

    if kpts_band is None:
        kpts_band = kpts.kpts_ibz

    if len(dm[0]) != kpts.nkpts_ibz:
        raise KeyError('Shape of the input density matrix does not '
                       'match the number of IBZ k-points: '
                       f'{len(dm[0])} vs {kpts.nkpts_ibz}.')
    dm_bz = kpts.transform_dm(dm)

    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm_bz, hermi,
                                       kpts.kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.info(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    ks.initialize_grids(cell, dm_bz, kpts.kpts, ground_state)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm_bz,
                                kpts=kpts.kpts, kpts_band=kpts_band,
                                max_memory=max_memory)
        logger.info(ks, 'nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm_bz[0]+dm_bz[1],
                                          0, hermi, kpts.kpts, max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.info(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    weight = kpts.weights_ibz

    if not hybrid:
        vj = ks.get_j(cell, dm[0]+dm[1], hermi, kpts, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if omega == 0:
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vj = vj[0] + vj[1]
        vxc += vj - vk

        if ground_state:
            exc -= (np.einsum('K,Kij,Kji', weight, dm[0], vk[0]) +
                    np.einsum('K,Kij,Kji', weight, dm[1], vk[1])).real * .5

    if ground_state:
        ecoul = np.einsum('K,Kij,Kji', weight, dm[0]+dm[1], vj) * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def get_rho(mf, dm=None, grids=None, kpts=None):
    from pyscf.pbc.dft import krks_ksymm
    if dm is None:
        dm = mf.make_rdm1()
    return krks_ksymm.get_rho(mf, dm[0]+dm[1], grids, kpts)


class KsymAdaptedKUKS(kuks.KUKS, kuhf_ksymm.KUHF):

    get_veff = get_veff
    get_rho = get_rho

    kpts = khf_ksymm.KsymAdaptedKSCF.kpts
    get_ovlp = khf_ksymm.KsymAdaptedKSCF.get_ovlp
    get_hcore = khf_ksymm.KsymAdaptedKSCF.get_hcore
    get_jk = khf_ksymm.KsymAdaptedKSCF.get_jk
    init_guess_by_chkfile = khf_ksymm.KsymAdaptedKSCF.init_guess_by_chkfile
    dump_chk = khf_ksymm.KsymAdaptedKSCF.dump_chk

    nelec = kuhf_ksymm.KUHF.nelec
    get_init_guess = kuhf_ksymm.KUHF.get_init_guess
    get_occ = kuhf_ksymm.KUHF.get_occ
    eig = kuhf_ksymm.KUHF.eig
    get_orbsym = kuhf_ksymm.KUHF.get_orbsym
    orbsym = kuhf_ksymm.KUHF.orbsym
    _finalize = kuhf_ksymm.KUHF._finalize

    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 **kwargs):
        kuhf_ksymm.KUHF.__init__(self, cell, kpts, exxdiv=exxdiv, **kwargs)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kuhf_ksymm.KUHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.cell, dm_kpts)

        weight = self.kpts.weights_ibz
        e1 = np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[0]+dm_kpts[1])
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
        from pyscf.pbc.scf.kuhf_ksymm import KUHF
        return self._transfer_attrs_(KUHF(self.cell, self.kpts))


KUKS = KsymAdaptedKUKS
