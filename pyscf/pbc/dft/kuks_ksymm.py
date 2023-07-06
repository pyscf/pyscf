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
from pyscf.pbc.scf import kuhf_ksymm
from pyscf.pbc.dft import gen_grid, multigrid
from pyscf.pbc.dft import rks, kuks

@lib.with_doc(kuks.get_veff.__doc__)
def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    if isinstance(kpts, np.ndarray):
        return kuks.get_veff(ks, cell, dm, dm_last, vhf_last, hermi, kpts, kpts_band)
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    # ndim = 4 : dm.shape = ([alpha,beta], nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and dm.shape[0] == 2 and kpts_band is None)

    if kpts_band is None: kpts_band = kpts.kpts_ibz
    dm_bz = dm
    if ground_state:
        if len(dm[0]) != kpts.nkpts_ibz:
            raise RuntimeError("Number of input density matrices does not \
                               match the number of IBZ kpts: %d vs %d."
                               % (len(dm[0]), kpts.nkpts_ibz))
        dm_bz = kpts.transform_dm(dm)

    hybrid = ks._numint.libxc.is_hybrid_xc(ks.xc)

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm_bz, hermi,
                                       kpts.kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_bz, ks.grids, kpts.kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, dm_bz,
                                        kpts=kpts.kpts, kpts_band=kpts_band,
                                        max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    weight = kpts.weights_ibz

    if not hybrid:
        vj = ks.get_j(cell, dm[0]+dm[1], hermi, kpts, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            logger.warn(ks, 'df.j_only cannot be used with hybrid functional')
            ks.with_df._j_only = False
            # Rebuild df object due to the change of parameter _j_only
            if ks.with_df._cderi is not None:
                ks.with_df.build()
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vj = vj[0] + vj[1]
        vk *= hyb
        if omega != 0:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= (np.einsum('K,Kij,Kji', weight, dm[0], vk[0]) +
                    np.einsum('K,Kij,Kji', weight, dm[1], vk[1])).real * .5

    if ground_state:
        ecoul = np.einsum('K,Kij,Kji', weight, dm[0]+dm[1], vj).real * .5
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
    def __init__(self, cell, kpts=libkpts.KPoints(), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kuhf_ksymm.KUHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kuhf_ksymm.KUHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.cell, dm_kpts)

        weight = self.kpts.weights_ibz
        e1 = np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[0]+dm_kpts[1])
        tot_e = e1 + vhf.ecoul + vhf.exc
        self.scf_summary['e1'] = e1.real
        self.scf_summary['coul'] = vhf.ecoul.real
        self.scf_summary['exc'] = vhf.exc.real
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
        return tot_e.real, vhf.ecoul + vhf.exc

    get_rho = get_rho
    density_fit = rks._patch_df_beckegrids(kuhf_ksymm.KUHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(kuhf_ksymm.KUHF.mix_density_fit)


KUKS = KsymAdaptedKUKS
