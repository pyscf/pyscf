#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
General spin-orbital Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : General spin-orbital Kohn-Sham for periodic
                           systems at a single k-point
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import kghf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyscf import __config__

def get_veff(ks, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional for KGKS

    Args:
        ks : an instance of :class:`GKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : (nkpts, 2*nao, 2*nao) or (*, nkpts, 2*nao, 2*nao) ndarray
        Veff = J + Vxc.
    '''
    if cell is None: cell = ks.cell
    if dm_kpts is None: dm_kpts = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    # TODO GKS with hybrid functional
    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10
    if hybrid:
        raise NotImplementedError

    # TODO GKS with multigrid method

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm_kpts, np.ndarray) and dm_kpts.ndim == 3 and
                    kpts_band is None)

    assert (hermi == 1)
    nso = dm_kpts.shape[-1]
    nao = nso // 2
    dm_a = dm_kpts[..., :nao, :nao]
    dm_b = dm_kpts[..., nao:, nao:]

    # For UniformGrids, grids.coords does not indicate whehter grids are initialized
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_a+dm_b, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    # vxc_spblk = ([alpha, beta], nkpts, nao, nao)
    n, exc, vxc_spblk = ks._numint.nr_uks(cell, ks.grids, ks.xc, (dm_a,dm_b), 0,
                                        kpts, kpts_band)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    # vxc = (vxc_aa,   0   )
    #       (   0  , vxc_bb)
    vxc = []
    for k in range(len(kpts)):
        vxc_k = np.asarray([vxc_spblk[0,k],vxc_spblk[1,k]], dtype=vxc_spblk.dtype)
        vxc_k = np.asarray(scipy.linalg.block_diag(*vxc_k))
        vxc.append(vxc_k)
    vxc = lib.asarray(vxc)

    weight = 1./len(kpts)
    if not hybrid:
        vk = None
        vj = ks.get_j(cell, dm_kpts, hermi, kpts, kpts_band)
        vxc += vj
    else:
        # TODO Hybrid functional
        raise NotImplementedError
        #if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
        #    ks.with_df._j_only = False
        #vj, vk = ks.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        #vk *= hyb
        #if abs(omega) > 1e-10:
        #    vklr = ks.get_k(cell, dm_kpts, hermi, kpts, kpts_band, omega=omega)
        #    vklr *= (alpha - hyb)
        #    vk += vklr
        #vxc += vj - vk * .5

        #if ground_state:
        #    exc -= np.einsum('Kij,Kji', dm_kpts, vk).real * .5 * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij,Kji', dm_kpts, vj).real * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

class KGKS(rks.KohnShamDFT, kghf.KGHF):
    '''GKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kghf.KGHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kghf.KGHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.cell, dm_kpts)

        weight = 1./len(h1e_kpts)
        e1 = weight * np.einsum('kij,kji', h1e_kpts, dm_kpts)
        tot_e = e1 + vhf.ecoul + vhf.exc
        self.scf_summary['e1'] = e1.real
        self.scf_summary['coul'] = vhf.ecoul.real
        self.scf_summary['exc'] = vhf.exc.real
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
        return tot_e.real, vhf.ecoul + vhf.exc

    # TODO Check if it is necessary
    #get_rho = None

    density_fit = rks._patch_df_beckegrids(kghf.KGHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(kghf.KGHF.mix_density_fit)

    stability = None
    def nuc_grad_method(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    mf = KGKS(cell, cell.make_kpts([2,1,1]))
    print(mf.kernel())
