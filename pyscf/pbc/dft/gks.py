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
# Authors: Chia-Nan Yeh <yehcanon@gmail.com>
#

'''
Generalized collinear Kohn-Sham in the spin-orbital basis for periodic systems at a single k-point

See Also:
    pyscf.pbc.dft.kgks.py : General spin-orbital Kohn-Sham for periodic
                            systems with k-point sampling
'''


import numpy
import scipy.linalg
import pyscf.dft
from pyscf import lib
from pyscf.pbc.scf import ghf as pbcghf
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf import __config__

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for GKS.'''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    if ks.nlc or ni.libxc.is_nlc(ks.xc):
        raise NotImplementedError(f'NLC functional {ks.xc} + {ks.nlc}')

    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    # TODO GKS with hybrid functional
    if hybrid:
        raise NotImplementedError

    # TODO GKS with multigrid method

    # ndim = 2, dm.shape = (2*nao, 2*nao)
    ground_state = (dm.ndim == 2 and kpts_band is None)

    assert (hermi == 1)
    dm = numpy.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_a = dm[..., :nao, :nao]
    dm_b = dm[..., nao:, nao:]

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_a+dm_b, ks.grids, kpt)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    # vxc = (vxc_aa, vxc_bb). vxc_ab is neglected in collinear DFT.
    max_memory = ks.max_memory - lib.current_memory()[0]
    n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, (dm_a,dm_b), 0,
                                    kpt, kpts_band, max_memory=max_memory)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)
    # vxc = (vxc_aa,   0   )
    #       (   0  , vxc_bb)
    vxc = numpy.asarray(scipy.linalg.block_diag(*vxc), dtype=dm.dtype)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
        vk *= hyb
        if omega != 0:
            vklr = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= numpy.einsum('ij,ji', dm, vk).real * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

class GKS(rks.KohnShamDFT, pbcghf.GHF):
    '''GKS class adapted for PBCs at a single k-point.

    This is a literal duplication of the molecular GKS class with some `mol`
    variables replaced by `cell`.
    '''
    def __init__(self, cell, kpt=numpy.zeros(3), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbcghf.GHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbcghf.GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = pyscf.dft.rks.energy_elec

    density_fit = rks._patch_df_beckegrids(pbcghf.GHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(pbcghf.GHF.mix_density_fit)

    def x2c1e(self):
        '''Adds spin-orbit coupling effects to H0 through the x2c1e approximation'''
        from pyscf.pbc.x2c.x2c1e import x2c1e_gscf
        return x2c1e_gscf(self)

    x2c = x2c1e

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
    mf = GKS(cell)
    print(mf.kernel())
