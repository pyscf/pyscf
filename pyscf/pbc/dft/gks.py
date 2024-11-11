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
from pyscf.pbc.dft import multigrid
from pyscf.pbc.dft.numint2c import NumInt2C
from pyscf.dft import gks as mol_ks
from pyscf import __config__


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for GKS.'''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    if ks.do_nlc():
        raise NotImplementedError(f'NLC functional {ks.xc} + {ks.nlc}')

    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    # TODO GKS with hybrid functional
    if hybrid:
        raise NotImplementedError

    # TODO GKS with multigrid method
    if isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        raise NotImplementedError

    # ndim = 2, dm.shape = (2*nao, 2*nao)
    ground_state = (dm.ndim == 2 and kpts_band is None)
    ks.initialize_grids(cell, dm, kpt, ground_state)

    # TODO: support non-symmetric density matrix
    assert (hermi == 1)
    dm = numpy.asarray(dm)

    # ndim = 2, dm.shape = (2*nao, 2*nao)
    ground_state = (dm.ndim == 2 and kpts_band is None)

    # vxc = (vxc_aa, vxc_bb). vxc_ab is neglected in collinear DFT.
    max_memory = ks.max_memory - lib.current_memory()[0]
    ni = ks._numint
    n, exc, vxc = ni.get_vxc(cell, ks.grids, ks.xc, dm, hermi=hermi, kpt=kpt,
                             kpts_band=kpts_band, max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if omega == 0:
            vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(cell, dm, hermi, kpt, kpts_band)
            vk = ks.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpts_band)
            vk *= hyb
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

    collinear = mol_ks.GKS.collinear
    spin_samples = mol_ks.GKS.spin_samples
    get_veff = get_veff
    energy_elec = mol_ks.energy_elec

    def __init__(self, cell, kpt=numpy.zeros(3), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbcghf.GHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = NumInt2C()

    def dump_flags(self, verbose=None):
        pbcghf.GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def x2c1e(self):
        '''Adds spin-orbit coupling effects to H0 through the x2c1e approximation'''
        from pyscf.pbc.x2c.x2c1e import x2c1e_gscf
        return x2c1e_gscf(self)
    x2c = x2c1e

    def stability(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    def to_hf(self):
        '''Convert to GHF object.'''
        from pyscf.pbc import scf
        return self._transfer_attrs_(scf.GHF(self.cell, self.kpt))

    to_gpu = lib.to_gpu
