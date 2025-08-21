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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic unrestricted Kohn-Sham for periodic systems at a single k-point

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''


import numpy
import pyscf.dft
from pyscf import lib
from pyscf.pbc.scf import uhf as pbcuhf
from pyscf.lib import logger
from pyscf.dft import uks as mol_uks
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyscf import __config__

get_rho = rks.get_rho

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    t0 = (logger.process_clock(), logger.perf_counter())

    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))

    ni = ks._numint
    if isinstance(ni, multigrid.MultiGridNumInt):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        j_in_xc = ni.xc_with_j
    else:
        ks.initialize_grids(cell, dm, kpt)
        j_in_xc = False

    max_memory = ks.max_memory - lib.current_memory()[0]
    n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm, 0, hermi,
                            kpt, kpts_band, max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    if ks.do_nlc():
        if ni.libxc.is_nlc(ks.xc):
            xc = ks.xc
        else:
            assert ni.libxc.is_nlc(ks.nlc)
            xc = ks.nlc
        n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm[0]+dm[1],
                                      0, hermi, kpt, max_memory=max_memory)
        exc += enlc
        vxc += vnlc
        logger.info(ks, 'nelec with nlc grids = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    ground_state = kpts_band is None
    vj, vk = rks._get_jk(ks, cell, dm, hermi, kpt, kpts_band, with_j=not j_in_xc)
    if j_in_xc:
        ecoul = vxc.ecoul
    else:
        vj = vj[0] + vj[1]
        vxc += vj
        ecoul = None
        if ground_state:
            ecoul = numpy.einsum('nij,ji->', dm, vj).real * .5
    if ni.libxc.is_hybrid_xc(ks.xc):
        vxc -= vk
        if ground_state:
            exc -= numpy.einsum('nij,nji->', dm, vk).real * .5
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    logger.timer(ks, 'veff', *t0)
    return vxc

def gen_response(mf, mo_coeff=None, mo_occ=None,
                 with_j=True, hermi=0, max_memory=None, with_nlc=True):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpt = mf.kpt
    ni = mf._numint
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    j_in_xc = getattr(ni, 'xc_with_j', False)

    if with_nlc and mf.do_nlc():
        raise NotImplementedError

    rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, 1, kpt)
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1):
        if hermi == 2:
            v1 = numpy.zeros_like(dm1)
        else:
            v1 = ni.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                               rho0, vxc, fxc, kpt, max_memory=max_memory)
        vj, vk = rks._get_jk(mf, cell, dm1, hermi, kpt, with_j=not j_in_xc)
        if not j_in_xc:
            v1 += vj[0] + vj[1]
        if hybrid:
            v1 -= vk
        return v1
    return vind

class UKS(rks.KohnShamDFT, pbcuhf.UHF):
    '''PBC-UKS at a single point (default: gamma point).

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''

    get_rho = get_rho
    get_vsap = mol_uks.UKS.get_vsap
    init_guess_by_vsap = mol_uks.UKS.init_guess_by_vsap
    get_veff = get_veff
    energy_elec = pyscf.dft.uks.energy_elec
    gen_response = gen_response

    def __init__(self, cell, kpt=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbcuhf.UHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbcuhf.UHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert to UHF object.'''
        from pyscf.pbc import scf
        return self._transfer_attrs_(scf.UHF(self.cell, self.kpt))

    def Gradients(self):
        from pyscf.pbc.grad import uks
        from pyscf.pbc.dft.multigrid import MultiGridNumInt2
        if not isinstance(self._numint, MultiGridNumInt2):
            raise NotImplementedError('pbc-UKS must be computed with MultiGridNumInt2')
        return uks.Gradients(self)

    multigrid_numint = rks.RKS.multigrid_numint

    to_gpu = lib.to_gpu
