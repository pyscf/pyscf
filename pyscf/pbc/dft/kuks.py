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
Unrestricted Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.uks.py : PBC-UKS at a single k-point
'''


import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import khf, kuhf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks, krks
from pyscf.pbc.dft.krks import get_rho
from pyscf.pbc.dft import multigrid
from pyscf import __config__


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    if isinstance(ni, multigrid.MultiGridNumInt):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        j_in_xc = ni.xc_with_j
    else:
        ks.initialize_grids(cell, dm, kpts)
        j_in_xc = False

    max_memory = ks.max_memory - lib.current_memory()[0]
    n, exc, vxc = ni.nr_uks(cell, ks.grids, ks.xc, dm, 0, hermi,
                            kpts, kpts_band, max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    if ks.do_nlc():
        if ni.libxc.is_nlc(ks.xc):
            xc = ks.xc
        else:
            assert ni.libxc.is_nlc(ks.nlc)
            xc = ks.nlc
        n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm[0]+dm[1],
                                      0, hermi, kpts, max_memory=max_memory)
        exc += enlc
        vxc += vnlc
        logger.info(ks, 'nelec with nlc grids = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    ground_state = kpts_band is None
    nkpts = len(kpts)
    weight = 1. / nkpts
    vj, vk = krks._get_jk(ks, cell, dm, hermi, kpts, kpts_band, with_j=not j_in_xc)
    if j_in_xc:
        ecoul = vxc.ecoul
    else:
        vj = vj[0] + vj[1]
        vxc += vj
        ecoul = None
        if ground_state:
            ecoul = np.einsum('nKij,Kji->', dm, vj).real * .5 * weight
    if ni.libxc.is_hybrid_xc(ks.xc):
        vxc -= vk
        if ground_state:
            exc -= np.einsum('nKij,nKji->', dm, vk).real * .5 * weight
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    logger.timer(ks, 'veff', *t0)
    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = 1./len(h1e_kpts)
    e1 = weight *(np.einsum('kij,kji', h1e_kpts, dm_kpts[0]) +
                  np.einsum('kij,kji', h1e_kpts, dm_kpts[1]))
    ecoul = vhf.ecoul
    exc = vhf.exc
    tot_e = e1 + ecoul + exc
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exc.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
    if khf.CHECK_COULOMB_IMAG and abs(ecoul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    ecoul.imag)
    return tot_e.real, ecoul.real + exc.real

def gen_response(mf, mo_coeff=None, mo_occ=None,
                 with_j=True, hermi=0, max_memory=None, with_nlc=True):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    ni = mf._numint
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    j_in_xc = getattr(ni, 'xc_with_j', False)

    if with_nlc and mf.do_nlc():
        raise NotImplementedError

    rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, 1, kpts)
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1, kshift=0):
        if hermi == 2:
            v1 = np.zeros_like(dm1)
        else:
            assert kshift == 0
            v1 = ni.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                               rho0, vxc, fxc, kpts, max_memory=max_memory)
        vj, vk = krks._get_jk(mf, cell, dm1, hermi, kpts, with_j=not j_in_xc,
                              kshift=kshift)
        if not j_in_xc:
            v1 += vj[0] + vj[1]
        if hybrid:
            v1 -= vk
        return v1
    return vind

class KUKS(rks.KohnShamDFT, kuhf.KUHF):
    '''UKS class adapted for PBCs with k-point sampling (default: gamma point).
    '''

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = get_rho
    gen_response = gen_response

    def __init__(self, cell, kpts=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kuhf.KUHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kuhf.KUHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def Gradients(self):
        from pyscf.pbc.grad import kuks
        return kuks.Gradients(self)

    def to_hf(self):
        '''Convert to KUHF object.'''
        from pyscf.pbc import scf, df
        out = self._transfer_attrs_(scf.KUHF(self.cell, self.kpts))
        # Pure functionals only construct J-type integrals. Enable all integrals for KHF.
        if (not self._numint.libxc.is_hybrid_xc(self.xc) and
            len(self.kpts) > 1 and getattr(self.with_df, '_j_only', False)):
            out.with_df._j_only = False
            out.with_df.reset()
        return out

    multigrid_numint = krks.KRKS.multigrid_numint

    to_gpu = lib.to_gpu
