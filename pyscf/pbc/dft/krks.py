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
Non-relativistic Restricted Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : Non-relativistic Restricted Kohn-Sham for periodic
                           systems at a single k-point
'''


import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import khf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyscf import __config__


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : ``(nkpts, nao, nao)`` or ``(*, nkpts, nao, nao)`` ndarray
        Veff = J + Vxc.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.info(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 3 and
                    kpts_band is None)
    ks.initialize_grids(cell, dm, kpts, ground_state)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, dm, 0, hermi,
                                kpts, kpts_band, max_memory=max_memory)
        logger.info(ks, 'nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm, 0, hermi, kpts,
                                          max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.info(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    nkpts = len(kpts)
    weight = 1. / nkpts
    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
        if omega == 0:
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=-omega)
            vk *= hyb
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        elif hyb == 0: # SR=0, only LR exchange
            vk = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vk *= alpha
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk * .5

        if ground_state:
            exc -= np.einsum('Kij,Kji', dm, vk).real * .5 * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij,Kji', dm, vj) * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

@lib.with_doc(khf.get_rho.__doc__)
def get_rho(mf, dm=None, grids=None, kpts=None):
    if dm is None: dm = mf.make_rdm1()
    if grids is None: grids = mf.grids
    if kpts is None: kpts = mf.kpts
    if dm[0].ndim == 3:  # the KUKS density matrix
        dm = dm[0] + dm[1]
    if isinstance(mf.with_df, multigrid.MultiGridFFTDF):
        rho = mf.with_df.get_rho(dm, kpts)
    else:
        rho = mf._numint.get_rho(mf.cell, dm, grids, kpts, mf.max_memory)
    return rho

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = 1./len(h1e_kpts)
    e1 = weight * np.einsum('kij,kji', h1e_kpts, dm_kpts)
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

class KRKS(rks.KohnShamDFT, khf.KRHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = get_rho

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        khf.KRHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        khf.KRHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    def nuc_grad_method(self):
        from pyscf.pbc.grad import krks
        return krks.Gradients(self)

    def to_hf(self):
        '''Convert to KRHF object.'''
        from pyscf.pbc import scf, df
        out = self._transfer_attrs_(scf.KRHF(self.cell, self.kpts))
        # Pure functionals only construct J-type integrals. Enable all integrals for KHF.
        if (not self._numint.libxc.is_hybrid_xc(self.xc) and
            len(self.kpts) > 1 and getattr(self.with_df, '_j_only', False)):
            out.with_df._j_only = False
            out.with_df.reset()
        return out

    to_gpu = lib.to_gpu
