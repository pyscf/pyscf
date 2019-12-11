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
Generalized Kohn-Sham
'''

import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ghf
from pyscf.dft import rks


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for GKS.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.clock(), time.time())

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)

    assert(hermi == 1)
    dm = numpy.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_a = dm[...,:nao,:nao].real
    dm_b = dm[...,nao:,nao:].real

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm_a+dm_b, ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)
    if ks.nlc != '':
        if ks.nlcgrids.coords is None:
            ks.nlcgrids.build(with_non0tab=True)
            if ks.small_rho_cutoff > 1e-20 and ground_state:
                ks.nlcgrids = rks.prune_small_rho_grids_(ks, mol, dm_a+dm_b, ks.nlcgrids)
            t0 = logger.timer(ks, 'setting up nlc grids', *t0)

    max_memory = ks.max_memory - lib.current_memory()[0]
    ni = ks._numint
    n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, (dm_a,dm_b), max_memory=max_memory)
    if ks.nlc != '':
        assert('VV10' in ks.nlc.upper())
        _, enlc, vnlc = ni.nr_rks(mol, ks.nlcgrids, ks.xc+'__'+ks.nlc, dm_a+dm_b,
                                  max_memory=max_memory)
        exc += enlc
        vxc += vnlc
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)
    if vxc.ndim == 4:
        raise NotImplementedError
    vxc = numpy.asarray(scipy.linalg.block_diag(*vxc), dtype=dm.dtype)

    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc += vj
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, ddm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vj += vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(mol, dm, hermi, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= numpy.einsum('ij,ji', dm, vk).real * .5
    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


class GKS(ghf.GHF, rks.KohnShamDFT):
    '''Generalized Kohn-Sham'''
    def __init__(self, mol):
        ghf.GHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self)

    def dump_flags(self, verbose=None):
        ghf.GHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = rks.energy_elec

    def nuc_grad_method(self):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = 'H 0 0 0; H 0 0 1; O .5 .6 .2'
    mol.basis = 'ccpvdz'
    mol.build()

    mf = GKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    dm = mf.init_guess_by_1e(mol)
    dm = dm + 0j
    nao = mol.nao_nr()
    numpy.random.seed(12)
    dm[:nao,nao:] = numpy.random.random((nao,nao)) * .1j
    dm[nao:,:nao] = dm[:nao,nao:].T.conj()
    mf.kernel(dm)
    mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    mf.analyze()
    print(mf.spin_square())
    print(mf.e_tot - -76.2760115704274)
