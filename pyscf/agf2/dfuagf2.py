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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory for
unrestricted references with density fitting
'''

import time
import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo, df
from pyscf.agf2 import uagf2, dfragf2, aux, mpi_helper
from pyscf.agf2.ragf2 import _get_blksize, _cholesky_build

BLKMIN = getattr(__config__, 'agf2_dfuragf2_blkmin', 1)


def build_se_part(agf2, eri, gf_occ, gf_vir):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : GreensFunction
            Occupied Green's function
        gf_vir : GreensFunction
            Virtual Green's function

    Returns:
        :class:`SelfEnergy`
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_occ[1]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction
    assert type(gf_vir[1]) is aux.GreensFunction

    nmoa, nmob = agf2.nmo
    nocca, nvira = gf_occ[0].naux, gf_vir[0].naux
    noccb, nvirb = gf_occ[1].naux, gf_vir[1].naux
    naux = agf2.with_df.get_naoaux()
    tol = agf2.weight_tol

    qeri = _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir)

    def _build_se_part_spin(spin=0):
        ''' Perform the build for a single spin

        spin = 0: alpha
        spin = 1: beta
        '''

        if spin == 0:
            ab = slice(None)
        else:
            ab = slice(None, None, -1)

        nmoa, nmob = agf2.nmo[ab]
        gfo_a, gfo_b = gf_occ[ab]
        gfv_a, gfv_b = gf_vir[ab]
        noa, nob = gfo_a.naux, gfo_b.naux
        nva, nvb = gfv_a.naux, gfv_b.naux

        (qxi_a, qja_a), (qxi_b, qja_b) = qeri[ab]

        vv = np.zeros((nmoa, nmoa))
        vev = np.zeros((nmoa, nmoa))

        eja_a = lib.direct_sum('j,a->ja', gfo_a.energy, -gfv_a.energy).ravel()
        eja_b = lib.direct_sum('j,a->ja', gfo_b.energy, -gfv_b.energy).ravel()

        buf = (np.zeros((nmoa, noa*nva)), 
               np.zeros((nmoa, nob*nvb)),
               np.zeros((nmoa*noa, nva))) 

        for i in mpi_helper.nrange(noa):
            qx_a = qxi_a.reshape(naux, nmoa, noa)[:,:,i]
            xija_aa = lib.dot(qx_a.T, qja_a, c=buf[0])
            xija_ab = lib.dot(qx_a.T, qja_b, c=buf[1])
            xjia_aa = lib.dot(qxi_a.T, qja_a[:,i*nva:(i+1)*nva], c=buf[2])
            xjia_aa = xjia_aa.reshape(nmoa, -1)

            eija_aa = eja_a + gfo_a.energy[i]
            eija_ab = eja_b + gfo_a.energy[i]

            vv = lib.dot(xija_aa, xija_aa.T, alpha=1, beta=1, c=vv)
            vv = lib.dot(xija_aa, xjia_aa.T, alpha=-1, beta=1, c=vv)
            vv = lib.dot(xija_ab, xija_ab.T, alpha=1, beta=1, c=vv)

            exija_aa = xija_aa * eija_aa[None]
            exija_ab = xija_ab * eija_ab[None]

            vev = lib.dot(exija_aa, xija_aa.T, alpha=1, beta=1, c=vev)
            vev = lib.dot(exija_aa, xjia_aa.T, alpha=-1, beta=1, c=vev)
            vev = lib.dot(exija_ab, xija_ab.T, alpha=1, beta=1, c=vev)

        e, c = _cholesky_build(vv, vev, gfo_a, gfv_a)
        se = aux.SelfEnergy(e, c, chempot=gfo_a.chempot)
        se.remove_uncoupled(tol=tol)

        return se

    se_a = _build_se_part_spin(0)

    cput0 = log.timer_debug1('se part (alpha)', *cput0)

    se_b = _build_se_part_spin(1)

    cput0 = log.timer_debug1('se part (beta)', *cput0)

    return (se_a, se_b)


class DFUAGF2(uagf2.UAGF2):
    #TODO: add .density_fit() to parent method
    ''' Unrestricted AGF2 with canonical HF reference with density fitting

    Attributes:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            Convergence threshold for AGF2 energy. Default value is 1e-7
        conv_tol_rdm1 : float
            Convergence threshold for first-order reduced density matrix.
            Default value is 1e-6.
        conv_tol_nelec : float
            Convergence threshold for the number of electrons. Default 
            value is 1e-6.
        max_cycle : int
            Maximum number of AGF2 iterations. Default value is 50.
        max_cycle_outer : int
            Maximum number of outer Fock loop iterations. Default 
            value is 20.
        max_cycle_inner : int
            Maximum number of inner Fock loop iterations. Default
            value is 50.
        weight_tol : float
            Threshold in spectral weight of auxiliaries to be considered
            zero. Default 1e-11.
        diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        diis_min_space : 
            Minimum space of DIIS. Default value is 1.

    Saved results

        e_corr : float
            AGF2 correlation energy
        e_tot : float
            Total energy (HF + correlation)
        e_1b : float
            One-body part of :attr:`e_tot`
        e_2b : float
            Two-body part of :attr:`e_tot`
        e_mp2 : float
            MP2 correlation energy
        converged : bool
            Whether convergence was successful
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin
        gf : tuple of GreensFunction 
            Auxiliaries of the Green's function for each spin
    '''

    conv_tol = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_conv_tol', 1e-7)
    conv_tol_rdm1 = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_conv_tol_rdm1', 1e-6)
    conv_tol_nelec = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_conv_tol_nelec', 1e-6)
    max_cycle = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_max_cycle', 50)
    max_cycle_outer = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_max_cycle_outer', 20)
    max_cycle_inner = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_max_cycle_inner', 50)
    weight_tol = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_weight_tol', 1e-11)
    diis_space = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_diis_space', 6)
    diis_min_space = getattr(__config__, 'agf2_dfuagf2_DFUAGF2_diis_min_space', 1)

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        uagf2.UAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                             mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None) is not None:
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self._keys.update(['_with_df'])

    build_se_part = build_se_part
    get_jk = dfragf2.get_jk

    def ao2mo(self, mo_coeff=None):
        ''' Get the density-fitted electronic repulsion integrals in
            MO basis.
        '''

        eri = _make_mo_eris_incore(self)

        return eri

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return uagf2.UAGF2.reset(self, mol)

    @property
    def with_df(self):
        return self._with_df
    @with_df.setter
    def with_df(self, val):
        self._with_df = val
        self._with_df.__class__ = dfragf2.DF


class _ChemistsERIs(uagf2._ChemistsERIs):
    ''' (pq|rs) as (pq|J)(J|rs)

    MO tensors are stored in tril from, we only need QMO tensors
    in low-symmetry
    '''
    pass

def _make_mo_eris_incore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    with_df = agf2.with_df
    moa, mob = eris.mo_coeff
    nmoa, nmob = moa.shape[1], mob.shape[1]
    npaira, npairb = nmoa*(nmoa+1)//2, nmob*(nmob+1)//2
    naux = with_df.get_naoaux()

    qxy_a = np.zeros((naux, npaira))
    qxy_b = np.zeros((naux, npairb))
    moa = np.asarray(moa, order='F')
    mob = np.asarray(mob, order='F')
    sija = (0, nmoa, 0, nmoa)
    sijb = (0, nmob, 0, nmob)
    sym = dict(aosym='s2', mosym='s2')

    p1 = 0
    for eri0 in with_df.loop():
        p0, p1 = p1, p1 + eri0.shape[0]
        qxy_a[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, moa, sija, out=qxy_a[p0:p1], **sym)
        qxy_b[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, mob, sijb, out=qxy_b[p0:p1], **sym)

    eris.eri_a = qxy_a
    eris.eri_b = qxy_b
    eris.eri_aa = (eris.eri_a, eris.eri_a)
    eris.eri_ab = (eris.eri_a, eris.eri_b)
    eris.eri_ba = (eris.eri_b, eris.eri_a)
    eris.eri_bb = (eris.eri_b, eris.eri_b)
    eris.eri = (eris.eri_a, eris.eri_b)

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir):
    ''' Returns nested tuple of ndarray
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmoa, nmob = agf2.nmo
    npaira, npairb = nmoa*(nmoa+1)//2, nmob*(nmob+1)//2
    with_df = agf2.with_df
    naux = with_df.get_naoaux()
    cxa, cxb = np.eye(nmoa), np.eye(nmob)
    cia = cja = gf_occ[0].coupling
    cib = cjb = gf_occ[1].coupling
    caa, cab = gf_vir[0].coupling, gf_vir[1].coupling

    xisym_a, nxi_a, cxi_a, sxi_a = ao2mo.incore._conc_mos(cxa, cia, compact=False)
    jasym_a, nja_a, cja_a, sja_a = ao2mo.incore._conc_mos(cja, caa, compact=False)
    xisym_b, nxi_b, cxi_b, sxi_b = ao2mo.incore._conc_mos(cxb, cib, compact=False)
    jasym_b, nja_b, cja_b, sja_b = ao2mo.incore._conc_mos(cjb, cab, compact=False)
    sym = dict(aosym='s2', mosym='s1')

    qxi_a = np.zeros((naux, nxi_a))
    qxi_b = np.zeros((naux, nxi_b))
    qja_a = np.zeros((naux, nja_a))
    qja_b = np.zeros((naux, nja_b))
    buf = (np.zeros((with_df.blockdim, npaira)), np.zeros((with_df.blockdim, npairb)))

    for p0, p1 in mpi_helper.prange(0, naux, with_df.blockdim):
        naux0 = p1 - p0
        bufa0 = buf[0][:naux0]
        bufb0 = buf[1][:naux0]
        bufa0[:] = eri.eri[0][p0:p1]
        bufb0[:] = eri.eri[1][p0:p1]

        qxi_a[p0:p1] = ao2mo._ao2mo.nr_e2(bufa0, cxi_a, sxi_a, out=qxi_a[p0:p1], **sym)
        qxi_b[p0:p1] = ao2mo._ao2mo.nr_e2(bufb0, cxi_b, sxi_b, out=qxi_b[p0:p1], **sym)
        qja_a[p0:p1] = ao2mo._ao2mo.nr_e2(bufa0, cja_a, sja_a, out=qja_a[p0:p1], **sym)
        qja_b[p0:p1] = ao2mo._ao2mo.nr_e2(bufb0, cja_b, sja_b, out=qja_b[p0:p1], **sym)

    qxi_a = qxi_a.reshape(naux, -1)
    qxi_b = qxi_b.reshape(naux, -1)
    qja_a = qja_a.reshape(naux, -1)
    qja_b = qja_b.reshape(naux, -1)

    qxi_a = np.array(qxi_a, order='F')
    qxi_b = np.array(qxi_b, order='F')
    qja_a = np.array(qja_a, order='F')
    qja_b = np.array(qja_b, order='F')

    log.timer_debug1('QMO integral transformation', *cput0)

    return ((qxi_a, qja_a), (qxi_b, qja_b))



if __name__ == '__main__':
    from pyscf import gto, scf, mp
    import pyscf.scf.stability

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', charge=-1, spin=1, verbose=3)
    uhf = scf.UHF(mol).density_fit()
    uhf.conv_tol = 1e-11
    uhf.run()

    for niter in range(1, 11):
        stability = scf.stability.uhf_stability(uhf)
        if isinstance(stability, tuple):
            sint, sext = stability
        else:
            sint = stability
        if np.allclose(sint, uhf.mo_coeff):
            break
        else:
            rdm1 = uhf.make_rdm1(sint, uhf.mo_occ)
        uhf.scf(dm0=rdm1)

    uagf2 = DFUAGF2(uhf)

    uagf2.run()
    uagf2.ipagf2(nroots=5)
    uagf2.eaagf2(nroots=5)

    print()
    keys = ['1b', '2b', 'mp2', 'corr', 'tot']
    print('  '.join(['%s %16.12f' % (key, getattr(uagf2, 'e_'+key, None)) for key in keys]))
