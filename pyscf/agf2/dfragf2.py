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
Auxiliary second-order Green's function perturbation theory
with density fitting
'''

import time
import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo, df
from pyscf.agf2 import ragf2, aux, mpi_helper
from pyscf.agf2.ragf2 import _get_blksize, _cholesky_build

BLKMIN = getattr(__config__, 'agf2_dfragf2_blkmin', 1)


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
    #NOTE: in my original implementation I transform qxi as ixq and do:
    #    xija = np.dot(ixq[i*nmo:(i+1)*nmo], qja, out=buf[0])
    #    xjia = np.dot(ixq, qja[:,i*nvir:(i+1)*nvir], out=buf[1])
    #    xjia = xjia.reshape(nocc, nmo, nvir).swapaxes(0,1).reshape(nmo, -1)
    # I need to benchmark these two versions, but this version makes
    # the code a lot clearer

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction

    nmo = agf2.nmo
    nocc, nvir = gf_occ.naux, gf_vir.naux
    naux = agf2.with_df.get_naoaux()
    tol = agf2.weight_tol

    mem_incore = (naux*(nmo*gf_occ.naux+gf_occ.naux*gf_vir.naux)) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mem_incore+mem_now < agf2.max_memory):
        qeri = _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir)
    else:
        qeri = _make_qmo_eris_outcore(agf2, eri, gf_occ, gf_vir)

    qxi, qja = qeri
    vv = np.zeros((nmo, nmo))
    vev = np.zeros((nmo, nmo))

    eja = lib.direct_sum('j,a->ja', gf_occ.energy, -gf_vir.energy)
    eja = eja.ravel()

    buf = (np.zeros((nmo, nocc*nvir)), np.zeros((nmo*nocc, nvir)))

    for i in mpi_helper.nrange(gf_occ.naux):
        qx = qxi.reshape(naux, nmo, nocc)[:,:,i]
        xija = lib.dot(qx.T, qja, c=buf[0])
        xjia = lib.dot(qxi.T, qja[:,i*nvir:(i+1)*nvir], c=buf[1])
        xjia = xjia.reshape(nmo, nocc*nvir)

        eija = eja + gf_occ.energy[i]

        vv = lib.dot(xija, xija.T, alpha=2, beta=1, c=vv)
        vv = lib.dot(xija, xjia.T, alpha=-1, beta=1, c=vv)

        exija = xija * eija[None]

        vev = lib.dot(exija, xija.T, alpha=2, beta=1, c=vev)
        vev = lib.dot(exija, xjia.T, alpha=-1, beta=1, c=vev)

    e, c = _cholesky_build(vv, vev, gf_occ, gf_vir)
    se = aux.SelfEnergy(e, c, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    log.timer_debug1('se part', *cput0)

    return se

def get_jk(agf2, eri, rdm1, with_j=True, with_k=True):
    ''' Get the J/K matrices.

    Args:
        eri : ndarray or H5 dataset
            Electronic repulsion integrals (NOT as _ChemistsERIs)
        rdm1 : 2D array
            Reduced density matrix

    Kwargs:
        with_j : bool
            Whether to compute J. Default value is True
        with_k : bool
            Whether to compute K. Default value is True

    Returns:
        tuple of ndarrays correspond to J and K, if either are
        not requested then they are set to None.
    '''

    nmo = agf2.nmo
    npair = nmo*(nmo+1)//2
    naux = agf2.with_df.get_naoaux()
    vj = vk = None

    if with_j:
        rdm1_tril = lib.pack_tril(rdm1 + np.tril(rdm1, k=-1))
        vj = np.zeros((npair,))
    else:
        vk = np.zeros((nmo, nmo))

    fdrv = ao2mo._ao2mo.libao2mo.AO2MOnr_e2_drv
    fmmm = ao2mo._ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    ftrans = ao2mo._ao2mo.libao2mo.AO2MOtranse2_nr_s2

    blksize = _get_blksize(agf2.max_memory, (npair, 1, nmo**2, nmo**2))
    blksize = min(nmo, max(BLKMIN, blksize))
    buf = np.empty((2, blksize, nmo, nmo))

    for p0, p1 in mpi_helper.prange(0, naux, blksize):
        eri1 = eri[p0:p1]
        rho = np.dot(eri1, rdm1_tril)

        if with_j:
            vj += np.dot(rho, eri1)

        if with_k:
            buf1 = buf[0,:p1-p0]
            fdrv(ftrans, fmmm, 
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 rdm1.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(p1-p0), ctypes.c_int(nmo),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 lib.c_null_ptr(), ctypes.c_int(0))

            buf2 = lib.unpack_tril(eri1, out=buf[1])
            buf1 = buf1.reshape(-1, nmo)
            buf2 = buf2.reshape(-1, nmo)

            vk = lib.dot(buf1.T, buf2, c=vk, beta=1)

    if with_j:
        vj = lib.unpack_tril(vj)
    
    return vj, vk


class DFRAGF2(ragf2.RAGF2):
    # Set with_df and add a .density_fit() method to parent method
    __doc__ = ragf2.RAGF2.__doc__

    conv_tol = getattr(__config__, 'agf2_dfragf2_DFRAGF2_conv_tol', 1e-7)
    conv_tol_rdm1 = getattr(__config__, 'agf2_dfragf2_DFRAGF2_conv_tol_rdm1', 1e-6)
    conv_tol_nelec = getattr(__config__, 'agf2_dfragf2_DFRAGF2_conv_tol_nelec', 1e-6)
    max_cycle = getattr(__config__, 'agf2_dfragf2_DFRAGF2_max_cycle', 50)
    max_cycle_outer = getattr(__config__, 'agf2_dfragf2_DFRAGF2_max_cycle_outer', 20)
    max_cycle_inner = getattr(__config__, 'agf2_dfragf2_DFRAGF2_max_cycle_inner', 50)
    weight_tol = getattr(__config__, 'agf2_dfragf2_DFRAGF2_weight_tol', 1e-11)
    diis_space = getattr(__config__, 'agf2_dfragf2_DFRAGF2_diis_space', 6)
    diis_min_space = getattr(__config__, 'agf2_dfragf2_DFRAGF2_diis_min_space', 1)

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        ragf2.RAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                             mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None) is not None:
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            #NOTE: how do we want to build this by default? this will also be the result of RAGF2.density_fit()
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self.with_df.prange = mpi_helper.prange

        self._keys.update(['with_df'])

    build_se_part = build_se_part
    get_jk = get_jk

    def ao2mo(self, mo_coeff=None):
        ''' Get the density-fitted electronic repulsion integrals in
            MO basis.
        '''

        mem_incore = (self.nmo*(self.nmo+1)//2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if mem_incore+mem_now < self.max_memory:
            eri = _make_mo_eris_incore(self)
        else:
            eri = _make_mo_eris_outcore(self)

        return eri

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return ragf2.RAGF2.reset(self, mol)


class _ChemistsERIs(ragf2._ChemistsERIs):
    ''' (pq|rs) as (pq|J)(J|rs)

    MO tensors stored in tril form, we only need QMO tensors
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
    nmo = eris.fock.shape[0]
    naux = with_df.get_naoaux()

    qxy = np.zeros_like(with_df._cderi)
    mo = np.asarray(eris.mo_coeff, order='F')
    sij = (0, nmo, 0, nmo)
    sym = dict(aosym='s2', mosym='s2')

    p1 = 0
    for eri0 in with_df.loop():
        p0, p1 = p1, p1 + eri0.shape[0]
        qxy[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, mo, sij, out=qxy[p0:p1], **sym)

    eris.eri = eris.qxy = qxy
    
    log.timer('MO integral transformation', *cput0)

    return eris

def _make_mo_eris_outcore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    with_df = agf2.with_df
    nmo = agf2.nmo
    npair = nmo*(nmo+1)//2
    naux = with_df.get_naoaux()

    qxy = eris.feri.create_dataset('mo', (naux, npair), 'f8')
    mo = np.asarray(eris.mo_coeff, order='F')
    sij = (0, nmo, 0, nmo)
    sym = dict(aosym='s2', mosym='s2')

    p1 = 0
    for eri0 in with_df.loop():
        p0, p1 = p1, p1 + eri0.shape[0]
        qxy[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, mo, sij, out=qxy[p0:p1], **sym)

    eris.eri = eris.feri['mo']

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir):
    ''' Returns tuple of ndarray
    '''
    
    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmo = eri.fock.shape[0]
    npair = nmo*(nmo+1)//2
    with_df = agf2.with_df
    naux = with_df.get_naoaux()

    cx = np.eye(nmo)
    ci = cj = gf_occ.coupling
    ca = gf_vir.coupling

    xisym, nxi, cxi, sxi = ao2mo.incore._conc_mos(cx, ci, compact=False)
    jasym, nja, cja, sja = ao2mo.incore._conc_mos(cj, ca, compact=False)
    sym = dict(aosym='s2', mosym='s1')

    qxi = np.zeros((naux, nxi))
    qja = np.zeros((naux, nja))
    buf = np.zeros((with_df.blockdim, npair))

    for p0, p1 in mpi_helper.prange(0, naux, with_df.blockdim):
        naux0 = p1 - p0
        buf0 = buf[:naux0]
        buf0[:] = eri.eri[p0:p1]

        qxi[p0:p1] = ao2mo._ao2mo.nr_e2(buf0, cxi, sxi, out=qxi[p0:p1], **sym)
        qja[p0:p1] = ao2mo._ao2mo.nr_e2(buf0, cja, sja, out=qja[p0:p1], **sym)

    qxi = qxi.reshape(naux, -1)
    qja = qja.reshape(naux, -1)

    log.timer_debug1('QMO integral transformation', *cput0)

    return (qxi, qja)

def _make_qmo_eris_outcore(agf2, eri, gf_occ, gf_vir):
    ''' Returns tuple of H5 dataset
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmo = eri.fock.shape[0]
    npair = nmo*(nmo+1)//2
    with_df = agf2.with_df
    naux = with_df.get_naoaux()

    cx = np.eye(nmo)
    ci = cj = gf_occ.coupling
    ca = gf_vir.coupling

    xisym, nxi, cxi, sxi = ao2mo.incore._conc_mos(cx, ci, compact=False)
    jasym, nja, cja, sja = ao2mo.incore._conc_mos(cj, ca, compact=False)
    sym = dict(aosym='s2', mosym='s1')

    # possible to have incore MO, outcore QMO
    if getattr(eri, 'feri', None) is None:
        eri.feri = lib.H5TmpFile()
    elif 'qmo/xi' in eri.feri:
        del eri.feri['qmo/xi']
        del eri.feri['qmo/ja']

    qxi = eri.feri.create_dataset('qmo/xi', (naux, nxi), 'f8')
    qja = eri.feri.create_dataset('qmo/ja', (naux, nja), 'f8')
    buf = np.zeros((with_df.blockdim, npair))

    for p0, p1 in mpi_helper.prange(0, naux, with_df.blockdim):
        naux0 = p1 - p0
        buf0 = buf[:naux0]

        if isinstance(eri.eri, ndarray):
            buf0[:] = eri.eri[p0:p1]
        else:
            eri.eri.read_direct(buf0, slice(p0, p1), slice(None))

        qxi[p0:p1] = ao2mo._ao2mo.nr_e2(buf0, cxi, sxi, out=qxi[p0:p1], **sym)
        qja[p0:p1] = ao2mo._ao2mo.nr_e2(buf0, cja, sja, out=qja[p0:p1], **sym)

    log.timer_debug1('QMO integral transformation', *cput0)

    return (qxi, qja)


if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=3)
    rhf = scf.RHF(mol).density_fit()
    rhf.conv_tol = 1e-11
    rhf.run()
    rhf.max_memory = 10

    ragf2 = DFRAGF2(rhf)
    ragf2.run()
    ragf2.ipragf2(nroots=5)
    ragf2.earagf2(nroots=5)
