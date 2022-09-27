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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory
with density fitting
'''

import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo, df
from pyscf.agf2 import ragf2, mpi_helper, _agf2
from pyscf.agf2 import aux_space as aux

BLKMIN = getattr(__config__, 'agf2_blkmin', 100)


def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : GreensFunction
            Occupied Green's function
        gf_vir : GreensFunction
            Virtual Green's function

    Kwargs:
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0

    Returns:
        :class:`SelfEnergy`
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction

    nmo = eri.nmo
    nocc, nvir = gf_occ.naux, gf_vir.naux
    naux = agf2.with_df.get_naoaux()
    tol = agf2.weight_tol
    facs = dict(os_factor=os_factor, ss_factor=ss_factor)

    ei, ci = gf_occ.energy, gf_occ.coupling
    ea, ca = gf_vir.energy, gf_vir.coupling

    qxi, qja = _make_qmo_eris_incore(agf2, eri, (ci, ci, ca))

    himem_required = naux*(nvir+nmo) + (nocc*nvir)*(2*nmo+1) + (2*nmo**2)
    himem_required *= 8e-6
    himem_required *= lib.num_threads()

    if ((himem_required*1.05 + lib.current_memory()[0]) > agf2.max_memory
            and agf2.allow_lowmem_build) or agf2.allow_lowmem_build == 'force':
        log.debug('Thread-private memory overhead %.3f exceeds max_memory, using '
                  'low-memory version.', himem_required)
        vv, vev = _agf2.build_mats_dfragf2_lowmem(qxi, qja, ei, ea, **facs)
    else:
        vv, vev = _agf2.build_mats_dfragf2_incore(qxi, qja, ei, ea, **facs)

    e, c = _agf2.cholesky_build(vv, vev)
    se = aux.SelfEnergy(e, c, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = ragf2.get_frozen_mask(agf2)
        coupling = np.zeros((nmo, se.naux))
        coupling[mask] = se.coupling
        se = aux.SelfEnergy(se.energy, coupling, chempot=se.chempot)

    log.timer('se part', *cput0)

    return se

def get_jk(agf2, eri, rdm1, with_j=True, with_k=True):
    ''' Get the J/K matrices.

    Args:
        eri : ndarray or H5 dataset
            Electronic repulsion integrals (NOT as _ChemistsERIs). In
            the case of no bra/ket symmetry, a tuple can be passed.
        rdm1 : 2D array
            Reduced density matrix

    Kwargs:
        with_j : bool
            Whether to compute J. Default value is True
        with_k : bool
            Whether to compute K. Default value is True

    Returns:
        tuple of ndarrays corresponding to J and K, if either are
        not requested then they are set to None.
    '''

    nmo = rdm1.shape[0]
    npair = nmo*(nmo+1)//2
    naux = agf2.with_df.get_naoaux()
    vj = vk = None

    if with_j:
        rdm1_tril = lib.pack_tril(rdm1 + np.tril(rdm1, k=-1))
        vj = np.zeros((npair,))

    if with_k:
        vk = np.zeros((nmo, nmo))

    fdrv = ao2mo._ao2mo.libao2mo.AO2MOnr_e2_drv
    fmmm = ao2mo._ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    ftrans = ao2mo._ao2mo.libao2mo.AO2MOtranse2_nr_s2

    if isinstance(eri, tuple):
        bra, ket = eri
    else:
        bra = ket = eri

    blksize = _agf2.get_blksize(agf2.max_memory, (npair, npair, 1, nmo**2, nmo**2))
    blksize = min(nmo, max(BLKMIN, blksize))
    logger.debug1(agf2, 'blksize (dfragf2.get_jk) = %d' % blksize)
    buf = (np.empty((blksize, nmo, nmo)), np.empty((blksize, nmo, nmo)))

    for p0, p1 in mpi_helper.prange(0, naux, blksize):
        bra0 = bra[p0:p1]
        ket0 = ket[p0:p1]
        rho = np.dot(ket0, rdm1_tril)

        if with_j:
            vj += np.dot(rho, bra0)

        if with_k:
            buf1 = buf[0][:p1-p0]
            fdrv(ftrans, fmmm,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 bra0.ctypes.data_as(ctypes.c_void_p),
                 rdm1.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(p1-p0), ctypes.c_int(nmo),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 lib.c_null_ptr(), ctypes.c_int(0))

            buf2 = lib.unpack_tril(ket0, out=buf[1])
            buf1 = buf1.reshape(-1, nmo)
            buf2 = buf2.reshape(-1, nmo)

            vk = lib.dot(buf1.T, buf2, c=vk, beta=1)


    if with_j:
        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vj)
        mpi_helper.barrier()
        vj = lib.unpack_tril(vj)

    if with_k:
        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vk)

    return vj, vk


class DFRAGF2(ragf2.RAGF2):
    ''' Restricted AGF2 with canonical HF reference with density fitting

    Attributes:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        allow_lowmem_build : bool or str
            Allow the self-energy build to switch to a serially slower
            code with lower thread-private memory overhead if needed. One
            of True, False or 'force'. Default value is True.
        conv_tol : float
            Convergence threshold for AGF2 energy. Default value is 1e-7
        conv_tol_rdm1 : float
            Convergence threshold for first-order reduced density matrix.
            Default value is 1e-8.
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
        diis : bool or lib.diis.DIIS
            Whether to use DIIS, can also be a lib.diis.DIIS object. Default
            value is True.
        diis_space : int
            DIIS space size. Default value is 8.
        diis_min_space : int
            Minimum space of DIIS. Default value is 1.
        fock_diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        fock_diis_min_space :
            Minimum space of DIIS. Default value is 1.
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        damping : float
            Damping factor for the self-energy. Default value is 0.0

    Saved results

        e_corr : float
            AGF2 correlation energy
        e_tot : float
            Total energy (HF + correlation)
        e_1b : float
            One-body part of :attr:`e_tot`
        e_2b : float
            Two-body part of :attr:`e_tot`
        e_init : float
            Initial correlation energy (truncated MP2)
        converged : bool
            Whether convergence was successful
        se : SelfEnergy
            Auxiliaries of the self-energy
        gf : GreensFunction
            Auxiliaries of the Green's function
    '''

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        ragf2.RAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                             mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None) is not None:
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.allow_lowmem_build = True

        self._keys.update(['_with_df', 'allow_lowmem_build'])

    build_se_part = build_se_part
    get_jk = get_jk

    def ao2mo(self, mo_coeff=None):
        ''' Get the density-fitted electronic repulsion integrals in
            MO basis.
        '''

        eri = _make_mo_eris_incore(self, mo_coeff)

        return eri

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return ragf2.RAGF2.reset(self, mol)

    @property
    def with_df(self):
        return self._with_df
    @with_df.setter
    def with_df(self, val):
        self._with_df = val
        self._with_df.__class__ = DF


class DF(df.DF):
    ''' Replaces the :class:`DF.prange` function with one which
        natively supports MPI, if used.
    '''
    def prange(self, start=None, stop=None, step=None):
        if start is None: start = 0
        if stop is None: stop = self.get_naoaux()
        if step is None: step = self.blockdim

        for p0, p1 in mpi_helper.prange(start, stop, step):
            yield p0, p1


class _ChemistsERIs(ragf2._ChemistsERIs):
    ''' (pq|rs) as (pq|J)(J|rs)

    MO tensors stored in tril form, we only need QMO tensors
    in low-symmetry
    '''
    pass


def _make_mo_eris_incore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    with_df = agf2.with_df
    nmo = eris.fock.shape[0]
    npair = nmo*(nmo+1)//2
    naux = with_df.get_naoaux()

    qxy = np.zeros((naux, npair))
    mo = np.asarray(eris.mo_coeff, order='F')
    sij = (0, nmo, 0, nmo)
    sym = dict(aosym='s2', mosym='s2')

    for p0, p1 in with_df.prange():
        eri0 = with_df._cderi[p0:p1]
        qxy[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, mo, sij, out=qxy[p0:p1], **sym)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qxy)

    eris.eri = eris.qxy = qxy

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, coeffs):
    ''' Returns tuple of ndarray
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    cx = np.eye(agf2.nmo)
    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = ragf2.get_frozen_mask(agf2)
        cx = cx[:,mask]

    nmo = eri.fock.shape[0]
    npair = nmo*(nmo+1)//2
    with_df = agf2.with_df
    naux = with_df.get_naoaux()
    ci, cj, ca = coeffs

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

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qxi)
    mpi_helper.allreduce_safe_inplace(qja)

    log.timer('QMO integral transformation', *cput0)

    return (qxi, qja)



if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=3)
    rhf = scf.RHF(mol).density_fit()
    rhf.conv_tol = 1e-11
    rhf.run()

    ragf2 = DFRAGF2(rhf)

    ragf2.run()
    ragf2.ipagf2(nroots=5)
    ragf2.eaagf2(nroots=5)
