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
Auxiliary second-order Green's function perturbation theory for
unrestricted references with density fitting
'''

import numpy as np
import ctypes
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo, df
from pyscf.agf2 import uagf2, dfragf2, mpi_helper, _agf2
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

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_occ[1]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction
    assert type(gf_vir[1]) is aux.GreensFunction

    nmoa, nmob = eri.nmo
    nocca, nvira = gf_occ[0].naux, gf_vir[0].naux
    noccb, nvirb = gf_occ[1].naux, gf_vir[1].naux
    naux = agf2.with_df.get_naoaux()
    tol = agf2.weight_tol
    facs = dict(os_factor=os_factor, ss_factor=ss_factor)

    ci_a, ei_a = gf_occ[0].coupling, gf_occ[0].energy
    ci_b, ei_b = gf_occ[1].coupling, gf_occ[1].energy
    ca_a, ea_a = gf_vir[0].coupling, gf_vir[0].energy
    ca_b, ea_b = gf_vir[1].coupling, gf_vir[1].energy

    qeri = _make_qmo_eris_incore(agf2, eri, (ci_a, ci_a, ca_a), (ci_b, ci_b, ca_b))
    (qxi_a, qja_a), (qxi_b, qja_b) = qeri
    qxi = (qxi_a, qxi_b)
    qja = (qja_a, qja_b)

    himem_required = naux*(nvira+nmoa) + (nocca*nvira+noccb*nvirb)*(1+2*nmoa) + (2*nmoa**2)
    himem_required *= 8e-6
    himem_required *= lib.num_threads()

    if ((himem_required*1.05 + lib.current_memory()[0]) > agf2.max_memory
            and agf2.allow_lowmem_build) or agf2.allow_lowmem_build == 'force':
        log.debug('Thread-private memory overhead %.3f exceeds max_memory, using '
                  'low-memory version.', himem_required)
        build_mats_dfuagf2 = _agf2.build_mats_dfuagf2_lowmem
    else:
        build_mats_dfuagf2 = _agf2.build_mats_dfuagf2_incore

    vv, vev = build_mats_dfuagf2(qxi, qja, (ei_a, ei_b), (ea_a, ea_b), **facs)
    e, c = _agf2.cholesky_build(vv, vev)
    se_a = aux.SelfEnergy(e, c, chempot=gf_occ[0].chempot)
    se_a.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = uagf2.get_frozen_mask(agf2)
        coupling = np.zeros((nmoa, se_a.naux))
        coupling[mask[0]] = se_a.coupling
        se_a = aux.SelfEnergy(se_a.energy, coupling, chempot=se_a.chempot)

    cput0 = log.timer('se part (alpha)', *cput0)

    himem_required = naux*(nvirb+nmob) + (noccb*nvirb+nocca*nvira)*(1+2*nmob) + (2*nmob**2)
    himem_required *= 8e-6
    himem_required *= lib.num_threads()

    if ((himem_required*1.05 + lib.current_memory()[0]) > agf2.max_memory
            and agf2.allow_lowmem_build) or agf2.allow_lowmem_build == 'force':
        log.debug('Thread-private memory overhead %.3f exceeds max_memory, using '
                  'low-memory version.', himem_required)
        build_mats_dfuagf2 = _agf2.build_mats_dfuagf2_lowmem
    else:
        build_mats_dfuagf2 = _agf2.build_mats_dfuagf2_incore

    rv = np.s_[::-1]
    vv, vev = build_mats_dfuagf2(qxi[rv], qja[rv], (ei_b, ei_a), (ea_b, ea_a), **facs)
    e, c = _agf2.cholesky_build(vv, vev)
    se_b = aux.SelfEnergy(e, c, chempot=gf_occ[1].chempot)
    se_b.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = uagf2.get_frozen_mask(agf2)
        coupling = np.zeros((nmoa, se_b.naux))
        coupling[mask[1]] = se_b.coupling
        se_b = aux.SelfEnergy(se_b.energy, coupling, chempot=se_b.chempot)

    cput0 = log.timer('se part (beta)', *cput0)

    return (se_a, se_b)


class DFUAGF2(uagf2.UAGF2):
    ''' Unrestricted AGF2 with canonical HF reference with density fitting

    Attributes:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        allow_lowmem_build : bool
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
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin
    '''

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        uagf2.UAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                             mo_coeff=mo_coeff, mo_occ=mo_occ)

        if getattr(mf, 'with_df', None) is not None:
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        self.allow_lowmem_build = True

        self._keys.update(['_with_df', 'allow_lowmem_build'])

    build_se_part = build_se_part
    get_jk = dfragf2.get_jk

    def ao2mo(self, mo_coeff=None):
        ''' Get the density-fitted electronic repulsion integrals in
            MO basis.
        '''

        eri = _make_mo_eris_incore(self, mo_coeff)

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

    cput0 = (logger.process_clock(), logger.perf_counter())
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

    for p0, p1 in with_df.prange():
        eri0 = with_df._cderi[p0:p1]
        qxy_a[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, moa, sija, out=qxy_a[p0:p1], **sym)
        qxy_b[p0:p1] = ao2mo._ao2mo.nr_e2(eri0, mob, sijb, out=qxy_b[p0:p1], **sym)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qxy_a)
    mpi_helper.allreduce_safe_inplace(qxy_b)

    eris.eri_a = qxy_a
    eris.eri_b = qxy_b
    eris.eri_aa = (eris.eri_a, eris.eri_a)
    eris.eri_ab = (eris.eri_a, eris.eri_b)
    eris.eri_ba = (eris.eri_b, eris.eri_a)
    eris.eri_bb = (eris.eri_b, eris.eri_b)
    eris.eri = (eris.eri_a, eris.eri_b)

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, coeffs_a, coeffs_b):
    ''' Returns nested tuple of ndarray
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    cxa, cxb = np.eye(agf2.nmo[0]), np.eye(agf2.nmo[1])
    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = uagf2.get_frozen_mask(agf2)
        cxa = cxa[:,mask[0]]
        cxb = cxb[:,mask[1]]

    nmoa, nmob = agf2.nmo
    npaira, npairb = nmoa*(nmoa+1)//2, nmob*(nmob+1)//2
    with_df = agf2.with_df
    naux = with_df.get_naoaux()
    cia, cja, caa = coeffs_a
    cib, cjb, cab = coeffs_b

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

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qxi_a)
    mpi_helper.allreduce_safe_inplace(qxi_b)
    mpi_helper.allreduce_safe_inplace(qja_a)
    mpi_helper.allreduce_safe_inplace(qja_b)

    qxi_a = qxi_a.reshape(naux, -1)
    qxi_b = qxi_b.reshape(naux, -1)
    qja_a = qja_a.reshape(naux, -1)
    qja_b = qja_b.reshape(naux, -1)

    log.timer('QMO integral transformation', *cput0)

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

