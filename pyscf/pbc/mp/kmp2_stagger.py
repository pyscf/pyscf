#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

'''
k-point spin-restricted periodic MP2 calculation using the staggered mesh method
Author: Xin Xing (xxing@berkeley.edu)
Reference: Staggered Mesh Method for Correlation Energy Calculations of Solids: Second-Order
        Møller–Plesset Perturbation Theory, J. Chem. Theory Comput. 2021, 17, 8, 4733-4745
'''

import h5py
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import df, dft, scf
from pyscf.pbc.scf.khf import get_occ
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp.kmp2 import padding_k_idx, _add_padding
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import unique, round_to_fbz
from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
from pyscf.pbc.df.df import CDERIArray

#   Kernel function for computing the MP2 energy
def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE):
    """Computes k-point RMP2 energy using the staggered mesh method

    Args:
        mp (KMP2_stagger): an instance of KMP2_stagger
        mo_energy (list): a list of numpy.ndarray. Each array contains MO energies of
                          shape (Nmo,) for one kpt
        mo_coeff (list): a list of numpy.ndarray. Each array contains MO coefficients
                         of shape (Nao, Nmo) for one kpt
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).

    Returns:
        KMP2 energy
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    nkpts_ov = mp.nkpts_ov
    kconserv = mp.khelper.kconserv

    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts_ov * (nocc * nvir)**2) * 16 / 1e6
    if mp.with_df_ints:
        naux = mp._scf.with_df.auxcell.nao_nr()
        mem_usage += (nkpts_ov**2 * naux * nocc * nvir) * 16 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))

    if mp.with_df_ints:
        Lov = _init_mp_df_eris_stagger(mp)
    else:
        with_df = df.FFTDF(mp.cell, mp.kpts)
        fao2mo = with_df.ao2mo

    #   info of occupied and virtual meshes
    nkpts_ov     = mp.nkpts_ov          #   number of points on the occupied/virtual mesh
    kpts_idx_vir = mp.kpts_idx_vir
    kpts_idx_occ = mp.kpts_idx_occ

    eia   = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    emp2     = 0.0
    oovv_ij  = np.zeros((nkpts_ov,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    #   get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    for iki, ki in enumerate(kpts_idx_occ):
        for ikj, kj in enumerate(kpts_idx_occ):
            for ika, ka in enumerate(kpts_idx_vir):
                kb = kconserv[ki,ka,kj]
                ikb = np.where(kpts_idx_vir == kb)[0][0]

                if mp.with_df_ints:
                    oovv_ij[ika] = (1./nkpts_ov) * lib.einsum(
                        "Lia,Ljb->iajb",
                        Lov[iki, ika], Lov[ikj, ikb]
                    ).transpose(0,2,1,3)
                else:
                    orbo_i = mo_coeff[ki][:,:nocc]
                    orbo_j = mo_coeff[kj][:,:nocc]
                    orbv_a = mo_coeff[ka][:,nocc:]
                    orbv_b = mo_coeff[kb][:,nocc:]
                    oovv_ij[ika] = fao2mo(
                        (orbo_i,orbv_a,orbo_j,orbv_b),
                        (mp.kpts[ki],mp.kpts[ka],mp.kpts[kj],mp.kpts[kb]),
                        compact=False
                    ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts_ov
            for ika, ka in enumerate(kpts_idx_vir):
                kb = kconserv[ki,ka,kj]
                ikb = np.where(kpts_idx_vir == kb)[0][0]

                #   remove zero/padded elements from denominator
                eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka][:nvir])
                eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb][:nvir])
                ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

                #   energy calculation
                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                t2_ijab = np.conj(oovv_ij[ika]/eijab)
                woovv = 2*oovv_ij[ika] - oovv_ij[ikb].transpose(0,1,3,2)
                emp2 += np.einsum('ijab,ijab', t2_ijab, woovv).real

    log.timer("KMP2_stagger", *cput0)
    emp2 /= nkpts_ov
    return emp2


def _init_mp_df_eris_stagger(mp):
    """
    Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals that lie on two staggered uniform meshes.
    Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Arguments:
        mp (KMP2_stagger) -- A KMP2_stagger instance

    Returns:
        Lov (numpy.ndarray) -- 3-center DF ints, with shape (nkpts_ov, nkpts_ov, naux, nocc, nvir)
    """

    log = logger.Logger(mp.stdout, mp.verbose)

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    if mp._scf.with_df._cderi is not None and mp.flag_submesh is True:
        #   When using submeshes for staggered mesh calculation, the 3c DF tensor from mean-field
        #   calculation can be used directly.
        with_df = mp._scf.with_df

    else:
        with_df = df.GDF(mp.cell, mp.kpts).build()

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    #   info of occupied and virtual meshes
    nkpts_ov     = mp.nkpts_ov          #   number of points on the occupied/virtual mesh
    kpts_idx_vir = mp.kpts_idx_vir
    kpts_idx_occ = mp.kpts_idx_occ

    mo_coeff = _add_padding(mp, mp.mo_coeff, mp.mo_energy)[0]
    dtype = np.result_type(np.complex128, *mo_coeff)

    Lov = np.empty((nkpts_ov, nkpts_ov), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())
    bra_start = 0
    bra_end = nocc
    ket_start = nmo+nocc
    ket_end = ket_start + nvir
    cderi_array = CDERIArray(with_df._cderi)
    tao = []
    ao_loc = None
    for iki, ki in enumerate(kpts_idx_occ):
        for ika, ka in enumerate(kpts_idx_vir):
            Lpq_ao = cderi_array[ki, ka]

            mo = np.hstack((mo_coeff[ki], mo_coeff[ka]))
            mo = np.asarray(mo, dtype=dtype, order='F')

            #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
            if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
            out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
            Lov[iki, ika] = out.reshape(-1, nocc, nvir)

    log.timer_debug1("transforming DF-MP2 integrals", *cput0)

    return Lov


#   Class for staggered mesh MP2
class KMP2_stagger(kmp2.KMP2):
    def __init__(self, mf, frozen=None, flag_submesh=False):
        self.cell = mf.cell
        self._scf = mf
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = self.cell.max_memory

        #   MP2 energy
        self.e_corr = None

        #   nocc and nmo variables
        self._nocc = None
        self._nmo = None

        #   Construction of orbitals and orbital energies on two staggered meshes
        nks = get_monkhorst_pack_size(mf.cell, mf.kpts)  #   mesh size
        self.flag_submesh = flag_submesh
        if flag_submesh is True:
            #   Choice 1: Use two staggered submeshes of mf.kpts of half size along each dimension
            if np.any( nks % 2) :
                log = logger.Logger(self.stdout, self.verbose)
                log.warn(
                    'The k-point mesh in the input mf has odd size in certain dimension. \n'
                    'Cannot use its submeshes for staggered mesh calculation!\n'
                    'You may set "flag_submesh" to False')
                raise RuntimeError

            nks_ov = nks // 2
            shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks_ov] )
            kpts_vir = shift + mf.cell.make_kpts(nks_ov, with_gamma_point=True)
            kpts_occ = kpts_vir + half_shift

            self.kpts = mf.kpts
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = get_occ(mf, mo_energy_kpts=mf.mo_energy)

        else:
            #   Choice 2: Apply non-SCF calculation to compute orbitals and orbital energies on a
            #             staggered mesh of the same size as mf.kpts.
            if mf.cell.dimension < 3:
                # The non-SCF calculation of orbitals/orbital energies is currently only valid
                # for 3D systems in PySCF. For example, when using get_band with a reference HF
                # system of mesh 1*1*k to compute orbitals and orbital energies on a 1*1*2k mesh,
                # the obtained mo_energy and  mo_coeff are not in good agreement with those computed
                # from an SCF-HF calculation with mesh 1*1*2k.
                raise NotImplementedError

            half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks] )
            kpts_vir = mf.kpts
            kpts_occ = kpts_vir + half_shift
            kpts = np.concatenate( (kpts_occ, kpts_vir), axis=0)
            if isinstance(mf, scf.khf.KRHF):
                with lib.temporary_env(mf, exxdiv='vcut_sph', with_df=df.FFTDF(mf.cell, mf.kpts)):
                    mo_energy, mo_coeff = mf.get_bands(kpts)
                    mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)

            self.kpts = kpts
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ

        if isinstance(self._scf.with_df, df.df.GDF):
            self.with_df_ints = True
        else:
            self.with_df_ints = False

        #   Basic info
        self.nkpts = len(self.kpts)
        self.frozen = frozen
        self.khelper = kpts_helper.KptsHelper(self.cell, self.kpts)

        #   Mesh info for the staggered mesh method
        self.kpts_vir = kpts_vir
        self.kpts_occ = kpts_occ
        self.nkpts_ov = len(self.kpts_vir)

        #   Map kpts, kpts_vir, and kpts_occ to the first BZ for better processing
        kpts_scaled = self.cell.get_scaled_kpts(self.kpts)
        kpts_occ_scaled = self.cell.get_scaled_kpts(self.kpts_occ)
        kpts_vir_scaled = self.cell.get_scaled_kpts(self.kpts_vir)
        kpts_bz = round_to_fbz(kpts_scaled, wrap_around=True, tol=1e-8)
        kpts_occ_bz = round_to_fbz(kpts_occ_scaled, wrap_around=True, tol=1e-8)
        kpts_vir_bz = round_to_fbz(kpts_vir_scaled, wrap_around=True, tol=1e-8)

        #   indices of virtual k-points in self.kpts
        self.kpts_idx_vir = [ np.asarray(np.sum((kpts_bz - kvir)**2, axis=-1) < 1e-10).nonzero()[0]
                                for kvir in kpts_vir_bz]
        self.kpts_idx_vir = np.concatenate(self.kpts_idx_vir).astype(int)

        #   indices of occupied k-points in self.kpts
        self.kpts_idx_occ = [ np.asarray(np.sum((kpts_bz - kocc)**2, axis=-1) < 1e-10).nonzero()[0]
                                for kocc in kpts_occ_bz]
        self.kpts_idx_occ = np.concatenate(self.kpts_idx_occ).astype(int)

        if (len(self.kpts_idx_vir) != self.nkpts_ov or
            len(np.unique(self.kpts_idx_vir)) != self.nkpts_ov or
            len(self.kpts_idx_occ) != self.nkpts_ov or
            len(np.unique(self.kpts_idx_occ)) != self.nkpts_ov):
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('Cannot locate the provided occupied/virtual submeshes in the large k-point mesh')
            raise RuntimeError

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('Staggerd mesh method for MP2 nocc = %d, nvir = %d', nocc, nvir)

        nks_ov = get_monkhorst_pack_size(self.cell, self.kpts_vir)  #   mesh size
        if self.flag_submesh is True:
            log.info('Two %d*%d*%d-sized submeshes of mf.kpts are used', nks_ov[0], nks_ov[1], nks_ov[2])
        else:
            log.info('Two %d*%d*%d-sized meshes are used based on non-SCF calculation', nks_ov[0], nks_ov[1], nks_ov[2])

        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        log.info('KMP2 energy = %.15g' % (self.e_corr))
        return self

    def kernel(self):
        if self.mo_energy is None or self.mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not provided.')
            raise RuntimeError

        mo_coeff, mo_energy = _add_padding(self, self.mo_coeff, self.mo_energy)

        self.e_corr = kernel(self, mo_energy, mo_coeff)
        self.dump_flags()
        return self.e_corr


if __name__ == '__main__':
    from pyscf.pbc import gto, mp
    '''
    Example calculation on hydrogen dimer
    '''
    cell = gto.Cell()
    cell.pseudo = 'gth-pade'
    cell.basis = 'gth-szv'
    cell.ke_cutoff=100
    cell.atom='''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()


    #   HF calculation using FFTDF
    nks_mf = [2,2,2]
    kpts = cell.make_kpts(nks_mf, with_gamma_point=True)
    kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
    ehf = kmf.kernel()

    #   staggered mesh KMP2 calculation using two submeshes of size [1,1,1] in kmf.kpts
    kmp = KMP2_stagger(kmf, flag_submesh=True)
    emp2 = kmp.kernel()
    assert((abs(emp2 - -0.0160902544091997))<1e-5)

    #   staggered mesh KMP2 calculation using two meshes of size [2,2,2], one of them is kmf.kpts
    kmp = KMP2_stagger(kmf, flag_submesh=False)
    emp2 = kmp.kernel()
    assert((abs(emp2 -  -0.0140289970302513))<1e-5)

    #   standard KMP2 calculation
    kmp = mp.KMP2(kmf)
    emp2, _ = kmp.kernel()
    assert((abs(emp2 - -0.0143904878990777))<1e-5)


    #   HF calculation using GDF
    nks_mf = [2,2,2]
    kpts = cell.make_kpts(nks_mf, with_gamma_point=True)
    kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
    gdf = df.GDF(cell, kpts).build()
    kmf.with_df = gdf
    ehf = kmf.kernel()

    #   staggered mesh KMP2 calculation using two submeshes of size [1,1,1] in kmf.kpts
    kmp = KMP2_stagger(kmf, flag_submesh=True)
    emp2 = kmp.kernel()
    assert((abs(emp2 - -0.0158364523431071))<1e-5)

    #   staggered mesh KMP2 calculation using two meshes of size [2,2,2], one of them is kmf.kpts
    kmp = KMP2_stagger(kmf, flag_submesh=False)
    emp2 = kmp.kernel()
    assert((abs(emp2 -  -0.0140280303691396))<1e-5)

    #   standard KMP2 calculation
    kmp = mp.KMP2(kmf)
    emp2, _ = kmp.kernel()
    assert((abs(emp2 - -0.0141829343769316))<1e-5)
