#!/usr/bin/env python
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

r'''
Range-separated Gaussian Density Fitting (RSGDF)

ref.:
[1] For the RSGDF method:
    Hong-Zhou Ye and Timothy C. Berkelbach, J. Chem. Phys. 154, 131104 (2021).
[2] For the SR lattice sum integral screening:
    Hong-Zhou Ye and Timothy C. Berkelbach, arXiv:2107.09704.

In RSGDF, the two-center and three-center Coulomb integrals are calculated in two pars:
    j2c = j2c_SR(omega) + j2c_LR(omega)
    j3c = j3c_SR(omega) + j3c_LR(omega)
where the SR and LR integrals correpond to using the following potentials
    g_SR(r_12;omega) = erfc(omega * r_12) / r_12
    g_LR(r_12;omega) = erf(omega * r_12) / r_12
The SR integrals are evaluated in real space using a lattice summation, while the LR integrals are evaluated in reciprocal space with a plane wave basis.
'''

import os
import h5py
import scipy.linalg
import tempfile
import numpy as np

from pyscf import gto as mol_gto
from pyscf.pbc import df
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import rsdf_helper
from pyscf.df.outcore import _guess_shell_ranges
from pyscf.pbc import tools as pbctools
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf import lib
from pyscf.lib import logger


def kpts_to_kmesh(cell, kpts):
    """ Check if kpt mesh includes the Gamma point. Generate the bvk kmesh only if it does.
    """
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    if np.any(abs(scaled_k).sum(axis=1) < KPT_DIFF_TOL):
        kmesh = (len(np.unique(scaled_k[:,0])),
                 len(np.unique(scaled_k[:,1])),
                 len(np.unique(scaled_k[:,2])))
    else:
        kmesh = None
    return kmesh


def weighted_coulG(cell, omega, kpt=np.zeros(3), exx=False, mesh=None):
    if cell.omega != 0:
        raise RuntimeError('RSGDF cannot be used '
                           'to evaluate the long-range HF exchange in RSH '
                           'functional')
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    if abs(omega) < 1.e-10:
        omega_ = None
    else:
        omega_ = omega
    coulG = pbctools.get_coulG(cell, kpt, False, None, mesh, Gv,
                               omega=omega_)
    coulG *= kws
    return coulG
def get_aux_chg(auxcell):
    r""" Compute charge of the auxiliary basis, \int_Omega dr chi_P(r)

    Returns:
        The function returns a 1d numpy array of size auxcell.nao_nr().
    """
    def get_nd(l):
        if auxcell.cart:
            return (l+1) * (l+2) // 2
        else:
            return 2 * l + 1

    naux = auxcell.nao_nr()
    qs = np.zeros(naux)
    shift = 0
    half_sph_norm = np.sqrt(4*np.pi)
    for ib in range(auxcell.nbas):
        l = auxcell.bas_angular(ib)
        if l == 0:
            npm = auxcell.bas_nprim(ib)
            nc = auxcell.bas_nctr(ib)
            es = auxcell.bas_exp(ib)
            ptr = auxcell._bas[ib,mol_gto.PTR_COEFF]
            cs = auxcell._env[ptr:ptr+npm*nc].reshape(nc,npm).T
            norms = mol_gto.gaussian_int(l+2, es)
            q = np.einsum("i,ij->j",norms,cs)[0] * half_sph_norm
        else:   # higher angular momentum AOs carry no charge
            q = 0.
        nd = get_nd(l)
        qs[shift:shift+nd] = q
        shift += nd

    return qs


# kpti == kptj: s2 symmetry
# kpti == kptj == 0 (gamma point): real
def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mydf.stdout, mydf.verbose)
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])

    omega = abs(mydf.omega)

    if mydf.use_bvk:
        bvk_kmesh = kpts_to_kmesh(cell, mydf.kpts)
        if bvk_kmesh is None:
            log.debug("Non-Gamma-inclusive kmesh is found. bvk kmesh is not used.")
        else:
            log.debug("Using bvk kmesh= [%d %d %d]", *bvk_kmesh)
    else:
        bvk_kmesh = None

    # The ideal way to hold the temporary integrals is to store them in the
    # cderi_file and overwrite them inplace in the second pass.  The current
    # HDF5 library does not have an efficient way to manage free space in
    # overwriting.  It often leads to the cderi_file ~2 times larger than the
    # necessary size.  For now, dumping the DF integral intermediates to a
    # separated temporary file can avoid this issue.  The DF intermediates may
    # be terribly huge. The temporary file should be placed in the same disk
    # as cderi_file.
    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash
    swapfile = None

    # get charge of auxbasis
    if cell.dimension == 3:
        qaux = get_aux_chg(auxcell)
    else:
        quax = np.zeros(auxcell.nao_nr())

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

# compute j2c first as it informs the integral screening in computing j3c
    # short-range part of j2c ~ (-kpt_ji | kpt_ji)
    omega_j2c = abs(mydf.omega_j2c)
    j2c = rsdf_helper.intor_j2c(auxcell, omega_j2c, kpts=uniq_kpts)

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    qaux2 = None
    g0_j2c = np.pi/omega_j2c**2./cell.vol
    mesh_j2c = mydf.mesh_j2c
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh_j2c)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
    blksize = max(2048, int(max_memory*.5e6/16/auxcell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)

    for k, kpt in enumerate(uniq_kpts):
        # short-range charge part
        if is_zero(kpt) and cell.dimension == 3:
            if qaux2 is None:
                qaux2 = np.outer(qaux,qaux)
            j2c[k] -= qaux2 * g0_j2c
        # long-range part via aft
        coulG_lr = weighted_coulG(cell, omega_j2c, kpt, False, mesh_j2c)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, b, gxyz[p0:p1],
                                Gvbase, kpt).T
            LkR = np.asarray(aoaux.real, order='C')
            LkI = np.asarray(aoaux.imag, order='C')
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2c[k] += lib.ddot(LkR*coulG_lr[p0:p1], LkR.T)
                j2c[k] += lib.ddot(LkI*coulG_lr[p0:p1], LkI.T)
            else:
                j2cR, j2cI = df.df_jk.zdotCN(LkR*coulG_lr[p0:p1],
                                             LkI*coulG_lr[p0:p1], LkR.T, LkI.T)
                j2c[k] += j2cR + j2cI * 1j

            LkR = LkI = None

        fswap['j2c/%d'%k] = j2c[k]
    j2c = coulG_lr = None

    t1 = log.timer_debug1('2c2e', *t1)

    def cholesky_decomposed_metric(uniq_kptji_id):
        j2c = np.asarray(fswap['j2c/%d'%uniq_kptji_id])
        j2c_negative = None
        try:
            if mydf.j2c_eig_always:
                raise scipy.linalg.LinAlgError
            j2c = scipy.linalg.cholesky(j2c, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that mesh is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([str(e), msg]))
            w, v = scipy.linalg.eigh(j2c)
            ndrop = np.count_nonzero(w<mydf.linear_dep_threshold)
            if ndrop > 0:
                log.debug('DF metric linear dependency for kpt %s',
                          uniq_kptji_id)
                log.debug('cond = %.4g, drop %d bfns', w[-1]/w[0], ndrop)
            v1 = v[:,w>mydf.linear_dep_threshold].conj().T
            v1 /= np.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = np.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = (v[:,idx]/np.sqrt(-w[idx])).conj().T
            w = v = None
            j2ctag = 'eig'
        return j2c, j2c_negative, j2ctag

# compute j3c
    # inverting j2c, and use it's column max to determine an extra precision for 3c2e prescreening

    # short-range part
    rsdf_helper._aux_e2_nospltbas(
                    cell, auxcell, omega, fswap, 'int3c2e', aosym='s2',
                    kptij_lst=kptij_lst, dataname='j3c-junk',
                    max_memory=max_memory,
                    bvk_kmesh=bvk_kmesh,
                    precision=mydf.precision_R)
    t1 = log.timer_debug1('3c2e', *t1)

    prescreening_data = None

    # recompute g0 and Gvectors for j3c
    g0 = np.pi/omega**2./cell.vol
    mesh = mydf.mesh_compact
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    # Add (1) short-range G=0 (i.e., charge) part and (2) long-range part
    tspans = np.zeros((3,2))    # lr, j2c_inv, j2c_cntr
    tspannames = ["ftaop+pw", "j2c_inv", "j2c_cntr"]
    feri = h5py.File(cderi_file, 'w')
    feri['j3c-kptij'] = kptij_lst
    nsegs = len(fswap['j3c-junk/0'])
    def make_kpt(uniq_kptji_id, cholesky_j2c):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        log.debug1('kpt = %s', kpt)
        adapted_ji_idx = np.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        j2c, j2c_negative, j2ctag = cholesky_j2c

        shls_slice = (0, auxcell.nbas)
        Gaux = ft_ao.ft_ao(auxcell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        wcoulG_lr = weighted_coulG(cell, omega, kpt, False, mesh)
        Gaux *= wcoulG_lr.reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        Gaux = None

        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            nao_pair = nao*(nao+1)//2

            if cell.dimension == 3:
                vbar = qaux * g0
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=adapted_kptjs)
                ovlp = [lib.pack_tril(s) for s in ovlp]
        else:
            aosym = 's1'
            nao_pair = nao**2

        mem_now = lib.current_memory()[0]
        log.debug2('memory = %s', mem_now)
        max_memory = max(2000, mydf.max_memory-mem_now)
        # nkptj for 3c-coulomb arrays plus 1 Lpq array
        buflen = min(max(int(max_memory*.38e6/16/naux/(nkptj+1)), 1), nao_pair)
        shranges = _guess_shell_ranges(cell, buflen, aosym)
        buflen = max([x[2] for x in shranges])
        # +1 for a pqkbuf
        if aosym == 's2':
            Gblksize = max(16, int(max_memory*.1e6/16/buflen/(nkptj+1)))
        else:
            Gblksize = max(16, int(max_memory*.2e6/16/buflen/(nkptj+1)))
        Gblksize = min(Gblksize, ngrids, 16384)

        def load(aux_slice):
            col0, col1 = aux_slice
            j3cR = []
            j3cI = []
            for k, idx in enumerate(adapted_ji_idx):
                v = np.vstack([fswap['j3c-junk/%d/%d'%(idx,i)][0,col0:col1].T
                               for i in range(nsegs)])
                # vbar is the interaction between the background charge
                # and the auxiliary basis.  0D, 1D, 2D do not have vbar.
                if is_zero(kpt) and cell.dimension == 3:
                    for i in np.where(vbar != 0)[0]:
                        v[i] -= vbar[i] * ovlp[k][col0:col1]
                j3cR.append(np.asarray(v.real, order='C'))
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    j3cI.append(None)
                else:
                    j3cI.append(np.asarray(v.imag, order='C'))
                v = None
            return j3cR, j3cI

        pqkRbuf = np.empty(buflen*Gblksize)
        pqkIbuf = np.empty(buflen*Gblksize)
        # buf for ft_aopair
        buf = np.empty(nkptj*buflen*Gblksize, dtype=np.complex128)
        cols = [sh_range[2] for sh_range in shranges]
        locs = np.append(0, np.cumsum(cols))
        tasks = zip(locs[:-1], locs[1:])
        for istep, (j3cR, j3cI) in enumerate(lib.map_with_prefetch(load, tasks)):
            bstart, bend, ncol = shranges[istep]
            log.debug1('int3c2e [%d/%d], AO [%d:%d], ncol = %d',
                       istep+1, len(shranges), bstart, bend, ncol)
            if aosym == 's2':
                shls_slice = (bstart, bend, 0, bend)
            else:
                shls_slice = (bstart, bend, 0, cell.nbas)

            tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
            for p0, p1 in lib.prange(0, ngrids, Gblksize):
                dat = ft_ao.ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                           b, gxyz[p0:p1], Gvbase, kpt,
                                           adapted_kptjs, out=buf,
                                           bvk_kmesh=bvk_kmesh)
                nG = p1 - p0
                for k, ji in enumerate(adapted_ji_idx):
                    aoao = dat[k].reshape(nG,ncol)
                    pqkR = np.ndarray((ncol,nG), buffer=pqkRbuf)
                    pqkI = np.ndarray((ncol,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao.real.T
                    pqkI[:] = aoao.imag.T

                    lib.dot(kLR[p0:p1].T, pqkR.T, 1, j3cR[k][:], 1)
                    lib.dot(kLI[p0:p1].T, pqkI.T, 1, j3cR[k][:], 1)
                    if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                        lib.dot(kLR[p0:p1].T, pqkI.T, 1, j3cI[k][:], 1)
                        lib.dot(kLI[p0:p1].T, pqkR.T, -1, j3cI[k][:], 1)
            tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[0] += tock_ - tick_

            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                    v = j3cR[k]
                else:
                    v = j3cR[k] + j3cI[k] * 1j
                if j2ctag == 'CD':
                    v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                    feri['j3c/%d/%d'%(ji,istep)] = v
                else:
                    feri['j3c/%d/%d'%(ji,istep)] = lib.dot(j2c, v)

                # low-dimension systems
                if j2c_negative is not None:
                    feri['j3c-/%d/%d'%(ji,istep)] = lib.dot(j2c_negative, v)
            j3cR = j3cI = None
            tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
            tspans[2] += tick_ - tock_

        for ji in adapted_ji_idx:
            del(fswap['j3c-junk/%d'%ji])

    # Wrapped around boundary and symmetry between k and -k can be used
    # explicitly for the metric integrals.  We consider this symmetry
    # because it is used in the df_ao2mo module when contracting two 3-index
    # integral tensors to the 4-index 2e integral tensor. If the symmetry
    # related k-points are treated separately, the resultant 3-index tensors
    # may have inconsistent dimension due to the numerial noise when handling
    # linear dependency of j2c.
    def conj_j2c(cholesky_j2c):
        j2c, j2c_negative, j2ctag = cholesky_j2c
        if j2c_negative is None:
            return j2c.conj(), None, j2ctag
        else:
            return j2c.conj(), j2c_negative.conj(), j2ctag

    a = cell.lattice_vectors() / (2*np.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = np.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = np.rint(kdif)
        mask = np.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = np.where(mask)[0]
        return uniq_kptji_ids

    done = np.zeros(len(uniq_kpts), dtype=bool)
    for k, kpt in enumerate(uniq_kpts):
        if done[k]:
            continue

        log.debug1('Cholesky decomposition for j2c at kpt %s', k)
        tick_ = np.asarray((logger.process_clock(), logger.perf_counter()))
        cholesky_j2c = cholesky_decomposed_metric(k)
        tock_ = np.asarray((logger.process_clock(), logger.perf_counter()))
        tspans[1] += tock_ - tick_

        # The k-point k' which has (k - k') * a = 2n pi. Metric integrals have the
        # symmetry S = S
        uniq_kptji_ids = kconserve_indices(-kpt)
        log.debug1("Symmetry pattern (k - %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for uniq_kptji_ids %s", uniq_kptji_ids)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

        # The k-point k' which has (k + k') * a = 2n pi. Metric integrals have the
        # symmetry S = S*
        uniq_kptji_ids = kconserve_indices(kpt)
        log.debug1("Symmetry pattern (k + %s)*a= 2n pi", kpt)
        log.debug1("    make_kpt for %s", uniq_kptji_ids)
        cholesky_j2c = conj_j2c(cholesky_j2c)
        for uniq_kptji_id in uniq_kptji_ids:
            if not done[uniq_kptji_id]:
                make_kpt(uniq_kptji_id, cholesky_j2c)
        done[uniq_kptji_ids] = True

    feri.close()

    # report time for aft part
    for tspan, tspanname in zip(tspans, tspannames):
        log.debug1("    CPU time for %s %9.2f sec, wall time %9.2f sec",
                   "%10s"%tspanname, *tspan)
    log.debug1("%s", "")


class RSGDF(df.df.GDF):
    '''Range Separated Gaussian Density Fitting
    '''

    # class methods defined outside the class
    _make_j3c = _make_j3c

    def __init__(self, cell, kpts=np.zeros((1,3))):
        if cell.dimension < 3:
            raise NotImplementedError("RSGDF for low-dimensional systems are not available yet. We recommend using cell.dimension=3 with large vacuum.")

        # if True and kpts are gamma-inclusive, RSDF will use the bvk cell trick for computing both j3c_SR and j3c_LR. If kpts are not gamma-inclusive, this attribute will be ignored.
        self.use_bvk = True

        # precision for real-space lattice sum (R) and reciprocal-space Fourier transform (G).
        self.precision_R = cell.precision * 1e-2
        self.precision_G = cell.precision

        # One of {omega, npw_max} must be provided, and the other will be deduced automatically from it. The priority when both are given is omega > npw_max.
        # If omega deduced from npw_max is smaller than self._omega_min, omega = omega_min is used.
        # The default is npw_max = 350 ~ 7x7x7 PWs for 3D isotropic systems.
        # Once omega is determined, mesh_compact is determined for (L|g^lr|pq) to achieve given accuracy.
        self.npw_max = 350
        self._omega_min = 0.1
        self.omega = None
        self.ke_cutoff = None
        self.mesh_compact = None

        # omega and mesh for j2c. Since j2c is to be inverted, it is desirable to use (1) a "fixed" omega for reproducibility, and (2) a higher precision to minimize any round-off error caused by inversion.
        # The extra computational cost due to the higher precision is negligible compared to the j3c build.
        # FOR EXPERTS: if you want omega_j2c to be the same as the omega used for j3c build, set omega_j2c to be any negative number.
        self.omega_j2c = 0.4
        self.mesh_j2c = None
        self.precision_j2c = 1e-4 * self.precision_G

        # set True to force calculating j2c^(-1/2) using eigenvalue decomposition (ED); otherwise, Cholesky decomposition (CD) is used first, and ED is called only if CD fails.
        self.j2c_eig_always = False

        df.df.GDF.__init__(self, cell, kpts=kpts)

    def dump_flags(self, verbose=None):
        cell = self.cell
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s ********', self.__class__)
        log.info('cell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 cell.nbas, cell.nao_nr(), cell.npgto_nr())
        log.info('use_bvk = %s', self.use_bvk)
        log.info('precision_R = %s', self.precision_R)
        log.info('precision_G = %s', self.precision_G)
        log.info('j2c_eig_always = %s', self.j2c_eig_always)
        log.info('omega = %s', self.omega)
        log.info('ke_cutoff = %s', self.ke_cutoff)
        log.info('mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        log.info('mesh_compact = %s (%d PWs)', self.mesh_compact,
                 np.prod(self.mesh_compact))
        if self.auxcell is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = %s', self.auxcell.basis)
            log.info('auxcell precision= %s', self.auxcell.precision)
            log.info('auxcell rcut = %s', self.auxcell.rcut)
            log.info('omega_j2c = %s', self.omega_j2c)
            log.info('mesh_j2c = %s (%d PWs)', self.mesh_j2c,
                     np.prod(self.mesh_j2c))

        auxcell = self.auxcell
        log.info('auxcell num shells = %d, num cGTOs = %d, num pGTOs = %d',
                 auxcell.nbas, auxcell.nao_nr(),
                 auxcell.npgto_nr())

        log.info('exp_to_discard = %s', self.exp_to_discard)
        if isinstance(self._cderi, str):
            log.info('_cderi = %s  where DF integrals are loaded (readonly).',
                     self._cderi)
        elif isinstance(self._cderi_to_save, str):
            log.info('_cderi_to_save = %s', self._cderi_to_save)
        else:
            log.info('_cderi_to_save = %s', self._cderi_to_save.name)
        log.info('len(kpts) = %d', len(self.kpts))
        log.debug1('    kpts = %s', self.kpts)
        if self.kpts_band is not None:
            log.info('len(kpts_band) = %d', len(self.kpts_band))
            log.debug1('    kpts_band = %s', self.kpts_band)

        return self

    def _rsh_build(self):
        # find kmax
        kpts = self.kpts if self.kpts_band is None else np.vstack([self.kpts,
                                                                self.kpts_band])
        b = self.cell.reciprocal_vectors()
        scaled_kpts = np.linalg.solve(b.T, kpts.T).T
        scaled_kpts[scaled_kpts > 0.49999999] -= 1
        kpts = np.dot(scaled_kpts, b)
        kmax = np.linalg.norm(kpts, axis=-1).max()
        scaled_kpts = kpts = None
        if kmax < 1.e-3: kmax = (0.75/np.pi/self.cell.vol)**0.33333333*2*np.pi

        # If omega is not given, estimate it from npw_max
        r2o = True
        if self.omega is None:
            self.omega, self.ke_cutoff, self.mesh_compact = \
                                rsdf_helper.estimate_omega_for_npw(
                                                self.cell, self.npw_max,
                                                self.precision_G,
                                                kmax=kmax,
                                                round2odd=r2o)
            # if omega from npw_max is too small, use omega_min
            if self.omega < self._omega_min:
                self.omega = self._omega_min
                self.ke_cutoff, self.mesh_compact = \
                                    rsdf_helper.estimate_mesh_for_omega(
                                                    self.cell, self.omega,
                                                    self.precision_G,
                                                    kmax=kmax,
                                                    round2odd=r2o)
        else:
            self.ke_cutoff, self.mesh_compact = \
                                rsdf_helper.estimate_mesh_for_omega(
                                                self.cell, self.omega,
                                                self.precision_G,
                                                kmax=kmax,
                                                round2odd=r2o)

        # As explained in __init__, if negative omega_j2c --> use omega
        if self.omega_j2c < 0: self.omega_j2c = self.omega

        # build auxcell
        from pyscf.df.addons import make_auxmol
        auxcell = make_auxmol(self.cell, self.auxbasis)

        # determine mesh for computing j2c
        auxcell.precision = self.precision_j2c
        auxcell.rcut = max([auxcell.bas_rcut(ib, auxcell.precision)
                            for ib in range(auxcell.nbas)])
        self.mesh_j2c = rsdf_helper.estimate_mesh_for_omega(
                                auxcell, self.omega_j2c, round2odd=True)[1]
        self.auxcell = auxcell

    def _kpts_build(self, kpts_band=None):
        if self.kpts_band is not None:
            self.kpts_band = np.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = np.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(np.vstack((self.kpts_band,kpts_band)))[0]

    def _gdf_build(self, j_only=None, with_j3c=True):
        # Remove duplicated k-points. Duplicated kpts may lead to a buffer
        # located in incore.wrap_int3c larger than necessary. Integral code
        # only fills necessary part of the buffer, leaving some space in the
        # buffer unfilled.
        uniq_idx = unique(self.kpts)[1]
        kpts = np.asarray(self.kpts)[uniq_idx]
        if self.kpts_band is None:
            kband_uniq = np.zeros((0,3))
        else:
            kband_uniq = [k for k in self.kpts_band if len(member(k, kpts))==0]
        if j_only is None:
            j_only = self._j_only
        if j_only:
            kall = np.vstack([kpts,kband_uniq])
            kptij_lst = np.hstack((kall,kall)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
            kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
            kptij_lst.extend([(ki, ki) for ki in kband_uniq])
            kptij_lst = np.asarray(kptij_lst)

        if with_j3c:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'DF integrals in %s (specified by '
                                '._cderi) is overwritten by GDF '
                                'initialization. ', cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .',
                                cderi)
            self._cderi = cderi
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, kptij_lst, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        # formatting k-points
        self._kpts_build(kpts_band=kpts_band)

        # build for range-separation hybrid
        self._rsh_build()

        # dump flags before the final build
        self.check_sanity()
        self.dump_flags()

        # do normal gdf build with the modified _make_j3c
        self._gdf_build(j_only=j_only, with_j3c=with_j3c)

        return self

RSDF = RSGDF


if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell(
        atom="H 0 0 0; H 0.75 0 0",
        a = np.eye(3)*2.5,
        basis={"H": [[0,(0.5,1.)],[1,(0.3,1.)]]},
    )
    cell.build()
    cell.verbose = 0

    scaled_center = None
    # scaled_center = np.random.rand(3)

    log = logger.Logger(cell.stdout, 6)

    for kmesh in ([1,1,1,],[2,1,1]):
        kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
        log.info("kmesh= %s", kmesh)
        log.info("kpts = %s", kpts)

        from pyscf.pbc import scf, mp, cc
        mf = scf.KRHF(cell, kpts=kpts).rs_density_fit()
        mf.kernel()

        mf2 = scf.KRHF(cell, kpts=kpts).density_fit()
        mf2.kernel()
        log.info("HF/GDF   energy   : % .10f", mf2.e_tot)
        log.info("HF/RSGDF energy   : % .10f", mf.e_tot)
        log.info("difference        : % .3g", (mf.e_tot-mf2.e_tot))

        mc = cc.KCCSD(mf)
        mc.kernel()
        mc2 = cc.KCCSD(mf2)
        mc2.kernel()
        log.info("CCSD/GDF   energy : % .10f", mc2.e_corr)
        log.info("CCSD/RSGDF energy : % .10f", mc.e_corr)
        log.info("difference        : % .3g\n", (mc.e_corr-mc2.e_corr))
