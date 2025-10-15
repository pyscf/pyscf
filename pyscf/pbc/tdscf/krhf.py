#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import linalg_helper
from pyscf.lib import logger
from pyscf.tdscf import rhf
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from pyscf.pbc import scf
from pyscf.pbc.tdscf.rhf import TDBase
from pyscf.pbc.scf import _response_functions  # noqa
from pyscf.pbc.lib.kpts_helper import is_gamma_point, get_kconserv_ria, conj_mapping
from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
from pyscf.pbc import df as pbcdf
from pyscf.data import nist
from pyscf import __config__

REAL_EIG_THRESHOLD = getattr(__config__, 'pbc_tdscf_rhf_TDDFT_pick_eig_threshold', 1e-3)

def get_ab(mf, kshift=0):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ai||jb)
    B[i,a,j,b] = (ai||bj)

    Ref: Chem Phys Lett, 256, 454

    Kwargs:
        kshift : integer
            The index of the k-point that represents the transition between
            k-points in the excitation coefficients.
    '''
    cell = mf.cell
    mo_energy = scf.addons.mo_energy_with_exxdiv_none(mf)
    mo = numpy.asarray(mf.mo_coeff)
    mo_occ = numpy.asarray(mf.mo_occ)
    kpts = mf.kpts
    nkpts, nao, nmo = mo.shape
    noccs = numpy.count_nonzero(mo_occ==2, axis=1)
    nocc = noccs[0]
    nvir = nmo - nocc
    assert all(noccs == nocc)
    orbo = mo[:,:,:nocc]
    orbv = mo[:,:,nocc:]

    kconserv = get_kconserv_ria(cell, kpts)[kshift]
    e_ia = numpy.asarray(_get_e_ia(mo_energy, mo_occ, kconserv)).astype(mo.dtype)
    a = numpy.diag(e_ia.ravel()).reshape(nkpts,nocc,nvir,nkpts,nocc,nvir)
    b = numpy.zeros_like(a)
    weight = 1./nkpts

    def add_hf_(a, b, hyb=1):
        eri = mf.with_df.ao2mo_7d([mo,orbo,mo,mo], kpts)
        eri *= weight
        eri = eri.reshape(nkpts,nkpts,nkpts,nmo,nocc,nmo,nmo)
        for ki, ka in enumerate(kconserv):
            for kj, kb in enumerate(kconserv):
                a[ki,:,:,kj] += numpy.einsum('aijb->iajb', eri[ka,ki,kj,nocc:,:,:nocc,nocc:]) * 2
                a[ki,:,:,kj] -= numpy.einsum('jiab->iajb', eri[kj,ki,ka,:nocc,:,nocc:,nocc:]) * hyb

            for kb, kj in enumerate(kconserv):
                b[ki,:,:,kj] += numpy.einsum('aibj->iajb', eri[ka,ki,kb,nocc:,:,nocc:,:nocc]) * 2
                b[ki,:,:,kj] -= numpy.einsum('ajbi->iajb', eri[ka,kj,kb,nocc:,:,nocc:,:nocc]) * hyb

    if isinstance(mf, scf.hf.KohnShamDFT):
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, cell.spin)

        add_hf_(a, b, hyb)
        if omega != 0:  # For RSH
            raise NotImplementedError

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo, mo_occ)
        make_rho = ni._gen_rho_evaluator(cell, dm0, hermi=1, with_lapl=False)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)
        cmap = conj_mapping(cell, kpts)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpts, None, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc[0,0] * weight

                rho_o = lib.einsum('krp,kpi->kri', ao, orbo)
                rho_v = lib.einsum('krp,kpi->kri', ao, orbv)
                rho_ov = numpy.einsum('kri,kra->kria', rho_o, rho_v)
                rho_vo = rho_ov.conj()[cmap]
                w_vo = numpy.einsum('kria,r->kria', rho_vo, wfxc) * (2/nkpts)
                a += lib.einsum('kria,lrjb->kialjb', w_vo, rho_ov)
                b += lib.einsum('kria,lrjb->kialjb', w_vo, rho_vo)

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpts, None, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o = lib.einsum('kxrp,kpi->kxri', ao, orbo)
                rho_v = lib.einsum('kxrp,kpi->kxri', ao, orbv)
                rho_ov = numpy.einsum('kxri,kra->kxria', rho_o, rho_v[:,0])
                rho_ov[:,1:4] += numpy.einsum('kri,kxra->kxria', rho_o[:,0], rho_v[:,1:4])
                rho_vo = rho_ov.conj()[cmap]
                w_vo = numpy.einsum('xyr,kxria->kyria', wfxc, rho_vo) * (2/nkpts)
                a += lib.einsum('kxria,lxrjb->kialjb', w_vo, rho_ov)
                b += lib.einsum('kxria,lxrjb->kialjb', w_vo, rho_vo)

        elif xctype == 'HF':
            pass

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, _, mask, weight, coords \
                    in ni.block_loop(cell, mf.grids, nao, ao_deriv, kpts, None, max_memory):
                rho = make_rho(0, ao, mask, xctype)
                fxc = ni.eval_xc_eff(mf.xc, rho, deriv=2, xctype=xctype)[2]
                wfxc = fxc * weight
                rho_o = lib.einsum('kxrp,kpi->kxri', ao, orbo)
                rho_v = lib.einsum('kxrp,kpi->kxri', ao, orbv)
                rho_ov = numpy.einsum('kxri,kra->kxria', rho_o, rho_v[:,0])
                rho_ov[:,1:4] += numpy.einsum('kri,kxra->kxria', rho_o[:,0], rho_v[:,1:4])
                tau_ov = numpy.einsum('kxri,kxra->kria', rho_o[:,1:4], rho_v[:,1:4]) * .5
                rho_ov = numpy.concatenate([rho_ov, tau_ov[:,numpy.newaxis]], axis=1)
                rho_vo = rho_ov.conj()[cmap]
                w_vo = numpy.einsum('xyr,kxria->kyria', wfxc, rho_vo) * (2/nkpts)
                a += lib.einsum('kxria,lxrjb->kialjb', w_vo, rho_ov)
                b += lib.einsum('kxria,lxrjb->kialjb', w_vo, rho_vo)
    else:
        add_hf_(a, b)

    return a, b

class KTDBase(TDBase):
    '''
    Attributes:
        kshift_lst : list of integers
            Each element in the list is the index of the k-point that
            represents the transition between k-points in the excitation
            coefficients. For excitation amplitude T_{ai}[k] at point k,
            the kshift connects orbital i at k and orbital a at k+kshift
    '''

    conv_tol = getattr(__config__, 'pbc_tdscf_rhf_TDA_conv_tol', 1e-4)

    _keys = {'kshift_lst'}

    def __init__(self, mf, kshift_lst=None):
        assert isinstance(mf, scf.khf.KSCF)
        TDBase.__init__(self, mf)
        warn_pbc2d_eri(mf)

        if kshift_lst is None: kshift_lst = [0]
        self.kshift_lst = kshift_lst

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        if self.singlet is None:
            log.info('nstates = %d', self.nstates)
        elif self.singlet:
            log.info('nstates = %d singlet', self.nstates)
        else:
            log.info('nstates = %d triplet', self.nstates)
        log.info('deg_eia_thresh = %.3e', self.deg_eia_thresh)
        log.info('kshift_lst = %s', self.kshift_lst)
        log.info('wfnsym = %s', self.wfnsym)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('eigh lindep = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
        log.info('eigh max_cycle = %d', self.max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')

    def check_sanity(self):
        TDBase.check_sanity(self)
        mf = self._scf
        if any(k != 0 for k in self.kshift_lst):
            if mf.rsjk is not None or not isinstance(mf.with_df, pbcdf.df.DF):
                logger.error(self, 'Solutions with non-zero kshift for %s are '
                             'only supported by GDF/RSDF')
                raise NotImplementedError
        assert isinstance(mf, scf.khf.KSCF)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        for k,kshift in enumerate(self.kshift_lst):
            if not all(self.converged[k]):
                logger.note(self, 'kshift = %d  TD-SCF states %s not converged.',
                            kshift, [i for i, x in enumerate(self.converged[k]) if not x])
            logger.note(self, 'kshift = %d  Excited State energies (eV)\n%s',
                        kshift, self.e[k] * nist.HARTREE2EV)
        return self

    get_nto = lib.invalid_method('get_nto')

class TDA(KTDBase):

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None, kshift=0):
        if mf is None: mf = self._scf
        return get_ab(mf, kshift)

    def gen_vind(self, mf=None, kshift=0):
        '''Compute Ax

        Kwargs:
            kshift : integer
                The index of the k-point that represents the transition between
                k-points in the excitation coefficients.
        '''
        assert mf is self._scf
        singlet = self.singlet
        kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff[0].shape
        occidx = [mo_occ[k]==2 for k in range(nkpts)]
        viridx = [mo_occ[k]==0 for k in range(nkpts)]
        orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
        orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpts)]
        dtype = numpy.result_type(*mo_coeff)
        e_ia = _get_e_ia(scf.addons.mo_energy_with_exxdiv_none(mf), mo_occ, kconserv)
        hdiag = numpy.hstack([x.ravel() for x in e_ia])

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = self.gen_response(singlet=singlet, hermi=0, max_memory=max_memory)

        def vind(zs):
            nz = len(zs)
            z1s = [_unpack(z, mo_occ, kconserv) for z in zs]
            dms = numpy.empty((nz,nkpts,nao,nao), dtype=dtype)
            for i in range(nz):
                for k, kp in enumerate(kconserv):
                    # *2 for double occupancy
                    dm1 = z1s[i][k] * 2
                    dms[i,kp] = lib.einsum('ov,pv,qo->pq', dm1, orbv[kp], orbo[k].conj())

            with lib.temporary_env(mf, exxdiv=None):
                v1ao = vresp(dms, kshift)
            v1s = []
            for i in range(nz):
                dm1 = z1s[i]
                v1 = [None] * nkpts
                for k, kp in enumerate(kconserv):
                    v1mo = lib.einsum('pq,qo,pv->ov', v1ao[i,kp], orbo[k], orbv[kp].conj())
                    v1mo += e_ia[k] * dm1[k]
                    v1[k] = v1mo.ravel()
                v1s.append( numpy.concatenate(v1) )
            return numpy.stack(v1s)
        return vind, hdiag

    def get_init_guess(self, mf, kshift, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]
        e_ia = numpy.concatenate( [x.reshape(-1) for x in
                                   _get_e_ia(mo_energy, mo_occ, kconserv)] )

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = numpy.partition(e_ia, nstates-1)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None):
        '''TDA diagonalization solver

        Args:
            x0: list of init guess arrays for each k-shift specified in :attr:`self.kshift_lst`
                [x0_1, x0_2, ..., x0_nshift]
            x0_i ~ (nstates, nkpts*nocc*nvir)
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()

        log = logger.new_logger(self)

        mf = self._scf
        mo_occ = mf.mo_occ

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        log = logger.Logger(self.stdout, self.verbose)

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.get_init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, e, x1 = lr_eigh(
                vind, x0k, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, pick=pickeig, max_cycle=self.max_cycle,
                max_memory=self.max_memory, verbose=log)
            self.converged.append( converged )
            self.e.append( e )
            # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
            self.xy.append( [(_unpack(xi*numpy.sqrt(.5), mo_occ, kconserv), 0) for xi in x1] )

        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
CIS = KTDA = TDA


class TDHF(KTDBase):

    get_ab = TDA.get_ab

    def gen_vind(self, mf, kshift=0):
        '''
        [ A   B ][X]
        [-B* -A*][Y]
        '''
        assert mf is self._scf
        assert kshift == 0

        singlet = self.singlet

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff[0].shape
        occidx = [mo_occ[k]==2 for k in range(nkpts)]
        viridx = [mo_occ[k]==0 for k in range(nkpts)]
        orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
        orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpts)]
        dtype = numpy.result_type(*mo_coeff)

        kconserv = numpy.arange(nkpts)
        e_ia = _get_e_ia(scf.addons.mo_energy_with_exxdiv_none(mf), mo_occ, kconserv)
        hdiag = numpy.hstack([x.ravel() for x in e_ia])
        tot_x = hdiag.size
        hdiag = numpy.hstack((hdiag, -hdiag))

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = self.gen_response(singlet=singlet, hermi=0, max_memory=max_memory)

        def vind(xys):
            nz = len(xys)
            z1xs = [_unpack(xy[:tot_x], mo_occ, kconserv) for xy in xys]
            z1ys = [_unpack(xy[tot_x:], mo_occ, kconserv) for xy in xys]
            dms = numpy.zeros((nz,nkpts,nao,nao), dtype=dtype)
            for i in range(nz):
                for k in range(nkpts):
                    # *2 for double occupancy
                    dmx = z1xs[i][k] * 2
                    dmy = z1ys[i][k] * 2
                    dms[i,k] += lib.einsum('ov,pv,qo->pq', dmx, orbv[k], orbo[k].conj())
                    dms[i,k] += lib.einsum('ov,qv,po->pq', dmy, orbv[k].conj(), orbo[k])

            with lib.temporary_env(mf, exxdiv=None):
                v1ao = vresp(dms, kshift) # = <mb||nj> Xjb + <mj||nb> Yjb
            v1s = []
            for i in range(nz):
                dmx = z1xs[i]
                dmy = z1ys[i]
                v1xs = [0] * nkpts
                v1ys = [0] * nkpts
                for k in range(nkpts):
                    # AX + BY
                    # = <aj||ib> Xjb + <ab||ij> Yjb
                    # = (<mj||nb> Xjb + <mb||nj> Yjb) Cma* Cni
                    v1x = lib.einsum('pq,qo,pv->ov', v1ao[i,k], orbo[k], orbv[k].conj())
                    # (B*)X + (A*)Y
                    # = <ij||ab> Xjb + <ib||aj> Yjb
                    # = (<mj||nb> Xjb + <mb||nj> Yjb) Cmi* Cna
                    v1y = lib.einsum('pq,po,qv->ov', v1ao[i,k], orbo[k].conj(), orbv[k])
                    v1x += e_ia[k] * dmx[k]
                    v1y += e_ia[k] * dmy[k]
                    v1xs[k] += v1x.ravel()
                    v1ys[k] -= v1y.ravel()
                v1s.append( numpy.concatenate(v1xs + v1ys) )
            return numpy.stack(v1s)
        return vind, hdiag

    def get_init_guess(self, mf, kshift, nstates=None):
        x0 = TDA.get_init_guess(self, mf, kshift, nstates)
        y0 = numpy.zeros_like(x0)
        return numpy.hstack([x0, y0])

    get_precond = rhf.TDHF.get_precond

    def kernel(self, x0=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()

        log = logger.new_logger(self)

        mf = self._scf
        mo_occ = mf.mo_occ

        real_system = (is_gamma_point(self._scf.kpts) and
                       self._scf.mo_coeff[0].dtype == numpy.double)

        if any(k != 0 for k in self.kshift_lst):
            # It's not clear how to define the Y matrix for kshift!=0 .
            # When the A tensor is constructed against the X(kshift) matrix,
            # the diagonal terms e_ia are calculated as e_i[k] - e_k[k+kshift].
            # Given the k-conserve relation in the A tensor, the j-b indices in
            # the A tensor should follow j[k'], b[k'+kshift]. This leads to the
            # j-b indices in the B tensor being defined as (j[k'+shift], b[k']).
            # To form the square A-B-B-A matrix, the diagonal terms for the
            # -A* part need to be constructed as e_i[k+kshift] - e_a[k], which
            # conflict to the diagonal terms of the A tensor.
            raise RuntimeError('kshift != 0 for TDHF')

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        log = logger.Logger(self.stdout, self.verbose)

        def norm_xy(z, kconserv):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm < 0:
                log.warn('TDDFT amplitudes |X| smaller than |Y|')
            norm = abs(.5/norm)**.5
            x *= norm
            y *= norm
            return _unpack(x, mo_occ, kconserv), _unpack(y, mo_occ, kconserv)

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.get_init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, e, x1 = lr_eig(
                vind, x0k, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, pick=pickeig, max_cycle=self.max_cycle,
                max_memory=self.max_memory, verbose=log)
            self.converged.append( converged )
            self.e.append( e )
            self.xy.append( [norm_xy(z, kconserv) for z in x1] )

        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
RPA = KTDHF = TDHF

def _get_e_ia(mo_energy, mo_occ, kconserv=None):
    nkpts = len(mo_occ)
    e_ia = [None] * nkpts
    if kconserv is None: kconserv = numpy.arange(nkpts)
    for k, kp in enumerate(kconserv):
        moeocc = mo_energy[k][mo_occ[k] > 1e-6]
        moevir = mo_energy[kp][mo_occ[kp] < 1e-6]
        e_ia[k] = -moeocc[:,None] + moevir
    return e_ia

def _unpack(vo, mo_occ, kconserv):
    z = []
    p1 = 0
    no_kpts = [numpy.count_nonzero(occ) for occ in mo_occ]
    for k, no in enumerate(no_kpts):
        kp = kconserv[k]
        nv = mo_occ[kp].size - no_kpts[kp]
        p0, p1 = p1, p1 + no * nv
        z.append(vo[p0:p1].reshape(no,nv))
    return z


scf.khf.KRHF.TDA  = lib.class_as_method(KTDA)
scf.khf.KRHF.TDHF = lib.class_as_method(KTDHF)
scf.krohf.KROHF.TDA  = NotImplemented
scf.krohf.KROHF.TDHF = NotImplemented
