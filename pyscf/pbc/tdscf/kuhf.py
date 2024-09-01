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

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.tdscf import uhf
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig
from pyscf.pbc import scf
from pyscf.pbc.tdscf.krhf import KTDBase, _get_e_ia
from pyscf.pbc.lib.kpts_helper import is_gamma_point, get_kconserv_ria
from pyscf.pbc.scf import _response_functions  # noqa
from pyscf import __config__

REAL_EIG_THRESHOLD = getattr(__config__, 'pbc_tdscf_uhf_TDDFT_pick_eig_threshold', 1e-3)

class TDA(KTDBase):

    def get_ab(self, mf=None, kshift=0):
        raise NotImplementedError

    def gen_vind(self, mf, kshift=0):
        '''Compute Ax

        Kwargs:
            kshift : integer
                The index of the k-point that represents the transition between
                k-points in the excitation coefficients.
        '''
        kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ[0])
        nao, nmo = mo_coeff[0][0].shape
        occidxa = [numpy.where(mo_occ[0][k]> 0)[0] for k in range(nkpts)]
        occidxb = [numpy.where(mo_occ[1][k]> 0)[0] for k in range(nkpts)]
        viridxa = [numpy.where(mo_occ[0][k]==0)[0] for k in range(nkpts)]
        viridxb = [numpy.where(mo_occ[1][k]==0)[0] for k in range(nkpts)]
        orboa = [mo_coeff[0][k][:,occidxa[k]] for k in range(nkpts)]
        orbob = [mo_coeff[1][k][:,occidxb[k]] for k in range(nkpts)]
        orbva = [mo_coeff[0][kconserv[k]][:,viridxa[kconserv[k]]] for k in range(nkpts)]
        orbvb = [mo_coeff[1][kconserv[k]][:,viridxb[kconserv[k]]] for k in range(nkpts)]

        moe = scf.addons.mo_energy_with_exxdiv_none(mf)
        e_ia_a = _get_e_ia(moe[0], mo_occ[0], kconserv)
        e_ia_b = _get_e_ia(moe[1], mo_occ[1], kconserv)
        hdiag = numpy.hstack([x.ravel() for x in (e_ia_a + e_ia_b)])

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = mf.gen_response(hermi=0, max_memory=max_memory)

        def vind(zs):
            nz = len(zs)
            zs = [_unpack(z, mo_occ, kconserv) for z in zs]
            dmov = numpy.empty((2,nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                dm1a, dm1b = zs[i]
                for k in range(nkpts):
                    dmov[0,i,k] = reduce(numpy.dot, (orboa[k], dm1a[k], orbva[k].conj().T))
                    dmov[1,i,k] = reduce(numpy.dot, (orbob[k], dm1b[k], orbvb[k].conj().T))

            with lib.temporary_env(mf, exxdiv=None):
                dmov = dmov.reshape(2,nz,nkpts,nao,nao)
                v1ao = vresp(dmov, kshift)
                v1ao = v1ao.reshape(2,nz,nkpts,nao,nao)

            v1s = []
            for i in range(nz):
                dm1a, dm1b = zs[i]
                v1as = []
                v1bs = []
                for k in range(nkpts):
                    v1a = reduce(numpy.dot, (orboa[k].conj().T, v1ao[0,i,k], orbva[k]))
                    v1b = reduce(numpy.dot, (orbob[k].conj().T, v1ao[1,i,k], orbvb[k]))
                    v1a += e_ia_a[k] * dm1a[k]
                    v1b += e_ia_b[k] * dm1b[k]
                    v1as.append(v1a.ravel())
                    v1bs.append(v1b.ravel())
                v1s.append( numpy.concatenate(v1as + v1bs) )
            return lib.asarray(v1s).reshape(nz,-1)

        return vind, hdiag

    def init_guess(self, mf, kshift, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]
        e_ia_a = _get_e_ia(mo_energy[0], mo_occ[0], kconserv)
        e_ia_b = _get_e_ia(mo_energy[1], mo_occ[1], kconserv)
        e_ia = numpy.hstack([x.ravel() for x in (e_ia_a + e_ia_b)])

        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = numpy.sort(e_ia)[nstates-1]
        e_threshold += self.deg_eia_thresh

        idx = numpy.where(e_ia <= e_threshold)[0]
        x0 = numpy.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None):
        '''TDA diagonalization solver
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
                x0k = self.init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, e, x1 = lr_eigh(
                vind, x0k, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, pick=pickeig, max_cycle=self.max_cycle,
                max_memory=self.max_memory, verbose=log)
            self.converged.append( converged )
            self.e.append( e )
            self.xy.append( [(_unpack(xi, mo_occ, kconserv),  # (X_alpha, X_beta)
                        (0, 0))  # (Y_alpha, Y_beta)
                       for xi in x1] )
        #TODO: analyze CIS wfn point group symmetry
        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
CIS = KTDA = TDA


class TDHF(KTDBase):

    def get_ab(self, mf=None, kshift=0):
        raise NotImplementedError

    def gen_vind(self, mf, kshift=0):
        assert kshift == 0

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ[0])
        nao, nmo = mo_coeff[0][0].shape
        occidxa = [numpy.where(mo_occ[0][k]> 0)[0] for k in range(nkpts)]
        occidxb = [numpy.where(mo_occ[1][k]> 0)[0] for k in range(nkpts)]
        viridxa = [numpy.where(mo_occ[0][k]==0)[0] for k in range(nkpts)]
        viridxb = [numpy.where(mo_occ[1][k]==0)[0] for k in range(nkpts)]
        orboa = [mo_coeff[0][k][:,occidxa[k]] for k in range(nkpts)]
        orbob = [mo_coeff[1][k][:,occidxb[k]] for k in range(nkpts)]
        orbva = [mo_coeff[0][k][:,viridxa[k]] for k in range(nkpts)]
        orbvb = [mo_coeff[1][k][:,viridxb[k]] for k in range(nkpts)]

        kconserv = numpy.arange(nkpts)
        moe = scf.addons.mo_energy_with_exxdiv_none(mf)
        e_ia_a = _get_e_ia(moe[0], mo_occ[0], kconserv)
        e_ia_b = _get_e_ia(moe[1], mo_occ[1], kconserv)
        hdiag = numpy.hstack([x.ravel() for x in (e_ia_a + e_ia_b)])
        hdiag = numpy.hstack((hdiag, -hdiag))
        tot_x_a = sum(x.size for x in e_ia_a)
        tot_x_b = sum(x.size for x in e_ia_b)
        tot_x = tot_x_a + tot_x_b

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = mf.gen_response(hermi=0, max_memory=max_memory)

        def vind(xys):
            nz = len(xys)
            x1s = [_unpack(x[:tot_x], mo_occ, kconserv) for x in xys]
            y1s = [_unpack(x[tot_x:], mo_occ, kconserv) for x in xys]
            dmov = numpy.empty((2,nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                xa, xb = x1s[i]
                ya, yb = y1s[i]
                for k in range(nkpts):
                    dmx = reduce(numpy.dot, (orboa[k], xa[k]  , orbva[k].conj().T))
                    dmy = reduce(numpy.dot, (orbva[k], ya[k].T, orboa[k].conj().T))
                    dmov[0,i,k] = dmx + dmy
                    dmx = reduce(numpy.dot, (orbob[k], xb[k]  , orbvb[k].conj().T))
                    dmy = reduce(numpy.dot, (orbvb[k], yb[k].T, orbob[k].conj().T))
                    dmov[1,i,k] = dmx + dmy

            with lib.temporary_env(mf, exxdiv=None):
                dmov = dmov.reshape(2,nz,nkpts,nao,nao)
                v1ao = vresp(dmov, kshift)
                v1ao = v1ao.reshape(2,nz,nkpts,nao,nao)

            v1s = []
            for i in range(nz):
                xa, xb = x1s[i]
                ya, yb = y1s[i]
                v1xsa = [0] * nkpts
                v1xsb = [0] * nkpts
                v1ysa = [0] * nkpts
                v1ysb = [0] * nkpts
                for k in range(nkpts):
                    v1xa = reduce(numpy.dot, (orboa[k].conj().T, v1ao[0,i,k], orbva[k]))
                    v1xb = reduce(numpy.dot, (orbob[k].conj().T, v1ao[1,i,k], orbvb[k]))
                    v1ya = reduce(numpy.dot, (orbva[k].conj().T, v1ao[0,i,k], orboa[k])).T
                    v1yb = reduce(numpy.dot, (orbvb[k].conj().T, v1ao[1,i,k], orbob[k])).T
                    v1xa += e_ia_a[k] * xa[k]
                    v1xb += e_ia_b[k] * xb[k]
                    v1ya += e_ia_a[k].conj() * ya[k]
                    v1yb += e_ia_b[k].conj() * yb[k]
                    v1xsa[k] += v1xa.ravel()
                    v1xsb[k] += v1xb.ravel()
                    v1ysa[k] += -v1ya.ravel()
                    v1ysb[k] += -v1yb.ravel()
                v1s.append( numpy.concatenate(v1xsa + v1xsb + v1ysa + v1ysb) )
            return numpy.hstack(v1s).reshape(nz,-1)

        return vind, hdiag

    def init_guess(self, mf, kshift, nstates=None, wfnsym=None):
        x0 = TDA.init_guess(self, mf, kshift, nstates)
        y0 = numpy.zeros_like(x0)
        return numpy.hstack([x0, y0])

    get_precond = uhf.TDHF.get_precond

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
                       self._scf.mo_coeff[0][0].dtype == numpy.double)

        if any(k != 0 for k in self.kshift_lst):
            raise RuntimeError('kshift != 0 for TDHF')

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = get_kconserv_ria(mf.cell, mf.kpts)[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, w, x1 = lr_eig(
                vind, x0k, precond, tol_residual=self.conv_tol, lindep=self.lindep,
                nroots=self.nstates, pick=pickeig, max_cycle=self.max_cycle,
                max_memory=self.max_memory, verbose=log)
            self.converged.append( converged )

            e = []
            xy = []
            for i, z in enumerate(x1):
                xs, ys = z.reshape(2,-1)
                norm = lib.norm(xs)**2 - lib.norm(ys)**2
                if norm > 0:
                    norm = 1/numpy.sqrt(norm)
                    xs *= norm
                    ys *= norm
                    e.append(w[i])
                    xy.append((_unpack(xs, mo_occ, kconserv), _unpack(ys, mo_occ, kconserv)))
            self.e.append( numpy.array(e) )
            self.xy.append( xy )

        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
RPA = KTDHF = TDHF

def _unpack(vo, mo_occ, kconserv):
    za = []
    zb = []
    p1 = 0
    for k, occ in enumerate(mo_occ[0]):
        no = numpy.count_nonzero(occ > 0)
        no1 = numpy.count_nonzero(mo_occ[0][kconserv[k]] > 0)
        nv = occ.size - no1
        p0, p1 = p1, p1 + no * nv
        za.append(vo[p0:p1].reshape(no,nv))

    for k, occ in enumerate(mo_occ[1]):
        no = numpy.count_nonzero(occ > 0)
        no1 = numpy.count_nonzero(mo_occ[1][kconserv[k]] > 0)
        nv = occ.size - no1
        p0, p1 = p1, p1 + no * nv
        zb.append(vo[p0:p1].reshape(no,nv))
    return za, zb


scf.kuhf.KUHF.TDA  = lib.class_as_method(KTDA)
scf.kuhf.KUHF.TDHF = lib.class_as_method(KTDHF)
