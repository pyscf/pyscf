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
from pyscf.pbc import scf
from pyscf.pbc.tdscf.rhf import TDBase
from pyscf.pbc.scf import _response_functions  # noqa
from pyscf.pbc.lib.kpts_helper import gamma_point, get_kconserv_ria
from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
from pyscf.pbc import df as pbcdf
from pyscf.data import nist
from pyscf import __config__

REAL_EIG_THRESHOLD = getattr(__config__, 'pbc_tdscf_rhf_TDDFT_pick_eig_threshold', 1e-3)

class KTDBase(TDBase):
    _keys = {'kconserv', 'kshift_lst'}

    def __init__(self, mf, kshift_lst=None):
        assert isinstance(mf, scf.khf.KSCF)
        TDBase.__init__(self, mf)
        warn_pbc2d_eri(mf)

        if kshift_lst is None: kshift_lst = [0]

        self.kconserv = get_kconserv_ria(mf.cell, mf.kpts)
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
        log.info('eigh max_space = %d', self.max_space)
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
        if any([k != 0 for k in self.kshift_lst]):
            if mf.rsjk is not None or not isinstance(mf.with_df, pbcdf.df.DF):
                logger.error(self, 'Solutions with non-zero kshift for %s are '
                             'only supported by GDF/RSDF')
                raise NotImplementedError

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
    conv_tol = getattr(__config__, 'pbc_tdscf_rhf_TDA_conv_tol', 1e-6)

    def gen_vind(self, mf, kshift):
        # exxdiv corrections are kept in hdiag while excluding them when calling
        # the contractions between two-electron integrals and X/Y amplitudes.
        # See also the relevant comments in function pbc.tdscf.rhf.TDA.gen_vind
        singlet = self.singlet
        kconserv = self.kconserv[kshift]

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff[0].shape
        occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
        viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]
        orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
        orbv = [mo_coeff[kconserv[k]][:,viridx[kconserv[k]]] for k in range(nkpts)]
        e_ia = _get_e_ia(scf.addons.mo_energy_with_exxdiv_none(mf), mo_occ, kconserv)
        hdiag = numpy.hstack([x.ravel() for x in e_ia])

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = mf.gen_response(singlet=singlet, hermi=0, max_memory=max_memory)

        def vind(zs):
            nz = len(zs)
            z1s = [_unpack(z, mo_occ, kconserv) for z in zs]
            dmov = numpy.empty((nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                for k in range(nkpts):
                    # *2 for double occupancy
                    dm1 = z1s[i][k] * 2
                    dmov[i,k] = reduce(numpy.dot, (orbo[k], dm1, orbv[k].conj().T))

            with lib.temporary_env(mf, exxdiv=None):
                v1ao = vresp(dmov, kshift)
            v1s = []
            for i in range(nz):
                dm1 = z1s[i]
                v1 = []
                for k in range(nkpts):
                    v1vo = reduce(numpy.dot, (orbo[k].conj().T, v1ao[i,k], orbv[k]))
                    v1vo += e_ia[k] * dm1[k]
                    v1.append(v1vo.ravel())
                v1s.append( numpy.concatenate(v1) )
            return lib.asarray(v1s).reshape(nz,-1)
        return vind, hdiag

    def init_guess(self, mf, kshift, nstates=None):
        if nstates is None: nstates = self.nstates

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        kconserv = self.kconserv[kshift]
        e_ia = numpy.concatenate( [x.reshape(-1) for x in
                                   _get_e_ia(mo_energy, mo_occ, kconserv)] )

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
        precision = self.cell.precision * 1e-2

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = self.kconserv[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, e, x1 = \
                    lib.davidson1(vind, x0k, precond,
                                  tol=self.conv_tol,
                                  max_cycle=self.max_cycle,
                                  nroots=self.nstates, lindep=self.lindep,
                                  max_space=self.max_space, pick=pickeig,
                                  fill_heff=purify_krlyov_heff(precision, 0, log),
                                  verbose=self.verbose)
            self.converged.append( converged )
            self.e.append( e )
            # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
            self.xy.append( [(_unpack(xi*numpy.sqrt(.5), mo_occ, kconserv), 0) for xi in x1] )

        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
CIS = KTDA = TDA


class TDHF(TDA):
    def gen_vind(self, mf, kshift):
        '''
        [ A   B ][X]
        [-B* -A*][Y]
        '''
        singlet = self.singlet
        kconserv = self.kconserv[kshift]

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff[0].shape
        occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
        viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]
        orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
        orbv = [mo_coeff[kconserv[k]][:,viridx[kconserv[k]]] for k in range(nkpts)]
        e_ia = _get_e_ia(scf.addons.mo_energy_with_exxdiv_none(mf), mo_occ, kconserv)
        hdiag = numpy.hstack([x.ravel() for x in e_ia])
        tot_x = hdiag.size
        hdiag = numpy.hstack((hdiag, -hdiag))

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)
        vresp = mf.gen_response(singlet=singlet, hermi=0, max_memory=max_memory)

        def vind(xys):
            nz = len(xys)
            z1xs = [_unpack(xy[:tot_x], mo_occ, kconserv) for xy in xys]
            z1ys = [_unpack(xy[tot_x:], mo_occ, kconserv) for xy in xys]
            dmov = numpy.empty((nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                for k in range(nkpts):
                    # *2 for double occupancy
                    dmx = z1xs[i][k] * 2
                    dmy = z1ys[i][k] * 2
                    dmov[i,k] = reduce(numpy.dot, (orbo[k], dmx, orbv[k].T.conj()))
                    dmov[i,k]+= reduce(numpy.dot, (orbv[k], dmy.T, orbo[k].T.conj()))

            with lib.temporary_env(mf, exxdiv=None):
                v1ao = vresp(dmov, kshift)
            v1s = []
            for i in range(nz):
                dmx = z1xs[i]
                dmy = z1ys[i]
                v1xs = []
                v1ys = []
                for k in range(nkpts):
                    v1x = reduce(numpy.dot, (orbo[k].T.conj(), v1ao[i,k], orbv[k]))
                    v1y = reduce(numpy.dot, (orbv[k].T.conj(), v1ao[i,k], orbo[k])).T
                    v1x+= e_ia[k] * dmx[k]
                    v1y+= e_ia[k] * dmy[k]
                    v1xs.append(v1x.ravel())
                    v1ys.append(-v1y.ravel())
                # v1s += v1xs + v1ys
                v1s.append( numpy.concatenate(v1xs + v1ys) )
            return lib.asarray(v1s).reshape(nz,-1)
        return vind, hdiag

    def init_guess(self, mf, kshift, nstates=None):
        x0 = TDA.init_guess(self, mf, kshift, nstates)
        y0 = numpy.zeros_like(x0)
        return numpy.asarray(numpy.block([[x0, y0], [y0, x0.conj()]]))

    def kernel(self, x0=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()

        log = logger.new_logger(self)

        mf = self._scf
        mo_occ = mf.mo_occ

        real_system = (gamma_point(self._scf.kpts) and
                       self._scf.mo_coeff[0].dtype == numpy.double)

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        log = logger.Logger(self.stdout, self.verbose)
        precision = self.cell.precision * 1e-2
        hermi = 0

        def norm_xy(z, kconserv):
            x, y = z.reshape(2,-1)
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            x *= norm
            y *= norm
            return _unpack(x, mo_occ, kconserv), _unpack(y, mo_occ, kconserv)

        self.converged = []
        self.e = []
        self.xy = []
        for i,kshift in enumerate(self.kshift_lst):
            kconserv = self.kconserv[kshift]

            vind, hdiag = self.gen_vind(self._scf, kshift)
            precond = self.get_precond(hdiag)

            if x0 is None:
                x0k = self.init_guess(self._scf, kshift, self.nstates)
            else:
                x0k = x0[i]

            converged, e, x1 = \
                    lib.davidson_nosym1(vind, x0k, precond,
                                        tol=self.conv_tol,
                                        max_cycle=self.max_cycle,
                                        nroots=self.nstates,
                                        lindep=self.lindep,
                                        max_space=self.max_space, pick=pickeig,
                                        fill_heff=purify_krlyov_heff(precision, hermi, log),
                                        verbose=self.verbose)
            self.converged.append( converged )
            self.e.append( e )
            self.xy.append( [norm_xy(z, kconserv) for z in x1] )

        log.timer(self.__class__.__name__, *cpu0)
        self._finalize()
        return self.e, self.xy
RPA = KTDHF = TDHF

def _get_e_ia(mo_energy, mo_occ, kconserv=None):
    e_ia = []
    nkpts = len(mo_occ)
    if kconserv is None: kconserv = numpy.arange(nkpts)
    for k in range(nkpts):
        kp = kconserv[k]
        moeocc = mo_energy[k][mo_occ[k] > 1e-6]
        moevir = mo_energy[kp][mo_occ[kp] < 1e-6]
        e_ia.append( -moeocc[:,None] + moevir )
    return e_ia

def _unpack(vo, mo_occ, kconserv):
    z = []
    p1 = 0
    for k, occ in enumerate(mo_occ):
        no = numpy.count_nonzero(occ > 0)
        no1 = numpy.count_nonzero(mo_occ[kconserv[k]] > 0)
        nv = occ.size - no1
        p0, p1 = p1, p1 + no * nv
        z.append(vo[p0:p1].reshape(no,nv))
    return z

def purify_krlyov_heff(precision, hermi, log):
    def fill_heff(heff, xs, ax, xt, axt, dot):
        if hermi == 1:
            heff = linalg_helper._fill_heff_hermitian(heff, xs, ax, xt, axt, dot)
        else:
            heff = linalg_helper._fill_heff(heff, xs, ax, xt, axt, dot)
        space = len(axt)
        # TODO: PBC integrals has larger errors than molecule systems.
        # purify the effective Hamiltonian with symmetry and other
        # possible conditions.
        if abs(heff[:space,:space].imag).max() < precision:
            log.debug('Remove imaginary part of the Krylov space effective Hamiltonian')
            heff[:space,:space].imag = 0
        return heff
    return fill_heff


scf.khf.KRHF.TDA  = lib.class_as_method(KTDA)
scf.khf.KRHF.TDHF = lib.class_as_method(KTDHF)
scf.krohf.KROHF.TDA  = None
scf.krohf.KROHF.TDHF = None


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf
    from pyscf.pbc import df
    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = '''
    C  0.          0.          0.
    C  1.68506879  1.68506879  1.68506879
    '''
    cell.a = '''
    0.          3.37013758  3.37013758
    3.37013758  0.          3.37013758
    3.37013758  3.37013758  0.
    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [25]*3
    cell.build()
    mf = scf.KRHF(cell, cell.make_kpts([2,1,1])).set(exxdiv=None)
    #mf.with_df = df.MDF(cell, cell.make_kpts([2,1,1]))
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-mdf.h5'
    #mf.with_df.build(with_j3c=False)
    mf.run()
    #mesh=9  -8.65192427146353
    #mesh=12 -8.65192352289817
    #mesh=15 -8.6519235231529
    #MDF mesh=5 -8.6519301815144

    td = TDA(mf)
    td.verbose = 5
    print(td.kernel()[0][0] * 27.2114)
    #mesh=9  [ 6.0073749   6.09315355  6.3479901 ]
    #mesh=12 [ 6.00253282  6.09317929  6.34799109]
    #mesh=15 [ 6.00253396  6.09317949  6.34799109]
    #MDF mesh=5 [ 6.09317489  6.09318265  6.34798637]

    #from pyscf.pbc import tools
    #scell = tools.super_cell(cell, [2,1,1])
    #mf = scf.RHF(scell).run()
    #td = rhf.TDA(mf)
    #td.verbose = 5
    #print(td.kernel()[0] * 27.2114)

    td = TDHF(mf)
    td.verbose = 5
    print(td.kernel()[0][0] * 27.2114)
    #mesh=9  [ 6.03860914  6.21664545  8.20305225]
    #mesh=12 [ 6.03868259  6.03860343  6.2167623 ]
    #mesh=15 [ 6.03861321  6.03861324  6.21675868]
    #MDF mesh=5 [ 6.03861693  6.03861775  6.21675694]
