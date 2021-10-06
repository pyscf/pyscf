#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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


import numpy
from pyscf import lib
from pyscf.tdscf import dhf
from pyscf.data import nist
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__

# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)


class TDA(dhf.TDA):
    pass

class TDDFT(dhf.TDHF):
    pass

RPA = TDDKS = TDDFT

class TDDFTNoHybrid(TDDFT, TDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    init_guess = TDA.init_guess

    def gen_vind(self, mf):
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        n2c = nmo // 2
        occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
        viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        e_ia = (mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]).T
        d_ia = numpy.sqrt(e_ia)
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
            dmov = lib.einsum('xov,ov,po,qv->xpq', zs, d_ia, orbo, orbv.conj())
            # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            dmov = dmov + dmov.conj().transpose(0,2,1)

            v1ao = vresp(dmov)
            v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)

            # numpy.sqrt(e_ia) * (e_ia*d_ia*z + v1ov)
            v1ov += numpy.einsum('xov,ov->xov', zs, ed_ia)
            v1ov *= d_ia
            return v1ov.reshape(v1ov.shape[0],-1)

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        cpu0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError(f'{self.__class__} cannot be used with hybrid functional')

        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = lib.logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > POSTIVE_EIG_THRESHOLD**2)[0]
            return w[idx], v[:,idx], idx

        self.converged, w2, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_cycle=self.max_cycle,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        mo_energy = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        n2c = mo_occ.size // 2
        occidx = n2c + numpy.where(mo_occ[n2c:] == 1)[0]
        viridx = n2c + numpy.where(mo_occ[n2c:] == 0)[0]
        e_ia = (mo_energy[viridx,None] - mo_energy[occidx]).T
        e_ia = numpy.sqrt(e_ia)
        def norm_xy(w, z):
            zp = e_ia * z.reshape(e_ia.shape)
            zm = w/e_ia * z.reshape(e_ia.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(1./norm)
            return (x*norm, y*norm)

        idx = numpy.where(w2 > POSTIVE_EIG_THRESHOLD**2)[0]
        self.e = numpy.sqrt(w2[idx])
        self.xy = [norm_xy(self.e[i], x1[i]) for i in idx]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy


def tddft(mf):
    '''Driver to create TDDFT or TDDFTNoHybrid object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf)
    else:
        return TDDFTNoHybrid(mf)

from pyscf import dft
dft.dks.DKS.TDA           = lib.class_as_method(TDA)
dft.dks.DKS.TDHF          = None
dft.dks.DKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.dks.DKS.TDDFT         = tddft
