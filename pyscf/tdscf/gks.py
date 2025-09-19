#!/usr/bin/env python
# Copyright 2021-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf import symm
from pyscf.tdscf import ghf, rhf
from pyscf.tdscf._lr_eig import eigh as lr_eigh
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__


class TDA(ghf.TDA):
    pass

class TDDFT(ghf.TDHF):
    pass

RPA = TDGKS = TDDFT

class CasidaTDDFT(TDDFT, TDA):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''

    get_init_guess = TDA.get_init_guess

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        wfnsym = self.wfnsym
        mol = mf.mol
        mask = self.get_frozen_mask()
        mo_coeff = mf.mo_coeff[:, mask]
        assert mo_coeff.dtype == numpy.double
        mo_energy = mf.mo_energy[mask]
        mo_occ = mf.mo_occ[mask]
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==1)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        if wfnsym is not None and mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            sym_forbid = ghf._get_x_sym_table(self) != wfnsym

        e_ia = (mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]).T
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia)
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = self.gen_response(mo_coeff, mo_occ, hermi=1)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
            if wfnsym is not None and mol.symmetry:
                zs = numpy.copy(zs)
                zs[:,sym_forbid] = 0

            dms = lib.einsum('xov,pv,qo->xpq', zs*d_ia, orbv, orbo)
            # +cc for A+B because K_{ai,jb} in A == K_{ai,bj} in B
            dms = dms + dms.transpose(0,2,1)

            v1ao = vresp(dms)
            v1mo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv)

            # numpy.sqrt(e_ia) * (e_ia*d_ia*z + v1mo)
            v1mo += numpy.einsum('xov,ov->xov', zs, ed_ia)
            v1mo *= d_ia
            if wfnsym is not None and mol.symmetry:
                v1mo[:,sym_forbid] = 0
            return v1mo.reshape(v1mo.shape[0],-1)

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        cpu0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        mol = self.mol
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be used with hybrid functional'
                               % self.__class__)
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = lib.logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0, x0sym = self.get_init_guess(
                self._scf, self.nstates, return_symmetry=True)
        elif mol.symmetry:
            x_sym = ghf._get_x_sym_table(self).ravel()
            x0sym = [rhf._guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, w2, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mask = self.get_frozen_mask()
        mo_energy = self._scf.mo_energy[mask]
        mo_occ = self._scf.mo_occ[mask]
        occidx = numpy.where(mo_occ==1)[0]
        viridx = numpy.where(mo_occ==0)[0]
        e_ia = (mo_energy[viridx,None] - mo_energy[occidx]).T
        e_ia = numpy.sqrt(e_ia)
        def norm_xy(w, z):
            zp = e_ia * z.reshape(e_ia.shape)
            zm = w/e_ia * z.reshape(e_ia.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm < 0:
                log.warn('TDDFT amplitudes |X| smaller than |Y|')
            norm = abs(norm)**-.5
            return (x*norm, y*norm)

        idx = numpy.where(w2 > self.positive_eig_threshold)[0]
        self.e = numpy.sqrt(w2[idx])
        self.xy = [norm_xy(self.e[i], x1[i]) for i in idx]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDDFTNoHybrid = CasidaTDDFT


def tddft(mf, frozen=None):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if (not mf._numint.libxc.is_hybrid_xc(mf.xc) and
        # Casida formula can be applied for real orbitals only
        mf.mo_coeff.dtype == numpy.double and mf.collinear[0] != 'm'):
        return CasidaTDDFT(mf, frozen)
    else:
        return TDDFT(mf, frozen)

from pyscf import dft
dft.gks.GKS.TDA           = dft.gks_symm.GKS.TDA           = lib.class_as_method(TDA)
dft.gks.GKS.TDHF          = dft.gks_symm.GKS.TDHF          = None
dft.gks.GKS.TDDFTNoHybrid = dft.gks_symm.GKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.gks.GKS.CasidaTDDFT   = dft.gks_symm.GKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.gks.GKS.TDDFT         = dft.gks_symm.GKS.TDDFT         = tddft
