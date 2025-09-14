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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import symm
from pyscf.tdscf import rhf
from pyscf.tdscf._lr_eig import eigh as lr_eigh
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__


class TDA(rhf.TDA):
    def Gradients(self):
        if getattr(self._scf, 'with_df', None):
            logger.warn(self, 'TDDFT Gradients with DF approximation is not available. '
                        'TDDFT Gradients are computed using exact integrals')
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)

class TDDFT(rhf.TDHF):
    Gradients = TDA.Gradients

RPA = TDRKS = TDDFT

class CasidaTDDFT(TDDFT, TDA):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''

    get_init_guess = TDA.get_init_guess
    get_precond = TDA.get_precond

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mask = self.get_frozen_mask()
        mo_coeff = mf.mo_coeff[:, mask]
        assert mo_coeff.dtype == numpy.double
        mo_energy = mf.mo_energy[mask]
        mo_occ = mf.mo_occ[mask]
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        if wfnsym is not None and mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            sym_forbid = rhf._get_x_sym_table(self) != wfnsym

        e_ia = (mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]).T
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia)
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = self.gen_response(singlet=singlet, hermi=1)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
            # *2 for double occupancy
            dms = lib.einsum('xov,pv,qo->xpq', zs * (d_ia*2), orbv, orbo)
            # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            dms = dms + dms.transpose(0,2,1)

            v1ao = vresp(dms)
            v1mo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo, orbv)

            # numpy.sqrt(e_ia) * (e_ia*d_ia*z + v1mo)
            v1mo += numpy.einsum('xov,ov->xov', zs, ed_ia)
            v1mo *= d_ia
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
            x_sym = rhf._get_x_sym_table(self).ravel()
            x0sym = [rhf._guess_wfnsym_id(self, x_sym, x) for x in x0]

        self.converged, w2, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mask = self.get_frozen_mask()
        mo_energy = self._scf.mo_energy[mask]
        mo_occ = self._scf.mo_occ[mask]
        occidx = numpy.where(mo_occ==2)[0]
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
            norm = abs(.5/norm)**.5  # normalize to 0.5 for alpha spin
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

class dRPA(TDDFTNoHybrid):
    def __init__(self, mf, frozen=None):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct RPA can only be applied with DFT; for HF+dRPA, use .xc='hf'")
        mf = mf.to_rks()
        # commit fc8d1967995b7e033b60d4428ddcca87aac78e4f handles xc='' .
        # xc='0*LDA' is equivalent to xc=''
        #mf.xc = '0.0*LDA'
        mf.xc = ''
        TDDFTNoHybrid.__init__(self, mf, frozen)

TDH = dRPA

class dTDA(TDA):
    def __init__(self, mf, frozen=None):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct TDA can only be applied with DFT; for HF+dTDA, use .xc='hf'")
        mf = mf.to_rks()
        # commit fc8d1967995b7e033b60d4428ddcca87aac78e4f handles xc='' .
        # xc='0*LDA' is equivalent to xc=''
        #mf.xc = '0.0*LDA'
        mf.xc = ''
        TDA.__init__(self, mf, frozen)


def tddft(mf, frozen=None):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf, frozen)
    else:
        return CasidaTDDFT(mf, frozen)

from pyscf import dft
dft.rks.RKS.TDA           = dft.rks_symm.RKS.TDA           = lib.class_as_method(TDA)
dft.rks.RKS.TDHF          = dft.rks_symm.RKS.TDHF          = None
#dft.rks.RKS.TDDFT         = dft.rks_symm.RKS.TDDFT         = lib.class_as_method(TDDFT)
dft.rks.RKS.TDDFTNoHybrid = dft.rks_symm.RKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.rks.RKS.CasidaTDDFT   = dft.rks_symm.RKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.rks.RKS.TDDFT         = dft.rks_symm.RKS.TDDFT         = tddft
dft.rks.RKS.dTDA          = dft.rks_symm.RKS.dTDA          = lib.class_as_method(dTDA)
dft.rks.RKS.dRPA          = dft.rks_symm.RKS.dRPA          = lib.class_as_method(dRPA)
dft.roks.ROKS.TDA           = dft.rks_symm.ROKS.TDA           = None
dft.roks.ROKS.TDHF          = dft.rks_symm.ROKS.TDHF          = None
dft.roks.ROKS.TDDFT         = dft.rks_symm.ROKS.TDDFT         = None
dft.roks.ROKS.TDDFTNoHybrid = dft.rks_symm.ROKS.TDDFTNoHybrid = None
dft.roks.ROKS.dTDA          = dft.rks_symm.ROKS.dTDA          = None
dft.roks.ROKS.dRPA          = dft.rks_symm.ROKS.dRPA          = None
