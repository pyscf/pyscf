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
from pyscf import symm
from pyscf.tdscf import rhf
from pyscf.scf import hf_symm
from pyscf.data import nist
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__


class TDA(rhf.TDA):
    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)

class TDDFT(rhf.TDHF):
    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)

RPA = TDRKS = TDDFT

class CasidaTDDFT(TDDFT, TDA):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    init_guess = TDA.init_guess

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert mo_coeff.dtype == numpy.double
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
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
            orbsym = hf_symm.get_orbsym(mol, mo_coeff)
            orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
            sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

        e_ia = (mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]).T
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia)
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = mf.gen_response(singlet=singlet, hermi=1)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
            # *2 for double occupancy
            dmov = lib.einsum('xov,ov,po,qv->xpq', zs, d_ia*2, orbo, orbv.conj())
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
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > self.positive_eig_threshold)[0]
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
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
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

    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)

TDDFTNoHybrid = CasidaTDDFT

class dRPA(TDDFTNoHybrid):
    def __init__(self, mf):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct RPA can only be applied with DFT; for HF+dRPA, use .xc='hf'")
        mf = mf.to_rks()
        # commit fc8d1967995b7e033b60d4428ddcca87aac78e4f handles xc='' .
        # xc='0*LDA' is equivalent to xc=''
        #mf.xc = '0.0*LDA'
        mf.xc = ''
        TDDFTNoHybrid.__init__(self, mf)

TDH = dRPA

class dTDA(TDA):
    def __init__(self, mf):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct TDA can only be applied with DFT; for HF+dTDA, use .xc='hf'")
        mf = mf.to_rks()
        # commit fc8d1967995b7e033b60d4428ddcca87aac78e4f handles xc='' .
        # xc='0*LDA' is equivalent to xc=''
        #mf.xc = '0.0*LDA'
        mf.xc = ''
        TDA.__init__(self, mf)


def tddft(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf)
    else:
        return CasidaTDDFT(mf)

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
