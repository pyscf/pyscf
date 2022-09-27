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


import numpy
from pyscf import symm
from pyscf import lib
from pyscf.lib import logger
from pyscf.tdscf import uhf
from pyscf.scf import uhf_symm
from pyscf.data import nist
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__


class TDA(uhf.TDA):
    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

class TDDFT(uhf.TDHF):
    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

RPA = TDUKS = TDDFT


class CasidaTDDFT(TDDFT, TDA):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''

    init_guess = TDA.init_guess

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        wfnsym = self.wfnsym

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert mo_coeff[0].dtype == numpy.double
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff[0].shape
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]

        if wfnsym is not None and mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
            orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsyma_in_d2h = numpy.asarray(orbsyma) % 10
            orbsymb_in_d2h = numpy.asarray(orbsymb) % 10
            sym_forbida = (orbsyma_in_d2h[occidxa,None] ^ orbsyma_in_d2h[viridxa]) != wfnsym
            sym_forbidb = (orbsymb_in_d2h[occidxb,None] ^ orbsymb_in_d2h[viridxb]) != wfnsym
            sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

        e_ia_a = (mo_energy[0][viridxa,None] - mo_energy[0][occidxa]).T
        e_ia_b = (mo_energy[1][viridxb,None] - mo_energy[1][occidxb]).T
        e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia).ravel()
        ed_ia = e_ia.ravel() * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

        def vind(zs):
            nz = len(zs)
            zs = numpy.asarray(zs).reshape(nz,-1)
            if wfnsym is not None and mol.symmetry:
                zs = numpy.copy(zs)
                zs[:,sym_forbid] = 0

            dmsa = (zs[:,:nocca*nvira] * d_ia[:nocca*nvira]).reshape(nz,nocca,nvira)
            dmsb = (zs[:,nocca*nvira:] * d_ia[nocca*nvira:]).reshape(nz,noccb,nvirb)
            dmsa = lib.einsum('xov,po,qv->xpq', dmsa, orboa, orbva.conj())
            dmsb = lib.einsum('xov,po,qv->xpq', dmsb, orbob, orbvb.conj())
            dmsa = dmsa + dmsa.conj().transpose(0,2,1)
            dmsb = dmsb + dmsb.conj().transpose(0,2,1)

            v1ao = vresp(numpy.asarray((dmsa,dmsb)))

            v1a = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
            v1b = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)

            hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
            hx += ed_ia * zs
            hx *= d_ia
            return hx

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
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
        log = logger.Logger(self.stdout, self.verbose)

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
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        e_ia_a = (mo_energy[0][viridxa,None] - mo_energy[0][occidxa]).T
        e_ia_b = (mo_energy[1][viridxb,None] - mo_energy[1][occidxb]).T
        e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
        e_ia = numpy.sqrt(e_ia)

        e = []
        xy = []
        for i, z in enumerate(x1):
            if w2[i] < self.positive_eig_threshold:
                continue
            w = numpy.sqrt(w2[i])
            zp = e_ia * z
            zm = w/e_ia * z
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                e.append(w)
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
        self.e = numpy.array(e)
        self.xy = xy

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

TDDFTNoHybrid = CasidaTDDFT


class dRPA(TDDFTNoHybrid):
    def __init__(self, mf):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct RPA can only be applied with DFT; for HF+dRPA, use .xc='hf'")
        mf = mf.to_uks()
        mf.xc = ''
        TDDFTNoHybrid.__init__(self, mf)

TDH = dRPA

class dTDA(TDA):
    def __init__(self, mf):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct TDA can only be applied with DFT; for HF+dTDA, use .xc='hf'")
        mf = mf.to_uks()
        mf.xc = ''
        TDA.__init__(self, mf)


def tddft(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf)
    else:
        return CasidaTDDFT(mf)

from pyscf import dft
dft.uks.UKS.TDA           = dft.uks_symm.UKS.TDA           = lib.class_as_method(TDA)
dft.uks.UKS.TDHF          = dft.uks_symm.UKS.TDHF          = None
#dft.uks.UKS.TDDFT         = dft.uks_symm.UKS.TDDFT         = lib.class_as_method(TDDFT)
dft.uks.UKS.TDDFTNoHybrid = dft.uks_symm.UKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.uks.UKS.CasidaTDDFT   = dft.uks_symm.UKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.uks.UKS.TDDFT         = dft.uks_symm.UKS.TDDFT         = tddft
dft.uks.UKS.dTDA          = dft.uks_symm.UKS.dTDA          = lib.class_as_method(dTDA)
dft.uks.UKS.dRPA          = dft.uks_symm.UKS.dRPA          = lib.class_as_method(dRPA)
