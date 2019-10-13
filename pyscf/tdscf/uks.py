#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

import time
from functools import reduce
import numpy
from pyscf import symm
from pyscf import lib
from pyscf.dft import numint
from pyscf.tdscf import uhf
from pyscf.scf import uhf_symm
from pyscf.data import nist
from pyscf.ao2mo import _ao2mo
from pyscf.soscf.newton_ah import _gen_uhf_response
from pyscf import __config__

# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)


class TDA(uhf.TDA):
    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

class TDDFT(uhf.TDHF):
    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

RPA = TDUKS = TDDFT


class TDDFTNoHybrid(TDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    def get_vind(self, mf):
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert(mo_coeff[0].dtype == numpy.double)
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
            orbsyma = orbsyma % 10
            orbsymb = orbsymb % 10
            sym_forbida = (orbsyma[occidxa,None] ^ orbsyma[viridxa]) != wfnsym
            sym_forbidb = (orbsymb[occidxb,None] ^ orbsymb[viridxb]) != wfnsym
            sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

        e_ia_a = (mo_energy[0][viridxa,None] - mo_energy[0][occidxa]).T
        e_ia_b = (mo_energy[1][viridxb,None] - mo_energy[1][occidxb]).T
        e_ia = numpy.hstack((e_ia_a.reshape(-1), e_ia_b.reshape(-1)))
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia).ravel()
        ed_ia = e_ia.ravel() * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)

        def vind(zs):
            nz = len(zs)
            if wfnsym is not None and mol.symmetry:
                zs = numpy.copy(zs)
                zs[:,sym_forbid] = 0
            dmov = numpy.empty((2,nz,nao,nao))
            for i in range(nz):
                z = d_ia * zs[i]
                za = z[:nocca*nvira].reshape(nocca,nvira)
                zb = z[nocca*nvira:].reshape(noccb,nvirb)
                dm = reduce(numpy.dot, (orboa, za, orbva.T))
                dmov[0,i] = dm + dm.T
                dm = reduce(numpy.dot, (orbob, zb, orbvb.T))
                dmov[1,i] = dm + dm.T

            v1ao = vresp(dmov)
            v1a = _ao2mo.nr_e2(v1ao[0], mo_coeff[0], (0,nocca,nocca,nmo))
            v1b = _ao2mo.nr_e2(v1ao[1], mo_coeff[1], (0,noccb,noccb,nmo))
            hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
            for i, z in enumerate(zs):
                hx[i] += ed_ia * z
                hx[i] *= d_ia
            return hx

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        cpu0 = (time.clock(), time.time())
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

        vind, hdiag = self.get_vind(self._scf)
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
            if w2[i] < POSTIVE_EIG_THRESHOLD**2:
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
        log.note('Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)


class dRPA(TDDFTNoHybrid):
    def __init__(self, mf):
        if not getattr(mf, 'xc', None):
            raise RuntimeError("direct RPA can only be applied with DFT; for HF+dRPA, use .xc='hf'")
        from pyscf import scf
        mf = scf.addons.convert_to_uhf(mf)
        mf.xc = '0.0*LDA'
        TDDFTNoHybrid.__init__(self, mf)

TDH = dRPA

class dTDA(TDA):
    def __init__(self, mf):
        if not getattr(mf, 'xc', None):
            raise RuntimeError("direct TDA can only be applied with DFT; for HF+dTDA, use .xc='hf'")
        from pyscf import scf
        mf = scf.addons.convert_to_uhf(mf)
        mf.xc = '0.0*LDA'
        TDA.__init__(self, mf)


def tddft(mf):
    '''Driver to create TDDFT or TDDFTNoHybrid object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf)
    else:
        return TDDFTNoHybrid(mf)

from pyscf import dft
dft.uks.UKS.TDA           = dft.uks_symm.UKS.TDA           = lib.class_as_method(TDA)
dft.uks.UKS.TDHF          = dft.uks_symm.UKS.TDHF          = None
#dft.uks.UKS.TDDFT         = dft.uks_symm.UKS.TDDFT         = lib.class_as_method(TDDFT)
dft.uks.UKS.TDDFTNoHybrid = dft.uks_symm.UKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.uks.UKS.TDDFT         = dft.uks_symm.UKS.TDDFT         = tddft
dft.uks.UKS.dTDA          = dft.uks_symm.UKS.dTDA          = lib.class_as_method(dTDA)
dft.uks.UKS.dRPA          = dft.uks_symm.UKS.dRPA          = lib.class_as_method(dRPA)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'lda, vwn_rpa'
    mf.scf()
    td = mf.TDDFTNoHybrid()
    #td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.08754011   9.08754011   9.7422721    9.7422721   12.48375928]

    mf = dft.UKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = mf.TDDFT()
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.09321047   9.09321047   9.82203065   9.82203065  12.29842071]

    mf = dft.UKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = mf.TDA()
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.01393088   9.01393088   9.68872733   9.68872733  12.42444633]

    mol.spin = 2
    mf = dft.UKS(mol)
    mf.xc = 'lda, vwn_rpa'
    mf.scf()
    td = TDDFTNoHybrid(mf)
    #td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  0.0765857    3.16823079  15.20150204  18.40379107  21.11477253]

    mf = dft.UKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  0.05161674   3.57883843  15.0960023   18.33537454  20.76914967]

    mf = dft.UKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  0.16142061   3.22811366  14.98443928  18.29273507  21.18410081]

