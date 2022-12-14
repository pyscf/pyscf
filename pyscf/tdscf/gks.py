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
from pyscf.tdscf import ghf
from pyscf.scf import ghf_symm
from pyscf.scf import _response_functions  # noqa
from pyscf.data import nist
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
    init_guess = TDA.init_guess

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        wfnsym = self.wfnsym
        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert mo_coeff.dtype == numpy.double
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
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
            orbsym = ghf_symm.get_orbsym(mol, mo_coeff)
            orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
            sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

        e_ia = (mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]).T
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia)
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
            if wfnsym is not None and mol.symmetry:
                zs = numpy.copy(zs)
                zs[:,sym_forbid] = 0

            dmov = lib.einsum('xov,qv,po->xpq', zs*d_ia, orbv, orbo)
            # +cc for A+B because K_{ai,jb} in A == K_{ai,bj} in B
            dmov = dmov + dmov.transpose(0,2,1)

            v1ao = vresp(dmov)
            v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo, orbv)

            # numpy.sqrt(e_ia) * (e_ia*d_ia*z + v1ov)
            v1ov += numpy.einsum('xov,ov->xov', zs, ed_ia)
            v1ov *= d_ia
            if wfnsym is not None and mol.symmetry:
                v1ov[:,sym_forbid] = 0
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
            norm = numpy.sqrt(1./norm)
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
        raise NotImplementedError

TDDFTNoHybrid = CasidaTDDFT


def tddft(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if (not mf._numint.libxc.is_hybrid_xc(mf.xc) and
        # Casida formula can be applied for real orbitals only
        mf.mo_coeff.dtype == numpy.double and mf.collinear[0] != 'm'):
        return CasidaTDDFT(mf)
    else:
        return TDDFT(mf)

from pyscf import dft
dft.gks.GKS.TDA           = dft.gks_symm.GKS.TDA           = lib.class_as_method(TDA)
dft.gks.GKS.TDHF          = dft.gks_symm.GKS.TDHF          = None
dft.gks.GKS.TDDFTNoHybrid = dft.gks_symm.GKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.gks.GKS.CasidaTDDFT   = dft.gks_symm.GKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.gks.GKS.TDDFT         = dft.gks_symm.GKS.TDDFT         = tddft
