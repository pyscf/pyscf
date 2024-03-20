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

import numpy
from pyscf import symm
from pyscf import lib
from pyscf.lib import logger
from pyscf.tdscf import uhf
from pyscf.scf import uhf_symm
from pyscf.data import nist
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__


class TDA_SF(uhf.TDA_SF):
    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

class TDDFT_SF(uhf.TDA_SF):
    pass
    # print('Remember to set collinear_samples in SF-TDDFT, \
    #        the default value is 200.')
    # def nuc_grad_method(self):
    #     from pyscf.grad import tduks
    #     return tduks.Gradients(self)
    
TDUKS_SF = TDDFT_SF

class CasidaTDDFT(TDDFT_SF, TDA_SF):
    '''Solve the Casida TDDFT formula
       [ A  B][X]
       [-B -A][Y]
    '''
    
    init_guess = TDA_SF.init_guess

    def gen_vind(self, mf=None,extype=0,collinear_samples=200):
        # spin flip up: exytpe=0, spin flip down: exytpe=1
        if mf is None:
            mf = self._scf
        wfnsym = self.wfnsym

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert mo_coeff[0].dtype == numpy.double
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
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
            raise NotImplementedError("UKS Spin Flip TDA/ TDDFT haven't taken symmetry\
                                      into account.")
            
        e_ia_b2a = (mo_energy[0][viridxa,None] - mo_energy[1][occidxb]).T
        e_ia_a2b = (mo_energy[1][viridxb,None] - mo_energy[0][occidxa]).T
        
        if self.extype==0:
            hdiag = numpy.hstack((e_ia_b2a.ravel(), -e_ia_a2b.ravel()))
        elif self.extype==1:
            hdiag = numpy.hstack((e_ia_a2b.ravel(), -e_ia_b2a.ravel()))

        vresp = mf.gen_response_sf(mo_coeff, mo_occ, hermi=0,extype=extype,\
                                   collinear_samples=collinear_samples)
            
        def vind(zs):
            nz = len(zs)
            zs = numpy.asarray(zs).reshape(nz,-1)
            if self.extype==0:
                zsb2a = (zs[:,:noccb*nvira]).reshape(nz,noccb,nvira)
                zsa2b = (zs[:,noccb*nvira:]).reshape(nz,nocca,nvirb)
            elif self.extype==1:
                zsb2a = (zs[:,nocca*nvirb:]).reshape(nz,noccb,nvira)
                zsa2b = (zs[:,:nocca*nvirb]).reshape(nz,nocca,nvirb)
                
            dmsb2a = lib.einsum('xov,po,qv->xpq', zsb2a, orbob, orbva.conj())
            dmsa2b = lib.einsum('xov,po,qv->xpq', zsa2b, orboa, orbvb.conj())

            v1aoA_b2a,v1aoA_a2b,v1aoB_b2a,v1aoB_a2b = vresp(numpy.asarray((dmsb2a,dmsa2b)))
            v1A_b2a = lib.einsum('xpq,po,qv->xov', v1aoA_b2a, orbob, orbva.conj())
            v1A_a2b = lib.einsum('xpq,po,qv->xov', v1aoA_a2b, orboa, orbvb.conj())
            v1B_b2a = lib.einsum('xpq,po,qv->xov', v1aoB_b2a, orbob, orbva.conj())
            v1B_a2b = lib.einsum('xpq,po,qv->xov', v1aoB_a2b, orboa, orbvb.conj())

            # add the orbital energy difference in A matrix.
            v1A_b2a += lib.einsum('ov,xov->xov', e_ia_b2a, zsb2a)
            v1A_a2b += lib.einsum('ov,xov->xov', e_ia_a2b, zsa2b)
            
            if self.extype==0:
                v1_top = v1A_b2a + v1B_b2a
                v1_bom = - v1B_a2b - v1A_a2b
            elif self.extype==1:
                v1_top = v1A_a2b + v1B_a2b
                v1_bom =-v1B_b2a - v1A_b2a
                
            hx = numpy.hstack((v1_top.reshape(nz,-1), v1_bom.reshape(nz,-1)))
            return hx
            
        return vind, hdiag

    def kernel(self, x0=None, nstates=None,extype=None,collinear_samples=None):
        '''SF-TDDFT diagonalization solver
        '''
        cpu0 = (logger.process_clock(), logger.perf_counter())
        mf = self._scf

        self.check_sanity()
        self.dump_flags()
        
        if extype is None:
            extype = self.extype
        else:
            self.extype = extype
        
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
            
        if collinear_samples is None:
            collinear_samples = self.collinear_samples
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf,extype=extype,collinear_samples=collinear_samples)
        precond = hdiag

        def pickeig(w, v, nroots, envs):
            # ToDo: as a function serving for davidson2
            idx = numpy.where(w.real>0)[0]
            return w[idx], v[:,idx], idx

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)
        
        # Because the degeneracy has been dealt with by init_guess_sf function.
        nstates_new = x0.shape[0]   
        converged, w, x1 = \
            lib.davidson2(vind, x0, precond,
                          tol=self.conv_tol,
                          nroots=nstates_new,
                          max_cycle=self.max_cycle,
                          max_space=self.max_space)

        mo_occ = self._scf.mo_occ
        occidxa = numpy.where(mo_occ[0]>0)[0]
        occidxb = numpy.where(mo_occ[1]>0)[0]
        viridxa = numpy.where(mo_occ[0]==0)[0]
        viridxb = numpy.where(mo_occ[1]==0)[0]
        nocca = len(occidxa)
        noccb = len(occidxb)
        nvira = len(viridxa)
        nvirb = len(viridxb)
        
        self.e = w[:nstates]
        x1 = x1[:nstates]
        self.converged = converged[:nstates]
        
        if self.extype==0:
            def norm_xy(z):
                x = z[:noccb*nvira].reshape(noccb,nvira)
                y = z[noccb*nvira:].reshape(nocca,nvirb)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                norm = numpy.sqrt(1./norm)
                return x*norm, y*norm
        elif self.extype==1:
            def norm_xy(z):
                x = z[:nocca*nvirb].reshape(nocca,nvirb)
                y = z[nocca*nvirb:].reshape(noccb,nvira)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                norm = numpy.sqrt(1./norm)
                return x*norm, y*norm

        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)

        self._finalize()
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tduks
        return tduks.Gradients(self)

def tddft(mf):
    '''Driver to create TDDFT_SF or CasidaTDDFT_SF object'''
    return CasidaTDDFT(mf)

from pyscf import dft
dft.uks.UKS.TDDFT_SF = lib.class_as_method(TDDFT_SF)
dft.uks.UKS.TDDFT_SF = tddft