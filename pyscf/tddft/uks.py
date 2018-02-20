#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf import dft
from pyscf.tddft import uhf
from pyscf.scf import uhf_symm
from pyscf.ao2mo import _ao2mo


TDA = uhf.TDA

RPA = TDDFT = uhf.TDHF


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
            orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
            sym_forbida = (orbsyma[viridxa].reshape(-1,1) ^ orbsyma[occidxa]) != wfnsym
            sym_forbidb = (orbsymb[viridxb].reshape(-1,1) ^ orbsymb[occidxb]) != wfnsym
            sym_forbid = numpy.hstack((sym_forbida.ravel(), sym_forbidb.ravel()))

        e_ai_a = mo_energy[0][viridxa].reshape(-1,1) - mo_energy[0][occidxa]
        e_ai_b = mo_energy[1][viridxb].reshape(-1,1) - mo_energy[1][occidxb]
        e_ai = numpy.hstack((e_ai_a.reshape(-1), e_ai_b.reshape(-1)))
        if wfnsym is not None and mol.symmetry:
            e_ai[sym_forbid] = 0
        dai = numpy.sqrt(e_ai).ravel()
        edai = e_ai.ravel() * dai
        hdiag = e_ai.ravel() ** 2

        if mf.grids.coords is None:
            mf.grids.build()
        ni = mf._numint
        hyb = ni.hybrid_coeff(mf.xc, spin=mol.spin)
        dm0 = None # mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, spin=1)

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.8-mem_now)

        def vind(zs):
            nz = len(zs)
            if wfnsym is not None and mol.symmetry:
                zs = numpy.copy(zs)
                zs[:,sym_forbid] = 0
            dmvo = numpy.empty((2,nz,nao,nao))
            for i in range(nz):
                z = dai * zs[i]
                za = z[:nocca*nvira].reshape(nvira,nocca)
                zb = z[nocca*nvira:].reshape(nvirb,noccb)
                dm = reduce(numpy.dot, (orbva, za, orboa.T))
                dmvo[0,i] = dm + dm.T
                dm = reduce(numpy.dot, (orbvb, zb, orbob.T))
                dmvo[1,i] = dm + dm.T
            v1ao = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dmvo, 0, 1, rho0,
                                 vxc, fxc, max_memory)
            if self.singlet:
                vj = mf.get_j(mf.mol, dmvo, hermi=1)
                v1ao += vj[0] + vj[1]

            v1a = _ao2mo.nr_e2(v1ao[0], mo_coeff[0], (nocca,nmo,0,nocca))
            v1b = _ao2mo.nr_e2(v1ao[1], mo_coeff[1], (noccb,nmo,0,noccb))
            hx = numpy.hstack((v1a.reshape(nz,-1), v1b.reshape(nz,-1)))
            for i, z in enumerate(zs):
                hx[i] += edai * z
                hx[i] *= dai
            return hx

        return vind, hdiag

    def kernel(self, x0=None):
        '''TDDFT diagonalization solver
        '''
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be applied with hybrid functional'
                               % self.__class__)
        self.check_sanity()
        self.dump_flags()

        vind, hdiag = self.get_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        w2, x1 = lib.davidson1(vind, x0, precond,
                               tol=self.conv_tol,
                               nroots=self.nstates, lindep=self.lindep,
                               max_space=self.max_space,
                               verbose=self.verbose)[1:]

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
        e_ai_a = mo_energy[0][viridxa].reshape(-1,1) - mo_energy[0][occidxa]
        e_ai_b = mo_energy[1][viridxb].reshape(-1,1) - mo_energy[1][occidxb]
        eai = numpy.hstack((e_ai_a.reshape(-1), e_ai_b.reshape(-1)))
        eai = numpy.sqrt(eai)

        e = []
        xy = []
        for i, z in enumerate(x1):
            if w2[i] < 0:
                continue
            w = numpy.sqrt(w2[i])
            zp = eai * z
            zm = w/eai * z
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                e.append(w)
                xy.append(((x[:nocca*nvira].reshape(nvira,nocca) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(nvirb,noccb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nvira,nocca) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(nvirb,noccb) * norm)))# Y_beta
        self.e = numpy.array(e)
        self.xy = xy
        return self.e, self.xy


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
    td = TDDFTNoHybrid(mf)
    #td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.08754011   9.08754011   9.7422721    9.7422721   12.48375928]

    mf = dft.UKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.09321047   9.09321047   9.82203065   9.82203065  12.29842071]

    mf = dft.UKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
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
# [  0.16429701   3.207094    15.26015883  18.41945506  21.11157069]

    mf = dft.UKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  0.03964157   3.57928871  15.09594298  20.76987534  18.33539087]

    mf = dft.UKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  0.15334039   3.22003211  15.02627671  18.33258354  21.17695386]

