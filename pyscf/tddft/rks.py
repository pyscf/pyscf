#!/usr/bin/env python
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
from pyscf.dft import numint
from pyscf import dft
from pyscf.tddft import rhf
from pyscf.ao2mo import _ao2mo
from pyscf.scf.newton_ah import _gen_rhf_response


TDA = rhf.TDA

RPA = TDDFT = rhf.TDHF


class TDDFTNoHybrid(TDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    def gen_vind(self, mf):
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
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
            orbsym = hf_symm.get_orbsym(mol, mo_coeff)
            sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym

        eai = mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]
        if wfnsym is not None and mol.symmetry:
            eai[sym_forbid] = 0
        dai = numpy.sqrt(eai).ravel()
        edai = eai.ravel() * dai
        hdiag = eai.ravel() ** 2

        vresp = _gen_rhf_response(mf, singlet=singlet, hermi=1)

        def vind(zs):
            nz = len(zs)
            dmvo = numpy.empty((nz,nao,nao))
            for i, z in enumerate(zs):
                # *2 for double occupancy
                dm = reduce(numpy.dot, (orbv, (dai*z).reshape(nvir,nocc)*2, orbo.T))
                dmvo[i] = dm + dm.T # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            v1ao = vresp(dmvo)
            v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
            for i, z in enumerate(zs):
                # numpy.sqrt(eai) * (eai*dai*z + v1vo)
                v1vo[i] += edai*z
                v1vo[i] *= dai
            return v1vo.reshape(nz,-1)

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

        vind, hdiag = self.gen_vind(self._scf)
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
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        eai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])
        eai = numpy.sqrt(eai)
        def norm_xy(w, z):
            zp = eai * z.reshape(eai.shape)
            zm = w/eai * z.reshape(eai.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            return (x*norm, y*norm)

        self.e = numpy.sqrt(w2)
        self.xy = [norm_xy(self.e[i], z) for i, z in enumerate(x1)]
        return self.e, self.xy


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'lda, vwn_rpa'
    mf.scf()
    td = TDDFTNoHybrid(mf)
    #td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.74227238   9.74227238  14.85153818  30.35019348  30.35019348]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [  9.08754045   9.08754045  12.48375957  29.66870808  29.66870808]

    mf = dft.RKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.82204435   9.82204435  15.0410193   30.01373062  30.01373062]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [  9.09322358   9.09322358  12.29843139  29.26731075  29.26731075]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
    print(td.kernel()[0] * 27.2114)
# [  9.68872769   9.68872769  15.07122478]
    td.singlet = False
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.0139312    9.0139312   12.42444659]


