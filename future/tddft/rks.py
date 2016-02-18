#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

import time
import copy
from functools import reduce
import numpy
import pyscf.lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf import dft
from pyscf.tddft import rhf
from pyscf.ao2mo import _ao2mo


# dmvo = (X+Y) in AO representation
def _contract_xc_kernel(td, xc_code, dmvo, singlet=True, max_memory=2000):
    mf = td._scf
    mol = td.mol
    grids = mf.grids

    ni = copy.copy(mf._numint)
    try:
        ni.libxc = dft.xcfun
        xctype = ni._xc_type(xc_code)
    except ImportError, KeyError:
        ni.libxc = dft.libxc
        xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    ndm = len(dmvo)

    dmvo = numpy.asarray(dmvo)
    dmvo = (dmvo + dmvo.transpose(0,2,1)) * .5
    v1ao = numpy.zeros((ndm,nao,nao))
    if xctype == 'LDA':
        if singlet:
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, ni.non0tab):
                rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'LDA')
                rho *= .5  # alpha density
                fxc = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[2]
                u_u, u_d, d_d = v2rho2 = fxc[0].T
                frho = u_u + u_d
                for i, dm in enumerate(dmvo):
                    rho1 = ni.eval_rho(mol, ao, dm, mask, xctype)
                    aow = numpy.einsum('pi,p->pi', ao, weight*frho*rho1)
                    v1ao[i] += numint._dot_ao_ao(mol, aow, ao, nao, weight.size, mask)
                    rho1 = aow = None

            for i in range(ndm):
                v1ao[i] = (v1ao[i] + v1ao[i].T) * .5
        else:
            # because frho = u_u - u_d = 0
            pass

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory, ni.non0tab):
            rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'GGA')
            rho *= .5  # alpha density
            vxc, fxc = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[1:3]

            vsigma = vxc[1].T
            u_u, u_d, d_d = fxc[0].T  # v2rho2
            u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T  # v2rhosigma
            uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T  # v2sigma2
            if singlet:
                fgamma = 2*vsigma[0] + vsigma[1]
                frho = u_u + u_d
                fgg = uu_uu + .5*ud_ud + 2*uu_ud + uu_dd
                frhogamma = u_uu + u_dd + u_ud
            else:
                fgamma = 2*vsigma[0] - vsigma[1]
                frho = u_u - u_d
                fgg = uu_uu - uu_dd
                frhogamma = u_uu - u_dd

            for i, dm in enumerate(dmvo):
                # rho1[0 ] = |b><j| z_{bj}
                # rho1[1:] = \nabla(|b><j|) z_{bj}
                rho1 = ni.eval_rho(mol, ao, dm, mask, 'GGA')
                # sigma1 = \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
                # *2 for alpha + beta
                sigma1 = numpy.einsum('xi,xi->i', rho[1:], rho1[1:]) * 2

                wv  = frho * rho1[0]
                wv += frhogamma * sigma1
                wv *= weight
                aow = numpy.einsum('pi,p->pi', ao[0], wv)
                v1ao[i] += numint._dot_ao_ao(mol, aow, ao[0], nao, weight.size, mask)

                for k in range(3):
                    wv  = fgg * sigma1 * rho[1+k]
                    wv += frhogamma * rho1[0] * rho[1+k]
                    wv *= 2 # *2 because \nabla\rho = \nabla(\rho_\alpha+\rho_\beta)
                    wv += fgamma * rho1[1+k]
                    wv *= weight
                    aow = numpy.einsum('ip,i->ip', ao[0], wv)
                    #v1ao += numint._dot_ao_ao(mol, aow, ao[1+k], nao, weight.size, mask)
                    #v1ao += numint._dot_ao_ao(mol, ao[1+k], aow, nao, weight.size, mask)
                    # v1ao+v1ao.T at the end
                    v1ao[i] += 2*numint._dot_ao_ao(mol, aow, ao[1+k], nao, weight.size, mask)
                aow = None
        for i in range(ndm):
            v1ao[i] = (v1ao[i] + v1ao[i].T) * .5
    else:
        raise NotImplementedError('meta-GGA')

    return v1ao


class TDA(rhf.TDA):
    # z_{ai} = X_{ai}
    def get_vind(self, zs):
        '''Compute Ax'''
        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(zs)
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))

        hyb = self._scf._numint.hybrid_coeff(self._scf.xc, spin=(mol.spin>0)+1)

        mem_now = pyscf.lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        v1ao = _contract_xc_kernel(self, self._scf.xc, dmvo,
                                   singlet=self.singlet, max_memory=max_memory)

        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        if abs(hyb) > 1e-10:
            vj, vk = self._scf.get_jk(mol, dmvo, hermi=0)
            if self.singlet:
                v1ao += vj*2 - hyb * vk
            else:
                v1ao = -hyb * vk
        else:
            if self.singlet:
                dm = dmvo + dmvo.transpose(0,2,1)
                vj = self._scf.get_j(mol, dm, hermi=1)
                v1ao += vj

        v1vo = _ao2mo.nr_e2_(v1ao, mo_coeff, (nocc,nvir,0,nocc)).reshape(-1,nvir*nocc)
        for i, z in enumerate(zs):
            v1vo[i] += eai * z
        return v1vo.reshape(nz,-1)


class TDDFT(rhf.TDHF):
    def get_vind(self, xys):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        nz = len(xys)
        dms = numpy.empty((nz*2,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            dmx = reduce(numpy.dot, (orbv, x, orbo.T))
            dmy = reduce(numpy.dot, (orbv, y, orbo.T))
            dms[i   ] = dmx + dmy.T  # AX + BY
            dms[i+nz] = dms[i].T # = dmy + dmx.T  # AY + BX

        hyb = self._scf._numint.hybrid_coeff(self._scf.xc, spin=(mol.spin>0)+1)

        if abs(hyb) > 1e-10:
            vj, vk = self._scf.get_jk(self.mol, dms, hermi=0)
            if self.singlet:
                veff = vj * 2 - hyb * vk
            else:
                veff = -hyb * vk
        else:
            if self.singlet:
                vj = self._scf.get_j(self.mol, dms, hermi=1)
                veff = vj * 2
            else:
                veff = numpy.zeros((nz*2,nao,nao))

        mem_now = pyscf.lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        v1xc = _contract_xc_kernel(self, self._scf.xc, dms[:nz],
                                   singlet=self.singlet, max_memory=max_memory)
        veff[:nz] += v1xc
        veff[nz:] += v1xc

        veff = _ao2mo.nr_e2_(veff, mo_coeff, (nocc,nvir,0,nocc)).reshape(-1,nvir*nocc)
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        for i, z in enumerate(xys):
            x, y = z.reshape(2,-1)
            veff[i   ] += eai * x  # AX
            veff[i+nz] += eai * y  # AY
        hx = numpy.hstack((veff[:nz], -veff[nz:]))
        return hx.reshape(nz,-1)


class TDDFTNoHybrid(TDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    def get_vind(self, zs):
        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (self._scf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        dai = numpy.sqrt(eai).ravel()

        nz = len(zs)
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            dm = reduce(numpy.dot, (orbv, (dai*z).reshape(nvir,nocc), orbo.T))
            dmvo[i] = dm + dm.T # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B

        if self.singlet:
            vj = self._scf.get_j(mol, dmvo, hermi=1)
            v1ao = vj * 2
        else:
            v1ao = 0

        mem_now = pyscf.lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)
        v1xc = _contract_xc_kernel(self, self._scf.xc, dmvo,
                                   singlet=self.singlet, max_memory=max_memory)
        v1ao += v1xc

        v1vo = _ao2mo.nr_e2_(v1ao, mo_coeff, (nocc,nvir,0,nocc)).reshape(-1,nvir*nocc)
        edai = eai.ravel() * dai
        for i, z in enumerate(zs):
            # numpy.sqrt(eai) * (eai*dai*z + v1vo)
            v1vo[i] += edai*z
            v1vo[i] *= dai
        return v1vo.reshape(nz,-1)

    def kernel(self, x0=None):
        '''TDDFT diagonalization solver
        '''
        if self._scf._numint.libxc.is_hybrid_xc(self._scf.xc):
            raise RuntimeError('%s cannot be applied with hybrid functional'
                               % self.__class__)

        self.check_sanity()

        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel()**2)

        w2, x1 = pyscf.lib.davidson1(self.get_vind, x0, precond,
                                     tol=self.conv_tol,
                                     nroots=self.nstates, lindep=self.lindep,
                                     max_space=self.max_space,
                                     verbose=self.verbose)
        self.e = numpy.sqrt(w2)
        eai = numpy.sqrt(eai)
        def norm_xy(w, z):
            zp = eai * z.reshape(eai.shape)
            zm = w/eai * z.reshape(eai.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = 2*(pyscf.lib.norm(x)**2 - pyscf.lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            return x*norm,y*norm

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
    td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.74237664   9.74237664  14.85155348  30.35031573  30.35031573]

    mf = dft.RKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFTNoHybrid(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.8221162    9.8221162   15.04101953  30.01385422  30.01385422]


    mf = dft.RKS(mol)
    mf.xc = 'lda, vwn_rpa'
    mf.scf()
    td = TDDFT(mf)
    td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.74237664   9.74237664  14.85155348  30.35031573  30.35031573]

    mf = dft.RKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.8221162    9.8221162   15.04101953  30.01385422  30.01385422]

    mf = dft.RKS(mol)
    mf.xc = 'b3pw91'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [ 10.30905882  10.30905882  15.49864105  30.81478949  30.81478949]

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.90274533   9.90274533  15.17706912  30.55785413  30.55785413]


    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.scf()
    td = TDA(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.91680714   9.91680714  15.44064902  30.56325275  30.56325275]

    mf = dft.RKS(mol)
    mf.xc = 'b3pw91'
    mf.scf()
    td = TDA(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [ 10.32227199  10.32227199  15.75328791  30.82032753  30.82032753]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.68883108   9.68883108  15.07124069]

#NOTE b3lyp by libxc is quite different to b3lyp from xcfun
    dft.numint._NumInt.libxc = dft.xcfun
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.88975514   9.88975514  15.16643994  30.55289462  30.55289462]

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.scf()
    td = TDA(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.90383269   9.90383269  15.43035113  30.55830068  30.55830068]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.68883108   9.68883108  15.07124069]
