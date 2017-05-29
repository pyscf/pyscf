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
from pyscf import lib
from pyscf.dft import numint
from pyscf import dft
from pyscf.tddft import rhf
from pyscf.ao2mo import _ao2mo

#
# Libxc may have bug or numerical instability for high order derivatives.
#
USE_XCFUN = True

# dmvo = (X+Y) in AO representation
def _contract_xc_kernel(mf, dmvo, singlet=True,
                        rho0=None, vxc=None, fxc=None, max_memory=2000):
    mol = mf.mol
    xc_code = mf.xc
    grids = mf.grids

    if USE_XCFUN:
        ni = copy.copy(mf._numint)
        try:
            ni.libxc = dft.xcfun
            xctype = ni._xc_type(xc_code)
        except (ImportError, KeyError, NotImplementedError):
            ni.libxc = dft.libxc
            xctype = ni._xc_type(xc_code)
    else:
        ni = mf._numint
        xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    ndm = len(dmvo)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmvo = numpy.asarray(dmvo)
    dmvo = (dmvo + dmvo.transpose(0,2,1)) * .5
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    v1ao = numpy.zeros((ndm,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'LDA')
                rho *= .5  # alpha density
                fxc0 = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[2]
                u_u, u_d, d_d = fxc0[0].T
            else:
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T
                ip += ngrid
            if singlet:
                frho = u_u + u_d
            else:
                frho = u_u - u_d

            for i, dm in enumerate(dmvo):
                rho1 = ni.eval_rho(mol, ao, dm, mask, xctype)
                aow = numpy.einsum('pi,p->pi', ao, weight*frho*rho1)
                v1ao[i] += numint._dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)
                rho1 = aow = None

        for i in range(ndm):
            v1ao[i] = (v1ao[i] + v1ao[i].T.conj()) * .5

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ngrid = weight.size
            if vxc is None or fxc is None:
                rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'GGA')
                rho *= .5  # alpha density
                vxc0, fxc0 = ni.eval_xc(xc_code, (rho,rho), 1, deriv=2)[1:3]

                vsigma = vxc0[1].T
                u_u, u_d, d_d = fxc0[0].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc0[1].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc0[2].T  # v2sigma2
            else:
                rho = rho0[0][:,ip:ip+ngrid]
                vsigma = vxc[1][ip:ip+ngrid].T
                u_u, u_d, d_d = fxc[0][ip:ip+ngrid].T  # v2rho2
                u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1][ip:ip+ngrid].T  # v2rhosigma
                uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2][ip:ip+ngrid].T  # v2sigma2

            if singlet:
                fgamma = vsigma[0] + vsigma[1] * .5
                frho = u_u + u_d
                fgg = uu_uu + .5*ud_ud + 2*uu_ud + uu_dd
                frhogamma = u_uu + u_dd + u_ud
            else:
                fgamma = vsigma[0] - vsigma[1] * .5
                frho = u_u - u_d
                fgg = uu_uu - uu_dd
                frhogamma = u_uu - u_dd

            for i, dm in enumerate(dmvo):
                # rho1[0 ] = |b><j| z_{bj}
                # rho1[1:] = \nabla(|b><j|) z_{bj}
                rho1 = ni.eval_rho(mol, ao, dm, mask, 'GGA')
                wv = numint._rks_gga_wv(rho, rho1, (None,fgamma),
                                        (frho,frhogamma,fgg), weight)
                aow = numpy.einsum('nip,ni->ip', ao, wv)
                v1ao[i] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

        for i in range(ndm):
            v1ao[i] = (v1ao[i] + v1ao[i].T.conj()) * .5
    else:
        raise NotImplementedError('meta-GGA')

    return v1ao


class TDA(rhf.TDA):
    # z_{ai} = X_{ai}
    def get_vind(self, mf):
        '''Compute Ax'''
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (mf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        mo_occ = [mf.mo_occ*.5, mf.mo_occ*.5]
        rho0, vxc, fxc = \
                mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                           [mo_coeff,mo_coeff], mo_occ, spin=1)
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=(mf.mol.spin>0)+1)

        def vind(zs):
            nz = len(zs)
            dmvo = numpy.empty((nz,nao,nao))
            for i, z in enumerate(zs):
                dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc), orbo.T))

            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.max_memory*.9-mem_now)
            v1ao = _contract_xc_kernel(mf, dmvo, self.singlet, rho0, vxc, fxc,
                                       max_memory=max_memory)
            if abs(hyb) > 1e-10:
                vj, vk = mf.get_jk(mf.mol, dmvo, hermi=0)
                if self.singlet:
                    v1ao += vj * 2 - hyb * vk
                else:
                    v1ao += -hyb * vk
            else:
                if self.singlet:
                    vj = mf.get_j(mf.mol, dmvo, hermi=1)
                    v1ao += vj * 2

            v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
            for i, z in enumerate(zs):
                v1vo[i] += eai * z
            return v1vo.reshape(nz,-1)

        return vind


class TDDFT(rhf.TDHF):
    def get_vind(self, mf):
        '''
        [ A  B][X]
        [-B -A][Y]
        '''
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (mf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        mo_occ = [mf.mo_occ*.5, mf.mo_occ*.5]
        rho0, vxc, fxc = \
                mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                           [mo_coeff,mo_coeff], mo_occ, spin=1)
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        eai = eai.ravel()
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=(mf.mol.spin>0)+1)

        def vind(xys):
            nz = len(xys)
            dms = numpy.empty((nz,nao,nao))
            for i in range(nz):
                x, y = xys[i].reshape(2,nvir,nocc)
                dms[i] = reduce(numpy.dot, (orbv, x, orbo.T))
                dms[i]+= reduce(numpy.dot, (orbo, y.T, orbv.T))

            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.max_memory*.9-mem_now)
            v1ao = _contract_xc_kernel(mf, dms, self.singlet, rho0, vxc, fxc,
                                       max_memory=max_memory)
            if abs(hyb) > 1e-10:
                vj, vk = mf.get_jk(mf.mol, dms, hermi=0)
                if self.singlet:
                    v1ao += vj * 2 - hyb * vk
                else:
                    v1ao -= hyb * vk
            else:
                if self.singlet:
                    vj = mf.get_j(mf.mol, dms, hermi=1)
                    v1ao += vj * 2

            nov = nocc*nvir
            v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nov)
            v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0,nocc,nocc,nmo))
            v1ov = v1ov.reshape(-1,nocc,nvir).transpose(0,2,1).reshape(-1,nov)
            hx = numpy.empty((nz,nov*2))
            for i, z in enumerate(xys):
                x, y = z.reshape(2,-1)
                hx[i,:nov] = v1vo[i] + eai * x  # AX
                hx[i,nov:] =-v1ov[i] - eai * y  #-AY
            return hx

        return vind


class TDDFTNoHybrid(TDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    def get_vind(self, mf):
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        nao, nmo = mo_coeff.shape
        nocc = (mf.mo_occ>0).sum()
        nvir = nmo - nocc
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        mo_occ = [mf.mo_occ*.5, mf.mo_occ*.5]
        rho0, vxc, fxc = \
                mf._numint.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                           [mo_coeff,mo_coeff], mo_occ, spin=1)
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        dai = numpy.sqrt(eai).ravel()
        edai = eai.ravel() * dai

        def vind(zs):
            nz = len(zs)
            dmvo = numpy.empty((nz,nao,nao))
            for i, z in enumerate(zs):
                dm = reduce(numpy.dot, (orbv, (dai*z).reshape(nvir,nocc), orbo.T))
                dmvo[i] = dm + dm.T # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B

            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.max_memory*.9-mem_now)
            v1ao = _contract_xc_kernel(mf, dmvo, self.singlet, rho0, vxc, fxc,
                                       max_memory=max_memory)
            if self.singlet:
                vj = mf.get_j(mf.mol, dmvo, hermi=1)
                v1ao += vj * 2

            v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir*nocc)
            for i, z in enumerate(zs):
                # numpy.sqrt(eai) * (eai*dai*z + v1vo)
                v1vo[i] += edai*z
                v1vo[i] *= dai
            return v1vo.reshape(nz,-1)
        return vind

    def kernel(self, x0=None):
        '''TDDFT diagonalization solver
        '''
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be applied with hybrid functional'
                               % self.__class__)

        self.check_sanity()
        mo_energy = mf.mo_energy
        nocc = (mf.mo_occ>0).sum()
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])

        if x0 is None:
            x0 = self.init_guess(eai, self.nstates)

        precond = self.get_precond(eai.ravel()**2)
        vind = self.get_vind(self._scf)

        w2, x1 = lib.davidson1(vind, x0, precond,
                               tol=self.conv_tol,
                               nroots=self.nstates, lindep=self.lindep,
                               max_space=self.max_space,
                               verbose=self.verbose)[1:]
        self.e = numpy.sqrt(w2)
        eai = numpy.sqrt(eai)
        def norm_xy(w, z):
            zp = eai * z.reshape(eai.shape)
            zm = w/eai * z.reshape(eai.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
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
    #td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.74227238   9.74227238  14.85153818  30.35019348  30.35019348]

    mf = dft.RKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.82204435   9.82204435  15.0410193   30.01373062  30.01373062]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = TDA(mf)
    td.singlet = False
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.0139312    9.0139312   12.42444659]

