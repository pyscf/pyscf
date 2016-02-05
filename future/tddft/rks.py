#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

import time
from functools import reduce
import numpy
import pyscf.lib
from pyscf.lib import logger
from pyscf.dft import numint
import pyscf.dft.vxc
from pyscf.tddft import rhf


# dmai = (X+Y) in AO representation
def _contract_xc_kernel(td, x_id, c_id, dmai, singlet=True, max_memory=2000):
    mf = td._scf
    mol = td.mol
    ni = mf._numint
    grids = mf.grids

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape

    # dmai ~ reduce(numpy.dot, (orbv, Xai, orbo.T))
    dmai = (dmai + dmai.T) * .5 # because K_{ai,bj} == K_{ai,bj}

    xctype = numint._xc_type(x_id, c_id)
    ngrids = len(grids.weights)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)

    v1ao = numpy.zeros((nao,nao))
    if xctype == 'LDA':
        buf = numpy.empty((blksize,nao))
        for ip0, ip1 in numint.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = ni.non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=0, non0tab=non0tab, out=buf)
            rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab, 'LDA')
            rho *= .5  # alpha density
            fxc = ni.eval_xc(x_id, c_id, (rho,rho), 1, deriv=2)[2]
            u_u, u_d, d_d = v2rho2 = fxc[0].T
            if singlet:
                frho = u_u + u_d
            else:
                frho = u_u - u_d
            rho1 = ni.eval_rho(mol, ao, dmai, non0tab, xctype)
            aow = numpy.einsum('pi,p->pi', ao, weight*frho*rho1)
            v1ao += numint._dot_ao_ao(mol, aow, ao, nao, ip1-ip0, non0tab)
            rho1 = aow = None

    elif xctype == 'GGA':
        buf = numpy.empty((4,blksize,nao))
        for ip0, ip1 in numint.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = ni.non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=1, non0tab=non0tab, out=buf)
            rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab, 'GGA')
            rho *= .5  # alpha density
            vxc, fxc = ni.eval_xc(x_id, c_id, (rho,rho), 1, deriv=2)[1:3]

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

            # rho1[0 ] = |b><j| z_{bj}
            # rho1[1:] = \nabla(|b><j|) z_{bj}
            rho1 = ni.eval_rho(mol, ao, dmai, non0tab, 'GGA')
            # sigma1 = \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
            sigma1 = numpy.einsum('xi,xi->i', rho[1:], rho1[1:]) * 2

            wv  = frho * rho1[0]
            wv += frhogamma * sigma1
            wv *= weight
            if c_id == 131 or x_id in (402, 404, 411, 416, 419):
                # second derivative of LYP functional in libxc library diverge
                wv[rho[0] < 4.57e-11] = 0
            aow = numpy.einsum('pi,p->pi', ao[0], wv)
            v1ao += numint._dot_ao_ao(mol, aow, ao[0], nao, ip1-ip0, non0tab)

            for k in range(3):
                wv  = fgg * sigma1 * rho[1+k]
                wv += frhogamma * rho1[0] * rho[1+k]
                wv *= 2 # *2 because \nabla\rho = \nabla(\rho_\alpha+\rho_\beta)
                wv += fgamma * rho1[1+k]
                wv *= weight
                if c_id == 131 or x_id in (402, 404, 411, 416, 419):
                    # second derivative of LYP functional in libxc library diverge
                    wv[rho[0] < 4.57e-11] = 0
                aow = numpy.einsum('ip,i->ip', ao[0], wv)
                #v1ao += numint._dot_ao_ao(mol, aow, ao[1+k], nao, ip1-ip0, non0tab)
                #v1ao += numint._dot_ao_ao(mol, ao[1+k], aow, nao, ip1-ip0, non0tab)
                # v1ao+v1ao.T at the end
                v1ao += 2*numint._dot_ao_ao(mol, aow, ao[1+k], nao, ip1-ip0, non0tab)
            aow = None
        v1ao = (v1ao + v1ao.T) * .5
    else:
        raise NotImplementedError('meta-GGA')

    return v1ao


class TDA(rhf.TDHF):
    # z_{ai} = X_{ai}
    def get_vind(self, z):
        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        z = z.reshape(eai.shape)

        mem_now = pyscf.lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)

        x_code, c_code = pyscf.dft.vxc.parse_xc_name(self._scf.xc)
        hyb = self._scf._numint.hybrid_coeff(x_code, spin=(mol.spin>0)+1)
        dmz = reduce(numpy.dot, (orbv, z, orbo.T))
        v1ao = _contract_xc_kernel(self, x_code, c_code, dmz,
                                   singlet=self.singlet, max_memory=max_memory)

        dm = reduce(numpy.dot, (orbv, z, orbo.T))
        if abs(hyb) > 1e-10:
            vj, vk = self._scf.get_jk(mol, dm, hermi=0)
            v1ao += vj*2 - hyb * vk
        else:
            dm = dm + dm.T  # symmetrized to handle vj*2
            vj = self._scf.get_j(mol, dm, hermi=1)
            v1ao += vj

        v1vo = reduce(numpy.dot, (orbv.T, v1ao, orbo))
        v1vo = eai*z + v1vo
        return v1vo.ravel()


#class TDRKS?
class TDDFT(TDA):
    # z_{ai} = [(A-B)^{-1/2}(X+Y)]_{ai}
    def get_vind(self, z):
        mol = self.mol
        mo_coeff = self._scf.mo_coeff
        mo_energy = self._scf.mo_energy
        nocc = (self._scf.mo_occ>0).sum()
        orbv = mo_coeff[:,nocc:]
        orbo = mo_coeff[:,:nocc]
        eai = pyscf.lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        dai = numpy.sqrt(eai) * z.reshape(eai.shape)

        mem_now = pyscf.lib.current_memory()[0]
        max_memory = max(2000, self.max_memory*.9-mem_now)

        x_code, c_code = pyscf.dft.vxc.parse_xc_name(self._scf.xc)
        hyb = self._scf._numint.hybrid_coeff(x_code, spin=(mol.spin>0)+1)
        dmai = reduce(numpy.dot, (orbv, dai, orbo.T))
        v1ao = _contract_xc_kernel(self, x_code, c_code, dmai,
                                   singlet=self.singlet, max_memory=max_memory)

        dm = reduce(numpy.dot, (orbv, dai, orbo.T))
        if abs(hyb) > 1e-10:
            raise NotImplementedError
            #vj, vk = self._scf.get_jk(mol, (dm, dm.T), hermi=0)
            #v1ao += (vj[0]+vj[1] - .5 * hyb * (vk[0]+vk[1])) * 2
            vj, vk = self._scf.get_jk(mol, (dm, dm.T))
            a, b = vj - .5 * hyb * vk
        else:
            dm = dm + dm.T  # dai*2 because of spin trace
            vj = self._scf.get_j(mol, dm)
            v1ao += vj

        v1ao *= 2  # *2 becauce A+B and K_{ai,jb} in A == K_{ai,bj} in B

        v1vo = reduce(numpy.dot, (orbv.T, v1ao, orbo))
        v1vo = numpy.sqrt(eai) * (eai*dai + v1vo)
        return v1vo.ravel()

    def get_precond(self, eai):
        eai2 = eai.ravel()**2
        def precond(x, e, x0):
            diagd = eai2 - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def kernel(self, x0=None):
        '''TDDFT diagonalization solver
        '''
        w2, x1 = TDA.kernel(self, x0)
        self.e = numpy.sqrt(w2)
        self.x = x1
        self.y = x1
        return self.e, x1


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
    td = TDDFT(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [  9.74237664   9.74237664  14.85155348  30.35031573  30.35031573]

    mf = dft.RKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = TDDFT(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [  9.8221162    9.8221162   15.04101953  30.01385422  30.01385422]

#    mf = dft.RKS(mol)
#    mf.xc = 'b3pw91'
#    mf.scf()
#    td = TDDFT(mf)
#    td.verbose = 5
#    print td.kernel()[0] * 27.2114
## [ 11.4529811   11.4529811   16.57578312  32.20049155  32.20049155]
#
#    mf = dft.RKS(mol)
#    mf.xc = 'b3lyp'
#    mf.scf()
#    td = TDDFT(mf)
#    td.verbose = 5
#    print td.kernel()[0] * 27.2114
## [ 11.04237434  11.04237434  16.26219172  31.9508519   31.9508519 ]


    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.scf()
    td = TDA(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [  9.91680714   9.91680714  15.44064902  30.56325275  30.56325275]

    mf = dft.RKS(mol)
    mf.xc = 'b3pw91'
    mf.scf()
    td = TDA(mf)
    td.verbose = 5
    print td.kernel()[0] * 27.2114
# [ 10.32227199  10.32227199  15.75328791  30.82032753  30.82032753]

