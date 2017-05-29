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
from pyscf.ao2mo import _ao2mo
from pyscf.pbc.tddft import rhf

# dmvo = (X+Y) in AO representation
def _contract_xc_kernel(mf, dmvo, singlet=True,
                        rho0=None, vxc=None, fxc=None, max_memory=2000):
    cell = mf.cell
    grids = mf.grids
    xc_code = mf.xc
    kpts = mf.kpts
    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    ndm = len(dmvo)
    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]

    dmvo = numpy.asarray(dmvo)
    dmvo = (dmvo + dmvo.transpose(0,1,3,2)) * .5
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    v1ao = [0] * ndm
    if xctype == 'LDA':
        ao_deriv = 0
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if fxc is None:
                rho = ni.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask, 'LDA')
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
                rho1 = ni.eval_rho(cell, ao_k1, dm, mask, xctype)
                wv = weight * frho * rho1
                v1ao[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    elif xctype == 'GGA':
        ao_deriv = 1
        ip = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, None, max_memory):
            ngrid = weight.size
            if vxc is None or fxc is None:
                rho = ni.eval_rho2(cell, ao_k1, mo_coeff, mo_occ, mask, 'GGA')
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
                rho1 = ni.eval_rho(cell, ao_k1, dm, mask, 'GGA')
                wv = numint._rks_gga_wv(rho, rho1, (None,fgamma),
                                        (frho,frhogamma,fgg), weight)
                v1ao[i] += ni._fxc_mat(cell, ao_k1, wv, mask, xctype, ao_loc)

    else:
        raise NotImplementedError('meta-GGA')

    for i in range(ndm):
        v1ao[i] = (v1ao[i] + v1ao[i].swapaxes(-2,-1).conj()) * .5

    return lib.asarray(v1ao)


class TDA(rhf.TDA):
    def get_vind(self, mf):
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff.shape[1:]
        orbo = []
        orbv = []
        for k in range(nkpts):
            nocc = numpy.count_nonzero(mo_occ[k]>0)
            nvir = nmo - nocc
            orbo.append(mo_coeff[k,:,:nocc])
            orbv.append(mo_coeff[k,:,nocc:])
        rho0, vxc, fxc = mf._numint.cache_xc_kernel(mf.cell, mf.grids, mf.xc,
                                                    [mo_coeff, mo_coeff],
                                                    [mo_occ*.5, mo_occ*.5], spin=1)
        eai = rhf._get_eai(mo_energy, mo_occ)
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=1)

        def vind(zs):
            nz = len(zs)
            dm1s = [rhf._split_vo(z, mo_occ) for z in zs]
            dmvo = numpy.empty((nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                dm1 = dm1s[i]
                for k in range(nkpts):
                    dmvo[i,k] = reduce(numpy.dot, (orbv[k], dm1[k], orbo[k].T.conj()))

            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.max_memory*.9-mem_now)
            v1ao = _contract_xc_kernel(mf, dmvo, self.singlet, rho0, vxc, fxc,
                                       max_memory=max_memory)
            if abs(hyb) > 1e-10:
                vj, vk = mf.get_jk(self.cell, dmvo, hermi=0)
                if self.singlet:
                    v1ao += vj * 2 - hyb * vk
                else:
                    v1ao += -hyb * vk
            else:
                if self.singlet:
                    vj = mf.get_j(self.cell, dmvo, hermi=1)
                    v1ao += vj * 2

            v1s = []
            for i in range(nz):
                dm1 = dm1s[i]
                for k in range(nkpts):
                    v1vo = reduce(numpy.dot, (orbv[k].T.conj(), v1ao[i,k], orbo[k]))
                    v1vo += eai[k] * dm1[k]
                    v1s.append(v1vo.ravel())
            return lib.asarray(v1s).reshape(nz,-1)
        return vind


class TDDFT(rhf.TDHF):
    def get_vind(self, mf):
        '''
        [ A   B ][X]
        [-B* -A*][Y]
        '''
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nkpts = len(mo_occ)
        nao, nmo = mo_coeff.shape[1:]
        orbo = []
        orbv = []
        for k in range(nkpts):
            nocc = numpy.count_nonzero(mo_occ[k]>0)
            orbo.append(mo_coeff[k,:,:nocc])
            orbv.append(mo_coeff[k,:,nocc:])
        rho0, vxc, fxc = mf._numint.cache_xc_kernel(mf.cell, mf.grids, mf.xc,
                                                    [mo_coeff, mo_coeff],
                                                    [mo_occ*.5, mo_occ*.5], spin=1)
        eai = rhf._get_eai(mo_energy, mo_occ)
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=(mf.cell.spin>0)+1)

        def vind(xys):
            nz = len(xys)
            nx = xys[0].size // 2
            dmxs = [rhf._split_vo(xy[:nx], mo_occ) for xy in xys]
            dmys = [rhf._split_vo(xy[nx:], mo_occ) for xy in xys]
            dmvo = numpy.empty((nz,nkpts,nao,nao), dtype=numpy.complex128)
            for i in range(nz):
                dmx = dmxs[i]
                dmy = dmys[i]
                for k in range(nkpts):
                    dmvo[i,k] = reduce(numpy.dot, (orbv[k], dmx[k], orbo[k].T.conj()))
                    dmvo[i,k]+= reduce(numpy.dot, (orbo[k], dmy[k].T, orbv[k].T.conj()))

            mem_now = lib.current_memory()[0]
            max_memory = max(2000, self.max_memory*.9-mem_now)
            v1ao = _contract_xc_kernel(mf, dmvo, self.singlet, rho0, vxc, fxc,
                                       max_memory=max_memory)
            if abs(hyb) > 1e-10:
                vj, vk = mf.get_jk(mf.cell, dmvo, hermi=0)
                if self.singlet:
                    v1ao += vj * 2 - hyb * vk
                else:
                    v1ao -= hyb * vk
            else:
                if self.singlet:
                    vj = mf.get_j(mf.cell, dmvo, hermi=1)
                    v1ao += vj * 2

            v1s = []
            for i in range(nz):
                dmx = dmxs[i]
                dmy = dmys[i]
                v1xs = []
                v1ys = []
                for k in range(nkpts):
                    v1x = reduce(numpy.dot, (orbv[k].T.conj(), v1ao[i,k], orbo[k]))
                    v1y = reduce(numpy.dot, (orbo[k].T.conj(), v1ao[i,k], orbv[k])).T
                    v1x+= eai[k] * dmx[k]
                    v1y+= eai[k] * dmy[k]
                    v1xs.append(v1x.ravel())
                    v1ys.append(-v1y.ravel())
                v1s.extend(v1xs)
                v1s.extend(v1ys)
            return lib.asarray(v1s).reshape(nz,-1)
        return vind


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf
    from pyscf.pbc import dft, df
    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = '''
    C  0.          0.          0.        
    C  1.68506879  1.68506879  1.68506879
    '''
    cell.a = '''
    0.          3.37013758  3.37013758
    3.37013758  0.          3.37013758
    3.37013758  3.37013758  0.
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [12]*3
    cell.build()

    mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
    #mf.with_df = df.MDF(cell, cell.make_kpts([2,1,1]))
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-mdf.h5'
    #mf.with_df.build(with_j3c=False)
    mf.xc = 'lda'
    mf.kernel()
#gs=12 -10.3077341607895
#gs=5  -10.3086623157515

    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
#gs=12 [ 6.08108297  6.10231481  6.10231478  6.38355803  6.38355804]
#MDF gs=5 [ 6.07919157  6.10251718  6.10253961  6.37202499  6.37565246]

    td = TDA(mf)
    td.singlet = False
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
#gs=12 [ 4.01539192  5.1750807   5.17508071]
#MDF gs=5 [ 4.01148649  5.18043397  5.18043459]
