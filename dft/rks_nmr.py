#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


import sys
import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import rhf_nmr
import pyscf.dft.vxc
from pyscf.dft import numint


def _get_vxc_giao(ni, mol, grids, x_id, c_id, dm,
                  max_memory=2000, verbose=None):
    natocc, natorb = scipy.linalg.eigh(dm)
    nao = dm.shape[0]

    xctype = numint._xc_type(x_id, c_id)
    ngrids = len(grids.weights)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory/6*1e6/12/nao/BLKSIZE)*BLKSIZE, ngrids)

    vmat = numpy.zeros((3,nao,nao))
    if xctype == 'LDA':
        buf = numpy.empty((4,blksize,nao))
        for ip0, ip1 in numint.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = ni.non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=0, non0tab=non0tab, out=buf)
            rho = ni.eval_rho2(mol, ao, natorb, natocc, non0tab, xctype)
            vxc = ni.eval_xc(x_id, c_id, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao, weight*vrho)
            giao = mol.eval_gto('GTOval_ig_sph', coords, comp=3,
                                non0tab=non0tab, out=buf[1:])
            vmat[0] += numint._dot_ao_ao(mol, aow, giao[0], nao, ip1-ip0, non0tab)
            vmat[1] += numint._dot_ao_ao(mol, aow, giao[1], nao, ip1-ip0, non0tab)
            vmat[2] += numint._dot_ao_ao(mol, aow, giao[2], nao, ip1-ip0, non0tab)
            rho = vxc = vrho = aow = giao = None
    elif xctype == 'GGA':
        buf = numpy.empty((10,blksize,nao))
        XX, XY, XZ = 0, 1, 2
        YX, YY, YZ = 3, 4, 5
        ZX, ZY, ZZ = 6, 7, 8
        for ip0, ip1 in numint.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = ni.non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=1, non0tab=non0tab, out=buf)
            rho = ni.eval_rho2(mol, ao, natorb, natocc, non0tab, xctype)
            vxc = ni.eval_xc(x_id, c_id, rho, 0, deriv=1)[1]
            vrho, vsigma = vxc[:2]
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho
            wv[1:] = rho[1:] * (weight * vsigma * 2)

            aow = numpy.einsum('npi,np->pi', ao[:4], wv)
            giao = mol.eval_gto('GTOval_ig_sph', coords, 3,
                                non0tab=non0tab, out=buf[4:])
            vmat[0] += numint._dot_ao_ao(mol, aow, giao[0], nao, ip1-ip0, non0tab)
            vmat[1] += numint._dot_ao_ao(mol, aow, giao[1], nao, ip1-ip0, non0tab)
            vmat[2] += numint._dot_ao_ao(mol, aow, giao[2], nao, ip1-ip0, non0tab)

            giao = mol.eval_gto('GTOval_ipig_sph', coords, 9,
                                non0tab=non0tab, out=buf[1:])
            aow = numpy.einsum('pi,p->pi', giao[XX], wv[1])
            aow+= numpy.einsum('pi,p->pi', giao[YX], wv[2])
            aow+= numpy.einsum('pi,p->pi', giao[ZX], wv[3])
            vmat[0] += numint._dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0tab)
            aow = numpy.einsum('pi,p->pi', giao[XY], wv[1])
            aow+= numpy.einsum('pi,p->pi', giao[YY], wv[2])
            aow+= numpy.einsum('pi,p->pi', giao[ZY], wv[3])
            vmat[1] += numint._dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0tab)
            aow = numpy.einsum('pi,p->pi', giao[XZ], wv[1])
            aow+= numpy.einsum('pi,p->pi', giao[YZ], wv[2])
            aow+= numpy.einsum('pi,p->pi', giao[ZZ], wv[3])
            vmat[2] += numint._dot_ao_ao(mol, ao[0], aow, nao, ip1-ip0, non0tab)
            rho = vxc = vrho = vsigma = wv = aow = giao = None
    else:
        raise NotImplementedError('meta-GGA')

    return vmat - vmat.transpose(0,2,1)


class NMR(rhf_nmr.NMR):
    def __init__(self, scf_method):
        rhf_nmr.NMR.__init__(self, scf_method)
        if pyscf.dft.vxc.is_hybrid_xc(self._scf.xc) is None:
            self.cphf = False

    def make_h10(self, mol=None, dm0=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if dm0 is None: dm0 = self._scf.make_rdm1()
        if gauge_orig is None: gauge_orig = self.gauge_orig

        if gauge_orig is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.debug('First-order GIAO Fock matrix')

            h1 = .5 * mol.intor('cint1e_giao_irjxp_sph', 3)
            h1 += mol.intor_asymmetric('cint1e_ignuc_sph', 3)
            h1 += mol.intor('cint1e_igkin_sph', 3)

            x_code, c_code = pyscf.dft.vxc.parse_xc_name(self._scf.xc)
            hyb = pyscf.dft.vxc.hybrid_coeff(x_code, spin=(mol.spin>0)+1)

            mem_now = pyscf.lib.current_memory()[0]
            max_memory = max(2000, self._scf.max_memory*.9-mem_now)
            h1 += _get_vxc_giao(self._scf._numint, mol, self._scf.grids,
                                x_code, c_code, dm0, max_memory=max_memory,
                                verbose=self._scf.verbose)

            if abs(hyb) > 1e-10:
                vj, vk = _vhf.direct_mapdm('cint2e_ig1_sph',  # (g i,j|k,l)
                                           'a4ij', ('lk->s1ij', 'jk->s1il'),
                                           dm0, 3, # xyz, 3 components
                                           mol._atm, mol._bas, mol._env)
                vk = vk - vk.transpose(0,2,1)
                h1 += vj - .5 * hyb * vk
            else:
                vj = _vhf.direct_mapdm('cint2e_ig1_sph', 'a4ij', 'lk->s1ij',
                                       dm0, 3, mol._atm, mol._bas, mol._env)
                h1 += vj
        else:
            mol.set_common_origin_(gauge_orig)
            h1 = .5 * mol.intor('cint1e_cg_irxp_sph', 3)
        pyscf.lib.chkfile.dump(self.chkfile, 'nmr/h1', h1)
        return h1

    def _vind(self, mo1):
        mol = self.mol
        hyb = pyscf.dft.vxc.hybrid_coeff(self._scf.xc, spin=(mol.spin>0)+1)

        if abs(hyb) > 1e-10:
            mo_coeff = self._scf.mo_coeff
            mo_occ = self._scf.mo_occ
            dm1 = self.make_rdm1_1(mo1, mo_coeff, mo_occ)
            direct_scf_bak, self._scf.direct_scf = self._scf.direct_scf, False
            vj, vk = self._scf.get_jk(self.mol, dm1, hermi=2)
            v_ao = -.5 * hyb * vk
            self._scf.direct_scf = direct_scf_bak
            return rhf_nmr._mat_ao2mo(v_ao, mo_coeff, mo_occ)
        else:
            nocc = (self._scf.mo_occ>0).sum()
            nmo = self._scf.mo_coeff.shape[1]
            return numpy.zeros((3,nmo,nocc))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['Ne' , (0. , 0. , 0.)], ]
    mol.basis='631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.kernel()
    nmr = NMR(mf)
    msc = nmr.kernel() # _xx,_yy,_zz = 55.131555
    print(msc)

    mol.atom = [
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.  )], ]
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    mf = dft.RKS(mol)
    mf.kernel()
    nmr = NMR(mf)
    msc = nmr.kernel() # _xx,_yy = 368.878091, _zz = 482.413409
    print(msc)

    mol.basis = 'ccpvdz'
    mol.build(0, 0)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()
    nmr = NMR(mf)
    msc = nmr.kernel() # _xx,_yy = 387.081157, _zz = 482.217461
    print(msc)

