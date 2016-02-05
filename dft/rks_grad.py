#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Non-relativistic DFT gradients'''

import time
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import rhf_grad
from pyscf.dft import numint
import pyscf.dft.vxc


def get_veff_(ks_grad, mol, dm):
    '''Coulomb + XC functional
    '''
    t0 = (time.clock(), time.time())

    mf = ks_grad._scf
    if mf.grids.coords is None:
        mf.grids.build_()
    grids = mf.grids
    if mf._numint.non0tab is None:
        mf._numint.non0tab = mf._numint.make_mask(mol, mf.grids.coords)
    x_code, c_code = pyscf.dft.vxc.parse_xc_name(mf.xc)
    hyb = mf._numint.hybrid_coeff(x_code, spin=(mol.spin>0)+1)

    mem_now = pyscf.lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    vxc = _get_vxc(mf._numint, mol, mf.grids, x_code, c_code, dm,
                   max_memory=max_memory, verbose=ks_grad.verbose)
    nao = vxc.shape[-1]
    vxc = vxc.reshape(-1,nao,nao)
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vhf = vj
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        vhf = vj - vk * (hyb * .5)

    return vhf + vxc


def _get_vxc(ni, mol, grids, x_id, c_id, dms, relativity=0, hermi=1,
             max_memory=2000, verbose=None):
    natocc = []
    natorb = []
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        e, c = scipy.linalg.eigh(dms)
        natocc.append(e)
        natorb.append(c)
        nao = dms.shape[0]
    else:
        for dm in dms:
            e, c = scipy.linalg.eigh(dm)
            natocc.append(e)
            natorb.append(c)
        nao = dms[0].shape[0]

    xctype = numint._xc_type(x_id, c_id)
    ngrids = len(grids.weights)
    BLKSIZE = numint.BLKSIZE
    blksize = min(int(max_memory/12*1e6/8/nao/BLKSIZE)*BLKSIZE, ngrids)

    nset = len(natocc)
    vmat = numpy.zeros((nset,3,nao,nao))
    if xctype == 'LDA':
        buf = numpy.empty((4,blksize,nao))
        for ip0, ip1 in numint.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = ni.non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=1, non0tab=non0tab, out=buf)
            for idm in range(nset):
                rho = ni.eval_rho2(mol, ao[0], natorb[idm], natocc[idm], non0tab, xctype)
                vxc = ni.eval_xc(x_id, c_id, rho, 0, relativity, 1, verbose)[1]
                vrho = vxc[0]
                aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
                vmat[idm,0] += numint._dot_ao_ao(mol, ao[1], aow, nao, ip1-ip0, non0tab)
                vmat[idm,1] += numint._dot_ao_ao(mol, ao[2], aow, nao, ip1-ip0, non0tab)
                vmat[idm,2] += numint._dot_ao_ao(mol, ao[3], aow, nao, ip1-ip0, non0tab)
                rho = vxc = vrho = aow = None
    elif xctype == 'GGA':
        buf = numpy.empty((10,blksize,nao))
        XX, XY, XZ = 4, 5, 6
        YX, YY, YZ = 5, 7, 8
        ZX, ZY, ZZ = 6, 8, 9
        for ip0, ip1 in numint.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0tab = ni.non0tab[ip0//BLKSIZE:]
            ao = ni.eval_ao(mol, coords, deriv=2, non0tab=non0tab, out=buf)
            for idm in range(nset):
                rho = ni.eval_rho2(mol, ao, natorb[idm], natocc[idm], non0tab, xctype)
                vxc = ni.eval_xc(x_id, c_id, rho, 0, relativity, 1, verbose)[1]
                vrho, vsigma = vxc[:2]
                wv = numpy.empty_like(rho)
                wv[0]  = weight * vrho
                wv[1:] = rho[1:] * (weight * vsigma * 2)

                aow = numpy.einsum('npi,np->pi', ao[:4], wv)
                vmat[idm,0] += numint._dot_ao_ao(mol, ao[1], aow, nao, ip1-ip0, non0tab)
                vmat[idm,1] += numint._dot_ao_ao(mol, ao[2], aow, nao, ip1-ip0, non0tab)
                vmat[idm,2] += numint._dot_ao_ao(mol, ao[3], aow, nao, ip1-ip0, non0tab)

                aow = numpy.einsum('pi,p->pi', ao[XX], wv[1])
                aow+= numpy.einsum('pi,p->pi', ao[XY], wv[2])
                aow+= numpy.einsum('pi,p->pi', ao[XZ], wv[3])
                vmat[idm,0] += numint._dot_ao_ao(mol, aow, ao[0], nao, ip1-ip0, non0tab)
                aow = numpy.einsum('pi,p->pi', ao[YX], wv[1])
                aow+= numpy.einsum('pi,p->pi', ao[YY], wv[2])
                aow+= numpy.einsum('pi,p->pi', ao[YZ], wv[3])
                vmat[idm,1] += numint._dot_ao_ao(mol, aow, ao[0], nao, ip1-ip0, non0tab)
                aow = numpy.einsum('pi,p->pi', ao[ZX], wv[1])
                aow+= numpy.einsum('pi,p->pi', ao[ZY], wv[2])
                aow+= numpy.einsum('pi,p->pi', ao[ZZ], wv[3])
                vmat[idm,2] += numint._dot_ao_ao(mol, aow, ao[0], nao, ip1-ip0, non0tab)
                rho = vxc = vrho = vsigma = wv = aow = None
    else:
        raise NotImplementedError('meta-GGA')

    if nset == 1:
        vmat = vmat.reshape(3,nao,nao)
    # - sign because nabla_X = -nabla_x
    return -vmat


class Gradients(rhf_grad.Gradients):
    def dump_flags(self):
        rhf_grad.Gradients.dump_flags(self)
        if callable(self._scf.grids.prune_scheme):
            logger.info(self, 'Grid pruning %s may affect DFT gradients accuracy.'
                        'Call mf.grids.run(prune_scheme=False) to mute grid pruning',
                        self._scf.grids.prune_scheme)
        return self
    def get_veff(self, mol=None, dm=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self._scf.make_rdm1()
        return get_veff_(self, mol, dm)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#'out_h2o'
    h2o.atom = [
        ['O' , (0. , 0.     , 0)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. ,  0.757 , 0.587)] ]
    h2o.basis = {'H': '631g',
                 'O': '631g',}
    h2o.build()
    mf = dft.RKS(h2o)
    mf.conv_tol = 1e-15
    print mf.scf()
    g = Gradients(mf)
    print(g.grad())
#[[ -1.20185763e-15   5.85017075e-16   2.10514006e-02]
# [ -1.51765862e-17   2.82055434e-02  -1.05252592e-02]
# [ -1.00778811e-16  -2.82055434e-02  -1.05252592e-02]]

    #mf.grids.level = 6
    mf.xc = 'b88,p86'
    print mf.scf()
    g = Gradients(mf)
    print(g.grad())
#[[ -6.53044528e-16   1.61440998e-15   2.44607362e-02]
# [  2.99909644e-16   2.73756804e-02  -1.22322688e-02]
# [ -2.24487619e-16  -2.73756804e-02  -1.22322688e-02]]

    mf.xc = 'b3lyp'
    print mf.scf()
    g = Gradients(mf)
    print(g.grad())
#[[ -3.44790653e-16  -2.31083509e-15   1.21670343e-02]
# [  7.15579513e-17   2.11176116e-02  -6.08866586e-03]
# [ -6.40735965e-17  -2.11176116e-02  -6.08866586e-03]]

    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.   )], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.conv_tol = 1e-15
    mf.kernel()
    print(Gradients(mf).grad())
# sum over z direction non-zero, due to meshgrid response?
#[[ -3.01281832e-17  -2.88065811e-17  -2.69065384e-03]
# [  3.04095683e-16   5.85874058e-16   2.72243724e-03]]
    mf = dft.RKS(mol)
    mf.grids.prune_scheme = None
    mf.grids.level = 6
    mf.conv_tol = 1e-15
    mf.kernel()
    print(Gradients(mf).grad())
#[[  5.02557583e-18   6.72559117e-17  -2.68931547e-03]
# [  1.14142464e-16  -6.66172332e-16   2.68911282e-03]]

