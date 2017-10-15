#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Non-relativistic UKS analytical nuclear gradients'''

import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint, gen_grid


def get_veff(ks_grad, mol=None, dm=None):
    '''Coulomb + XC functional
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad._scf.make_rdm1()
    t0 = (time.clock(), time.time())

    mf = ks_grad._scf
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    hyb = mf._numint.libxc.hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(mf._numint, mol, mf.grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
    else:
        exc, vxc = get_vxc(mf._numint, mol, mf.grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
    nao = vxc.shape[-1]
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vxc += vj[0] + vj[1]
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        vxc += vj[0] + vj[1] - vk * hyb

    return lib.tag_array(vxc, exc1_grid=exc)


def grad_elec(grad_mf, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    mf = grad_mf._scf
    mol = grad_mf.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(grad_mf.stdout, grad_mf.verbose)

    h1 = grad_mf.get_hcore(mol)
    s1 = grad_mf.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (time.clock(), time.time())
    log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    vhf = grad_mf.get_veff(mol, dm0)
    log.timer('gradients of 2e part', *t0)

    f1 = h1 + vhf
    dme0 = grad_mf.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    atom_slices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = atom_slices[ia]
# h1, s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        vrinv = grad_mf._grad_rinv(mol, ia)
        de[k] += numpy.einsum('sxij,sij->x', f1[:,:,p0:p1], dm0[:,p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vrinv, dm0_sf) * 2
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2
        if grad_mf.grid_response:
            de[k] += vhf.exc1_grid[ia]
    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        rhf_grad._write(log, mol, de, atmlst)
        if grad_mf.grid_response:
            log.debug('grids response contributions')
            rhf_grad._write(log, mol, vhf.exc1_grid[atmlst], atmlst)
            log.debug1('sum(de) %s', vhf.exc1_grid.sum(axis=0))
    return de


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,0])
            vmat[0,0] += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            vmat[0,1] += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            vmat[0,2] += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,1])
            vmat[1,0] += numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            vmat[1,1] += numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            vmat[1,2] += numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:4], mask, 'GGA')
            rho_b = make_rho(1, ao[:4], mask, 'GGA')
            vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[1]
            vrho, vsigma = vxc[:2]

            wva = numpy.empty_like(rho_a)
            wva[0]  = weight * vrho[:,0]
            wva[1:] = rho_a[1:] * (weight * vsigma[:,0] * 2)  # sigma_uu
            wva[1:]+= rho_b[1:] * (weight * vsigma[:,1])      # sigma_ud
            wvb = numpy.empty_like(rho_b)
            wvb[0]  = weight * vrho[:,1]
            wvb[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
            wvb[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud

            vmat[0] += rks_grad._gga_grad_sum(mol, ao, wva, mask, shls_slice, ao_loc)
            vmat[1] += rks_grad._gga_grad_sum(mol, ao, wvb, mask, shls_slice, ao_loc)
            rho_a = rho_b = vxc = vrho = vsigma = wva = wvb = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        raise NotImplementedError('meta-GGA')

    exc = numpy.zeros((mol.natm,3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    atom_slices = mol.aoslice_by_atom()

    excsum = 0
    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        vtmp = numpy.empty((3,nao,nao))
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            ngrids = weight.size
            sh0, sh1 = atom_slices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho_a = make_rho(0, ao[0], mask, 'LDA')
            rho_b = make_rho(1, ao[0], mask, 'LDA')
            exc, vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[:2]
            vrho = vxc[0]

            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,0])
            vtmp[0] = numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            vtmp[1] = numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            vtmp[2] = numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a+rho_b, weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho[:,1])
            vtmp[0] = numint._dot_ao_ao(mol, ao[1], aow, mask, shls_slice, ao_loc)
            vtmp[1] = numint._dot_ao_ao(mol, ao[2], aow, mask, shls_slice, ao_loc)
            vtmp[2] = numint._dot_ao_ao(mol, ao[3], aow, mask, shls_slice, ao_loc)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2
            rho = vxc = vrho = aow = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            ngrids = weight.size
            sh0, sh1 = atom_slices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)
            rho_a = make_rho(0, ao[:4], mask, 'GGA')
            rho_b = make_rho(1, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc(xc_code, (rho_a,rho_b), 1, relativity, 1, verbose)[:2]
            vrho, vsigma = vxc[:2]

            wva = numpy.empty_like(rho_a)
            wva[0]  = weight * vrho[:,0]
            wva[1:] = rho_a[1:] * (weight * vsigma[:,0] * 2)  # sigma_uu
            wva[1:]+= rho_b[1:] * (weight * vsigma[:,1])      # sigma_ud
            wvb = numpy.empty_like(rho_b)
            wvb[0]  = weight * vrho[:,1]
            wvb[1:] = rho_b[1:] * (weight * vsigma[:,2] * 2)  # sigma_dd
            wvb[1:]+= rho_a[1:] * (weight * vsigma[:,1])      # sigma_ud

            vtmp = rks_grad._gga_grad_sum(mol, ao, wva, mask, shls_slice, ao_loc)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = rks_grad._gga_grad_sum(mol, ao, wvb, mask, shls_slice, ao_loc)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2
            rho_a = rho_b = vxc = vrho = vsigma = wva = wvb = None

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')
    else:
        raise NotImplementedError('meta-GGA')

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


class Gradients(uhf_grad.Gradients):
    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.grid_response = False
        self._keys = self._keys.union(['grid_response', 'grids'])

    def dump_flags(self):
        uhf_grad.Gradients.dump_flags(self)
        logger.info('grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff
    grad_elec = grad_elec

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. ,  0.757 , 0.587)] ]
    mol.basis = '631g'
    mol.charge = 1
    mol.spin = 1
    mol.build()
    mf = dft.UKS(mol)
    mf.conv_tol = 1e-12
    #mf.grids.atom_grid = (20,86)
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -4.20040265e-16  -6.59462771e-16   2.10150467e-02]
# [  1.42178271e-16   2.81979579e-02  -1.05137653e-02]
# [  6.34069238e-17  -2.81979579e-02  -1.05137653e-02]]
    g.grid_response = True
    print(g.kernel())

    mf.xc = 'b88,p86'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -8.20194970e-16  -2.04319288e-15   2.44405835e-02]
# [  4.36709255e-18   2.73690416e-02  -1.22232039e-02]
# [  3.44483899e-17  -2.73690416e-02  -1.22232039e-02]]
    g.grid_response = True
    print(g.kernel())

    mf.xc = 'b3lypg'
    e0 = mf.scf()
    g = Gradients(mf)
    print(g.kernel())
#[[ -3.59411142e-16  -2.68753987e-16   1.21557501e-02]
# [  4.04977877e-17   2.11112794e-02  -6.08181640e-03]
# [  1.52600378e-16  -2.11112794e-02  -6.08181640e-03]]


    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.   )], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.charge = -1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
# sum over z direction non-zero, due to meshgrid response
#[[ 0  0  -2.68934738e-03]
# [ 0  0   2.69333577e-03]]
    mf = dft.UKS(mol)
    mf.grids.prune = None
    mf.grids.level = 6
    mf.conv_tol = 1e-14
    mf.kernel()
    print(Gradients(mf).kernel())
#[[ 0  0  -2.68931547e-03]
# [ 0  0   2.68911282e-03]]

