#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic RKS analytical Hessian
'''

import time
import copy
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.hessian import rhf
from pyscf.dft import numint
from pyscf import dft


USE_XCFUN = True
XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def hess_elec(hess_mf, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None, max_memory=4000, verbose=None):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(hess_mf.stdout, hess_mf.verbose)

    time0 = (time.clock(), time.time())

    mf = hess_mf._scf
    mol = hess_mf.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    nocc = int(mo_occ.sum()) // 2
    mocc = mo_coeff[:,:nocc]
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    ni = copy.copy(mf._numint)
    if USE_XCFUN:
        try:
            ni.libxc = dft.xcfun
            xctype = ni._xc_type(mf.xc)
        except (ImportError, KeyError, NotImplementedError):
            ni.libxc = dft.libxc
            xctype = ni._xc_type(mf.xc)
    else:
        xctype = ni._xc_type(mf.xc)

    if mf.grids.coords is None:
        mf.grids.build(with_non0tab=True)
    grids = mf.grids
    hyb = ni.libxc.hybrid_coeff(mf.xc)
    max_memory = 4000

    h1aos = hess_mf.make_h1(mo_coeff, mo_occ, hess_mf.chkfile, atmlst, log)
    t1 = log.timer('making H1', *time0)
    def fx(mo1):
        # *2 for alpha + beta
        dm1 = numpy.einsum('xai,pa,qi->xpq', mo1, mo_coeff, mocc*2)
        dm1 = dm1 + dm1.transpose(0,2,1)
        vindxc = _contract_xc_kernel(mf, mf.xc, dm1, max_memory)
        if abs(hyb) > 1e-10:
            vj, vk = mf.get_jk(mol, dm1)
            veff = vj - hyb * .5 * vk + vindxc
        else:
            vj = mf.get_j(mol, dm1)
            veff = vj + vindxc
        v1 = numpy.einsum('xpq,pa,qi->xai', veff, mo_coeff, mocc)
        return v1.reshape(v1.shape[0],-1)
    mo1s, e1s = hess_mf.solve_mo1(mo_energy, mo_coeff, mo_occ, h1aos,
                                  fx, atmlst, max_memory, log)
    t1 = log.timer('solving MO1', *t1)

    tmpf = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    with h5py.File(tmpf.name, 'w') as f:
        for i0, ia in enumerate(atmlst):
            mol.set_rinv_origin(mol.atom_coord(ia))
            f['rinv2aa/%d'%ia] = (mol.atom_charge(ia) *
                                  mol.intor('int1e_ipiprinv', comp=9))
            f['rinv2ab/%d'%ia] = (mol.atom_charge(ia) *
                                  mol.intor('int1e_iprinvip', comp=9))

    h1aa =(mol.intor('int1e_ipipkin', comp=9) +
           mol.intor('int1e_ipipnuc', comp=9))
    h1ab =(mol.intor('int1e_ipkinip', comp=9) +
           mol.intor('int1e_ipnucip', comp=9))
    s1aa = mol.intor('int1e_ipipovlp', comp=9)
    s1ab = mol.intor('int1e_ipovlpip', comp=9)
    s1a =-mol.intor('int1e_ipovlp', comp=3)

    # Energy weighted density matrix
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[:nocc]) * 2

    int2e_ipip1 = mol._add_suffix('int2e_ipip1')
    if abs(hyb) > 1e-10:
        vj1, vk1 = _vhf.direct_mapdm(int2e_ipip1, 's2kl',
                                     ('lk->s1ij', 'jk->s1il'), dm0, 9,
                                     mol._atm, mol._bas, mol._env)
        veff1ii = vj1 - hyb * .5 * vk1
    else:
        vj1 = _vhf.direct_mapdm(int2e_ipip1, 's2kl', 'lk->s1ij', dm0, 9,
                                mol._atm, mol._bas, mol._env)
        veff1ii = vj1.copy()
    vj1[:] = 0
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
            for i in range(6):
                vj1[i] += lib.dot(ao[i+4].T, aow)
            aow = aow1 = None
    elif xctype == 'GGA':
        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]
            vrho, vgamma = vxc[:2]
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho
            wv[1:] = rho[1:] * (weight * vgamma * 2)
            aow = numpy.einsum('npi,np->pi', ao[:4], wv)
            for i in range(6):
                vj1[i] += lib.dot(ao[i+4].T, aow)
            aow = numpy.einsum('npi,np->pi', ao[[XXX,XXY,XXZ]], wv[1:4])
            vj1[0] += lib.dot(aow.T, ao[0])
            aow = numpy.einsum('npi,np->pi', ao[[XXY,XYY,XYZ]], wv[1:4])
            vj1[1] += lib.dot(aow.T, ao[0])
            aow = numpy.einsum('npi,np->pi', ao[[XXZ,XYZ,XZZ]], wv[1:4])
            vj1[2] += lib.dot(aow.T, ao[0])
            aow = numpy.einsum('npi,np->pi', ao[[XYY,YYY,YYZ]], wv[1:4])
            vj1[3] += lib.dot(aow.T, ao[0])
            aow = numpy.einsum('npi,np->pi', ao[[XYZ,YYZ,YZZ]], wv[1:4])
            vj1[4] += lib.dot(aow.T, ao[0])
            aow = numpy.einsum('npi,np->pi', ao[[XZZ,YZZ,ZZZ]], wv[1:4])
            vj1[5] += lib.dot(aow.T, ao[0])
            rho = vxc = vrho = vgamma = wv = aow = None
    else:
        raise NotImplementedError('meta-GGA')
    veff1ii += vj1[[0,1,2,1,3,4,2,4,5]]
    vj1 = vk1 = None

    t1 = log.timer('contracting int2e_ipip1', *t1)

    offsetdic = mol.offset_nr_by_atom()
    frinv = h5py.File(tmpf.name, 'r')
    rinv2aa = frinv['rinv2aa']
    rinv2ab = frinv['rinv2ab']

    de2 = numpy.zeros((mol.natm,mol.natm,3,3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h_2 = rinv2ab[str(ia)] + rinv2aa[str(ia)].value.transpose(0,2,1)
        h_2[:,p0:p1] += h1ab[:,p0:p1]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        int2e_ip1ip2 = mol._add_suffix('int2e_ip1ip2')
        int2e_ipvip1 = mol._add_suffix('int2e_ipvip1')
        if abs(hyb) > 1e-10:
            vj1, vk1, vk2 = _vhf.direct_bindm(int2e_ip1ip2, 's1',
                                              ('ji->s1kl', 'li->s1kj', 'lj->s1ki'),
                                              (dm0[:,p0:p1], dm0[:,p0:p1], dm0), 9,
                                              mol._atm, mol._bas, mol._env,
                                              shls_slice=shls_slice)
            veff2 = vj1 * 2 - hyb * .5 * vk1
            veff2[:,:,p0:p1] -= hyb * .5 * vk2
            t1 = log.timer('contracting int2e_ip1ip2 for atom %d'%ia, *t1)

            vj1, vk1 = _vhf.direct_bindm(int2e_ipvip1, 's2kl',
                                         ('lk->s1ij', 'li->s1kj'),
                                         (dm0, dm0[:,p0:p1]), 9,
                                         mol._atm, mol._bas, mol._env,
                                         shls_slice=shls_slice)
            veff2[:,:,p0:p1] += vj1.transpose(0,2,1)
            veff2 -= hyb * .5 * vk1.transpose(0,2,1)
            vj1 = vk1 = vk2 = None
            t1 = log.timer('contracting int2e_ipvip1 for atom %d'%ia, *t1)
        else:
            vj1 = _vhf.direct_bindm(int2e_ip1ip2, 's1',
                                    'ji->s1kl', dm0[:,p0:p1], 9,
                                    mol._atm, mol._bas, mol._env,
                                    shls_slice=shls_slice)
            veff2 = vj1 * 2
            t1 = log.timer('contracting int2e_ip1ip2 for atom %d'%ia, *t1)

            vj1 = _vhf.direct_bindm(int2e_ipvip1, 's2kl',
                                    'lk->s1ij', dm0, 9,
                                    mol._atm, mol._bas, mol._env,
                                    shls_slice=shls_slice)
            veff2[:,:,p0:p1] += vj1.transpose(0,2,1)
            t1 = log.timer('contracting int2e_ipvip1 for atom %d'%ia, *t1)

        if xctype == 'LDA':
            ao_deriv = 1
            vj1[:] = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
                vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
                vrho = vxc[0]
                frr = fxc[0]
                half = lib.dot(ao[0], dm0[:,p0:p1].copy())
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], half) * 2
                aow = numpy.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                veff2[0] += lib.dot(ao[1].T, aow[0]) # d/d X_i ~ aow ~ rho1
                veff2[1] += lib.dot(ao[2].T, aow[0])
                veff2[2] += lib.dot(ao[3].T, aow[0])
                veff2[3] += lib.dot(ao[1].T, aow[1])
                veff2[4] += lib.dot(ao[2].T, aow[1])
                veff2[5] += lib.dot(ao[3].T, aow[1])
                veff2[6] += lib.dot(ao[1].T, aow[2])
                veff2[7] += lib.dot(ao[2].T, aow[2])
                veff2[8] += lib.dot(ao[3].T, aow[2])
                aow = numpy.einsum('xpi,p->xpi', ao[1:,:,p0:p1], weight*vrho)
                vj1[0] += lib.dot(aow[0].T, ao[1])
                vj1[1] += lib.dot(aow[0].T, ao[2])
                vj1[2] += lib.dot(aow[0].T, ao[3])
                vj1[3] += lib.dot(aow[1].T, ao[1])
                vj1[4] += lib.dot(aow[1].T, ao[2])
                vj1[5] += lib.dot(aow[1].T, ao[3])
                vj1[6] += lib.dot(aow[2].T, ao[1])
                vj1[7] += lib.dot(aow[2].T, ao[2])
                vj1[8] += lib.dot(aow[2].T, ao[3])
                half = aow = None

            veff2[:,:,p0:p1] += vj1.transpose(0,2,1)

        elif xctype == 'GGA':
            def get_wv(rho, rho1, weight, vxc, fxc):
                vgamma = vxc[1]
                frr, frg, fgg = fxc[:3]
                ngrid = weight.size
                sigma1 = numpy.einsum('xi,xi->i', rho[1:], rho1[1:])
                wv = numpy.empty((4,ngrid))
                wv[0]  = frr * rho1[0]
                wv[0] += frg * sigma1 * 2
                wv[1:]  = (fgg * sigma1 * 4 + frg * rho1[0] * 2) * rho[1:]
                wv[1:] += vgamma * rho1[1:] * 2
                wv *= weight
                return wv
            ao_deriv = 2
            vj1[:] = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
                vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
                vrho, vgamma = vxc[:2]
                # (d_X \nabla_x mu) nu DM_{mu,nu}
                half = lib.dot(ao[0], dm0[:,p0:p1].copy())
                rho1X = numpy.einsum('xpi,pi->xp', ao[[1,XX,XY,XZ],:,p0:p1], half)
                rho1Y = numpy.einsum('xpi,pi->xp', ao[[2,YX,YY,YZ],:,p0:p1], half)
                rho1Z = numpy.einsum('xpi,pi->xp', ao[[3,ZX,ZY,ZZ],:,p0:p1], half)
                # (d_X mu) (\nabla_x nu) DM_{mu,nu}
                half = lib.dot(ao[1], dm0[:,p0:p1].copy())
                rho1X[1] += numpy.einsum('pi,pi->p', ao[1,:,p0:p1], half)
                rho1Y[1] += numpy.einsum('pi,pi->p', ao[2,:,p0:p1], half)
                rho1Z[1] += numpy.einsum('pi,pi->p', ao[3,:,p0:p1], half)
                half = lib.dot(ao[2], dm0[:,p0:p1].copy())
                rho1X[2] += numpy.einsum('pi,pi->p', ao[1,:,p0:p1], half)
                rho1Y[2] += numpy.einsum('pi,pi->p', ao[2,:,p0:p1], half)
                rho1Z[2] += numpy.einsum('pi,pi->p', ao[3,:,p0:p1], half)
                half = lib.dot(ao[3], dm0[:,p0:p1].copy())
                rho1X[3] += numpy.einsum('pi,pi->p', ao[1,:,p0:p1], half)
                rho1Y[3] += numpy.einsum('pi,pi->p', ao[2,:,p0:p1], half)
                rho1Z[3] += numpy.einsum('pi,pi->p', ao[3,:,p0:p1], half)

                wv = get_wv(rho, rho1X, weight, vxc, fxc) * 2  # ~ vj1*2
                aow = numpy.einsum('npi,np->pi', ao[[1,XX,XY,XZ]], wv)  # dX
                veff2[0] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[2,YX,YY,YZ]], wv)  # dY
                veff2[1] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[3,ZX,ZY,ZZ]], wv)  # dZ
                veff2[2] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[1:4], wv[1:4])
                veff2[0] += lib.dot(ao[1].T, aow)
                veff2[1] += lib.dot(ao[2].T, aow)
                veff2[2] += lib.dot(ao[3].T, aow)
                wv = get_wv(rho, rho1Y, weight, vxc, fxc) * 2
                aow = numpy.einsum('npi,np->pi', ao[[1,XX,XY,XZ]], wv)
                veff2[3] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[2,YX,YY,YZ]], wv)
                veff2[4] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[3,ZX,ZY,ZZ]], wv)
                veff2[5] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[1:4], wv[1:4])
                veff2[3] += lib.dot(ao[1].T, aow)
                veff2[4] += lib.dot(ao[2].T, aow)
                veff2[5] += lib.dot(ao[3].T, aow)
                wv = get_wv(rho, rho1Z, weight, vxc, fxc) * 2
                aow = numpy.einsum('npi,np->pi', ao[[1,XX,XY,XZ]], wv)
                veff2[6] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[2,YX,YY,YZ]], wv)
                veff2[7] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[3,ZX,ZY,ZZ]], wv)
                veff2[8] += lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[1:4], wv[1:4])
                veff2[6] += lib.dot(ao[1].T, aow)
                veff2[7] += lib.dot(ao[2].T, aow)
                veff2[8] += lib.dot(ao[3].T, aow)

                wv = numpy.empty_like(rho)
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:] * (weight * vgamma * 2)
                aowx = numpy.einsum('npi,np->pi', ao[[1,XX,XY,XZ]], wv)
                aowy = numpy.einsum('npi,np->pi', ao[[2,YX,YY,YZ]], wv)
                aowz = numpy.einsum('npi,np->pi', ao[[3,ZX,ZY,ZZ]], wv)
                ao1 = aowx[:,p0:p1].copy()
                ao2 = aowy[:,p0:p1].copy()
                ao3 = aowz[:,p0:p1].copy()
                vj1[0] += lib.dot(ao1.T, ao[1])
                vj1[1] += lib.dot(ao1.T, ao[2])
                vj1[2] += lib.dot(ao1.T, ao[3])
                vj1[3] += lib.dot(ao2.T, ao[1])
                vj1[4] += lib.dot(ao2.T, ao[2])
                vj1[5] += lib.dot(ao2.T, ao[3])
                vj1[6] += lib.dot(ao3.T, ao[1])
                vj1[7] += lib.dot(ao3.T, ao[2])
                vj1[8] += lib.dot(ao3.T, ao[3])
                ao1 = ao[1,:,p0:p1].copy()
                ao2 = ao[2,:,p0:p1].copy()
                ao3 = ao[3,:,p0:p1].copy()
                vj1[0] += lib.dot(ao1.T, aowx)
                vj1[1] += lib.dot(ao1.T, aowy)
                vj1[2] += lib.dot(ao1.T, aowz)
                vj1[3] += lib.dot(ao2.T, aowx)
                vj1[4] += lib.dot(ao2.T, aowy)
                vj1[5] += lib.dot(ao2.T, aowz)
                vj1[6] += lib.dot(ao3.T, aowx)
                vj1[7] += lib.dot(ao3.T, aowy)
                vj1[8] += lib.dot(ao3.T, aowz)

            veff2[:,:,p0:p1] += vj1.transpose(0,2,1)

        else:
            raise NotImplementedError('meta-GGA')

        for j0, ja in enumerate(atmlst):
            q0, q1 = offsetdic[ja][2:]
# *2 for double occupancy, *2 for +c.c.
            mo1  = lib.chkfile.load(hess_mf.chkfile, 'scf_mo1/%d'%ja)
            h1ao = lib.chkfile.load(hess_mf.chkfile, 'scf_h1ao/%d'%ia)
            dm1 = numpy.einsum('ypi,qi->ypq', mo1, mocc)
            de  = numpy.einsum('xpq,ypq->xy', h1ao, dm1) * 4
            dm1 = numpy.einsum('ypi,qi,i->ypq', mo1, mocc, mo_energy[:nocc])
            de -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            de -= numpy.einsum('xpq,ypq->xy', s1oo, e1s[j0]) * 2

            de = de.reshape(-1)
            v2aa = rinv2aa[str(ja)].value
            v2ab = rinv2ab[str(ja)].value
            de += numpy.einsum('xpq,pq->x', v2aa[:,p0:p1], dm0[p0:p1])*2
            de += numpy.einsum('xpq,pq->x', v2ab[:,p0:p1], dm0[p0:p1])*2
            de += numpy.einsum('xpq,pq->x', h_2[:,:,q0:q1], dm0[:,q0:q1])*2
            de += numpy.einsum('xpq,pq->x', veff2[:,q0:q1], dm0[q0:q1])*2
            de -= numpy.einsum('xpq,pq->x', s1ab[:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            if ia == ja:
                de += numpy.einsum('xpq,pq->x', h1aa[:,p0:p1], dm0[p0:p1])*2
                de -= numpy.einsum('xpq,pq->x', v2aa, dm0)*2
                de -= numpy.einsum('xpq,pq->x', v2ab, dm0)*2
                de += numpy.einsum('xpq,pq->x', veff1ii[:,p0:p1], dm0[p0:p1])*2
                de -= numpy.einsum('xpq,pq->x', s1aa[:,p0:p1], dme0[p0:p1])*2

            de2[i0,j0] = de.reshape(3,3)

    frinv.close()
    log.timer('RHF hessian', *time0)
    return de2

def make_h1(mf, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, mf.verbose)
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    ni = copy.copy(mf._numint)
    if USE_XCFUN:
        try:
            ni.libxc = dft.xcfun
            xctype = ni._xc_type(mf.xc)
        except (ImportError, KeyError, NotImplementedError):
            ni.libxc = dft.libxc
            xctype = ni._xc_type(mf.xc)
    else:
        xctype = ni._xc_type(mf.xc)
    grids = mf.grids
    hyb = ni.libxc.hybrid_coeff(mf.xc)
    max_memory = 4000

    h1a =-(mol.intor('int1e_ipkin', comp=3) +
           mol.intor('int1e_ipnuc', comp=3))

    offsetdic = mol.offset_nr_by_atom()
    h1aos = []
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1ao[:,p0:p1] += h1a[:,p0:p1]
        h1ao = h1ao + h1ao.transpose(0,2,1)

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        int2e_ip1 = mol._add_suffix('int2e_ip1')
        if abs(hyb) > 1e-10:
            vj1, vj2, vk1, vk2 = \
                    _vhf.direct_bindm(int2e_ip1, 's2kl',
                                      ('ji->s2kl', 'lk->s1ij', 'li->s1kj', 'jk->s1il'),
                                      (-dm0[:,p0:p1], -dm0, -dm0[:,p0:p1], -dm0),
                                      3, mol._atm, mol._bas, mol._env,
                                      shls_slice=shls_slice)
            for i in range(3):
                lib.hermi_triu(vj1[i], 1)
            veff = vj1 - hyb*.5*vk1
            veff[:,p0:p1] += vj2 - hyb*.5*vk2
        else:
            vj1, vj2 = \
                    _vhf.direct_bindm(int2e_ip1, 's2kl',
                                      ('ji->s2kl', 'lk->s1ij'),
                                      (-dm0[:,p0:p1], -dm0),
                                      3, mol._atm, mol._bas, mol._env,
                                      shls_slice=shls_slice)
            for i in range(3):
                lib.hermi_triu(vj1[i], 1)
            veff = vj1
            veff[:,p0:p1] += vj2

        if xctype == 'LDA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
                vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
                vrho = vxc[0]
                frr = fxc[0]
                half = lib.dot(ao[0], dm0[:,p0:p1].copy())
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], half)
                aow = numpy.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                aow1 = numpy.einsum('xpi,p->xpi', ao[1:,:,p0:p1], weight*vrho)
                aow[:,:,p0:p1] += aow1
                veff[0] += lib.dot(-aow[0].T, ao[0])
                veff[1] += lib.dot(-aow[1].T, ao[0])
                veff[2] += lib.dot(-aow[2].T, ao[0])
                half = aow = aow1 = None

        elif xctype == 'GGA':
            def get_wv(rho, rho1, weight, vxc, fxc):
                vgamma = vxc[1]
                frr, frg, fgg = fxc[:3]
                ngrid = weight.size
                sigma1 = numpy.einsum('xi,xi->i', rho[1:], rho1[1:])
                wv = numpy.empty((4,ngrid))
                wv[0]  = frr * rho1[0]
                wv[0] += frg * sigma1 * 2
                wv[1:]  = (fgg * sigma1 * 4 + frg * rho1[0] * 2) * rho[1:]
                wv[1:] += vgamma * rho1[1:] * 2
                wv *= weight
                return wv
            ao_deriv = 2
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
                rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
                vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
                vrho, vgamma = vxc[:2]
                # (d_X \nabla_x mu) nu DM_{mu,nu}
                half = lib.dot(ao[0], dm0[:,p0:p1].copy())
                rho1X = numpy.einsum('xpi,pi->xp', ao[[1,XX,XY,XZ],:,p0:p1], half)
                rho1Y = numpy.einsum('xpi,pi->xp', ao[[2,YX,YY,YZ],:,p0:p1], half)
                rho1Z = numpy.einsum('xpi,pi->xp', ao[[3,ZX,ZY,ZZ],:,p0:p1], half)
                # (d_X mu) (\nabla_x nu) DM_{mu,nu}
                half = lib.dot(ao[1], dm0[:,p0:p1].copy())
                rho1X[1] += numpy.einsum('pi,pi->p', ao[1,:,p0:p1], half)
                rho1Y[1] += numpy.einsum('pi,pi->p', ao[2,:,p0:p1], half)
                rho1Z[1] += numpy.einsum('pi,pi->p', ao[3,:,p0:p1], half)
                half = lib.dot(ao[2], dm0[:,p0:p1].copy())
                rho1X[2] += numpy.einsum('pi,pi->p', ao[1,:,p0:p1], half)
                rho1Y[2] += numpy.einsum('pi,pi->p', ao[2,:,p0:p1], half)
                rho1Z[2] += numpy.einsum('pi,pi->p', ao[3,:,p0:p1], half)
                half = lib.dot(ao[3], dm0[:,p0:p1].copy())
                rho1X[3] += numpy.einsum('pi,pi->p', ao[1,:,p0:p1], half)
                rho1Y[3] += numpy.einsum('pi,pi->p', ao[2,:,p0:p1], half)
                rho1Z[3] += numpy.einsum('pi,pi->p', ao[3,:,p0:p1], half)

                wv = get_wv(rho, rho1X, weight, vxc, fxc)
                wv[0] *= .5
                aow = numpy.einsum('npi,np->pi', ao[:4], wv)
                veff[0] -= lib.transpose_sum(lib.dot(aow.T, ao[0]))
                wv = get_wv(rho, rho1Y, weight, vxc, fxc)
                wv[0] *= .5
                aow = numpy.einsum('npi,np->pi', ao[:4], wv)
                veff[1] -= lib.transpose_sum(lib.dot(aow.T, ao[0]))
                wv = get_wv(rho, rho1Z, weight, vxc, fxc)
                wv[0] *= .5
                aow = numpy.einsum('npi,np->pi', ao[:4], wv)
                veff[2] -= lib.transpose_sum(lib.dot(aow.T, ao[0]))

                wv = numpy.empty_like(rho)
                wv[0]  = weight * vrho
                wv[1:] = rho[1:] * (weight * vgamma * 2)
                aow = numpy.einsum('npi,np->pi', ao[:4], wv)
                veff[0,p0:p1] -= lib.dot(ao[1,:,p0:p1].T.copy(), aow)
                veff[1,p0:p1] -= lib.dot(ao[2,:,p0:p1].T.copy(), aow)
                veff[2,p0:p1] -= lib.dot(ao[3,:,p0:p1].T.copy(), aow)

                aow = numpy.einsum('npi,np->pi', ao[[XX,XY,XZ],:,p0:p1], wv[1:4])
                veff[0,p0:p1] -= lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[YX,YY,YZ],:,p0:p1], wv[1:4])
                veff[1,p0:p1] -= lib.dot(aow.T, ao[0])
                aow = numpy.einsum('npi,np->pi', ao[[ZX,ZY,ZZ],:,p0:p1], wv[1:4])
                veff[2,p0:p1] -= lib.dot(aow.T, ao[0])
        else:
            raise NotImplementedError('meta-GGA')

        veff = veff + veff.transpose(0,2,1)

        if chkfile is None:
            h1aos.append(h1ao+veff)
        else:
            key = 'scf_h1ao/%d' % ia
            lib.chkfile.save(chkfile, key, h1ao+veff)
    if chkfile is None:
        return h1aos
    else:
        return chkfile


def _contract_xc_kernel(mf, xc_code, dms, max_memory=2000):
    mol = mf.mol
    grids = mf.grids

    ni = copy.copy(mf._numint)
    if USE_XCFUN:
        try:
            ni.libxc = dft.xcfun
            xctype = ni._xc_type(xc_code)
        except (ImportError, KeyError, NotImplementedError):
            ni.libxc = dft.libxc
            xctype = ni._xc_type(xc_code)
    else:
        xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    ndm = len(dms)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dms = numpy.asarray(dms)
    dms = (dms + dms.transpose(0,2,1)) * .5
    v1ao = numpy.zeros((ndm,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, 'LDA')
            fxc = ni.eval_xc(xc_code, rho, 0, deriv=2)[2]
            frho = fxc[0]

            for i, dm in enumerate(dms):
                rho1 = ni.eval_rho(mol, ao, dm, mask, xctype)
                aow = numpy.einsum('pi,p->pi', ao, weight*frho*rho1)
                v1ao[i] += numint._dot_ao_ao(mol, aow, ao, mask, shls_slice, ao_loc)
                rho1 = aow = None

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc, fxc = ni.eval_xc(xc_code, rho, 0, deriv=2)[1:3]
            vgamma = vxc[1]
            frr, frg, fgg = fxc[:3]
            for i, dm in enumerate(dms):
                rho1 = ni.eval_rho(mol, ao, dm, mask, 'GGA')
                sigma1 = numpy.einsum('xi,xi->i', rho[1:], rho1[1:])
                ngrid = weight.size
                wv = numpy.empty((4,ngrid))
                wv[0]  = frr * rho1[0]
                wv[0] += frg * sigma1 * 2
                wv[1:]  = (fgg * sigma1 * 4 + frg * rho1[0] * 2) * rho[1:]
                wv[1:] += vgamma * rho1[1:] * 2
                wv[1:] *= 2  # for (\nabla\mu) \nu + \mu (\nabla\nu)
                wv *= weight
                aow = numpy.einsum('npi,np->pi', ao, wv)
                v1ao[i] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
    else:
        raise NotImplementedError('meta-GGA')

    for i in range(ndm):
        v1ao[i] = (v1ao[i] + v1ao[i].T) * .5
    return v1ao


class Hessian(rhf.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''
    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None,
                verbose=None):
        return make_h1(self._scf, mo_coeff, mo_occ, chkfile, atmlst, verbose)

    hess_elec = hess_elec


def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    from pyscf.dft import rks_grad
    dft.numint._NumInt.libxc = dft.xcfun
    #xc_code = 'lda,vwn'
    xc_code = 'blyp'

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)],
        ]
    mol.basis = '631g'
    #mol.unit = 'B'
    mol.build()
    mf = dft.RKS(mol)
    mf.grids.level = 4
    mf.grids.prune = False
    mf.xc = xc_code
    mf.conv_tol = 1e-14
    mf.kernel()
    n3 = mol.natm * 3
    h = Hessian(mf)
    e2 = h.kernel().transpose(0,2,1,3).reshape(n3,n3)

    def grad1(coord, ptr, x, inc):
        coord = coord.copy()
        mol._env[ptr:ptr+3] = coord + numpy.asarray(x)*inc
        mf = dft.RKS(mol).set(conv_tol=1e-14)
        mf.grids.level = 4
        mf.grids.prune = False
        mf.xc = xc_code
        e1a = mf.run().apply(rks_grad.Gradients).kernel()
        mol._env[ptr:ptr+3] = coord - numpy.asarray(x)*inc
        mf = dft.RKS(mol).set(conv_tol=1e-14)
        mf.grids.level = 4
        mf.grids.prune = False
        mf.xc = xc_code
        e1b = mf.run().apply(rks_grad.Gradients).kernel()
        mol._env[ptr:ptr+3] = coord
        return (e1a-e1b)/(2*inc)
    e2ref = []
    for ia in range(mol.natm):
        coord = mol.atom_coord(ia)
        ptr = mol._atm[ia,gto.PTR_COORD]
        e2ref.append(grad1(coord, ptr, (1,0,0), .5e-3))
        e2ref.append(grad1(coord, ptr, (0,1,0), .5e-3))
        e2ref.append(grad1(coord, ptr, (0,0,1), .5e-3))
    e2ref = numpy.asarray(e2ref).reshape(-1,n3)
    print(abs(e2ref).sum())
    numpy.set_printoptions(2,linewidth=100)
    print(numpy.linalg.norm(e2-e2ref))
    for i in range(n3):
        print(e2ref[i]-e2[i], abs(e2ref[i]-e2[i]).max())

## partial derivative for C
#    e2 = h.hess_elec().transpose(0,2,1,3).reshape(n3,n3)
#    g1a = mf.apply(rks_grad.Gradients).grad_elec()
#    ia = 0
#    coord = mol.atom_coord(ia)
#    ptr = mol._atm[ia,gto.PTR_COORD]
#    inc = 1e-4
#    coord = coord.copy()
#    mol._env[ptr:ptr+3] = coord + numpy.asarray((0,1,0))*inc
#    mf = dft.RKS(mol)
#    mf.grids.level = 4
#    mf.grids.prune = False
#    mf.conv_tol = 1e-14
#    mf.xc = xc_code
#    mf.kernel()
#    mol._env[ptr:ptr+3] = coord
#    g1b = mf.apply(rks_grad.Gradients).grad_elec()
#    print((g1b-g1a)/inc)
#    print(e2[1].reshape(-1,3))

## partial derivative for R
#    e2 = h.hess_elec().transpose(0,2,1,3).reshape(n3,n3)
#    g1a = mf.apply(rks_grad.Gradients).grad_elec()
#    ia = 0
#    coord = mol.atom_coord(ia)
#    ptr = mol._atm[ia,gto.PTR_COORD]
#    inc = 1e-4
#    coord = coord.copy()
#    mol._env[ptr:ptr+3] = coord + numpy.asarray((0,1,0))*inc
#    g1b = mf.apply(rks_grad.Gradients).grad_elec()
#    print((g1b-g1a)/inc)
#    print(e2[1].reshape(-1,3))

#    g1a = mf.apply(rks_grad.Gradients).grad_elec()
#    ia = 1
#    coord = mol.atom_coord(ia)
#    ptr = mol._atm[ia,gto.PTR_COORD]
#    inc = 1e-4
#    coord = coord.copy()
#    mol._env[ptr:ptr+3] = coord + numpy.asarray((0,1,0))*inc
#    g1b = mf.apply(rks_grad.Gradients).grad_elec()
#    print((g1b-g1a)/inc)
#    print(e2[4].reshape(-1,3))

# h^1
#    e2 = h.hess_elec()
#    dm0 = mf.make_rdm1()
#    g1a = rks_grad.get_veff(rks_grad.Gradients(mf), mol)
#    #g1a = mf.get_veff(mol, dm0)
#    ia = 0
#    coord = mol.atom_coord(ia)
#    ptr = mol._atm[ia,gto.PTR_COORD]
#    inc = 1e-4
#    coord = coord.copy()
#    mol._env[ptr:ptr+3] = coord + numpy.asarray((0,1,0))*inc
#    mf._eri = None
#    g1b = rks_grad.get_veff(rks_grad.Gradients(mf), mol)
#    #g1b = mf.get_veff(mol, dm0)
#    print((g1b-g1a)/inc)
