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
from pyscf.hessian import rhf as rhf_hess
from pyscf.dft import numint
from pyscf import dft


def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (time.clock(), time.time())

    mol = hessobj.mol
    mf = hessobj._scf
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, hessobj.chkfile, atmlst, log)
        t1 = log.timer_debug1('making H1', *time0)
    if mo1 is None or mo_e1 is None:
        fx = rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       fx, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    if isinstance(h1ao, str):
        h1ao = lib.chkfile.load(h1ao, 'scf_f1ao')
        h1ao = dict([(int(k), h1ao[k]) for k in h1ao])
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1 = dict([(int(k), mo1[k]) for k in mo1])

    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    hyb = ni.hybrid_coeff(mf.xc)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    # Energy weighted density matrix
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[:nocc]) * 2

    h1aa, h1ab = rhf_hess.get_hcore(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    if abs(hyb) > 1e-10:
        vj1, vk1 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                                    ['lk->s1ij', dm0,   # vj1
                                     'jk->s1il', dm0])  # vk1
        veff_diag += (vj1 - hyb * .5 * vk1).reshape(3,3,nao,nao)
    else:
        vj1 = rhf_hess._get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                               ['lk->s1ij', dm0])
        veff_diag += vj1.reshape(3,3,nao,nao)
    vj1 = vk1 = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    de2 = numpy.zeros((mol.natm,mol.natm,3,3))  # (A,B,dR_A,dR_B)
    vxc = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if abs(hyb) > 1e-10:
            vj1, vk1, vk2 = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                             ['ji->s1kl', dm0[:,p0:p1],  # vj1
                                              'li->s1kj', dm0[:,p0:p1],  # vk1
                                              'lj->s1ki', dm0         ], # vk2
                                             shls_slice=shls_slice)
            veff = vj1 * 2 - hyb * .5 * vk1
            veff[:,:,p0:p1] -= hyb * .5 * vk2
            t1 = log.timer('contracting int2e_ip1ip2 for atom %d'%ia, *t1)

            vj1, vk1 = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                                        ['lk->s1ij', dm0,           # vj1
                                         'li->s1kj', dm0[:,p0:p1]], # vk1
                                        shls_slice=shls_slice)
            veff[:,:,p0:p1] += vj1.transpose(0,2,1)
            veff -= hyb * .5 * vk1.transpose(0,2,1)
            vj1 = vk1 = vk2 = None
            t1 = log.timer('contracting int2e_ipvip1 for atom %d'%ia, *t1)
        else:
            vj1 = rhf_hess._get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                   ['ji->s1kl', dm0[:,p0:p1]],
                                   shls_slice=shls_slice)
            veff = vj1 * 2
            t1 = log.timer('contracting int2e_ip1ip2 for atom %d'%ia, *t1)

            vj1 = rhf_hess._get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                                   ['lk->s1ij', dm0], shls_slice=shls_slice)
            veff[:,:,p0:p1] += vj1.transpose(0,2,1)
            vj1 = None
            t1 = log.timer('contracting int2e_ipvip1 for atom %d'%ia, *t1)
        veff = veff.reshape(3,3,nao,nao)
        veff += vxc[ia]

        rinv2aa, rinv2ab = rhf_hess._hess_rinv(mol, ia)
        hcore = rinv2ab + rinv2aa.transpose(0,1,3,2)
        hcore[:,:,p0:p1] += h1ab[:,:,p0:p1]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        for j0 in range(ia+1):
            ja = atmlst[j0]
            q0, q1 = aoslices[ja][2:]
# *2 for double occupancy, *2 for +c.c.
            dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)
            de  = numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 4
            dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_energy[:nocc])
            de -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            de -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja]) * 2

            v2aa, v2ab = rhf_hess._hess_rinv(mol, ja)
            de += numpy.einsum('xypq,pq->xy', v2aa[:,:,p0:p1], dm0[p0:p1])*2
            de += numpy.einsum('xypq,pq->xy', v2ab[:,:,p0:p1], dm0[p0:p1])*2
            de += numpy.einsum('xypq,pq->xy', hcore[:,:,:,q0:q1], dm0[:,q0:q1])*2
            de += numpy.einsum('xypq,pq->xy', veff[:,:,q0:q1], dm0[q0:q1])*2
            de -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            if ia == ja:
                de += numpy.einsum('xypq,pq->xy', h1aa[:,:,p0:p1], dm0[p0:p1])*2
                de -= numpy.einsum('xypq,pq->xy', v2aa, dm0)*2
                de -= numpy.einsum('xypq,pq->xy', v2ab, dm0)*2
                de += numpy.einsum('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
                de -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

            de2[i0,j0] = de
            de2[j0,i0] = de.T

    log.timer('RKS hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

    hyb = ni.hybrid_coeff(mf.xc)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)

    h1a =-(mol.intor('int1e_ipkin', comp=3) +
           mol.intor('int1e_ipnuc', comp=3))

    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, 4000)
    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1 = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1[:,p0:p1] += h1a[:,p0:p1]

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if abs(hyb) > 1e-10:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
            h1 += vj1 - hyb*.5*vk1
            h1[:,p0:p1] += vj2 - hyb*.5*vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice)
            h1 += vj1
            h1[:,p0:p1] += vj2

        h1ao[ia] += h1 + h1.transpose(0,2,1)

    if chkfile is None:
        return h1ao
    else:
        for ia in atmlst:
            lib.chkfile.save(chkfile, 'scf_f1ao/%d'%ia, h1ao[ia])
        return chkfile

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj._scf
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc = ni.eval_xc(mf.xc, rho, 0, deriv=1)[1]
            vrho = vxc[0]
            aow = numpy.einsum('pi,p->pi', ao[0], weight*vrho)
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
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
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
            aow = numpy.einsum('npi,np->pi', ao[[XXX,XXY,XXZ]], wv[1:4])
            vmat[0] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
            aow = numpy.einsum('npi,np->pi', ao[[XXY,XYY,XYZ]], wv[1:4])
            vmat[1] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
            aow = numpy.einsum('npi,np->pi', ao[[XXZ,XYZ,XZZ]], wv[1:4])
            vmat[2] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
            aow = numpy.einsum('npi,np->pi', ao[[XYY,YYY,YYZ]], wv[1:4])
            vmat[3] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
            aow = numpy.einsum('npi,np->pi', ao[[XYZ,YYZ,YZZ]], wv[1:4])
            vmat[4] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
            aow = numpy.einsum('npi,np->pi', ao[[XZZ,YZZ,ZZZ]], wv[1:4])
            vmat[5] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
            rho = vxc = vrho = vgamma = wv = aow = None

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[0]
    rho1 = numpy.zeros((3,4,ngrids))
    ao_dm0_0 = ao_dm0[0][:,p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[:,0] = numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0_0)
    rho1[0,1]+= numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_0)
    rho1[0,2]+= numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_0)
    rho1[0,3]+= numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_0)
    rho1[1,1]+= numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_0)
    rho1[1,2]+= numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_0)
    rho1[1,3]+= numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_0)
    rho1[2,1]+= numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_0)
    rho1[2,2]+= numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_0)
    rho1[2,3]+= numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_0)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    rho1[:,1] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[1][:,p0:p1])
    rho1[:,2] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[2][:,p0:p1])
    rho1[:,3] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[3][:,p0:p1])
    return rho1

def _make_dR_dao_w(ao, wv):
    aow = numpy.einsum('npi,p->npi', ao[1:4], wv[0])
    aow[0] += numpy.einsum('pi,p->pi', ao[XX], wv[1])  # dX
    aow[0] += numpy.einsum('pi,p->pi', ao[XY], wv[2])  # dX
    aow[0] += numpy.einsum('pi,p->pi', ao[XZ], wv[3])  # dX
    aow[1] += numpy.einsum('pi,p->pi', ao[YX], wv[1])  # dY
    aow[1] += numpy.einsum('pi,p->pi', ao[YY], wv[2])  # dY
    aow[1] += numpy.einsum('pi,p->pi', ao[YZ], wv[3])  # dY
    aow[2] += numpy.einsum('pi,p->pi', ao[ZX], wv[1])  # dZ
    aow[2] += numpy.einsum('pi,p->pi', ao[ZY], wv[2])  # dZ
    aow[2] += numpy.einsum('pi,p->pi', ao[ZZ], wv[3])  # dZ
    return aow

def _make_dR_dao_w1(ao, wv):
    aow = [0] * 3
    aow[0] += numpy.einsum('pi,p->pi', ao[XX], wv[1])  # dX
    aow[0] += numpy.einsum('pi,p->pi', ao[XY], wv[2])  # dX
    aow[0] += numpy.einsum('pi,p->pi', ao[XZ], wv[3])  # dX
    aow[1] += numpy.einsum('pi,p->pi', ao[YX], wv[1])  # dY
    aow[1] += numpy.einsum('pi,p->pi', ao[YY], wv[2])  # dY
    aow[1] += numpy.einsum('pi,p->pi', ao[YZ], wv[3])  # dY
    aow[2] += numpy.einsum('pi,p->pi', ao[ZX], wv[1])  # dZ
    aow[2] += numpy.einsum('pi,p->pi', ao[ZY], wv[2])  # dZ
    aow[2] += numpy.einsum('pi,p->pi', ao[ZZ], wv[3])  # dZ
    return aow

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d1], ao2[d2], mask,
                                                 shls_slice, ao_loc)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d2], ao2[d1], mask,
                                                 shls_slice, ao_loc)

def _d1_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:
        vmat[0] += numint._dot_ao_ao(mol, ao1[0], ao2, mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, ao1[1], ao2, mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, ao1[2], ao2, mask, shls_slice, ao_loc)
    else:
        vmat[0] += numint._dot_ao_ao(mol, ao1, ao2[0], mask, shls_slice, ao_loc)
        vmat[1] += numint._dot_ao_ao(mol, ao1, ao2[1], mask, shls_slice, ao_loc)
        vmat[2] += numint._dot_ao_ao(mol, ao1, ao2[2], mask, shls_slice, ao_loc)

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj._scf
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((mol.natm,3,3,nao,nao))
    ipip = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho = vxc[0]
            frr = fxc[0]
            aow = numpy.einsum('xpi,p->xpi', ao[1:], weight*vrho)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                # aow ~ rho1 ~ d/dR1
                aow = numpy.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho, vgamma = vxc[:2]

            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho * .5
            wv[1:] = rho[1:] * (weight * vgamma * 2)
            aow = _make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices)
                wv[0] = _get_wv(rho, dR_rho1[0], weight, vxc, fxc)
                wv[1] = _get_wv(rho, dR_rho1[1], weight, vxc, fxc)
                wv[2] = _get_wv(rho, dR_rho1[2], weight, vxc, fxc)
                wv[:] *= 2

                aow = _make_dR_dao_w(ao, wv[0])
                _d1_dot_(vmat[ia,0], mol, aow, ao[0], mask, ao_loc, True)
                aow = _make_dR_dao_w(ao, wv[1])
                _d1_dot_(vmat[ia,1], mol, aow, ao[0], mask, ao_loc, True)
                aow = _make_dR_dao_w(ao, wv[2])
                _d1_dot_(vmat[ia,2], mol, aow, ao[0], mask, ao_loc, True)

                aow = numpy.einsum('npi,Xnp->Xpi', ao[1:4], wv[:,1:4])
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return vmat

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj._scf
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    veff = numpy.zeros((mol.natm,3,nao,nao))
    max_memory = max(2000, max_memory-veff.size*8/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, 'LDA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho = vxc[0]
            frr = fxc[0]
            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            aow1 = numpy.einsum('xpi,p->xpi', ao[1:], weight*vrho)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                aow = numpy.einsum('pi,xp->xpi', ao[0], weight*frr*rho1)
                aow[:,:,p0:p1] += aow1[:,:,p0:p1]
                _d1_dot_(veff[ia], mol, aow, ao[0], mask, ao_loc, True)
            ao_dm0 = aow = aow1 = None

        for ia in range(mol.natm):
            veff[ia] = -veff[ia] - veff[ia].transpose(0,2,1)

    elif xctype == 'GGA':
        ao_deriv = 2
        v_ip = numpy.zeros((3,nao,nao))
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, 'GGA')
            vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
            vrho, vgamma = vxc[:2]
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho
            wv[1:] = rho[1:] * (weight * vgamma * 2)
            aow = numpy.einsum('npi,np->pi', ao[:4], wv)
            _d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)
            aow = _make_dR_dao_w1(ao, wv)
            _d1_dot_(v_ip, mol, aow, ao[0], mask, ao_loc, True)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            for ia in range(mol.natm):
                wv = dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices)
                wv[0] = _get_wv(rho, dR_rho1[0], weight, vxc, fxc)
                wv[1] = _get_wv(rho, dR_rho1[1], weight, vxc, fxc)
                wv[2] = _get_wv(rho, dR_rho1[2], weight, vxc, fxc)
                wv[:,1:] *= 2
                aow = numpy.einsum('npi,Xnp->Xpi', ao[:4], wv)
                _d1_dot_(veff[ia], mol, aow, ao[0], mask, ao_loc, True)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            veff[ia,:,p0:p1] += v_ip[:,p0:p1]
            veff[ia] = -veff[ia] - veff[ia].transpose(0,2,1)

    elif xctype == 'MGGA':
        raise NotImplementedError('meta-GGA')

    return veff

def _get_wv(rho, rho1, weight, vxc, fxc):
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


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''
    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self._keys = self._keys.union(['grids'])

    hess_elec = hess_elec
    make_h1 = make_h1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    from pyscf.dft import rks_grad
    dft.numint._NumInt.libxc = dft.xcfun
    xc_code = 'lda,vwn'
    #xc_code = 'blyp'
    #xc_code = 'b3lyp'

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
    hobj = Hessian(mf)
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)

    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            coord = mol.atom_coord(ia).copy()
            ptr = mol._atm[ia,gto.PTR_COORD]
            mol._env[ptr+i] = coord[i] + inc
            mf = dft.RKS(mol).set(conv_tol=1e-14, xc=xc_code).run()
            e1a = mf.nuc_grad_method().set(grid_response=True).kernel()
            mol._env[ptr+i] = coord[i] - inc
            mf = dft.RKS(mol).set(conv_tol=1e-14, xc=xc_code).run()
            e1b = mf.nuc_grad_method().set(grid_response=True).kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_full(ia, .5e-4) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-4))

# \partial^2 E / \partial R \partial R'
    h1ao = hobj.make_h1(mf.mo_coeff, mf.mo_occ)
    mo1, mo_e1 = hobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao)
    e2 = hobj.hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ,
                        numpy.zeros_like(mo1), numpy.zeros_like(mo_e1),
                        numpy.zeros_like(h1ao))
    e2 += hobj.hess_nuc(mol)
    e2 = e2.transpose(0,2,1,3).reshape(n3,n3)
    def grad_partial_R(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_partial_R(ia, .5e-4) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-8))
