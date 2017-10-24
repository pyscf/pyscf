#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic UHF analytical Hessian
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.scf.newton_ah import _gen_uhf_response
from pyscf.grad import rhf as rhf_grad
from pyscf.hessian import rhf as rhf_hess
_get_jk = rhf_hess._get_jk


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

    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, hessobj.chkfile, atmlst, log)
        t1 = log.timer_debug1('making H1', *time0)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    if isinstance(h1ao, str):
        h1ao = lib.chkfile.load(h1ao, 'scf_f1ao')
        h1aoa = h1ao['0']
        h1aob = h1ao['1']
        h1aoa = dict([(int(k), h1aoa[k]) for k in h1aoa])
        h1aob = dict([(int(k), h1aob[k]) for k in h1aob])
    else:
        h1aoa, h1aob = h1ao
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1a = mo1['0']
        mo1b = mo1['1']
        mo1a = dict([(int(k), mo1a[k]) for k in mo1a])
        mo1b = dict([(int(k), mo1b[k]) for k in mo1b])
    else:
        mo1a, mo1b = mo1
    mo_e1a, mo_e1b = mo_e1

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1ooa = numpy.einsum('xpq,pi,qj->xij', s1ao, mocca, mocca)
        s1oob = numpy.einsum('xpq,pi,qj->xij', s1ao, moccb, moccb)

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            dm1a = numpy.einsum('ypi,qi->ypq', mo1a[ja], mocca)
            dm1b = numpy.einsum('ypi,qi->ypq', mo1b[ja], moccb)
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1aoa[ia], dm1a) * 2
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1aob[ia], dm1b) * 2
            dm1a = numpy.einsum('ypi,qi,i->ypq', mo1a[ja], mocca, mo_ea)
            dm1b = numpy.einsum('ypi,qi,i->ypq', mo1b[ja], moccb, mo_eb)
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1a) * 2
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1b) * 2
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ooa, mo_e1a[ja])
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oob, mo_e1b[ja])

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UHF hessian', *time0)
    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (time.clock(), time.time())

    mol = hessobj.mol
    mf = hessobj._scf
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    dm0 = dm0a + dm0b
    # Energy weighted density matrix
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    dme0 = numpy.einsum('pi,qi,i->pq', mocca, mocca, mo_ea)
    dme0+= numpy.einsum('pi,qi,i->pq', moccb, moccb, mo_eb)

    h1aa, h1ab = rhf_hess.get_hcore(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)

    vj1a, vj1b, vk1a, vk1b = \
            _get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                    ['lk->s1ij', dm0a, 'lk->s1ij', dm0b,
                     'jk->s1il', dm0a, 'jk->s1il', dm0b])
    vj1 = vj1a + vj1b
    vhfa_diag = vj1 - vk1a
    vhfb_diag = vj1 - vk1b
    vhfa_diag = vhfa_diag.reshape(3,3,nao,nao)
    vhfb_diag = vhfb_diag.reshape(3,3,nao,nao)
    vj1 = vj1a = vj1b = vk1a = vk1b = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    aoslices = mol.aoslice_by_atom()
    de2 = numpy.zeros((mol.natm,mol.natm,3,3))  # (A,B,dR_A,dR_B)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1a, vj1b, vk1a, vk1b, vk2a, vk2b = \
                _get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                        ['ji->s1kl', dm0a[:,p0:p1], 'ji->s1kl', dm0b[:,p0:p1],
                         'li->s1kj', dm0a[:,p0:p1], 'li->s1kj', dm0b[:,p0:p1],
                         'lj->s1ki', dm0a         , 'lj->s1ki', dm0b         ],
                        shls_slice=shls_slice)
        vj1 = vj1a + vj1b
        vhfa = vj1 * 2 - vk1a
        vhfb = vj1 * 2 - vk1b
        vhfa[:,:,p0:p1] -= vk2a
        vhfb[:,:,p0:p1] -= vk2b
        t1 = log.timer_debug1('contracting int2e_ip1ip2 for atom %d'%ia, *t1)
        vj1a, vj1b, vk1a, vk1b = \
                _get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                        ['lk->s1ij', dm0a         , 'lk->s1ij', dm0b         ,
                         'li->s1kj', dm0a[:,p0:p1], 'li->s1kj', dm0b[:,p0:p1]],
                        shls_slice=shls_slice)
        vj1 = vj1a + vj1b
        vhfa[:,:,p0:p1] += vj1.transpose(0,2,1)
        vhfb[:,:,p0:p1] += vj1.transpose(0,2,1)
        vhfa -= vk1a.transpose(0,2,1)
        vhfb -= vk1b.transpose(0,2,1)
        vj1 = vj1a = vj1b = vk1a = vk1b = vk2a = vk2b = None
        t1 = log.timer_debug1('contracting int2e_ipvip1 for atom %d'%ia, *t1)
        vhfa = vhfa.reshape(3,3,nao,nao)
        vhfb = vhfb.reshape(3,3,nao,nao)

        rinv2aa, rinv2ab = rhf_hess._hess_rinv(mol, ia)
        hcore = rinv2ab + rinv2aa.transpose(0,1,3,2)
        hcore[:,:,p0:p1] += h1ab[:,:,p0:p1]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1ooa = numpy.einsum('xpq,pi,qj->xij', s1ao, mocca, mocca)
        s1oob = numpy.einsum('xpq,pi,qj->xij', s1ao, moccb, moccb)

        de2[i0,i0] += numpy.einsum('xypq,pq->xy', h1aa[:,:,p0:p1], dm0[p0:p1])*2
        de2[i0,i0] -= numpy.einsum('xypq,pq->xy', rinv2aa, dm0)*2
        de2[i0,i0] -= numpy.einsum('xypq,pq->xy', rinv2ab, dm0)*2
        de2[i0,i0] += numpy.einsum('xypq,pq->xy', vhfa_diag[:,:,p0:p1], dm0a[p0:p1])*2
        de2[i0,i0] += numpy.einsum('xypq,pq->xy', vhfb_diag[:,:,p0:p1], dm0b[p0:p1])*2
        de2[i0,i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0 in range(i0, len(atmlst)):
            ja = atmlst[j0]
            q0, q1 = aoslices[ja][2:]
            de2[j0,i0] += numpy.einsum('xypq,pq->xy', rinv2aa[:,:,q0:q1], dm0[q0:q1])*2
            de2[j0,i0] += numpy.einsum('xypq,pq->xy', rinv2ab[:,:,q0:q1], dm0[q0:q1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', hcore[:,:,:,q0:q1], dm0[:,q0:q1])*2
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', vhfa[:,:,q0:q1], dm0a[q0:q1])*2
            de2[i0,j0] += numpy.einsum('xypq,pq->xy', vhfb[:,:,q0:q1], dm0b[q0:q1])*2
            de2[i0,j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UHF partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    time0 = t1 = (time.clock(), time.time())
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    dR_h1_a = rhf_grad.get_hcore(mol)

    aoslices = mol.aoslice_by_atom()
    h1aoa = [None] * mol.natm
    h1aob = [None] * mol.natm
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1 = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1[:,p0:p1] += dR_h1_a[:,p0:p1]

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1a, vj1b, vj2a, vj2b, vk1a, vk1b, vk2a, vk2b = \
                _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                        ['ji->s2kl', -dm0a[:,p0:p1], 'ji->s2kl', -dm0b[:,p0:p1],
                         'lk->s1ij', -dm0a         , 'lk->s1ij', -dm0b         ,
                         'li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                         'jk->s1il', -dm0a         , 'jk->s1il', -dm0b         ],
                        shls_slice=shls_slice)
        vj1 = vj1a + vj1b
        vj2 = vj2a + vj2b
        h1a = h1 + vj1 - vk1a
        h1b = h1 + vj1 - vk1b
        h1a[:,p0:p1] += vj2 - vk2a
        h1b[:,p0:p1] += vj2 - vk2b
        h1a = h1a + h1a.transpose(0,2,1)
        h1b = h1b + h1b.transpose(0,2,1)

        if chkfile is None:
            h1aoa[ia] = h1a
            h1aob[ia] = h1b
        else:
            lib.chkfile.save(chkfile, 'scf_f1ao/0/%d' % ia, h1a)
            lib.chkfile.save(chkfile, 'scf_f1ao/1/%d' % ia, h1b)
    if chkfile is None:
        return (h1aoa,h1aob)
    else:
        return chkfile

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
              fx=None, atmlst=None, max_memory=4000, verbose=None):
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat, mo_coeff, mocc):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    blksize = max(2, int(max_memory*1e6/8 / (nao*(nocca+noccb)*3*6)))
    mo1sa = [None] * mol.natm
    mo1sb = [None] * mol.natm
    e1sa = [None] * mol.natm
    e1sb = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1voa = []
        s1vob = []
        h1voa = []
        h1vob = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1voa.append(_ao2mo(s1ao, mo_coeff[0], mocca))
            s1vob.append(_ao2mo(s1ao, mo_coeff[1], moccb))
            if isinstance(h1ao_or_chkfile, str):
                h1aoa = lib.chkfile.load(h1ao_or_chkfile, 'scf_f1ao/0/%d'%ia)
                h1aob = lib.chkfile.load(h1ao_or_chkfile, 'scf_f1ao/1/%d'%ia)
            else:
                h1aoa = h1ao_or_chkfile[0][ia]
                h1aob = h1ao_or_chkfile[1][ia]
            h1voa.append(_ao2mo(h1aoa, mo_coeff[0], mocca))
            h1vob.append(_ao2mo(h1aob, mo_coeff[1], moccb))

        h1vo = (numpy.vstack(h1voa), numpy.vstack(h1vob))
        s1vo = (numpy.vstack(s1voa), numpy.vstack(s1vob))
        mo1, e1 = ucphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo)
        mo1a = numpy.einsum('pq,xqi->xpi', mo_coeff[0], mo1[0]).reshape(-1,3,nao,nocca)
        mo1b = numpy.einsum('pq,xqi->xpi', mo_coeff[1], mo1[1]).reshape(-1,3,nao,noccb)
        e1a = e1[0].reshape(-1,3,nocca,nocca)
        e1b = e1[1].reshape(-1,3,noccb,noccb)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            if isinstance(h1ao_or_chkfile, str):
                lib.chkfile.save(h1ao_or_chkfile, 'scf_mo1/0/%d'%ia, mo1a[k])
                lib.chkfile.save(h1ao_or_chkfile, 'scf_mo1/1/%d'%ia, mo1b[k])
            else:
                mo1sa[ia] = mo1a[k]
                mo1sb[ia] = mo1b[k]
            e1sa[ia] = e1a[k].reshape(3,nocca,nocca)
            e1sb[ia] = e1b[k].reshape(3,noccb,noccb)
        mo1 = e1 = mo1a = mo1b = e1a = e1b = None

    if isinstance(h1ao_or_chkfile, str):
        return h1ao_or_chkfile, (e1sa,e1sb)
    else:
        return (mo1sa,mo1sb), (e1sa,e1sb)

def gen_vind(mf, mo_coeff, mo_occ):
    nao, nmoa = mo_coeff[0].shape
    nmob = mo_coeff[1].shape[1]
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    vresp = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        nset = len(mo1)
        dm1 = numpy.empty((2,nset,nao,nao))
        for i, x in enumerate(mo1):
            xa = x[:nmoa*nocca].reshape(nmoa,nocca)
            xb = x[nmoa*nocca:].reshape(nmob,noccb)
            dma = reduce(numpy.dot, (mo_coeff[0], xa, mocca.T))
            dmb = reduce(numpy.dot, (mo_coeff[1], xb, moccb.T))
            dm1[0,i] = dma + dma.T
            dm1[1,i] = dmb + dmb.T
        v1 = vresp(dm1)
        v1vo = numpy.empty_like(mo1)
        for i in range(nset):
            v1vo[i,:nmoa*nocca] = reduce(numpy.dot, (mo_coeff[0].T, v1[0,i], mocca)).ravel()
            v1vo[i,nmoa*nocca:] = reduce(numpy.dot, (mo_coeff[1].T, v1[1,i], moccb)).ravel()
        return v1vo
    return fx


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic UHF hessian'''
    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self._scf, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                         fx, atmlst, max_memory, verbose)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.scf import rhf_grad

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = Hessian(mf)
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(lib.finger(e2) - -0.50693144355876429)

    mol.spin = 2
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = Hessian(mf)
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)

    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            mf = scf.UHF(mol).run(conv_tol=1e-14)
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            mf = scf.UHF(mol).run(conv_tol=1e-14)
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_full(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-4))

# \partial^2 E / \partial R \partial R'
    e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
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
