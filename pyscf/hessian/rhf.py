#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic RHF analytical Hessian
'''

import time
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import cphf


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
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    h1aos = hess_mf.make_h1(mo_coeff, mo_occ, hess_mf.chkfile, atmlst, log)
    t1 = log.timer('making H1', *time0)
    mo1s, e1s = hess_mf.solve_mo1(mo_energy, mo_coeff, mo_occ, h1aos,
                                  None, atmlst, max_memory, log)
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
    vj1, vk1 = _vhf.direct_mapdm(int2e_ipip1, 's2kl',
                                 ('lk->s1ij', 'jk->s1il'), dm0, 9,
                                 mol._atm, mol._bas, mol._env)
    vhf1ii = vj1 - vk1*.5
    vj1 = vk1 = None
    t1 = log.timer('contracting int2e_ipip1', *t1)

    offsetdic = mol.offset_nr_by_atom()
    frinv = h5py.File(tmpf.name, 'r')
    rinv2aa = frinv['rinv2aa']
    rinv2ab = frinv['rinv2ab']

    de2 = numpy.zeros((mol.natm,mol.natm,3,3))
    int2e_ip1ip2 = mol._add_suffix('int2e_ip1ip2')
    int2e_ipvip1 = mol._add_suffix('int2e_ipvip1')
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        h_2 = rinv2ab[str(ia)] + rinv2aa[str(ia)].value.transpose(0,2,1)
        h_2[:,p0:p1] += h1ab[:,p0:p1]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vk1, vk2 = _vhf.direct_bindm(int2e_ip1ip2, 's1',
                                          ('ji->s1kl', 'li->s1kj', 'lj->s1ki'),
                                          (dm0[:,p0:p1], dm0[:,p0:p1], dm0), 9,
                                          mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice)
        vhf2 = vj1 * 2 - vk1 * .5
        vhf2[:,:,p0:p1] -= vk2 * .5
        t1 = log.timer('contracting int2e_ip1ip2 for atom %d'%ia, *t1)

        vj1, vk1 = _vhf.direct_bindm(int2e_ipvip1, 's2kl',
                                     ('lk->s1ij', 'li->s1kj'),
                                     (dm0, dm0[:,p0:p1]), 9,
                                     mol._atm, mol._bas, mol._env,
                                     shls_slice=shls_slice)
        vhf2[:,:,p0:p1] += vj1.transpose(0,2,1)
        vhf2 -= vk1.transpose(0,2,1) * .5
        vj1 = vk1 = vk2 = None
        t1 = log.timer('contracting int2e_ipvip1 for atom %d'%ia, *t1)

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
            de += numpy.einsum('xpq,pq->x', vhf2[:,q0:q1], dm0[q0:q1])*2
            de -= numpy.einsum('xpq,pq->x', s1ab[:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            if ia == ja:
                de += numpy.einsum('xpq,pq->x', h1aa[:,p0:p1], dm0[p0:p1])*2
                de -= numpy.einsum('xpq,pq->x', v2aa, dm0)*2
                de -= numpy.einsum('xpq,pq->x', v2ab, dm0)*2
                de += numpy.einsum('xpq,pq->x', vhf1ii[:,p0:p1], dm0[p0:p1])*2
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

    h1a =-(mol.intor('int1e_ipkin', comp=3) +
           mol.intor('int1e_ipnuc', comp=3))

    offsetdic = mol.offset_nr_by_atom()
    h1aos = []
    int2e_ip1 = mol._add_suffix('int2e_ip1')
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1ao[:,p0:p1] += h1a[:,p0:p1]
        h1ao = h1ao + h1ao.transpose(0,2,1)

        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vj2, vk1, vk2 = \
                _vhf.direct_bindm(int2e_ip1, 's2kl',
                                  ('ji->s2kl', 'lk->s1ij', 'li->s1kj', 'jk->s1il'),
                                  (-dm0[:,p0:p1], -dm0, -dm0[:,p0:p1], -dm0),
                                  3, mol._atm, mol._bas, mol._env,
                                  shls_slice=shls_slice)
        for i in range(3):
            lib.hermi_triu(vj1[i], 1)
        vhf = vj1 - vk1*.5
        vhf[:,p0:p1] += vj2 - vk2*.5
        vhf = vhf + vhf.transpose(0,2,1)

        if chkfile is None:
            h1aos.append(h1ao+vhf)
        else:
            key = 'scf_h1ao/%d' % ia
            lib.chkfile.save(chkfile, key, h1ao+vhf)
    if chkfile is None:
        return h1aos
    else:
        return chkfile

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
              fx=None, atmlst=None, max_memory=4000, verbose=None):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, mf.verbose)
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    if fx is None:
        def fx(mo1):
            dm1 = numpy.einsum('xai,pa,qi->xpq', mo1, mo_coeff, mocc*2)
            dm1 = dm1 + dm1.transpose(0,2,1)
            v1 = mf.get_veff(mol, dm1)
            return numpy.einsum('xpq,pa,qi->xai', v1, mo_coeff, mocc)

    offsetdic = mol.offset_nr_by_atom()
    mem_now = lib.current_memory()[0]
    max_memory = max(4000, max_memory*.9-mem_now)
    blksize = max(2, int(max_memory*1e6/8 / (nmo*nocc*3*6)))
    s1a =-mol.intor('int1e_ipovlp', comp=3)
    mo1s = []
    e1s = []
    for ia0, ia1 in prange(0, len(atmlst), blksize):
        s1vo = []
        h1vo = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = offsetdic[ia]
            s1ao = numpy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1vo.append(numpy.einsum('xpq,pi,qj->xij', s1ao, mo_coeff, mocc))
            if isinstance(h1ao_or_chkfile, str):
                key = 'scf_h1ao/%d' % ia
                h1ao = lib.chkfile.load(h1ao_or_chkfile, key)
            else:
                h1ao = h1ao_or_chkfile[i0]
            h1vo.append(numpy.einsum('xpq,pi,qj->xij', h1ao, mo_coeff, mocc))
        h1vo = numpy.vstack(h1vo)
        s1vo = numpy.vstack(s1vo)
        mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo)
        mo1 = numpy.einsum('pq,xqi->xpi', mo_coeff, mo1).reshape(-1,3,nmo,nocc)
        if isinstance(h1ao_or_chkfile, str):
            for k in range(ia1-ia0):
                key = 'scf_mo1/%d' % atmlst[k+ia0]
                lib.chkfile.save(h1ao_or_chkfile, key, mo1[k])
                mo1s.append(key)
        else:
            mo1s.append(mo1)
        e1s.append(e1.reshape(-1,3,nocc,nocc))

    e1s = numpy.vstack(e1s)
    if isinstance(h1ao_or_chkfile, str):
        return h1ao_or_chkfile, e1s
    else:
        return numpy.vstack(mo1s), e1s

def hess_nuc(mol, atmlst=None):
    gs = numpy.zeros((mol.natm,mol.natm,3,3))
    qs = numpy.asarray([mol.atom_charge(i) for i in range(mol.natm)])
    rs = numpy.asarray([mol.atom_coord(i) for i in range(mol.natm)])
    for i in range(mol.natm):
        r12 = rs[i] - rs
        s12 = numpy.sqrt(numpy.einsum('ki,ki->k', r12, r12))
        s12[i] = 1e60
        tmp1 = qs[i] * qs / s12**3
        tmp2 = numpy.einsum('k, ki,kj->kij',-3*qs[i]*qs/s12**5, r12, r12)

        gs[i,i,0,0] = \
        gs[i,i,1,1] = \
        gs[i,i,2,2] = -tmp1.sum()
        gs[i,i] -= numpy.einsum('kij->ij', tmp2)

        gs[i,:,0,0] += tmp1
        gs[i,:,1,1] += tmp1
        gs[i,:,2,2] += tmp1
        gs[i,:] += tmp2

    if atmlst is not None:
        gs = gs[atmlst][:,atmlst]
    return gs


class Hessian(lib.StreamObject):
    '''Non-relativistic restricted Hartree-Fock hessian'''
    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self._scf = scf_method
        self.chkfile = scf_method.chkfile
        self.max_memory = self.mol.max_memory

        self.de = numpy.zeros((0,0,3,3))
        self._keys = set(self.__dict__.keys())

    hess_elec = hess_elec
    make_h1 = make_h1

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self._scf, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                         fx, atmlst, max_memory, verbose)

    def hess_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        return hess_nuc(mol, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (time.clock(), time.time())
        if mo_energy is None: mo_energy = self._scf.mo_energy
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None: mo_occ = self._scf.mo_occ
        if atmlst is None: atmlst = range(self.mol.natm)

        de = self.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        self.de = de = de + self.hess_nuc(self.mol, atmlst=atmlst)
        return self.de


def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.scf import rhf_grad
    from pyscf.hessian import rhf_o0

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
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    h = Hessian(mf)
    e2 = h.kernel().transpose(0,2,1,3).reshape(n3,n3)
    e2ref = rhf_o0.Hessian(mf).kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(numpy.allclose(e2,e2ref))

#    def grad1(coord, ptr, x, inc):
#        coord = coord.copy()
#        mol._env[ptr:ptr+3] = coord + numpy.asarray(x)*inc
#        e1a = scf.RHF(mol).run(conv_tol=1e-14).apply(rhf_grad.Gradients).kernel()
#        mol._env[ptr:ptr+3] = coord - numpy.asarray(x)*inc
#        e1b = scf.RHF(mol).run(conv_tol=1e-14).apply(rhf_grad.Gradients).kernel()
#        mol._env[ptr:ptr+3] = coord
#        return (e1a-e1b)/(2*inc)
#    e2ref = []
#    for ia in range(mol.natm):
#        coord = mol.atom_coord(ia)
#        ptr = mol._atm[ia,gto.PTR_COORD]
#        e2ref.append(grad1(coord, ptr, (1,0,0), .5e-4))
#        e2ref.append(grad1(coord, ptr, (0,1,0), .5e-4))
#        e2ref.append(grad1(coord, ptr, (0,0,1), .5e-4))
#    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
#    numpy.set_printoptions(2,linewidth=100)
#    print(numpy.linalg.norm(e2-e2ref))
#    print(numpy.allclose(e2,e2ref,atol=1e-6))
#    #for i in range(n3):
#    #    print(e2ref[i]-e2[i], abs(e2ref[i]-e2[i]).max())
