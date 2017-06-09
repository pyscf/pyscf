#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# J. Chem. Phys. 117, 7433
#

import time
from functools import reduce
import numpy
import pyscf.lib
from pyscf.lib import logger
from pyscf.scf import rhf_grad
from pyscf.scf import cphf


#
# Given Y = 0, TDHF gradients (XAX+XBY+YBX+YAY)^1 turn to TDA gradients (XAX)^1
#
def kernel(td_grad, x_y, singlet=True, atmlst=None,
           max_memory=2000, verbose=logger.INFO):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(td_grad.stdout, verbose)
    time0 = time.clock(), time.time()

    mol = td_grad.mol
    mf = td_grad._td._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x+y).reshape(nvir,nocc)
    xmy = (x-y).reshape(nvir,nocc)
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmzvop = reduce(numpy.dot, (orbv, xpy, orbo.T))
    dmzvom = reduce(numpy.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(numpy.dot, (orbo, doo, orbo.T))
    dmzoo+= reduce(numpy.dot, (orbv, dvv, orbv.T))

    vj, vk = mf.get_jk(mol, (dmzoo, dmzvop+dmzvop.T, dmzvom-dmzvom.T), hermi=0)
    veff0doo = vj[0] * 2 - vk[0]
    wvo = reduce(numpy.dot, (orbv.T, veff0doo, orbo)) * 2
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
    veff0mop = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mop[nocc:,nocc:], xpy) * 2
    veff = -vk[2]
    veff0mom = reduce(numpy.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy) * 2
    wvo += numpy.einsum('ac,ai->ci', veff0mom[nocc:,nocc:], xmy) * 2
    def fvind(x):  # For singlet, closed shell ground state
        dm = reduce(numpy.dot, (orbv, x.reshape(nvir,nocc), orbo.T))
        vj, vk = mf.get_jk(mol, (dm+dm.T))
        return reduce(numpy.dot, (orbv.T, vj*2-vk, orbo)).ravel()
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.max_cycle_cphf, tol=td_grad.conv_tol)[0]
    z1 = z1.reshape(nvir,nocc)
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao = reduce(numpy.dot, (orbv, z1, orbo.T))
    vj, vk = mf.get_jk(mol, z1ao, hermi=0)
    veff = vj * 2 - vk

    im0 = numpy.zeros((nmo,nmo))
    im0[:nocc,:nocc] = reduce(numpy.dot, (orbo.T, veff0doo+veff, orbo))
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mop[nocc:,:nocc], xpy)
    im0[:nocc,:nocc]+= numpy.einsum('ak,ai->ki', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,nocc:] = numpy.einsum('ci,ai->ac', veff0mop[nocc:,:nocc], xpy)
    im0[nocc:,nocc:]+= numpy.einsum('ci,ai->ac', veff0mom[nocc:,:nocc], xmy)
    im0[nocc:,:nocc] = numpy.einsum('ki,ai->ak', veff0mop[:nocc,:nocc], xpy)*2
    im0[nocc:,:nocc]+= numpy.einsum('ki,ai->ak', veff0mom[:nocc,:nocc], xmy)*2

    zeta = pyscf.lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[:nocc,:nocc] = doo
    dm1[nocc:,nocc:] = dvv
    dm1[nocc:,:nocc] = z1
    dm1[:nocc,:nocc] += numpy.eye(nocc)*2 # for ground state
    im0 = reduce(numpy.dot, (mo_coeff, im0+zeta*dm1, mo_coeff.T))

    h1 = td_grad.get_hcore(mol)
    s1 = td_grad.get_ovlp(mol)

    dmz1doo = z1ao + dmzoo
    oo0 = reduce(numpy.dot, (orbo, orbo.T))
    vj, vk = td_grad.get_jk(mol, (oo0, dmz1doo+dmz1doo.T, dmzvop+dmzvop.T,
                                  dmzvom-dmzvom.T))
    vj = vj.reshape(-1,3,nao,nao)
    vk = vk.reshape(-1,3,nao,nao)
    if singlet:
        vhf1 = vj * 2 - vk
    else:
        vhf1 = numpy.vstack((vj[:2]*2-vk[:2], -vk[2:]))
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1ao[:,p0:p1] += h1[:,p0:p1] + vhf1[0,:,p0:p1]

        # Ground state gradients
        # h1ao*2 for +c.c, oo0*2 for doubly occupied orbitals
        de[k] = numpy.einsum('xpq,pq->x', h1ao, oo0) * 4

        de[k] += numpy.einsum('xpq,pq->x', h1ao, dmz1doo)
        de[k] += numpy.einsum('xqp,pq->x', h1ao, dmz1doo)
        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', vhf1[1,:,p0:p1], oo0[p0:p1])
        de[k] += numpy.einsum('xij,ij->x', vhf1[2,:,p0:p1], dmzvop[p0:p1,:]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1[3,:,p0:p1], dmzvom[p0:p1,:]) * 2
        de[k] += numpy.einsum('xji,ij->x', vhf1[2,:,p0:p1], dmzvop[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', vhf1[3,:,p0:p1], dmzvom[:,p0:p1]) * 2

    log.timer('TDHF nuclear gradients', *time0)
    return de


class Gradients(rhf_grad.Gradients):
    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self._td = td
        self._scf = td._scf
        self.chkfile = td.chkfile
        self.max_memory = td.max_memory
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-8

        self.de = 0
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** LR %s gradients for %s ********',
                 self._td.__class__, self._td._scf.__class__)
        log.info('CPHF conv_tol = %g', self.conv_tol)
        log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pyscf.lib.current_memory()[0])
        log.info('\n')
        return self

    def grad_elec(self, xy, singlet, atmlst=None):
        return kernel(self, xy, singlet, atmlst, self.max_memory, self.verbose)

    def kernel(self, xy=None, state=0, singlet=None, atmlst=None):
        cput0 = (time.clock(), time.time())
        if xy is None: xy = self._td.xy[state]
        if singlet is None: singlet = self._td.singlet
        if atmlst is None: atmlst = range(self.mol.natm)
        self.check_sanity()
        de = self.grad_elec(xy, singlet, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)

        logger.note(self, '--------------')
        logger.note(self, '           x                y                z')
        for k, ia in enumerate(atmlst):
            logger.note(self, '%d %s  %15.9f  %15.9f  %15.9f', ia,
                        self.mol.atom_symbol(ia), de[k,0], de[k,1], de[k,2])
        logger.note(self, '--------------')
        logger.timer(self, 'TD gradients', *cput0)
        return self.de


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import dft
    import pyscf.tddft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.)], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol)
    mf.scf()
    td = pyscf.tddft.TDA(mf)
    td.nstates = 3
    e, z = td.kernel()
    tdg = Gradients(td)
    #tdg.verbose = 5
    g1 = tdg.kernel(z[0])
    print(g1)
#[[ 0  0  -2.67023832e-01]
# [ 0  0   2.67023832e-01]]

    td = pyscf.tddft.TDDFT(mf)
    td.nstates = 3
    e, z = td.kernel()
    tdg = Gradients(td)
    g1 = tdg.kernel(state=0)
    print(g1)
# [[ 0  0  -2.71041021e-01]
#  [ 0  0   2.71041021e-01]]

    td = pyscf.tddft.TDA(mf)
    td.nstates = 3
    td.singlet = False
    e, z = td.kernel()
    tdg = Gradients(td)
    g1 = tdg.kernel(state=0)
    print(g1)
# [[ 0  0  -2.81048403e-01]
#  [ 0  0   2.81048403e-01]]

    td = pyscf.tddft.TDDFT(mf)
    td.nstates = 3
    td.singlet = False
    e, z = td.kernel()
    tdg = Gradients(td)
    g1 = tdg.kernel(state=0)
    print(g1)
# [[ 0  0  -2.86250870e-01]
#  [ 0  0   2.86250870e-01]]

