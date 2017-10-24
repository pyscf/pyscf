#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
See also pyscf/hessian/rhf.py
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.hessian import rhf as rhf_hess


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    time0 = t1 = (time.clock(), time.time())
    mol = hobj.mol
    mf = hobj._scf

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    max_memory = max(2000, hobj.max_memory - lib.current_memory()[0])
    de2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, range(natm),
                                 max_memory, log)
    de2 += hobj.hess_nuc()

    aoslices = mol.aoslice_by_atom()
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    fvind = rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    def h_op(x):
        x = x.reshape(natm,3)
        hx = numpy.einsum('abxy,ax->by', de2, x)
        h1ao = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao_a = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/%d' % ia)
            h1ao += numpy.einsum('x,xij->ij', x[ia], h1ao_a)
            s1ao_a = numpy.zeros((3,nao,nao))
            s1ao_a[:,p0:p1] += s1a[:,p0:p1]
            s1ao_a[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_a)

        s1vo = reduce(numpy.dot, (mo_coeff.T, s1ao, mocc))
        h1vo = reduce(numpy.dot, (mo_coeff.T, h1ao, mocc))
        mo1, mo_e1 = cphf.solve(fvind, mo_energy, mo_occ, h1vo, s1vo)
        mo1 = numpy.dot(mo_coeff, mo1)
        mo_e1 = mo_e1.reshape(nocc,nocc)
        dm1 = numpy.einsum('pi,qi->pq', mo1, mocc)
        dme1 = numpy.einsum('pi,qi,i->pq', mo1, mocc, mo_energy[mo_occ>0])
        dme1 = dme1 + dme1.T + reduce(numpy.dot, (mocc, mo_e1, mocc.T))

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            h1ao = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/%s'%ja)
            hx[ja] += numpy.einsum('xpq,pq->x', h1ao, dm1) * 4
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1]) * 2
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1]) * 2
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag

def kernel(hobj):
    h_op, hdiag = gen_hop(hobj)

    def precond(x, e, *args):
        hdiagd = hdiag-e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return x/hdiagd

    e, c = lib.davidson(hop, hdiag, precond, tol=hobj.conv_tol,
                        nroots=hobj.nroots, verbose=5)
    return e, c

class Frequency(rhf_hess.Hessian):
    def __init__(self, mf):
        self.nroots = 3
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-2
        rhf_hess.Hessian.__init__(self, mf)

    def kernel(self):
        self.freq, self.mode = kernel(self)
        return self.freq, self.mode

Freq = Frequency


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.hessian import rhf

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
    hobj = rhf.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.19160804881270971)
    hop = gen_hop(hobj)[0]
    print(lib.finger(hop(x)) - -0.19160804881270971)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print Freq(mf).kernel()[0]
    print numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0]
