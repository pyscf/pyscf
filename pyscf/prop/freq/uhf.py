#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
See also pyscf/hessian/uhf.py
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.hessian import uhf as uhf_hess
from pyscf.prop.freq import rhf as rhf_freq


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    mol = hobj.mol
    mf = hobj._scf

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    atmlst = range(natm)
    max_memory = max(2000, hobj.max_memory - lib.current_memory()[0])
    de2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 max_memory, log)
    de2 += hobj.hess_nuc()

    # Compute H1 integrals and store in hobj.chkfile
    hobj.make_h1(mo_coeff, mo_occ, hobj.chkfile, atmlst, log)

    aoslices = mol.aoslice_by_atom()
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    fvind = uhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    def h_op(x):
        x = x.reshape(natm,3)
        hx = numpy.einsum('abxy,ax->by', de2, x)
        h1aoa = 0
        h1aob = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao_i = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/0/%d' % ia)
            h1aoa += numpy.einsum('x,xij->ij', x[ia], h1ao_i)
            h1ao_i = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/1/%d' % ia)
            h1aob += numpy.einsum('x,xij->ij', x[ia], h1ao_i)
            s1ao_i = numpy.zeros((3,nao,nao))
            s1ao_i[:,p0:p1] += s1a[:,p0:p1]
            s1ao_i[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_i)

        s1voa = reduce(numpy.dot, (mo_coeff[0].T, s1ao, mocca))
        s1vob = reduce(numpy.dot, (mo_coeff[1].T, s1ao, moccb))
        h1voa = reduce(numpy.dot, (mo_coeff[0].T, h1aoa, mocca))
        h1vob = reduce(numpy.dot, (mo_coeff[1].T, h1aob, moccb))
        mo1, mo_e1 = ucphf.solve(fvind, mo_energy, mo_occ,
                                 (h1voa,h1vob), (s1voa,s1vob))
        mo1a = numpy.dot(mo_coeff[0], mo1[0])
        mo1b = numpy.dot(mo_coeff[1], mo1[1])
        mo_e1a = mo_e1[0].reshape(nocca,nocca)
        mo_e1b = mo_e1[1].reshape(noccb,noccb)
        dm1a = numpy.einsum('pi,qi->pq', mo1a, mocca)
        dm1b = numpy.einsum('pi,qi->pq', mo1b, moccb)
        dme1a = numpy.einsum('pi,qi,i->pq', mo1a, mocca, mo_ea)
        dme1a = dme1a + dme1a.T + reduce(numpy.dot, (mocca, mo_e1a, mocca.T))
        dme1b = numpy.einsum('pi,qi,i->pq', mo1b, moccb, mo_eb)
        dme1b = dme1b + dme1b.T + reduce(numpy.dot, (moccb, mo_e1b, moccb.T))
        dme1 = dme1a + dme1b

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            h1aoa = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/0/%d' % ja)
            h1aob = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/1/%d' % ja)
            hx[ja] += numpy.einsum('xpq,pq->x', h1aoa, dm1a) * 2
            hx[ja] += numpy.einsum('xpq,pq->x', h1aob, dm1b) * 2
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1])
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1])
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag


class Frequency(uhf_hess.Hessian):
    def __init__(self, mf):
        self.nroots = 3
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-2
        uhf_hess.Hessian.__init__(self, mf)

    def kernel(self):
        self.freq, self.mode = rhf_freq.kernel(self)
        return self.freq, self.mode

    gen_hop = gen_hop

Freq = Frequency


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.spin = 2
    mol.unit = 'B'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = uhf_hess.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.075282233847343283)
    hop = gen_hop(Freq(mf))[0]
    print(lib.finger(hop(x)) - -0.075282233847343283)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print(Freq(mf).kernel()[0])
    print(numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0])
