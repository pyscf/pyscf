#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
See also pyscf/hessian/rhf.py
'''

import numpy
from pyscf import lib
from pyscf.hessian import rhf as rhf_hess


def kernel(hobj):
    natm = hobj.mol.natm
    if hobj.nroots is None:
        h = hobj.hess().transpose(2,0,3,1).reshape(natm*3,natm*3)
        e, c = numpy.linalg.eigh(h)
        c = c.T
    else:
        # Solve part of the roots
        h_op, hdiag = hobj.gen_hop()
        def precond(x, e, *args):
            hdiagd = hdiag-e
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            return x/hdiagd
        e, c = lib.davidson(h_op, hdiag, precond, tol=hobj.conv_tol,
                            nroots=hobj.nroots, verbose=5)
        c = numpy.asarray(c)
    hobj.freq = e
    hobj.mode = c.reshape(-1,natm,3)
    return e, c

class Frequency(rhf_hess.Hessian):
    def __init__(self, mf):
        self.nroots = None
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-3
        rhf_hess.Hessian.__init__(self, mf)

    kernel = kernel

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
    mol.unit = 'B'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = rhf_hess.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.19160804881270971)
    hop = Freq(mf).gen_hop()[0]
    print(lib.finger(hop(x)) - -0.19160804881270971)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print(Freq(mf).set(nroots=1).kernel()[0])
    print(numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0])
