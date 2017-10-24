#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
See also pyscf/hessian/uks.py
'''

from pyscf.hessian import uks as uks_hess
from pyscf.prop.freq import rhf as rhf_freq
from pyscf.prop.freq import uhf as uhf_freq

class Frequency(uks_hess.Hessian):
    def __init__(self, mf):
        self.nroots = 3
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-2
        uks_hess.Hessian.__init__(self, mf)

    def kernel(self):
        self.freq, self.mode = rhf_freq.kernel(self)
        return self.freq, self.mode

    gen_hop = uhf_freq.gen_hop

Freq = Frequency


if __name__ == '__main__':
    import numpy
    from pyscf import lib
    from pyscf import gto
    from pyscf import scf, dft

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
    mf = dft.UKS(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = uks_hess.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.15366054253827535)
    hop = Freq(mf).gen_hop()[0]
    print(lib.finger(hop(x)) - -0.15366054253827535)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print(Freq(mf).kernel()[0])
    print(numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0])
