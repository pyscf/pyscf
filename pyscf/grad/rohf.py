#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic ROHF analytical nuclear gradients
'''

from pyscf.scf import addons
from pyscf.grad import uhf as uhf_grad


class Gradients(uhf_grad.Gradients):
    '''Non-relativistic ROHF gradients
    '''
    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, addons.convert_to_uhf(mf))

Grad = Gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {'He': 'ccpvdz'}
    mol.build()
    mf = scf.ROHF(mol)
    mf.scf()
    g = Gradients(mf)
    print(g.grad())

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.build()
    rhf = scf.ROHF(mol)
    rhf.conv_tol = 1e-14
    e0 = rhf.scf()
    g = Gradients(rhf)
    print(g.grad())
#[[ 0   0               -2.41134256e-02]
# [ 0   4.39690522e-03   1.20567128e-02]
# [ 0  -4.39690522e-03   1.20567128e-02]]

    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = '631g'
    mol.charge = 1
    mol.spin = 1
    mol.build()
    rhf = scf.ROHF(mol)
    rhf.conv_tol = 1e-14
    e0 = rhf.scf()
    g = Gradients(rhf)
    print(g.grad())
#[[ 0   0                3.27774948e-03]
# [ 0   4.31591309e-02  -1.63887474e-03]
# [ 0  -4.31591309e-02  -1.63887474e-03]]
