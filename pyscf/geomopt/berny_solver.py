'''
Interface to geometry optimizer pyberny
(In testing)
'''
from __future__ import absolute_import
try:
    from berny import Berny, geomlib, Logger
except ImportError:
    raise ImportError('Geometry optimizer pyberny not found.\npyberny library '
                      'can be found on github https://github.com/azag0/pyberny')

from pyscf import lib
from pyscf.geomopt.grad import gen_grad_solver

def kernel(method, mol=None):
    '''Optimize the
    '''
    if mol is None:
        mol = method.mol
    geom = to_berny_geom(mol)
    g_solver = gen_grad_solver(method)
    optimizer = Berny(geom, log=Logger(out=method.stdout), **kwargs)
    dm0 = None
    for geom in optimizer:
        atom = geom_to_atom(geom)
        mol.set_geom_(atom)
        energy, gradients = g_solver(mol)
        optimizer.send((energy, gradients))
    return geom

def to_berny_geom(mol):
    species = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = mol.atom_coords() * lib.param.BOHR
    return geomlib.Molecule(species, coords)

def geom_to_atom(geom):
    return list(geom)

def as_berny_solver(method, mol=None):
    '''
    Generate a solver for berny optimize function
    '''
    if mol is None:
        mol = method.mol
    g_solver = gen_grad_solver(method)
    atom = yield
    while True:
        mol.set_geom_(atom)
        energy, gradients = g_solver(mol)
        atom = yield energy, gradients


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf, dft, cc
    from berny import optimize
    mol = gto.M(atom='''
C       1.1879  -0.3829 0.0000
C       0.0000  0.5526  0.0000
O       -1.1867 -0.2472 0.0000
H       -1.9237 0.3850  0.0000
H       2.0985  0.2306  0.0000
H       1.1184  -1.0093 0.8869
H       1.1184  -1.0093 -0.8869
H       -0.0227 1.1812  0.8852
H       -0.0227 1.1812  -0.8852
                ''',
                basis='3-21g')
    mol1 = mol.copy()

    mf = scf.RHF(mol)
    print(kernel(mf).dumps('xyz'))
    print(optimize(as_berny_solver(mf), to_berny_geom(mol)).dumps('xyz'))

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.conv_tol = 1e-7
    print(kernel(mf, mol1).dumps('xyz'))

    mycc = cc.CCSD(scf.RHF(mol))
    print(kernel(mycc).dumps('xyz'))

