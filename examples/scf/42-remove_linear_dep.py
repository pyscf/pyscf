#!/usr/bin/env python

'''
There is a list of methods of the SCF object one can modify to control the SCF
calculations. You can find these methods in function pyscf.scf.hf.kernel.
This example shows how to remove basis linear dependency by modifying the SCF
eig method.

The pyscf.scf.addons module provides a function remove_linear_dep_ to remove
basis linear dependency in a similar manner, but it also can handle
pathological linear dependencies via the partial Cholesky procedure.
'''


import numpy
import numpy.linalg
from pyscf import gto, scf, mcscf

mol = gto.M(atom=['H 0 0 %f'%i for i in range(10)], unit='Bohr',
            basis='ccpvtz')

#
# A regular SCF calculation for this sytem will raise a warning message
#
# Warn: Singularity detected in overlap matrix (condition number = 5.47e+09). SCF may be inaccurate and hard to converge.
#
# The linear dependency can cause HF, MCSCF etc methods converging to wrong
# answer. This example shows how to remove linear dependency from overlap
# matrix and use the linearly independent basis in the HF, MCSCF calculations.
#
# There is a shortcut function to remove linear-dependency, e.g.
#
#       mf = scf.RHF(mol).apply(scf.addons.remove_linear_dep_)
#
# which also implements the partial Cholesky ortogonalization method.
#
# This example demonstrated how the linear dependency is removed in our
# implementation.
#

#
# The smallest eigenvalue of overlap matrix is 10^{-9}
#
s = mol.intor('cint1e_ovlp_sph')
print(numpy.linalg.eigh(s)[0][:8])
#[  1.96568587e-09   8.58358923e-08   7.86870520e-07   1.89728026e-06
#   2.14355169e-06   8.96267338e-06   2.46812168e-05   3.26534277e-05]

def eig(h, s):
    d, t = numpy.linalg.eigh(s)
# Removing the eigenvectors assoicated to the smallest eigenvalue, the new
# basis defined by x matrix has 139 vectors.
    x = t[:,d>1e-8] / numpy.sqrt(d[d>1e-8])
    xhx = reduce(numpy.dot, (x.T, h, x))
    e, c = numpy.linalg.eigh(xhx)
    c = numpy.dot(x, c)
# Return 139 eigenvalues and 139 eigenvectors.
    return e, c
#
# Replacing the default eig function with the above one,  the HF solver
# generate only 139 canonical orbitals
#
mf = scf.RHF(mol)
mf.eig = eig
mf.verbose = 4
mf.kernel()

#
# The CASSCF solver takes the HF orbital as initial guess.  The MCSCF problem
# size is (0 core, 10 active, 129 external) orbitals.  This information can be
# found in the output.
#
mc = mcscf.CASSCF(mf, 10, 10)
mc.verbose = 4
mc.kernel()



#
# For symmetry adapted calculation, similar treatments can be applied.
#
# Here by assigning symmetry=1, mol.irrep_name, mol.irrep_id and mol.symm_orb
# (see pyscf/gto/mole.py) are initialized in the mol object.  They are the
# irrep symbols, IDs, and symmetry-adapted-basis.
#
mol = gto.M(atom=['H 0 0 %f'%i for i in range(10)], unit='Bohr',
            basis='ccpvtz', symmetry=1)

#
# The smallest eigenvalue is associated to A1u irrep.  Removing the relevant
# basis will not break the symmetry
#
s = mol.intor('cint1e_ovlp_sph')
for i, c in enumerate(mol.symm_orb):
    s1 = reduce(numpy.dot, (c.T, s, c))
    print(mol.irrep_name[i], numpy.linalg.eigh(s1)[0])
#A1g [  8.58358928e-08   2.14355169e-06   2.46812168e-05   3.26534277e-05
#...
#E1gx [  1.67409011e-04   2.38132838e-03   4.51022127e-03   9.89429994e-03
#...
#E1gy [  1.67409011e-04   2.38132838e-03   4.51022127e-03   9.89429994e-03
#...
#A1u [  1.96568605e-09   7.86870519e-07   1.89728026e-06   8.96267338e-06
#...

# pyscf/scf/hf_symm.py
def eig(h, s):
    from pyscf import symm
    nirrep = len(mol.symm_orb)
    h = symm.symmetrize_matrix(h, mol.symm_orb)
    s = symm.symmetrize_matrix(s, mol.symm_orb)
    cs = []
    es = []
#
# Linear dependency are removed by looping over different symmetry irreps.
#
    for ir in range(nirrep):
        d, t = numpy.linalg.eigh(s[ir])
        x = t[:,d>1e-8] / numpy.sqrt(d[d>1e-8])
        xhx = reduce(numpy.dot, (x.T, h[ir], x))
        e, c = numpy.linalg.eigh(xhx)
        cs.append(reduce(numpy.dot, (mol.symm_orb[ir], x, c)))
        es.append(e)
    e = numpy.hstack(es)
    c = numpy.hstack(cs)
    return e, c
mf = scf.RHF(mol)
mf.eig = eig
mf.verbose = 4
mf.kernel()

mc = mcscf.CASSCF(mf, 10, 10)
mc.verbose = 4
mc.kernel()
