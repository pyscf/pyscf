#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Transform a operator from four-component picture to two-component picture
'''

from pyscf import lib
from pyscf import gto
from pyscf.x2c import x2c, sfx2c1e

mol = gto.M(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

# Transform operator 1/|r-R_O|
x2cobj = x2c.X2C(mol)
with mol.with_rinv_origin((0., 0., 0.)):
    # Pass the integral names to the function picture_change
    even_operator = ('int1e_rinv_spinor', 'int1e_sprinvsp_spinor')
    v = x2cobj.picture_change(even_operator)
    print(v.shape)

# Transform dipole operator r
x2cobj = x2c.X2C(mol)
with mol.with_common_orig((0., 0., 0.)):
    # Function picture_change also supports operators in matrix representation
    c = lib.param.LIGHT_SPEED
    xmol = x2cobj.get_xmol()[0]
    rLL = xmol.intor('int1e_r_spinor')
    rSS = xmol.intor('int1e_sprsp_spinor') * (.5/c)**2
    even_operator = (rLL, rSS)
    v = x2cobj.picture_change(even_operator)
    print(v.shape)

# Transform operator 1/|r-R_O| under spin-free X2c framework
x2cobj = sfx2c1e.SpinFreeX2C(mol)
with mol.with_rinv_origin((0., 0., 0.)):
    v = x2cobj.picture_change(('int1e_rinv', 'int1e_prinvp'))
    print(v.shape)

# Transform the Hamiltonian using the picture_change function
x2cobj = sfx2c1e.SpinFreeX2C(mol)
c = lib.param.LIGHT_SPEED
xmol = x2cobj.get_xmol()[0]
t = xmol.intor_symmetric('int1e_kin')
w = xmol.intor_symmetric('int1e_pnucp')
v = 'int1e_nuc'
h1 = x2cobj.picture_change(even_operator=(v, w*(.5/c)**2-t), odd_operator=t)
print(abs(h1 - x2cobj.get_hcore()).max())
