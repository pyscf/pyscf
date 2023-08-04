#!/usr/bin/env python
#
# Author: Shiv Upadhyay <shivnupadhyay@gmail.com>
#

'''
VVO and LIVVO generation for a water molecule
'''

from pyscf import gto, lo, tools, dft
from pyscf.lo import iao, orth


mol = gto.Mole()
mol.verbose = 1
mol.atom = '''
O	 0.0000000	 0.0000000	 0.0000000
H	 0.7569685	 0.0000000	-0.5858752
H	-0.7569685	 0.0000000	-0.5858752'''
mol.unit = 'Angstrom'
mol.basis = 'aug-cc-pvtz'
mol.symmetry = 1
mol.build()

mf = dft.RKS(mol)
mf.xc = 'HF*0.2 + .08*LDA + .72*B88, .81*LYP + .19*VWN5'
mf.kernel()

orbocc = mf.mo_coeff[:,0:mol.nelec[0]]
orbvirt = mf.mo_coeff[:,mol.nelec[0]:]
mocoeff = mf.mo_coeff

ovlpS = mol.intor_symmetric('int1e_ovlp')

# plot canonical mos
iaos = iao.iao(mol, orbocc)
iaos = orth.vec_lowdin(iaos, ovlpS)
for i in range(iaos.shape[1]):
    tools.cubegen.orbital(mol, 'h2o_cmo_{:02d}.cube'.format(i+1), mocoeff[:,i])

# plot intrinsic atomic orbitals
for i in range(iaos.shape[1]):
    tools.cubegen.orbital(mol, 'h2o_iao_{:02d}.cube'.format(i+1), iaos[:,i])

# plot intrinsic bonding orbitals
count = 0
ibos = lo.ibo.ibo(mol, orbocc, locmethod='IBO')
for i in range(ibos.shape[1]):
    count += 1
    tools.cubegen.orbital(mol, 'h2o_ibo_{:02d}.cube'.format(count), ibos[:,i])

# plot valence virtual orbitals and localized valence virtual orbitals
vvo = lo.vvo.vvo(mol, orbocc, orbvirt)
livvo = lo.vvo.livvo(mol, orbocc, orbvirt)
for i in range(vvo.shape[1]):
    count += 1
    tools.cubegen.orbital(mol, 'h2o_vvo_{:02d}.cube'.format(count), vvo[:,i])
    tools.cubegen.orbital(mol, 'h2o_livvo_{:02d}.cube'.format(count), livvo[:,i])

