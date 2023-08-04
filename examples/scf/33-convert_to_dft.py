#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Converion between HF object and DFT (KS) object. The SCF energy and SCF
wavefunction are not changed during conversion.
'''

from pyscf import gto

mol = gto.M(
    atom = '''
O 0  0      0
H 0  -.757  .587
H 0   .757  .587''',
    basis = '6-311++g')

#
# Convet an HF object to a DFT object
# 
mf = mol.HF().run()
mf_dft = mf.to_rks(xc='b88,lyp')
print('Convert %s to %s', mf, mf_dft)


#
# Convet a DFT object to an HF object
#
mf = mol.KS(xc='b88,lyp').run()
mf_hf = mol.to_rhf()
print('Convert %s to %s', mf, mf_hf)

