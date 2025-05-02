#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to specify auxiliary basis for density fitting integrals.
The format and input convention of auxbasis are the same to the AO basis.

See also examples/gto/04-input_basis.py
'''

import tempfile
from pyscf import gto, scf, df

#
# If auxbasis is not specified, default optimal auxiliary basis (if possible)
# or even-tempered gaussian functions will be generated as auxbasis
#
mol = gto.M(atom='N1 0 0 0; N2 0 0 1.2', basis={'N1':'ccpvdz', 'N2':'tzp'})
mf = scf.RHF(mol).density_fit()
mf.kernel()
print('Default auxbasis', mf.with_df.auxmol.basis)
#
# The default basis is generated in the function df.make_auxbasis.  It returns
# a basis dict for the DF auxiliary basis. In the real calculations, you can
# first generate the default basis then make modification.
#
auxbasis = df.make_auxbasis(mol)
print(mf.with_df.auxmol.basis == auxbasis)
auxbasis['N2'] = 'ccpvdz jkfit'
mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
mf.kernel()

#
# Input with key argument auxbasis='xxx' in .density_fit function
# This auxbasis will be used for all elements in the system.
#
mol = gto.M(atom='N1 0 0 0; N2 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol).density_fit(auxbasis='weigend')
mf.kernel()

#
# The DF basis can be assigned to with_df.auxbasis attribute.
# Like the AO basis input, DF basis can be specified separately for each element.
#
mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = {'default': 'weigend', 'N2': 'ahlrichs'}
mf.kernel()

#
# Combined basis set is also supported in DF basis input.
#
mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = ('weigend','sto3g')
mf.kernel()

#
# Even-tempered Gaussian DF basis can be generated based on the AO basis.
# In the following example, the exponents of auxbasis are
#    alpha = a * 1.7^i   i = 0..N
# where a and N are determined by the smallest and largest exponets of AO basis.
#
mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = df.aug_etb(mol, beta=1.7)
mf.kernel()

# Generating auxiliary basis using the AutoAux algorithms proposed by Stoychev
# (JCTC, 13, 554)
mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = df.autoaux(mol)
mf.kernel()

# The automatic generation of auxiliary basis set (see also 10.1021/acs.jctc.3c00670)
mf = scf.RHF(mol).density_fit()
mf.with_df.auxbasis = df.autoabs(mol)
mf.kernel()
