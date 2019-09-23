#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Accuracy of DFT analytical nuclear gradients may be affected by the treatments
of numerical integration.  Dense grids or the response of grids can be used to
improve accuracy of DFT nuclear gradients results.
'''

from pyscf import gto, dft

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = '631g')

grad = dft.RKS(mol).run(xc='b3lyp').nuc_grad_method()
grad.kernel()
# O     0.0000000000     0.0000000000     0.0124683365
# H     0.0000000000     0.0213007595    -0.0062381016
# H    -0.0000000000    -0.0213007595    -0.0062381016

#
# Second grids can be used in the nuclear gradients method.
#
grad.grids = dft.gen_grid.Grids(mol)
grad.grids.level = 6
grad.grids.prune = None
grad.kernel()
# O     0.0000000000     0.0000000000     0.0124749929
# H     0.0000000000     0.0213007958    -0.0062375176
# H    -0.0000000000    -0.0213007958    -0.0062375176

#
# The response of integration mesh was not included in the default force
# calculations.  Enabling the grids' response can give better accuracy.
#
dft.RKS(mol).run(xc='b3lyp').nuc_grad_method().run(grid_response=True)
# O     0.0000000000     0.0000000000     0.0124733388
# H     0.0000000000     0.0213006425    -0.0062366694
# H    -0.0000000000    -0.0213006425    -0.0062366694

#
# Range-separated XC functional is supported
#
dft.RKS(mol).run(xc='wb97x').nuc_grad_method().run()
# O    -0.0000000000     0.0000000000     0.0041289735
# H    -0.0000000000     0.0178316686    -0.0020388217
# H     0.0000000000    -0.0178316686    -0.0020388217

