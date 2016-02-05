#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import dft
from pyscf import grad

mol = gto.M(
    atom = [
    ['O' , (0. , 0.     , 0)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. ,  0.757 , 0.587)] ],
    basis = '631g')

dft.RKS(mol).run(conv_tol=1e-15, xc='b3lyp').apply(grad.RKS).run()

#[[ -3.44790653e-16  -2.31083509e-15   1.21670343e-02]
# [  7.15579513e-17   2.11176116e-02  -6.08866586e-03]
# [ -6.40735965e-17  -2.11176116e-02  -6.08866586e-03]]
