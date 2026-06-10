# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

from pyscf.df.grad import uks as dfuks_grad
from pyscf.sgx.grad.rhf import get_jk, _GradientsMixin

class Gradients(_GradientsMixin, dfuks_grad.Gradients):
    '''Restricted SGX RKS gradients'''
    get_jk = get_jk

Grad = Gradients
