# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

from pyscf.df.grad import rks as dfrks_grad
from pyscf.sgx.grad.rhf import get_jk, _GradientsMixin

__all__ = ['Gradients', 'Grad']

class Gradients(_GradientsMixin, dfrks_grad.Gradients):
    '''Restricted SGX RKS gradients'''
    get_jk = get_jk

Grad = Gradients
