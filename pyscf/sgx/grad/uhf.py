# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

from pyscf.df.grad import uhf as dfuhf_grad
from pyscf.sgx.grad.rhf import get_jk, _GradientsMixin

class Gradients(_GradientsMixin, dfuhf_grad.Gradients):
    '''Unrestricted SGX HF gradients'''
    get_jk = get_jk

Grad = Gradients
