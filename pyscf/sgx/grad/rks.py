# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

from pyscf.df.grad import rks as dfrks_grad
from pyscf.sgx.grad.rhf import get_jk

class Gradients(dfrks_grad.Gradients):
    '''Unrestricted SGX HF gradients'''
    def __init__(self, mf):
        self.sgx_grid_response = True
        if mf.with_df.direct_j:
            raise ValueError("direct_j setting not supported for gradients")
        dfrks_grad.Gradients.__init__(self, mf)

    get_jk = get_jk

Grad = Gradients
