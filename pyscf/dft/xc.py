#!/usr/bin/env python

'''
Python XC functional implementation. The backup module in case libxc and xcfun
libraries are not available
'''

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    r'''XC functional, potential and functional derivatives.
    '''
    raise NotImplementedError
