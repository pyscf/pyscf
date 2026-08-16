#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Composite 3c methods: B97-3c and r2SCAN-3c
'''

from pyscf import lib

# method name -> (basis, xc)
_DFT3C_METHODS = {
    'b97-3c': ('def2mtzvp', 'b97-3c'),
    'r2scan-3c': ('def2mtzvpp', 'r2scan-3c'),
}

def dft3c(mf, method='b97-3c', **kwargs):
    '''Apply a composite 3c method to the Kohn-Sham object.

    The composite 3c methods combine an XC functional with a dispersion
    correction and the gCP/SRB correction, together with a tailored basis
    set and its RI-J auxiliary basis:

    B97-3c
        B97 functional (libxc GGA_XC_B97_3C) with the D3(BJ) dispersion
        correction and the SRB correction, in the def2-mTZVP basis.
        J. G. Brandenburg et al., J. Chem. Phys. 148, 064104 (2018).

    r2SCAN-3c
        r2SCAN functional with the D4 dispersion correction and the gCP
        correction, in the def2-mTZVPP basis.
        S. Grimme et al., J. Chem. Phys. 154, 064103 (2021).

    The molecular basis is set to the tailored basis of the method and
    ``mf.xc`` is set to the method name, from which the dispersion and
    gCP/SRB corrections are derived automatically.  The RI-J auxiliary
    basis (def2-mTZVPP-RIJ) is resolved from the basis_set_exchange
    package at runtime when density fitting is enabled.

    The method returns a new DFT3C object; the input object is not
    modified.  Density fitting can be applied before or after the 3c
    setup:

    >>> mf = dft.RKS(mol).dft3c('b97-3c').density_fit().run()
    >>> mf = dft.RKS(mol).density_fit().dft3c('b97-3c').run()
    >>> mf = dft.UKS(mol).dft3c('r2scan-3c').run()

    The dispersion and gCP/SRB corrections require the pyscf-dispersion
    package (``pip install pyscf-dispersion``).  The RI-J auxiliary basis
    is resolved from the basis_set_exchange package at runtime
    (``pip install basis-set-exchange``).
    '''
    mf_class = mf.__class__
    name = mf_class.__name__
    if issubclass(mf_class, DFT3C):
        raise RuntimeError('Object %s already has the DFT3C mixin' % name)
    dft3cmf = DFT3C(mf, method, **kwargs)
    return lib.set_class(dft3cmf, (DFT3C, mf_class))

# 1. A tag to label the derived SCF class
# 2. A hook to register DFT3C specific methods
class DFT3C:
    '''
    Composite 3c method class

    Attributes for the composite 3c methods:
        method3c : str
            The name of the composite 3c method (b97-3c or r2scan-3c).
    '''

    __name_mixin__ = 'DFT3C'

    _keys = {'method3c'}

    def __init__(self, mf, method='b97-3c'):
        self.__dict__.update(mf.__dict__)
        self.method3c = method
        self._apply_dft3c(method)

    def _apply_dft3c(self, method):
        method_lower = method.lower().replace('_', '-')
        if method_lower not in _DFT3C_METHODS:
            raise NotImplementedError(
                f'Unknown 3c method {method}. Supported methods: b97-3c, r2scan-3c.')
        basis, xc = _DFT3C_METHODS[method_lower]
        self.mol.basis = basis
        self.mol.build()
        self.xc = xc
        if getattr(self, 'with_df', None) is not None:
            # density_fit was applied before dft3c.  Rebuild the density
            # fitting object for the 3c basis with the RI-J auxiliary basis.
            self.with_df = _make_df(self)

    @property
    def method(self):
        return self.method3c
    @method.setter
    def method(self, value):
        self.method3c = value
        self._apply_dft3c(value)

    def undo_dft3c(self):
        '''Remove the DFT3C mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, DFT3C))
        del obj.method3c
        return obj


def _make_df(mf):
    from pyscf import df
    from pyscf.scf import dhf
    if isinstance(mf, dhf.UHF):
        with_df = df.DF4C(mf.mol, 'def2-mTZVPP-RIJ')
    else:
        with_df = df.DF(mf.mol, 'def2-mTZVPP-RIJ')
    with_df.max_memory = mf.max_memory
    with_df.stdout = mf.stdout
    with_df.verbose = mf.verbose
    return with_df
