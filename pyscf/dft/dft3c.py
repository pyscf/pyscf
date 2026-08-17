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
Composite 3c methods: B97-3c, r2SCAN-3c and wB97X-3c
'''

from pyscf import lib

# method name -> (basis, xc, ecp, auxbasis)
_DFT3C_METHODS = {
    'b97-3c': ('def2mtzvp', 'b97-3c', None, 'def2-mTZVPP-RIJ'),
    'r2scan-3c': ('def2mtzvpp', 'r2scan-3c', None, 'def2-mTZVPP-RIJ'),
    # wB97X-3c uses the ECP-based Grimme vDZP basis; it has no dedicated
    # auxiliary basis (density fitting falls back to even-tempered
    # functions; an auto-aux basis can be requested explicitly).
    'wb97x-3c': ('Grimme vDZP', 'wb97x-3c', 'Grimme vDZP', None),
}

def dft3c(mf, method='b97-3c', **kwargs):
    '''Apply a composite 3c method to the Kohn-Sham object.

    The composite 3c methods combine an XC functional with a dispersion
    correction and, for B97-3c and r2SCAN-3c, the gCP/SRB correction,
    together with a tailored basis set:

    B97-3c
        B97 functional (libxc GGA_XC_B97_3C) with the D3(BJ) dispersion
        correction and the SRB correction, in the def2-mTZVP basis.
        J. G. Brandenburg et al., J. Chem. Phys. 148, 064104 (2018).

    r2SCAN-3c
        r2SCAN functional with the D4 dispersion correction and the gCP
        correction, in the def2-mTZVPP basis.
        S. Grimme et al., J. Chem. Phys. 154, 064103 (2021).

    wB97X-3c
        wB97X-V functional (range-separated hybrid) with the D4 dispersion
        correction, in the ECP-based Grimme vDZP basis.  The VV10 NLC of
        wB97X-V is replaced by the D4 correction and no gCP/SRB correction
        is applied.
        M. Mueller, A. Hansen, S. Grimme, J. Chem. Phys. 158, 014103 (2023).

    The molecular basis (and ECPs, for wB97X-3c) is set to the tailored
    basis of the method and ``mf.xc`` is set to the method name, from
    which the dispersion and gCP/SRB corrections are derived
    automatically.  The RI-J auxiliary basis (def2-mTZVPP-RIJ) of B97-3c
    and r2SCAN-3c is resolved from the basis_set_exchange package at
    runtime when density fitting is enabled; wB97X-3c has no dedicated
    auxiliary basis and density fitting falls back to even-tempered
    functions (an auto-aux basis can be requested explicitly).

    The method returns a new DFT3C object; the input object is not
    modified.  Density fitting can be applied before or after the 3c
    setup:

    >>> mf = dft.RKS(mol).dft3c('b97-3c').density_fit().run()
    >>> mf = dft.RKS(mol).density_fit().dft3c('b97-3c').run()
    >>> mf = dft.UKS(mol).dft3c('r2scan-3c').run()
    >>> mf = dft.RKS(mol).dft3c('wb97x-3c').run()

    The dispersion and gCP/SRB corrections require the pyscf-dispersion
    package (``pip install pyscf-dispersion``).  The Grimme vDZP basis
    and the RI-J auxiliary basis are resolved from the
    basis_set_exchange package at runtime
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
            The name of the composite 3c method
            (b97-3c, r2scan-3c or wb97x-3c).
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
                f'Unknown 3c method {method}. Supported methods: '
                f'b97-3c, r2scan-3c, wb97x-3c.')
        basis, xc, ecp, auxbasis = _DFT3C_METHODS[method_lower]
        self.mol.basis = basis
        if ecp is not None:
            # The ECPs of the basis are defined only for the heavy elements.
            # Set the ECP per element from the BSE record so that light
            # elements without an ECP are not looked up.
            from pyscf.gto.basis import bse as bse_mod
            if bse_mod.basis_set_exchange is None:
                raise RuntimeError('basis_set_exchange is required for the '
                                   'ECPs of the %s basis' % ecp)
            atoms = sorted({a[0] for a in self.mol._atom})
            bse_obj = bse_mod.basis_set_exchange.api.get_basis(ecp, elements=atoms)
            ecp_basis = bse_mod._ecp_basis(bse_obj)
            self.mol.ecp = {a: ecp for a in ecp_basis}
        self.mol.build()
        self.xc = xc
        if getattr(self, 'with_df', None) is not None:
            # density_fit was applied before dft3c.  Rebuild the density
            # fitting object for the 3c basis and auxiliary basis.
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
    auxbasis = _DFT3C_METHODS[mf.method3c.lower().replace('_', '-')][3]
    if isinstance(mf, dhf.UHF):
        with_df = df.DF4C(mf.mol, auxbasis)
    else:
        with_df = df.DF(mf.mol, auxbasis)
    with_df.max_memory = mf.max_memory
    with_df.stdout = mf.stdout
    with_df.verbose = mf.verbose
    return with_df
