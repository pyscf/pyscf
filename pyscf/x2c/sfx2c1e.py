#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

'''
1-electron Spin-free X2C approximation
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import ghf
from pyscf.x2c import x2c


def sfx2c1e(mf):
    '''Spin-free X2C.
    For the given SCF object, it updates the hcore constructor.  All integrals
    are computed in the real spherical GTO basis.

    Args:
        mf : an SCF object

    Returns:
        An SCF object

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol).sfx2c1e()
    >>> mf.scf()

    >>> import pyscf.x2c.sfx2c1e
    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = pyscf.x2c.sfx2c1e.sfx2c1e(scf.UHF(mol))
    >>> mf.scf()
    '''
    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            return mf.__class__(mf)
        else:
            return mf

    assert(isinstance(mf, hf.SCF))

    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__
    class SFX2C1E_SCF(mf_class, x2c._X2C_SCF):
        __doc__ = doc + \
        '''
        Attributes for spin-free X2C:
            with_x2c : X2C object
        '''
        def __init__(self, mf):
            self.__dict__.update(mf.__dict__)
            self.with_x2c = SpinFreeX2C(mf.mol)
            self._keys = self._keys.union(['with_x2c'])

        def get_hcore(self, mol=None):
            if self.with_x2c:
                hcore = self.with_x2c.get_hcore(mol)
                if isinstance(self, ghf.GHF):
                    hcore = scipy.linalg.block_diag(hcore, hcore)
                return hcore
            else:
                return mf_class.get_hcore(self, mol)

        def dump_flags(self):
            mf_class.dump_flags(self)
            if self.with_x2c:
                self.with_x2c.dump_flags()
            return self

    return SFX2C1E_SCF(mf)

sfx2c = sfx2c1e


class SpinFreeX2C(x2c.X2C):
    '''1-component X2c (spin-free part only)
    '''
    def get_hcore(self, mol=None):
        '''1-component X2c Foldy-Wouthuysen (FW Hamiltonian  (spin-free part only)
        '''
        if mol is None: mol = self.mol
        xmol, contr_coeff = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert('1E' in self.approx.upper())
        t = xmol.intor_symmetric('int1e_kin')
        v = xmol.intor_symmetric('int1e_nuc')
        s = xmol.intor_symmetric('int1e_ovlp')
        w = xmol.intor_symmetric('int1e_pnucp')
        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            nao = xmol.nao_nr()
            x = numpy.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = xmol.intor('int1e_kin', shls_slice=shls_slice)
                s1 = xmol.intor('int1e_ovlp', shls_slice=shls_slice)
                with xmol.with_rinv_as_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = z * xmol.intor('int1e_rinv', shls_slice=shls_slice)
                    w1 = z * xmol.intor('int1e_prinvp', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
            h1 = x2c._get_hcore_fw(t, v, w, s, x, c)
        else:
            h1 = x2c._x2c1e_get_hcore(t, v, w, s, c)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp')
            s21 = gto.intor_cross('int1e_ovlp', xmol, mol)
            c = lib.cho_solve(s22, s21)
            h1 = reduce(numpy.dot, (c.T, h1, c))
        if self.xuncontract and contr_coeff is not None:
            h1 = reduce(numpy.dot, (contr_coeff.T, h1, contr_coeff))
        return h1

    def hcore_deriv_generator(self, mol=None, deriv=1):
        from pyscf.x2c import sfx2c1e_grad
        from pyscf.x2c import sfx2c1e_hess
        if deriv == 1:
            return sfx2c1e_grad.hcore_grad_generator(self, mol)
        elif deriv == 2:
            return sfx2c1e_hess.hcore_hess_generator(self, mol)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz-dk',
    )

    method = hf.RHF(mol)
    enr = method.kernel()
    print('E(NR) = %.12g' % enr)

    method = sfx2c1e(hf.RHF(mol))
    esfx2c = method.kernel()
    print('E(SFX2C1E) = %.12g' % esfx2c)
    method.with_x2c.basis = 'unc-ccpvqz-dk'
    print('E(SFX2C1E) = %.12g' % method.kernel())
    method.with_x2c.approx = 'atom1e'
    print('E(SFX2C1E) = %.12g' % method.kernel())

