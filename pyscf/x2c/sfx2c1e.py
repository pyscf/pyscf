#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
from pyscf.data import nist


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
    assert isinstance(mf, hf.SCF)

    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinFreeX2CHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SpinFreeX2CHelper):
            # An object associated to x2c.SpinOrbitalX2CHelper
            raise NotImplementedError
        else:
            return mf

    return lib.set_class(SFX2C1E_SCF(mf), (SFX2C1E_SCF, mf.__class__))

sfx2c = sfx2c1e

class SFX2C1E_SCF(x2c._X2C_SCF):
    '''
    Attributes for spin-free X2C:
        with_x2c : X2C object
    '''

    __name_mixin__ = 'sfX2C1e'

    _keys = {'with_x2c'}

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinFreeX2CHelper(mf.mol)

    def undo_x2c(self):
        '''Remove the X2C Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, SFX2C1E_SCF))
        del obj.with_x2c
        return obj

    def get_hcore(self, mol=None):
        if self.with_x2c:
            hcore = self.with_x2c.get_hcore(mol)
            if isinstance(self, ghf.GHF):
                hcore = scipy.linalg.block_diag(hcore, hcore)
            return hcore
        else:
            return super(x2c._X2C_SCF, self).get_hcore(mol)

    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   picture_change=True, **kwargs):
        r''' Dipole moment calculation with picture change correction

        Args:
             mol: an instance of :class:`Mole`
             dm : a 2D ndarrays density matrices

        Kwarg:
            picture_chang (bool) : Whether to compute the dipole moment with
            picture change correction.

        Return:
            A list: the dipole moment on x, y and z component
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        log = logger.new_logger(mol, verbose)

        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            # UHF density matrices
            dm = dm[0] + dm[1]

        if isinstance(self, ghf.GHF):
            nao = mol.nao_nr()
            dm = dm[:nao,:nao] + dm[nao:,nao:]

        with mol.with_common_orig((0,0,0)):
            if picture_change:
                xmol = self.with_x2c.get_xmol()[0]
                nao = xmol.nao
                prp = xmol.intor_symmetric('int1e_sprsp').reshape(3,4,nao,nao)[:,3]
                c1 = 0.5/lib.param.LIGHT_SPEED
                ao_dip = self.with_x2c.picture_change(('int1e_r', prp*c1**2))
            else:
                ao_dip = mol.intor_symmetric('int1e_r')

        el_dip = numpy.einsum('xij,ji->x', ao_dip, dm).real

        charges = mol.atom_charges()
        coords  = mol.atom_coords()
        nucl_dip = numpy.einsum('i,ix->x', charges, coords)
        mol_dip = nucl_dip - el_dip

        if unit.upper() == 'DEBYE':
            mol_dip *= nist.AU2DEBYE
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an X2C object. Convert dst to X2C object.')
            dst = dst.sfx2c()
        return hf.SCF._transfer_attrs_(self, dst)

    def to_gpu(self):
        obj = self.undo_x2c().to_gpu().sfx2c1e()
        return lib.to_gpu(self, obj)


class SpinFreeX2CHelper(x2c.X2CHelperBase):
    '''1-component X2c (spin-free part only)
    '''
    def get_hcore(self, mol=None):
        '''1-component X2c Foldy-Wouthuysen (FW Hamiltonian  (spin-free part only)
        '''
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())
        t = xmol.intor_symmetric('int1e_kin')
        v = xmol.intor_symmetric('int1e_nuc')
        s = xmol.intor_symmetric('int1e_ovlp')
        w = xmol.intor_symmetric('int1e_pnucp')
        if 'get_xmat' in self.__dict__:
            # If the get_xmat method is overwritten by user, build the X
            # matrix with the external get_xmat method
            x = self.get_xmat(xmol)
            h1 = x2c._get_hcore_fw(t, v, w, s, x, c)

        elif 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            nao = xmol.nao_nr()
            x = numpy.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = xmol.intor('int1e_kin', shls_slice=shls_slice)
                s1 = xmol.intor('int1e_ovlp', shls_slice=shls_slice)
                with xmol.with_rinv_at_nucleus(ia):
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

    def picture_change(self, even_operator=(None, None), odd_operator=None):
        '''Picture change for even_operator + odd_operator

        even_operator has two terms at diagonal blocks
        [ v  0 ]
        [ 0  w ]

        odd_operator has the term at off-diagonal blocks
        [ 0    p ]
        [ p^T  0 ]

        v, w, and p can be strings (integral name) or matrices.
        '''
        mol = self.mol
        xmol, c = self.get_xmol(mol)
        pc_mat = self._picture_change(xmol, even_operator, odd_operator)

        if self.basis is not None:
            s22 = xmol.intor_symmetric('int1e_ovlp')
            s21 = gto.mole.intor_cross('int1e_ovlp', xmol, mol)
            c = lib.cho_solve(s22, s21)

        elif self.xuncontract:
            pass

        else:
            return pc_mat

        if pc_mat.ndim == 2:
            return lib.einsum('pi,pq,qj->ij', c, pc_mat, c)
        else:
            return lib.einsum('pi,xpq,qj->xij', c, pc_mat, c)

    def get_xmat(self, mol=None):
        if mol is None:
            xmol = self.get_xmol(mol)[0]
        else:
            xmol = mol
        c = lib.param.LIGHT_SPEED
        assert ('1E' in self.approx.upper())

        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            nao = xmol.nao_nr()
            x = numpy.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                t1 = xmol.intor('int1e_kin', shls_slice=shls_slice)
                s1 = xmol.intor('int1e_ovlp', shls_slice=shls_slice)
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    v1 = z * xmol.intor('int1e_rinv', shls_slice=shls_slice)
                    w1 = z * xmol.intor('int1e_prinvp', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
        else:
            t = xmol.intor_symmetric('int1e_kin')
            v = xmol.intor_symmetric('int1e_nuc')
            s = xmol.intor_symmetric('int1e_ovlp')
            w = xmol.intor_symmetric('int1e_pnucp')
            x = x2c._x2c1e_xmatrix(t, v, w, s, c)
        return x

    def _get_rmat(self, x=None):
        '''The matrix (in AO basis) that changes metric from NESC metric to NR metric'''
        xmol = self.get_xmol()[0]
        if x is None:
            x = self.get_xmat(xmol)
        c = lib.param.LIGHT_SPEED
        s = xmol.intor_symmetric('int1e_ovlp')
        t = xmol.intor_symmetric('int1e_kin')
        s1 = s + reduce(numpy.dot, (x.conj().T, t, x)) * (.5/c**2)
        return x2c._get_r(s, s1)

    def hcore_deriv_generator(self, mol=None, deriv=1):
        from pyscf.x2c import sfx2c1e_grad
        from pyscf.x2c import sfx2c1e_hess
        if deriv == 1:
            return sfx2c1e_grad.hcore_grad_generator(self, mol)
        elif deriv == 2:
            return sfx2c1e_hess.hcore_hess_generator(self, mol)
        else:
            raise NotImplementedError

SpinFreeX2C = SpinFreeX2CHelper


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

    mf = method.density_fit().undo_x2c().run()
    print('E(DF-NR) = %.12g' % mf.e_tot)
