#!/usr/bin/env python
# Copyright 2020-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

from abc import ABC, abstractmethod
import numpy as np
from pyscf import lib
from pyscf.pbc.symm import geom
from pyscf.pbc.symm.tables import SchoenfliesNotation

def _round_zero(a, tol=1e-9):
    a[np.where(abs(a) < tol)] = 0
    return a

class GroupElement(ABC):
    '''
    The abstract class for group elements.
    '''
    def __call__(self, other):
        return self.__matmul__(other)

    @abstractmethod
    def __matmul__(self, other):
        pass

    def __mul__(self, other):
        assert isinstance(other, self.__class__)
        return self @ other

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def inv(self):
        '''
        Inverse of the group element.
        '''
        pass


class PGElement(GroupElement):
    '''
    The class for crystallographic point group elements.
    The group elements are rotation matrices represented
    in lattice translation vector basis.

    Attributes:
        matrix : (d,d) array of ints
            Rotation matrix in lattice translation vector basis.
        dimension : int
            Dimension of the space: `d`.
    '''
    def __init__(self, matrix):
        self.matrix = matrix
        self.dimension = matrix.shape[0]

    def __matmul__(self, other):
        if not isinstance(other, PGElement):
            raise TypeError(f"{other} is not a point group element.")
        return PGElement(np.dot(self.matrix, other.matrix))

    def __repr__(self):
        return self.matrix.__repr__()

    def __hash__(self):
        def _id(op):
            s = op.flatten() + 1
            return lib.inv_base_repr_int(s, op.shape[0])

        r = _id(self.rot)
        # move identity to the first place
        d = self.dimension
        r -= _id(np.eye(d, dtype=int))
        if r < 0:
            r += _id(np.ones((d,d), dtype=int)) + 1
        return int(r)

    @staticmethod
    def decrypt_hash(h, dimension=3):
        if dimension == 3:
            id_eye = int('211121112', 3)
            id_max = int('222222222', 3)
        elif dimension == 2:
            id_eye = int('2112', 3)
            id_max = int('2222', 3)
        else:
            raise NotImplementedError

        r = h + id_eye
        if r > id_max:
            r -= id_max + 1
        #s = np.base_repr(r, 3)
        #s = '0'*(dimension**2-len(s)) + s
        #rot = np.asarray([int(i) for i in s]) - 1
        rot = np.asarray(lib.base_repr_int(r, 3, dimension**2)) - 1
        rot = rot.reshape(dimension,dimension)
        # sanity check
        #element = PGElement(rot)
        #assert hash(element) == h
        return rot

    def __lt__(self, other):
        if not isinstance(other, PGElement):
            raise TypeError(f"{other} is not a point group element.")
        return self.__hash__() < other.__hash__()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, a):
        self._matrix = a

    rot = matrix

    def inv(self):
        return PGElement(np.asarray(np.linalg.inv(self.matrix), dtype=np.int32))


class FiniteGroup(ABC):
    '''
    The class for finite groups.

    Attributes:
        elements : list
            Group elements.
        order : int
            Group order.
    '''
    def __init__(self, elements, from_hash=False):
        if from_hash:
            elements = self.__class__.elements_from_hash(elements)
        self.elements = np.asarray(elements)
        self._order = None
        self._hash_table = None
        self._inverse_table = None
        self._multiplication_table = None
        self._conjugacy_table = None
        self._conjugacy_mask = None
        self._chartab_full = None
        self._chartab = None
        self._group_name = None
        self._group_index = None
        self._check_sanity()

    @staticmethod
    @abstractmethod
    def elements_from_hash(hashes, **kwargs):
        pass

    def __len__(self):
        return self.order

    def __getitem__(self, i):
        return self.elements[i]

    def __and__(self, other):
        if type(self) is not type(other):
            raise TypeError(f'{self} and {other} must be the same type')
        if self is other:
            return self
        hi = list(self.hash_table.keys())
        hj = list(other.hash_table.keys())
        hij = np.intersect1d(hi, hj)
        return self.__class__(hij, from_hash=True)

    def __or__(self, other):
        if type(self) is not type(other):
            raise TypeError(f'{self} and {other} must be the same type')
        if self is other:
            return self
        hi = list(self.hash_table.keys())
        hj = list(other.hash_table.keys())
        hij = np.union1d(hi, hj)
        return self.__class__(hij, from_hash=True)

    def issubset(self, other):
        if type(self) is not type(other):
            raise TypeError(f'{self} and {other} must be the same type')
        if self is other:
            return True
        hi = list(self.hash_table.keys())
        hj = list(other.hash_table.keys())
        return set(hi).issubset(set(hj))

    @property
    def order(self):
        if self._order is None:
            self._order = len(self.elements)
        return self._order

    @order.setter
    def order(self, n):
        self._order = n

    @property
    def hash_table(self):
        '''
        Hash table for group elements: {hash : index}.
        '''
        if self._hash_table is None:
            self._hash_table = {hash(g) : i for i, g in enumerate(self.elements)}
        return self._hash_table

    @hash_table.setter
    def hash_table(self, table):
        self._hash_table = table

    @property
    def inverse_table(self):
        '''
        Table for inverse of the group elements.

        Return : (n,) array of ints
            The indices of elements.
        '''
        if self._inverse_table is None:
            _table = [self.hash_table[hash(g.inv())] for g in self.elements]
            self._inverse_table = np.asarray(_table)
        return self._inverse_table

    @inverse_table.setter
    def inverse_table(self, table):
        self._inverse_table = table

    @property
    def multiplication_table(self):
        '''
        Multiplication table of the group.

        Return : (n, n) array of ints
             The indices of elements.
        '''
        if self._multiplication_table is None:
            prod = self.elements[:,None] * self.elements[None,:]
            _table = [self.hash_table[hash(gh)] for gh in prod.flatten()]
            self._multiplication_table = np.asarray(_table).reshape(prod.shape)
        return self._multiplication_table

    @multiplication_table.setter
    def multiplication_table(self, table):
        self._multiplication_table = table

    @property
    def conjugacy_table(self):
        '''
        conjugacy_table[`index_g`, `index_x`] returns the index of element `h`,
        where :math:`h = x * g * x^{-1}`.
        '''
        if self._conjugacy_table is None:
            prod_table = self.multiplication_table
            g_xinv = prod_table[:,self.inverse_table]
            self._conjugacy_table = prod_table[np.arange(self.order)[None,:], g_xinv]
        return self._conjugacy_table

    @conjugacy_table.setter
    def conjugacy_table(self, table):
        self._conjugacy_table = table

    @property
    def conjugacy_mask(self):
        '''
        Boolean mask array indicating whether two elements
        are conjugate with each other.
        '''
        if self._conjugacy_mask is None:
            n = self.order
            is_conjugate = np.zeros((n,n), dtype=bool)
            is_conjugate[np.arange(n)[:,None], self.conjugacy_table] = True
            self._conjugacy_mask = is_conjugate
        return self._conjugacy_mask

    def conjugacy_classes(self):
        '''
        Compute conjugacy classes.

        Returns:
            classes : (n_irrep,n) boolean array
                The indices of `True` correspond to the
                indices of elements in this class.
            representatives : (n_irrep,) array of ints
                Representive elements' indices in each class.
            inverse : (n,) array of ints
                The indices to reconstruct `conjugacy_mask` from `classes`.
        '''
        _, idx = np.unique(self.conjugacy_mask, axis=0, return_index=True)
        representatives = np.sort(idx)
        classes = self.conjugacy_mask[representatives]
        inverse = -np.ones((self.order), dtype=int)
        diff = (self.conjugacy_mask[None,:,:]==classes[:,None,:]).all(axis=-1)
        for i, a in enumerate(diff):
            inverse[np.where(a)[0]] = i
        assert (inverse >= 0).all()
        assert (classes[inverse] == self.conjugacy_mask).all()
        return classes, representatives, inverse

    def character_table(self, return_full_table=False, recompute=False):
        '''
        Character table of the group.

        Args:
            return_full_table : bool
                If True, return the characters for all elements.
            recompute : bool
                Whether to recompute the character table. Default is False,
                meaning to use the cached table if possible.

        Returns:
            chartab : array
                Character table for classes.
            chartab_full : array, optional
                Character table for all elements.
        '''
        if not recompute:
            if not return_full_table and self._chartab is not None:
                return self._chartab
            if return_full_table and self._chartab_full is not None:
                return self._chartab_full

        classes, _, inverse = self.conjugacy_classes()
        class_sizes = classes.sum(axis=1)

        ginv_h = self.multiplication_table[self.inverse_table]
        M  = classes @ np.random.rand(self.order)[ginv_h] @ classes.T
        M /= class_sizes

        _, Rchi = np.linalg.eig(M)
        chi = Rchi.T / class_sizes

        norm = np.sum(np.abs(chi) ** 2 * class_sizes[None,:], axis=1) ** 0.5
        chi  = chi / norm[:,None] * self.order ** 0.5
        chi /= (chi[:, 0] / np.abs(chi[:, 0]))[:,None]
        chi  = np.round(chi, 9)
        chi_copy = chi.copy()
        chi_copy[:,1:] *= -1
        idx = np.lexsort(np.rot90(chi_copy))
        chi = chi[idx]
        chi = _round_zero(chi)
        self._chartab = chi
        self._chartab_full = chi[:, inverse]
        if return_full_table:
            return self._chartab_full
        else:
            return self._chartab

    def project_chi(self, chi, other):
        '''
        Project characters to another group.
        '''
        if self is other:
            return chi
        i_ind, j_ind = self.get_elements_map(other)
        chi_j = np.zeros((other.order,))
        chi_j[j_ind] = chi[i_ind]
        return chi_j

    def get_elements_map(self, other):
        if not (other.issubset(self) or self.issubset(other)):
            raise KeyError(f'{self} or {other} must be a subset of the other.')
        hi = [hash(g) for g in self.elements]
        hj = [hash(g) for g in other.elements]
        _, i_ind, j_ind = np.intersect1d(hi, hj, return_indices=True)
        return i_ind, j_ind

    def get_irrep_chi(self, ir):
        chartab = self.character_table(True)
        return chartab[ir]

    def _check_sanity(self):
        try:
            # check duplication
            assert len(self.hash_table) == self.order
            # check inversion
            assert (abs(np.sort(self.inverse_table)
                       -np.arange(self.order)).max() == 0)
            # check multiplication
            assert (abs(np.sort(self.multiplication_table, axis=-1)
                       -np.arange(self.order)[None,:]).max() == 0)
        except (AssertionError, KeyError, ValueError):
            raise ValueError('The elements do not form a group.')


class PointGroup(FiniteGroup):
    '''
    The class for crystallographic point groups.
    '''
    def group_name(self, notation='international'):
        if self._group_name is not None and notation=='international':
            return self._group_name
        name = geom.get_crystal_class(None, self.elements)[0]
        self._group_name = name
        if notation.lower().startswith('scho'): # Schoenflies
            name = SchoenfliesNotation[name]
        return name

    @property
    def group_index(self):
        if self._group_index is None:
            name = self.group_name()
            self._group_index = list(SchoenfliesNotation.keys()).index(name)
        return self._group_index

    @staticmethod
    def elements_from_hash(hashes, dimension=3):
        elements = [PGElement(PGElement.decrypt_hash(h, dimension)) for h in hashes]
        return elements


class Representation:
    '''
    Helper class for representation reductions.
    Only characters are stored at the moment.
    '''
    def __init__(self, group, rep=None, chi=None):
        self.group = group
        self.rep = rep
        self.chi = chi

    @property
    def rep(self):
        if self._rep is None:
            self._rep = self.chi_to_rep(self.chi)
        return self._rep

    @rep.setter
    def rep(self, value):
        self._rep = value

    @property
    def chi(self):
        if self._chi is None:
            self._chi = self.rep_to_chi(self.rep)
        return self._chi

    @chi.setter
    def chi(self, value):
        self._chi = value

    def rep_to_chi(self, rep):
        chartab = self.group.character_table(True)
        chi = np.einsum('ni,n->i', chartab, rep)
        return chi

    def chi_to_rep(self, chi):
        group = self.group
        chartab = group.character_table(True)
        assert len(chi) == chartab.shape[1]
        nA = np.einsum('ni,i->n', chartab.conj(), chi) / group.order
        assert (abs(nA - nA.round()) < 1e-9).all()
        nA = np.rint(nA).astype(int)
        return nA

    def __matmul__(self, other):
        g1, chi1 =  self.group,  self.chi
        g2, chi2 = other.group, other.chi
        g12 = g1 & g2
        chi1_proj = g1.project_chi(chi1, g12)
        chi2_proj = g2.project_chi(chi2, g12)
        chi12_proj = chi1_proj * chi2_proj
        return self.__class__(g12, chi=chi12_proj)
