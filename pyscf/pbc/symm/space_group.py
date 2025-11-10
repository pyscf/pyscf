#!/usr/bin/env python
# Copyright 2020-2023 The PySCF Developers. All Rights Reserved.
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

import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.symm import geom
from functools import reduce
from pyscf.pbc.symm.group import PGElement

SYMPREC = getattr(__config__, 'pbc_symm_space_group_symprec', 1e-6)
XYZ = np.eye(3)

def transform_rot(op, a, b, allow_non_integer=False):
    r'''
    Transform rotation operator from :math:`\mathbf{a}` basis system to :math:`\mathbf{b}` basis system.

    Note:
        This function raises error when the point-group symmetries of the two basis systems are different.

    Arguments:
        op : (3,3) array
            Rotation operator in :math:`\mathbf{a}` basis system.
        a : (3,3) array
            Basis vectors of :math:`\mathbf{a}` basis system (row-major).
        b : (3,3) array
            Basis vectors of :math:`\mathbf{b}` basis system (row-major).
        allow_non_integer : bool
            Whether to allow non-integer rotation matrix in the new basis system.
            Default value is False.

    Returns:
        A (3,3) array
            Rotation operator in :math:`\mathbf{b}` basis system.
    '''
    P = np.dot(np.linalg.inv(b.T), a.T)
    R = reduce(np.dot,(P, op, np.linalg.inv(P))).round(15)
    R[np.where(abs(R) < 1e-9)] = 0
    if not allow_non_integer:
        if(np.amax(np.absolute(R-R.round())) > SYMPREC):
            raise RuntimeError("Point-group symmetries of the two coordinate systems are different.")
        return R.round().astype(int)
    else:
        return R

def transform_trans(op, a, b):
    r'''
    Transform translation operator from :math:`\mathbf{a}` basis system to :math:`\mathbf{b}` basis system.

    Arguments:
        op : (3,) array
            Translation operator in :math:`\mathbf{a}` basis system.
        a : (3,3) array
            Basis vectors of :math:`\mathbf{a}` basis system (row-major).
        b : (3,3) array
            Basis vectors of :math:`\mathbf{b}` basis system (row-major).

    Returns:
        A (3,) array
            Translation operator in :math:`\mathbf{b}` basis system.
    '''
    P = np.dot(np.linalg.inv(b.T), a.T)
    return np.dot(op, P.T)


class SPGElement():
    '''
    Matrix representation of space group operations

    Attributes:
        rot : (d,d) array
            Rotation operator.
        trans : (d,) array
            Translation operator.
        dimension : int
            Dimension of the space: `d`.
    '''
    def __init__(self,
                 rot=np.eye(3, dtype=np.int32),
                 trans=np.zeros((3)), dimension=3):
        self.rot = np.asarray(rot)
        self.trans = np.asarray(trans)
        self.dimension = dimension
        if dimension != 3:
            raise NotImplementedError

    def dot(self, r_or_op):
        '''
        Operates on a point or multiplication of two operators
        '''
        if isinstance(r_or_op, np.ndarray) and r_or_op.ndim==1 and len(r_or_op)==3:
            return np.dot(r_or_op, self.rot.T) + self.trans
        elif isinstance(r_or_op, SPGElement):
            beta = self.rot
            b = self.trans
            alpha = r_or_op.rot
            a = r_or_op.trans
            op = SPGElement(np.dot(beta, alpha), b + np.dot(a, beta.T))
            return op
        else:
            raise KeyError("Input has wrong type: %s" % type(r_or_op))

    def dot_rot(self, r):
        '''
        Rotate a point (without translation)
        '''
        return np.dot(r, self.rot.T)

    def inv(self):
        '''
        Inverse of self
        '''
        inv_rot = np.linalg.inv(self.rot)
        trans = -np.dot(self.trans, inv_rot.T)
        return SPGElement(inv_rot, trans)

    def transform(self, a, b, allow_non_integer=False):
        r'''
        Transform from :math:`\mathbf{a}` basis system to :math:`\mathbf{b}` basis system.
        '''
        rot = transform_rot(self.rot, a, b, allow_non_integer)
        trans = transform_trans(self.trans, a, b)
        return SPGElement(rot, trans)

    @property
    def rot_is_eye(self):
        return ((self.rot - np.eye(self.dimension,dtype=int)) == 0).all()

    @property
    def rot_is_inversion(self):
        return ((self.rot + np.eye(self.dimension,dtype=int)) == 0).all()

    @property
    def trans_is_zero(self):
        return (np.abs(self.trans) < SYMPREC).all()

    @property
    def is_eye(self):
        '''
        Whether self is identity operation.
        '''
        return self.rot_is_eye and self.trans_is_zero

    @property
    def is_inversion(self):
        '''
        Whether self is inversion operation.
        '''
        return self.rot_is_inversion and self.trans_is_zero

    def __eq__(self, other):
        if not isinstance(other, SPGElement):
            raise TypeError
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        if not isinstance(other, SPGElement):
            raise TypeError
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, SPGElement):
            raise TypeError
        return self.__hash__() < other.__hash__()

    def __le__(self, other):
        if not isinstance(other, SPGElement):
            raise TypeError
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other):
        if not isinstance(other, SPGElement):
            raise TypeError
        return not self.__le__(other)

    def __ge__(self, other):
        if not isinstance(other, SPGElement):
            raise TypeError
        return not self.__lt__(other)

    def __hash__(self):
        r = PGElement.__hash__(self)

        trans = self.trans
        t = 0
        for i in range(self.dimension):
            t += int(np.round(trans[i] * 12.)) * 12**(2-i)

        return int(t * (3**(self.dimension ** 2)) + r)

    def a2b(self, cell):
        '''
        Transform from direct lattice system to reciprocal lattice system.
        '''
        return self.transform(cell.lattice_vectors(), cell.reciprocal_vectors())

    def a2r(self, cell):
        '''
        Transform from direct lattice system to Cartesian coordinate system.
        '''
        return self.transform(cell.lattice_vectors(), XYZ, True)

    def b2a(self, cell):
        '''
        Transform from reciprocal lattice system to direct lattice system.
        '''
        return self.transform(cell.reciprocal_vectors(), cell.lattice_vectors())

    def b2r(self, cell):
        '''
        Transform from reciprocal lattice system to Cartesian coordinate system.
        '''
        return self.transform(cell.reciprocal_vectors(), XYZ)

    def r2a(self, cell):
        '''
        Transform from Cartesian coordinate system to direct lattice system.
        '''
        return self.transform(XYZ, cell.lattice_vectors())

    def r2b(self, cell):
        '''
        Transform from Cartesian coordinate system to reciprocal lattice system.
        '''
        return self.transform(XYZ, cell.reciprocal_vectors())

    def __str__(self):
        s = ''
        for x in range(3):
            s += '%2d %2d %2d %10.6f\n' % (self.rot[x][0], self.rot[x][1], self.rot[x][2], self.trans[x])
        return s


class SpaceGroup(lib.StreamObject):
    '''
    Determines the space group of a lattice.

    Attributes:
        cell : :class:`Cell` object

        symprec : float
            Numerical tolerance for determining the space group.
            Default value is 1e-6 in the unit of length.
        verbose : int
            Print level. Default value equals to `cell.verbose`.
        backend: str
            Choose which backend to use for symmetry detection.
            Default is `pyscf` and other choices are `spglib`.
        ops : list of :class:`SPGElement` objects
            Matrix representation of the space group operations (in direct lattice system).
        nop : int
            Order of the space group.
        groupname : dict
            Standard symbols for symmetry groups.
            groupname['point_group_symbol']: point group symbol
            groupname['international_symbol']: space group symbol
            groupname['international_number']: space group number
    '''
    def __init__(self, cell, symprec=SYMPREC):
        self.cell = cell
        self.symprec = symprec
        self.verbose = cell.verbose
        self.stdout = cell.stdout
        self.backend = 'pyscf'

        # Followings are not input variables
        self.ops = []
        self.nop = 0
        self.groupname = {}

    def build(self, dump_info=True):
        if self.cell.dimension < 3 and self.backend == 'spglib':
            logger.warn(self, 'spglib only works for 3D system; '
                        'setting symmetry detection backend to pyscf native implementation.')
            self.backend = 'pyscf'

        if self.backend == 'spglib':
            from pyscf.pbc.symm.pyscf_spglib import cell_to_spgcell, get_symmetry_dataset, get_symmetry
            spgcell = cell_to_spgcell(self.cell)
            dataset = get_symmetry_dataset(spgcell, symprec=self.symprec)
            spg_symbol = dataset['international']
            spg_no = dataset['number']
            pg_symbol = dataset['pointgroup']
            symmetry = get_symmetry(spgcell, symprec=self.symprec)
            for rot, trans in zip(symmetry['rotations'], symmetry['translations']):
                self.ops.append(SPGElement(rot, trans))
        elif self.backend == 'pyscf':
            self.ops = geom.search_space_group_ops(self.cell, tol=self.symprec)
            pg_symbol = geom.get_crystal_class(self.cell, tol=self.symprec)[0]
            #TODO add space group symbol
            spg_symbol = None
            spg_no = None

        self.ops.sort()
        self.nop = len(self.ops)
        self.groupname['point_group_symbol'] = pg_symbol
        self.groupname['international_symbol'] = spg_symbol
        self.groupname['international_number'] = spg_no

        if dump_info:
            self.dump_info()
        return self

    def dump_info(self, ops=None):
        if ops is None:
            ops = self.ops
        if self.verbose >= logger.INFO:
            gn = self.groupname
            if gn['international_symbol'] is not None:
                logger.info(self, '[Cell] International symbol:  %s (%d)',
                            gn['international_symbol'], gn['international_number'])
            logger.info(self, '[Cell] Point group symbol:  %s', gn['point_group_symbol'])
        if self.verbose >= logger.DEBUG:
            if len(ops) < len(self.ops):
                message = 'Subset of space group symmetry operations:'
            else:
                message = 'Space group symmetry operations:'
            logger.debug(self, message)
            for op in ops:
                logger.debug(self, op.__str__())

if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = """
      Cu 1.000000     1.00000      0.0000
      O  0.000000     1.00000      0.0000
      O  1.000000     2.00000      0.0000
      Cu 1.000000     3.00000      0.0000
      O  1.000000     4.00000      0.0000
      O  2.000000     3.00000      0.0000
      Cu 3.000000     3.00000      0.0000
      O  4.000000     3.00000      0.0000
      O  3.000000     2.00000      0.0000
      Cu 3.000000     1.00000      0.0000
      O  3.000000     0.00000      0.0000
      O  2.000000     1.00000      0.0000
    """
    cell.a = [[4.0, 0., 0.], [0., 4.0, 0.], [0., 0., 16.0]]
    cell.verbose = 5
    cell.dimension = 3
    cell.magmom = [1., 1., -1., -1., 1., -1., 1., 1., -1., -1., 1., -1.]
    cell.build()
    sg = SpaceGroup(cell)
    #sg.backend = 'spglib'
    sg.build()
    print(sg.groupname['point_group_symbol'])
    print(sg.ops[0])
    print(hash(sg.ops[0]))
    sg.ops.reverse()
    print(sg.ops[0])
    sg.ops.sort()
    print(sg.ops[0])
