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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
#
# The symmetry detection method implemented here is not strictly follow the
# point group detection flowchart.  The detection is based on the degeneracy
# of cartesian basis of multipole momentum, eg
# http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=604&option=4
# see the column of "linear functions, quadratic functions and cubic functions".
#
# Different point groups have different combinations of degeneracy for each
# type of cartesian functions.  Based on the degeneracy of cartesian function
# basis, one can quickly filter out a few candidates of point groups for the
# given molecule.  Regular operations (rotation, mirror etc) can be applied
# then to identify the symmetry.  Current implementation only checks the
# rotation functions and it's roughly enough for D2h and subgroups.
#
# There are special cases this detection method may break down, eg two H8 cube
# molecules sitting on the same center but with random orientation.  The
# system is in C1 while this detection method gives O group because the
# 3 rotation bases are degenerated.  In this case, the code use the regular
# method (point group detection flowchart) to detect the point group.
#

'''
References:

[1] SOFI. M. Gunde, et. al. arXiv:2408.06131.

[2] libmsym. M. Johansson and V. Veryazov, J. Cheminformatics 9, 8 (2017).

[3] SymMol. T. Pilati and A. Forni, J. Appl. Crystallogr. 31, 503-504 (1998).

[4] R. J. Largent, W. F. Polik, and J. R. Schmidt, J. Comput. Chem. 33, 1637-1642 (2012),
'''

import re
import numpy
import scipy.linalg
from pyscf.gto import mole
from pyscf.lib import norm
from pyscf.lib import logger
from pyscf.lib.exceptions import PointGroupSymmetryError
from pyscf.symm.param import OPERATOR_TABLE
from pyscf import __config__

TOLERANCE = getattr(__config__, 'symm_geom_tol', 1e-5)


def parallel_vectors(v1, v2, tol=TOLERANCE):
    if numpy.allclose(v1, 0, atol=tol) or numpy.allclose(v2, 0, atol=tol):
        return True
    else:
        cos = numpy.dot(_normalize(v1), _normalize(v2))
        return (abs(cos-1) < TOLERANCE) | (abs(cos+1) < TOLERANCE)

def argsort_coords(coords, decimals=None, tol=0.05):
    # * np.round for decimal places can lead to more errors than the actual
    # difference between two numbers. For example,
    # np.round([0.1249999999,0.1250000001], 2) => [0.12, 0.13]
    # np.round([0.1249999999,0.1250000001], 3) => [0.125, 0.125]
    # When loosen tolerance is used, compared to the more strict tolerance,
    # the coordinates might look more different.
    # * Using the power of two as the factor can reduce such errors, although not
    # faithfully rounding to the required decimals.
    # * For normal molecules, tol~=0.1 in coordinates is enough to distinguish
    # atoms in molecule. A very tight threshold is not appropriate here. With
    # tight threshold, small differences in coordinates may lead to different
    # arg orders.
    if decimals is None:
        fac = 2**int(-numpy.log2(tol))
    else:
        fac = 2**int(3.3219281 * decimals)
    # +.5 for rounding to the nearest integer
    coords = (coords*fac + .5).astype(int)
    idx = numpy.lexsort((coords[:,2], coords[:,1], coords[:,0]))
    return idx

def sort_coords(coords, decimals=None, tol=0.05):
    coords = numpy.asarray(coords)
    idx = argsort_coords(coords, tol=tol)
    return coords[idx]

# ref. http://en.wikipedia.org/wiki/Rotation_matrix
def rotation_mat(vec, theta):
    '''rotate angle theta along vec
    new(x,y,z) = R * old(x,y,z)'''
    vec = _normalize(vec)
    uu = vec.reshape(-1,1) * vec.reshape(1,-1)
    ux = numpy.array((
        ( 0     ,-vec[2], vec[1]),
        ( vec[2], 0     ,-vec[0]),
        (-vec[1], vec[0], 0     )))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    r = c * numpy.eye(3) + s * ux + (1-c) * uu
    return r

# reflection operation with householder
def householder(vec):
    vec = _normalize(vec)
    return numpy.eye(3) - vec[:,None]*vec*2

def closest_axes(axes, ref):
    xcomp, ycomp, zcomp = numpy.einsum('ix,jx->ji', axes, ref)
    zmax = numpy.amax(abs(zcomp))
    zmax_idx = numpy.where(abs(abs(zcomp)-zmax)<TOLERANCE)[0]
    z_id = numpy.amax(zmax_idx)
    #z_id = numpy.argmax(abs(zcomp))
    xcomp[z_id] = ycomp[z_id] = 0       # remove z
    xmax = numpy.amax(abs(xcomp))
    xmax_idx = numpy.where(abs(abs(xcomp)-xmax)<TOLERANCE)[0]
    x_id = numpy.amax(xmax_idx)
    #x_id = numpy.argmax(abs(xcomp))
    ycomp[x_id] = 0                     # remove x
    y_id = numpy.argmax(abs(ycomp))
    return x_id, y_id, z_id

def alias_axes(axes, ref):
    '''Rename axes, make it as close as possible to the ref axes
    '''
    x_id, y_id, z_id = closest_axes(axes, ref)
    new_axes = axes[[x_id,y_id,z_id]]
    if numpy.linalg.det(new_axes) < 0:
        new_axes = axes[[y_id,x_id,z_id]]
    return new_axes

def _adjust_planar_c2v(atom_coords, axes):
    '''Adjust axes for planar molecules'''
    # Following http://iopenshell.usc.edu/resources/howto/symmetry/
    # See also discussions in issue #1201
    # * planar C2v molecules should be oriented such that the X axis is perpendicular
    # to the plane of the molecule, and the Z axis is the axis of symmetry;
    natm = len(atom_coords)
    tol = TOLERANCE / numpy.sqrt(1+natm)
    atoms_on_xz = abs(atom_coords.dot(axes[1])) < tol
    if all(atoms_on_xz):
        # rotate xy
        axes = numpy.array([-axes[1], axes[0], axes[2]])
    return axes

def _adjust_planar_d2h(atom_coords, axes):
    '''Adjust axes for planar molecules'''
    # Following http://iopenshell.usc.edu/resources/howto/symmetry/
    # See also discussions in issue #1201
    # * planar D2h molecules should be oriented such that the X axis is
    # perpendicular to the plane of the molecule, and the Z axis passes through
    # the greatest number of atoms.
    natm = len(atom_coords)
    tol = TOLERANCE / numpy.sqrt(1+natm)
    natm_with_x = numpy.count_nonzero(abs(atom_coords.dot(axes[0])) > tol)
    natm_with_y = numpy.count_nonzero(abs(atom_coords.dot(axes[1])) > tol)
    natm_with_z = numpy.count_nonzero(abs(atom_coords.dot(axes[2])) > tol)
    if natm_with_z == 0:  # atoms on xy plane
        if natm_with_x >= natm_with_y:  # atoms-on-y >= atoms-on-x
            # rotate xz
            axes = numpy.array([-axes[2], axes[1], axes[0]])
        else:
            # rotate xy then rotate xz
            axes = numpy.array([axes[2], axes[0], axes[1]])
    elif natm_with_y == 0:  # atoms on xz plane
        if natm_with_x >= natm_with_z:  # atoms-on-z >= atoms-on-x
            # rotate xy
            axes = numpy.array([-axes[1], axes[0], axes[2]])
        else:
            # rotate xz then rotate xy
            axes = numpy.array([axes[1], axes[2], axes[0]])
    elif natm_with_x == 0:  # atoms on yz plane
        if natm_with_y < natm_with_z:  # atoms-on-z < atoms-on-y
            # rotate yz
            axes = numpy.array([axes[0], -axes[2], axes[1]])
    return axes

def detect_symm(atoms, basis=None, verbose=logger.WARN):
    '''Detect the point group symmetry for given molecule.

    Return group name, charge center, and nex_axis (three rows for x,y,z)
    '''
    log = logger.new_logger(verbose=verbose)

    tol = TOLERANCE / numpy.sqrt(1+len(atoms))
    decimals = int(-numpy.log10(tol))
    log.debug('geometry tol = %g', tol)

    rawsys = SymmSys(atoms, basis)
    w1, u1 = rawsys.cartesian_tensor(1)
    axes = u1.T
    log.debug('principal inertia moments %s', w1)
    charge_center = rawsys.charge_center

    if numpy.allclose(w1, 0, atol=tol):
        gpname = 'SO3'
        return gpname, charge_center, numpy.eye(3)

    elif numpy.allclose(w1[:2], 0, atol=tol): # linear molecule
        if rawsys.has_icenter():
            gpname = 'Dooh'
        else:
            gpname = 'Coov'
        return gpname, charge_center, axes

    else:
        w1_degeneracy, w1_degen_values = _degeneracy(w1, decimals)
        w2, u2 = rawsys.cartesian_tensor(2)
        w2_degeneracy, w2_degen_values = _degeneracy(w2, decimals)
        log.debug('2d tensor %s', w2)

        n = None
        c2x = None
        mirrorx = None
        if 3 in w1_degeneracy: # T, O, I
            # Because rotation vectors Rx Ry Rz are 3-degenerated representation
            # See http://www.webqc.org/symmetrypointgroup-td.html

            w3, u3 = rawsys.cartesian_tensor(3)
            w3_degeneracy, w3_degen_values = _degeneracy(w3, decimals)
            log.debug('3d tensor %s', w3)
            if (5 in w2_degeneracy and
                4 in w3_degeneracy and len(w3_degeneracy) == 3):  # I group
                gpname, new_axes = _search_i_group(rawsys)
                if gpname is not None:
                    return gpname, charge_center, _refine(new_axes)

            elif 3 in w2_degeneracy and len(w2_degeneracy) <= 3:  # T/O group
                gpname, new_axes = _search_ot_group(rawsys)
                if gpname is not None:
                    return gpname, charge_center, _refine(new_axes)

        elif (2 in w1_degeneracy and
              numpy.any(w2_degeneracy[w2_degen_values>0] >= 2)):
            if numpy.allclose(w1[1], w1[2], atol=tol):
                axes = axes[[1,2,0]]
            n = rawsys.search_c_highest(axes[2])[1]
            if n == 1:
                n = None
            else:
                c2x = rawsys.search_c2x(axes[2], n)
                mirrorx = rawsys.search_mirrorx(axes[2], n)

        else:
            n = -1  # tag as D2h and subgroup

# They must not be I/O/T group, at most one C3 or higher rotation axis
        if n is None:
            zaxis, n = rawsys.search_c_highest()
            if n > 1:
                c2x = rawsys.search_c2x(zaxis, n)
                mirrorx = rawsys.search_mirrorx(zaxis, n)
                if c2x is not None:
                    axes = _make_axes(zaxis, c2x)
                elif mirrorx is not None:
                    axes = _make_axes(zaxis, mirrorx)
                else:
                    for axis in numpy.eye(3):
                        if not parallel_vectors(axis, zaxis):
                            axes = _make_axes(zaxis, axis)
                            break
            else:  # Ci or Cs or C1 with degenerated w1
                mirror = rawsys.search_mirrorx(None, 1)
                if mirror is not None:
                    xaxis = numpy.array((1.,0.,0.))
                    axes = _make_axes(mirror, xaxis)
                else:
                    axes = numpy.eye(3)

        log.debug('Highest C_n = C%d', n)
        if n >= 2:
            if c2x is not None:
                if rawsys.has_mirror(axes[2]):
                    gpname = 'D%dh' % n
                elif rawsys.has_improper_rotation(axes[2], n):
                    gpname = 'D%dd' % n
                else:
                    gpname = 'D%d' % n
                # yaxis = numpy.cross(axes[2], c2x)
                axes = _make_axes(axes[2], c2x)
            elif mirrorx is not None:
                gpname = 'C%dv' % n
                axes = _make_axes(axes[2], mirrorx)
            elif rawsys.has_mirror(axes[2]):
                gpname = 'C%dh' % n
            elif rawsys.has_improper_rotation(axes[2], n):
                gpname = 'S%d' % (n*2)
            else:
                gpname = 'C%d' % n
            return gpname, charge_center, _refine(axes)

        else:
            is_c2x = rawsys.has_rotation(axes[0], 2)
            is_c2y = rawsys.has_rotation(axes[1], 2)
            is_c2z = rawsys.has_rotation(axes[2], 2)
# rotate to old axes, as close as possible?
            if is_c2z and is_c2x and is_c2y:
                if rawsys.has_icenter():
                    gpname = 'D2h'
                    # _adjust_planar_d2h is unlikely to be called
                    axes = _adjust_planar_d2h(rawsys.atom_coords, axes)
                else:
                    gpname = 'D2'
                axes = alias_axes(axes, numpy.eye(3))
            elif is_c2z or is_c2x or is_c2y:
                if is_c2x:
                    axes = axes[[1,2,0]]
                if is_c2y:
                    axes = axes[[2,0,1]]
                if rawsys.has_mirror(axes[2]):
                    gpname = 'C2h'
                elif rawsys.has_mirror(axes[0]):
                    gpname = 'C2v'
                    axes = _adjust_planar_c2v(rawsys.atom_coords, axes)
                else:
                    gpname = 'C2'
            else:
                if rawsys.has_icenter():
                    gpname = 'Ci'
                elif rawsys.has_mirror(axes[0]):
                    gpname = 'Cs'
                    axes = axes[[1,2,0]]
                elif rawsys.has_mirror(axes[1]):
                    gpname = 'Cs'
                    axes = axes[[2,0,1]]
                elif rawsys.has_mirror(axes[2]):
                    gpname = 'Cs'
                else:
                    gpname = 'C1'
                    axes = numpy.eye(3)
                    charge_center = numpy.zeros(3)
    return gpname, charge_center, axes


# reduce to D2h and its subgroups
# FIXME, CPL, 209, 506
def get_subgroup(gpname, axes):
    if gpname in ('D2h', 'D2' , 'C2h', 'C2v', 'C2' , 'Ci' , 'Cs' , 'C1'):
        return gpname, axes
    elif gpname in ('SO3',):
        #return 'D2h', alias_axes(axes, numpy.eye(3))
        return 'SO3', axes
    elif gpname in ('Dooh',):
        #return 'D2h', alias_axes(axes, numpy.eye(3))
        return 'Dooh', axes
    elif gpname in ('Coov',):
        #return 'C2v', axes
        return 'Coov', axes
    elif gpname in ('Oh',):
        return 'D2h', alias_axes(axes, numpy.eye(3))
    elif gpname in ('O',):
        return 'D2', alias_axes(axes, numpy.eye(3))
    elif gpname in ('Ih',):
        return 'Ci', alias_axes(axes, numpy.eye(3))
    elif gpname in ('I',):
        return 'C1', axes
    elif gpname in ('Td', 'T', 'Th'):
        return 'D2', alias_axes(axes, numpy.eye(3))
    elif re.search(r'S\d+', gpname):
        n = int(re.search(r'\d+', gpname).group(0))
        if n % 2 == 0:
            return 'C%d'%(n//2), axes
        else:
            return 'Ci', axes
    else:
        n = int(re.search(r'\d+', gpname).group(0))
        if n % 2 == 0:
            if re.search(r'D\d+d', gpname):
                subname = 'D2'
            elif re.search(r'D\d+h', gpname):
                subname = 'D2h'
            elif re.search(r'D\d+', gpname):
                subname = 'D2'
            elif re.search(r'C\d+h', gpname):
                subname = 'C2h'
            elif re.search(r'C\d+v', gpname):
                subname = 'C2v'
            else:
                subname = 'C2'
        else:
            # rotate axes and
            # Dnh -> C2v
            # Dn  -> C2
            # Dnd -> Ci
            # Cnh -> Cs
            # Cnv -> Cs
            if re.search(r'D\d+h', gpname):
                subname = 'C2v'
                axes = axes[[1,2,0]]
            elif re.search(r'D\d+d', gpname):
                subname = 'C2h'
                axes = axes[[1,2,0]]
            elif re.search(r'D\d+', gpname):
                subname = 'C2'
                axes = axes[[1,2,0]]
            elif re.search(r'C\d+h', gpname):
                subname = 'Cs'
            elif re.search(r'C\d+v', gpname):
                subname = 'Cs'
                axes = axes[[1,2,0]]
            else:
                subname = 'C1'
        return subname, axes
subgroup = get_subgroup

def as_subgroup(topgroup, axes, subgroup=None):
    from pyscf.symm import std_symb
    from pyscf.symm.param import SUBGROUP

    groupname, axes = get_subgroup(topgroup, axes)

    if isinstance(subgroup, str):
        subgroup = std_symb(subgroup)
        if groupname == 'C2v' and subgroup == 'Cs':
            axes = numpy.einsum('ij,kj->ki', rotation_mat(axes[1], numpy.pi/2), axes)

        elif (groupname == 'D2' and re.search(r'D\d+d', topgroup) and
              subgroup in ('C2v', 'Cs')):
            # Special treatment for D2d, D4d, .... get_subgroup gives D2 by
            # default while C2v is also D2d's subgroup.
            groupname = 'C2v'
            axes = numpy.einsum('ij,kj->ki', rotation_mat(axes[2], numpy.pi/4), axes)

        elif topgroup in ('Td', 'T', 'Th') and subgroup == 'C2v':
            x, y, z = axes
            x = _normalize(x+y)
            y = numpy.cross(z, x)
            axes = numpy.array((x,y,z))

        elif subgroup not in SUBGROUP[groupname]:
            raise PointGroupSymmetryError('%s not in Ablien subgroup of %s' %
                                          (subgroup, topgroup))

        groupname = subgroup
    return groupname, axes

def symm_ops(gpname, axes=None):
    if axes is not None:
        raise PointGroupSymmetryError('TODO: non-standard orientation')
    op1 = numpy.eye(3)
    opi = -1

    opc2z = -numpy.eye(3)
    opc2z[2,2] = 1
    opc2x = -numpy.eye(3)
    opc2x[0,0] = 1
    opc2y = -numpy.eye(3)
    opc2y[1,1] = 1

    opcsz = numpy.dot(opc2z, opi)
    opcsx = numpy.dot(opc2x, opi)
    opcsy = numpy.dot(opc2y, opi)
    opdic = {'E'  : op1,
             'C2z': opc2z,
             'C2x': opc2x,
             'C2y': opc2y,
             'i'  : opi,
             'sz' : opcsz,  # the mirror perpendicular to z
             'sx' : opcsx,  # the mirror perpendicular to x
             'sy' : opcsy,}
    return opdic

def symm_identical_atoms(gpname, atoms):
    '''Symmetry identical atoms'''
    # from pyscf import gto
    # Dooh Coov for linear molecule
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        idx0 = argsort_coords(coords)
        coords0 = coords[idx0]
        opdic = symm_ops(gpname)
        newc = numpy.dot(coords, opdic['sz'])
        idx1 = argsort_coords(newc)
        dup_atom_ids = numpy.sort((idx0,idx1), axis=0).T
        uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
        eql_atom_ids = dup_atom_ids[uniq_idx]
        eql_atom_ids = [sorted(set(i)) for i in eql_atom_ids]
        return eql_atom_ids
    elif gpname == 'Coov':
        eql_atom_ids = [[i] for i,a in enumerate(atoms)]
        return eql_atom_ids

    coords = numpy.array([a[1] for a in atoms])

#    charges = numpy.array([gto.charge(a[0]) for a in atoms])
#    center = numpy.einsum('z,zr->r', charges, coords)/charges.sum()
#    if not numpy.allclose(center, 0, atol=TOLERANCE):
#        sys.stderr.write('WARN: Molecular charge center %s is not on (0,0,0)\n'
#                        % center)
    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in OPERATOR_TABLE[gpname]]
    idx = argsort_coords(coords)
    coords0 = coords[idx]

    dup_atom_ids = []
    for op in ops:
        newc = numpy.dot(coords, op)
        idx = argsort_coords(newc)
        if not numpy.allclose(coords0, newc[idx], atol=TOLERANCE):
            raise PointGroupSymmetryError(
                'Symmetry identical atoms not found. This may be due to '
                'the strict setting of the threshold symm.geom.TOLERANCE. '
                'Consider adjusting the tolerance.')

        dup_atom_ids.append(idx)

    dup_atom_ids = numpy.sort(dup_atom_ids, axis=0).T
    uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
    eql_atom_ids = dup_atom_ids[uniq_idx]
    eql_atom_ids = [sorted(set(i)) for i in eql_atom_ids]
    return eql_atom_ids

def check_symm(gpname, atoms, basis=None):
    '''
    Check whether the declared symmetry (gpname) exists in the system

    If basis is specified, this function checks also the basis functions have
    the required symmetry.

    Args:
        gpname: str
            point group name
        atoms: list
            [[symbol, [x, y, z]], [symbol, [x, y, z]], ...]
    '''

    #FIXME: compare the basis set when basis is given
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        if numpy.allclose(coords[:,:2], 0, atol=TOLERANCE):
            opdic = symm_ops(gpname)
            rawsys = SymmSys(atoms, basis)
            return rawsys.has_icenter() and numpy.allclose(rawsys.charge_center, 0)
        else:
            return False
    elif gpname == 'Coov':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        return numpy.allclose(coords[:,:2], 0, atol=TOLERANCE)
    elif gpname == 'SO3':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        return abs(coords).max() < TOLERANCE

    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in OPERATOR_TABLE[gpname]]
    rawsys = SymmSys(atoms, basis)

    # A fast check using Casimir tensors
    coords = rawsys.atoms[:,1:]
    weights = rawsys.atoms[:,0]
    for op in ops:
        if not is_identical_geometry(coords, coords.dot(op), weights):
            return False

    for lst in rawsys.atomtypes.values():
        coords = rawsys.atoms[lst,1:]
        idx = argsort_coords(coords)
        coords0 = coords[idx]

        for op in ops:
            newc = numpy.dot(coords, op)
            idx = argsort_coords(newc)
            if not numpy.allclose(coords0, newc[idx], atol=TOLERANCE):
                return False
    return True

check_given_symm = check_symm

def shift_atom(atoms, orig, axis):
    c = numpy.array([a[1] for a in atoms])
    c = numpy.dot(c - orig, numpy.array(axis).T)
    return [[atoms[i][0], c[i]] for i in range(len(atoms))]

def is_identical_geometry(coords1, coords2, weights):
    '''A fast check to compare the geometry of two molecules using Casimir tensors'''
    if coords1.shape != coords2.shape:
        return False
    for order in range(1, 4):
        if abs(casimir_tensors(coords1, weights, order) -
               casimir_tensors(coords2, weights, order)).max() > TOLERANCE:
            return False
    return True

def casimir_tensors(r, q, order=1):
    if order == 1:
        return q.dot(r)
    elif order == 2:
        return numpy.einsum('i,ix,iy->xy', q, r, r)
    elif order == 3:
        return numpy.einsum('i,ix,iy,iz->xyz', q, r, r, r)
    else:
        raise NotImplementedError


class RotationAxisNotFound(PointGroupSymmetryError):
    pass

class SymmSys:
    def __init__(self, atoms, basis=None):
        self.atomtypes = mole.atom_types(atoms, basis)
        # fake systems, which treats the atoms of different basis as different atoms.
        # the fake systems do not have the same symmetry as the potential
        # it's only used to determine the main (Z-)axis
        chg1 = numpy.pi - 2
        coords = []
        fake_chgs = []
        idx = []
        for k, lst in self.atomtypes.items():
            idx.append(lst)
            coords.append([atoms[i][1] for i in lst])
            ksymb = mole._rm_digit(k)
            if ksymb != k:
                # Put random charges on the decorated atoms
                fake_chgs.append([chg1] * len(lst))
                chg1 *= numpy.pi-2
            elif mole.is_ghost_atom(k):
                if ksymb == 'X' or ksymb.upper() == 'GHOST':
                    fake_chgs.append([.3] * len(lst))
                elif ksymb[0] == 'X':
                    fake_chgs.append([mole.charge(ksymb[1:])+.3] * len(lst))
                elif ksymb[:5] == 'GHOST':
                    fake_chgs.append([mole.charge(ksymb[5:])+.3] * len(lst))
            else:
                fake_chgs.append([mole.charge(ksymb)] * len(lst))
        coords = numpy.array(numpy.vstack(coords), dtype=float)
        fake_chgs = numpy.hstack(fake_chgs)
        self.charge_center = numpy.einsum('i,ij->j', fake_chgs, coords)/fake_chgs.sum()
        coords = coords - self.charge_center

        idx = numpy.argsort(numpy.hstack(idx))
        self.atoms = numpy.hstack((fake_chgs.reshape(-1,1), coords))[idx]

        self.group_atoms_by_distance = []
        decimals = int(-numpy.log10(TOLERANCE)) - 1
        for index in self.atomtypes.values():
            index = numpy.asarray(index)
            c = self.atoms[index,1:]
            dists = numpy.around(norm(c, axis=1), decimals)
            u, idx = numpy.unique(dists, return_inverse=True)
            for i, s in enumerate(u):
                self.group_atoms_by_distance.append(index[idx == i])

    @property
    def atom_coords(self):
        return self.atoms[:,1:]

    def cartesian_tensor(self, n):
        z = self.atoms[:,0]
        r = self.atoms[:,1:]
        ncart = (n+1)*(n+2)//2
        natm = len(z)
        tensor = numpy.sqrt(numpy.copy(z).reshape(natm,-1) / z.sum())
        for i in range(n):
            tensor = numpy.einsum('zi,zj->zij', tensor, r).reshape(natm,-1)
        e, c = scipy.linalg.eigh(numpy.dot(tensor.T,tensor))
        return e[-ncart:], c[:,-ncart:]

    def symmetric_for(self, op):
        for lst in self.group_atoms_by_distance:
            r0 = self.atoms[lst,1:]
            r1 = numpy.dot(r0, op)
            # FIXME: compare whether two sets of coordinates are identical
            yield all((_vec_in_vecs(x, r0) for x in r1))

    def has_icenter(self):
        return all(self.symmetric_for(-1))

    def has_rotation(self, axis, n):
        op = rotation_mat(axis, numpy.pi*2/n).T
        return all(self.symmetric_for(op))

    def has_mirror(self, perp_vec):
        return all(self.symmetric_for(householder(perp_vec).T))

    def has_improper_rotation(self, axis, n):
        s_op = numpy.dot(householder(axis), rotation_mat(axis, numpy.pi/n)).T
        return all(self.symmetric_for(s_op))

    def search_possible_rotations(self, zaxis=None):
        '''If zaxis is given, the rotation axis is parallel to zaxis'''
        maybe_cn = []
        for lst in self.group_atoms_by_distance:
            natm = len(lst)
            if natm > 1:
                coords = self.atoms[lst,1:]
# possible C2 axis
                for i in range(1, natm):
                    if abs(coords[0]+coords[i]).sum() > TOLERANCE:
                        maybe_cn.append((coords[0]+coords[i], 2))
                    else: # abs(coords[0]-coords[i]).sum() > TOLERANCE:
                        maybe_cn.append((coords[0]-coords[i], 2))

# atoms of equal distances may be associated with rotation axis > C2.
                r0 = coords - coords[0]
                distance = norm(r0, axis=1)
                eq_distance = abs(distance[:,None] - distance) < TOLERANCE
                for i in range(2, natm):
                    for j in numpy.where(eq_distance[i,:i])[0]:
                        cos = numpy.dot(r0[i],r0[j]) / (distance[i]*distance[j])
                        ang = numpy.arccos(cos)
                        nfrac = numpy.pi*2 / (numpy.pi-ang)
                        n = int(numpy.around(nfrac))
                        if abs(nfrac-n) < TOLERANCE:
                            maybe_cn.append((numpy.cross(r0[i],r0[j]),n))

        # remove zero-vectors and duplicated vectors
        vecs = numpy.vstack([x[0] for x in maybe_cn])
        idx = norm(vecs, axis=1) > TOLERANCE
        ns = numpy.hstack([x[1] for x in maybe_cn])
        vecs = _normalize(vecs[idx])
        ns = ns[idx]

        if zaxis is not None:  # Keep parallel rotation axes
            cos = numpy.dot(vecs, _normalize(zaxis))
            vecs = vecs[(abs(cos-1) < TOLERANCE) | (abs(cos+1) < TOLERANCE)]
            ns = ns[(abs(cos-1) < TOLERANCE) | (abs(cos+1) < TOLERANCE)]

        possible_cn = []
        seen = numpy.zeros(len(vecs), dtype=bool)
        for k, v in enumerate(vecs):
            if not seen[k]:
                where1 = numpy.einsum('ix->i', abs(vecs[k:] - v)) < TOLERANCE
                where1 = numpy.where(where1)[0] + k
                where2 = numpy.einsum('ix->i', abs(vecs[k:] + v)) < TOLERANCE
                where2 = numpy.where(where2)[0] + k
                seen[where1] = True
                seen[where2] = True

                vk = _normalize((numpy.einsum('ix->x', vecs[where1]) -
                                 numpy.einsum('ix->x', vecs[where2])))
                for n in (set(ns[where1]) | set(ns[where2])):
                    possible_cn.append((vk,n))
        return possible_cn

    def search_c2x(self, zaxis, n):
        '''C2 axis which is perpendicular to z-axis'''
        decimals = int(-numpy.log10(TOLERANCE)) - 1
        for lst in self.group_atoms_by_distance:
            if len(lst) > 1:
                r0 = self.atoms[lst,1:]
                zcos = numpy.around(numpy.einsum('ij,j->i', r0, zaxis),
                                    decimals=decimals)
                uniq_zcos = numpy.unique(zcos)
                maybe_c2x = []
                for d in uniq_zcos:
                    if d > TOLERANCE:
                        mirrord = abs(zcos+d)<TOLERANCE
                        if mirrord.sum() == (zcos==d).sum():
                            above = r0[zcos==d]
                            below = r0[mirrord]
                            nelem = len(below)
                            maybe_c2x.extend([above[0] + below[i]
                                              for i in range(nelem)])
                    elif abs(d) < TOLERANCE: # plane which crosses the orig
                        r1 = r0[zcos==d][0]
                        maybe_c2x.append(r1)
                        r2 = numpy.dot(rotation_mat(zaxis, numpy.pi*2/n), r1)
                        if abs(r1+r2).sum() > TOLERANCE:
                            maybe_c2x.append(r1+r2)
                        else:
                            maybe_c2x.append(r2-r1)

                if len(maybe_c2x) > 0:
                    idx = norm(maybe_c2x, axis=1) > TOLERANCE
                    maybe_c2x = _normalize(maybe_c2x)[idx]
                    maybe_c2x = _remove_dupvec(maybe_c2x)
                    for c2x in maybe_c2x:
                        if (not parallel_vectors(c2x, zaxis) and
                            self.has_rotation(c2x, 2)):
                            return c2x

    def search_mirrorx(self, zaxis, n):
        '''mirror which is parallel to z-axis'''
        if n > 1:
            for lst in self.group_atoms_by_distance:
                natm = len(lst)
                r0 = self.atoms[lst[0],1:]
                if natm > 1 and not parallel_vectors(r0, zaxis):
                    r1 = numpy.dot(rotation_mat(zaxis, numpy.pi*2/n), r0)
                    mirrorx = _normalize(r1-r0)
                    if self.has_mirror(mirrorx):
                        return mirrorx
        else:
            for lst in self.group_atoms_by_distance:
                natm = len(lst)
                r0 = self.atoms[lst,1:]
                if natm > 1:
                    maybe_mirror = [r0[i]-r0[0] for i in range(1, natm)]
                    for mirror in _normalize(maybe_mirror):
                        if self.has_mirror(mirror):
                            return mirror

    def search_c_highest(self, zaxis=None):
        possible_cn = self.search_possible_rotations(zaxis)
        nmax = 1
        cmax = numpy.array([0.,0.,1.])
        for cn, n in possible_cn:
            if n > nmax and self.has_rotation(cn, n):
                nmax = n
                cmax = cn
        return cmax, nmax


def _normalize(vecs):
    vecs = numpy.asarray(vecs)
    if vecs.ndim == 1:
        return vecs / (numpy.linalg.norm(vecs) + 1e-200)
    else:
        return vecs / (norm(vecs, axis=1).reshape(-1,1) + 1e-200)

def _vec_in_vecs(vec, vecs):
    norm = numpy.sqrt(len(vecs))
    return min(numpy.einsum('ix->i', abs(vecs-vec))/norm) < TOLERANCE

def _search_i_group(rawsys):
    possible_cn = rawsys.search_possible_rotations()
    c5_axes = [c5 for c5, n in possible_cn
               if n == 5 and rawsys.has_rotation(c5, 5)]
    if len(c5_axes) <= 1:
        return None,None

    zaxis = c5_axes[0]
    cos = numpy.dot(c5_axes, zaxis)
    assert (numpy.all((abs(cos[1:]+1/numpy.sqrt(5)) < TOLERANCE) |
                     (abs(cos[1:]-1/numpy.sqrt(5)) < TOLERANCE)))

    if rawsys.has_icenter():
        gpname = 'Ih'
    else:
        gpname = 'I'

    c5 = c5_axes[1]
    if numpy.dot(c5, zaxis) < 0:
        c5 = -c5
    c5a = numpy.dot(rotation_mat(zaxis, numpy.pi*6/5), c5)
    xaxis = c5a + c5
    return gpname, _make_axes(zaxis, xaxis)

def _search_ot_group(rawsys):
    possible_cn = rawsys.search_possible_rotations()
    c4_axes = [c4 for c4, n in possible_cn
               if n == 4 and rawsys.has_rotation(c4, 4)]

    if len(c4_axes) > 0:  # T group
        assert (len(c4_axes) > 1)
        if rawsys.has_icenter():
            gpname = 'Oh'
        else:
            gpname = 'O'
        return gpname, _make_axes(c4_axes[0], c4_axes[1])

    else:  # T group
        c3_axes = [c3 for c3, n in possible_cn
                   if n == 3 and rawsys.has_rotation(c3, 3)]
        if len(c3_axes) <= 1:
            return None, None

        cos = numpy.dot(c3_axes, c3_axes[0])
        assert (numpy.all((abs(cos[1:]+1./3) < TOLERANCE) |
                         (abs(cos[1:]-1./3) < TOLERANCE)))

        if rawsys.has_icenter():
            gpname = 'Th'
# Because C3 axes are on the mirror of Td, two C3 can determine a mirror.
        elif rawsys.has_mirror(numpy.cross(c3_axes[0], c3_axes[1])):
            gpname = 'Td'
        else:
            gpname = 'T'

        c3a = c3_axes[0]
        if numpy.dot(c3a, c3_axes[1]) > 0:
            c3a = -c3a
        c3b = numpy.dot(rotation_mat(c3a,-numpy.pi*2/3), c3_axes[1])
        c3c = numpy.dot(rotation_mat(c3a, numpy.pi*2/3), c3_axes[1])
        zaxis, xaxis = c3a+c3b, c3a+c3c
        return gpname, _make_axes(zaxis, xaxis)

def _degeneracy(e, decimals):
    e = numpy.around(e, decimals)
    u, idx = numpy.unique(e, return_inverse=True)
    degen = numpy.array([numpy.count_nonzero(idx==i) for i in range(len(u))])
    return degen, u

def _pseudo_vectors(vs):
    idy0 = abs(vs[:,1])<TOLERANCE
    idz0 = abs(vs[:,2])<TOLERANCE
    vs = vs.copy()
    # ensure z component > 0
    vs[vs[:,2]<0] *= -1
    # if z component == 0, ensure y component > 0
    vs[(vs[:,1]<0) & idz0] *= -1
    # if y and z component == 0, ensure x component > 0
    vs[(vs[:,0]<0) & idy0 & idz0] *= -1
    return vs

def _remove_dupvec(vs):
    def rm_iter(vs):
        if len(vs) <= 1:
            return vs
        else:
            x = numpy.sum(abs(vs[1:]-vs[0]), axis=1)
            rest = rm_iter(vs[1:][x>TOLERANCE])
            return numpy.vstack((vs[0], rest))
    return rm_iter(_pseudo_vectors(vs))

def _make_axes(z, x):
    y = numpy.cross(z, x)
    x = numpy.cross(y, z)  # because x might not perp to z
    return _normalize(numpy.array((x,y,z)))

def _refine(axes):
    # Make sure the axes can be rotated from continuous unitary transformation
    if axes[2,2] < 0:
        axes[2] *= -1
    if abs(axes[0,0]) > abs(axes[1,0]):
        x_id, y_id = 0, 1
    else:
        x_id, y_id = 1, 0
    if axes[x_id,0] < 0:
        axes[x_id] *= -1
    if numpy.linalg.det(axes) < 0:
        axes[y_id] *= -1
    return axes


if __name__ == "__main__":
    atom = [["O" , (1. , 0.    , 0.   ,)],
            ['H' , (0. , -.757 , 0.587,)],
            ['H' , (0. , 0.757 , 0.587,)] ]
    gpname, orig, axes = detect_symm(atom)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['H', (0,0,1)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, orig, axes)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0., 0., 0.)],
            ['H', (0., 0., 1.)],
            ['H', (0., 1., 0.)],
            ['H', (1., 0., 0.)],
            ['H', (-1, 0., 0.)],
            ['H', (0.,-1., 0.)],
            ['H', (0., 0.,-1.)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, orig, axes)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(subgroup(gpname, axes)[0], atom))
