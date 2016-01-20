#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import re
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.gto import mole
from pyscf.lib import logger
import pyscf.symm.param

GEOM_THRESHOLD = 1e-5
PLACE = int(-numpy.log10(GEOM_THRESHOLD))

def get_charge_center(atoms):
    charge = numpy.array([mole._charge(a[0]) for a in atoms])
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    rbar = numpy.einsum('i,ij->j', charge, coords)/charge.sum()
    return rbar

def get_mass_center(atoms):
    mass = numpy.array([pyscf.lib.parameters.ELEMENTS[mole._charge(a[0])][1]
                        for a in atoms])
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    rbar = numpy.einsum('i,ij->j', mass, coords)/mass.sum()
    return rbar

def get_inertia_momentum(atoms, basis):
    charge = numpy.array([mole._charge(a[0]) for a in atoms])
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    rbar = numpy.einsum('i,ij->j', charge, coords)/charge.sum()
    coords = coords - rbar
    im = numpy.einsum('i,ij,ik->jk', charge, coords, coords)/charge.sum()
    return im

def parallel_vectors(v1, v2, tol=GEOM_THRESHOLD):
    if numpy.allclose(v1, 0, atol=tol) or numpy.allclose(v2, 0, atol=tol):
        return True
    else:
        v3 = numpy.cross(v1/numpy.linalg.norm(v1), v2/numpy.linalg.norm(v2))
        return numpy.linalg.norm(v3) < GEOM_THRESHOLD

def argsort_coords(coords):
    coords = numpy.around(coords, decimals=PLACE-1)
    idx = numpy.lexsort((coords[:,2], coords[:,1], coords[:,0]))
    return idx

def sort_coords(coords):
    coords = numpy.array(coords)
    idx = argsort_coords(coords)
    return coords[idx]

# ref. http://en.wikipedia.org/wiki/Rotation_matrix
def rotation_mat(vec, theta):
    '''rotate angle theta along vec
    new(x,y,z) = R * old(x,y,z)'''
    vec = vec / numpy.linalg.norm(vec)
    uu = vec.reshape(-1,1) * vec.reshape(1,-1)
    ux = numpy.array((
        ( 0     ,-vec[2], vec[1]),
        ( vec[2], 0     ,-vec[0]),
        (-vec[1], vec[0], 0     )))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    r = c * numpy.eye(3) + s * ux + (1-c) * uu
    return r

# refection operation via householder
def householder(vec):
    vec = numpy.array(vec)
    return numpy.eye(3) - vec[:,None]*vec*2

#TODO: Sn, T, Th, O, I
def detect_symm(atoms, basis=None, verbose=logger.WARN):
    '''Detect the point group symmetry for given molecule.

    Return group name, charge center, and nex_axis (three rows for x,y,z)
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)
# a tight threshold for classifying the main class of group.  Because if the
# main group class is incorrectly assigned, the following search _search_toi
# and search_c_highest is very likely to give wrong type of symmetry
    tol = GEOM_THRESHOLD / len(atoms)
    log.debug('geometry tol = %g', tol)

    rawsys = SymmSys(atoms, basis)
    w, axes = scipy.linalg.eigh(rawsys.im)
    axes = axes.T
    log.debug('principal inertia moments %s', str(w))

    if numpy.allclose(w, 0, atol=tol):
        gpname = 'SO3'
    elif numpy.allclose(w[:2], 0, atol=tol): # linear molecule
        if rawsys.detect_icenter():
            gpname = 'Dooh'
        else:
            gpname = 'Coov'
    elif numpy.allclose(w, w[0], atol=tol): # T, O, I
        gpname, axes = _search_toi(rawsys)
    elif (numpy.allclose(w[0], w[1], atol=tol) or
          numpy.allclose(w[1], w[2], atol=tol)):
        if numpy.allclose(w[1], w[2], atol=tol):
            axes = axes[[1,2,0]]
        n, c2x, mirrorx = rawsys.search_c_highest(axes[2])
        if c2x is not None:
            if rawsys.iden_op(householder(axes[2])):
                gpname = 'D%dh' % n
            elif rawsys.detect_icenter():
                gpname = 'D%dd' % n
            else:
                gpname = 'D%d' % n
            yaxis = numpy.cross(axes[2], c2x)
            axes = numpy.array((c2x, yaxis, axes[2]))
        elif mirrorx is not None:
            gpname = 'C%dv' % n
            yaxis = numpy.cross(axes[2], mirrorx)
            axes = numpy.array((mirrorx, yaxis, axes[2]))
        elif rawsys.iden_op(householder(axes[2])): # xy-mirror
            gpname = 'C%dh' % n
        elif rawsys.iden_op(-rotation_mat(axes[2], numpy.pi/n)): # rotate and inverse
            gpname = 'S%d' % (n*2)
        else:
            gpname = 'C%d' % n
    else:
        is_c2x = rawsys.iden_op(rotation_mat(axes[0], numpy.pi))
        is_c2y = rawsys.iden_op(rotation_mat(axes[1], numpy.pi))
        is_c2z = rawsys.iden_op(rotation_mat(axes[2], numpy.pi))
# rotate to old axes, as close as possible?
        if is_c2z and is_c2x and is_c2y:
            if rawsys.detect_icenter():
                gpname = 'D2h'
            else:
                gpname = 'D2'
        elif is_c2z or is_c2x or is_c2y:
            if is_c2x:
                axes = axes[[1,2,0]]
            if is_c2y:
                axes = axes[[2,0,1]]
            if rawsys.iden_op(householder(axes[2])):
                gpname = 'C2h'
            elif rawsys.iden_op(householder(axes[0])):
                gpname = 'C2v'
            else:
                gpname = 'C2'
        else:
            if rawsys.detect_icenter():
                gpname = 'Ci'
            elif rawsys.iden_op(householder(axes[0])):
                gpname = 'Cs'
                axes = axes[[1,2,0]]
            elif rawsys.iden_op(householder(axes[1])):
                gpname = 'Cs'
                axes = axes[[2,0,1]]
            elif rawsys.iden_op(householder(axes[2])):
                gpname = 'Cs'
            else:
                gpname = 'C1'
    return gpname, rawsys.charge_center, _pesudo_vectors(axes)


# reduce to D2h and its subgroups
# FIXME, CPL, 209, 506
def subgroup(gpname, axes):
    if gpname in ('D2h', 'D2' , 'C2h', 'C2v', 'C2' , 'Ci' , 'Cs' , 'C1'):
        return gpname, axes
    elif gpname in ('SO3',):
        #return 'D2h', axes
        return 'Dooh', axes
    elif gpname in ('Dooh',):
        #return 'D2h', axes
        return 'Dooh', axes
    elif gpname in ('Coov',):
        #return 'C2v', axes
        return 'Coov', axes
    elif gpname in ('Oh',):
        return 'D2h', axes
    elif gpname in ('Ih',):
        return 'Cs', axes[[2,0,1]]
    elif gpname in ('Td',):
        x,y,z = axes
        x = (x+y) / numpy.linalg.norm(x+y)
        y = numpy.cross(z, x)
        return 'C2v', numpy.array((x,y,z))
    elif re.search(r'S\d+', gpname):
        n = int(re.search(r'\d+', gpname).group(0))
        return 'C%d'%(n//2), axes
    else:
        n = int(re.search(r'\d+', gpname).group(0))
        if re.search(r'D\d+d', gpname):
            gpname = 'C%dv' % n
        if n % 2 == 0:
            subname = re.sub(r'\d+', '2', gpname)
        else:
            # Dnh -> C2v
            # Dn  -> C2
            # Cnh -> Cs
            # Cnv -> Cs
            if re.search(r'D\d+h', gpname):
                subname = 'C2v'
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
        if subname[-1] == 'd':
            subname = subname[:-1]
        return subname, axes


def symm_ops(gpname, axes=None):
    if axes is not None:
        raise RuntimeError('TODO: non-standard orientation')
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
             'sz' : opcsz,
             'sx' : opcsx,
             'sy' : opcsy,}
    return opdic

def symm_identical_atoms(gpname, atoms):
    ''' Requires '''
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
        eql_atom_ids = [list(sorted(set(i))) for i in eql_atom_ids]
        return eql_atom_ids
    elif gpname == 'Coov':
        eql_atom_ids = [[i] for i,a in enumerate(atoms)]
        return eql_atom_ids

    center = get_charge_center(atoms)
    if not numpy.allclose(center, 0, atol=GEOM_THRESHOLD):
        sys.stderr.write('WARN: Molecular charge center %s is not on (0,0,0)\n'
                        % str(center))
    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    coords = numpy.array([a[1] for a in atoms], dtype=float)
    idx = argsort_coords(coords)
    coords0 = coords[idx]

    dup_atom_ids = []
    for op in ops:
        newc = numpy.dot(coords, op)
        idx = argsort_coords(newc)
        if not numpy.allclose(coords0, newc[idx], atol=GEOM_THRESHOLD):
            raise RuntimeError('Symmetry identical atoms not found')
        dup_atom_ids.append(idx)

    dup_atom_ids = numpy.sort(dup_atom_ids, axis=0).T
    uniq_idx = numpy.unique(dup_atom_ids[:,0], return_index=True)[1]
    eql_atom_ids = dup_atom_ids[uniq_idx]
    eql_atom_ids = [list(sorted(set(i))) for i in eql_atom_ids]
    return eql_atom_ids

def check_given_symm(gpname, atoms, basis=None):
# more strict than symm_identical_atoms, we required not only the coordinates
# match, but also the symbols and basis functions

#FIXME: compare the basis set when basis is given
    if gpname == 'Dooh':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        if numpy.allclose(coords[:,:2], 0, atol=GEOM_THRESHOLD):
            opdic = symm_ops(gpname)
            rawsys = SymmSys(atoms, basis)
            return rawsys.detect_icenter()
        else:
            return False
    elif gpname == 'Coov':
        coords = numpy.array([a[1] for a in atoms], dtype=float)
        return numpy.allclose(coords[:,:2], 0, atol=GEOM_THRESHOLD)

    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    rawsys = SymmSys(atoms, basis)
    for lst in rawsys.atomtypes.values():
        coords = rawsys.atoms[lst,1:]
        idx = argsort_coords(coords)
        coords0 = coords[idx]

        for op in ops:
            newc = numpy.dot(coords, op)
            idx = argsort_coords(newc)
            if not numpy.allclose(coords0, newc[idx], atol=GEOM_THRESHOLD):
                return False
    return True

def shift_atom(atoms, orig, axis):
    c = numpy.array([a[1] for a in atoms])
    c = numpy.dot(c - orig, numpy.array(axis).T)
    return [[atoms[i][0], c[i]] for i in range(len(atoms))]


class SymmSys(object):
    def __init__(self, atoms, basis=None):
        self.atomtypes = mole.atom_types(atoms, basis)
        # fake systems, which treates the atoms of different basis as different atoms.
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
            if ksymb != k or ksymb == 'GHOST':
                fake_chgs.append([chg1] * len(lst))
                chg1 *= numpy.pi-2
            else:
                fake_chgs.append([mole._charge(ksymb)] * len(lst))
        coords = numpy.array(numpy.vstack(coords), dtype=float)
        fake_chgs = numpy.hstack(fake_chgs)
        self.charge_center = numpy.einsum('i,ij->j', fake_chgs, coords)/fake_chgs.sum()
        coords = coords - self.charge_center
        self.im = numpy.einsum('i,ij,ik->jk', fake_chgs, coords, coords)/fake_chgs.sum()

        idx = numpy.argsort(numpy.hstack(idx))
        self.atoms = numpy.hstack((fake_chgs.reshape(-1,1), coords))[idx]


    def group_atoms_by_distance(self, index):
        c = self.atoms[index,1:]
        r = numpy.sqrt(numpy.einsum('ij,ij->i', c, c))
        lst = numpy.argsort(r)
        groups = [[index[lst[0]]]]
        for i in range(len(lst)-1):
            if numpy.allclose(r[lst[i]], r[lst[i+1]], atol=GEOM_THRESHOLD):
                groups[-1].append(index[lst[i+1]])
            else:
                groups.append([index[lst[i+1]]])
        return groups

    def detect_icenter(self):
        return self.iden_op(-1)

    def iden_op(self, op):
        for lst in self.atomtypes.values():
            r0 = self.atoms[lst,1:]
            r1 = numpy.dot(r0, op)
            if not numpy.allclose(sort_coords(r0), sort_coords(r1),
                                  atol=GEOM_THRESHOLD):
                return False
        return True

    def search_c_highest(self, zaxis):
        has_c2x = True
        has_mirrorx = True
        maybe_cn = []
        maybe_c2x = []
        maybe_mirrorx = []
        for atype in self.atomtypes.values():
            groups = self.group_atoms_by_distance(atype)
            for lst in groups:
                r0 = self.atoms[lst,1:]
                zcos = numpy.around(numpy.einsum('ij,j->i', r0, zaxis),
                                    decimals=PLACE-1)
                uniq_zcos = numpy.unique(zcos)
                for d in uniq_zcos:
                    cn = (zcos==d).sum()
                    if (cn == 1):
                        if not parallel_vectors(zaxis, r0[zcos==d][0]):
                            raise RuntimeError('Unknown symmetry')
                    else:
                        maybe_cn.append(cn)

                # The possible C2x are composed by those vectors, whose
                # distance to xy-mirror are identical
                if has_c2x:
                    for d in uniq_zcos:
                        if numpy.allclose(d, 0, atol=GEOM_THRESHOLD): # plain which cross the orig
                            r1 = r0[zcos==d]
                            i1, i2 = numpy.tril_indices(len(r1))
                            maybe_c2x.append(r1[i1] + r1[i2])
                        elif d > GEOM_THRESHOLD:
                            mirrord = abs(zcos+d)<GEOM_THRESHOLD
                            if mirrord.sum() == (zcos==d).sum():
                                above = r0[zcos==d]
                                below = r0[mirrord]
                                nelem = len(above)
                                i1, i2 = numpy.indices((nelem,nelem))
                                maybe_c2x.append(above[i1.flatten()] +
                                                 below[i2.flatten()])
                            else:
                                # if the number of mirrored vectors are diff,
                                # it's impossible to have c2x
                                has_c2x = False
                                break

                if has_mirrorx:
                    for d in uniq_zcos:
                        r1 = r0[zcos==d]
                        i1, i2 = numpy.tril_indices(len(r1))
                        maybe_mirrorx.append(numpy.cross(zaxis, r1[i1]+r1[i2]))

        possible_cn = []
        for n in set(maybe_cn):
            for i in range(2, n+1):
                if n % i == 0:
                    possible_cn.append(i)

        r0 = self.atoms[:,1:]
        n = 1
        for i in sorted(set(possible_cn), reverse=True):
            op = rotation_mat(zaxis, numpy.pi*2/i)
            r1 = numpy.dot(r0, op)
            if numpy.allclose(sort_coords(r0), sort_coords(r1),
                              atol=GEOM_THRESHOLD):
                n = i
                break

        #
        # C2 perp to Cn and mirros on Cn
        #

        def pick_vectors(maybe_vec):
            maybe_vec = numpy.vstack(maybe_vec)
            # remove zero-vectors and duplicated vectors
            d = numpy.einsum('ij,ij->i', maybe_vec, maybe_vec)
            maybe_vec = maybe_vec[d>GEOM_THRESHOLD**2]
            maybe_vec /= numpy.sqrt(d[d>GEOM_THRESHOLD**2]).reshape(-1,1)
            maybe_vec = _remove_dupvec(maybe_vec) # also transfer to pseudo-vector

            # remove the C2x which can be related by Cn rotation along z axis
            seen = numpy.zeros(len(maybe_vec), dtype=bool)
            for k, r1 in enumerate(maybe_vec):
                if not seen[k]:
                    cos2r1 = numpy.einsum('j,ij->i', r1, maybe_vec[k+1:])
                    for i in range(1,n):
                        c = numpy.cos(numpy.pi*i/n) # no 2pi because of pseudo-vector
                        seen[k+1:][abs(cos2r1-c) < GEOM_THRESHOLD] = True

            possible_vec = maybe_vec[numpy.logical_not(seen)]
            return possible_vec

        c2x = None
        if has_c2x:
            possible_c2x = pick_vectors(maybe_c2x)
            r0 = sort_coords(self.atoms[:,1:])
            for c in possible_c2x:
                op = rotation_mat(c, numpy.pi)
                r1 = numpy.dot(r0, op)
                if numpy.allclose(sort_coords(r1), r0, atol=GEOM_THRESHOLD):
                    c2x = c
                    break

        mirrorx = None
        if has_mirrorx:
            possible_mirrorx = pick_vectors(maybe_mirrorx)
            r0 = sort_coords(self.atoms[:,1:])
            for c in possible_mirrorx:
                op = householder(c)
                r1 = numpy.dot(r0, op)
                if numpy.allclose(sort_coords(r1), r0, atol=GEOM_THRESHOLD):
                    mirrorx = c
                    break

        return n, c2x, mirrorx


# T/Td/Th/O/Oh/I/Ih
def _search_toi(rawsys):
    def has_rotation(zaxis, n, coords):
        op = rotation_mat(zaxis, numpy.pi*2/n)
        r1 = numpy.dot(coords, op)
        return numpy.allclose(sort_coords(coords), sort_coords(r1),
                              atol=GEOM_THRESHOLD)

    def highest_c(zaxis, coords):
        for n in (5, 4, 3):
            if has_rotation(zaxis, n, coords):
                return n
        return 1

    maybe = []
    maybe_axes = []
    for atype in rawsys.atomtypes.values():
        groups = rawsys.group_atoms_by_distance(atype)
        for lst in groups:
            if len(lst) > 1:
                coords = rawsys.atoms[lst,1:]
                zaxis = rawsys.atoms[lst[0],1:]
                if not parallel_vectors(zaxis, rawsys.atoms[lst[1],1:]):
                    xaxis = rawsys.atoms[lst[1],1:]
                else:
                    xaxis = rawsys.atoms[lst[2],1:]
                maybe_axes.append([zaxis,xaxis])

                cn = highest_c(zaxis, coords)
                if cn == 5:
                    maybe.append('I')
                    break
                elif cn == 4:
                    maybe.append('O')
                    break
                else: # C3 axis or fullerene
                    zcos = numpy.around(numpy.einsum('ij,j->i', coords, zaxis),
                                        decimals=PLACE-1)
                    uniq_zcos = numpy.unique(zcos)
                    for d in reversed(uniq_zcos):
                        idx = numpy.where(zcos==d)[0]
                        if len(idx) >= 2:
                            r1, r2 = coords[idx[:2]]
                            zaxis1 = numpy.cross(r2-zaxis, r1-zaxis)
                            cn = highest_c(zaxis1, coords)
                            if cn == 5:
                                maybe.append('I')
                                # xaxis on mirror
                                maybe_axes[-1] = [zaxis1,zaxis]
                                break
                            elif cn == 4:
                                maybe.append('O')
                                maybe_axes[-1] = [zaxis1,zaxis]
                                break
                            elif cn == 3:
                                maybe.append('T')
                                xaxis = zaxis+r2
                                zaxis = zaxis+r1
                                maybe_axes[-1] = [zaxis,xaxis]
                                break
                            else:
                                raise RuntimeError('Unknown symmetry')
                    break # lst in rawsys.group_atoms_by_distance(atype)

    def make_axes(zx):
        z, x = zx
        y = numpy.cross(z, x)
        x = numpy.cross(y, z) # because x might not perp to z
        x /= numpy.linalg.norm(x)
        y /= numpy.linalg.norm(y)
        z /= numpy.linalg.norm(z)
        return numpy.array((x,y,z))
    if 'T' in maybe:
# FIXME
        gpname = 'Td'
        axes = make_axes(maybe_axes[maybe.index('T')])
    elif 'O' in maybe:
        if rawsys.detect_icenter():
            gpname = 'Oh'
        else:
            gpname = 'O'
        axes = make_axes(maybe_axes[maybe.index('O')])
    else:
        if rawsys.detect_icenter():
            gpname = 'Ih'
        else:
            gpname = 'I'
        axes = make_axes(maybe_axes[maybe.index('I')])
    return gpname, axes


# orientation of rotation axes
def _pesudo_vectors(vs):
    idy0 = abs(vs[:,1])<GEOM_THRESHOLD
    idz0 = abs(vs[:,2])<GEOM_THRESHOLD
    vs = vs.copy()
    # ensure z component > 0
    vs[vs[:,2]<0] *= -1
    # if z component == 0, ensure y component > 0
    vs[numpy.logical_and(vs[:,1]<0, idz0)] *= -1
    # if y and z component == 0, ensure x component > 0
    idx = numpy.logical_and(idy0, idz0)
    vs[numpy.logical_and(vs[:,0]<0, idx)] *= -1
    return vs


def _remove_dupvec(vs):
    def rm_iter(vs):
        if len(vs) <= 1:
            return vs
        else:
            x = numpy.sum(abs(vs[1:]-vs[0]), axis=1)
            rest = rm_iter(vs[1:][x>GEOM_THRESHOLD])
            return numpy.vstack((vs[0], rest))
    return rm_iter(_pesudo_vectors(vs))


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
