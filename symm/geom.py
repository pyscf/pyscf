#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import pyscf.lib
from pyscf.gto import mole
import pyscf.symm.param

GEOM_THRESHOLD = 1e-5
PLACE = int(-numpy.log(GEOM_THRESHOLD))

def get_charge_center(atoms):
    xbar = 0
    ybar = 0
    zbar = 0
    totchg = 0
    for atm in atoms:
        symb = atm[0]
        charge = mole._charge(symb)
        x,y,z = atm[1]
        xbar += charge * x
        ybar += charge * y
        zbar += charge * z
        totchg += charge
    return numpy.array((xbar,ybar,zbar), dtype=float) / totchg

def get_mass_center(atoms):
    xbar = 0
    ybar = 0
    zbar = 0
    totmass = 0
    for atm in atoms:
        symb = atm[0]
        charge = mole._charge(symb)
        mass = pyscf.lib.parameters.ELEMENTS[charge][1]
        x,y,z = atm[1]
        xbar += mass * x
        ybar += mass * y
        zbar += mass * z
        totmass += mass
    return numpy.array((xbar,ybar,zbar)) / totmass

def gen_new_axis(axisz, axisx=None, axisy=None):
    if axisx is not None:
        axisy = numpy.cross(axisz, axisx)
    elif axisy is not None:
        axisx = numpy.cross(axisy, axisz)
    elif abs(axisz[1]) < GEOM_THRESHOLD: # in xz plain
        axisy = numpy.array((0.,1.,0.))
        axisx = numpy.cross(axisy, axisz)
    elif abs(axisz[0]) < abs(axisz[1]): # close to old y axis
        axisx = numpy.array((axisz[1], -axisz[0], 0))
        axisy = numpy.cross(axisz, axisx)
    else:
        axisy = numpy.array((-axisz[1], axisz[0], 0))
        axisx = numpy.cross(axisy, axisz)
    new_axis = numpy.array((axisx/numpy.linalg.norm(axisx),
                            axisy/numpy.linalg.norm(axisy),
                            axisz/numpy.linalg.norm(axisz)))
    return new_axis

def normalize(v):
    rr = numpy.linalg.norm(v)
    if rr > 1e-12:
        return v/rr
    else:
        return v

def parallel_vectors(v1, v2):
    v3 = numpy.cross(v1, v2)
    return numpy.linalg.norm(v3) < GEOM_THRESHOLD

def vector_perp_to_vector(v1):
    if abs(v1[0]) > GEOM_THRESHOLD or abs(v1[1]) > GEOM_THRESHOLD:
        return numpy.array((-v1[1],v1[0],0.))
    else:
        return numpy.array((1.,0.,0.))
def vector_perp_to_vectors(v1, v2):
    v3 = numpy.cross(v1,v2)
    norm = numpy.linalg.norm(v3)
    if norm < GEOM_THRESHOLD:
        return vector_perp_to_vector(v1)
    else:
        return v3/norm

def vector_parallel_x(vec):
    return abs(vec[1]) < GEOM_THRESHOLD and abs(vec[2]) < GEOM_THRESHOLD
def vector_parallel_y(vec):
    return abs(vec[0]) < GEOM_THRESHOLD and abs(vec[2]) < GEOM_THRESHOLD
def vector_parallel_z(vec):
    return abs(vec[0]) < GEOM_THRESHOLD and abs(vec[1]) < GEOM_THRESHOLD
def order_vectors(vecs):
    # ordering vectors, move (0,0,1),(1,0,0),(0,1,0) to the front
    if list(filter(vector_parallel_y, vecs)):
        vecs = [numpy.array((0.,1.,0.))] + pyscf.lib.remove_if(vector_parallel_y, vecs)
    if list(filter(vector_parallel_x, vecs)):
        vecs = [numpy.array((1.,0.,0.))] + pyscf.lib.remove_if(vector_parallel_x, vecs)
    if list(filter(vector_parallel_z, vecs)):
        vecs = [numpy.array((0.,0.,1.))] + pyscf.lib.remove_if(vector_parallel_z, vecs)
    return vecs

def argsort_coords(coords):
    coords = numpy.array(coords)
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
    uu = vec.reshape(-1,1) * vec.reshape(1,-1)
    ux = numpy.array((
        ( 0     ,-vec[2], vec[1]),
        ( vec[2], 0     ,-vec[0]),
        (-vec[1], vec[0], 0     )))
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    r = c * numpy.eye(3) + s * ux + (1-c) * uu
    return r

def find_axis(vecs):
    nv = vecs.__len__()
    if nv < 3:
        return None
    elif list(filter(vector_parallel_x, vecs)) \
         and list(filter(vector_parallel_z, vecs)):
        return numpy.eye(3)
    else:
        orthtab = numpy.zeros((nv,nv),dtype=bool)
        for i in range(nv):
            for j in range(i):
                if abs(numpy.dot(vecs[i],vecs[j])) < GEOM_THRESHOLD:
                    orthtab[i,j] = orthtab[j,i] = True
        for i in range(nv):
            if orthtab[i].sum() >= 2:
                lst = [j for j in range(nv) if orthtab[i,j]]
                for j in lst:
                    if orthtab[j,lst].sum() > 0:
                        return gen_new_axis(vecs[i], axisx=vecs[j])

####################################
def atoms_in_line(atoms):
    if len(atoms) == 2:
        return True
    else:
        coords = numpy.array([a[1] for a in atoms])
        v12 = coords[1] - coords[0]
        return all([parallel_vectors(c-coords[0], v12) for c in coords[2:]])

def atoms_in_plain(atoms):
    coords = numpy.array([a[1] for a in atoms])
    v12 = coords[1] - coords[0]
    for c in coords[2:]:
        if not parallel_vectors(c-coords[0], v12):
            plain = numpy.cross(v12, c-coords[0])
            if numpy.allclose(numpy.dot(coords[2:], plain), 0, atol=GEOM_THRESHOLD):
                return plain
    return False

def detect_symm(atoms, basis=None):
    '''
    Return group name, charge center, and nex_axis (three rows for x,y,z)
    '''
    rawsys = SymmSys(atoms, basis)
    if atoms.__len__() == 1:
        return 'D2h', numpy.zeros(3), numpy.eye(3)
    elif atoms.__len__() == 2:
        rchg = rawsys.charge_center
        new_axis = gen_new_axis(numpy.array(atoms[0][1])-rchg, axisx=(1,0,0))
        if rawsys.detect_icenter():
            return 'D2h', rchg, new_axis
        else:
            return 'C2v', rchg, new_axis
    elif atoms_in_line(atoms):
        rchg = rawsys.charge_center
        if numpy.allclose(atoms[0][1], 0):
            zaxis = numpy.array(atoms[1][1]) - rchg
        else:
            zaxis = numpy.array(atoms[0][1]) - rchg
        new_axis = gen_new_axis(zaxis, axisx=(1,0,0))
        if rawsys.detect_icenter():
            return 'D2h', rchg, new_axis
        else:
            return 'C2v', rchg, new_axis
    else:
        rchg = rawsys.charge_center
        icenter = rawsys.detect_icenter()
        c2_axis = rawsys.detect_C2()
        c2_xyz = find_axis(c2_axis)
        if icenter:
            if not c2_axis:
                return 'Ci', rchg, numpy.eye(3)
            elif c2_axis.__len__() >= 3 and c2_xyz is not None:
                return 'D2h', rchg, c2_xyz
            else:
                new_axis = gen_new_axis(c2_axis[0])
                return 'C2h', rchg, new_axis
        else:
            mirror = rawsys.detect_mirror()
            if c2_axis.__len__() >= 3 and c2_xyz is not None:
                return 'D2', rchg, c2_xyz
            elif c2_axis and mirror:
                new_axis = gen_new_axis(c2_axis[0], axisy=mirror[0])
                return 'C2v', rchg, new_axis
            elif c2_axis:
                new_axis = gen_new_axis(c2_axis[0])
                return 'C2', rchg, new_axis
            elif mirror:
                new_axis = gen_new_axis(mirror[0])
                return 'Cs', rchg, new_axis
            else:
                return 'C1', rchg, numpy.eye(3)

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
    if not numpy.allclose(get_charge_center(atoms), 0):
        raise RuntimeError('''The molecule needs to be placed in the standard orientation.
It can be obtained using the return variables of detect_symm.''')
    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    rawsys = SymmSys(atoms)
    coords = numpy.array([a[1] for a in rawsys.atoms])
    idx = argsort_coords(coords)
    coords0 = coords[idx]

    dup_atom_ids = []
    for op in ops:
        newc = numpy.around(numpy.dot(coords, op), decimals=PLACE-1)
        idx = argsort_coords(newc)
        assert(numpy.allclose(coords0, newc[idx]))
        dup_atom_ids.append(idx)

    dup_atom_ids = numpy.sort(dup_atom_ids, axis=0).T
    uniq_idx = numpy.unique(dup_atom_ids[:,0])
    eql_atom_ids = dup_atom_ids[uniq_idx]
    eql_atom_ids = [list(set(i)) for i in eql_atom_ids]
    return eql_atom_ids

def check_given_symm(gpname, atoms, basis=None):
# more strict than symm_identical_atoms, we required not only the coordinates
# match, but also the symbols and basis functions
    opdic = symm_ops(gpname)
    ops = [opdic[op] for op in pyscf.symm.param.OPERATOR_TABLE[gpname]]
    rawsys = SymmSys(atoms)
    for lst in rawsys.uniq_atoms.values():
        coords = numpy.array([rawsys.atoms[i][1] for i in lst])
        idx = argsort_coords(coords)
        coords0 = coords[idx]

        for op in ops:
            newc = numpy.around(numpy.dot(coords, op), decimals=PLACE-1)
            idx = argsort_coords(newc)
            if not numpy.allclose(coords0, newc[idx]):
                return False
    return True

def shift_atom(atoms, orig, axis):
    c = numpy.array([a[1] for a in atoms])
    c = numpy.dot(c - orig, numpy.array(axis).T)
    return [[atoms[i][0], c[i]] for i in range(len(atoms))]

class SymmSys(object):
    def __init__(self, atoms, basis=None):
        atoms = [(a[0], numpy.array(a[1])) for a in atoms]
        self.charge_center = get_charge_center(atoms)
        self.mass_center = get_mass_center(atoms)
# it's important to round off the coordinates, so that we can use
# numpy.allclose to compare the coordinates
        self.atoms = [[a[0], numpy.around(a[1]-self.charge_center, decimals=PLACE-1)]
                      for a in atoms]
        self.uniq_atoms = mole.unique_atoms(atoms, basis)

    def group_atoms_by_distance(self, index):
        r = [numpy.linalg.norm(self.atoms[i][1]) for i in index]
        lst = numpy.argsort(r)
        groups = [[index[lst[0]]]]
        for i in range(len(lst)-1):
            if numpy.allclose(r[lst[i]], r[lst[i+1]]):
                groups[-1].append(index[lst[i+1]])
            else:
                groups.append([index[lst[i+1]]])
        return groups

    def check_op_valid(self, op_test, atom_groups):
        for atoms in atom_groups:
            for atm in atoms:
                # check if op_test on atm can generate an atom of the mole
                if not pyscf.lib.member(op_test, atm, atoms):
                    return False
        return True

    def remove_dupvec(self, vs):
        def p_normalize(vs):
            # pseudo vector
            idy0 = abs(vs[:,1])<GEOM_THRESHOLD
            idz0 = abs(vs[:,2])<GEOM_THRESHOLD
            vs = vs.copy()
            vs[vs[:,2]<0] *= -1
            vs[numpy.logical_and(vs[:,1]<0, idz0)] *= -1
            idx = numpy.logical_and(idy0, idz0)
            vs[numpy.logical_and(vs[:,0]<0, idx)] *= -1
            return vs
        def rm_iter(vs):
            if len(vs) <= 1:
                return vs
            else:
                x = numpy.sum(abs(vs[1:]-vs[0]), axis=1)
                rest = rm_iter(vs[1:][x>GEOM_THRESHOLD])
                return numpy.vstack((vs[0], rest))
        return rm_iter(p_normalize(vs))

    def detect_icenter(self):
        for lst in self.uniq_atoms.values():
            coords = sort_coords([self.atoms[i][1] for i in lst])
            newc = sort_coords(-coords)
            if not numpy.allclose(coords, newc, atol=GEOM_THRESHOLD):
                return False
        return True

    def detect_C2(self):
        def search_maybe_c2(groups):
            maybe_c2 = [] # normalized vector
            for g in groups:
                if g.__len__() == 1:
                    if not numpy.allclose(self.atoms[g[0]][1], 0):
                        maybe_c2.append(normalize(self.atoms[g[0]][1]))
                else:
                    for i,gi in enumerate(g):
                        for j in range(i):
                            gj = g[j]
                            if numpy.allclose(self.atoms[gi][1]+self.atoms[gj][1],0):
                                maybe_c2.append(normalize(self.atoms[gi][1]))
                                # linear molecules are handled as a special case
                                #maybe_c2.append(normalize(vector_perp_to_vector(gi[1])))
                            else:
                                maybe_c2.append(normalize(self.atoms[gi][1]+self.atoms[gj][1]))
                                vec = vector_perp_to_vectors(self.atoms[gi][1], self.atoms[gj][1])
                                maybe_c2.append(normalize(vec))
            return self.remove_dupvec(numpy.array(maybe_c2))
        maybe_c2 = []
        for lst in self.uniq_atoms.values():
            groups = self.group_atoms_by_distance(lst)
            maybe_c2.extend(search_maybe_c2(groups))
        #c2 perp to plain. e.g.
        # H   He
        #  \ /
        #   H
        #  / \
        # He  H
        if len(self.atoms) > 2:
            plain_vec = atoms_in_plain(self.atoms)
            if isinstance(plain_vec, numpy.ndarray):
                maybe_c2.append(plain_vec)

        if len(maybe_c2) == 0:
            return False
        else:
            maybe_c2 = self.remove_dupvec(numpy.array(maybe_c2))

            axes = []
            for axis in maybe_c2:
                rotmat = rotation_mat(axis, numpy.pi)
                alltrue = True
                for lst in self.uniq_atoms.values():
                    coords = sort_coords([self.atoms[i][1] for i in lst])
                    newc = numpy.einsum('ij,kj->ki', rotmat, coords)
                    newc = sort_coords(numpy.around(newc, decimals=PLACE-1))
                    if not numpy.allclose(coords, newc, atol=GEOM_THRESHOLD):
                        alltrue = False
                        break
                if alltrue:
                    axes.append(axis)
            return order_vectors(axes)

    def detect_mirror(self):
        maybe_mirror = [] # normalized vector perp to the mirror
        for lst in self.uniq_atoms.values():
            groups = self.group_atoms_by_distance(lst)
            for g in groups:
                for i,gi in enumerate(g):
                    for j in range(i):
                        gj = g[j]
                        maybe_mirror.append(normalize(self.atoms[gj][1]
                                                     -self.atoms[gi][1]))

        if len(self.atoms) > 2:
            plain_vec = atoms_in_plain(self.atoms)
            if isinstance(plain_vec, numpy.ndarray):
                maybe_mirror.append(plain_vec)

        if len(maybe_mirror) == 0:
            return False
        else:
            maybe_mirror = self.remove_dupvec(numpy.array(maybe_mirror))

            mirrors = []
            for mir_vec in maybe_mirror:
                alltrue = True
                rotmat = -rotation_mat(mir_vec, numpy.pi)
                for lst in self.uniq_atoms.values():
                    coords = sort_coords([self.atoms[i][1] for i in lst])
                    newc = numpy.einsum('ij,kj->ki', rotmat, coords)
                    newc = sort_coords(numpy.around(newc, decimals=PLACE-1))
                    if not numpy.allclose(coords, newc, atol=GEOM_THRESHOLD):
                        alltrue = False
                        break
                if alltrue:
                    mirrors.append(mir_vec)
            return order_vectors(mirrors)


if __name__ == "__main__":
    atom = [["O" , (1. , 0.    , 0.   ,)],
            ['H' , (0. , -.757 , 0.587,)],
            ['H' , (0. , 0.757 , 0.587,)] ]
    gpname, orig, axes = detect_symm(atom)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['H', (0,0,1)]]
    gpname, orig, axes = detect_symm(atom)
    atom = shift_atom(atom, orig, axes)
    print(gpname, symm_identical_atoms(gpname, atom))
