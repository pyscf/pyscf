#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import pyscf.gto
import pyscf.lib

GEOM_THRESHOLD = 1e-5

def get_charge_center(atoms):
    xbar = 0
    ybar = 0
    zbar = 0
    totchg = 0
    for atm in atoms:
        symb = atm[0]
        charge = pyscf.gto.mole._charge(symb)
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
        charge = pyscf.gto.mole._charge(symb)
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

def same_vectors(v1, v2):
    return numpy.linalg.norm(v1-v2) < GEOM_THRESHOLD
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

def related_by_icenter(atm1, atm2):
    if atm1[0] != atm2[0]:
        return False
    r1 = atm1[1]
    r2 = atm2[1]
    return numpy.linalg.norm(r1+r2)<GEOM_THRESHOLD
def related_by_C2(c2axis, atm1, atm2):
    # return True if rotated atm1 matches atm2
    if atm1[0] != atm2[0]:
        return False
    r1 = atm1[1]
    r2 = atm2[1]
    if numpy.allclose(r1, r2):
        # atm1 atm2 are same and on the plain
        return parallel_vectors(r1, c2axis)
    elif abs(numpy.linalg.norm(r1+r2)) < GEOM_THRESHOLD:
        if abs(numpy.dot(r1,c2axis)) < GEOM_THRESHOLD:
            # r1 perp to c2axis
            return abs(numpy.linalg.norm(r1+r2)) < GEOM_THRESHOLD
        else:
            return False
    else:
        return parallel_vectors(r1+r2,c2axis)
def related_by_mirror(mir_vec, atm1, atm2):
    # return True if mirrored atm1 matches atm2
    if atm1[0] != atm2[0]:
        return False
    r1 = atm1[1]
    r2 = atm2[1]
    if numpy.allclose(r1, r2):
        # atm1 atm2 are same and on the plain
        return abs(numpy.dot(atm1[1],mir_vec)) < GEOM_THRESHOLD
    else:
        return parallel_vectors(r2-r1,mir_vec)

class SymmOperator(object):
    def __init__(self, atoms):
        atoms = [(a[0], numpy.array(a[1])) for a in atoms]
        self.charge_center = get_charge_center(atoms)
        self.mass_center = get_mass_center(atoms)
        self.atoms = []
        for atm in atoms:
            r = atm[1] - self.charge_center
            self.atoms.append((atm[0], r, numpy.linalg.norm(r)))

    def group_atoms_by_distance(self):
        natom = self.atoms.__len__()
        seen = [False] * natom
        groups = []
        for i, atm in enumerate(self.atoms):
            if not seen[i]:
                seen[i] = True
                groups.append([atm])
                for j in range(i+1, natom):
                    atj = self.atoms[j]
                    if not seen[j] \
                       and atm[0] == atj[0] \
                       and abs(atm[2]-atj[2])<GEOM_THRESHOLD:
                        seen[j] = True
                        groups[-1].append(atj)
        return groups

    def check_op_valid(self, op_test, atom_groups):
        for atoms in atom_groups:
            for atm in atoms:
                # check if op_test on atm can generate an atom of the mole
                if not pyscf.lib.member(op_test, atm, atoms):
                    return False
        return True

    def detect_icenter(self):
        def icenter_is_proper(atm1, atm2):
            if abs(numpy.linalg.norm(atm1[1]))<GEOM_THRESHOLD:
                return True
            else:
                return related_by_icenter(atm1,atm2)
        groups = self.group_atoms_by_distance()
        return self.check_op_valid(icenter_is_proper, groups)

    def detect_C2(self):
        #FIXME: missing c2 perp to plain. e.g.
        # H   H
        #  \ /
        #   H
        #  / \
        # H   H
        groups = self.group_atoms_by_distance()
        maybe_c2 = [] # normalized vector
        for g in groups:
            if g.__len__() == 1:
               if g[0][2] > GEOM_THRESHOLD:
                   maybe_c2.append(normalize(g[0][1]))
            else:
                for i,gi in enumerate(g):
                    for j in range(i):
                        gj = g[j]
                        if numpy.linalg.norm(gi[1]+gj[1]) < GEOM_THRESHOLD:
                            maybe_c2.append(normalize(gi[1]))
                            maybe_c2.append(normalize(vector_perp_to_vector(gi[1])))
                        else:
                            maybe_c2.append(normalize(gi[1]+gj[1]))
        for i, ati in enumerate(self.atoms):
            for j in range(i):
                atj = self.atoms[j]
                vec = vector_perp_to_vectors(ati[1],atj[1])
                maybe_c2.append(normalize(vec))
        maybe_c2 = pyscf.lib.remove_dup(parallel_vectors, maybe_c2, True)
        axes = []
        for axis in maybe_c2:
            def c2_is_proper(atm1,atm2):
                if parallel_vectors(atm1[1], axis):
                    # c2axis is a proper operator when the atom on C2 axis
                    return True
                else:
                    return related_by_C2(axis,atm1,atm2)
            if self.check_op_valid(c2_is_proper, groups):
                axes.append(axis)
        return order_vectors(axes)

    def detect_mirror(self):
        groups = self.group_atoms_by_distance()
        maybe_mirror = [] # normalized vector perp to the mirror
        for g in groups:
            if g.__len__() == 1 \
               and g[0][2] > GEOM_THRESHOLD:
                mir_vec = normalize(g[0][1])
                maybe_mirror.append(mir_vec)
                maybe_mirror.append(vector_perp_to_vector(mir_vec))
            else:
                for i,gi in enumerate(g):
                    for j in range(i):
                        gj = g[j]
                        maybe_mirror.append(vector_perp_to_vectors(gi[1],gj[1]))
                        maybe_mirror.append(normalize(gj[1]-gi[1]))
        maybe_mirror = pyscf.lib.remove_dup(parallel_vectors, maybe_mirror, True)
        mirrors = []
        for mir_vec in maybe_mirror:
            def mirror_is_proper(atm1, atm2):
                if abs(numpy.dot(atm1[1],mir_vec)) < GEOM_THRESHOLD:
                    # mirror is a proper operator when the atom on mirror
                    return True
                else:
                    return related_by_mirror(mir_vec,atm1,atm2)
            if self.check_op_valid(mirror_is_proper, groups):
                mirrors.append(mir_vec)
        return order_vectors(mirrors)

def vector_parallel_x(vec):
    return abs(vec[1]) < GEOM_THRESHOLD \
            and abs(vec[2]) < GEOM_THRESHOLD
def vector_parallel_y(vec):
    return abs(vec[0]) < GEOM_THRESHOLD \
            and abs(vec[2]) < GEOM_THRESHOLD
def vector_parallel_z(vec):
    return abs(vec[0]) < GEOM_THRESHOLD \
            and abs(vec[1]) < GEOM_THRESHOLD
def order_vectors(vecs):
    # ordering vectors, move (0,0,1),(1,0,0),(0,1,0) to the front
    if list(filter(vector_parallel_y, vecs)):
        vecs = [numpy.array((0.,1.,0.))] + pyscf.lib.remove_if(vector_parallel_y, vecs)
    if list(filter(vector_parallel_x, vecs)):
        vecs = [numpy.array((1.,0.,0.))] + pyscf.lib.remove_if(vector_parallel_x, vecs)
    if list(filter(vector_parallel_z, vecs)):
        vecs = [numpy.array((0.,0.,1.))] + pyscf.lib.remove_if(vector_parallel_z, vecs)
    return vecs

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
        return None

####################################
def atoms_in_line(atoms):
    if len(atoms) == 2:
        return True
    else:
        coords = numpy.array([a[1] for a in atoms])
        v12 = coords[1] - coords[0]
        return all([parallel_vectors(c-coords[0], v12) for c in coords[2:]])

def detect_symm(atoms):
    if atoms.__len__() == 1:
        return 'D2h', numpy.zeros(3), numpy.eye(3)
    elif atoms.__len__() == 2:
        rchg = get_charge_center(atoms)
        new_axis = gen_new_axis(numpy.array(atoms[0][1])-rchg, axisx=(1,0,0))
        if atoms[0][0] == atoms[1][0]:
            return 'D2h', rchg, new_axis
        else:
            return 'C2v', rchg, new_axis
    elif atoms_in_line(atoms):
        rchg = get_charge_center(atoms)
        if numpy.allclose(atoms[0][1], 0):
            zaxis = numpy.array(atoms[1][1])-rchg
        else:
            zaxis = numpy.array(atoms[0][1])-rchg
        new_axis = gen_new_axis(zaxis, axisx=(1,0,0))
        symbs = [a[0] for a in atoms]
        if symbs == list(reversed(symbs)):
            return 'D2h', rchg, new_axis
        else:
            return 'C2v', rchg, new_axis
    else:
        rchg = get_charge_center(atoms)
        ops = SymmOperator(atoms)
        icenter = ops.detect_icenter()
        c2_axis = ops.detect_C2()
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
            mirror = ops.detect_mirror()
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

def symm_identical_atoms(gpname, atoms):
    natoms = atoms.__len__()
    ex, ey, ez = numpy.eye(3)

    if gpname == 'C1':
        return [(i,) for i in range(natoms)]
    elif gpname == 'Ci':
        tests = (related_by_icenter,)
    elif gpname == 'Cs':
        tests = (lambda a1,a2:related_by_mirror(ez,a1,a2),)
    elif gpname == 'C2':
        tests = (lambda a1,a2:related_by_C2(ez,a1,a2),)
    elif gpname == 'C2v':
        tests = (lambda a1,a2:related_by_C2(ez,a1,a2), \
                 lambda a1,a2:related_by_mirror(ex,a1,a2), \
                 lambda a1,a2:related_by_mirror(ey,a1,a2))
    elif gpname == 'C2h':
        tests = (related_by_icenter, \
                 lambda a1,a2:related_by_C2(ez,a1,a2), \
                 lambda a1,a2:related_by_mirror(ez,a1,a2))
    else: # D2 or D2h
        tests = [lambda a1,a2:related_by_C2(ex,a1,a2), \
                 lambda a1,a2:related_by_C2(ey,a1,a2), \
                 lambda a1,a2:related_by_C2(ez,a1,a2)]
        if gpname == 'D2h':
            tests.extend((related_by_icenter, \
                          lambda a1,a2:related_by_mirror(ex,a1,a2), \
                          lambda a1,a2:related_by_mirror(ey,a1,a2), \
                          lambda a1,a2:related_by_mirror(ez,a1,a2)))

    def check(tests, atm1, atm2):
        if atm1[0] == atm2[0] and abs(atm1[2]-atm2[2])<GEOM_THRESHOLD:
            for test in tests:
                if test(atm1,atm2):
                    return True
        return False

    ops = SymmOperator(atoms)
    seen = [False] * natoms
    eql_atom_ids = []
    for ia,ati in enumerate(ops.atoms):
        if not seen[ia]:
            eql_atom_ids.append([ia])
            seen[ia] = True
            for ja,atj in enumerate(ops.atoms):
                if not seen[ja] and check(tests,ati,atj):
                    eql_atom_ids[-1].append(ja)
                    seen[ja] = True
    return eql_atom_ids

def check_given_symm(gpname, atoms):
    eql_atoms = symm_identical_atoms(gpname, atoms)
    #fixme
    pass

if __name__ == "__main__":
    atom = [["O" , (1. , 0.    , 0.   ,)],
            [1   , (0. , -.757 , 0.587,)],
            [1   , (0. , 0.757 , 0.587,)] ]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, symm_identical_atoms(gpname, atom))

    atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['H', (0,0,1)]]
    gpname, orig, axes = detect_symm(atom)
    print(gpname, symm_identical_atoms(gpname, atom))
