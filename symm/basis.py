#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf import gto
import geom
import param

OP_TEST = {
    'E'  : lambda x,y:True,
    'C2x': lambda x,y:geom.related_by_C2(numpy.array((1.,0.,0.)),x,y),
    'C2y': lambda x,y:geom.related_by_C2(numpy.array((0.,1.,0.)),x,y),
    'C2z': lambda x,y:geom.related_by_C2(numpy.array((0.,0.,1.)),x,y),
    'i'  : geom.related_by_icenter,
    'sx' : lambda x,y:geom.related_by_mirror(numpy.array((1.,0.,0.)),x,y),
    'sy' : lambda x,y:geom.related_by_mirror(numpy.array((0.,1.,0.)),x,y),
    'sz' : lambda x,y:geom.related_by_mirror(numpy.array((0.,0.,1.)),x,y),
}

OP_PARITY_ODD = {
    'E'  : (0, 0, 0),
    'C2x': (0, 1, 1),
    'C2y': (1, 0, 1),
    'C2z': (1, 1, 0),
    'i'  : (1, 1, 1),
    'sx' : (1, 0, 0),
    'sy' : (0, 1, 0),
    'sz' : (0, 0, 1),
}

def tot_parity_odd(op, l, m):
    if op == 'E':
        return 0
    else:
        ox,oy,oz = OP_PARITY_ODD[op]
        gx,gy,gz = param.SPHERIC_GTO_PARITY_ODD[l][l+m]
        return (ox and gx)^(oy and gy)^(oz and gz)

def basis_offset_for_atoms(atoms, basis_tab):
    basoff = [0]
    n = 0
    for at in atoms:
        symb = gto.mole._symbol(at[0])
        for b in basis_tab[symb]:
            angl = b[0]
            if isinstance(b[1], int):
                nctr = b[2].__len__() - 1
            else:
                nctr = b[1].__len__() - 1
            n += nctr * (angl*2+1)
        basoff.append(n)
    return n, basoff

def symm_adapted_basis(gpname, eql_atom_ids, atoms, basis_tab):
    atoms = [(a[0], numpy.array(a[1])) for a in atoms]
    ops = param.OPERATOR_TABLE[gpname]
    chartab = param.CHARACTER_TABLE[gpname]
    nirrep = chartab.__len__()
    so = [[] for i in range(nirrep)]
    nfn, basoff = basis_offset_for_atoms(atoms, basis_tab)

    for atom_ids in eql_atom_ids:
        at0 = atoms[atom_ids[0]]
        symb = gto.mole._symbol(at0[0])
        def op_test(x):
            return OP_TEST[op](at0, atoms[x])
        op_relate_atoms = [lib.find_if(op_test, atom_ids) for op in ops]

        ib = 0
        for b in basis_tab[symb]:
            angl = b[0]
            if isinstance(b[1], int):
                nctr = b[2].__len__() - 1
            else:
                nctr = b[1].__len__() - 1
            for i in range(nctr):
                for m in range(-angl,angl+1):
                    sign = [-1 if tot_parity_odd(op,angl,m) else 1 \
                                     for op in ops]
                    c = numpy.zeros((nirrep,nfn))
                    for op_id,atm_id in enumerate(op_relate_atoms):
                        idx = basoff[atm_id] + ib
                        for ir, irrep in enumerate(chartab):
                            c[ir,idx] += irrep[op_id+1] * sign[op_id]
                    for ir in range(nirrep):
                        if abs(numpy.linalg.norm(c[ir])) > 1e-14:
                            so[ir].append(c[ir])
                    ib += 1
    for ir,c in enumerate(so):
        if len(c) > 0:
            so[ir] = numpy.array(c).T
        else:
            so[ir] = numpy.empty((nfn,0))
    return so

def dump_symm_adapted_basis(mol, so):
    pass

def irrep_name(groupname, irrep_id):
    return param.CHARACTER_TABLE[groupname][irrep_id][0]

def symmetrize_matrix(mat, so):
    return [reduce(numpy.dot, (c.T,mat,c)) for c in so]


if __name__ == "__main__":
    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#"out_h2o"
    h2o.atom = [['O' , (1. , 0.    , 0.   ,)],
                [1   , (0. , -.757 , 0.587,)],
                [1   , (0. , 0.757 , 0.587,)] ]
    h2o.basis = {'H': '6_31g',
                 'O': '6_31g',}
    gpname, origin, axes = symm.detect_symm(h2o.atom)
    atoms = gto.mole.format_atom(atoms, origin, axes)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(gpname, eql_atoms, atoms, h2o.basis))
