#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate symmetry adapted basis
'''

from functools import reduce
import numpy
import pyscf.lib
import pyscf.gto
from pyscf.symm import geom
from pyscf.symm import param

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

def symm_adapted_basis(gpname, eql_atom_ids, atoms, basis_tab):
    if gpname in ('Dooh', 'Coov'):
        return linearmole_symm_adapted_basis(gpname, eql_atom_ids, atoms, basis_tab)
    decimals = int(-numpy.log10(geom.TOLERANCE)) - 1
    opdic = geom.symm_ops(gpname)
    ops = [opdic[op] for op in param.OPERATOR_TABLE[gpname]]
    chartab = param.CHARACTER_TABLE[gpname]
    nirrep = chartab.__len__()
    nfn, basoff = _basis_offset_for_atoms(atoms, basis_tab)
    coords = numpy.around([a[1] for a in atoms], decimals=decimals)

    sodic = {}
    def add_so(ir, c):
        if ir in sodic:
            sodic[ir].append(c)
        else:
            sodic[ir] = [c]
    for atom_ids in eql_atom_ids:
        at0 = atoms[atom_ids[0]]
        symb = pyscf.gto.mole._symbol(at0[0])
        op_coords = numpy.around([numpy.dot(at0[1], op) for op in ops],
                                 decimals=decimals)
# num atoms in atom_ids smaller than num ops?
## the coord-ids generated from the ops sequence
#        id_coords = numpy.argsort(geom.argsort_coords(op_coords))
## which atom_id that each coord-id corresponds to
#        from_atom_id = numpy.argsort(geom.argsort_coords(coords[atom_ids]))
#        op_relate_atoms = atom_ids[from_atom_id[id_coords]]
        coords0 = coords[atom_ids]
        idx = []
        for c in op_coords:
            idx.append(numpy.argwhere(numpy.sum(abs(coords0-c),axis=1)
                                      <geom.TOLERANCE)[0,0])
        op_relate_atoms = numpy.array(atom_ids)[idx]

        ib = 0
        if symb in basis_tab:
            bas0 = basis_tab[symb]
        else:
            bas0 = basis_tab[pyscf.gto.mole._rm_digit(symb)]
        for b in bas0:
            angl = b[0]
            for i in range(_num_contract(b)):
                for m in range(-angl,angl+1):
                    sign = [-1 if tot_parity_odd(op,angl,m) else 1
                            for op in param.OPERATOR_TABLE[gpname]]
                    c = numpy.zeros((nirrep,nfn))
                    for op_id, atm_id in enumerate(op_relate_atoms):
                        idx = basoff[atm_id] + ib
                        for ir, irrep in enumerate(chartab):
                            c[ir,idx] += irrep[op_id+1] * sign[op_id]
                    norms = numpy.sqrt(numpy.einsum('ij,ij->i', c, c))
                    for ir in range(nirrep):
                        if(norms[ir] > 1e-12):
                            add_so(ir, c[ir]/norms[ir])
                    ib += 1
    so = []
    irrep_ids = sorted(sodic.keys())
    for ir in irrep_ids:
        so.append(numpy.vstack(sodic[ir]).T)
    return so, irrep_ids

def dump_symm_adapted_basis(mol, so):
    raise RuntimeError('TODO')

def symmetrize_matrix(mat, so):
    return [reduce(numpy.dot, (c.T,mat,c)) for c in so]

def _basis_offset_for_atoms(atoms, basis_tab):
    basoff = [0]
    n = 0
    for at in atoms:
        symb = pyscf.gto.mole._symbol(at[0])
        if symb in basis_tab:
            bas0 = basis_tab[symb]
        else:
            bas0 = basis_tab[pyscf.gto.mole._rm_digit(symb)]
        for b in bas0:
            angl = b[0]
            n += _num_contract(b) * (angl*2+1)
        basoff.append(n)
    return n, basoff

def _num_contract(basis):
    if isinstance(basis[1], int):
# This branch should never be reached if basis_tab is formated by function mole.format_basis
        nctr = len(basis[2]) - 1
    else:
        nctr = len(basis[1]) - 1
    return nctr


###############################
# Linear molecule
# Irreps ID maps
# Dooh     ->  D2h        |   Coov      -> C2v
# A1g   0      Ag    0    |   A1    0      A1    0
# A2g   1      B1g   1    |   A2    1      A2    1
# A1u   5      B1u   5    |   E1x   2      B1    2
# A2u   4      Au    4    |   E1y   3      B2    3
# E1gx  2      B2g   2    |   E2x   10     A1    0
# E1gy  3      B3g   3    |   E2y   11     A2    1
# E1ux  7      B3u   7    |   E3x   12     B1    2
# E1uy  6      B2u   6    |   E3y   13     B2    3
# E2gx  10     Ag    0    |   E4x   20     A1    0
# E2gy  11     B1g   1    |   E4y   21     A2    1
# E2ux  15     B1u   5    |   E5x   22     B1    2
# E2uy  14     Au    4    |   E5y   23     B2    3
# E3gx  12     B2g   2    |
# E3gy  13     B3g   3    |
# E3ux  17     B3u   7    |
# E3uy  16     B2u   6    |
# E4gx  20     Ag    0    |
# E4gy  21     B1g   1    |
# E4ux  25     B1u   5    |
# E4uy  24     Au    4    |
# E5gx  22     B2g   2    |
# E5gy  23     B3g   3    |
# E5ux  27     B3u   7    |
# E5uy  26     B2u   6    |

DOOH_IRREP_ID_TABLE = {
    'A1g' : 0,
    'A2g' : 1,
    'A1u' : 5,
    'A2u' : 4,
    'E1gx': 2,
    'E1gy': 3,
    'E1ux': 7,
    'E1uy': 6,
    '_evengx': 0,
    '_evengy': 1,
    '_evenux': 5,
    '_evenuy': 4,
    '_oddgx': 2,
    '_oddgy': 3,
    '_oddux': 7,
    '_odduy': 6,
}
COOV_IRREP_ID_TABLE = {
    'A1' : 0,
    'A2' : 1,
    'E1x': 2,
    'E1y': 3,
    '_evenx': 0,
    '_eveny': 1,
    '_oddx': 2,
    '_oddy': 3,
}

def linearmole_symm_descent(gpname, irrepid):
    '''Map irreps to D2h or C2v'''
    if gpname in ('Dooh', 'Coov'):
        return irrepid % 10
    else:
        raise RuntimeError('%s is not proper for linear molecule.' % gpname)

def linearmole_irrep_symb2id(gpname, symb):
    if gpname == 'Dooh':
        if symb in DOOH_IRREP_ID_TABLE:
            return DOOH_IRREP_ID_TABLE[symb]
        else:
            n = int(''.join([i for i in symb if i.isdigit()]))
            if n % 2:
                return (n//2)*10 + DOOH_IRREP_ID_TABLE['_odd'+symb[-2:]]
            else:
                return (n//2)*10 + DOOH_IRREP_ID_TABLE['_even'+symb[-2:]]
    elif gpname == 'Coov':
        if symb in COOV_IRREP_ID_TABLE:
            return COOV_IRREP_ID_TABLE[symb]
        else:
            n = int(''.join([i for i in symb if i.isdigit()]))
            if n % 2:
                return (n//2)*10 + COOV_IRREP_ID_TABLE['_odd'+symb[-1]]
            else:
                return (n//2)*10 + COOV_IRREP_ID_TABLE['_even'+symb[-1]]
    else:
        raise RuntimeError('%s is not proper for linear molecule.' % gpname)

DOOH_IRREP_SYMBS = ('A1g' , 'A2g' , 'E1gx', 'E1gy' , 'A2u', 'A1u' , 'E1uy', 'E1ux')
DOOH_IRREP_SYMBS_EXT = ('gx' , 'gy' , 'gx', 'gy' , 'uy', 'ux' , 'uy', 'ux')
COOV_IRREP_SYMBS = ('A1' , 'A2' , 'E1x', 'E1y')
def linearmole_irrep_id2symb(gpname, irrep_id):
    if gpname == 'Dooh':
        if irrep_id < 10:
            return DOOH_IRREP_SYMBS[irrep_id]
        else:
            n = irrep_id % 10
            m = irrep_id // 10
            if n in (0, 1, 5, 4):
                rn = m*2
            else:
                rn = m*2+1
            return 'E%d%s' % (rn, DOOH_IRREP_SYMBS_EXT[n])
    elif gpname == 'Coov':
        if irrep_id < 10:
            return COOV_IRREP_SYMBS[irrep_id]
        else:
            n = irrep_id % 10
            m = irrep_id // 10
            if n < 2:
                rn = m*2
            else:
                rn = m*2+1
            if n % 2:
                xy = 'y'
            else:
                xy = 'x'
            return 'E%d%s' % (rn, xy)
    else:
        raise RuntimeError('%s is not proper for linear molecule.' % gpname)

def linearmole_symm_adapted_basis(gpname, eql_atom_ids, atoms, basis_tab):
    assert(gpname in ('Dooh', 'Coov'))
    nbf, basoff = _basis_offset_for_atoms(atoms, basis_tab)
    sodic = {}
    shalf = numpy.sqrt(.5)
    def plus(i0, i1):
        c = numpy.zeros(nbf)
        c[i0] = c[i1] = shalf
        return c
    def minus(i0, i1):
        c = numpy.zeros(nbf)
        c[i0] = shalf
        c[i1] =-shalf
        return c
    def identity(i0):
        c = numpy.zeros(nbf)
        c[i0] = 1
        return c
    def add_so(irrep_name, c):
        if irrep_name in sodic:
            sodic[irrep_name].append(c)
        else:
            sodic[irrep_name] = [c]

    if gpname == 'Dooh':
        for atom_ids in eql_atom_ids:
            if len(atom_ids) == 2:
                at0 = atom_ids[0]
                at1 = atom_ids[1]
                symb = pyscf.gto.mole._symbol(atoms[at0][0])
                ib = 0
                if symb in basis_tab:
                    bas0 = basis_tab[symb]
                else:
                    bas0 = basis_tab[pyscf.gto.mole._rm_digit(symb)]
                for b in bas0:
                    angl = b[0]
                    nc = _num_contract(b)
                    degen = angl * 2 + 1
                    if angl == 1:
                        for i in range(nc):
                            aoff = ib + i*degen + angl
# m = 0
                            idx0 = basoff[at0] + aoff + 1
                            idx1 = basoff[at1] + aoff + 1
                            add_so('A1g', minus(idx0, idx1))
                            add_so('A1u', plus (idx0, idx1))
# m = +/- 1
                            idx0 = basoff[at0] + aoff - 1
                            idy0 = basoff[at0] + aoff
                            idx1 = basoff[at1] + aoff - 1
                            idy1 = basoff[at1] + aoff
                            add_so('E1ux', plus (idx0, idx1))
                            add_so('E1uy', plus (idy0, idy1))
                            add_so('E1gx', minus(idx0, idx1))
                            add_so('E1gy', minus(idy0, idy1))
                    else:
                        for i in range(nc):
                            aoff = ib + i*degen + angl
# m = 0
                            idx0 = basoff[at0] + aoff
                            idx1 = basoff[at1] + aoff
                            if angl % 2: # p-sigma, f-sigma
                                add_so('A1g', minus(idx0, idx1))
                                add_so('A1u', plus (idx0, idx1))
                            else: # s-sigma, d-sigma
                                add_so('A1g', plus (idx0, idx1))
                                add_so('A1u', minus(idx0, idx1))
# +/-m
                            for m in range(1,angl+1):
                                idx0 = basoff[at0] + aoff + m
                                idy0 = basoff[at0] + aoff - m
                                idx1 = basoff[at1] + aoff + m
                                idy1 = basoff[at1] + aoff - m
                                if angl % 2: # odd parity
                                    add_so('E%dux'%m, plus (idx0, idx1))
                                    add_so('E%duy'%m, plus (idy0, idy1))
                                    add_so('E%dgx'%m, minus(idx0, idx1))
                                    add_so('E%dgy'%m, minus(idy0, idy1))
                                else:
                                    add_so('E%dgy'%m, plus (idy0, idy1))
                                    add_so('E%dgx'%m, plus (idx0, idx1))
                                    add_so('E%duy'%m, minus(idy0, idy1))
                                    add_so('E%dux'%m, minus(idx0, idx1))
                    ib += nc * degen
            elif len(atom_ids) == 1:
                at0 = atom_ids[0]
                symb = pyscf.gto.mole._symbol(atoms[at0][0])
                ib = 0
                if symb in basis_tab:
                    bas0 = basis_tab[symb]
                else:
                    bas0 = basis_tab[pyscf.gto.mole._rm_digit(symb)]
                for b in bas0:
                    angl = b[0]
                    nc = _num_contract(b)
                    degen = angl * 2 + 1
                    if angl == 1:
                        for i in range(nc):
                            aoff = ib + i*degen + angl
# m = 0
                            idx0 = basoff[at0] + aoff + 1
                            add_so('A1u', identity(idx0))
# m = +/- 1
                            idx0 = basoff[at0] + aoff - 1
                            idy0 = basoff[at0] + aoff
                            add_so('E1uy', identity(idy0))
                            add_so('E1ux', identity(idx0))
                    else:
                        for i in range(nc):
                            aoff = ib + i*degen + angl
                            idx0 = basoff[at0] + aoff
# m = 0
                            if angl % 2:
                                add_so('A1u', identity(idx0))
                            else:
                                add_so('A1g', identity(idx0))
# +/-m
                            for m in range(1,angl+1):
                                idx0 = basoff[at0] + aoff + m
                                idy0 = basoff[at0] + aoff - m
                                if angl % 2: # p, f functions
                                    add_so('E%dux'%m, identity(idx0))
                                    add_so('E%duy'%m, identity(idy0))
                                else: # d, g functions
                                    add_so('E%dgy'%m, identity(idy0))
                                    add_so('E%dgx'%m, identity(idx0))
                    ib += nc * degen
    elif gpname == 'Coov':
        for atom_ids in eql_atom_ids:
            at0 = atom_ids[0]
            symb = pyscf.gto.mole._symbol(atoms[at0][0])
            ib = 0
            if symb in basis_tab:
                bas0 = basis_tab[symb]
            else:
                bas0 = basis_tab[pyscf.gto.mole._rm_digit(symb)]
            for b in bas0:
                angl = b[0]
                nc = _num_contract(b)
                degen = angl * 2 + 1
                if angl == 1:
                    for i in range(nc):
                        aoff = ib + i*degen + angl
# m = 0
                        idx0 = basoff[at0] + aoff + 1
                        add_so('A1', identity(idx0))
# m = +/- 1
                        idx0 = basoff[at0] + aoff - 1
                        idy0 = basoff[at0] + aoff
                        add_so('E1x', identity(idx0))
                        add_so('E1y', identity(idy0))
                else:
                    for i in range(nc):
                        aoff = ib + i*degen + angl
                        idx0 = basoff[at0] + aoff
# m = 0
                        add_so('A1', identity(idx0))
# +/-m
                        for m in range(1,angl+1):
                            idx0 = basoff[at0] + aoff + m
                            idy0 = basoff[at0] + aoff - m
                            add_so('E%dx'%m, identity(idx0))
                            add_so('E%dy'%m, identity(idy0))
                ib += nc * degen

    so = []
    irrep_ids = []
    irrep_names = list(sodic.keys())
    for irname in irrep_names:
        irrep_ids.append(linearmole_irrep_symb2id(gpname, irname))
    idx = numpy.argsort(irrep_ids)
    for i in idx:
        so.append(numpy.vstack(sodic[irrep_names[i]]).T)
    irrep_ids = [irrep_ids[i] for i in idx]
    return so, irrep_ids


if __name__ == "__main__":
    h2o = pyscf.gto.Mole()
    h2o.verbose = 0
    h2o.output = None
    h2o.atom = [['O' , (1. , 0.    , 0.   ,)],
                [1   , (0. , -.757 , 0.587,)],
                [1   , (0. , 0.757 , 0.587,)] ]
    h2o.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    h2o.build()
    gpname, origin, axes = geom.detect_symm(h2o._atom)
    atoms = pyscf.gto.mole.format_atom(h2o._atom, origin, axes)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(gpname, eql_atoms, atoms, h2o._basis))

    mol = pyscf.gto.M(
        atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['H', (0,0,1)]],
        basis = 'ccpvtz', charge=1)
    gpname, orig, axes = geom.detect_symm(mol._atom)
    atoms = pyscf.gto.mole.format_atom(mol._atom, orig, axes)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(gpname, eql_atoms, atoms, mol._basis)[1])

    mol = pyscf.gto.M(
        atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['He', (0,0,1)]],
        basis = 'ccpvtz')
    gpname, orig, axes = geom.detect_symm(mol._atom)
    atoms = pyscf.gto.mole.format_atom(mol._atom, orig, axes)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(gpname, eql_atoms, atoms, mol._basis)[1])
