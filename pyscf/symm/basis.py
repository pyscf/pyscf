#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''
Generate symmetry adapted basis
'''

from functools import reduce
import numpy
from pyscf.data.elements import _symbol, _rm_digit
from pyscf import gto
from pyscf import lib
from pyscf.lib.exceptions import PointGroupSymmetryError
from pyscf.symm import geom
from pyscf.symm import param
from pyscf.symm.Dmatrix import Dmatrix, get_euler_angles

__all__ = ['tot_parity_odd',
           'symm_adapted_basis',
           'dump_symm_adapted_basis',
           'symmetrize_matrix',
           'linearmole_symm_descent',
           'linearmole_irrep_symb2id',
           'linearmole_irrep_id2symb',
           'linearmole_symm_adapted_basis',
           'so3_irrep_symb2id',
           'so3_irrep_id2symb']

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
        return (ox and gx) ^ (oy and gy) ^ (oz and gz)

def symm_adapted_basis(mol, gpname, orig=0, coordinates=None):
    if gpname == 'SO3':
        return so3_symm_adapted_basis(mol, gpname, orig, coordinates)
    elif gpname in ('Dooh', 'Coov'):
        return linearmole_symm_adapted_basis(mol, gpname, orig, coordinates)

    # prop_atoms are the atoms relocated wrt the charge center with proper
    # orientation
    if coordinates is None:
        coordinates = numpy.eye(3)
    prop_atoms = mol.format_atom(mol._atom, orig, coordinates, 'Bohr')
    eql_atom_ids = geom.symm_identical_atoms(gpname, prop_atoms)

    ops = numpy.asarray([param.D2H_OPS[op] for op in param.OPERATOR_TABLE[gpname]])
    chartab = numpy.array([x[1:] for x in param.CHARACTER_TABLE[gpname]])
    nirrep = chartab.__len__()
    aoslice = mol.aoslice_by_atom()
    nao = mol.nao_nr()
    atom_coords = numpy.array([a[1] for a in prop_atoms])

    sodic = [[] for i in range(8)]
    for atom_ids in eql_atom_ids:
        r0 = atom_coords[atom_ids[0]]
        op_coords = numpy.einsum('x,nxy->ny', r0, ops)
# Using ops to generate other atoms from atom_ids[0]
        coords0 = atom_coords[atom_ids]
        dc = abs(op_coords.reshape(-1,1,3) - coords0).max(axis=2)
        op_relate_idx = numpy.argwhere(dc < geom.TOLERANCE)[:,1]
        ao_loc = numpy.array([aoslice[atom_ids[i],2] for i in op_relate_idx])

        b0, b1 = aoslice[atom_ids[0],:2]
        ip = 0
        for ib in range(b0, b1):
            l = mol.bas_angular(ib)
            if mol.cart:
                degen = (l + 1) * (l + 2) // 2
                cbase = numpy.zeros((degen,nirrep,nao))
                for op_id, op in enumerate(ops):
                    n = 0
                    for x in range(l, -1, -1):
                        for y in range(l-x, -1, -1):
                            z = l-x-y
                            idx = ao_loc[op_id] + n
                            sign = op[0,0]**x * op[1,1]**y * op[2,2]**z
                            cbase[n,:,idx] += sign * chartab[:,op_id]
                            n += 1
            else:
                degen = l * 2 + 1
                cbase = numpy.zeros((degen,nirrep,nao))
                for op_id, op in enumerate(param.OPERATOR_TABLE[gpname]):
                    for n, m in enumerate(range(-l, l+1)):
                        idx = ao_loc[op_id] + n
                        if tot_parity_odd(op, l, m):
                            cbase[n,:,idx] -= chartab[:,op_id]
                        else:
                            cbase[n,:,idx] += chartab[:,op_id]
            norms = numpy.sqrt(numpy.einsum('mij,mij->mi', cbase, cbase))

            for i in range(mol.bas_nctr(ib)):
                for n, ir in numpy.argwhere(norms > 1e-12):
                    c = numpy.zeros(nao)
                    c[ip:] = cbase[n,ir,:nao-ip] / norms[n,ir]
                    sodic[ir].append(c)
                ip += degen

    ao_loc = mol.ao_loc_nr()
    l_idx = {}
    ANG_OF = 1
    for l in range(mol._bas[:,ANG_OF].max()+1):
        idx = [numpy.arange(ao_loc[ib], ao_loc[ib+1])
               for ib in numpy.where(mol._bas[:,ANG_OF] == l)[0]]
        if idx:
            l_idx[l] = numpy.hstack(idx)

    Ds = _momentum_rotation_matrices(mol, coordinates)
    so = []
    irrep_ids = []
    for ir, c in enumerate(sodic):
        if len(c) > 0:
            irrep_ids.append(ir)
            c_ir = numpy.vstack(c).T
            nso = c_ir.shape[1]
            for l, idx in l_idx.items():
                c = c_ir[idx].reshape(-1,Ds[l].shape[1],nso)
                c_ir[idx] = numpy.einsum('nm,smp->snp', Ds[l], c).reshape(-1,nso)

            so.append(c_ir)

    return so, irrep_ids

def _momentum_rotation_matrices(mol, axes):
    '''Cache the rotation matrices for each angular momentum'''
    alpha, beta, gamma = get_euler_angles(numpy.eye(3), axes)
    ANG_OF = 1
    l_max = mol._bas[:,ANG_OF].max()

    if not mol.cart:
        return [Dmatrix(l, alpha, beta, gamma, reorder_p=True)
                for l in range(l_max+1)]

    pp = Dmatrix(1, alpha, beta, gamma, reorder_p=True)

    Ds = [numpy.ones((1,1))]
    for l in range(1, l_max+1):
        # All possible x,y,z combinations
        cidx = numpy.sort(lib.cartesian_prod([(0, 1, 2)] * l), axis=1)

        addr = 0
        affine = numpy.ones((1,1))
        for i in range(l):
            nd = affine.shape[0] * 3
            affine = numpy.einsum('ik,jl->ijkl', affine, pp).reshape(nd, nd)
            addr = addr * 3 + cidx[:,i]

        uniq_addr, rev_addr = numpy.unique(addr, return_inverse=True)
        ncart = (l + 1) * (l + 2) // 2
        assert ncart == uniq_addr.size
        trans = numpy.zeros((ncart,ncart))
        for i, k in enumerate(rev_addr):
            trans[k] += affine[i,uniq_addr]
        Ds.append(trans)
    return Ds

def dump_symm_adapted_basis(mol, so):
    raise PointGroupSymmetryError('TODO')

def symmetrize_matrix(mat, so):
    return [reduce(numpy.dot, (c.conj().T,mat,c)) for c in so]

def _basis_offset_for_atoms(atoms, basis_tab):
    basoff = [0]
    n = 0
    for at in atoms:
        symb = _symbol(at[0])
        if symb in basis_tab:
            bas0 = basis_tab[symb]
        else:
            bas0 = basis_tab[_rm_digit(symb)]
        for b in bas0:
            angl = b[0]
            n += _num_contract(b) * (angl*2+1)
        basoff.append(n)
    return n, basoff

def _num_contract(basis):
    if isinstance(basis[1], int):
        # This branch should never be reached if basis_tab is formated by
        # function mole.format_basis
        nctr = len(basis[2]) - 1
    else:
        nctr = len(basis[1]) - 1
    return nctr

###############################
# SO3 (real spherical harmonics)
# Irreps ID maps
# SO3       ->  Dooh
# s       0 ->  A1g   0

# pz    105 ->  A1u   5
# py    106 ->  E1uy  6
# px    107 ->  E1ux  7

# dz2   200 ->  A1g   0
# dyz   203 ->  E1gy  3
# dxz   202 ->  E1gx  2
# dxy   211 ->  E2gy  11
# dx2y2 210 ->  E2gx  10

# f0    305 ->  A1u   5
# f-1   306 ->  E1uy  6
# f+1   307 ->  E1ux  7
# f-2   314 ->  E2uy  14
# f+2   315 ->  E2ux  15
# f-3   316 ->  E3uy  16
# f+3   317 ->  E3ux  17

# g0    400 ->  A1g   0
# g-1   403 ->  E1gy  3
# g+1   402 ->  E1gx  2
# g-2   411 ->  E2gy  11
# g+2   410 ->  E2gx  10
# g-3   413 ->  E3gy  13
# g+3   412 ->  E3gx  12
# g-4   421 ->  E4gy  21
# g+4   420 ->  E4gx  20
_SO3_SYMB2ID = {
    's+0' :     0,
    'p-1':    106,
    'p+0':    105,
    'p+1':    107,
    'd-2':    211,
    'd-1':    203,
    'd+0':    200,
    'd+1':    202,
    'd+2':    210,
    'f-3':    316,
    'f-2':    314,
    'f-1':    306,
    'f+0':    305,
    'f+1':    307,
    'f+2':    315,
    'f+3':    317,
    'g-4':    421,
    'g-3':    413,
    'g-2':    411,
    'g-1':    403,
    'g+0':    400,
    'g+1':    402,
    'g+2':    410,
    'g+3':    412,
    'g+4':    420,
    'h-5':    526,
    'h-4':    524,
    'h-3':    516,
    'h-2':    514,
    'h-1':    506,
    'h+0':    505,
    'h+1':    507,
    'h+2':    515,
    'h+3':    517,
    'h+4':    525,
    'h+5':    527,
    'i-6':    631,
    'i-5':    623,
    'i-4':    621,
    'i-3':    613,
    'i-2':    611,
    'i-1':    603,
    'i+0':    600,
    'i+1':    602,
    'i+2':    610,
    'i+3':    612,
    'i+4':    620,
    'i+5':    622,
    'i+6':    630,
}
_SO3_ID2SYMB = dict([(v, k) for k, v in _SO3_SYMB2ID.items()])
_ANGULAR = 'spdfghiklmnortu'

def so3_irrep_symb2id(symb):
    symb = symb.lower()
    if symb in _SO3_SYMB2ID:
        return _SO3_SYMB2ID[symb]
    else:
        s = ''.join([i for i in symb if i.isalpha()])
        n = int(''.join([i for i in symb if not i.isalpha()]))
        e2 = abs(n) // 2
        l = _ANGULAR.index(s)
        if abs(n) > l:
            raise KeyError(symb)
        gu = 'u' if l % 2 else 'g'
        xy = 'y' if n < 0 else 'x'
        parity = '_odd' if n % 2 else '_even'
        return l * 100 + e2 * 10 + DOOH_IRREP_ID_TABLE[parity + gu + xy]

def so3_irrep_id2symb(irrep_id):
    if irrep_id in _SO3_ID2SYMB:
        return _SO3_ID2SYMB[irrep_id]
    else:
        l, e2 = divmod(irrep_id, 100)
        e2, n = divmod(e2, 10)
        if n in (0, 1, 5, 4):
            rn = e2 * 2
        elif n in (2, 3, 6, 7):
            rn = e2 * 2 + 1
        else:
            raise KeyError(f'Unknown Irrep ID {irrep_id} for SO3 group')
        if l <= 1 or rn > l:
            raise KeyError(f'Unknown Irrep ID {irrep_id} for SO3 group')
        if n in (6, 4, 3, 1):
            rn = -rn
        return f'{_ANGULAR[l]}{rn:+d}'

def so3_symm_adapted_basis(mol, gpname, orig=0, coordinates=None):
    assert gpname == 'SO3'
    assert mol.natm == 1

    ao_loc = mol.ao_loc_nr(cart=False)
    nao_sph = ao_loc[-1]

    if mol.cart:
        coeff = mol.cart2sph(normalized='sp')
    else:
        coeff = numpy.eye(nao_sph)
    nao = coeff.shape[0]

    lmax = max(mol._bas[:,gto.ANG_OF])
    so = []
    irrep_names = []
    for l in range(lmax+1):
        bas_idx = mol._bas[:,gto.ANG_OF] == l
        if sum(bas_idx) == 0: # Skip any unused angular momentum channels
            continue
        cs = [coeff[:,p0:p1]
              for p0, p1 in zip(ao_loc[:-1][bas_idx], ao_loc[1:][bas_idx])]
        c_groups = numpy.hstack(cs).reshape(nao, -1, l*2+1)
        if l == 1:
            so.extend([c_groups[:,:,1], c_groups[:,:,2], c_groups[:,:,0]])
            irrep_names.extend(['p-1', 'p+0', 'p+1'])
        else:
            for m in range(-l, l+1):
                so.append(c_groups[:,:,l+m])
                irrep_names.append('%s%+d' % (_ANGULAR[l], m))

    irrep_ids = [so3_irrep_symb2id(ir) for ir in irrep_names]
    return so, irrep_ids


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

def linearmole_symm_descent(gpname, irrep_id):
    '''Map irreps to D2h or C2v'''
    if gpname in ('Dooh', 'Coov'):
        return irrep_id % 10
    else:
        raise PointGroupSymmetryError('%s is not proper for linear molecule.' % gpname)

def linearmole_irrep_symb2id(gpname, symb):
    if gpname == 'Dooh':
        if symb in DOOH_IRREP_ID_TABLE:
            return DOOH_IRREP_ID_TABLE[symb]
        else:
            try:
                n = int(''.join([i for i in symb if i.isdigit()]))
                if n % 2:
                    return (n//2)*10 + DOOH_IRREP_ID_TABLE['_odd'+symb[-2:]]
                else:
                    return (n//2)*10 + DOOH_IRREP_ID_TABLE['_even'+symb[-2:]]
            except (KeyError, ValueError):
                raise PointGroupSymmetryError(f'Incorrect Dooh irrep {symb}')
    elif gpname == 'Coov':
        if symb in COOV_IRREP_ID_TABLE:
            return COOV_IRREP_ID_TABLE[symb]
        else:
            if 'g' in symb or 'u' in symb:
                raise PointGroupSymmetryError(f'Incorrect Coov irrep {symb}')
            try:
                n = int(''.join([i for i in symb if i.isdigit()]))
                if n % 2:
                    return (n//2)*10 + COOV_IRREP_ID_TABLE['_odd'+symb[-1]]
                else:
                    return (n//2)*10 + COOV_IRREP_ID_TABLE['_even'+symb[-1]]
            except (KeyError, ValueError):
                raise PointGroupSymmetryError(f'Incorrect Coov irrep {symb}')
    else:
        raise PointGroupSymmetryError(f'Incorrect cylindrical symmetry group {gpname}')

DOOH_IRREP_SYMBS = ('A1g' , 'A2g' , 'E1gx', 'E1gy' , 'A2u', 'A1u' , 'E1uy', 'E1ux')
DOOH_IRREP_SYMBS_EXT = ('gx' , 'gy' , 'gx', 'gy' , 'uy', 'ux' , 'uy', 'ux')
COOV_IRREP_SYMBS = ('A1' , 'A2' , 'E1x', 'E1y')
def linearmole_irrep_id2symb(gpname, irrep_id):
    if gpname == 'Dooh':
        if irrep_id < 10:
            return DOOH_IRREP_SYMBS[irrep_id]
        else:
            l = abs(linearmole_irrep2momentum(irrep_id))
            n = irrep_id % 10
            return 'E%d%s' % (l, DOOH_IRREP_SYMBS_EXT[n])
    elif gpname == 'Coov':
        if irrep_id < 10:
            return COOV_IRREP_SYMBS[irrep_id]
        else:
            l = abs(linearmole_irrep2momentum(irrep_id))
            n = irrep_id % 10
            if n >= 4:
                raise PointGroupSymmetryError(f'Incorrect Coov irrep {irrep_id}')
            if n % 2:
                xy = 'y'
            else:
                xy = 'x'
            return 'E%d%s' % (l, xy)
    else:
        raise PointGroupSymmetryError(f'Incorrect cylindrical symmetry group {gpname}')

def linearmole_irrep2momentum(irrep_id):
    if irrep_id % 10 in (0, 1, 5, 4):
        l = irrep_id // 10 * 2
    else:
        l = irrep_id // 10 * 2 + 1
    if irrep_id % 10 in (1, 3, 4, 6):  # Ey
        l *= -1
    return l

def linearmole_symm_adapted_basis(mol, gpname, orig=0, coordinates=None):
    assert (gpname in ('Dooh', 'Coov'))
    assert (not mol.cart)

    if coordinates is None:
        coordinates = numpy.eye(3)
    prop_atoms = mol.format_atom(mol._atom, orig, coordinates, 'Bohr')
    eql_atom_ids = geom.symm_identical_atoms(gpname, prop_atoms)

    aoslice = mol.aoslice_by_atom()
    basoff = aoslice[:,2]
    nao = mol.nao_nr()
    sodic = {}
    shalf = numpy.sqrt(.5)
    def plus(i0, i1):
        c = numpy.zeros(nao)
        c[i0] = c[i1] = shalf
        return c
    def minus(i0, i1):
        c = numpy.zeros(nao)
        c[i0] = shalf
        c[i1] =-shalf
        return c
    def identity(i0):
        c = numpy.zeros(nao)
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
                ip = 0
                b0, b1, p0, p1 = aoslice[at0]
                for ib in range(b0, b1):
                    angl = mol.bas_angular(ib)
                    nc = mol.bas_nctr(ib)
                    degen = angl * 2 + 1
                    if angl == 1:
                        for i in range(nc):
                            aoff = ip + i*degen + angl
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
                            aoff = ip + i*degen + angl
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
                    ip += nc * degen
            elif len(atom_ids) == 1:
                at0 = atom_ids[0]
                ip = 0
                b0, b1, p0, p1 = aoslice[at0]
                for ib in range(b0, b1):
                    angl = mol.bas_angular(ib)
                    nc = mol.bas_nctr(ib)
                    degen = angl * 2 + 1
                    if angl == 1:
                        for i in range(nc):
                            aoff = ip + i*degen + angl
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
                            aoff = ip + i*degen + angl
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
                    ip += nc * degen
    elif gpname == 'Coov':
        for atom_ids in eql_atom_ids:
            at0 = atom_ids[0]
            ip = 0
            b0, b1, p0, p1 = aoslice[at0]
            for ib in range(b0, b1):
                angl = mol.bas_angular(ib)
                nc = mol.bas_nctr(ib)
                degen = angl * 2 + 1
                if angl == 1:
                    for i in range(nc):
                        aoff = ip + i*degen + angl
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
                        aoff = ip + i*degen + angl
                        idx0 = basoff[at0] + aoff
# m = 0
                        add_so('A1', identity(idx0))
# +/-m
                        for m in range(1,angl+1):
                            idx0 = basoff[at0] + aoff + m
                            idy0 = basoff[at0] + aoff - m
                            add_so('E%dx'%m, identity(idx0))
                            add_so('E%dy'%m, identity(idy0))
                ip += nc * degen

    irrep_ids = []
    irrep_names = list(sodic.keys())
    for irname in irrep_names:
        irrep_ids.append(linearmole_irrep_symb2id(gpname, irname))
    irrep_idx = numpy.argsort(irrep_ids)
    irrep_ids = [irrep_ids[i] for i in irrep_idx]

    ao_loc = mol.ao_loc_nr()
    l_idx = {}
    ANG_OF = 1
    for l in range(mol._bas[:,ANG_OF].max()+1):
        idx = [numpy.arange(ao_loc[ib], ao_loc[ib+1])
               for ib in numpy.where(mol._bas[:,ANG_OF] == l)[0]]
        if idx:
            l_idx[l] = numpy.hstack(idx)

    Ds = _momentum_rotation_matrices(mol, coordinates)
    so = []
    for i in irrep_idx:
        c_ir = numpy.vstack(sodic[irrep_names[i]]).T
        nso = c_ir.shape[1]
        for l, idx in l_idx.items():
            c = c_ir[idx].reshape(-1,Ds[l].shape[1],nso)
            c_ir[idx] = numpy.einsum('nm,smp->snp', Ds[l], c).reshape(-1,nso)

        so.append(c_ir)

    return so, irrep_ids


if __name__ == "__main__":
    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None
    h2o.atom = [['O' , (1. , 0.    , 0.   ,)],
                [1   , (0. , -.757 , 0.587,)],
                [1   , (0. , 0.757 , 0.587,)] ]
    h2o.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    h2o.build()
    gpname, origin, axes = geom.detect_symm(h2o._atom)
    atoms = gto.format_atom(h2o._atom, origin, axes)
    h2o.build(False, False, atom=atoms)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(h2o, gpname, eql_atoms)[1])

    mol = gto.M(
        atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['H', (0,0,1)]],
        basis = 'ccpvtz', charge=1)
    gpname, orig, axes = geom.detect_symm(mol._atom)
    atoms = gto.format_atom(mol._atom, orig, axes)
    mol.build(False, False, atom=atoms)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(mol, gpname, eql_atoms)[1])

    mol = gto.M(
        atom = [['H', (0,0,0)], ['H', (0,0,-1)], ['He', (0,0,1)]],
        basis = 'ccpvtz')
    gpname, orig, axes = geom.detect_symm(mol._atom)
    atoms = gto.format_atom(mol._atom, orig, axes)
    mol.build(False, False, atom=atoms)
    print(gpname)
    eql_atoms = geom.symm_identical_atoms(gpname, atoms)
    print(symm_adapted_basis(mol, gpname, eql_atoms)[1])
