#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
Mole class and helper functions to handle parameters and attributes for GTO
integrals. This module serves the interface to the integral library libcint.
'''

import os
import sys
import types
import re

import json
import ctypes
import numpy
import numpy as np
import h5py
import scipy.special
import scipy.linalg
import contextlib
import threading
from pyscf import lib
from pyscf.lib import param
from pyscf.data import elements
from pyscf.lib import logger
from pyscf.gto import cmd_args
from pyscf.gto import basis
from pyscf.gto import moleintor
from pyscf.gto.eval_gto import eval_gto
from pyscf.gto.ecp import core_configuration
from pyscf import __config__

from pyscf.data.elements import ELEMENTS, ELEMENTS_PROTON, \
        _rm_digit, charge, _symbol, _std_symbol, _atom_symbol, is_ghost_atom, \
        _std_symbol_without_ghost

from pyscf.lib.exceptions import BasisNotFoundError, PointGroupSymmetryError
import warnings


# for _atm, _bas, _env
CHARGE_OF  = 0
PTR_COORD  = 1
NUC_MOD_OF = 2
PTR_ZETA   = 3
PTR_FRAC_CHARGE = 4
PTR_RADIUS = 5
ATM_SLOTS  = 6
ATOM_OF    = 0
ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
RADI_POWER = 3 # for ECP
KAPPA_OF   = 4
SO_TYPE_OF = 4 # for ECP
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8
# pointer to env
PTR_EXPCUTOFF   = 0
PTR_COMMON_ORIG = 1
PTR_RINV_ORIG   = 4
PTR_RINV_ZETA   = 7
PTR_RANGE_OMEGA = 8
PTR_F12_ZETA    = 9
PTR_GTG_ZETA    = 10
NGRIDS          = 11
PTR_GRIDS       = 12
AS_RINV_ORIG_ATOM = 17
AS_ECPBAS_OFFSET = 18
AS_NECPBAS      = 19
PTR_ENV_START   = 20
# parameters from libcint
NUC_POINT = 1
NUC_GAUSS = 2
# nucleus with fractional charges. It can be used to mimic MM particles
NUC_FRAC_CHARGE = 3
NUC_ECP = 4  # atoms with pseudo potential

BASE = getattr(__config__, 'BASE', 0)
NORMALIZE_GTO = getattr(__config__, 'NORMALIZE_GTO', True)
DISABLE_EVAL = getattr(__config__, 'DISABLE_EVAL', False)
ARGPARSE = getattr(__config__, 'ARGPARSE', False)
DUMPINPUT = getattr(__config__, 'DUMPINPUT', True)

with open(os.path.abspath(os.path.join(__file__, '..', 'basis', 'bse_meta.json')), 'r') as f:
    BSE_META = json.load(f)
del f

def M(*args, **kwargs):
    r'''This is a shortcut to build up Mole object.

    Args: Same to :func:`Mole.build`

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    '''
    mol = Mole()
    mol.build(*args, **kwargs)
    return mol

def gaussian_int(n, alpha):
    r'''int_0^inf x^n exp(-alpha x^2) dx'''
    n1 = (n + 1) * .5
    return scipy.special.gamma(n1) / (2. * alpha**n1)

def gto_norm(l, expnt):
    r'''Normalized factor for GTO radial part   :math:`g=r^l e^{-\alpha r^2}`

    .. math::

        \frac{1}{\sqrt{\int g^2 r^2 dr}}
        = \sqrt{\frac{2^{2l+3} (l+1)! (2a)^{l+1.5}}{(2l+2)!\sqrt{\pi}}}

    Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant.  Chem., 54(1995), 83-87.

    Args:
        l (int):
            angular momentum
        expnt :
            exponent :math:`\alpha`

    Returns:
        normalization factor

    Examples:

    >>> print(gto_norm(0, 1))
    2.5264751109842591
    '''
    if numpy.all(l >= 0):
        #f = 2**(2*l+3) * math.factorial(l+1) * (2*expnt)**(l+1.5) \
        #        / (math.factorial(2*l+2) * math.sqrt(math.pi))
        #return math.sqrt(f)
        return 1/numpy.sqrt(gaussian_int(l*2+2, 2*expnt))
    else:
        raise ValueError('l should be >= 0')

def cart2sph(l, c_tensor=None, normalized=None):
    '''
    Cartesian to real spherical transformation matrix

    Kwargs:
        normalized :
            How the Cartesian GTOs are normalized.  'sp' means the s and p
            functions are normalized (this is the convention used by libcint
            library).
    '''
    nf = (l+1)*(l+2)//2
    if c_tensor is None:
        c_tensor = numpy.eye(nf)
    else:
        c_tensor = numpy.asarray(c_tensor, order='F').reshape(-1,nf)
    if l == 0 or l == 1:
        if normalized == 'sp':
            return c_tensor
        elif l == 0:
            return c_tensor * 0.282094791773878143
        else:
            return c_tensor * 0.488602511902919921
    else:
        assert l <= 15
        nd = l * 2 + 1
        ngrid = c_tensor.shape[0]
        c2sph = numpy.zeros((ngrid,nd), order='F')
        fn = moleintor.libcgto.CINTc2s_ket_sph
        fn(c2sph.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrid),
           c_tensor.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(l))
        return c2sph

def cart2spinor_kappa(kappa, l=None, normalized=None):
    '''Cartesian to spinor transformation matrix for kappa

    Kwargs:
        normalized :
            How the Cartesian GTOs are normalized.  'sp' means the s and p
            functions are normalized (this is the convention used by libcint
            library).
    '''
    if kappa < 0:
        l = -kappa - 1
        nd = l * 2 + 2
    elif kappa > 0:
        l = kappa
        nd = l * 2
    else:
        assert (l is not None)
        assert (l <= 12)
        nd = l * 4 + 2
    nf = (l+1)*(l+2)//2
    c2smat = numpy.zeros((nf*2,nd), order='F', dtype=numpy.complex128)
    cmat = numpy.eye(nf)
    fn = moleintor.libcgto.CINTc2s_ket_spinor_sf1
    fn(c2smat.ctypes.data_as(ctypes.c_void_p),
       c2smat[nf:].ctypes.data_as(ctypes.c_void_p),
       cmat.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nf*2), ctypes.c_int(nf),
       ctypes.c_int(1), ctypes.c_int(kappa), ctypes.c_int(l))
    if normalized != 'sp':
        if l == 0:
            c2smat *= 0.282094791773878143
        elif l == 1:
            c2smat *= 0.488602511902919921
    # c2smat[0] is the transformation for spin up
    # c2smat[1] is the transformation for spin down
    c2smat = c2smat.reshape(2,nf,nd)
    return c2smat
cart2j_kappa = cart2spinor_kappa

def cart2spinor_l(l, normalized=None):
    '''Cartesian to spinor transformation matrix for angular moment l

    Kwargs:
        normalized :
            How the Cartesian GTOs are normalized.  'sp' means the s and p
            functions are normalized (this is the convention used by libcint
            library).
    '''
    return cart2spinor_kappa(0, l, normalized)
cart2j_l = cart2spinor_l

def sph2spinor_kappa(kappa, l=None):
    '''Real spherical to spinor transformation matrix for kappa'''
    from pyscf.symm.sph import sph2spinor
    ua, ub = sph2spinor(l)
    if kappa < 0:
        l = -kappa - 1
        ua = ua[:,l*2:]
        ub = ub[:,l*2:]
    elif kappa > 0:
        l = kappa
        ua = ua[:,:l*2]
        ub = ub[:,:l*2]
    else:
        assert (l is not None)
        assert (l <= 12)
    return ua, ub

def sph2spinor_l(l):
    '''Real spherical to spinor transformation matrix for angular moment l'''
    return sph2spinor_kappa(0, l)

def ao_rotation_matrix(mol, orientation):
    '''Matrix u to rotate AO basis to a new orientation.

    atom_new_coords = mol.atom_coords().dot(orientation.T)
    new_AO = u * mol.AO
    new_orbitals_coef = u.dot(orbitals_coef)
    '''
    from pyscf.symm.basis import _momentum_rotation_matrices
    Ds = _momentum_rotation_matrices(mol, orientation)
    u = []
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        u.extend([Ds[l]] * nc)
    return scipy.linalg.block_diag(*u)

def atom_types(atoms, basis=None, magmom=None):
    '''symmetry inequivalent atoms'''
    atmgroup = {}
    for ia, a in enumerate(atoms):
        if 'GHOST' in a[0].upper():
            a = ['X'+a[0][5:]] + list(a[1:])
        if a[0] in atmgroup:
            atmgroup[a[0]].append(ia)
        elif basis is None:
            atmgroup[a[0]] = [ia]
        else:
            stdsymb = _std_symbol(a[0])
            if a[0] in basis:
                if stdsymb in basis and basis[a[0]] == basis[stdsymb]:
                    if stdsymb in atmgroup:
                        atmgroup[stdsymb].append(ia)
                    else:
                        atmgroup[stdsymb] = [ia]
                else:
                    atmgroup[a[0]] = [ia]
            elif stdsymb in atmgroup:
                atmgroup[stdsymb].append(ia)
            else:
                atmgroup[stdsymb] = [ia]

    if magmom is not None:
        atmgroup_new = {}
        suffix = {1:'u', -1:'d', 0:'o'}
        magmom = np.asarray(magmom)
        for elem, idx in atmgroup.items():
            uniq_mag = np.unique(magmom[idx])
            if len(uniq_mag) > 1:
                for i, mag in enumerate(uniq_mag):
                    subgrp = np.asarray(idx)[np.where(magmom[idx] == mag)[0]]
                    if mag not in suffix:
                        raise RuntimeError("Magmom should be chosen from [-1, 0, 1], but %s is given" % mag)
                    atmgroup_new[elem+'_'+suffix[mag]] = subgrp.tolist()
            else:
                atmgroup_new[elem] = idx
        atmgroup = atmgroup_new
    return atmgroup


def format_atom(atoms, origin=0, axes=None,
                unit=getattr(__config__, 'UNIT', 'Ang')):
    '''Convert the input :attr:`Mole.atom` to the internal data format.
    Including, changing the nuclear charge to atom symbol, converting the
    coordinates to AU, rotate and shift the molecule.
    If the :attr:`~Mole.atom` is a string, it takes ";" and "\\n"
    for the mark to separate atoms;  "," and arbitrary length of blank space
    to separate the individual terms for an atom.  Blank line will be ignored.

    Args:
        atoms : list or str
            the same to :attr:`Mole.atom`

    Kwargs:
        origin : ndarray
            new axis origin.
        axes : ndarray
            (new_x, new_y, new_z), new coordinates
        unit : str or number
            If unit is one of strings (B, b, Bohr, bohr, AU, au), the coordinates
            of the input atoms are the atomic unit;  If unit is one of strings
            (A, a, Angstrom, angstrom, Ang, ang), the coordinates are in the
            unit of angstrom;  If a number is given, the number are considered
            as the Bohr value (in angstrom), which should be around 0.53.
            Set unit=1 if wishing to preserve the unit of the coordinates.

    Returns:
        "atoms" in the internal format. The internal format is
            | atom = [[atom1, (x, y, z)],
            |         [atom2, (x, y, z)],
            |         ...
            |         [atomN, (x, y, z)]]

    Examples:

    >>> gto.format_atom('9,0,0,0; h@1 0 0 1', origin=(1,1,1))
    [['F', [-1.0, -1.0, -1.0]], ['H@1', [-1.0, -1.0, 0.0]]]
    >>> gto.format_atom(['9,0,0,0', (1, (0, 0, 1))], origin=(1,1,1))
    [['F', [-1.0, -1.0, -1.0]], ['H', [-1, -1, 0]]]
    '''
    def str2atm(line):
        dat = line.split()
        try:
            coords = [float(x) for x in dat[1:4]]
        except ValueError:
            if DISABLE_EVAL:
                raise ValueError('Failed to parse geometry %s' % line)
            else:
                coords = list(eval(','.join(dat[1:4])))
        if len(coords) != 3:
            raise ValueError('Coordinates error in %s' % line)
        return [_atom_symbol(dat[0]), coords]

    if isinstance(atoms, str):
        # The input atoms points to a geometry file
        if os.path.isfile(atoms):
            try:
                atoms = fromfile(atoms)
            except ValueError:
                sys.stderr.write('\nFailed to parse geometry file  %s\n\n' % atoms)
                raise

        atoms = atoms.replace(';','\n').replace(',',' ').replace('\t',' ')
        fmt_atoms = []
        for dat in atoms.split('\n'):
            dat = dat.strip()
            if dat and dat[0] != '#':
                fmt_atoms.append(dat)

        if len(fmt_atoms[0].split()) < 4:
            fmt_atoms = from_zmatrix('\n'.join(fmt_atoms))
        else:
            fmt_atoms = [str2atm(line) for line in fmt_atoms]
    else:
        fmt_atoms = []
        for atom in atoms:
            if isinstance(atom, str):
                if atom.lstrip()[0] != '#':
                    fmt_atoms.append(str2atm(atom.replace(',',' ')))
            else:
                if isinstance(atom[1], (int, float)):
                    fmt_atoms.append([_atom_symbol(atom[0]), atom[1:4]])
                else:
                    fmt_atoms.append([_atom_symbol(atom[0]), atom[1]])

    if len(fmt_atoms) == 0:
        return []

    if axes is None:
        axes = numpy.eye(3)

    if isinstance(unit, str):
        if is_au(unit):
            unit = 1.
        else:
            unit = 1./param.BOHR
    else:
        unit = 1./unit

    c = numpy.array([a[1] for a in fmt_atoms], dtype=numpy.double)
    c = numpy.einsum('ix,kx->ki', axes * unit, c - origin)
    z = [a[0] for a in fmt_atoms]
    return list(zip(z, c.tolist()))

#TODO: sort exponents
def format_basis(basis_tab, sort_basis=True):
    '''Convert the input :attr:`Mole.basis` to the internal data format.

    ``{ atom: [(l, ((-exp, c_1, c_2, ..),
                    (-exp, c_1, c_2, ..))),
               (l, ((-exp, c_1, c_2, ..),
                    (-exp, c_1, c_2, ..)))], ... }``

    Args:
        basis_tab : dict
            Similar to :attr:`Mole.basis`, it **cannot** be a str

    Returns:
        Formatted :attr:`~Mole.basis`

    Examples:

    >>> gto.format_basis({'H':'sto-3g', 'H^2': '3-21g'})
    {'H': [[0,
        [3.4252509099999999, 0.15432897000000001],
        [0.62391373000000006, 0.53532813999999995],
        [0.16885539999999999, 0.44463454000000002]]],
     'H^2': [[0,
        [5.4471780000000001, 0.15628500000000001],
        [0.82454700000000003, 0.90469100000000002]],
        [0, [0.18319199999999999, 1.0]]]}

    >>> gto.format_basis({'H':'gth-szv'})
    {'H': [[0,
        (8.3744350009, -0.0283380461),
        (1.8058681460, -0.1333810052),
        (0.4852528328, -0.3995676063),
        (0.1658236932, -0.5531027541)]]}
    '''
    basis_converter = _generate_basis_converter()
    fmt_basis = {}
    for atom, atom_basis in basis_tab.items():
        symb = _atom_symbol(atom)
        _basis = basis_converter(symb, atom_basis)
        if len(_basis) == 0:
            raise BasisNotFoundError('Basis not found for  %s' % symb)

        # Sort basis according to angular momentum. This is important for method
        # decontract_basis, which assumes that basis functions with the same
        # angular momentum are grouped together. Related to issue #1620 #1770
        if sort_basis:
            _basis = sorted([b for b in _basis if b], key=lambda b: b[0])
        fmt_basis[symb] = _basis
    return fmt_basis

def _generate_basis_converter():
    def nparray_to_list(item):
        val = []
        for x in item:
            if isinstance(x, (tuple, list)):
                val.append(nparray_to_list(x))
            elif isinstance(x, numpy.ndarray):
                val.append(x.tolist())
            else:
                val.append(x)
        return val

    def load(basis_name, symb):
        unc = basis_name.lower().startswith('unc')
        if unc:
            basis_name = basis_name[3:]
        if 'gth' in basis_name:
            from pyscf.pbc.gto.basis import load as pbc_basis_load
            _basis = pbc_basis_load(basis_name, symb)
        else:
            _basis = basis.load(basis_name, symb)
        if unc:
            _basis = uncontracted_basis(_basis)
        return _basis

    def converter(symb, raw_basis):
        if isinstance(raw_basis, str):
            _basis = load(raw_basis, _std_symbol_without_ghost(symb))
        elif (any(isinstance(x, str) for x in raw_basis)
              # The first element is the basis of internal format
              or not isinstance(raw_basis[0][0], (numpy.integer, int))):
            stdsymb = _std_symbol_without_ghost(symb)
            _basis = []
            for rawb in raw_basis:
                if isinstance(rawb, str):
                    _basis.extend(load(rawb, stdsymb))
                else:
                    _basis.extend(nparray_to_list(rawb))
        else:
            _basis = nparray_to_list(raw_basis)
        return _basis
    return converter

def uncontracted_basis(_basis):
    '''Uncontract internal format _basis

    Examples:

    >>> gto.uncontract(gto.load('sto3g', 'He'))
    [[0, [6.36242139, 1]], [0, [1.158923, 1]], [0, [0.31364979, 1]]]
    '''
    MAXL = 10
    ubasis_raw = [[] for l in range(MAXL)]
    ubasis_exp = [[] for l in range(MAXL)]
    for b in _basis:
        angl = b[0]
        kappa = b[1]
        if isinstance(kappa, int):
            coeffs = b[2:]
        else:
            coeffs = b[1:]

        if isinstance(kappa, int) and kappa != 0:
            warnings.warn('For basis with kappa != 0, the uncontract basis might be wrong. '
                          'Please double check the resultant attribute mol._basis')
            for p in coeffs:
                ubasis_raw[angl].append([angl, kappa, [p[0], 1]])
                ubasis_exp[angl].append(p[0])
        else:
            for p in coeffs:
                ubasis_raw[angl].append([angl, [p[0], 1]])
                ubasis_exp[angl].append(p[0])

    # Check linear dependency
    ubasis = []
    for l in range(MAXL):
        basis_l = ubasis_raw[l]
        if basis_l:
            es = numpy.array(ubasis_exp[l])
            # Remove duplicated primitive basis functions
            es, e_idx = numpy.unique(es.round(9), True)
            # from large exponent to small exponent
            for i in reversed(e_idx):
                ubasis.append(basis_l[i])
    return ubasis
uncontract = uncontracted_basis
contract = contracted_basis = basis.to_general_contraction

def to_uncontracted_cartesian_basis(mol):
    '''Decontract the basis of a Mole or a Cell.  Returns a Mole (Cell) object
    with uncontracted Cartesian basis and a list of coefficients that
    transform the uncontracted basis to the original basis. Each element in
    the coefficients list corresponds to one shell of the original Mole (Cell).

    Examples:

    >>> mol = gto.M(atom='Ne', basis='ccpvdz')
    >>> pmol, ctr_coeff = mol.to_uncontracted_cartesian_basis()
    >>> c = scipy.linalg.block_diag(*ctr_coeff)
    >>> s = reduce(numpy.dot, (c.T, pmol.intor('int1e_ovlp'), c))
    >>> abs(s-mol.intor('int1e_ovlp')).max()
    0.0
    '''
    return decontract_basis(mol, to_cart=True)

def decontract_basis(mol, atoms=None, to_cart=False, aggregate=False):
    '''Decontract the basis of a Mole or a Cell.  Returns a Mole (Cell) object
    with the uncontracted basis environment and a list of coefficients that
    transform the uncontracted basis to the original basis. Each element in
    the coefficients list corresponds to one shell of the original Mole (Cell).

    Kwargs:
        atoms: list or str
            Atoms on which the basis to be decontracted. By default, all basis
            are decontracted
        to_cart: bool
            Decontract basis and transfer to Cartesian basis
        aggregate: bool
            Whether to aggregate the transformation coefficients into a giant
            transformation matrix

    Examples:

    >>> mol = gto.M(atom='Ne', basis='ccpvdz')
    >>> pmol, ctr_coeff = mol.decontract_basis()
    >>> c = scipy.linalg.block_diag(*ctr_coeff)
    >>> s = reduce(numpy.dot, (c.T, pmol.intor('int1e_ovlp'), c))
    >>> abs(s-mol.intor('int1e_ovlp')).max()
    0.0
    '''
    pmol = mol.copy(deep=False)

    # Some input basis may be segmented basis from a general contracted set.
    # This may lead to duplicated pGTOs. First contract all basis to remove
    # duplicated primitive functions.
    bas_exps = mol.bas_exps()
    def _to_full_contraction(mol, bas_idx):
        es = numpy.hstack([bas_exps[i] for i in bas_idx])
        _, e_idx, rev_idx = numpy.unique(es.round(9), True, True)
        if aggregate:
            cs = scipy.linalg.block_diag(
                *[mol._libcint_ctr_coeff(i) for i in bas_idx])
            if len(es) != len(e_idx):
                cs_new = numpy.zeros((e_idx.size, cs.shape[1]))
                for i, j in enumerate(rev_idx):
                    cs_new[j] += cs[i]
                es = es[e_idx][::-1]
                cs = cs_new[::-1]
            yield es, cs
        else:
            if len(es) != len(e_idx):
                raise RuntimeError('Duplicated pGTOs across shells')
            for i in bas_idx:
                yield bas_exps[i], mol._libcint_ctr_coeff(i)

    _bas = []
    env = [mol._env.copy()]
    contr_coeff = []
    pexp = env[0].size

    lmax = mol._bas[:,ANG_OF].max()
    if mol.cart:
        c2s = [numpy.eye((l+1)*(l+2)//2) for l in range(lmax+1)]
    elif to_cart:
        c2s = [cart2sph(l, normalized='sp') for l in range(lmax+1)]
        pmol.cart = True
    else:
        c2s = [numpy.eye(l*2+1) for l in range(lmax+1)]

    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        if ib0 == ib1: # No basis on atom ia
            continue

        if atoms is not None:
            if isinstance(atoms, str):
                to_apply = ((atoms == mol.atom_pure_symbol(ia)) or
                            (atoms == mol.atom_symbol(ia)))
            elif isinstance(atoms, (tuple, list)):
                to_apply = ((mol.atom_pure_symbol(ia) in atoms) or
                            (mol.atom_symbol(ia) in atoms) or
                            (ia in atoms))
            else:
                to_apply = True
            if not to_apply:
                for ib in range(ib0, ib1):
                    l = mol.bas_angular(ib)
                    nc = mol.bas_nctr(ib)
                    c = numpy.einsum('pi,xm->pxim', numpy.eye(nc), c2s[l])
                    contr_coeff.append(c.reshape(nc * c2s[l].shape[0], -1))
                _bas.append(mol._bas[ib0:ib1])
                continue

        lmax = mol._bas[ib0:ib1,ANG_OF].max()

        for l in range(lmax+1):
            bas_idx = ib0 + numpy.where(mol._bas[ib0:ib1,ANG_OF] == l)[0]
            if len(bas_idx) == 0:
                continue
            if bas_idx[0] + len(bas_idx) != bas_idx[-1] + 1:
                raise NotImplementedError('Discontinuous bases of same angular momentum')

            for mol_exps, b_coeff in _to_full_contraction(mol, bas_idx):
                nprim, nc = b_coeff.shape
                bs = numpy.zeros((nprim, BAS_SLOTS), dtype=numpy.int32)
                bs[:,ATOM_OF] = ia
                bs[:,ANG_OF ] = l
                bs[:,NCTR_OF] = bs[:,NPRIM_OF] = 1
                bs[:,PTR_EXP] = pexp + numpy.arange(nprim)
                bs[:,PTR_COEFF] = pexp + numpy.arange(nprim, nprim*2)
                norm = gto_norm(l, mol_exps)
                env.append(mol_exps)
                env.append(norm)
                pexp += nprim * 2
                _bas.append(bs)

                c = numpy.einsum('pi,p,xm->pxim', b_coeff, 1./norm, c2s[l])
                contr_coeff.append(c.reshape(nprim * c2s[l].shape[0], -1))

    pmol._bas = numpy.asarray(numpy.vstack(_bas), dtype=numpy.int32)
    pmol._env = numpy.hstack(env)
    if aggregate:
        contr_coeff = scipy.linalg.block_diag(*contr_coeff)
    return pmol, contr_coeff

def format_ecp(ecp_tab):
    r'''Convert the input :attr:`ecp` (dict) to the internal data format::

      { atom: (nelec,  # core electrons
               ((l,  # l=-1 for UL, l>=0 for Ul to indicate |l><l|
                 (((exp_1, c_1),  # for r^0
                   (exp_2, c_2),
                   ...),
                  ((exp_1, c_1),  # for r^1
                   (exp_2, c_2),
                   ...),
                  ((exp_1, c_1),  # for r^2
                   ...))))),
       ...}
    '''
    fmt_ecp = {}
    for atom, atom_ecp in ecp_tab.items():
        symb = _atom_symbol(atom)

        if isinstance(atom_ecp, str):
            stdsymb = _std_symbol_without_ghost(symb)
            ecp_dat = basis.load_ecp(str(atom_ecp), stdsymb)
            if ecp_dat is None or len(ecp_dat) == 0:
                #raise BasisNotFoundError('ECP not found for  %s' % symb)
                sys.stderr.write('ECP %s not found for  %s\n' %
                                 (atom_ecp, symb))
            else:
                fmt_ecp[symb] = ecp_dat
        else:
            fmt_ecp[symb] = atom_ecp
    return fmt_ecp

def format_pseudo(pseudo_tab):
    r'''Convert the input :attr:`pseudo` (dict) to the internal data format::

       { atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }

    Args:
        pseudo_tab : dict
            Similar to :attr:`pseudo` (a dict), it **cannot** be a str

    Returns:
        Formatted :attr:`pseudo`

    Examples:

    >>> pbc.format_pseudo({'H':'gth-blyp', 'He': 'gth-pade'})
    {'H': [[1],
        0.2, 2, [-4.19596147, 0.73049821], 0],
     'He': [[2],
        0.2, 2, [-9.1120234, 1.69836797], 0]}
    '''
    from pyscf.pbc.gto.pseudo import load
    fmt_pseudo = {}
    for atom, atom_pp in pseudo_tab.items():
        symb = _symbol(atom)

        if isinstance(atom_pp, str):
            stdsymb = _std_symbol_without_ghost(symb)
            fmt_pseudo[symb] = load(atom_pp, stdsymb)
        else:
            fmt_pseudo[symb] = atom_pp
    return fmt_pseudo

# transform etb to basis format
def expand_etb(l, n, alpha, beta):
    r'''Generate the exponents of even tempered basis for :attr:`Mole.basis`.
    .. math::

        e = e^{-\alpha * \beta^{i-1}} for i = 1 .. n

    Args:
        l : int
            Angular momentum
        n : int
            Number of GTOs

    Returns:
        Formatted :attr:`~Mole.basis`

    Examples:

    >>> gto.expand_etb(1, 3, 1.5, 2)
    [[1, [6.0, 1]], [1, [3.0, 1]], [1, [1.5, 1]]]
    '''
    return [[l, [alpha*beta**i, 1]] for i in reversed(range(n))]
def expand_etbs(etbs):
    r'''Generate even tempered basis.  See also :func:`expand_etb`

    Args:
        etbs = [(l, n, alpha, beta), (l, n, alpha, beta),...]

    Returns:
        Formatted :attr:`~Mole.basis`

    Examples:

    >>> gto.expand_etbs([(0, 2, 1.5, 2.), (1, 2, 1, 2.)])
    [[0, [6.0, 1]], [0, [3.0, 1]], [1, [1., 1]], [1, [2., 1]]]
    '''
    return lib.flatten([expand_etb(*etb) for etb in etbs])
etbs = expand_etbs

# concatenate two mol
def conc_env(atm1, bas1, env1, atm2, bas2, env2):
    r'''Concatenate two Mole's integral parameters.  This function can be used
    to construct the environment for cross integrals like

    .. math::

        \langle \mu | \nu \rangle, \mu \in mol1, \nu \in mol2

    Returns:
        Concatenated atm, bas, env

    Examples:
        Compute the overlap between H2 molecule and O atom

    >>> mol1 = gto.M(atom='H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mol2 = gto.M(atom='O 0 0 0', basis='sto3g')
    >>> atm3, bas3, env3 = gto.conc_env(mol1._atm, mol1._bas, mol1._env,
    ...                                 mol2._atm, mol2._bas, mol2._env)
    >>> gto.moleintor.getints('int1e_ovlp_sph', atm3, bas3, env3, range(2), range(2,5))
    [[ 0.04875181  0.44714688  0.          0.37820346  0.        ]
     [ 0.04875181  0.44714688  0.          0.          0.37820346]]
    '''
    off = len(env1)
    natm_off = len(atm1)
    atm2 = numpy.copy(atm2)
    bas2 = numpy.copy(bas2)
    atm2[:,PTR_COORD] += off
    atm2[:,PTR_ZETA ] += off
    bas2[:,ATOM_OF  ] += natm_off
    bas2[:,PTR_EXP  ] += off
    bas2[:,PTR_COEFF] += off
    return (numpy.asarray(numpy.vstack((atm1,atm2)), dtype=numpy.int32),
            numpy.asarray(numpy.vstack((bas1,bas2)), dtype=numpy.int32),
            numpy.hstack((env1,env2)))

def conc_mol(mol1, mol2):
    '''Concatenate two Mole objects.
    '''
    if not mol1._built:
        logger.warn(mol1, 'Warning: object %s not initialized. Initializing %s',
                    mol1, mol1)
        mol1.build()
    if not mol2._built:
        logger.warn(mol2, 'Warning: object %s not initialized. Initializing %s',
                    mol2, mol2)
        mol2.build()

    mol3 = Mole()
    mol3._built = True

    mol3._atm, mol3._bas, mol3._env = \
            conc_env(mol1._atm, mol1._bas, mol1._env,
                     mol2._atm, mol2._bas, mol2._env)
    off = len(mol1._env)
    natm_off = len(mol1._atm)
    if len(mol2._ecpbas) == 0:
        mol3._ecpbas = mol1._ecpbas
    else:
        ecpbas2 = numpy.copy(mol2._ecpbas)
        ecpbas2[:,ATOM_OF  ] += natm_off
        ecpbas2[:,PTR_EXP  ] += off
        ecpbas2[:,PTR_COEFF] += off
        if len(mol1._ecpbas) == 0:
            mol3._ecpbas = ecpbas2
        else:
            mol3._ecpbas = numpy.vstack((mol1._ecpbas, ecpbas2))

    mol3.verbose = mol1.verbose
    mol3.output = mol1.output
    mol3.max_memory = mol1.max_memory
    mol3.charge = mol1.charge + mol2.charge
    mol3.spin = abs(mol1.spin - mol2.spin)
    mol3.symmetry = False
    mol3.symmetry_subgroup = None
    mol3.cart = mol1.cart and mol2.cart

    mol3._atom = mol1._atom + mol2._atom
    mol3.atom = mol3._atom
    mol3.unit = 'Bohr'

    mol3._basis.update(mol2._basis)
    mol3._basis.update(mol1._basis)
    mol3._pseudo.update(mol2._pseudo)
    mol3._pseudo.update(mol1._pseudo)
    mol3._ecp.update(mol2._ecp)
    mol3._ecp.update(mol1._ecp)
    mol3._pseudo.update(mol2._pseudo)
    mol3._pseudo.update(mol1._pseudo)
    mol3.basis = mol3._basis
    mol3.ecp = mol3._ecp
    mol3.pseudo = mol3._pseudo

    mol3.nucprop.update(mol1.nucprop)
    mol3.nucprop.update(mol2.nucprop)
    return mol3

# <bas-of-mol1|intor|bas-of-mol2>
def intor_cross(intor, mol1, mol2, comp=None, grids=None):
    r'''1-electron integrals from two molecules like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in mol1, \nu \in mol2

    Args:
        intor : str
            Name of the 1-electron integral, such as int1e_ovlp_sph (spherical overlap),
            int1e_nuc_cart (cartesian nuclear attraction), int1e_ipovlp_spinor
            (spinor overlap gradients), etc.  Ref to :func:`getints` for the
            full list of available 1-electron integral names
        mol1, mol2:
            :class:`Mole` objects

    Kwargs:
        comp : int
            Components of the integrals, e.g. int1e_ipovlp_sph has 3 components
        grids : ndarray
            Coordinates of grids for the int1e_grids integrals

    Returns:
        ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

    Examples:
        Compute the overlap between H2 molecule and O atom

    >>> mol1 = gto.M(atom='H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mol2 = gto.M(atom='O 0 0 0', basis='sto3g')
    >>> gto.intor_cross('int1e_ovlp_sph', mol1, mol2)
    [[ 0.04875181  0.44714688  0.          0.37820346  0.        ]
     [ 0.04875181  0.44714688  0.          0.          0.37820346]]
    '''
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = conc_env(mol1._atm, mol1._bas, mol1._env,
                                mol2._atm, mol2._bas, mol2._env)
    if '_grids' in intor:
        assert grids is not None
        envc = numpy.append(envc, grids.ravel())
        envc[NGRIDS] = grids.shape[0]
        envc[PTR_GRIDS] = envc.size - grids.size

    shls_slice = (0, nbas1, nbas1, nbas1+nbas2)

    if (intor.endswith('_sph') or intor.startswith('cint') or
        intor.endswith('_spinor') or intor.endswith('_cart')):
        return moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart == mol2.cart:
        intor = mol1._add_suffix(intor)
        return moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart:
        mat = moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return numpy.dot(mat, mol2.cart2sph_coeff())
    else:
        mat = moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return numpy.dot(mol1.cart2sph_coeff().T, mat)

# append (charge, pointer to coordinates, nuc_mod) to _atm
def make_atm_env(atom, ptr=0, nuclear_model=NUC_POINT, nucprop={}):
    '''Convert the internal format :attr:`Mole._atom` to the format required
    by ``libcint`` integrals
    '''
    nuc_charge = charge(atom[0])
    if nuclear_model == NUC_POINT:
        zeta = 0
    elif nuclear_model == NUC_GAUSS:
        zeta = dyall_nuc_mod(nuc_charge, nucprop)
    else:  # callable(nuclear_model)
        zeta = nuclear_model(nuc_charge, nucprop)
        nuclear_model = NUC_GAUSS

    _env = numpy.hstack((atom[1], zeta))
    _atm = numpy.zeros(6, dtype=numpy.int32)
    _atm[CHARGE_OF] = nuc_charge
    _atm[PTR_COORD] = ptr
    _atm[NUC_MOD_OF] = nuclear_model
    _atm[PTR_ZETA ] = ptr + 3
    return _atm, _env

# append (atom, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0) to bas
# absorb normalization into GTO contraction coefficients
def make_bas_env(basis_add, atom_id=0, ptr=0):
    '''Convert :attr:`Mole.basis` to the argument ``bas`` for ``libcint`` integrals
    '''
    _bas = []
    _env = []
    for b in basis_add:
        angl = b[0]
        if angl > 14:
            sys.stderr.write('Warning: integral library does not support basis '
                             'with angular momentum > 14\n')

        if isinstance(b[1], int):
            kappa = b[1]
            b_coeff = numpy.array(sorted(b[2:], reverse=True))
        else:
            kappa = 0
            b_coeff = numpy.array(sorted(b[1:], reverse=True))
        es = b_coeff[:,0]
        cs = b_coeff[:,1:]
        nprim, nctr = cs.shape
        cs = numpy.einsum('pi,p->pi', cs, gto_norm(angl, es))
        if NORMALIZE_GTO:
            cs = _nomalize_contracted_ao(angl, es, cs)

        _env.append(es)
        _env.append(cs.T.reshape(-1))
        ptr_exp = ptr
        ptr_coeff = ptr_exp + nprim
        ptr = ptr_coeff + nprim * nctr
        _bas.append([atom_id, angl, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0])
    _env = lib.flatten(_env) # flatten nested lists
    return (numpy.array(_bas, numpy.int32).reshape(-1,BAS_SLOTS),
            numpy.array(_env, numpy.double))

def _nomalize_contracted_ao(l, es, cs):
    #ee = numpy.empty((nprim,nprim))
    #for i in range(nprim):
    #    for j in range(i+1):
    #        ee[i,j] = ee[j,i] = gaussian_int(angl*2+2, es[i]+es[j])
    #s1 = 1/numpy.sqrt(numpy.einsum('pi,pq,qi->i', cs, ee, cs))
    ee = es.reshape(-1,1) + es.reshape(1,-1)
    ee = gaussian_int(l*2+2, ee)
    s1 = 1. / numpy.sqrt(numpy.einsum('pi,pq,qi->i', cs, ee, cs))
    return numpy.einsum('pi,i->pi', cs, s1)

def make_env(atoms, basis, pre_env=[], nucmod={}, nucprop={}):
    '''Generate the input arguments for ``libcint`` library based on internal
    format :attr:`Mole._atom` and :attr:`Mole._basis`
    '''
    _atm = []
    _bas = []
    _env = [pre_env]
    ptr_env = len(pre_env)

    for ia, atom in enumerate(atoms):
        symb = atom[0]
        stdsymb = _rm_digit(symb)

        if ia+1 in nucprop:
            prop = nucprop[ia+1]
        elif symb in nucprop:
            prop = nucprop[symb]
        else:
            prop = nucprop.get(stdsymb, {})

        nuclear_model = NUC_POINT
        if nucmod:
            if nucmod is None:
                nuclear_model = NUC_POINT
            elif isinstance(nucmod, (int, str, types.FunctionType)):
                nuclear_model = _parse_nuc_mod(nucmod)
            elif ia+1 in nucmod:
                nuclear_model = _parse_nuc_mod(nucmod[ia+1])
            elif symb in nucmod:
                nuclear_model = _parse_nuc_mod(nucmod[symb])
            elif stdsymb in nucmod:
                nuclear_model = _parse_nuc_mod(nucmod[stdsymb])
        atm0, env0 = make_atm_env(atom, ptr_env, nuclear_model, prop)
        ptr_env = ptr_env + len(env0)
        _atm.append(atm0)
        _env.append(env0)

    _basdic = {}
    for symb, basis_add in basis.items():
        bas0, env0 = make_bas_env(basis_add, 0, ptr_env)
        ptr_env = ptr_env + len(env0)
        _basdic[symb] = bas0
        _env.append(env0)

    for ia, atom in enumerate(atoms):
        symb = atom[0]
        puresymb = _rm_digit(symb)
        if symb in _basdic:
            b = _basdic[symb].copy()
        elif puresymb in _basdic:
            b = _basdic[puresymb].copy()
        else:
            if symb[:2].upper() == 'X-':
                symb = symb[2:]
            elif symb[:6].upper() == 'GHOST-':
                symb = symb[6:]
            puresymb = _rm_digit(symb)
            if symb in _basdic:
                b = _basdic[symb].copy()
            elif puresymb in _basdic:
                b = _basdic[puresymb].copy()
            else:
                sys.stderr.write('Warning: Basis not found for atom %d %s\n' % (ia, symb))
                continue
        b[:,ATOM_OF] = ia
        _bas.append(b)

    if _atm:
        _atm = numpy.asarray(numpy.vstack(_atm), numpy.int32).reshape(-1, ATM_SLOTS)
    else:
        _atm = numpy.zeros((0,ATM_SLOTS), numpy.int32)
    if _bas:
        _bas = numpy.asarray(numpy.vstack(_bas), numpy.int32).reshape(-1, BAS_SLOTS)
    else:
        _bas = numpy.zeros((0,BAS_SLOTS), numpy.int32)
    _env = numpy.asarray(numpy.hstack(_env), dtype=numpy.float64)
    return _atm, _bas, _env

def make_ecp_env(mol, _atm, ecp, pre_env=[]):
    '''Generate the input arguments _ecpbas for ECP integrals
    '''
    _env = []
    ptr_env = len(pre_env)

    _ecpdic = {}
    for symb, ecp_add in ecp.items():
        ecp0 = []
        nelec = ecp_add[0]
        for lb in ecp_add[1]:
            for rorder, bi in enumerate(lb[1]):
                if len(bi) > 0:
                    ec = numpy.array(sorted(bi, reverse=True))
                    nexp, ncol = ec.shape
                    _env.append(ec[:,0])
                    _env.append(ec[:,1])
                    ptr_exp, ptr_coeff = ptr_env, ptr_env + nexp
                    ecp0.append([0, lb[0], nexp, rorder, 0,
                                 ptr_exp, ptr_coeff, 0])
                    ptr_env += nexp * 2

                    if ncol == 3:  # Has SO-ECP
                        _env.append(ec[:,2])
                        ptr_coeff, ptr_env = ptr_env, ptr_env + nexp
                        ecp0.append([0, lb[0], nexp, rorder, 1,
                                     ptr_exp, ptr_coeff, 0])

        _ecpdic[symb] = (nelec, numpy.asarray(ecp0, dtype=numpy.int32))

    _ecpbas = []
    if _ecpdic:
        _atm = _atm.copy()
        for ia, atom in enumerate(mol._atom):
            symb = atom[0]
            if symb in _ecpdic:
                ecp0 = _ecpdic[symb]
            elif _rm_digit(symb) in _ecpdic:
                ecp0 = _ecpdic[_rm_digit(symb)]
            else:
                ecp0 = None
            if ecp0 is not None:
                _atm[ia,CHARGE_OF ] = charge(symb) - ecp0[0]
                _atm[ia,NUC_MOD_OF] = NUC_ECP
                b = ecp0[1].copy().reshape(-1,BAS_SLOTS)
                b[:,ATOM_OF] = ia
                _ecpbas.append(b)
    if _ecpbas:
        _ecpbas = numpy.asarray(numpy.vstack(_ecpbas), numpy.int32)
        _env = numpy.hstack([pre_env] + _env)
    else:
        _ecpbas = numpy.zeros((0,BAS_SLOTS), numpy.int32)
        _env = pre_env
    return _atm, _ecpbas, _env

def tot_electrons(mol):
    '''Total number of electrons for the given molecule

    Returns:
        electron number in integer

    Examples:

    >>> mol = gto.M(atom='H 0 1 0; C 0 0 1', charge=1)
    >>> mol.tot_electrons()
    6
    '''
    if mol._atm.size != 0:
        nelectron = mol.atom_charges().sum()
    elif mol._atom:
        nelectron = sum(charge(a[0]) for a in mol._atom)
    else:
        nelectron = sum(charge(a[0]) for a in format_atom(mol.atom))
    nelectron -= mol.charge
    nelectron_int = int(round(nelectron))

    if abs(nelectron - nelectron_int) > 1e-4:
        logger.warn(mol, 'Found fractional number of electrons %f. Round it to %d',
                    nelectron, nelectron_int)
    return nelectron_int

def copy(mol, deep=True):
    '''Deepcopy of the given :class:`Mole` object

    Some attributes are shared between the original and copied objects.
    Deepcopy is utilized here to ensure that operations on the copied object do
    not affect the original object.
    '''
    # Avoid copy.copy(mol) for shallow copy because copy.copy automatically
    # calls __copy__, __reduce__, __getstate__, __setstate__ methods
    newmol = mol.view(mol.__class__)
    if not deep:
        return newmol

    import copy
    newmol._atm    = numpy.copy(mol._atm)
    newmol._bas    = numpy.copy(mol._bas)
    newmol._env    = numpy.copy(mol._env)
    newmol._ecpbas = numpy.copy(mol._ecpbas)

    newmol.atom    = copy.deepcopy(mol.atom)
    newmol._atom   = copy.deepcopy(mol._atom)
    newmol.basis   = copy.deepcopy(mol.basis)
    newmol._basis  = copy.deepcopy(mol._basis)
    newmol.ecp     = copy.deepcopy(mol.ecp)
    newmol._ecp    = copy.deepcopy(mol._ecp)
    newmol.pseudo  = copy.deepcopy(mol.pseudo)
    newmol._pseudo = copy.deepcopy(mol._pseudo)
    if mol.magmom is not None:
        newmol.magmom  = list(mol.magmom)
    return newmol

def pack(mol):
    '''Pack the input args of :class:`Mole` to a dict.

    Note this function only pack the input arguments (not the entire Mole
    class).  Modifications to mol._atm, mol._bas, mol._env are not tracked.
    Use :func:`dumps` to serialize the entire Mole object.
    '''
    mdic = {'atom'    : mol.atom,
            'unit'    : mol.unit,
            'basis'   : mol.basis,
            'charge'  : mol.charge,
            'spin'    : mol.spin,
            'cart'    : mol.cart,
            'symmetry': mol.symmetry,
            'symmetry_subgroup': mol.symmetry_subgroup,
            'nucmod'  : mol.nucmod,
            'nucprop' : mol.nucprop,
            'ecp'     : mol.ecp,
            'pseudo'  : mol.pseudo,
            '_nelectron': mol._nelectron,
            'magmom'  : mol.magmom,
            'verbose' : mol.verbose}
    return mdic
def unpack(moldic):
    '''Unpack a dict which is packed by :func:`pack`, to generate the input
    arguments for :class:`Mole` object.
    '''
    mol = Mole()
    mol.__dict__.update(moldic)
    return mol


def dumps(mol):
    '''Serialize Mole object to a JSON formatted str.
    '''
    exclude_keys = {'output', 'stdout', '_keys', '_ctx_lock',
                    # Constructing in function loads
                    'symm_orb', 'irrep_id', 'irrep_name'}
    moldic = dict(mol.__dict__)
    for k in exclude_keys:
        if k in moldic:
            del (moldic[k])
    for k in moldic:
        if isinstance(moldic[k], (numpy.ndarray, numpy.generic)):
            moldic[k] = moldic[k].tolist()
    moldic['atom'] = repr(mol.atom)
    moldic['basis']= repr(mol.basis)
    moldic['ecp' ] = repr(mol.ecp)
    moldic['pseudo'] = repr(mol.pseudo)

    try:
        return json.dumps(moldic)
    except TypeError:
        def skip_value(dic):
            dic1 = {}
            for k,v in dic.items():
                if (v is None or
                    isinstance(v, (str, bool, int, float))):
                    dic1[k] = v
                elif isinstance(v, (list, tuple)):
                    dic1[k] = v   # Should I recursively skip_vaule?
                elif isinstance(v, set):
                    dic1[k] = list(v)
                elif isinstance(v, dict):
                    dic1[k] = skip_value(v)
                elif isinstance(v, np.generic):
                    dic1[k] = v.tolist()
                else:
                    msg =('Function mol.dumps drops attribute %s because '
                          'it is not JSON-serializable' % k)
                    warnings.warn(msg)
            return dic1
        return json.dumps(skip_value(moldic), skipkeys=True)

def loads(molstr):
    '''Deserialize a str containing a JSON document to a Mole object.
    '''
    # the numpy function array is used by eval function
    from numpy import array  # noqa
    moldic = json.loads(molstr)
    mol = Mole()
    mol.__dict__.update(moldic)
    mol.atom = eval(mol.atom)
    mol.basis= eval(mol.basis)
    mol.ecp  = eval(mol.ecp)
    if 'pseudo' in moldic:
        # backward compatibility with old dumps function, which does not have
        # the pseudo attribute
        mol.pseudo  = eval(mol.pseudo)
    mol._atm = numpy.array(mol._atm, dtype=numpy.int32)
    mol._bas = numpy.array(mol._bas, dtype=numpy.int32)
    mol._env = numpy.array(mol._env, dtype=numpy.double)
    mol._ecpbas = numpy.array(mol._ecpbas, dtype=numpy.int32)

    # Objects related to symmetry cannot be serialized by dumps function.
    # Recreate it manually
    if mol.symmetry and mol._symm_orig is not None:
        from pyscf import symm
        mol._symm_orig = numpy.array(mol._symm_orig)
        mol._symm_axes = numpy.array(mol._symm_axes)
        mol.symm_orb, mol.irrep_id = \
                symm.symm_adapted_basis(mol, mol.groupname,
                                        mol._symm_orig, mol._symm_axes)
        mol.irrep_name = [symm.irrep_id2name(mol.groupname, ir)
                           for ir in mol.irrep_id]

    elif mol.symmetry and mol.symm_orb is not None:
        # Backward compatibility. To load symm_orb from chkfile of pyscf-1.6
        # and earlier.
        symm_orb = []

        # decompress symm_orb
        for val, x, y, shape in mol.symm_orb:
            if isinstance(val[0], list):
                # backward compatibility for chkfile of pyscf-1.4 in which val
                # is an array of real floats. In pyscf-1.5, val can be a list
                # of list, to include the imaginary part
                val_real, val_imag = val
            else:
                val_real, val_imag = val, None
            if val_imag is None:
                c = numpy.zeros(shape)
                c[numpy.array(x),numpy.array(y)] = numpy.array(val_real)
            else:
                c = numpy.zeros(shape, dtype=numpy.complex128)
                val = numpy.array(val_real) + numpy.array(val_imag) * 1j
                c[numpy.array(x),numpy.array(y)] = val
            symm_orb.append(c)
        mol.symm_orb = symm_orb

    return mol


def len_spinor(l, kappa):
    '''The number of spinor associated with given angular momentum and kappa.  If kappa is 0,
    return 4l+2
    '''
    if kappa == 0:
        n = (l * 4 + 2)
    elif kappa < 0:
        n = (l * 2 + 2)
    else:
        n = (l * 2)
    return n

def len_cart(l):
    '''The number of Cartesian function associated with given angular momentum.
    '''
    return (l + 1) * (l + 2) // 2

def npgto_nr(mol, cart=None):
    '''Total number of primitive spherical GTOs for the given :class:`Mole` object'''
    if cart is None:
        cart = mol.cart
    l = mol._bas[:,ANG_OF]
    if cart:
        return int(((l+1)*(l+2)//2 * mol._bas[:,NPRIM_OF]).sum())
    else:
        return int(((l*2+1) * mol._bas[:,NPRIM_OF]).sum())
def nao_nr(mol, cart=None):
    '''Total number of contracted GTOs for the given :class:`Mole` object'''
    if cart is None:
        cart = mol.cart
    if cart:
        return nao_cart(mol)
    else:
        return int(((mol._bas[:,ANG_OF]*2+1) * mol._bas[:,NCTR_OF]).sum())
def nao_cart(mol):
    '''Total number of contracted cartesian GTOs for the given :class:`Mole` object'''
    l = mol._bas[:,ANG_OF]
    return int(((l+1)*(l+2)//2 * mol._bas[:,NCTR_OF]).sum())

# nao_id0:nao_id1 corresponding to bas_id0:bas_id1
def nao_nr_range(mol, bas_id0, bas_id1):
    '''Lower and upper boundary of contracted spherical basis functions associated
    with the given shell range

    Args:
        mol :
            :class:`Mole` object
        bas_id0 : int
            start shell id
        bas_id1 : int
            stop shell id

    Returns:
        tuple of start basis function id and the stop function id

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.nao_nr_range(mol, 2, 4)
    (2, 6)
    '''
    ao_loc = moleintor.make_loc(mol._bas[:bas_id1], 'sph')
    nao_id0 = int(ao_loc[bas_id0])
    nao_id1 = int(ao_loc[-1])
    return nao_id0, nao_id1

def nao_2c(mol):
    '''Total number of contracted spinor GTOs for the given :class:`Mole` object'''
    l = mol._bas[:,ANG_OF]
    kappa = mol._bas[:,KAPPA_OF]
    dims = (l*4+2) * mol._bas[:,NCTR_OF]
    dims[kappa<0] = (l[kappa<0] * 2 + 2) * mol._bas[kappa<0,NCTR_OF]
    dims[kappa>0] = (l[kappa>0] * 2) * mol._bas[kappa>0,NCTR_OF]
    return int(dims.sum())

# nao_id0:nao_id1 corresponding to bas_id0:bas_id1
def nao_2c_range(mol, bas_id0, bas_id1):
    '''Lower and upper boundary of contracted spinor basis functions associated
    with the given shell range

    Args:
        mol :
            :class:`Mole` object
        bas_id0 : int
            start shell id, 0-based
        bas_id1 : int
            stop shell id, 0-based

    Returns:
        tuple of start basis function id and the stop function id

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.nao_2c_range(mol, 2, 4)
    (4, 12)
    '''
    ao_loc = moleintor.make_loc(mol._bas[:bas_id1], '')
    nao_id0 = int(ao_loc[bas_id0])
    nao_id1 = int(ao_loc[-1])
    return nao_id0, nao_id1

def ao_loc_nr(mol, cart=None):
    '''Offset of every shell in the spherical basis function spectrum

    Returns:
        list, each entry is the corresponding start basis function id

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.ao_loc_nr(mol)
    [0, 1, 2, 3, 6, 9, 10, 11, 12, 15, 18]
    '''
    if cart is None:
        cart = mol.cart
    if cart:
        return moleintor.make_loc(mol._bas, 'cart')
    else:
        return moleintor.make_loc(mol._bas, 'sph')

def ao_loc_2c(mol):
    '''Offset of every shell in the spinor basis function spectrum

    Returns:
        list, each entry is the corresponding start id of spinor function

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.ao_loc_2c(mol)
    [0, 2, 4, 6, 12, 18, 20, 22, 24, 30, 36]
    '''
    return moleintor.make_loc(mol._bas, 'spinor')

def time_reversal_map(mol):
    r'''The index to map the spinor functions and its time reversal counterpart.
    The returned indices have positive or negative values.  For the i-th basis function,
    if the returned j = idx[i] < 0, it means :math:`T|i\rangle = -|j\rangle`,
    otherwise :math:`T|i\rangle = |j\rangle`
    '''
    tao = []
    i = 0
    for b in mol._bas:
        l = b[ANG_OF]
        if b[KAPPA_OF] == 0:
            djs = (l * 2, l * 2 + 2)
        elif b[KAPPA_OF] > 0:
            djs = (l * 2,)
        else:
            djs = (l * 2 + 2,)
        if l % 2 == 0:
            for n in range(b[NCTR_OF]):
                for dj in djs:
                    for m in range(0, dj, 2):
                        tao.append(-(i + dj - m))
                        tao.append(  i + dj - m - 1)
                    i += dj
        else:
            for n in range(b[NCTR_OF]):
                for dj in djs:
                    for m in range(0, dj, 2):
                        tao.append(  i + dj - m)
                        tao.append(-(i + dj - m - 1))
                    i += dj
    return numpy.asarray(tao, dtype=numpy.int32)


CHECK_GEOM = getattr(__config__, 'gto_mole_check_geom', True)

def classical_coulomb_energy(mol, charges=None, coords=None):
    '''Compute nuclear repulsion energy (AU) or static Coulomb energy

    Returns
        float
    '''
    if charges is None: charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    rr = inter_distance(mol, coords)
    rr[numpy.diag_indices_from(rr)] = 1e200
    if CHECK_GEOM and numpy.any(rr < 1e-5):
        raise_err = False
        for atm_idx in numpy.argwhere(rr<1e-5):
            # Only raise error if atoms with charge != 0 have the same coordinates
            if charges[atm_idx[0]] * charges[atm_idx[1]] != 0:
                logger.warn(mol, 'Atoms %s have the same coordinates', atm_idx)
                raise_err = True
            # At least one of the atoms is a ghost atom; suppress divide by 0 warning
            else:
                rr[atm_idx[0], atm_idx[1]] = 1e200
        if raise_err: raise RuntimeError('Ill geometry')
    e = numpy.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

energy_nuc = classical_coulomb_energy

def inter_distance(mol, coords=None):
    '''
    Inter-particle distance array
    '''
    if coords is None: coords = mol.atom_coords()
    rr = numpy.linalg.norm(coords.reshape(-1,1,3) - coords, axis=2)
    rr[numpy.diag_indices_from(rr)] = 0
    return rr

def sph_labels(mol, fmt=True, base=BASE):
    '''Labels for spherical GTO functions

    Kwargs:
        fmt : str or bool
        if fmt is boolean, it controls whether to format the labels and the
        default format is "%d%3s %s%-4s".  if fmt is string, the string will
        be used as the print format.

    Returns:
        List of [(atom-id, symbol-str, nl-str, str-of-real-spherical-notation]
        or formatted strings based on the argument "fmt"

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; Cl 0 0 1', basis='sto-3g')
    >>> gto.sph_labels(mol)
    [(0, 'H', '1s', ''), (1, 'Cl', '1s', ''), (1, 'Cl', '2s', ''), (1, 'Cl', '3s', ''),
     (1, 'Cl', '2p', 'x'), (1, 'Cl', '2p', 'y'), (1, 'Cl', '2p', 'z'), (1, 'Cl', '3p', 'x'),
     (1, 'Cl', '3p', 'y'), (1, 'Cl', '3p', 'z')]
    '''
    count = numpy.zeros((mol.natm, 9), dtype=int)
    label = []
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        strl = param.ANGULAR[l]
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)
        nelec_ecp = mol.atom_nelec_core(ia)
        if nelec_ecp == 0 or l > 3:
            shl_start = count[ia,l]+l+1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[l]+count[ia,l]+l+1
        count[ia,l] += nc
        for n in range(shl_start, shl_start+nc):
            for m in range(-l, l+1):
                label.append((ia+base, symb, '%d%s' % (n, strl),
                              str(param.REAL_SPHERIC[l][l+m])))

    if isinstance(fmt, str):
        return [(fmt % x) for x in label]
    elif fmt:
        return ['%d %s %s%-4s' % x for x in label]
    else:
        return label
spheric_labels = spherical_labels = sph_labels

def cart_labels(mol, fmt=True, base=BASE):
    '''Labels of Cartesian GTO functions

    Kwargs:
        fmt : str or bool
        if fmt is boolean, it controls whether to format the labels and the
        default format is "%d%3s %s%-4s".  if fmt is string, the string will
        be used as the print format.

    Returns:
        List of [(atom-id, symbol-str, nl-str, str-of-xyz-notation)]
        or formatted strings based on the argument "fmt"
    '''
    cartxyz = []
    for l in range(max(mol._bas[:,ANG_OF])+1):
        xyz = []
        for x in range(l, -1, -1):
            for y in range(l-x, -1, -1):
                z = l-x-y
                xyz.append('x'*x + 'y'*y + 'z'*z)
        cartxyz.append(xyz)

    count = numpy.zeros((mol.natm, 9), dtype=int)
    label = []
    for ib in range(len(mol._bas)):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        strl = param.ANGULAR[l]
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)
        nelec_ecp = mol.atom_nelec_core(ia)
        if nelec_ecp == 0 or l > 3:
            shl_start = count[ia,l]+l+1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[l]+count[ia,l]+l+1
        count[ia,l] += nc
        ncart = (l + 1) * (l + 2) // 2
        for n in range(shl_start, shl_start+nc):
            for m in range(ncart):
                label.append((ia+base, symb, '%d%s' % (n, strl), cartxyz[l][m]))

    if isinstance(fmt, str):
        return [(fmt % x) for x in label]
    elif fmt:
        return ['%d%3s %s%-4s' % x for x in label]
    else:
        return label

def ao_labels(mol, fmt=True, base=BASE):
    '''Labels of AO basis functions

    Kwargs:
        fmt : str or bool
            if fmt is boolean, it controls whether to format the labels and the
            default format is "%d%3s %s%-4s".  if fmt is string, the string will
            be used as the print format.

    Returns:
        List of [(atom-id, symbol-str, nl-str, str-of-AO-notation)]
        or formatted strings based on the argument "fmt"
    '''
    if mol.cart:
        return mol.cart_labels(fmt, base)
    else:
        return mol.sph_labels(fmt, base)

def spinor_labels(mol, fmt=True, base=BASE):
    '''
    Labels of spinor GTO functions
    '''
    count = numpy.zeros((mol.natm, 9), dtype=int)
    label = []
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        kappa = mol.bas_kappa(ib)
        strl = param.ANGULAR[l]
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)
        nelec_ecp = mol.atom_nelec_core(ia)
        if nelec_ecp == 0 or l > 3:
            shl_start = count[ia,l]+l+1
        else:
            coreshl = core_configuration(nelec_ecp, atom_symbol=_std_symbol(symb))
            shl_start = coreshl[l]+count[ia,l]+l+1
        count[ia,l] += nc
        for n in range(shl_start, shl_start+nc):
            if kappa >= 0:
                for m in range(-l*2+1, l*2, 2):
                    label.append((ia+base, symb, '%d%s%d/2' % (n, strl, l*2-1),
                                  '%d/2'%m))
            if kappa <= 0:
                for m in range(-l*2-1, l*2+2, 2):
                    label.append((ia+base, symb, '%d%s%d/2' % (n, strl, l*2+1),
                                  '%d/2'%m))

    if isinstance(fmt, str):
        return [(fmt % x) for x in label]
    elif fmt:
        return ['%d %s %s,%-5s' % x for x in label]
    else:
        return label

def search_ao_label(mol, label):
    '''Find the index of the AO basis function based on the given ao_label

    Args:
        ao_label : string or a list of strings
            The regular expression pattern to match the orbital labels
            returned by mol.ao_labels()

    Returns:
        A list of index for the AOs that matches the given ao_label RE pattern

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; Cl 0 0 1', basis='ccpvtz')
    >>> mol.search_ao_label('Cl.*p')
    [19 20 21 22 23 24 25 26 27 28 29 30]
    >>> mol.search_ao_label('Cl 2p')
    [19 20 21]
    >>> mol.search_ao_label(['Cl.*d', 'Cl 4p'])
    [25 26 27 31 32 33 34 35 36 37 38 39 40]
    '''
    return _aolabels2baslst(mol, label)

def _aolabels2baslst(mol, aolabels_or_baslst, base=BASE):
    if callable(aolabels_or_baslst):
        baslst = [i for i,x in enumerate(mol.ao_labels(base=base))
                  if aolabels_or_baslst(x)]
    elif isinstance(aolabels_or_baslst, str):
        aolabels = re.sub(' +', ' ', aolabels_or_baslst.strip(), count=1)
        aolabels = re.compile(aolabels)
        baslst = [i for i,s in enumerate(mol.ao_labels(base=base))
                  if re.search(aolabels, s)]
    elif len(aolabels_or_baslst) > 0 and isinstance(aolabels_or_baslst[0], str):
        aolabels = [re.compile(re.sub(' +', ' ', x.strip(), count=1))
                    for x in aolabels_or_baslst]
        baslst = [i for i,t in enumerate(mol.ao_labels(base=base))
                  if any(re.search(x, t) for x in aolabels)]
    else:
        baslst = [i-base for i in aolabels_or_baslst]
    return numpy.asarray(baslst, dtype=int)

def search_shell_id(mol, atm_id, l):
    '''Search the first basis/shell id (**not** the basis function id) which
    matches the given atom-id and angular momentum

    Args:
        atm_id : int
            atom id, 0-based
        l : int
            angular momentum

    Returns:
        basis id, 0-based.  If not found, return None

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; Cl 0 0 1', basis='sto-3g')
    >>> mol.search_shell_id(1, 1) # Cl p shell
    4
    >>> mol.search_shell_id(1, 2) # Cl d shell
    None
    '''
    for ib in range(len(mol._bas)):
        ia = mol.bas_atom(ib)
        l1 = mol.bas_angular(ib)
        if ia == atm_id and l1 == l:
            return ib

def search_ao_nr(mol, atm_id, l, m, atmshell):
    '''Search the first basis function id (**not** the shell id) which matches
    the given atom-id, angular momentum magnetic angular momentum, principal shell.

    Args:
        atm_id : int
            atom id, 0-based
        l : int
            angular momentum
        m : int
            magnetic angular momentum
        atmshell : int
            principal quantum number

    Returns:
        basis function id, 0-based.  If not found, return None

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; Cl 0 0 1', basis='sto-3g')
    >>> mol.search_ao_nr(1, 1, -1, 3) # Cl 3px
    7
    '''
    ibf = 0
    for ib in range(len(mol._bas)):
        ia = mol.bas_atom(ib)
        l1 = mol.bas_angular(ib)
        if mol.cart:
            degen = (l1 + 1) * (l1 + 2) // 2
        else:
            degen = l1 * 2 + 1
        nc = mol.bas_nctr(ib)
        if ia == atm_id and l1 == l:
            if atmshell > nc+l1:
                atmshell = atmshell - nc
            else:
                return ibf + (atmshell-l1-1)*degen + (l1+m)
        ibf += degen * nc
    raise RuntimeError('Required AO not found')

def search_ao_r(mol, atm_id, l, j, m, atmshell):
    raise RuntimeError('TODO')
#TODO:    ibf = 0
#TODO:    for ib in range(len(mol._bas)):
#TODO:        ia = mol.bas_atom(ib)
#TODO:        l1 = mol.bas_angular(ib)
#TODO:        nc = mol.bas_nctr(ib)
#TODO:        k = mol.bas_kappa(bas_id)
#TODO:        degen = len_spinor(l1, k)
#TODO:        if ia == atm_id and l1 == l and k == kappa:
#TODO:            if atmshell > nc+l1:
#TODO:                atmshell = atmshell - nc
#TODO:            else:
#TODO:                return ibf + (atmshell-l1-1)*degen + (degen+m)
#TODO:        ibf += degen

def offset_2c_by_atom(mol):
    '''2-component AO offset for each atom.  Return a list, each item
    of the list gives (start-shell-id, stop-shell-id, start-AO-id, stop-AO-id)
    '''
    return aoslice_by_atom(mol, mol.ao_loc_2c())

def aoslice_by_atom(mol, ao_loc=None):
    '''AO offsets for each atom.  Return a list, each item of the list gives
    (start-shell-id, stop-shell-id, start-AO-id, stop-AO-id)
    '''
    if ao_loc is None:
        ao_loc = mol.ao_loc_nr()

    aorange = numpy.empty((mol.natm,4), dtype=int)

    if mol.natm == 0:
        return aorange

    bas_atom = mol._bas[:,ATOM_OF]
    delimiter = numpy.where(bas_atom[0:-1] != bas_atom[1:])[0] + 1

    if mol.natm == len(delimiter) + 1:
        aorange[:,0] = shell_start = numpy.append(0, delimiter)
        aorange[:,1] = shell_end = numpy.append(delimiter, mol.nbas)

    else:  # Some atoms miss basis
        shell_start = numpy.empty(mol.natm, dtype=int)
        shell_start[:] = -1
        shell_start[0] = 0
        shell_start[bas_atom[0]] = 0
        shell_start[bas_atom[delimiter]] = delimiter

        shell_end = numpy.empty(mol.natm, dtype=int)
        shell_end[0] = 0
        shell_end[bas_atom[delimiter-1]] = delimiter
        shell_end[bas_atom[-1]] = mol.nbas

        for i in range(1, mol.natm):
            if shell_start[i] == -1:
                shell_start[i] = shell_end[i] = shell_end[i-1]

    aorange[:,0] = shell_start
    aorange[:,1] = shell_end
    aorange[:,2] = ao_loc[shell_start]
    aorange[:,3] = ao_loc[shell_end]
    return aorange
offset_nr_by_atom = aoslice_by_atom

def same_basis_set(mol1, mol2):
    '''Check whether two molecules use the same basis sets.
    The two molecules can have different geometry.
    '''
    atomtypes1 = atom_types(mol1._atom, mol1._basis)
    atomtypes2 = atom_types(mol2._atom, mol2._basis)
    if set(atomtypes1.keys()) != set(atomtypes2.keys()):
        return False
    for k in atomtypes1:
        if len(atomtypes1[k]) != len(atomtypes2[k]):
            return False
        elif mol1._basis.get(k, None) != mol2._basis.get(k, None):
            return False
    return True

def same_mol(mol1, mol2, tol=1e-5, cmp_basis=True, ignore_chiral=False):
    '''Compare the two molecules whether they have the same structure.

    Kwargs:
        tol : float
            In Bohr
        cmp_basis : bool
            Whether to compare basis functions for the two molecules
    '''
    from pyscf import symm

    if mol1._atom.__len__() != mol2._atom.__len__():
        return False

    chg1 = mol1._atm[:,CHARGE_OF]
    chg2 = mol2._atm[:,CHARGE_OF]
    if not numpy.all(numpy.sort(chg1) == numpy.sort(chg2)):
        return False

    if cmp_basis and not same_basis_set(mol1, mol2):
        return False

    def finger(mol, chgs, coord):
        center = numpy.einsum('z,zr->r', chgs, coord) / chgs.sum()
        im = inertia_moment(mol, chgs, coord)
        # Divid im by chgs.sum(), to normalize im. Otherwise the input tol may
        # not reflect the actual deviation.
        im /= chgs.sum()

        w, v = scipy.linalg.eigh(im)
        axes = v.T
        if numpy.linalg.det(axes) < 0:
            axes *= -1
        r = numpy.dot(coord-center, axes.T)
        return w, r

    coord1 = mol1.atom_coords()
    coord2 = mol2.atom_coords()
    w1, r1 = finger(mol1, chg1, coord1)
    w2, r2 = finger(mol2, chg2, coord2)
    if not (numpy.allclose(w1, w2, atol=tol)):
        return False

    rotate_xy  = numpy.array([[-1., 0., 0.],
                              [ 0.,-1., 0.],
                              [ 0., 0., 1.]])
    rotate_yz  = numpy.array([[ 1., 0., 0.],
                              [ 0.,-1., 0.],
                              [ 0., 0.,-1.]])
    rotate_zx  = numpy.array([[-1., 0., 0.],
                              [ 0., 1., 0.],
                              [ 0., 0.,-1.]])

    def inspect(z1, r1, z2, r2):
        place = int(-numpy.log10(tol)) - 1
        idx = symm.argsort_coords(r2, place)
        z2 = z2[idx]
        r2 = r2[idx]
        for rot in (1, rotate_xy, rotate_yz, rotate_zx):
            r1new = numpy.dot(r1, rot)
            idx = symm.argsort_coords(r1new, place)
            if (numpy.all(z1[idx] == z2) and
                numpy.allclose(r1new[idx], r2, atol=tol)):
                return True
        return False

    return (inspect(chg1, r1, chg2, r2) or
            (ignore_chiral and inspect(chg1, r1, chg2, -r2)))
is_same_mol = same_mol

def chiral_mol(mol1, mol2=None):
    '''Detect whether the given molecule is chiral molecule or two molecules
    are chiral isomers.
    '''
    if mol2 is None:
        mol2 = mol1.copy()
        ptr_coord = mol2._atm[:,PTR_COORD]
        mol2._env[ptr_coord  ] *= -1
        mol2._env[ptr_coord+1] *= -1
        mol2._env[ptr_coord+2] *= -1
    return (not same_mol(mol1, mol2, ignore_chiral=False) and
            same_mol(mol1, mol2, ignore_chiral=True))

def inertia_moment(mol, mass=None, coords=None):
    if mass is None:
        mass = mol.atom_mass_list()
    if coords is None:
        coords = mol.atom_coords()
    mass_center = numpy.einsum('i,ij->j', mass, coords)/mass.sum()
    coords = coords - mass_center
    im = numpy.einsum('i,ij,ik->jk', mass, coords, coords)
    im = numpy.eye(3) * im.trace() - im
    return im

def atom_mass_list(mol, isotope_avg=False):
    '''A list of mass for all atoms in the molecule

    Kwargs:
        isotope_avg : boolean
            Whether to use the isotope average mass as the atomic mass
    '''
    if isotope_avg:
        mass_table = elements.MASSES
    else:
        mass_table = elements.ISOTOPE_MAIN

    nucprop = mol.nucprop
    if nucprop:
        mass = []
        for ia in range(mol.natm):
            z = mol.atom_charge(ia)
            symb = mol.atom_symbol(ia)
            stdsymb = _std_symbol(symb)
            if ia+1 in nucprop:
                prop = nucprop[ia+1]
            elif symb in nucprop:
                prop = nucprop[symb]
            else:
                prop = nucprop.get(stdsymb, {})

            mass.append(prop.get('mass', mass_table[z]))
    else:
        #mass = [mass_table[z] for z in mol.atom_charges()]
        mass = []
        for ia in range(mol.natm):
            z = charge(mol.atom_symbol(ia))
            mass.append(mass_table[z])

    return numpy.array(mass)

def condense_to_shell(mol, mat, compressor='max'):
    '''The given matrix is first partitioned to blocks, based on AO shell as
    delimiter. Then call compressor function to abstract each block.

    Args:
        compressor: string or function
            if compressor is a string, its value can be  sum, max, min, abssum,
            absmax, absmin, norm
    '''
    ao_loc = mol.ao_loc_nr()
    if callable(compressor):
        abstract = numpy.empty((mol.nbas, mol.nbas))
        for i, i0 in enumerate(ao_loc[:mol.nbas]):
            for j, j0 in enumerate(ao_loc[:mol.nbas]):
                abstract[i,j] = compressor(mat[i0:ao_loc[i+1],j0:ao_loc[j+1]])
    else:
        abstract = lib.condense(compressor, mat, ao_loc)
    return abstract

def get_overlap_cond(mol, shls_slice=None):
    '''Overlap magnitudes measured by -log(overlap) between two shells

    Args:
        mol : an instance of :class:`Mole`

    Returns:
        2D mask array of shape (nbas,nbas)
    '''
    nbas = mol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    cond = numpy.zeros((nbas, nbas))
    moleintor.libcgto.GTOoverlap_cond(
        cond.ctypes.data_as(ctypes.c_void_p), (ctypes.c_int * 4)(*shls_slice),
        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        mol._env.ctypes.data_as(ctypes.c_void_p))
    return cond


def tostring(mol, format='raw'):
    '''Convert molecular geometry to a string of the required format.

    Supported output formats:
        | raw: Each line is  <symbol> <x> <y> <z>
        | xyz: XYZ cartesian coordinates format
        | zmat: Z-matrix format
    '''
    format = format.lower()
    if format == 'xyz' or format == 'raw':
        coords = mol.atom_coords() * param.BOHR
        output = []
        if format == 'xyz':
            output.append('%d' % mol.natm)
            output.append('XYZ from PySCF')

        for i in range(mol.natm):
            symb = mol.atom_pure_symbol(i)
            x, y, z = coords[i]
            output.append('%-4s %17.8f %17.8f %17.8f' %
                          (symb, x, y, z))
        return '\n'.join(output)
    elif format == 'zmat':
        coords = mol.atom_coords() * param.BOHR
        zmat = cart2zmat(coords).splitlines()
        output = []
        for i, line in enumerate(zmat):
            symb = mol.atom_pure_symbol(i)
            output.append('%-4s   %s' % (symb, line))
        return '\n'.join(output)
    else:
        raise NotImplementedError(f'format={format}')

def tofile(mol, filename, format=None):
    '''Write molecular geometry to a file of the required format.

    Supported output formats:
        | raw: Each line is  <symbol> <x> <y> <z>
        | xyz: XYZ cartesian coordinates format
        | zmat: Z-matrix format
    '''
    if format is None:  # Guess format based on filename
        format = os.path.splitext(filename)[1][1:]
    string = tostring(mol, format)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(string)
        f.write('\n')
    return string

def fromfile(filename, format=None):
    '''Read molecular geometry from a file
    (in testing)

    Supported formats:
        | raw: Each line is  <symbol> <x> <y> <z>
        | xyz: XYZ cartesian coordinates format
        | zmat: Z-matrix format
    '''
    if format is None:  # Guess format based on filename
        format = os.path.splitext(filename)[1][1:].lower()
        if format not in ('xyz', 'zmat', 'sdf'):
            format = 'raw'
    with open(filename, 'r') as f:
        return fromstring(f.read(), format)


def fromstring(string, format='xyz'):
    '''Convert the string of the specified format to internal format
    (in testing)

    Supported formats:
        | raw: Each line is  <symbol> <x> <y> <z>
        | xyz: XYZ cartesian coordinates format
        | zmat: Z-matrix format
    '''
    format = format.lower()
    if format == 'zmat':
        return from_zmatrix(string)
    elif format == 'xyz':
        line, title, geom = string.split('\n', 2)
        return geom
    elif format == 'sdf':
        raw = string.splitlines()
        natoms, nbonds = raw[3].split()[:2]
        atoms = []
        for line in raw[4:4+int(natoms)]:
            d = line.split()
            atoms.append('%s %s %s %s' % (d[3], d[0], d[1], d[2]))
        return '\n'.join(atoms)
    elif format == 'raw':
        return string
    else:
        raise NotImplementedError

def is_au(unit):
    '''Return whether the unit is recognized as A.U. or not
    '''
    return unit.upper().startswith(('B', 'AU'))

#
# MoleBase handles three layers of basis data: input, internal format, libcint arguments.
# The relationship of the three layers are
#    .atom (input) <=>  ._atom (for python) <=> ._atm (for libcint)
#   .basis (input) <=> ._basis (for python) <=> ._bas (for libcint)
# input layer does not talk to libcint directly.  Data are held in python
# internal format layer.  Most of methods defined in this class only operates
# on the internal format.  Exceptions are make_env, make_atm_env, make_bas_env,
# set_common_orig_, set_rinv_orig_ which are used to manipulate the libcint arguments.
#
class MoleBase(lib.StreamObject):
    '''Basic class to hold molecular structure, integrals and global options

    Attributes:
        verbose : int
            Print level
        output : str or None
            Output file, default is None which dumps msg to sys.stdout
        max_memory : int, float
            Allowed memory in MB
        charge : int
            Charge of molecule. It affects the electron numbers
        spin : int or None
            2S, num. alpha electrons - num. beta electrons to control
            multiplicity. If spin = None is set, multiplicity will be guessed
            based on the neutral molecule.
        symmetry : bool or str
            Whether to use symmetry.  When this variable is set to True, the
            molecule will be rotated and the highest rotation axis will be
            placed z-axis.
            If a string is given as the name of point group, the given point
            group symmetry will be used.  Note that the input molecular
            coordinates will not be changed in this case.
        symmetry_subgroup : str
            subgroup

        atom : list or str
            To define molecular structure.  The internal format is

            | atom = [[atom1, (x, y, z)],
            |         [atom2, (x, y, z)],
            |         ...
            |         [atomN, (x, y, z)]]

        unit : str
            Angstrom or Bohr
        basis : dict or str
            To define basis set.
        nucmod : dict or str or [function(nuc_charge, nucprop) => zeta]
            Nuclear model.  0 or None means point nuclear model.  Other
            values will enable Gaussian nuclear model.  If a function is
            assigned to this attribute, the function will be called to
            generate the nuclear charge distribution value "zeta" and the
            relevant nuclear model will be set to Gaussian model.
            Default is point nuclear model.
        nucprop : dict
            Nuclear properties (like g-factor 'g', quadrupole moments 'Q').
            It is needed by pyscf.prop module and submodules.
        cart : boolean
            Using Cartesian GTO basis and integrals (6d,10f,15g)
        magmom : list
            Collinear spin of each atom. Default is [0,]*natm

        ** Following attributes are generated by :func:`Mole.build` **

        stdout : file object
            Default is sys.stdout if :attr:`Mole.output` is not set
        topgroup : str
            Point group of the system.
        groupname : str
            The supported subgroup of the point group. It can be one of SO3,
            Dooh, Coov, D2h, C2h, C2v, D2, Cs, Ci, C2, C1
        nelectron : int
            sum of nuclear charges - :attr:`Mole.charge`
        symm_orb : a list of numpy.ndarray
            Symmetry adapted basis.  Each element is a set of symm-adapted orbitals
            for one irreducible representation.  The list index does **not** correspond
            to the id of irreducible representation.
        irrep_id : a list of int
            Each element is one irreducible representation id associated with the basis
            stored in symm_orb.  One irrep id stands for one irreducible representation
            symbol.  The irrep symbol and the relevant id are defined in
            :attr:`symm.param.IRREP_ID_TABLE`
        irrep_name : a list of str
            Each element is one irreducible representation symbol associated with the basis
            stored in symm_orb.  The irrep symbols are defined in
            :attr:`symm.param.IRREP_ID_TABLE`
        _built : bool
            To label whether :func:`Mole.build` has been called.  It is to
            ensure certain functions being initialized only once.
        _basis : dict
            like :attr:`Mole.basis`, the internal format which is returned from the
            parser :func:`format_basis`

        ** Following attributes are arguments used by ``libcint`` library **

        _atm :
            :code:`[[charge, ptr-of-coord, nuc-model, ptr-zeta, 0, 0], [...]]`
            each element represents one atom
        natm :
            number of atoms
        _bas :
            :code:`[[atom-id, angular-momentum, num-primitive-GTO, num-contracted-GTO, 0, ptr-of-exps, ptr-of-contract-coeff, 0], [...]]`
            each element represents one shell
        nbas :
            number of shells
        _env :
            list of floats to store the coordinates, GTO exponents, contract-coefficients

    Examples:

    >>> mol = Mole(atom='H^2 0 0 0; H 0 0 1.1', basis='sto3g').build()
    >>> print(mol.atom_symbol(0))
    H^2
    >>> print(mol.atom_pure_symbol(0))
    H
    >>> print(mol.nao_nr())
    2
    >>> print(mol.intor('int1e_ovlp_sph'))
    [[ 0.99999999  0.43958641]
     [ 0.43958641  0.99999999]]
    >>> mol.charge = 1
    >>> mol.build()
    <class 'pyscf.gto.mole.Mole'> has no attributes Charge
    '''  # noqa: E501

    output = None
    max_memory = param.MAX_MEMORY

    verbose = getattr(__config__, 'VERBOSE', logger.NOTE)

    # the unit (angstrom/bohr) of the coordinates defined by the input self.atom
    unit = getattr(__config__, 'UNIT', 'angstrom')

    # Whether to hold everything in memory
    incore_anyway = getattr(__config__, 'INCORE_ANYWAY', False)

    # Using cartesian GTO (6d,10f,15g)
    cart = getattr(__config__, 'gto_mole_Mole_cart', False)
    charge = 0
    spin = 0 # 2j == nelec_alpha - nelec_beta
    symmetry = False
    symmetry_subgroup = None

    # Store the keys appeared in the module.  It is used to check misinput attributes
    _keys = {
        'verbose', 'unit', 'incore_anyway', 'output', 'max_memory',
        'cart', 'charge', 'spin', 'symmetry', 'symmetry_subgroup',
        'atom', 'basis', 'nucmod', 'ecp', 'nucprop', 'magmom', 'pseudo',
        'groupname', 'topgroup', 'symm_orb', 'irrep_id', 'irrep_name',
    }

    def __init__(self):
        # self.atom = [(symb/nuc_charge, (coord(Angstrom):0.,0.,0.)), ...]
        self.atom = []
        # self.basis = {atom_type/nuc_charge: [l, kappa, (expnt, c_1, c_2,..),..]}
        self.basis = 'sto-3g'
        # self.nucmod = {atom_symbol: nuclear_model, atom_id: nuc_mod}, atom_id is 1-based
        self.nucmod = {}
        # self.ecp = {atom_symbol: [[l, (r_order, expnt, c),...]]}
        self.ecp = {}
        # Nuclear property. self.nucprop = {atom_symbol: {key: value}}
        self.nucprop = {}
        # Collinear spin of each atom. self.magmom = [0, ...]
        self.magmom = []
        self.pseudo = None
##################################################
# don't modify the following private variables, they are not input options
        self._atm = numpy.zeros((0,6), dtype=numpy.int32)
        self._bas = numpy.zeros((0,8), dtype=numpy.int32)
        self._env = numpy.zeros(PTR_ENV_START)
        self._ecpbas = numpy.zeros((0,8), dtype=numpy.int32)

        self.groupname = 'C1'
        self.topgroup = 'C1'
        self.symm_orb = None
        self.irrep_id = None
        self.irrep_name = None
        self._symm_orig = None
        self._symm_axes = None
        self._nelectron = None
        self._nao = None
        self._enuc = None
        self._atom = []
        self._basis = {}
        self._ecp = {}
        self._pseudo = {}

        self._built = False
        # Some methods modify ._env. These method are executed in the context
        # _TemporaryMoleContext which is protected by the _ctx_lock.
        self._ctx_lock = None

    @property
    def natm(self):
        return len(self._atm)
    @property
    def nbas(self):
        return len(self._bas)

    @property
    def nelec(self):
        ne = self.nelectron
        nalpha = (ne + self.spin) // 2
        nbeta = nalpha - self.spin
        assert (nalpha >= 0 and nbeta >= 0)
        if nalpha + nbeta != ne:
            raise RuntimeError('Electron number %d and spin %d are not consistent\n'
                               'Note mol.spin = 2S = Nalpha - Nbeta, not 2S+1' %
                               (ne, self.spin))
        return nalpha, nbeta
    @nelec.setter
    def nelec(self, neleca_nelecb):
        neleca, nelecb = neleca_nelecb
        self._nelectron = neleca + nelecb
        self.spin = neleca - nelecb

    @property
    def nelectron(self):
        if self._nelectron is None:
            return self.tot_electrons()
        else:
            return self._nelectron
    @nelectron.setter
    def nelectron(self, n):
        self._nelectron = n

    @property
    def multiplicity(self):
        return self.spin + 1
    @multiplicity.setter
    def multiplicity(self, x):
        if x is None:
            self.spin = None
        else:
            self.spin = x - 1

    @property
    def ms(self):
        '''Spin quantum number. multiplicity = ms*2+1'''
        if self.spin % 2 == 0:
            return self.spin // 2
        else:
            return self.spin * .5
    @ms.setter
    def ms(self, x):
        if x is None:
            self.spin = None
        else:
            self.spin = int(round(2*x, 4))

    @property
    def enuc(self):
        '''nuclear repulsion energy'''
        if self._enuc is None:
            self._enuc = self.energy_nuc()
        return self._enuc
    @enuc.setter
    def enuc(self, x):
        self._enuc = x

    copy = copy

    pack = pack

    @classmethod
    @lib.with_doc(unpack.__doc__)
    def unpack(cls, moldic):
        return unpack(moldic)

    @lib.with_doc(unpack.__doc__)
    def unpack_(self, moldic):
        self.__dict__.update(moldic)
        return self

    dumps = dumps

    @classmethod
    @lib.with_doc(loads.__doc__)
    def loads(cls, molstr):
        return loads(molstr)

    @lib.with_doc(loads.__doc__)
    def loads_(self, molstr):
        self.__dict__.update(loads(molstr).__dict__)
        return self

    # when pickling, serialize as a JSON-formatted string
    __getstate__ = dumps
    __setstate__ = loads_

    def build(self, dump_input=DUMPINPUT, parse_arg=ARGPARSE,
              verbose=None, output=None, max_memory=None,
              atom=None, basis=None, unit=None, nucmod=None, ecp=None, pseudo=None,
              charge=None, spin=0, symmetry=None, symmetry_subgroup=None,
              cart=None, magmom=None):
        '''Setup molecule and initialize some control parameters.  Whenever you
        change the value of the attributes of :class:`Mole`, you need call
        this function to refresh the internal data of Mole.

        Kwargs:
            dump_input : bool
                whether to dump the contents of input file in the output file
            parse_arg : bool
                whether to read the sys.argv and overwrite the relevant parameters
            verbose : int
                Print level.  If given, overwrite :attr:`Mole.verbose`
            output : str or None
                Output file.  If given, overwrite :attr:`Mole.output`
            max_memory : int, float
                Allowd memory in MB.  If given, overwrite :attr:`Mole.max_memory`
            atom : list or str
                To define molecular structure.
            basis : dict or str
                To define basis set.
            nucmod : dict or str
                Nuclear model.  If given, overwrite :attr:`Mole.nucmod`
            charge : int
                Charge of molecule. It affects the electron numbers
                If given, overwrite :attr:`Mole.charge`
            spin : int
                2S, num. alpha electrons - num. beta electrons to control
                multiplicity. If setting spin = None , multiplicity will be
                guessed based on the neutral molecule.
                If given, overwrite :attr:`Mole.spin`
            symmetry : bool or str
                Whether to use symmetry.  If given a string of point group
                name, the given point group symmetry will be used.
            magmom : list
                Collinear spin of each atom. Default is [0.0,]*natm

        '''
        if isinstance(dump_input, str):
            sys.stderr.write('Assigning the first argument %s to mol.atom\n' %
                             dump_input)
            dump_input, atom = True, dump_input

        if verbose is not None: self.verbose = verbose
        if output is not None: self.output = output
        if max_memory is not None: self.max_memory = max_memory
        if atom is not None: self.atom = atom
        if basis is not None: self.basis = basis
        if unit is not None: self.unit = unit
        if nucmod is not None: self.nucmod = nucmod
        if ecp is not None: self.ecp = ecp
        if pseudo is not None: self.pseudo = pseudo
        if charge is not None: self.charge = charge
        if spin != 0: self.spin = spin
        if symmetry is not None: self.symmetry = symmetry
        if symmetry_subgroup is not None: self.symmetry_subgroup = symmetry_subgroup
        if cart is not None: self.cart = cart
        if magmom is not None: self.magmom = magmom

        if parse_arg:
            _update_from_cmdargs_(self)

        # avoid opening output file twice
        if (self.output is not None
            # StringIO() does not have attribute 'name'
            and getattr(self.stdout, 'name', None) != self.output):

            if self.verbose > logger.QUIET:
                if os.path.isfile(self.output):
                    print('overwrite output file: %s' % self.output)
                else:
                    print('output file: %s' % self.output)

            if self.output == '/dev/null':
                self.stdout = open(os.devnull, 'w', encoding='utf-8')
            else:
                self.stdout = open(self.output, 'w', encoding='utf-8')

        if self.atom:
            self._atom = self.format_atom(self.atom, unit=self.unit)
        uniq_atoms = {a[0] for a in self._atom}

        if self.basis:
            _basis = _parse_default_basis(self.basis, uniq_atoms)
            self._basis = self.format_basis(_basis)
        env = self._env[:PTR_ENV_START]
        self._atm, self._bas, self._env = \
                self.make_env(self._atom, self._basis, env, self.nucmod,
                              self.nucprop)

        if self.pseudo:
            self.ecp, self.pseudo = classify_ecp_pseudo(self, self.ecp, self.pseudo)

        if self.ecp:
            # Unless explicitly input, ECP should not be assigned to ghost atoms
            atoms_wo_ghost = [a for a in uniq_atoms if not is_ghost_atom(a)]
            _ecp = _parse_default_basis(self.ecp, atoms_wo_ghost)
            self._ecp = self.format_ecp(_ecp)
            if self._ecp:
                self._atm, self._ecpbas, self._env = \
                        self.make_ecp_env(self._atm, self._ecp, self._env)

        if self.pseudo:
            # Unless explicitly input, PP should not be assigned to ghost atoms
            atoms_wo_ghost = [a for a in uniq_atoms if not is_ghost_atom(a)]
            _pseudo = _parse_default_basis(self.pseudo, atoms_wo_ghost)
            self._pseudo = _pseudo = self.format_pseudo(_pseudo)
            if _pseudo:
                conflicts = set(_pseudo).intersection(self._ecp)
                if conflicts:
                    raise RuntimeError('Pseudo potential for atoms %s are defined '
                                       'in both .ecp and .pseudo.' % list(conflicts))

                for ia, atom in enumerate(self._atom):
                    symb = atom[0]
                    if (symb in _pseudo and
                        # skip ghost atoms
                        self._atm[ia,0] != 0):
                        self._atm[ia,0] = sum(_pseudo[symb][0])

        if self.spin is None:
            self.spin = self.nelectron % 2
        else:
            # Access self.nelec in which the code checks whether the spin and
            # number of electrons are consistent.
            self.nelec

        # reset nuclear energy
        self.enuc = None

        if not self.magmom:
            self.magmom = [0,] * self.natm
        elif len(self.magmom) != self.natm:
            logger.warn(self, 'len(magmom) != natm. Set magmom to zero')
            self.magmom = [0,] * self.natm
        elif isinstance(self.magmom, np.ndarray):
            self.magmom = self.magmom.tolist()
        if self.spin == 0 and abs(numpy.sum(self.magmom) - self.spin) > 1e-6:
            #don't check for unrestricted calcs.
            raise ValueError("mol.magmom is set incorrectly.")

        if self.symmetry:
            self._build_symmetry()

        if dump_input and not self._built and self.verbose > logger.NOTE:
            self.dump_input()

        if self.verbose >= logger.WARN:
            self.check_sanity()

        if self.verbose >= logger.DEBUG3:
            logger.debug3(self, 'arg.atm = %s', self._atm)
            logger.debug3(self, 'arg.bas = %s', self._bas)
            logger.debug3(self, 'arg.env = %s', self._env)
            logger.debug3(self, 'ecpbas  = %s', self._ecpbas)

        self._built = True
        return self
    kernel = build

    def check_sanity(self):
        if isinstance(self.ecp, str):
            return self

        if isinstance(self.basis, str) and not self.ecp:
            elements = [x for x, _ in self._atom]
            ecp, ecp_atoms = bse_predefined_ecp(self.basis, elements)
            if ecp_atoms:
                logger.warn(self, f'ECP not specified. The basis set {self.basis} '
                            f'include an ECP. Recommended ECP: {ecp}.')
        elif isinstance(self.basis, dict) and isinstance(self.ecp, dict):
            for element, basname in self.basis.items():
                if isinstance(basname, str) and not self.ecp.get(element):
                    ecp, ecp_atoms = bse_predefined_ecp(basname, element)
                    if ecp_atoms:
                        logger.warn(self, f'ECP for {element} not specified. '
                                    f'The basis set {basname} include an ECP. '
                                    f'Recommended ECP: {ecp}.')
        return self

    def _build_symmetry(self, *args, **kwargs):
        '''
        Update symmetry related attributes: topgroup, groupname, _symm_orig,
        _symm_axes, irrep_id, irrep_name, symm_orb
        '''
        from pyscf import symm

        # TODO: Consider ECP info in point group symmetry initialization
        self.topgroup, orig, axes = symm.detect_symm(self._atom, self._basis)

        if isinstance(self.symmetry, str):
            self.symmetry = str(symm.std_symb(self.symmetry))
            groupname = None
            if abs(axes - np.eye(3)).max() < symm.TOLERANCE:
                if symm.check_symm(self.symmetry, self._atom, self._basis):
                    # Try to use original axes (issue #1209)
                    groupname = self.symmetry
                    axes = np.eye(3)
                else:
                    logger.warn(self, 'Unable to to identify input symmetry using original axes.\n'
                                'Different symmetry axes will be used.')
            if groupname is None:
                try:
                    groupname, axes = symm.as_subgroup(self.topgroup, axes,
                                                       self.symmetry)
                except PointGroupSymmetryError as e:
                    raise PointGroupSymmetryError(
                        'Unable to identify input symmetry %s. Try symmetry="%s"' %
                        (self.symmetry, self.topgroup)) from e
        else:
            groupname, axes = symm.as_subgroup(self.topgroup, axes,
                                               self.symmetry_subgroup)
        self._symm_orig = orig
        self._symm_axes = axes

        if self.cart and groupname in ('Dooh', 'Coov', 'SO3'):
            if groupname == 'Coov':
                groupname, lgroup = 'C2v', groupname
            else:
                groupname, lgroup = 'D2h', groupname
            logger.warn(self, 'This version does not support symmetry %s '
                        'for cartesian GTO basis. Its subgroup %s is used',
                        lgroup, groupname)
        self.groupname = groupname

        self.symm_orb, self.irrep_id = \
                symm.symm_adapted_basis(self, groupname, orig, axes)
        self.irrep_name = [symm.irrep_id2name(groupname, ir)
                           for ir in self.irrep_id]
        return self

    format_atom = staticmethod(format_atom)
    format_basis = staticmethod(format_basis)
    format_pseudo = staticmethod(format_pseudo)
    format_ecp = staticmethod(format_ecp)
    expand_etb = staticmethod(expand_etb)
    expand_etbs = etbs = staticmethod(expand_etbs)

    @lib.with_doc(make_env.__doc__)
    def make_env(self, atoms, basis, pre_env=[], nucmod={}, nucprop=None):
        if nucprop is None:
            nucprop = self.nucprop
        return make_env(atoms, basis, pre_env, nucmod, nucprop)

    @lib.with_doc(make_atm_env.__doc__)
    def make_atm_env(self, atom, ptr=0, nucmod=NUC_POINT, nucprop=None):
        if nucprop is None:
            nucprop = self.nucprop.get(atom[0], {})
        return make_atm_env(atom, ptr, nucmod, nucprop)

    @lib.with_doc(make_bas_env.__doc__)
    def make_bas_env(self, basis_add, atom_id=0, ptr=0):
        return make_bas_env(basis_add, atom_id, ptr)

    @lib.with_doc(make_ecp_env.__doc__)
    def make_ecp_env(self, _atm, _ecp, pre_env=[]):
        if _ecp:
            _atm, _ecpbas, _env = make_ecp_env(self, _atm, _ecp, pre_env)
        else:
            _atm, _ecpbas, _env = _atm, numpy.zeros((0,BAS_SLOTS)), pre_env
        return _atm, _ecpbas, _env

    tot_electrons = tot_electrons

    @lib.with_doc(gto_norm.__doc__)
    def gto_norm(self, l, expnt):
        return gto_norm(l, expnt)

    def dump_input(self):
        import __main__
        if hasattr(__main__, '__file__'):
            try:
                filename = os.path.abspath(__main__.__file__)
                finput = open(filename, 'r')
                self.stdout.write('#INFO: **** input file is %s ****\n' % filename)
                self.stdout.write(finput.read())
                self.stdout.write('#INFO: ******************** input file end ********************\n')
                self.stdout.write('\n')
                self.stdout.write('\n')
                finput.close()
            except IOError:
                logger.warn(self, 'input file does not exist')

        self.stdout.write('\n'.join(lib.misc.format_sys_info()))

        self.stdout.write('\n\n')
        for key in os.environ:
            if 'PYSCF' in key:
                self.stdout.write('[ENV] %s %s\n' % (key, os.environ[key]))
        if self.verbose >= logger.DEBUG2:
            for key in dir(__config__):
                if key[:2] != '__':
                    self.stdout.write('[CONFIG] %s = %s\n' %
                                      (key, getattr(__config__, key)))
        else:
            conf_file = getattr(__config__, 'conf_file', None)
            self.stdout.write('[CONFIG] conf_file %s\n' % conf_file)

        self.stdout.write('[INPUT] verbose = %d\n' % self.verbose)
        if self.verbose >= logger.DEBUG:
            self.stdout.write('[INPUT] max_memory = %s \n' % self.max_memory)
        self.stdout.write('[INPUT] num. atoms = %d\n' % self.natm)
        self.stdout.write('[INPUT] num. electrons = %d\n' % self.nelectron)
        self.stdout.write('[INPUT] charge = %d\n' % self.charge)
        self.stdout.write('[INPUT] spin (= nelec alpha-beta = 2S) = %d\n' % self.spin)
        self.stdout.write('[INPUT] symmetry %s subgroup %s\n' %
                          (self.symmetry, self.symmetry_subgroup))
        self.stdout.write('[INPUT] Mole.unit = %s\n' % self.unit)
        if self.cart:
            self.stdout.write('[INPUT] Cartesian GTO integrals (6d 10f)\n')

        self.stdout.write('[INPUT] Symbol           X                Y                Z      unit'
                          '          X                Y                Z       unit  Magmom\n')
        for ia,atom in enumerate(self._atom):
            coorda = tuple([x * param.BOHR for x in atom[1]])
            coordb = tuple(atom[1])
            magmom = self.magmom[ia]
            self.stdout.write('[INPUT]%3d %-4s %16.12f %16.12f %16.12f AA  '
                              '%16.12f %16.12f %16.12f Bohr  %4.1f\n'
                              % ((ia+1, _symbol(atom[0])) + coorda + coordb + (magmom,)))
        if self.nucmod:
            if isinstance(self.nucmod, (int, str, types.FunctionType)):
                nucatms = [_symbol(atom[0]) for atom in self._atom]
            else:
                nucatms = self.nucmod.keys()
            self.stdout.write('[INPUT] Gaussian nuclear model for atoms %s\n' %
                              nucatms)

        if self.nucprop:
            self.stdout.write('[INPUT] nucprop %s\n' % self.nucprop)

        if self.verbose >= logger.DEBUG:
            self.stdout.write('[INPUT] ---------------- BASIS SET ---------------- \n')
            self.stdout.write('[INPUT] l, kappa, [nprim/nctr], '
                              'expnt,             c_1 c_2 ...\n')
            for atom, basis_set in self._basis.items():
                self.stdout.write('[INPUT] %s\n' % atom)
                for b in basis_set:
                    if isinstance(b[1], int):
                        kappa = b[1]
                        b_coeff = b[2:]
                    else:
                        kappa = 0
                        b_coeff = b[1:]
                    nprim = len(b_coeff)
                    nctr = len(b_coeff[0])-1
                    if nprim < nctr:
                        logger.warn(self, 'num. primitives smaller than num. contracted basis')
                    self.stdout.write('[INPUT] %d   %2d    [%-5d/%-4d]  '
                                      % (b[0], kappa, nprim, nctr))
                    for k, x in enumerate(b_coeff):
                        if k == 0:
                            self.stdout.write('%-15.12g  ' % x[0])
                        else:
                            self.stdout.write(' '*32+'%-15.12g  ' % x[0])
                        for c in x[1:]:
                            self.stdout.write(' %4.12g' % c)
                        self.stdout.write('\n')

        if self.verbose >= logger.INFO:
            self.stdout.write('\n')
            logger.info(self, 'nuclear repulsion = %.15g', self.enuc)

            if self.symmetry:
                if self.topgroup == self.groupname:
                    logger.info(self, 'point group symmetry = %s', self.topgroup)
                else:
                    logger.info(self, 'point group symmetry = %s, use subgroup %s',
                                self.topgroup, self.groupname)
                logger.info(self, "symmetry origin: %s", self._symm_orig)
                logger.info(self, "symmetry axis x: %s", self._symm_axes[0])
                logger.info(self, "symmetry axis y: %s", self._symm_axes[1])
                logger.info(self, "symmetry axis z: %s", self._symm_axes[2])
                for ir in range(self.symm_orb.__len__()):
                    logger.info(self, 'num. orbitals of irrep %s = %d',
                                self.irrep_name[ir], self.symm_orb[ir].shape[1])
            logger.info(self, 'number of shells = %d', self.nbas)
            logger.info(self, 'number of NR pGTOs = %d', self.npgto_nr())
            logger.info(self, 'number of NR cGTOs = %d', self.nao_nr())
            logger.info(self, 'basis = %s', self.basis)
            logger.info(self, 'ecp = %s', self.ecp)
        if self.verbose >= logger.DEBUG2:
            for i in range(len(self._bas)):
                exps = self.bas_exp(i)
                logger.debug1(self, 'bas %d, expnt(s) = %s', i, str(exps))

        logger.info(self, 'CPU time: %12.2f', logger.process_clock())
        return self

    def set_common_origin(self, coord):
        '''Update common origin for integrals of dipole, rxp etc.
        **Note** the unit of the coordinates needs to be Bohr

        Examples:

        >>> mol.set_common_origin(0)
        >>> mol.set_common_origin((1,0,0))
        '''
        self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3] = coord
        return self
    set_common_orig = set_common_origin
    set_common_orig_ = set_common_orig    # for backward compatibility
    set_common_origin_ = set_common_orig  # for backward compatibility

    def with_common_origin(self, coord):
        '''Return a temporary mol context which has the rquired common origin.
        The required common origin has no effects out of the temporary context.
        See also :func:`mol.set_common_origin`

        Examples:

        >>> with mol.with_common_origin((1,0,0)):
        ...     mol.intor('int1e_r', comp=3)
        '''
        coord0 = self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3].copy()
        return self._TemporaryMoleContext(self.set_common_origin, (coord,), (coord0,))
    with_common_orig = with_common_origin

    def set_rinv_origin(self, coord):
        r'''Update origin for operator :math:`\frac{1}{|r-R_O|}`.
        **Note** the unit is Bohr

        Examples:

        >>> mol.set_rinv_origin(0)
        >>> mol.set_rinv_origin((0,1,0))
        '''
        self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = coord[:3]
        return self
    set_rinv_orig = set_rinv_origin
    set_rinv_orig_ = set_rinv_orig    # for backward compatibility
    set_rinv_origin_ = set_rinv_orig  # for backward compatibility

    def with_rinv_origin(self, coord):
        '''Return a temporary mol context which has the rquired origin of 1/r
        operator.  The required origin has no effects out of the temporary
        context.  See also :func:`mol.set_rinv_origin`

        Examples:

        >>> with mol.with_rinv_origin((1,0,0)):
        ...     mol.intor('int1e_rinv')
        '''
        coord0 = self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3].copy()
        return self._TemporaryMoleContext(self.set_rinv_origin, (coord,), (coord0,))
    with_rinv_orig = with_rinv_origin

    def set_range_coulomb(self, omega):
        '''Switch on range-separated Coulomb operator for **all** 2e integrals

        Args:
            omega : double

                | = 0 : Regular electron repulsion integral
                | > 0 : Long-range operator  erf(omega r12) / r12
                | < 0 : Short-range operator  erfc(omega r12) /r12
        '''
        if omega is not None:
            self._env[PTR_RANGE_OMEGA] = omega
    set_range_coulomb_ = set_range_coulomb  # for backward compatibility

    @property
    def omega(self):
        return self._env[PTR_RANGE_OMEGA]
    omega = omega.setter(set_range_coulomb)

    def with_range_coulomb(self, omega):
        '''Return a temporary mol context which sets the required parameter
        omega for range-separated Coulomb operator.
        If omega = None, return the context for regular Coulomb integrals.
        See also :func:`mol.set_range_coulomb`

        Examples:

        >>> with mol.with_range_coulomb(omega=1.5):
        ...     mol.intor('int2e')
        '''
        if omega is None:
            return contextlib.nullcontext()
        omega0 = self._env[PTR_RANGE_OMEGA].copy()
        return self._TemporaryMoleContext(self.set_range_coulomb, (omega,), (omega0,))

    def with_long_range_coulomb(self, omega):
        '''Return a temporary mol context for long-range part of
        range-separated Coulomb operator.
        '''
        if omega is None:
            return contextlib.nullcontext()
        return self.with_range_coulomb(abs(omega))

    def with_short_range_coulomb(self, omega):
        '''Return a temporary mol context for short-range part of
        range-separated Coulomb operator.
        '''
        if omega is None:
            return contextlib.nullcontext()
        return self.with_range_coulomb(-abs(omega))

    def set_f12_zeta(self, zeta):
        '''Set zeta for YP exp(-zeta r12)/r12 or STG exp(-zeta r12) type integrals
        '''
        self._env[PTR_F12_ZETA] = zeta

    def set_nuc_mod(self, atm_id, zeta):
        '''Change the nuclear charge distribution of the given atom ID.  The charge
        distribution is defined as: rho(r) = nuc_charge * Norm * exp(-zeta * r^2).
        This function can **only** be called after .build() method is executed.

        Examples:

        >>> for ia in range(mol.natm):
        ...     zeta = gto.filatov_nuc_mod(mol.atom_charge(ia))
        ...     mol.set_nuc_mod(ia, zeta)
        '''
        ptr = self._atm[atm_id,PTR_ZETA]
        self._env[ptr] = zeta
        if zeta == 0:
            self._atm[atm_id,NUC_MOD_OF] = NUC_POINT
        else:
            self._atm[atm_id,NUC_MOD_OF] = NUC_GAUSS
        return self
    set_nuc_mod_ = set_nuc_mod  # for backward compatibility

    def set_rinv_zeta(self, zeta):
        '''Assume the charge distribution on the "rinv_origin".  zeta is the parameter
        to control the charge distribution: rho(r) = Norm * exp(-zeta * r^2).
        **Be careful** when call this function. It affects the behavior of
        int1e_rinv_* functions.  Make sure to set it back to 0 after using it!
        '''
        self._env[PTR_RINV_ZETA] = zeta
        return self
    set_rinv_zeta_ = set_rinv_zeta  # for backward compatibility

    def with_rinv_zeta(self, zeta):
        '''Return a temporary mol context which has the rquired Gaussian charge
        distribution placed at "rinv_origin": rho(r) = Norm * exp(-zeta * r^2).
        See also :func:`mol.set_rinv_zeta`

        Examples:

        >>> with mol.with_rinv_zeta(zeta=1.5), mol.with_rinv_origin((1.,0,0)):
        ...     mol.intor('int1e_rinv')
        '''
        zeta0 = self._env[PTR_RINV_ZETA].copy()
        return self._TemporaryMoleContext(self.set_rinv_zeta, (zeta,), (zeta0,))

    def with_rinv_at_nucleus(self, atm_id):
        '''Return a temporary mol context in which the rinv operator (1/r) is
        treated like the Coulomb potential of a Gaussian charge distribution
        rho(r) = Norm * exp(-zeta * r^2) at the place of the input atm_id.

        Examples:

        >>> with mol.with_rinv_at_nucleus(3):
        ...     mol.intor('int1e_rinv')
        '''
        zeta = self._env[self._atm[atm_id,PTR_ZETA]]
        rinv = self.atom_coord(atm_id)
        if zeta == 0:
            self._env[AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
            return self.with_rinv_origin(rinv)
        else:
            self._env[AS_RINV_ORIG_ATOM] = atm_id  # required by ecp gradients
            rinv0 = self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3].copy()
            zeta0 = self._env[PTR_RINV_ZETA].copy()

            def set_rinv(z, r):
                self._env[PTR_RINV_ZETA] = z
                self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = r
            return self._TemporaryMoleContext(set_rinv, (zeta,rinv), (zeta0,rinv0))
    with_rinv_as_nucleus = with_rinv_at_nucleus  # For backward compatibility

    def with_integral_screen(self, threshold):
        '''Return a temporary mol context which has the required integral
        screen threshold
        '''
        if threshold is None:
            # This calls the default cutoff settings in cint library
            expcutoff = 0
        else:
            expcutoff = abs(numpy.log(threshold))
        expcutoff0 = self._env[PTR_EXPCUTOFF]

        def set_cutoff(cut):
            self._env[PTR_EXPCUTOFF] = cut
        return self._TemporaryMoleContext(set_cutoff, (expcutoff,), (expcutoff0,))

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        '''Update geometry
        '''
        if inplace:
            mol = self
        else:
            mol = self.copy(deep=False)
            mol._env = mol._env.copy()
        if unit is None:
            unit = mol.unit
        else:
            mol.unit = unit
        if symmetry is None:
            symmetry = mol.symmetry

        if isinstance(atoms_or_coords, numpy.ndarray):
            mol.atom = list(zip([x[0] for x in mol._atom],
                                atoms_or_coords.tolist()))
        else:
            mol.atom = atoms_or_coords

        if isinstance(atoms_or_coords, numpy.ndarray) and not symmetry:
            if isinstance(unit, str):
                if is_au(unit):
                    unit = 1.
                else:
                    unit = 1./param.BOHR
            else:
                unit = 1./unit

            mol._atom = list(zip([x[0] for x in mol._atom],
                                 (atoms_or_coords * unit).tolist()))
            ptr = mol._atm[:,PTR_COORD]
            mol._env[ptr+0] = unit * atoms_or_coords[:,0]
            mol._env[ptr+1] = unit * atoms_or_coords[:,1]
            mol._env[ptr+2] = unit * atoms_or_coords[:,2]
            # reset nuclear energy
            mol.enuc = None
        else:
            mol.symmetry = symmetry
            mol.build(False, False)

        if mol.verbose >= logger.INFO:
            logger.info(mol, 'New geometry')
            for ia, atom in enumerate(mol._atom):
                coorda = tuple([x * param.BOHR for x in atom[1]])
                coordb = tuple(atom[1])
                coords = coorda + coordb
                logger.info(mol, ' %3d %-4s %16.12f %16.12f %16.12f AA  '
                            '%16.12f %16.12f %16.12f Bohr\n',
                            ia+1, mol.atom_symbol(ia), *coords)
        return mol

    def update(self, chkfile):
        return self.update_from_chk(chkfile)
    def update_from_chk(self, chkfile):
        with h5py.File(chkfile, 'r') as fh5:
            mol = loads(fh5['mol'][()])
            self.__dict__.update(mol.__dict__)
        return self

    def has_ecp(self):
        '''Whether pseudo potential is used in the system.'''
        return len(self._ecpbas) > 0 or self._pseudo

    def has_ecp_soc(self):
        '''Whether spin-orbit coupling is enabled in ECP.'''
        return (len(self._ecpbas) > 0 and
                numpy.any(self._ecpbas[:,SO_TYPE_OF] == 1))


    def atom_symbol(self, atm_id):
        r'''For the given atom id, return the input symbol (without striping special characters)

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H^2 0 0 0; H 0 0 1.1')
        >>> mol.atom_symbol(0)
        H^2
        '''
        return _symbol(self._atom[atm_id][0])

    def atom_pure_symbol(self, atm_id):
        r'''For the given atom id, return the standard symbol (striping special characters)

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H^2 0 0 0; H 0 0 1.1')
        >>> mol.atom_pure_symbol(0)
        H
        '''
        return _std_symbol(self._atom[atm_id][0])

    @property
    def elements(self):
        '''A list of elements in the molecule'''
        return [self.atom_pure_symbol(i) for i in range(self.natm)]

    def atom_charge(self, atm_id):
        r'''Nuclear effective charge of the given atom id
        Note "atom_charge /= charge(atom_symbol)" when ECP is enabled.
        Number of electrons screened by ECP can be obtained by charge(atom_symbol)-atom_charge

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.atom_charge(1)
        17
        '''
        if self._atm[atm_id,NUC_MOD_OF] != NUC_FRAC_CHARGE:
            # regular QM atoms
            return int(self._atm[atm_id,CHARGE_OF])
        else:
            # MM atoms with fractional charges
            return self._env[self._atm[atm_id,PTR_FRAC_CHARGE]]

    def atom_charges(self):
        '''np.asarray([mol.atom_charge(i) for i in range(mol.natm)])'''
        z = self._atm[:,CHARGE_OF]
        if numpy.any(self._atm[:,NUC_MOD_OF] == NUC_FRAC_CHARGE):
            # Create the integer nuclear charges first then replace the MM
            # particles with the MM charges that saved in _env[PTR_FRAC_CHARGE]
            z = numpy.array(z, dtype=numpy.double)
            idx = self._atm[:,NUC_MOD_OF] == NUC_FRAC_CHARGE
            # MM fractional charges can be positive or negative
            z[idx] = self._env[self._atm[idx,PTR_FRAC_CHARGE]]
        return z

    def atom_nelec_core(self, atm_id):
        '''Number of core electrons for pseudo potential.
        '''
        return charge(self.atom_symbol(atm_id)) - self.atom_charge(atm_id)

    def atom_coord(self, atm_id, unit='Bohr'):
        r'''Coordinates (ndarray) of the given atom id

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.atom_coord(1)
        [ 0.          0.          2.07869874]
        '''
        ptr = self._atm[atm_id,PTR_COORD]
        if not is_au(unit):
            return self._env[ptr:ptr+3] * param.BOHR
        else:
            return self._env[ptr:ptr+3].copy()

    def atom_coords(self, unit='Bohr'):
        '''np.asarray([mol.atom_coord(i) for i in range(mol.natm)])'''
        ptr = self._atm[:,PTR_COORD]
        c = self._env[ptr[:,None] + np.arange(3)]
        if not is_au(unit):
            c *= param.BOHR
        return c

    atom_mass_list = atom_mass_list

    def atom_nshells(self, atm_id):
        r'''Number of basis/shells of the given atom

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.atom_nshells(1)
        5
        '''
        return int((self._bas[:,ATOM_OF] == atm_id).sum())

    def atom_shell_ids(self, atm_id):
        r'''A list of the shell-ids of the given atom

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.atom_shell_ids(1)
        [3, 4, 5, 6, 7]
        '''
        return numpy.where(self._bas[:,ATOM_OF] == atm_id)[0]

    def bas_coord(self, bas_id):
        r'''Coordinates (ndarray) associated with the given basis id

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.bas_coord(1)
        [ 0.          0.          2.07869874]
        '''
        atm_id = self.bas_atom(bas_id)
        ptr = self._atm[atm_id,PTR_COORD]
        return self._env[ptr:ptr+3].copy()

    def bas_atom(self, bas_id):
        r'''The atom (0-based id) that the given basis sits on

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_atom(7)
        1
        '''
        return int(self._bas[bas_id,ATOM_OF])

    def bas_angular(self, bas_id):
        r'''The angular momentum associated with the given basis

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_angular(7)
        2
        '''
        return int(self._bas[bas_id,ANG_OF])

    def bas_nctr(self, bas_id):
        r'''The number of contracted GTOs for the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_nctr(3)
        3
        '''
        return int(self._bas[bas_id,NCTR_OF])

    def bas_nprim(self, bas_id):
        r'''The number of primitive GTOs for the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_nprim(3)
        11
        '''
        return int(self._bas[bas_id,NPRIM_OF])

    def bas_kappa(self, bas_id):
        r'''Kappa (if l < j, -l-1, else l) of the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_kappa(3)
        0
        '''
        return int(self._bas[bas_id,KAPPA_OF])

    def bas_exp(self, bas_id):
        r'''exponents (ndarray) of the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_exp(0)
        [ 13.01     1.962    0.4446]
        '''
        nprim = self.bas_nprim(bas_id)
        ptr = self._bas[bas_id,PTR_EXP]
        return self._env[ptr:ptr+nprim].copy()

    def bas_exps(self):
        '''exponents of all basis
        return [mol.bas_exp(i) for i in range(self.nbas)]
        '''
        nprims = self._bas[:,NPRIM_OF]
        pexps = self._bas[:,PTR_EXP]
        exps = [self._env[i0:i1] for i0, i1 in zip(pexps, pexps + nprims)]
        return exps

    def _libcint_ctr_coeff(self, bas_id):
        nprim = self.bas_nprim(bas_id)
        nctr = self.bas_nctr(bas_id)
        ptr = self._bas[bas_id,PTR_COEFF]
        return self._env[ptr:ptr+nprim*nctr].reshape(nctr,nprim).T

    def bas_ctr_coeff(self, bas_id):
        r'''Contract coefficients (ndarray) of the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.M(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_ctr_coeff(0)
        [[ 10.03400444]
         [  4.1188704 ]
         [  1.53971186]]
        '''
        l = self.bas_angular(bas_id)
        es = self.bas_exp(bas_id)
        cs = self._libcint_ctr_coeff(bas_id)
        cs = numpy.einsum('pi,p->pi', cs, 1/gto_norm(l, es))
        return cs

    def bas_len_spinor(self, bas_id):
        '''The number of spinor associated with given basis
        If kappa is 0, return 4l+2
        '''
        l = self.bas_angular(bas_id)
        k = self.bas_kappa(bas_id)
        return len_spinor(l, k)

    def bas_len_cart(self, bas_id):
        '''The number of Cartesian function associated with given basis
        '''
        return len_cart(self._bas[bas_id,ANG_OF])

    npgto_nr = npgto_nr

    nao_nr = nao_nr
    nao_2c = nao_2c
    nao_cart = nao_cart

    nao_nr_range = nao_nr_range
    nao_2c_range = nao_2c_range

    ao_loc_nr = ao_loc_nr
    ao_loc_2c = ao_loc_2c

    @property
    def nao(self):
        if self._nao is None:
            return self.nao_nr()
        else:
            return self._nao
    @nao.setter
    def nao(self, x):
        self._nao = x

    ao_loc = property(ao_loc_nr)

    tmap = time_reversal_map = time_reversal_map

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None, grids=None):
        '''Integral generator.

        Args:
            intor : str
                Name of the 1e or 2e AO integrals.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. int1e_ipovlp_sph has 3 components.
            hermi : int
                Symmetry of the integrals

                | 0 : no symmetry assumed (default)
                | 1 : hermitian
                | 2 : anti-hermitian

            shls_slice : 4-element, 6-element or 8-element tuple
                Label the start-stop shells for each index in the integral.
                For example, the 8-element tuple for the 2-electron integral
                tensor (ij|kl) = intor('int2e') are specified as
                (ish_start, ish_end, jsh_start, jsh_end, ksh_start, ksh_end, lsh_start, lsh_end)
            grids : ndarray
                Coordinates of grids for the int1e_grids integrals

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:

        >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
        >>> mol.intor('int1e_ipnuc_sph', comp=3) # <nabla i | V_nuc | j>
        [[[ 0.          0.        ]
          [ 0.          0.        ]]
         [[ 0.          0.        ]
          [ 0.          0.        ]]
         [[ 0.10289944  0.48176097]
          [-0.48176097 -0.10289944]]]
        >>> mol.intor('int1e_nuc_spinor')
        [[-1.69771092+0.j  0.00000000+0.j -0.67146312+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -1.69771092+0.j  0.00000000+0.j -0.67146312+0.j]
         [-0.67146312+0.j  0.00000000+0.j -1.69771092+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -0.67146312+0.j  0.00000000+0.j -1.69771092+0.j]]
        '''
        if not self._built:
            logger.warn(self, 'Warning: intor envs of %s not initialized.', self)
            # FIXME: Whether to check _built and call build?  ._bas and .basis
            # may not be consistent. calling .build() may leads to wrong intor env.
            #self.build(False, False)
        intor = self._add_suffix(intor)
        bas = self._bas
        env = self._env
        if 'ECP' in intor:
            assert (self._ecp is not None)
            bas = numpy.vstack((self._bas, self._ecpbas))
            env[AS_ECPBAS_OFFSET] = len(self._bas)
            env[AS_NECPBAS] = len(self._ecpbas)
            if shls_slice is None:
                shls_slice = (0, self.nbas, 0, self.nbas)
        elif '_grids' in intor:
            assert grids is not None
            env = numpy.append(env, grids.ravel())
            env[NGRIDS] = grids.shape[0]
            env[PTR_GRIDS] = env.size - grids.size
        return moleintor.getints(intor, self._atm, bas, env,
                                 shls_slice, comp, hermi, aosym, out=out)

    def _add_suffix(self, intor, cart=None):
        if not (intor[:4] == 'cint' or
                intor.endswith(('_sph', '_cart', '_spinor', '_ssc'))):
            if cart is None:
                cart = self.cart
            if cart:
                intor = intor + '_cart'
            else:
                intor = intor + '_sph'
        return intor

    def intor_symmetric(self, intor, comp=None, grids=None):
        '''One-electron integral generator. The integrals are assumed to be hermitian

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. int1e_ipovlp_sph has 3 components.
            grids : ndarray
                Coordinates of grids for the int1e_grids integrals

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:

        >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
        >>> mol.intor_symmetric('int1e_nuc_spinor')
        [[-1.69771092+0.j  0.00000000+0.j -0.67146312+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -1.69771092+0.j  0.00000000+0.j -0.67146312+0.j]
         [-0.67146312+0.j  0.00000000+0.j -1.69771092+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -0.67146312+0.j  0.00000000+0.j -1.69771092+0.j]]
        '''
        return self.intor(intor, comp, 1, aosym='s4', grids=grids)

    def intor_asymmetric(self, intor, comp=None, grids=None):
        '''One-electron integral generator. The integrals are assumed to be anti-hermitian

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. int1e_ipovlp has 3 components.
            grids : ndarray
                Coordinates of grids for the int1e_grids integrals

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:

        >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
        >>> mol.intor_asymmetric('int1e_nuc_spinor')
        [[-1.69771092+0.j  0.00000000+0.j  0.67146312+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -1.69771092+0.j  0.00000000+0.j  0.67146312+0.j]
         [-0.67146312+0.j  0.00000000+0.j -1.69771092+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -0.67146312+0.j  0.00000000+0.j -1.69771092+0.j]]
        '''
        return self.intor(intor, comp, 2, aosym='a4', grids=grids)

    @lib.with_doc(moleintor.getints_by_shell.__doc__)
    def intor_by_shell(self, intor, shells, comp=None, grids=None):
        intor = self._add_suffix(intor)
        if 'ECP' in intor:
            assert (self._ecp is not None)
            bas = numpy.vstack((self._bas, self._ecpbas))
            self._env[AS_ECPBAS_OFFSET] = len(self._bas)
            self._env[AS_NECPBAS] = len(self._ecpbas)
        else:
            bas = self._bas
        return moleintor.getints_by_shell(intor, shells, self._atm, bas,
                                          self._env, comp)

    eval_ao = eval_gto = eval_gto

    energy_nuc = energy_nuc
    def get_enuc(self):
        return self.enuc

    def get_ao_indices(self, bas_list, ao_loc=None):
        '''
        Generate (dis-continued) AO indices for basis specified in bas_list
        '''
        if ao_loc is None:
            ao_loc = self.ao_loc
        return lib.locs_to_indices(ao_loc, bas_list)

    sph_labels = spheric_labels = sph_labels
    cart_labels = cart_labels
    ao_labels = ao_labels

    spinor_labels = spinor_labels

    search_ao_label = search_ao_label

    def search_shell_id(self, atm_id, l):
        return search_shell_id(self, atm_id, l)

    search_ao_nr = search_ao_nr
    search_ao_r = search_ao_r

    aoslice_by_atom = aoslice_nr_by_atom = offset_ao_by_atom = offset_nr_by_atom = aoslice_by_atom
    aoslice_2c_by_atom = offset_2c_by_atom = offset_2c_by_atom

    condense_to_shell = condense_to_shell
    get_overlap_cond = get_overlap_cond

    to_uncontracted_cartesian_basis = to_uncontracted_cartesian_basis
    decontract_basis = decontract_basis

    ao_rotation_matrix = ao_rotation_matrix

    def cart2sph_coeff(self, normalized='sp'):
        '''Transformation matrix that transforms Cartesian GTOs to spherical
        GTOs for all basis functions

        Kwargs:
            normalized : string or boolean
                How the Cartesian GTOs are normalized.  Except s and p functions,
                Cartesian GTOs do not have the universal normalization coefficients
                for the different components of the same shell.  The value of this
                argument can be one of 'sp', 'all', None.  'sp' means the Cartesian s
                and p basis are normalized.  'all' means all Cartesian functions are
                normalized.  None means none of the Cartesian functions are normalized.
                The default value 'sp' is the convention used by libcint library.

        Examples:

        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvtz')
        >>> c = mol.cart2sph_coeff()
        >>> s0 = mol.intor('int1e_ovlp_sph')
        >>> s1 = c.T.dot(mol.intor('int1e_ovlp_cart')).dot(c)
        >>> print(abs(s1-s0).sum())
        >>> 4.58676826646e-15
        '''
        c2s_l = [cart2sph(l, normalized=normalized) for l in range(12)]
        c2s = []
        for ib in range(self.nbas):
            l = self.bas_angular(ib)
            for n in range(self.bas_nctr(ib)):
                c2s.append(c2s_l[l])
        return scipy.linalg.block_diag(*c2s)

    def sph2spinor_coeff(self):
        '''Transformation matrix that transforms real-spherical GTOs to spinor
        GTOs for all basis functions

        Examples:

        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvtz')
        >>> ca, cb = mol.sph2spinor_coeff()
        >>> s0 = mol.intor('int1e_ovlp_spinor')
        >>> s1 = ca.conj().T.dot(mol.intor('int1e_ovlp_sph')).dot(ca)
        >>> s1+= cb.conj().T.dot(mol.intor('int1e_ovlp_sph')).dot(cb)
        >>> print(abs(s1-s0).max())
        >>> 6.66133814775e-16
        '''
        from pyscf.symm import sph
        return sph.sph2spinor_coeff(self)

    def apply(self, fn, *args, **kwargs):
        if callable(fn):
            return lib.StreamObject.apply(self, fn, *args, **kwargs)
        elif isinstance(fn, str):
            method = getattr(self, fn.upper())
            return method(*args, **kwargs)
        else:
            raise TypeError('First argument of .apply method must be a '
                            'function/class or a name (string) of a method.')

    @contextlib.contextmanager
    def _TemporaryMoleContext(self, method, args, args_bak):
        '''Almost every method depends on the Mole environment. Ensure the
        modification in temporary environment being thread safe
        '''
        haslock = self._ctx_lock
        if haslock is None:
            self._ctx_lock = threading.RLock()

        with self._ctx_lock:
            method(*args)
            try:
                yield
            finally:
                method(*args_bak)
                if haslock is None:
                    self._ctx_lock = None


class Mole(MoleBase):
    '''A Mole object to hold the basic information of a molecule.
    '''

    __add__ = conc_mol
    inertia_moment = inertia_moment
    tostring = tostring
    tofile = tofile

    def __init__(self, **kwargs):
        MoleBase.__init__(self)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def fromstring(self, string, format='xyz'):
        '''Update the Mole object based on the input geometry string'''
        atom = self.format_atom(fromstring(string, format), unit=1)
        self.set_geom_(atom, unit='Angstrom', inplace=True)
        if format == 'sdf' and 'M  CHG' in string:
            raise NotImplementedError
            #FIXME self.charge = 0
        return self

    def fromfile(self, filename, format=None):
        '''Update the Mole object based on the input geometry file'''
        atom = self.format_atom(fromfile(filename, format), unit=1)
        self.set_geom_(atom, unit='Angstrom', inplace=True)
        if format == 'sdf':
            raise NotImplementedError
        return self

    def __getattr__(self, key):
        '''To support accessing methods (mol.HF, mol.KS, mol.CCSD, mol.CASSCF, ...)
        from Mole object.
        '''
        if key[0] == '_':  # Skip private attributes and Python builtins
            # https://bugs.python.org/issue45985
            # https://github.com/python/cpython/issues/103936
            # @property and __getattr__ conflicts. As a temporary fix, call
            # object.__getattribute__ method to re-raise AttributeError
            return object.__getattribute__(self, key)

        # Import all available modules. Some methods are registered to other
        # classes/modules when importing modules in __all__.
        from pyscf import __all__  # noqa
        from pyscf import scf, dft

        for mod in (scf, dft):
            method = getattr(mod, key, None)
            if callable(method):
                return method(self)

        if 'TD' in key[:3]:
            if key in ('TDHF', 'TDA'):
                mf = scf.HF(self)
            else:
                mf = dft.KS(self)
                xc = key.split('TD', 1)[1]
                if xc in dft.XC:
                    mf.xc = xc
                    key = 'TDDFT'
        elif 'CI' in key or 'CC' in key or 'CAS' in key or 'MP' in key:
            mf = scf.HF(self)
        else:
            return object.__getattribute__(self, key)

        method = getattr(mf, key)

        # Initialize SCF object for post-SCF methods if applicable
        if self.nelectron != 0:
            mf.run()
        return method

    def ao2mo(self, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
              **kwargs):
        '''Integral transformation for arbitrary orbitals and arbitrary
        integrals.  See more detailed documentation in func:`ao2mo.kernel`.

        Args:
            mo_coeffs (an np array or a list of arrays) : A matrix of orbital
                coefficients if it is a numpy ndarray, or four sets of orbital
                coefficients, corresponding to the four indices of (ij|kl).

        Kwargs:
            erifile (str or h5py File or h5py Group object) : The file/object
                to store the transformed integrals.  If not given, the return
                value is an array (in memory) of the transformed integrals.
            dataname : str
                *Note* this argument is effective if erifile is given.
                The dataset name in the erifile (ref the hierarchy of HDF5 format
                http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
                different dataname, the existed integral file can be reused.  If
                the erifile contains the specified dataname, the old integrals
                will be replaced by the new one under the key dataname.
            intor (str) : integral name Name of the 2-electron integral.  Ref
                to :func:`getints_by_shell`
                for the complete list of available 2-electron integral names

        Returns:
            An array of transformed integrals if erifile is not given.
            Otherwise, return the file/fileobject if erifile is assigned.


        Examples:

        >>> import pyscf
        >>> mol = pyscf.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
        >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
        >>> mo2 = numpy.random.random((mol.nao_nr(), 8))

        >>> eri1 = mol.ao2mo(mo1)
        >>> print(eri1.shape)
        (55, 55)

        >>> eri1 = mol.ao2mo(mo1, compact=False)
        >>> print(eri1.shape)
        (100, 100)

        >>> eri1 = mol.ao2mo(eri, (mo1,mo2,mo2,mo2))
        >>> print(eri1.shape)
        (80, 36)

        >>> eri1 = mol.ao2mo(eri, (mo1,mo2,mo2,mo2), erifile='water.h5')
        '''
        from pyscf import ao2mo
        return ao2mo.kernel(self, mo_coeffs, erifile, dataname, intor, **kwargs)

    def to_cell(self, a, dimension=3):
        '''Put a molecule in a cell with periodic boundary conditions

        Args:
            a : (3,3) ndarray
                Lattice primitive vectors. Each row is a lattice vector
        '''
        from pyscf.pbc.gto import Cell
        cell = Cell()
        cell.__dict__.update(self.__dict__)
        cell.dimension = dimension
        cell.build(False, False)
        return cell

def _parse_default_basis(basis, uniq_atoms):
    if isinstance(basis, (str, tuple, list)):
        # default basis for all atoms
        _basis = {a: basis for a in uniq_atoms}
    elif 'default' in basis:
        default_basis = basis['default']
        _basis = {a: default_basis for a in uniq_atoms}
        _basis.update(basis)
        del _basis['default']
    else:
        _basis = basis
    return _basis

def _parse_nuc_mod(str_or_int_or_fn):
    nucmod = NUC_POINT
    if callable(str_or_int_or_fn):
        nucmod = str_or_int_or_fn
    elif (isinstance(str_or_int_or_fn, str) and
          str_or_int_or_fn[0].upper() == 'G'): # 'gauss_nuc'
        nucmod = NUC_GAUSS
    elif str_or_int_or_fn != 0:
        nucmod = NUC_GAUSS
    return nucmod

def _update_from_cmdargs_(mol):
    try:
        # Detect whether in Ipython shell
        __IPYTHON__  # noqa: F821
        return
    except Exception:
        pass

    if not mol._built: # parse cmdline args only once
        opts = cmd_args.cmd_args()

        if opts.verbose:
            mol.verbose = opts.verbose
        if opts.max_memory:
            mol.max_memory = opts.max_memory

        if opts.output:
            mol.output = opts.output


def from_zmatrix(atomstr):
    '''>>> a = """H
    H 1 2.67247631453057
    H 1 4.22555607338457 2 50.7684795164077
    H 1 2.90305235726773 2 79.3904651036893 3 6.20854462618583"""
    >>> for x in zmat2cart(a): print(x)
    ['H', array([ 0.,  0.,  0.])]
    ['H', array([ 2.67247631,  0.        ,  0.        ])]
    ['H', array([ 2.67247631,  0.        ,  3.27310166])]
    ['H', array([ 0.53449526,  0.30859098,  2.83668811])]
    '''
    from pyscf.symm import rotation_mat
    atomstr = atomstr.replace(';','\n').replace(',',' ')
    symbols = []
    coord = []
    min_items_per_line = 1
    for line_id, line in enumerate(atomstr.splitlines()):
        line = line.strip()
        if line and line[0] != '#':
            rawd = line.split()
            if len(rawd) < min_items_per_line:
                raise ValueError('Zmatrix format error at L%d %s' % (line_id, line))

            symbols.append(rawd[0])
            if len(rawd) < 3:
                coord.append(numpy.zeros(3))
                min_items_per_line = 3
            elif len(rawd) == 3:
                if DISABLE_EVAL:
                    coord.append(numpy.array((float(rawd[2]), 0, 0)))
                else:
                    coord.append(numpy.array((eval(rawd[2]), 0, 0)))
                min_items_per_line = 5
            elif len(rawd) == 5:
                if DISABLE_EVAL:
                    vals = rawd[1:]
                else:
                    vals = eval(','.join(rawd[1:]))
                bonda = int(vals[0]) - 1
                bond  = float(vals[1])
                anga  = int(vals[2]) - 1
                ang   = float(vals[3])/180*numpy.pi
                assert (ang >= 0)
                v1 = coord[anga] - coord[bonda]
                if not numpy.allclose(v1[:2], 0):
                    vecn = numpy.cross(v1, numpy.array((0.,0.,1.)))
                else: # on z
                    vecn = numpy.array((0.,0.,1.))
                rmat = rotation_mat(vecn, ang)
                c = numpy.dot(rmat, v1) * (bond/numpy.linalg.norm(v1))
                coord.append(coord[bonda]+c)
                min_items_per_line = 7
            else:
                if DISABLE_EVAL:
                    vals = rawd[1:]
                else:
                    vals = eval(','.join(rawd[1:]))
                bonda = int(vals[0]) - 1
                bond  = float(vals[1])
                anga  = int(vals[2]) - 1
                ang   = float(vals[3])/180*numpy.pi
                assert (ang >= 0 and ang <= numpy.pi)
                v1 = coord[anga] - coord[bonda]
                v1 /= numpy.linalg.norm(v1)
                if ang < 1e-7:
                    c = v1 * bond
                elif numpy.pi-ang < 1e-7:
                    c = -v1 * bond
                else:
                    diha  = int(vals[4]) - 1
                    dih   = float(vals[5])/180*numpy.pi
                    v2 = coord[diha] - coord[anga]
                    vecn = numpy.cross(v2, -v1)
                    vecn_norm = numpy.linalg.norm(vecn)
                    if vecn_norm < 1e-7:
                        if not numpy.allclose(v1[:2], 0):
                            vecn = numpy.cross(v1, numpy.array((0.,0.,1.)))
                        else: # on z
                            vecn = numpy.array((0.,0.,1.))
                        rmat = rotation_mat(vecn, ang)
                        c = numpy.dot(rmat, v1) * bond
                    else:
                        rmat = rotation_mat(v1, -dih)
                        vecn = numpy.dot(rmat, vecn) / vecn_norm
                        rmat = rotation_mat(vecn, ang)
                        c = numpy.dot(rmat, v1) * bond
                coord.append(coord[bonda]+c)
    atoms = list(zip([_atom_symbol(x) for x in symbols], coord))
    return atoms
zmat2cart = zmat = from_zmatrix

def cart2zmat(coord):
    '''>>> c = numpy.array((
    (0.000000000000,  1.889726124565,  0.000000000000),
    (0.000000000000,  0.000000000000, -1.889726124565),
    (1.889726124565, -1.889726124565,  0.000000000000),
    (1.889726124565,  0.000000000000,  1.133835674739)))
    >>> print(cart2zmat(c))
    1
    1 2.67247631453057
    1 4.22555607338457 2 50.7684795164077
    1 2.90305235726773 2 79.3904651036893 3 6.20854462618583
    '''
    zstr = []
    zstr.append('1')
    if len(coord) > 1:
        r1 = coord[1] - coord[0]
        nr1 = numpy.linalg.norm(r1)
        zstr.append('1 %.15g' % nr1)
    if len(coord) > 2:
        r2 = coord[2] - coord[0]
        nr2 = numpy.linalg.norm(r2)
        a = numpy.arccos(numpy.dot(r1,r2)/(nr1*nr2))
        zstr.append('1 %.15g 2 %.15g' % (nr2, a*180/numpy.pi))
    if len(coord) > 3:
        o0, o1, o2 = coord[:3]
        p0, p1, p2 = 1, 2, 3
        for k, c in enumerate(coord[3:]):
            r0 = c - o0
            nr0 = numpy.linalg.norm(r0)
            r1 = o1 - o0
            nr1 = numpy.linalg.norm(r1)
            a1 = numpy.arccos(numpy.dot(r0,r1)/(nr0*nr1))
            b0 = numpy.cross(r0, r1)
            nb0 = numpy.linalg.norm(b0)

            if abs(nb0) < 1e-7: # o0, o1, c in line
                a2 = 0
                zstr.append('%d %.15g %d %.15g %d %.15g' %
                            (p0, nr0, p1, a1*180/numpy.pi, p2, a2))
            else:
                b1 = numpy.cross(o2-o0, r1)
                nb1 = numpy.linalg.norm(b1)

                if abs(nb1) < 1e-7:  # o0 o1 o2 in line
                    a2 = 0
                    zstr.append('%d %.15g %d %.15g %d %.15g' %
                                (p0, nr0, p1, a1*180/numpy.pi, p2, a2))
                    o2 = c
                    p2 = 4 + k
                else:
                    if numpy.dot(numpy.cross(b1, b0), r1) < 0:
                        a2 = numpy.arccos(numpy.dot(b1, b0) / (nb0*nb1))
                    else:
                        a2 =-numpy.arccos(numpy.dot(b1, b0) / (nb0*nb1))
                    zstr.append('%d %.15g %d %.15g %d %.15g' %
                                (p0, nr0, p1, a1*180/numpy.pi, p2, a2*180/numpy.pi))

    return '\n'.join(zstr)

def dyall_nuc_mod(nuc_charge, nucprop={}):
    ''' Generate the nuclear charge distribution parameter zeta
    rho(r) = nuc_charge * Norm * exp(-zeta * r^2)

    Ref. L. Visscher and K. Dyall, At. Data Nucl. Data Tables, 67, 207 (1997)
    '''
    mass = nucprop.get('mass', elements.ISOTOPE_MAIN[nuc_charge])
    r = (0.836 * mass**(1./3) + 0.570) / 52917.7249
    zeta = 1.5 / (r**2)
    return zeta

def filatov_nuc_mod(nuc_charge, nucprop={}):
    ''' Generate the nuclear charge distribution parameter zeta
    rho(r) = nuc_charge * Norm * exp(-zeta * r^2)

    Ref. M. Filatov and D. Cremer, Theor. Chem. Acc. 108, 168 (2002)
         M. Filatov and D. Cremer, Chem. Phys. Lett. 351, 259 (2002)
    '''
    c = param.LIGHT_SPEED
    nuc_charge = charge(nuc_charge)
    r = (-0.263188*nuc_charge + 106.016974 + 138.985999/nuc_charge) / c**2
    zeta = 1 / (r**2)
    return zeta

def fakemol_for_charges(coords, expnt=1e16):
    '''Construct a fake Mole object that holds the charges on the given
    coordinates (coords).  The shape of the charge can be a normal
    distribution with the Gaussian exponent (expnt).
    '''
    nbas = coords.shape[0]
    expnt = numpy.asarray(expnt).ravel()

    fakeatm = numpy.zeros((nbas,ATM_SLOTS), dtype=numpy.int32)
    fakebas = numpy.zeros((nbas,BAS_SLOTS), dtype=numpy.int32)
    fakeenv = [0] * PTR_ENV_START
    ptr = PTR_ENV_START
    fakeatm[:,PTR_COORD] = numpy.arange(ptr, ptr+nbas*3, 3)
    fakeenv.append(coords.ravel())
    ptr += nbas*3
    fakebas[:,ATOM_OF] = numpy.arange(nbas)
    fakebas[:,NPRIM_OF] = 1
    fakebas[:,NCTR_OF] = 1
    if expnt.size == 1:
        expnt = expnt[0]
        # approximate point charge with gaussian distribution exp(-1e16*r^2)
        fakebas[:,PTR_EXP] = ptr
        fakebas[:,PTR_COEFF] = ptr+1
        fakeenv.append([expnt, 1 / (2*numpy.sqrt(numpy.pi)*gaussian_int(2,expnt))])
        ptr += 2
    else:
        assert expnt.size == nbas
        # approximate point charge with gaussian distribution exp(-expnt*r^2)
        fakebas[:,PTR_EXP] = ptr + numpy.arange(nbas) * 2
        fakebas[:,PTR_COEFF] = ptr + numpy.arange(nbas) * 2 + 1
        coeff = 1 / (2 * numpy.sqrt(numpy.pi) * gaussian_int(2, expnt))
        fakeenv.append(numpy.vstack((expnt, coeff)).T.ravel())

    fakemol = Mole()
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = numpy.hstack(fakeenv)
    fakemol._built = True
    return fakemol

def fakemol_for_cgtf_charge(coord, expnt=1e16, contr_coeff=1):
    '''Constructs a "fake" Mole object that has a Gaussian charge
    distribution at the specified coordinate (coord).  The charge
    can be given as a linear combination of Gaussians with
    exponents expnt and contraction coefficients contr_coeff.
    '''
    assert coord.shape[0] == 1
    expnt = numpy.asarray(expnt).ravel()
    contr_coeff = numpy.asarray(contr_coeff).ravel()

    fakeatm = numpy.zeros((1,ATM_SLOTS), dtype=numpy.int32)
    fakebas = numpy.zeros((1,BAS_SLOTS), dtype=numpy.int32)
    fakeenv = [0] * PTR_ENV_START
    ptr = PTR_ENV_START
    fakeatm[:,PTR_COORD] = numpy.arange(ptr, ptr+3, 3)
    fakeenv.append(coord.ravel())
    ptr += 3
    fakebas[:,ATOM_OF] = 0#numpy.arange(nbas)
    fakebas[:,NPRIM_OF] = contr_coeff.size
    fakebas[:,NCTR_OF] = 1
    if expnt.size == 1:
        expnt = expnt[0]
        fakebas[:,PTR_EXP] = ptr
        fakebas[:,PTR_COEFF] = ptr+1
        fakeenv.append([expnt, 1 / (2*numpy.sqrt(numpy.pi)*gaussian_int(2,expnt))])
        ptr += 2
    else:
        assert expnt.size == contr_coeff.size
        fakebas[:,PTR_EXP] = ptr
        fakebas[:,PTR_COEFF] = ptr + contr_coeff.size
        coeff = contr_coeff / (2 * numpy.sqrt(numpy.pi) * gaussian_int(2, expnt))
        fakeenv.append(numpy.vstack((expnt, coeff)).ravel())

    fakemol = Mole()
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = numpy.hstack(fakeenv)
    fakemol._built = True
    return fakemol

def classify_ecp_pseudo(mol, ecp, pp):
    '''
    Check whether ecp keywords are presented in pp and whether pp keywords are
    presented in ecp.  The return (ecp, pp) should have only the ecp keywords and
    pp keywords in each dict.
    The "misplaced" ecp/pp keywords have the lowest priority. E.g., if an atom
    is defined in ecp, the same ecp atom found in pp does NOT replace the
    definition in ecp, and vise versa.
    '''
    def classify(ecp, pp_alias):
        if isinstance(ecp, str):
            if basis._format_pseudo_name(ecp)[0] in pp_alias:
                return {}, {'default': str(ecp)}
        elif isinstance(ecp, dict):
            ecp_as_pp = {}
            for atom in ecp:
                key = ecp[atom]
                if (isinstance(key, str) and
                    basis._format_pseudo_name(key)[0] in pp_alias):
                    ecp_as_pp[atom] = str(key)
            if ecp_as_pp:
                ecp_left = dict(ecp)
                for atom in ecp_as_pp:
                    ecp_left.pop(atom)
                return ecp_left, ecp_as_pp
        return ecp, {}
    ecp_left, ecp_as_pp = classify(ecp, basis.PP_ALIAS)
    pp_left , pp_as_ecp = classify(pp, basis.ALIAS)

    # ecp = ecp_left + pp_as_ecp
    # pp = pp_left + ecp_as_pp
    ecp = ecp_left
    if pp_as_ecp and not isinstance(ecp_left, str):
        # If ecp is a str, all atoms have ecp definition.
        # The misplaced ecp has no effects.
        logger.info(mol, 'pseudo-potentials keywords for %s found in .ecp',
                    pp_as_ecp.keys())
        if ecp_left:
            pp_as_ecp.update(ecp_left)
        ecp = pp_as_ecp
    pp = pp_left
    if ecp_as_pp and not isinstance(pp_left, str):
        logger.info(mol, 'ECP keywords for %s found in .pseudo',
                    ecp_as_pp.keys())
        if pp_left:
            ecp_as_pp.update(pp_left)
        pp = ecp_as_pp
    return ecp, pp

def bse_predefined_ecp(basis_name, elements):
    '''Find ECP names for a given list of atoms from BSE database
    '''
    ecp = ecp_atoms = None
    if not isinstance(basis_name, str):
        return ecp, ecp_atoms
    pyscf_basis_alias = basis._format_basis_name(basis_name).lower()
    basis_meta = BSE_META.get(pyscf_basis_alias)
    if basis_meta:
        if isinstance(elements, str):
            elements = [elements]
        ecp_elements = basis_meta[1]
        if ecp_elements:
            unique_atoms = {charge(a) for a in set(elements)}
            ecp_atoms = unique_atoms.intersection(ecp_elements)
            if ecp_atoms:
                ecp = basis_meta[0] # standard format basis set name
    return ecp, ecp_atoms
