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
Mole class and helper functions to handle paramters and attributes for GTO
integrals. This module serves the interface to the integral library libcint.
'''

import os, sys
import types
import re
import platform
import gc
import time
import json
import ctypes
import numpy
import h5py
import scipy.special
import scipy.linalg
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

# For code compatibility in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str


# for _atm, _bas, _env
CHARGE_OF  = 0
PTR_COORD  = 1
NUC_MOD_OF = 2
PTR_ZETA   = PTR_FRAC_CHARGE = 3
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
PTR_LIGHT_SPEED = 0
PTR_COMMON_ORIG = 1
PTR_RINV_ORIG   = 4
PTR_RINV_ZETA   = 7
PTR_RANGE_OMEGA = 8
PTR_F12_ZETA    = 9
PTR_GTG_ZETA    = 10
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

def M(**kwargs):
    r'''This is a shortcut to build up Mole object.

    Args: Same to :func:`Mole.build`

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    '''
    mol = Mole()
    mol.build(**kwargs)
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
    if l >= 0:
        #f = 2**(2*l+3) * math.factorial(l+1) * (2*expnt)**(l+1.5) \
        #        / (math.factorial(2*l+2) * math.sqrt(math.pi))
        #return math.sqrt(f)
        return 1/numpy.sqrt(gaussian_int(l*2+2, 2*expnt))
    else:
        raise ValueError('l should be > 0')

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
        assert(l <= 12)
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
        assert(l is not None)
        assert(l <= 12)
        nd = l * 4 + 2
    nf = (l+1)*(l+2)//2
    c2smat = numpy.zeros((nf*2,nd), order='F', dtype=numpy.complex)
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
        assert(l is not None)
        assert(l <= 12)
    return ua, ub

def sph2spinor_l(l):
    '''Real spherical to spinor transformation matrix for angular moment l'''
    return sph2spinor_kappa(0, l)

def atom_types(atoms, basis=None):
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
    return atmgroup


def format_atom(atoms, origin=0, axes=None,
                unit=getattr(__config__, 'UNIT', 'Ang')):
    '''Convert the input :attr:`Mole.atom` to the internal data format.
    Including, changing the nuclear charge to atom symbol, converting the
    coordinates to AU, rotate and shift the molecule.
    If the :attr:`~Mole.atom` is a string, it takes ";" and "\\n"
    for the mark to separate atoms;  "," and arbitrary length of blank space
    to spearate the individual terms for an atom.  Blank line will be ignored.

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
            as the Bohr value (in angstrom), which should be around 0.53

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
        assert(len(dat) == 4)
        return [_atom_symbol(dat[0]), [float(x) for x in dat[1:4]]]

    if isinstance(atoms, (str, unicode)):
        # The input atoms points to a geometry file
        if os.path.isfile(atoms):
            try:
                atoms = fromfile(atoms)
            except ValueError:
                sys.stderr.write('\nFailed to parse geometry file  %s\n\n' % atoms)
                raise

        atoms = str(atoms.replace(';','\n').replace(',',' ').replace('\t',' '))
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
            if isinstance(atom, (str, unicode)):
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

    if isinstance(unit, (str, unicode)):
        if unit.upper().startswith(('B', 'AU')):
            unit = 1.
        else: #unit[:3].upper() == 'ANG':
            unit = 1./param.BOHR
    else:
        unit = 1./unit

    c = numpy.array([a[1] for a in fmt_atoms], dtype=numpy.double)
    c = numpy.einsum('ix,kx->ki', axes * unit, c - origin)
    z = [a[0] for a in fmt_atoms]
    return list(zip(z, c.tolist()))

#TODO: sort exponents
def format_basis(basis_tab):
    '''Convert the input :attr:`Mole.basis` to the internal data format.

    ``{ atom: [(l, ((-exp, c_1, c_2, ..),
                    (-exp, c_1, c_2, ..))),
               (l, ((-exp, c_1, c_2, ..),
                    (-exp, c_1, c_2, ..)))], ... }``

    Args:
        basis_tab : dict
            Similar to :attr:`Mole.basis`, it **cannot** be a str

    Returns:
        Formated :attr:`~Mole.basis`

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
    '''
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

    def convert(basis_name, symb):
        if basis_name.lower().startswith('unc'):
            return uncontract(basis.load(basis_name[3:], symb))
        else:
            return basis.load(basis_name, symb)

    fmt_basis = {}
    for atom, atom_basis in basis_tab.items():
        symb = _atom_symbol(atom)
        stdsymb = _std_symbol_without_ghost(symb)

        if isinstance(atom_basis, (str, unicode)):
            bset = convert(str(atom_basis), stdsymb)
        elif (any(isinstance(x, (str, unicode)) for x in atom_basis)
              # The first element is the basis of internal format
              or not isinstance(atom_basis[0][0], int)):
            bset = []
            for rawb in atom_basis:
                if isinstance(rawb, (str, unicode)):
                    bset += convert(str(rawb), stdsymb)
                else:
                    bset += nparray_to_list(rawb)
        else:
            bset = nparray_to_list(atom_basis)
        fmt_basis[symb] = bset

        if len(fmt_basis[symb]) == 0:
            raise RuntimeError('Basis not found for  %s' % symb)
    return fmt_basis

def uncontracted_basis(_basis):
    '''Uncontract internal format _basis

    Examples:

    >>> gto.uncontract(gto.load('sto3g', 'He'))
    [[0, [6.36242139, 1]], [0, [1.158923, 1]], [0, [0.31364979, 1]]]
    '''
    ubasis = []
    for b in _basis:
        angl = b[0]
        if isinstance(b[1], int):
            kappa = b[1]
            for p in b[2:]:
                ubasis.append([angl, kappa, [p[0], 1]])
        else:
            for p in b[1:]:
                ubasis.append([angl, [p[0], 1]])
    return ubasis
uncontract = uncontracted_basis
contract = contracted_basis = basis.to_general_contraction

def to_uncontracted_cartesian_basis(mol):
    '''Decontract the basis of a Mole or a Cell.  Returns a Mole (Cell) object
    with the uncontracted basis environment and a list of coefficients that
    transform the uncontracted cartesian basis to the original basis.  Each
    element in the list corresponds to one shell of the original Mole (Cell).

    Examples:

    >>> mol = gto.M(atom='Ne', basis='ccpvdz')
    >>> pmol, ctr_coeff = mol.to_uncontracted_cartesian_basis()
    >>> c = scipy.linalg.block_diag(*ctr_coeff)
    >>> s = reduce(numpy.dot, (c.T, pmol.intor('int1e_ovlp'), c))
    >>> abs(s-mol.intor('int1e_ovlp')).max()
    0.0
    '''
    import copy
    lmax = mol._bas[:,ANG_OF].max()
    if mol.cart:
        c2s = [numpy.eye((l+1)*(l+2)//2) for l in range(lmax+1)]
    else:
        c2s = [cart2sph(l, normalized='sp') for l in range(lmax+1)]

    pmol = copy.copy(mol)
    pmol.cart = True
    _bas = []
    _env = mol._env.copy()
    contr_coeff = []
    for ib in range(mol.nbas):
        l = mol._bas[ib,ANG_OF]
        ncart = (l+1)*(l+2)//2
        es = mol.bas_exp(ib)
        cs = mol._libcint_ctr_coeff(ib)
        np, nc = cs.shape
        norm = gto_norm(l, es)
        c = numpy.einsum('pi,p,xm->pxim', cs, 1./norm, c2s[l])
        contr_coeff.append(c.reshape(np*ncart,-1))

        pexp = mol._bas[ib,PTR_EXP]
        pc = mol._bas[ib,PTR_COEFF]
        bs = numpy.empty((np,8), dtype=numpy.int32)
        bs[:] = mol._bas[ib]
        bs[:,NCTR_OF] = bs[:,NPRIM_OF] = 1
        bs[:,PTR_EXP] = numpy.arange(pexp, pexp+np)
        bs[:,PTR_COEFF] = numpy.arange(pc, pc+np)
        _env[pc:pc+np] = norm
        _bas.append(bs)

    pmol._bas = numpy.asarray(numpy.vstack(_bas), dtype=numpy.int32)
    pmol._env = _env
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

        if isinstance(atom_ecp, (str, unicode)):
            stdsymb = _std_symbol_without_ghost(symb)
            ecp_dat = basis.load_ecp(str(atom_ecp), stdsymb)
            if ecp_dat is None or len(ecp_dat) == 0:
                #raise RuntimeError('ECP not found for  %s' % symb)
                sys.stderr.write('ECP %s not found for  %s\n' %
                                 (atom_ecp, symb))
            else:
                fmt_ecp[symb] = ecp_dat
        else:
            fmt_ecp[symb] = atom_ecp
    return fmt_ecp

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
        Formated :attr:`~Mole.basis`

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
        Formated :attr:`~Mole.basis`

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
            mol3._ecpbas = numpy.hstack((mol1._ecpbas, ecpbas2))

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
    mol3.basis = mol3._basis
    mol3.ecp = mol3._ecp

    mol3.nucprop.update(mol1.nucprop)
    mol3.nucprop.update(mol2.nucprop)
    return mol3

# <bas-of-mol1|intor|bas-of-mol2>
def intor_cross(intor, mol1, mol2, comp=None):
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
        if not b:  # == []
            continue
        angl = b[0]
        #if angl in [6, 7]:
        #    print('libcint may have large error for ERI of i function')
        if isinstance(b[1], int):
            kappa = b[1]
            b_coeff = numpy.array(sorted(list(b[2:]), reverse=True))
        else:
            kappa = 0
            b_coeff = numpy.array(sorted(list(b[1:]), reverse=True))
        es = b_coeff[:,0]
        cs = b_coeff[:,1:]
        nprim, nctr = cs.shape
        if NORMALIZE_GTO:
            cs = numpy.einsum('pi,p->pi', cs, gto_norm(angl, es))
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
    _env = []
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
            elif isinstance(nucmod, (int, str, unicode, types.FunctionType)):
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
                sys.stderr.write('Warn: Basis not found for atom %d %s\n' % (ia, symb))
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
    if _env:
        _env = numpy.hstack((pre_env,numpy.hstack(_env)))
    else:
        _env = numpy.array(pre_env, copy=False)
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

def copy(mol):
    '''Deepcopy of the given :class:`Mole` object
    '''
    import copy
    newmol = copy.copy(mol)
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
            'symmetry': mol.symmetry,
            'nucmod'  : mol.nucmod,
            'nucprop' : mol.nucprop,
            'ecp'     : mol.ecp,
            '_nelectron': mol._nelectron,
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
    exclude_keys = set(('output', 'stdout', '_keys',
                        # Constructing in function loads
                        'symm_orb', 'irrep_id', 'irrep_name'))
    nparray_keys = set(('_atm', '_bas', '_env', '_ecpbas',
                        '_symm_orig', '_symm_axes'))

    moldic = dict(mol.__dict__)
    for k in exclude_keys:
        del(moldic[k])
    for k in nparray_keys:
        if isinstance(moldic[k], numpy.ndarray):
            moldic[k] = moldic[k].tolist()
    moldic['atom'] = repr(mol.atom)
    moldic['basis']= repr(mol.basis)
    moldic['ecp' ] = repr(mol.ecp)

    try:
        return json.dumps(moldic)
    except TypeError:
        import warnings
        def skip_value(dic):
            dic1 = {}
            for k,v in dic.items():
                if (v is None or
                    isinstance(v, (str, unicode, bool, int, float))):
                    dic1[k] = v
                elif isinstance(v, (list, tuple)):
                    dic1[k] = v   # Should I recursively skip_vaule?
                elif isinstance(v, set):
                    dic1[k] = list(v)
                elif isinstance(v, dict):
                    dic1[k] = skip_value(v)
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
    if sys.version_info < (3,):
# Convert to utf8 because JSON loads fucntion returns unicode.
        def byteify(inp):
            if isinstance(inp, dict):
                return dict([(byteify(k), byteify(v)) for k, v in inp.iteritems()])
            elif isinstance(inp, (tuple, list)):
                return [byteify(x) for x in inp]
            elif isinstance(inp, unicode):
                return inp.encode('utf-8')
            else:
                return inp
        moldic = byteify(moldic)
    mol = Mole()
    mol.__dict__.update(moldic)
    mol.atom = eval(mol.atom)
    mol.basis= eval(mol.basis)
    mol.ecp  = eval(mol.ecp)
    mol._atm = numpy.array(mol._atm, dtype=numpy.int32)
    mol._bas = numpy.array(mol._bas, dtype=numpy.int32)
    mol._env = numpy.array(mol._env, dtype=numpy.double)
    mol._ecpbas = numpy.array(mol._ecpbas, dtype=numpy.int32)

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
# backward compatibility for chkfile of pyscf-1.4 in which val is an array of
# real floats. In pyscf-1.5, val can be a list of list, to include the
# imaginary part
                val_real, val_imag = val
            else:
                val_real, val_imag = val, None
            if val_imag is None:
                c = numpy.zeros(shape)
                c[numpy.array(x),numpy.array(y)] = numpy.array(val_real)
            else:
                c = numpy.zeros(shape, dtype=numpy.complex)
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
        return ((l+1)*(l+2)//2 * mol._bas[:,NPRIM_OF]).sum()
    else:
        return ((l*2+1) * mol._bas[:,NPRIM_OF]).sum()
def nao_nr(mol, cart=None):
    '''Total number of contracted GTOs for the given :class:`Mole` object'''
    if cart is None:
        cart = mol.cart
    if cart:
        return nao_cart(mol)
    else:
        return ((mol._bas[:,ANG_OF]*2+1) * mol._bas[:,NCTR_OF]).sum()
def nao_cart(mol):
    '''Total number of contracted cartesian GTOs for the given :class:`Mole` object'''
    l = mol._bas[:,ANG_OF]
    return ((l+1)*(l+2)//2 * mol._bas[:,NCTR_OF]).sum()

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
        tupel of start basis function id and the stop function id

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.nao_nr_range(mol, 2, 4)
    (2, 6)
    '''
    ao_loc = moleintor.make_loc(mol._bas[:bas_id1], 'sph')
    nao_id0 = ao_loc[bas_id0]
    nao_id1 = ao_loc[-1]
    return nao_id0, nao_id1

def nao_2c(mol):
    '''Total number of contracted spinor GTOs for the given :class:`Mole` object'''
    l = mol._bas[:,ANG_OF]
    kappa = mol._bas[:,KAPPA_OF]
    dims = (l*4+2) * mol._bas[:,NCTR_OF]
    dims[kappa<0] = l[kappa<0] * 2 + 2
    dims[kappa>0] = l[kappa>0] * 2
    return dims.sum()

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
        tupel of start basis function id and the stop function id

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.nao_2c_range(mol, 2, 4)
    (4, 12)
    '''
    ao_loc = moleintor.make_loc(mol._bas[:bas_id1], '')
    nao_id0 = ao_loc[bas_id0]
    nao_id1 = ao_loc[-1]
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
    The returned indices have postive or negative values.  For the i-th basis function,
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

def energy_nuc(mol, charges=None, coords=None):
    '''Compute nuclear repulsion energy (AU) or static Coulomb energy

    Returns
        float
    '''
    if charges is None: charges = mol.atom_charges()
    if len(charges) <= 1:
        return 0
    #e = 0
    #for j in range(len(mol._atm)):
    #    q2 = charges[j]
    #    r2 = coords[j]
    #    for i in range(j):
    #        q1 = charges[i]
    #        r1 = coords[i]
    #        r = numpy.linalg.norm(r1-r2)
    #        e += q1 * q2 / r
    rr = inter_distance(mol, coords)
    rr[numpy.diag_indices_from(rr)] = 1e200
    if CHECK_GEOM and numpy.any(rr < 1e-5):
        for atm_idx in numpy.argwhere(rr<1e-5):
            logger.warn(mol, 'Atoms %s have the same coordinates', atm_idx)
        raise RuntimeError('Ill geometry')
    e = numpy.einsum('i,ij,j->', charges, 1./rr, charges) * .5
    return e

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
            coreshl = core_configuration(nelec_ecp)
            shl_start = coreshl[l]+count[ia,l]+l+1
        count[ia,l] += nc
        for n in range(shl_start, shl_start+nc):
            for m in range(-l, l+1):
                label.append((ia+base, symb, '%d%s' % (n, strl),
                              str(param.REAL_SPHERIC[l][l+m])))

    if isinstance(fmt, (str, unicode)):
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
            coreshl = core_configuration(nelec_ecp)
            shl_start = coreshl[l]+count[ia,l]+l+1
        count[ia,l] += nc
        ncart = (l + 1) * (l + 2) // 2
        for n in range(shl_start, shl_start+nc):
            for m in range(ncart):
                label.append((ia+base, symb, '%d%s' % (n, strl), cartxyz[l][m]))

    if isinstance(fmt, (str, unicode)):
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
            coreshl = core_configuration(nelec_ecp)
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

    if isinstance(fmt, (str, unicode)):
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
    '''Detect whether the given molelcule is chiral molecule or two molecules
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

def condense_to_shell(mol, mat, compressor=numpy.max):
    '''The given matrix is first partitioned to blocks, based on AO shell as
    delimiter.  Then call compressor function to abstract each block.
    '''
    ao_loc = mol.ao_loc_nr()
    abstract = numpy.empty((mol.nbas,mol.nbas))
    for i, i0 in enumerate(ao_loc[:mol.nbas]):
        for j, j0 in enumerate(ao_loc[:mol.nbas]):
            abstract[i,j] = compressor(mat[i0:ao_loc[i+1],j0:ao_loc[j+1]])
    return abstract


def tostring(mol, format='raw'):
    '''Convert molecular geometry to a string of the required format.

    Supported output formats:
        | raw: Each line is  <symobl> <x> <y> <z>
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
            output.append('%-4s %14.5f %14.5f %14.5f' %
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
        raise NotImplementedError

def tofile(mol, filename, format=None):
    '''Write molecular geometry to a file of the required format.

    Supported output formats:
        | raw: Each line is  <symobl> <x> <y> <z>
        | xyz: XYZ cartesian coordinates format
        | zmat: Z-matrix format
    '''
    if format is None:  # Guess format based on filename
        format = os.path.splitext(filename)[1][1:]
    string = tostring(mol, format)
    with open(filename, 'w') as f:
        f.write(string)
        f.write('\n')
    return string

def fromfile(filename, format=None):
    '''Read molecular geometry from a file
    (in testing)

    Supported formats:
        | raw: Each line is  <symobl> <x> <y> <z>
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
        | raw: Each line is  <symobl> <x> <y> <z>
        | xyz: XYZ cartesian coordinates format
        | zmat: Z-matrix format
    '''
    format = format.lower()
    if format == 'zmat':
        return from_zmatrix(string)
    elif format == 'xyz':
        dat = string.splitlines()
        natm = int(dat[0])
        return '\n'.join(dat[2:natm+2])
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


#
# Mole class handles three layers: input, internal format, libcint arguments.
# The relationship of the three layers are, eg
#    .atom (input) <=>  ._atom (for python) <=> ._atm (for libcint)
#   .basis (input) <=> ._basis (for python) <=> ._bas (for libcint)
# input layer does not talk to libcint directly.  Data are held in python
# internal fomrat layer.  Most of methods defined in this class only operates
# on the internal format.  Exceptions are make_env, make_atm_env, make_bas_env,
# set_common_orig_, set_rinv_orig_ which are used to manipulate the libcint arguments.
#
class Mole(lib.StreamObject):
    '''Basic class to hold molecular structure and global options

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
            To define molecluar structure.  The internal format is

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

        ** Following attributes are generated by :func:`Mole.build` **

        stdout : file object
            Default is sys.stdout if :attr:`Mole.output` is not set
        topgroup : str
            Point group of the system.
        groupname : str
            The supported subgroup of the point group. It can be one of Dooh,
            Coov, D2h, C2h, C2v, D2, Cs, Ci, C2, C1
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
        _keys : a set of str
            Store the keys appeared in the module.  It is used to check misinput attributes

        ** Following attributes are arguments used by ``libcint`` library **

        _atm :
            :code:`[[charge, ptr-of-coord, nuc-model, ptr-zeta, 0, 0], [...]]`
            each element reperesents one atom
        natm :
            number of atoms
        _bas :
            :code:`[[atom-id, angular-momentum, num-primitive-GTO, num-contracted-GTO, 0, ptr-of-exps, ptr-of-contract-coeff, 0], [...]]`
            each element reperesents one shell
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

    '''

    verbose = getattr(__config__, 'VERBOSE', logger.NOTE)

    # the unit (angstrom/bohr) of the coordinates defined by the input self.atom
    unit = getattr(__config__, 'UNIT', 'angstrom')

    # Whether to hold everything in memory
    incore_anyway = getattr(__config__, 'INCORE_ANYWAY', False)

    # Using cartesian GTO (6d,10f,15g)
    cart = getattr(__config__, 'gto_mole_Mole_cart', False)

    def __init__(self, **kwargs):
        self.output = None
        self.max_memory = param.MAX_MEMORY

        self.charge = 0
        self.spin = 0 # 2j == nelec_alpha - nelec_beta
        self.symmetry = False
        self.symmetry_subgroup = None
        self.cart = False

# Save inputs
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
##################################################
# don't modify the following private variables, they are not input options
        self._atm = numpy.zeros((0,6), dtype=numpy.int32)
        self._bas = numpy.zeros((0,8), dtype=numpy.int32)
        self._env = numpy.zeros(PTR_ENV_START)
        self._env[PTR_LIGHT_SPEED] = param.LIGHT_SPEED
        self._ecpbas = numpy.zeros((0,8), dtype=numpy.int32)

        self.stdout = sys.stdout
        self.groupname = 'C1'
        self.topgroup = 'C1'
        self.symm_orb = None
        self.irrep_id = None
        self.irrep_name = None
        self._symm_orig = None
        self._symm_axes = None
        self._nelectron = None
        self._atom = []
        self._basis = {}
        self._ecp = {}
        self._built = False

        # _pseudo is created to make the mol object consistenet with the mol
        # object converted from Cell.to_mol(). It is initialized in the
        # Cell.build() method only. Assigning _pseudo to mol object basically
        # has no effects. Mole.build() method does not have code to access the
        # contents of _pseudo.
        self._pseudo = {}

        keys = set(('verbose', 'unit', 'cart', 'incore_anyway'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.__dict__.update(kwargs)

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
        assert(nalpha >= 0 and nbeta >= 0)
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

    def __getattr__(self, key):
        '''To support accessing methods (mol.HF, mol.KS, mol.CCSD, mol.CASSCF, ...)
        from Mole object.
        '''
        if key[:2] == '__':  # Skip Python builtins
            raise AttributeError('Mole object has no attribute %s' % key)
        elif key in ('_ipython_canary_method_should_not_exist_',
                   '_repr_mimebundle_'):
            # https://github.com/mewwts/addict/issues/26
            # https://github.com/jupyter/notebook/issues/2014
            raise AttributeError

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
        else:
            mf = scf.HF(self)

        method = getattr(mf, key, None)
        if method is None:
            raise AttributeError('Mole object has no attribute %s' % key)

        # Initialize SCF object for post-SCF methods if applicable
        if self.nelectron != 0:
            mf.run()
        return method

# need "deepcopy" here because in shallow copy, _env may get new elements but
# with ptr_env unchanged
# def __copy__(self):
#        cls = self.__class__
#        newmol = cls.__new__(cls)
#        newmol = ...
# do not use __copy__ to aovid iteratively call copy.copy
    copy = copy

    pack = pack
    @lib.with_doc(unpack.__doc__)
    def unpack(self, moldic):
        return unpack(moldic)
    def unpack_(self, moldic):
        self.__dict__.update(moldic)
        return self

    dumps = dumps
    @lib.with_doc(loads.__doc__)
    def loads(self, molstr):
        return loads(molstr)
    def loads_(self, molstr):
        self.__dict__.update(loads(molstr).__dict__)
        return self

    def build(self, dump_input=True, parse_arg=True,
              verbose=None, output=None, max_memory=None,
              atom=None, basis=None, unit=None, nucmod=None, ecp=None,
              charge=None, spin=0, symmetry=None, symmetry_subgroup=None,
              cart=None):
        '''Setup moleclue and initialize some control parameters.  Whenever you
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
                To define molecluar structure.
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

        '''
        gc.collect()  # To release circular referred objects

        if isinstance(dump_input, (str, unicode)):
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
        if charge is not None: self.charge = charge
        if spin != 0: self.spin = spin
        if symmetry is not None: self.symmetry = symmetry
        if symmetry_subgroup is not None: self.symmetry_subgroup = symmetry_subgroup
        if cart is not None: self.cart = cart

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
                self.stdout = open(os.devnull, 'w')
            else:
                self.stdout = open(self.output, 'w')

        if self.verbose >= logger.WARN:
            self.check_sanity()

        self._atom = self.format_atom(self.atom, unit=self.unit)
        uniq_atoms = set([a[0] for a in self._atom])

        if isinstance(self.basis, (str, unicode, tuple, list)):
            # specify global basis for whole molecule
            _basis = dict(((a, self.basis) for a in uniq_atoms))
        elif 'default' in self.basis:
            default_basis = self.basis['default']
            _basis = dict(((a, default_basis) for a in uniq_atoms))
            _basis.update(self.basis)
            del(_basis['default'])
        else:
            _basis = self.basis
        self._basis = self.format_basis(_basis)

# TODO: Consider ECP info in point group symmetry initialization
        if self.ecp:
            # Unless explicitly input, ECP should not be assigned to ghost atoms
            if isinstance(self.ecp, (str, unicode)):
                _ecp = dict([(a, str(self.ecp))
                             for a in uniq_atoms if not is_ghost_atom(a)])
            elif 'default' in self.ecp:
                default_ecp = self.ecp['default']
                _ecp = dict(((a, default_ecp)
                             for a in uniq_atoms if not is_ghost_atom(a)))
                _ecp.update(self.ecp)
                del(_ecp['default'])
            else:
                _ecp = self.ecp
            self._ecp = self.format_ecp(_ecp)

        env = self._env[:PTR_ENV_START]
        self._atm, self._bas, self._env = \
                self.make_env(self._atom, self._basis, env, self.nucmod,
                              self.nucprop)
        self._atm, self._ecpbas, self._env = \
                self.make_ecp_env(self._atm, self._ecp, self._env)

        if self.spin is None:
            self.spin = self.nelectron % 2
        else:
            # Access self.nelec in which the code checks whether the spin and
            # number of electrons are consistent.
            self.nelec

        if self.symmetry:
            from pyscf import symm
            self.topgroup, orig, axes = symm.detect_symm(self._atom, self._basis)

            if isinstance(self.symmetry, (str, unicode)):
                self.symmetry = str(symm.std_symb(self.symmetry))
                self.groupname, axes = symm.subgroup(self.symmetry, axes)
                prop_atoms = self.format_atom(self._atom, orig, axes, 'Bohr')
                if symm.check_given_symm(self.groupname, prop_atoms, self._basis):
                    self.topgroup = self.symmetry
                else:
                    raise RuntimeWarning('Unable to identify input symmetry %s.\n'
                                         'Try symmetry="%s" with geometry (unit="Bohr")\n%s' %
                                         (self.symmetry, self.topgroup,
                                          '\n'.join([str(a) for a in prop_atoms])))
            else:
                self.groupname, axes = symm.as_subgroup(self.topgroup, axes,
                                                        self.symmetry_subgroup)
            self._symm_orig = orig
            self._symm_axes = axes

            if self.cart and self.groupname in ('Dooh', 'Coov'):
                if self.groupname == 'Dooh':
                    self.groupname, lgroup = 'D2h', 'Dooh'
                else:
                    self.groupname, lgroup = 'C2v', 'Coov'
                logger.warn(self, 'This version does not support linear molecule '
                            'symmetry %s for cartesian GTO basis.  Its subgroup '
                            '%s is used', lgroup, self.groupname)

            self.symm_orb, self.irrep_id = \
                    symm.symm_adapted_basis(self, self.groupname, orig, axes)
            self.irrep_name = [symm.irrep_id2name(self.groupname, ir)
                               for ir in self.irrep_id]

        if dump_input and not self._built and self.verbose > logger.NOTE:
            self.dump_input()

        logger.debug3(self, 'arg.atm = %s', str(self._atm))
        logger.debug3(self, 'arg.bas = %s', str(self._bas))
        logger.debug3(self, 'arg.env = %s', str(self._env))
        logger.debug3(self, 'ecpbas  = %s', str(self._ecpbas))

        self._built = True
        return self
    kernel = build

    @lib.with_doc(format_atom.__doc__)
    def format_atom(self, atom, origin=0, axes=None, unit='Ang'):
        return format_atom(atom, origin, axes, unit)

    @lib.with_doc(format_basis.__doc__)
    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    @lib.with_doc(format_ecp.__doc__)
    def format_ecp(self, ecp_tab):
        return format_ecp(ecp_tab)

    @lib.with_doc(expand_etb.__doc__)
    def expand_etb(self, l, n, alpha, beta):
        return expand_etb(l, n, alpha, beta)

    @lib.with_doc(expand_etbs.__doc__)
    def expand_etbs(self, etbs):
        return expand_etbs(etbs)
    etbs = expand_etbs

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

        self.stdout.write('System: %s  Threads %s\n' %
                          (str(platform.uname()), lib.num_threads()))
        self.stdout.write('Python %s\n' % sys.version)
        self.stdout.write('numpy %s  scipy %s\n' %
                          (numpy.__version__, scipy.__version__))
        self.stdout.write('Date: %s\n' % time.ctime())
        import pyscf
        pyscfdir = os.path.abspath(os.path.join(__file__, '..', '..'))
        self.stdout.write('PySCF version %s\n' % pyscf.__version__)
        self.stdout.write('PySCF path  %s\n' % pyscfdir)
        try:
            with open(os.path.join(pyscfdir, '..', '.git', 'ORIG_HEAD'), 'r') as f:
                self.stdout.write('GIT ORIG_HEAD %s' % f.read())
        except IOError:
            pass
        try:
            head = os.path.join(pyscfdir, '..', '.git', 'HEAD')
            with open(head, 'r') as f:
                head = f.read().splitlines()[0]
                self.stdout.write('GIT HEAD      %s\n' % head)
            # or command(git log -1 --pretty=%H)
            if head.startswith('ref:'):
                branch = os.path.basename(head)
                head = os.path.join(pyscfdir, '..', '.git', head.split(' ')[1])
                with open(head, 'r') as f:
                    self.stdout.write('GIT %s branch  %s' % (branch, f.read()))
        except IOError:
            pass

        self.stdout.write('\n')
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

        for ia,atom in enumerate(self._atom):
            coorda = tuple([x * param.BOHR for x in atom[1]])
            coordb = tuple([x for x in atom[1]])
            self.stdout.write('[INPUT]%3d %-4s %16.12f %16.12f %16.12f AA  '\
                              '%16.12f %16.12f %16.12f Bohr\n' \
                              % ((ia+1, _symbol(atom[0])) + coorda + coordb))
        if self.nucmod:
            if isinstance(self.nucmod, (int, str, unicode,
                                        types.FunctionType)):
                nucatms = [_symbol(atom[0]) for atom in self._atom]
            else:
                nucatms = self.nucmod.keys()
            self.stdout.write('[INPUT] Gaussian nuclear model for atoms %s\n' %
                              nucatms)

        if self.nucprop:
            self.stdout.write('[INPUT] nucprop %s\n' % self.nucprop)

        if self.verbose >= logger.DEBUG:
            self.stdout.write('[INPUT] ---------------- BASIS SET ---------------- \n')
            self.stdout.write('[INPUT] l, kappa, [nprim/nctr], ' \
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
                    self.stdout.write('[INPUT] %d   %2d    [%-5d/%-4d]  '
                                      % (b[0], kappa, b_coeff.__len__(),
                                         b_coeff[0].__len__()-1))
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
            logger.info(self, 'nuclear repulsion = %.15g', self.energy_nuc())
            if self.symmetry:
                if self.topgroup == self.groupname:
                    logger.info(self, 'point group symmetry = %s', self.topgroup)
                else:
                    logger.info(self, 'point group symmetry = %s, use subgroup %s',
                                self.topgroup, self.groupname)
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

        logger.info(self, 'CPU time: %12.2f', time.clock())
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
        '''Retuen a temporary mol context which has the rquired common origin.
        The required common origin has no effects out of the temporary context.
        See also :func:`mol.set_common_origin`

        Examples:

        >>> with mol.with_common_origin((1,0,0)):
        ...     mol.intor('int1e_r', comp=3)
        '''
        coord0 = self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3].copy()
        return _TemporaryMoleContext(self.set_common_origin, (coord,), (coord0,))
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
        '''Retuen a temporary mol context which has the rquired origin of 1/r
        operator.  The required origin has no effects out of the temporary
        context.  See also :func:`mol.set_rinv_origin`

        Examples:

        >>> with mol.with_rinv_origin((1,0,0)):
        ...     mol.intor('int1e_rinv')
        '''
        coord0 = self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3].copy()
        return _TemporaryMoleContext(self.set_rinv_origin, (coord,), (coord0,))
    with_rinv_orig = with_rinv_origin

    def set_range_coulomb(self, omega):
        '''Apply the long range part of range-separated Coulomb operator for
        **all** 2e integrals
        erf(omega r12) / r12
        set omega to 0 to siwtch off the range-separated Coulomb
        '''
        if omega is None:
            self._env[PTR_RANGE_OMEGA] = 0
        else:
            self._env[PTR_RANGE_OMEGA] = omega
    set_range_coulomb_ = set_range_coulomb  # for backward compatibility

    @property
    def omega(self):
        return self._env[PTR_RANGE_OMEGA]
    omega = omega.setter(set_range_coulomb)

    def with_range_coulomb(self, omega):
        '''Retuen a temporary mol context which has the required parameter
        omega for long range part of range-separated Coulomb operator.
        If omega = None, it will be treated as the regular Coulomb operator.
        See also :func:`mol.set_range_coulomb`

        Examples:

        >>> with mol.with_range_coulomb(omega=1.5):
        ...     mol.intor('int2e')
        '''
        omega0 = self._env[PTR_RANGE_OMEGA].copy()
        return _TemporaryMoleContext(self.set_range_coulomb, (omega,), (omega0,))

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
        '''Retuen a temporary mol context which has the rquired Gaussian charge
        distribution placed at "rinv_origin": rho(r) = Norm * exp(-zeta * r^2).
        See also :func:`mol.set_rinv_zeta`

        Examples:

        >>> with mol.with_rinv_zeta(zeta=1.5), mol.with_rinv_origin((1.,0,0)):
        ...     mol.intor('int1e_rinv')
        '''
        zeta0 = self._env[PTR_RINV_ZETA].copy()
        return _TemporaryMoleContext(self.set_rinv_zeta, (zeta,), (zeta0,))

    def with_rinv_at_nucleus(self, atm_id):
        '''Retuen a temporary mol context in which the rinv operator (1/r) is
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
            return _TemporaryMoleContext(set_rinv, (zeta,rinv), (zeta0,rinv0))
    with_rinv_as_nucleus = with_rinv_at_nucleus  # For backward compatibility

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        '''Update geometry
        '''
        import copy
        if inplace:
            mol = self
        else:
            mol = copy.copy(self)
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
            mol._atom = mol.atom

            if isinstance(unit, (str, unicode)):
                if unit.upper().startswith(('B', 'AU')):
                    unit = 1.
                else: #unit[:3].upper() == 'ANG':
                    unit = 1./param.BOHR
            else:
                unit = 1./unit
            ptr = mol._atm[:,PTR_COORD]
            mol._env[ptr+0] = unit * atoms_or_coords[:,0]
            mol._env[ptr+1] = unit * atoms_or_coords[:,1]
            mol._env[ptr+2] = unit * atoms_or_coords[:,2]
        else:
            mol.symmetry = symmetry
            mol.build(False, False)

        if mol.verbose >= logger.INFO:
            logger.info(mol, 'New geometry (unit %s)', unit)
            coords = mol.atom_coords()
            for ia in range(mol.natm):
                logger.info(mol, ' %3d %-4s %16.12f %16.12f %16.12f',
                            ia+1, mol.atom_symbol(ia), *coords[ia])
        return mol

    def update(self, chkfile):
        return self.update_from_chk(chkfile)
    def update_from_chk(self, chkfile):
        with h5py.File(chkfile, 'r') as fh5:
            mol = loads(fh5['mol'].value)
            self.__dict__.update(mol.__dict__)
        return self

    def has_ecp(self):
        '''Whether pseudo potential is used in the system.'''
        return len(self._ecpbas) > 0 or self._pseudo


#######################################################
#NOTE: atm_id or bas_id start from 0
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
        >>> mol.atom_symbol(0)
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
            return self._atm[atm_id,CHARGE_OF]
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
        if unit[:3].upper() == 'ANG':
            return self._env[ptr:ptr+3] * param.BOHR
        else:
            return self._env[ptr:ptr+3]

    def atom_coords(self, unit='Bohr'):
        '''np.asarray([mol.atom_coords(i) for i in range(mol.natm)])'''
        ptr = self._atm[:,PTR_COORD]
        c = self._env[numpy.vstack((ptr,ptr+1,ptr+2)).T]
        if unit[:3].upper() == 'ANG':
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
        return (self._bas[:,ATOM_OF] == atm_id).sum()

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
        return self._env[ptr:ptr+3]

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
        return self._bas[bas_id,ATOM_OF]

    def bas_angular(self, bas_id):
        r'''The angular momentum associated with the given basis

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_atom(7)
        2
        '''
        return self._bas[bas_id,ANG_OF]

    def bas_nctr(self, bas_id):
        r'''The number of contracted GTOs for the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_atom(3)
        3
        '''
        return self._bas[bas_id,NCTR_OF]

    def bas_nprim(self, bas_id):
        r'''The number of primitive GTOs for the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_atom(3)
        11
        '''
        return self._bas[bas_id,NPRIM_OF]

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
        return self._bas[bas_id,KAPPA_OF]

    def bas_exp(self, bas_id):
        r'''exponents (ndarray) of the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_kappa(0)
        [ 13.01     1.962    0.4446]
        '''
        nprim = self.bas_nprim(bas_id)
        ptr = self._bas[bas_id,PTR_EXP]
        return self._env[ptr:ptr+nprim]

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

    nao = property(nao_nr)
    ao_loc = property(ao_loc_nr)

    tmap = time_reversal_map = time_reversal_map

    inertia_moment = inertia_moment

    tostring = tostring
    tofile = tofile

    def fromstring(self, string, format='xyz'):
        '''Update the Mole object based on the input geometry string'''
        self.atom = string
        self._atom = self.format_atom(fromstring(string, format))
        self.set_geom_(self, self._atom, unit='Angstrom', inplace=True)
        if format == 'sdf' and 'M  CHG' in string:
            raise NotImplementedError
            #FIXME self.charge = 0
        return self

    def fromfile(self, filename, format=None):
        '''Update the Mole object based on the input geometry file'''
        self.atom = filename
        self._atom = self.format_atom(fromfile(filename, format))
        self.set_geom_(self, self._atom, unit='Angstrom', inplace=True)
        return self

    def intor(self, intor, comp=None, hermi=0, aosym='s1', out=None,
              shls_slice=None):
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
        if 'ECP' in intor:
            assert(self._ecp is not None)
            bas = numpy.vstack((self._bas, self._ecpbas))
            self._env[AS_ECPBAS_OFFSET] = len(self._bas)
            self._env[AS_NECPBAS] = len(self._ecpbas)
            if shls_slice is None:
                shls_slice = (0, self.nbas, 0, self.nbas)
        else:
            bas = self._bas
        return moleintor.getints(intor, self._atm, bas, self._env,
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

    def intor_symmetric(self, intor, comp=None):
        '''One-electron integral generator. The integrals are assumed to be hermitian

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. int1e_ipovlp_sph has 3 components.

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
        return self.intor(intor, comp, 1, aosym='s4')

    def intor_asymmetric(self, intor, comp=None):
        '''One-electron integral generator. The integrals are assumed to be anti-hermitian

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. int1e_ipovlp has 3 components.

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
        return self.intor(intor, comp, 2, aosym='a4')

    @lib.with_doc(moleintor.getints_by_shell.__doc__)
    def intor_by_shell(self, intor, shells, comp=None):
        intor = self._add_suffix(intor)
        if 'ECP' in intor:
            assert(self._ecp is not None)
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
        return self.energy_nuc()

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

    to_uncontracted_cartesian_basis = to_uncontracted_cartesian_basis

    __add__ = conc_mol

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
        elif isinstance(fn, (str, unicode)):
            method = getattr(self, fn.upper())
            return method(*args, **kwargs)
        else:
            raise TypeError('First argument of .apply method must be a '
                            'function/class or a name (string) of a method.')

    def ao2mo(self, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
              **kwargs):
        '''Integral transformation for arbitrary orbitals and arbitrary
        integrals.  See more detalied documentation in func:`ao2mo.kernel`.

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

def _parse_nuc_mod(str_or_int_or_fn):
    nucmod = NUC_POINT
    if callable(str_or_int_or_fn):
        nucmod = str_or_int_or_fn
    elif (isinstance(str_or_int_or_fn, (str, unicode)) and
          str_or_int_or_fn[0].upper() == 'G'): # 'gauss_nuc'
        nucmod = NUC_GAUSS
    elif str_or_int_or_fn != 0:
        nucmod = NUC_GAUSS
    return nucmod

def _update_from_cmdargs_(mol):
    # Ipython shell conflicts with optparse
    # pass sys.args when using ipython
    try:
        __IPYTHON__
        #sys.stderr.write('Warn: Ipython shell catchs sys.args\n')
        return
    except:
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
    symb = []
    coord = []
    min_items_per_line = 1
    for line in atomstr.splitlines():
        line = line.strip()
        if line and line[0] != '#':
            rawd = line.split()
            assert(len(rawd) >= min_items_per_line)

            symb.append(rawd[0])
            if len(rawd) < 3:
                coord.append(numpy.zeros(3))
                min_items_per_line = 3
            elif len(rawd) == 3:
                coord.append(numpy.array((float(rawd[2]), 0, 0)))
                min_items_per_line = 5
            elif len(rawd) == 5:
                bonda = int(rawd[1]) - 1
                bond  = float(rawd[2])
                anga  = int(rawd[3]) - 1
                ang   = float(rawd[4])/180*numpy.pi
                assert(ang >= 0)
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
                bonda = int(rawd[1]) - 1
                bond  = float(rawd[2])
                anga  = int(rawd[3]) - 1
                ang   = float(rawd[4])/180*numpy.pi
                assert(ang >= 0 and ang <= numpy.pi)
                v1 = coord[anga] - coord[bonda]
                v1 /= numpy.linalg.norm(v1)
                if ang < 1e-7:
                    c = v1 * bond
                elif numpy.pi-ang < 1e-7:
                    c = -v1 * bond
                else:
                    diha  = int(rawd[5]) - 1
                    dih   = float(rawd[6])/180*numpy.pi
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
    atoms = list(zip([_atom_symbol(x) for x in symb], coord))
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
    r = (0.836 * mass**(1./3) + 0.570) / 52917.7249;
    zeta = 1.5 / (r**2);
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
# approximate point charge with gaussian distribution exp(-1e16*r^2)
    fakebas[:,PTR_EXP] = ptr
    fakebas[:,PTR_COEFF] = ptr+1
    fakeenv.append([expnt, 1/(2*numpy.sqrt(numpy.pi)*gaussian_int(2,expnt))])
    ptr += 2
    fakemol = Mole()
    fakemol._atm = fakeatm
    fakemol._bas = fakebas
    fakemol._env = numpy.hstack(fakeenv)
    fakemol._built = True
    return fakemol

class _TemporaryMoleContext(object):
    import copy
    def __init__(self, method, args, args_bak):
        self.method = method
        self.args = args
        self.args_bak = args_bak
    def __enter__(self):
        self.method(*self.args)
    def __exit__(self, type, value, traceback):
        self.method(*self.args_bak)

del(BASE)
