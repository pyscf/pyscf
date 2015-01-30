#!/usr/bin/env python
# -*- coding: utf-8
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import gc
import time
import math
import itertools
from functools import reduce
import numpy
import pyscf.lib.parameters as param
import pyscf.lib.logger as log
from pyscf.gto import cmd_args
from pyscf.gto import basis
from pyscf.gto import moleintor


def M(*args, **kwargs):
    r'''This is a simple way to build up Mole object quickly.

    Args: Same to :func:`Mole.build`

    Examples:

    >>> from pyscf import gto
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='6-31g')
    '''
    mol = Mole()
    mol.build_(*args, **kwargs)
    return mol

def gto_norm(l, expnt):
    r'''Normalized factor for GTO   :math:`g=r^l e^{-\alpha r^2}`

    .. math::

        \frac{1}{\sqrt{\int g^2 r^2 dr}}
        = \sqrt{\frac{2^{2l+3} (l+1)! (2a)^{l+1.5}}{(2l+2)!\sqrt{\pi}}}

    Ref: H. B. Schlegel and M. J. Frisch, Int. J. Quant.  Chem., 54(1955), 83-87.

    Args:
        l (int):
            angular momentum
        expnt :
            exponent :math:`\alpha`

    Returns:
        normalization factor

    Examples:

    >>> print gto_norm(0, 1)
    2.5264751109842591
    '''
    if l >= 0:
        f = 2**(2*l+3) * math.factorial(l+1) * (2*expnt)**(l+1.5) \
                / (math.factorial(2*l+2) * math.sqrt(math.pi))
        return math.sqrt(f)
    else:
        raise ValueError('l should be > 0')


def format_atom(atoms, origin=0, axes=1):
    '''Convert the input :attr:`Mole.atom` to the internal data format.
    Including, changing the nuclear charge to atom symbol, rotate and shift
    molecule.  If the :attr:`~Mole.atom` is a string, it takes ";" and "\\n"
    for the mark to separate atoms;  "," and arbitrary length of blank space
    to spearate the individual terms for an atom.  Blank line will be ignored.

    Args:
        atoms : list or str
            the same to :attr:`Mole.atom`

    Kwargs:
        origin : ndarray
            new axis origin.
        axes : ndarray
            (new_x, new_y, new_z), each entry is a length-3 array

    Returns:
        formated :attr:`~Mole.atom`

    Examples:

    >>> gto.format_atom('9,0,0,0; h@1 0 0 1', origin=(1,1,1))
    [['F', [-1.0, -1.0, -1.0]], ['H@1', [-1.0, -1.0, 0.0]]]
    >>> gto.format_atom(['9,0,0,0', (1, (0, 0, 1))], origin=(1,1,1))
    [['F', [-1.0, -1.0, -1.0]], ['H', [-1, -1, 0]]]
    '''
    fmt_atoms = []
    def str2atm(line):
        dat = line.split()
        if dat[0].isdigit():
            symb = param.ELEMENTS[int(dat[0])][0]
        else:
            rawsymb = _rm_digit(dat[0])
            stdsymb = param.ELEMENTS[_ELEMENTDIC[rawsymb.upper()]][0]
            symb = dat[0].replace(rawsymb, stdsymb)
        c = numpy.array([float(x) for x in dat[1:4]]) - origin
        return [symb, numpy.dot(axes, c).tolist()]

    if isinstance(atoms, str):
        atoms = atoms.replace(';','\n').replace(',',' ')
        for line in atoms.split('\n'):
            if line.strip():
                fmt_atoms.append(str2atm(line))
    else:
        for atom in atoms:
            if isinstance(atom, str):
                fmt_atoms.append(str2atm(atom.replace(',',' ')))
            else:
                if isinstance(atom[0], int):
                    symb = param.ELEMENTS[atom[0]][0]
                else:
                    rawsymb = _rm_digit(atom[0])
                    stdsymb = param.ELEMENTS[_ELEMENTDIC[rawsymb.upper()]][0]
                    symb = atom[0].replace(rawsymb, stdsymb)
                c = numpy.array(atom[1]) - origin
                fmt_atoms.append([symb, numpy.dot(axes, c).tolist()])
    return fmt_atoms

#TODO: sort exponents
def format_basis(basis_tab):
    '''Convert the input :attr:`Mole.basis` to the internal data format.

    ``{ atom: (l, kappa, ((-exp, c_1, c_2, ..), nprim, nctr, ptr-exps, ptr-contraction-coeff)), ... }``

    Args:
        basis_tab : list
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
    fmt_basis = {}
    for atom in basis_tab.keys():
        symb = _symbol(atom)

        if isinstance(basis_tab[atom], str):
            rawsymb = _rm_digit(symb)
            stdsymb = param.ELEMENTS[_ELEMENTDIC[rawsymb.upper()]][0]
            symb = symb.replace(rawsymb, stdsymb)
            fmt_basis[symb] = basis.load(basis_tab[atom], stdsymb)
        else:
            fmt_basis[symb] = basis_tab[atom]
    return fmt_basis

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
    basis = [expand_etb(*etb) for etb in etbs]
    return list(itertools.chain.from_iterable(basis))

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
    >>> gto.moleintor.getints('cint1e_ovlp_sph', atm3, bas3, env3, range(2), range(2,5))
    [[ 0.04875181  0.44714688  0.          0.37820346  0.        ]
     [ 0.04875181  0.44714688  0.          0.          0.37820346]]
    '''
    off = len(env1)
    natm_off = len(atm1)
    atm2 = numpy.array(atm2)
    bas2 = numpy.array(bas2)
    atm2[:,PTR_COORD] += off
    atm2[:,PTR_MASS ] += off
    bas2[:,ATOM_OF  ] += natm_off
    bas2[:,PTR_EXP  ] += off
    bas2[:,PTR_COEFF] += off
    return atm1+atm2.tolist(), bas1+bas2.tolist(), env1+env2

# <bas-of-mol1|intor|bas-of-mol2>
def intor_cross(intor, mol1, mol2, comp=1):
    r'''Cross 1-electron integrals like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in mol1, \nu \in mol2

    Args:
        intor : str
            Name of the 1-electron integral, such as cint1e_ovlp_sph (spherical overlap),
            cint1e_nuc_cart (cartesian nuclear attraction), cint1e_ipovlp
            (spinor overlap gradients), etc.  Ref to :func:`getints` for the
            full list of available 1-electron integral names
        mol1, mol2:
            :class:`Mole` objects

    Kwargs:
        comp : int
            Components of the integrals, e.g. cint1e_ipovlp has 3 components

    Returns:
        ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

    Examples:
        Compute the overlap between H2 molecule and O atom

    >>> mol1 = gto.M(atom='H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mol2 = gto.M(atom='O 0 0 0', basis='sto3g')
    >>> gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
    [[ 0.04875181  0.44714688  0.          0.37820346  0.        ]
     [ 0.04875181  0.44714688  0.          0.          0.37820346]]
    '''
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = conc_env(mol1._atm, mol1._bas, mol1._env, \
                                mol2._atm, mol2._bas, mol2._env)
    bras = range(nbas1)
    kets = range(nbas1, nbas1+nbas2)
    return moleintor.getints(intor, atmc, basc, envc, bras, kets, comp, 0)

# append (charge, pointer to coordinates, nuc_mod) to _atm
def make_atm_env(atom, ptr=0):
    '''Convert :attr:`Mole.atom` to the argument ``atm`` for ``libcint`` integrals
    '''
    _atm = [0] * 6
    _env = [x/param.BOHR for x in atom[1]]
    _env.append(param.ELEMENTS[_atm[CHARGE_OF]][1])
    _atm[CHARGE_OF] = _charge(atom[0])
    _atm[PTR_COORD] = ptr
    _atm[NUC_MOD_OF] = param.MI_NUC_POINT
    _atm[PTR_MASS ] = ptr + 3
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
        assert(angl < 8)
        if angl in [6, 7]:
            print('libcint may have large error for ERI of i function')
        if isinstance(b[1], int):
            kappa = b[1]
            b_coeff = numpy.array(b[2:])
        else:
            kappa = 0
            b_coeff = numpy.array(b[1:])
        es = b_coeff[:,0]
        cs = b_coeff[:,1:]
        nprim, nctr = cs.shape
        cs = numpy.array([cs[i] * gto_norm(angl, es[i]) \
                          for i in range(nprim)], order='F')
        _env.append(es)
        _env.append(cs.ravel(order='K'))
        ptr_exp = ptr
        ptr_coeff = ptr_exp + nprim
        ptr = ptr_coeff + nprim * nctr
        _bas.append([atom_id, angl, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0])
    _env = list(itertools.chain.from_iterable(_env)) # flatten nested lists
    return _bas, _env

def make_env(atoms, basis, pre_env=[], nucmod={}, mass={}):
    '''Generate the input arguments for ``libcint`` library in terms of
    :attr:`Mole.atoms` and :attr:`Mole.basis`
    '''
    _atm = []
    _bas = []
    _env = []
    ptr_env = len(pre_env)

    for ia, atom in enumerate(atoms):
        symb = atom[0]
        atm0, env0 = make_atm_env(atom, ptr_env)
        ptr_env = ptr_env + len(env0)
        if isinstance(nucmod, int):
            assert(nucmod in (0, 1))
            atm0[NUC_MOD_OF] = nucmod
        elif ia+1 in nucmod:
            atm0[NUC_MOD_OF] = nucmod[ia+1]
        elif symb in nucmod:
            atm0[NUC_MOD_OF] = nucmod[symb]
        elif _rm_digit(symb) in nucmod:
            atm0[NUC_MOD_OF] = nucmod[_rm_digit(symb)]
        if ia+1 in mass:
            atm0[PTR_MASS] = ptr_env
            env0.append(mass[ia+1])
            ptr_env = ptr_env + 1
        elif symb in mass:
            atm0[PTR_MASS] = ptr_env
            env0.append(mass[symb])
            ptr_env = ptr_env + 1
        elif _rm_digit(symb) in mass:
            atm0[PTR_MASS] = ptr_env
            env0.append(mass[_rm_digit(symb)])
            ptr_env = ptr_env + 1
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
        if symb in _basdic:
            bas0 = _basdic[symb]
        else:
            bas0 = _basdic[_rm_digit(symb)]
        _bas.append([[ia] + b[1:] for b in bas0])

    _bas = list(itertools.chain.from_iterable(_bas))
    _env = list(itertools.chain.from_iterable(_env))
    return _atm, _bas, pre_env+_env

def tot_electrons(mol):
    '''Total number of electrons for the given molecule

    Returns:
        electron number in integer

    Examples:

    >>> mol = gto.M(atom='H 0 1 0; C 0 0 1', charge=1)
    >>> gto.tot_electrons(mol)
    6
    '''
    nelectron = -mol.charge
    for ia in range(mol.natm):
        nelectron += mol.atom_charge(ia)
    return nelectron

def copy(mol):
    '''Deepcopy of the given :class:`Mole` object
    '''
    import copy
    newmol = copy.copy(mol)
    newmol._atm = copy.deepcopy(mol._atm)
    newmol._bas = copy.deepcopy(mol._bas)
    newmol._env = copy.deepcopy(mol._env)
    newmol.atom    = copy.deepcopy(mol.atom)
    newmol.basis   = copy.deepcopy(mol.basis)
    newmol._basis  = copy.deepcopy(mol._basis)
    return newmol

def pack(mol):
    '''Pack the given :class:`Mole` to a dict, which can be serialized with :mod:`pickle`
    '''
    return {'atom'    : mol.atom,
            'basis'   : mol._basis,
            'charge'  : mol.charge,
            'spin'    : mol.spin,
            'symmetry': mol.symmetry,
            'nucmod'  : mol.nucmod,
            'mass'    : mol.mass,
            'grids'   : mol.grids,
            'light_speed': mol.light_speed}
def unpack(moldic):
    '''Unpack a dict which is packed by :func:`pack`, return a :class:`Mole` object.
    '''
    mol = Mole()
    mol.__dict__.update(moldic)
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

def npgto_nr(mol):
    '''Total number of primitive spherical GTOs for the given :class:`Mole` object'''
    return reduce(lambda n, b: n + (mol.bas_angular(b) * 2 + 1) \
                                * mol.bas_nprim(b),
                  range(len(mol._bas)), 0)
def nao_nr(mol):
    '''Total number of contracted spherical GTOs for the given :class:`Mole` object'''
    return sum([(mol.bas_angular(b) * 2 + 1) * mol.bas_nctr(b) \
                for b in range(len(mol._bas))])

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
    nao_id0 = sum([(mol.bas_angular(b) * 2 + 1) * mol.bas_nctr(b) \
                   for b in range(bas_id0)])
    n = sum([(mol.bas_angular(b) * 2 + 1) * mol.bas_nctr(b) \
             for b in range(bas_id0, bas_id1)])
    return nao_id0, nao_id0+n

def nao_2c(mol):
    '''Total number of contracted spinor GTOs for the given :class:`Mole` object'''
    return sum([mol.bas_len_spinor(b) * mol.bas_nctr(b) \
                for b in range(len(mol._bas))])

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
    nao_id0 = sum([mol.bas_len_spinor(b) * mol.bas_nctr(b) \
                   for b in range(bas_id0)])
    n = sum([mol.bas_len_spinor(b) * mol.bas_nctr(b) \
             for b in range(bas_id0, bas_id1)])
    return nao_id0, nao_id0+n

def ao_loc_nr(mol):
    '''Offset of every shell in the spherical basis function spectrum

    Returns:
        list, each entry is the corresponding start basis function id

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.ao_loc_nr(mol)
    [0, 1, 2, 3, 6, 9, 10, 11, 12, 15, 18]
    '''
    off = 0
    ao_loc = []
    for i in range(len(mol._bas)):
        ao_loc.append(off)
        off += (mol.bas_angular(i) * 2 + 1) * mol.bas_nctr(i)
    ao_loc.append(off)
    return ao_loc

def ao_loc_2c(mol):
    '''Offset of every shell in the spinor basis function spectrum

    Returns:
        list, each entry is the corresponding start id of spinor function

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; C 0 0 1', basis='6-31g')
    >>> gto.ao_loc_2c(mol)
    [0, 2, 4, 6, 12, 18, 20, 22, 24, 30, 36]
    '''
    off = 0
    ao_loc = []
    for i in range(len(mol._bas)):
        ao_loc.append(off)
        off += mol.bas_len_spinor(i) * mol.bas_nctr(i)
    ao_loc.append(off)
    return ao_loc

def time_reversal_map(mol):
    r'''The index to map the spinor functions and its time reversal counterpart.
    The returned indices have postive and negative value.  For the i-th basis function,
    if the returned j = idx[i] < 0, it means :math:`T|i\rangle = -|j\rangle`,
    otherwise :math:`T|i\rangle = |j\rangle`
    '''
    tao = []
    i = 0
    for b in mol._bas:
        l = b[ANG_OF];
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
    return tao

def energy_nuc(mol):
    '''Nuclear repulsion energy, (AU)

    Returns
        float
    '''
    if mol.natm == 0:
        return 0
    #e = 0
    #chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    #coords = [mol.atom_coord(i) for i in range(len(mol._atm))]
    #for j in range(len(mol._atm)):
    #    q2 = chargs[j]
    #    r2 = coords[j]
    #    for i in range(j):
    #        q1 = chargs[i]
    #        r1 = coords[i]
    #        r = numpy.linalg.norm(r1-r2)
    #        e += q1 * q2 / r
    chargs = numpy.array([mol.atom_charge(i) for i in range(len(mol._atm))])
    coords = numpy.array([mol.atom_coord(i) for i in range(len(mol._atm))])
    rr = numpy.dot(coords, coords.T)
    rd = rr.diagonal()
    rr = rd[:,None] + rd - rr*2
    rr[numpy.diag_indices_from(rr)] = 1e-60
    r = numpy.sqrt(rr)
    qq = chargs[:,None] * chargs[None,:]
    qq[numpy.diag_indices_from(qq)] = 0
    e = (qq/r).sum() * .5
    return e

def spheric_labels(mol):
    '''Labels for spheric GTO functions

    Returns:
        List of [(atom-id, symbol-str, nl-str, str-of-real-spheric-notation]

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; Cl 0 0 1', basis='sto-3g')
    >>> gto.spheric_labels(mol)
    [(0, 'H', '1s', ''), (1, 'Cl', '1s', ''), (1, 'Cl', '2s', ''), (1, 'Cl', '3s', ''), (1, 'Cl', '2p', 'x'), (1, 'Cl', '2p', 'y'), (1, 'Cl', '2p', 'z'), (1, 'Cl', '3p', 'x'), (1, 'Cl', '3p', 'y'), (1, 'Cl', '3p', 'z')]
    '''
    count = numpy.zeros((mol.natm, 9), dtype=int)
    label = []
    for ib in range(len(mol._bas)):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        strl = param.ANGULAR[l]
        nc = mol.bas_nctr(ib)
        symb = mol.atom_symbol(ia)
        for n in range(count[ia,l]+l+1, count[ia,l]+l+1+nc):
            for m in range(-l, l+1):
                label.append((ia, symb, '%d%s' % (n, strl), \
                              '%s' % param.REAL_SPHERIC[l][l+m]))
        count[ia,l] += nc
    return label

def spinor_labels(mol):
    raise RuntimeError('TODO')

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
        nc = mol.bas_nctr(ib)
        if ia == atm_id and l1 == l:
            if atmshell > nc+l1:
                atmshell = atmshell - nc
            else:
                return ibf + (atmshell-l1-1)*(l1*2+1) + (l1+m)
        ibf += (l1*2+1) * nc

#TODO:def search_ao_r(mol, atm_id, l, j, m, atmshell):
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

def is_same_mol(mol1, mol2):
    if mol1.atom.__len__() != mol2.atom.__len__():
        return False
    for a1, a2 in zip(mol1.atom, mol2.atom):
        if a1[0] != a2[0] \
           or numpy.linalg.norm(numpy.array(a1[1])-numpy.array(a2[1])) > 2:
            return False
    return True

# for _atm, _bas, _env
CHARGE_OF  = 0
PTR_COORD  = 1
NUC_MOD_OF = 2
PTR_MASS   = 3
RAD_GRIDS  = 4
ANG_GRIDS  = 5
ATOM_OF    = 0
ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
KAPPA_OF   = 4
PTR_EXP    = 5
PTR_COEFF  = 6
# pointer to env
PTR_LIGHT_SPEED = 0
PTR_COMMON_ORIG = 1
PTR_RINV_ORIG   = 4
PTR_ENV_START   = 20


class Mole(object):
    '''Basic class to hold molecular structure and global options

    Attributes:
        verbose : int
            Print level
        output : str or None
            Output file, default is None which dumps msg to sys.stdout
        max_memory : int, float
            Allowed memory in MB
        light_speed :
            Default is set in lib.parameters.LIGHTSPEED
        charge : int
            Charge of molecule. It affects the electron numbers
        spin : int
            2S, num. alpha electrons - num. beta electrons
        symmetry : bool
            The parameter controls whether to use symmetry in calculation

        atom : list or str
            To define molecluar structure.  The internal format is

            | atom = [[atom1, (x, y, z)],
            |         [atom2, (x, y, z)],
            |         ...
            |         [atomN, (x, y, z)]]

        basis : dict or str
            To define basis set.
        nucmod : dict or str
            Nuclear model
        mass : dict
            Similar struct as :attr:`Mole.nucmod`
        grids : dict
            Define (radial grids, angular grids) for given atom or symbol

        ** Following attributes are generated by :func:`Mole.build` **

        stdout : file object
            Default is sys.stdout if :attr:`Mole.output` is not set
        groupname : str
            One of D2h, C2h, C2v, D2, Cs, Ci, C2, C1
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
            :attr:`symm.parameters.IRREP_ID_TABLE`
        irrep_name : a list of str
            Each element is one irreducible representation symbol associated with the basis
            stored in symm_orb.  The irrep symbols are defined in
            :attr:`symm.parameters.IRREP_ID_TABLE`
        _built : bool
            To label whether :func:`Mole.build` has been called.  It ensures some functions
            being initialized once.
        _basis : dict
            like :attr:`Mole.basis`, the internal format which is returned from the
            parser :func:`format_basis`
        _keys : a set of str
            Store the keys appeared in the module.  It is used to check misinput attributes

        ** Following attributes are arguments used by ``libcint`` library **

        _atm :
            :code:`[[charge, ptr-of-coord, nuc-model, mass, 0, 0], [...]]`
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

    >>> mol = Mole(atom='H^2 0 0 0; H 0 0 1.1', basis='sto3g')
    >>> print(mol.atom_symbol(0))
    H^2
    >>> print(mol.atom_pure_symbol(0))
    H
    >>> print(mol.nao_nr())
    2
    >>> print(mol.intor('cint1e_ovlp_sph'))
    [[ 0.99999999  0.43958641]
     [ 0.43958641  0.99999999]]
    >>> mol.charge = 1
    >>> mol.build()
    <class 'pyscf.gto.mole.Mole'> has no attributes Charge

    '''
    def __init__(self):
        self.verbose = log.NOTE
        self.output = None
        self.max_memory = param.MEMORY_MAX

        self.light_speed = param.LIGHTSPEED
        self.charge = 0
        self.spin = 0 # 2j
        self.symmetry = False

# atom, etb, basis, nucmod, mass, grids to save inputs
# self.atom = [(symb/nuc_charge, (coord(Angstrom):0.,0.,0.)), ...]
        self.atom = []
# self.basis = {atom_type/nuc_charge: [l, kappa, (expnt, c_1, c_2,..),..]}
        self.basis = 'sto-3g'
# self.nucmod = {atom_symbol: nuclear_model, atom_id: nuc_mod}, atom_id is 1-based
        self.nucmod = {}
# self.mass = {atom_symbol: mass, atom_id: mass}, atom_id is 1-based
        self.mass = {}
# self.grids = {atom_type/nuc_charge: [num_grid_radial, num_grid_angular]}
        self.grids = {}
##################################################
# don't modify the following private variables, they are not input options
        self._atm = []
        self.natm = 0
        self._bas = []
        self.nbas = 0
        self._env = [0] * PTR_ENV_START

        self.stdout = sys.stdout
        self.groupname = 'C1'
        self.nelectron = 0
        self.symm_orb = None
        self.irrep_id = None
        self.irrep_name = None
        self._basis = None
        self._built = False
        self._keys = set(self.__dict__.keys())

    def check_sanity(self, obj):
        '''Check misinput of a class attribute due to typos, check whether a
        class method is overwritten.  It does not check the attributes which
        are prefixed with "_".

        Args:
            obj : this object should have attribute _keys to store all the
            name of the attributes of the object
        '''
        if hasattr(obj, '_keys'):
            if self.verbose > log.QUIET:
                objkeys = [x for x in obj.__dict__.keys() if x[0] != '_']
                keysub = set(objkeys) - set(obj._keys)
                if keysub:
                    keyin = keysub.intersection(dir(obj.__class__))
                    if keyin:
                        log.warn(self, 'overwrite keys %s of %s',
                                 ' '.join(keyin), str(obj.__class__))

                    keydiff = keysub - set(dir(obj.__class__))
                    if keydiff:
                        sys.stderr.write('%s has no attributes %s\n' %
                                         (str(obj.__class__), ' '.join(keydiff)))

# need "deepcopy" here because in shallow copy, _env may get new elements but
# with ptr_env unchanged
# def __copy__(self):
#        cls = self.__class__
#        newmol = cls.__new__(cls)
#        newmol = ...
# do not use __copy__ to aovid iteratively call copy.copy
    def copy(self):
        return copy(self)

    def pack(self):
        return pack(self)
    def unpack(self, moldic):
        return unpack(moldic)
    def unpack_(self, moldic):
        self.__dict__.update(moldic)
        return self


    def build(self, *args, **kwargs):
        return self.build_(*args, **kwargs)
    def build_(self, dump_input=True, parse_arg=True, \
               verbose=None, output=None, max_memory=None, \
               atom=None, basis=None, nucmod=None, mass=None, grids=None, \
               charge=None, spin=None, symmetry=None, light_speed=None):
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
                To define molecluar structure.  If given, overwrite :attr:`Mole.atom`
            basis : dict or str
                To define basis set.  If given, overwrite :attr:`Mole.basis`
            nucmod : dict or str
                Nuclear model.  If given, overwrite :attr:`Mole.nucmod`
            mass : dict
                If given, overwrite :attr:`Mole.mass`
            grids : dict
                Define (radial grids, angular grids) for given atom or symbol
                If given, overwrite :attr:`Mole.grids`
            charge : int
                Charge of molecule. It affects the electron numbers
                If given, overwrite :attr:`Mole.charge`
            spin : int
                2S, num. alpha electrons - num. beta electrons
                If given, overwrite :attr:`Mole.spin`
            symmetry : bool
                Whether to use symmetry.  If given, overwrite :attr:`Mole.symmetry`
            light_speed :
                If given, overwrite :attr:`Mole.light_speed`

        '''
# release circular referred objs
# Note obj.x = obj.member_function causes circular referrence
        gc.collect()

        if verbose is not None: self.verbose = verbose
        if output is not None: self.output = output
        if max_memory is not None: self.max_memory = max_memory
        if atom is not None: self.atom = atom
        if basis is not None: self.basis = basis
        if nucmod is not None: self.nucmod = nucmod
        if mass is not None: self.mass = mass
        if grids is not None: self.grids = grids
        if charge is not None: self.charge = charge
        if spin is not None: self.spin = spin
        if symmetry is not None: self.symmetry = symmetry
        if light_speed is not None: self.light_speed = light_speed

        if parse_arg:
            _update_from_cmdargs_(self)

        # avoid to open output file twice
        if parse_arg and self.output is not None \
           and self.stdout.name != self.output:
            self.stdout = open(self.output, 'w')

        self.check_sanity(self)

        self.atom = self.format_atom(self.atom)
        if self.symmetry:
            import pyscf.symm
            #if self.symmetry in pyscf.symm.param.POINTGROUP
            #    self.groupname = self.symmetry
            #    #todo: pyscf.symm.check_given_symm(self.symmetric, self.atom)
            #    pass
            #else:
            #    self.groupname, inp_atoms = pyscf.symm.detect_symm(self.atom)
            self.groupname, origin, axes = pyscf.symm.detect_symm(self.atom)
            self.atom = self.format_atom(self.atom, origin, axes)

        if isinstance(self.basis, str):
            # specify global basis for whole molecule
            uniq_atoms = set([a[0] for a in self.atom])
            self._basis = self.format_basis(dict([(a, self.basis)
                                                  for a in uniq_atoms]))
        else:
            self._basis = self.format_basis(self.basis)

        self._env[PTR_LIGHT_SPEED] = self.light_speed
        self._atm, self._bas, self._env = \
                self.make_env(self.atom, self._basis, self._env, \
                              self.nucmod, self.mass)
        self.natm = self._atm.__len__()
        self.nbas = self._bas.__len__()
        self.nelectron = self.tot_electrons()
        if (self.nelectron+self.spin) % 2 != 0:
            sys.stderr.write('Electron number %d and spin %d are not consistent\n' %
                             (self.nelectron, self.nspin))

        if self.symmetry:
            import pyscf.symm
            eql_atoms = pyscf.symm.symm_identical_atoms(self.groupname, self.atom)
            symm_orb = pyscf.symm.symm_adapted_basis(self.groupname, eql_atoms,\
                                                     self.atom, self._basis)
            self.irrep_id = [ir for ir in range(len(symm_orb)) \
                             if symm_orb[ir].size > 0]
            self.irrep_name = [pyscf.symm.irrep_name(self.groupname, ir) \
                               for ir in self.irrep_id]
            self.symm_orb = [c for c in symm_orb if c.size > 0]

        if dump_input and not self._built and self.verbose >= log.NOTICE:
            self.dump_input()

        log.debug2(self, 'arg.atm = %s', self._atm)
        log.debug2(self, 'arg.bas = %s', self._bas)
        log.debug2(self, 'arg.env = %s', self._env)

        self._built = True
        #return self._atm, self._bas, self._env

    def format_atom(self, atom, origin=0, axes=1):
        return format_atom(atom, origin, axes)

    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    def expand_etb(self, l, n, alpha, beta):
        return expand_etb(l, n, alpha, beta)

    def expand_etbs(self, etbs):
        return expand_etbs(etbs)

    def make_env(self, atoms, basis, pre_env=[], nucmod={}, mass={}):
        return make_env(atoms, basis, pre_env, nucmod, mass)

    def make_atm_env(self, atom, ptr=0):
        return make_atm_env(atom, ptr)

    def make_bas_env(self, basis_add, atom_id=0, ptr=0):
        return make_bas_env(basis_add, atom_id, ptr)

    def tot_electrons(self):
        return tot_electrons(self)

    def gto_norm(self, l, expnt):
        return gto_norm(l, expnt)


    def dump_input(self):
        import __main__
        if hasattr(__main__, '__file__'):
            try:
                filename = os.path.abspath(__main__.__file__)
                finput = open(filename, 'r')
                self.stdout.write('\n')
                self.stdout.write('INFO: **** input file is %s ****\n' % filename)
                self.stdout.write(finput.read())
                self.stdout.write('INFO: ******************** input file end ********************\n')
                self.stdout.write('\n')
            except:
                log.warn(self, 'input file does not exist')

        self.stdout.write('System: %s\n' % str(os.uname()))
        self.stdout.write('Date: %s\n' % time.ctime())
        try:
            dn = os.path.dirname(os.path.realpath(__file__))
            self.stdout.write('GIT version: ')
            # or command(git log -1 --pretty=%H)
            for branch in 'dev', 'master':
                fname = '/'.join((dn, "../.git/refs/heads", branch))
                fin = open(fname, 'r')
                d = fin.readline()
                fin.close()
                self.stdout.write(' '.join((branch, d[:-1], '; ')))
            self.stdout.write('\n\n')
        except:
            pass

        self.stdout.write('[INPUT] VERBOSE %d\n' % self.verbose)
        self.stdout.write('[INPUT] light speed = %s\n' % self.light_speed)
        self.stdout.write('[INPUT] num atoms = %d\n' % self.natm)
        self.stdout.write('[INPUT] num electrons = %d\n' % self.nelectron)
        self.stdout.write('[INPUT] charge = %d\n' % self.charge)
        self.stdout.write('[INPUT] spin (= nelec alpha-beta = 2S) = %d\n' % self.spin)

        for nuc,(rad,ang) in self.grids.items():
            self.stdout.write('[INPUT] %s (%d, %d)\n' % (nuc, rad, ang))

        for ia,atom in enumerate(self.atom):
            self.stdout.write('[INPUT] %d %s %s AA, '\
                              '%s Bohr\n' \
                              % (ia+1, _symbol(atom[0]), atom[1],
                                 [x/param.BOHR for x in atom[1]]))
        for kn, vn in self.nucmod.items():
            if kn in self.mass:
                mass = self.mass[kn]
            else:
                if isinstance(kn, int):
                    symb = _symbol(self.atom[kn-1][0])
                    mass = param.ELEMENTS[_charge(symb)][1]
                else:
                    mass = param.ELEMENTS[_charge(kn)][1]

            self.stdout.write('[INPUT] Gaussian nuclear model for atom %s, mass = %f\n' %
                              (str(kn), mass))

        self.stdout.write('[INPUT] basis\n')
        self.stdout.write('[INPUT] l, kappa, [nprim/nctr], ' \
                          'expnt,             c_1 c_2 ...\n')
        for atom, basis in self._basis.items():
            self.stdout.write('[INPUT] %s\n' % atom)
            for b in basis:
                if isinstance(b[1], int):
                    kappa = b[1]
                    b_coeff = b[2:]
                else:
                    kappa = 0
                    b_coeff = b[1:]
                self.stdout.write('[INPUT] %d   %2d    [%-5d/%-4d]  ' \
                                  % (b[0], kappa, b_coeff.__len__(), \
                                     b_coeff[0].__len__()-1))
                for k, x in enumerate(b_coeff):
                    if k == 0:
                        self.stdout.write('%-15.12g  ' % x[0])
                    else:
                        self.stdout.write(' '*32+'%-15.12g  ' % x[0])
                    for c in x[1:]:
                        self.stdout.write(' %4.12g' % c)
                    self.stdout.write('\n')

        log.info(self, 'nuclear repulsion = %.15g', self.energy_nuc())
        if self.symmetry:
            log.info(self, 'point group symmetry = %s', self.groupname)
            for ir in range(self.symm_orb.__len__()):
                log.info(self, 'num. orbitals of %s = %d', \
                         self.irrep_name[ir], self.symm_orb[ir].shape[1])
        log.info(self, 'number of shells = %d', self.nbas)
        log.info(self, 'number of NR pGTOs = %d', self.npgto_nr())
        log.info(self, 'number of NR cGTOs = %d', self.nao_nr())
        if self.verbose >= log.DEBUG1:
            for i in range(len(self._bas)):
                exps = self.bas_exp(i)
                log.debug1(self, 'bas %d, expnt(s) = %s', i, str(exps))

        log.info(self, 'CPU time: %12.2f', time.clock())

    def set_common_origin_(self, coord):
        '''Update common origin which held in :class`Mole`._env.  **Note** the unit is Bohr

        Examples:

        >>> mol.set_common_origin_((0,0,0))
        '''
        self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3] = coord

    def set_rinv_orig_(self, coord):
        r'''Update origin for operator :math:`\frac{1}{|r-R_O|}`.  **Note** the unit is Bohr

        Examples:

        >>> mol.set_rinv_orig_((0,0,0))
        '''
        self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = coord[:3]

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
        return _symbol(self.atom[atm_id][0])

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
        return _symbol(self.atom_charge(atm_id))

    def atom_charge(self, atm_id):
        r'''Nuclear charge of the given atom id

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.atom_charge(1)
        17
        '''
        return self._atm[atm_id][CHARGE_OF]

    def atom_coord(self, atm_id):
        r'''Coordinates (ndarray) of the given atom id

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.atom_coord(1)
        [ 0.          0.          2.07869874]
        '''
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

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
        symb = self.atom_symbol(atm_id)
        return len(self._basis[symb])

    def atom_shell_ids(self, atm_id):
        r'''A list of the shell-ids of the given atom

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.atom_nshells(1)
        [3, 4, 5, 6, 7]
        '''
        return [ib for ib in range(len(self._bas)) \
                if self.bas_atom(ib) == atm_id]

    def bas_coord(self, bas_id):
        r'''Coordinates (ndarray) associated with the given basis id

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.bas_coord(2)
        [ 0.          0.          2.07869874]
        '''
        atm_id = self.bas_atom(bas_id) - 1
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

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
        return self._bas[bas_id][ATOM_OF]

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
        return self._bas[bas_id][ANG_OF]

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
        return self._bas[bas_id][NCTR_OF]

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
        return self._bas[bas_id][NPRIM_OF]

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
        return self._bas[bas_id][KAPPA_OF]

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
        ptr = self._bas[bas_id][PTR_EXP]
        return numpy.array(self._env[ptr:ptr+nprim])

    def bas_ctr_coeff(self, bas_id):
        r'''Contract coefficients (ndarray) of the given shell

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_ctr_coeff(3)
        [[ 0.34068924]
         [ 0.57789106]
         [ 0.65774031]]
        '''
        nprim = self.bas_nprim(bas_id)
        nctr = self.bas_nctr(bas_id)
        ptr = self._bas[bas_id][PTR_COEFF]
        return numpy.array(self._env[ptr:ptr+nprim*nctr]).reshape(nprim,nctr)

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
        return len_cart(self._bas[bas_id][ANG_OF])


    def npgto_nr(self):
        return npgto_nr(self)

    def nao_nr(self):
        return nao_nr(self)

    def nao_nr_range(self, bas_id0, bas_id1):
        return nao_nr_range(self, bas_id0, bas_id1)

    def nao_2c(self):
        return nao_2c(self)

    def nao_2c_range(self, bas_id0, bas_id1):
        return nao_2c_range(self, bas_id0, bas_id1)

    def ao_loc_nr(self):
        return ao_loc_nr(self)

    def ao_loc_2c(self):
        return ao_loc_2c(self)

    def tmap(self):
        return time_reversal_map(self)
    def time_reversal_map(self):
        return time_reversal_map(self)

    def intor(self, intor, comp=1, hermi=0):
        '''One-electron integral generator.

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. cint1e_ipovlp has 3 components.
            hermi : int
                Symmetry of the integrals

                | 0 : no symmetry assumed (default)
                | 1 : hermitian
                | 2 : anti-hermitian

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:

        >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
        >>> mol.intor('cint1e_ipnuc_sph', comp=3) # <nabla i | V_nuc | j>
        [[[ 0.          0.        ]
          [ 0.          0.        ]]
         [[ 0.          0.        ]
          [ 0.          0.        ]]
         [[ 0.10289944  0.48176097]
          [-0.48176097 -0.10289944]]]
        >>> mol.intor('cint1e_nuc')
        [[-1.69771092+0.j  0.00000000+0.j -0.67146312+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -1.69771092+0.j  0.00000000+0.j -0.67146312+0.j]
         [-0.67146312+0.j  0.00000000+0.j -1.69771092+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -0.67146312+0.j  0.00000000+0.j -1.69771092+0.j]]
        '''
        return moleintor.getints(intor, self._atm, self._bas, self._env,
                                 comp=comp, hermi=hermi)

    def intor_symmetric(self, intor, comp=1):
        '''One-electron integral generator. The integrals are assumed to be hermitian

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. cint1e_ipovlp has 3 components.

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:

        >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
        >>> mol.intor_symmetric('cint1e_nuc')
        [[-1.69771092+0.j  0.00000000+0.j -0.67146312+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -1.69771092+0.j  0.00000000+0.j -0.67146312+0.j]
         [-0.67146312+0.j  0.00000000+0.j -1.69771092+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -0.67146312+0.j  0.00000000+0.j -1.69771092+0.j]]
        '''
        return self.intor(intor, comp, 1)

    def intor_asymmetric(self, intor, comp=1):
        '''One-electron integral generator. The integrals are assumed to be anti-hermitian

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                complete list of available 1-electron integral names

        Kwargs:
            comp : int
                Components of the integrals, e.g. cint1e_ipovlp has 3 components.

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:

        >>> mol.build(atom='H 0 0 0; H 0 0 1.1', basis='sto-3g')
        >>> mol.intor_asymmetric('cint1e_nuc')
        [[-1.69771092+0.j  0.00000000+0.j  0.67146312+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -1.69771092+0.j  0.00000000+0.j  0.67146312+0.j]
         [-0.67146312+0.j  0.00000000+0.j -1.69771092+0.j  0.00000000+0.j]
         [ 0.00000000+0.j -0.67146312+0.j  0.00000000+0.j -1.69771092+0.j]]
        '''
        return self.intor(intor, comp, 2)

    def intor_cross(self, intor, bras, kets, comp=1):
        r'''Cross 1-electron integrals like

        .. math::

            \langle \mu | intor | \nu \rangle, \mu \in bras, \nu \in kets

        Args:
            intor : str
                Name of the 1-electron integral.  Ref to :func:`getints` for the
                full list of available 1-electron integral names
            bras : list of int
                A list of shell ids for bra
            kets : list of int
                A list of shell ids for ket

        Kwargs:
            comp : int
                Components of the integrals, e.g. cint1e_ipovlp has 3 components

        Returns:
            ndarray of 1-electron integrals, can be either 2-dim or 3-dim, depending on comp

        Examples:
            Compute the overlap between H2 molecule and O atom

        >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
        >>> mol.intor_cross('cint1e_ovlp_sph', range(0,3), range(3,5))
        [[ 0.04875181  0.04875181]
         [ 0.44714688  0.44714688]
         [ 0.          0.        ]
         [ 0.37820346  0.        ]
         [ 0.          0.37820346]]
        '''
        return moleintor.getints(intor, self._atm, self._bas, self._env,
                                 bras, kets, comp, 0)

    def energy_nuc(self):
        return energy_nuc(self)
    def get_enuc(self):
        return energy_nuc(self)

    def spheric_labels(self):
        return spheric_labels(self)

    def search_shell_id(self, atm_id, l):
        return search_shell_id(self, atm_id, l)

    def search_ao_nr(self, atm_id, l, m, atmshell):
        return search_ao_nr(self, atm_id, l, m, atmshell)
    def search_ao_r(self, atm_id, l, m, kappa, atmshell):
        return search_ao_r(self, atm_id, l, m, kappa, atmshell)

    def spinor_labels(self):
        return spinor_labels(self)

_ELEMENTDIC = dict((k.upper(),v) for k,v in param.ELEMENTS_PROTON.items())

def _rm_digit(symb):
    if symb.isalpha():
        return symb
    else:
        return ''.join([i for i in symb if i.isalpha()])
def _charge(symb_or_chg):
    if isinstance(symb_or_chg, int):
        return symb_or_chg
    else:
        return param.ELEMENTS_PROTON[_rm_digit(symb_or_chg)]

def _symbol(symb_or_chg):
    if isinstance(symb_or_chg, int):
        return param.ELEMENTS[symb_or_chg][0]
    else:
        return symb_or_chg

def _update_from_cmdargs_(mol):
    # Ipython shell conflicts with optparse
    # pass sys.args when using ipython
    try:
        __IPYTHON__
        print('Warn: Ipython shell catchs sys.args')
        return None
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

    if mol.output is not None:
        if os.path.isfile(mol.output):
            #os.remove(mol.output)
            if mol.verbose > log.QUIET:
                print('overwrite output file: %s' % mol.output)
        else:
            if mol.verbose > log.QUIET:
                print('output file: %s' % mol.output)

