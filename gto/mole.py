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

def gto_norm(l, expnt):
    ''' normalized factor for GTO   g=r^l e^(-a r^2)
    norm = 1/sqrt( \int_0^\infty g**2 r^2 dr )
         = sqrt( (2^(2l+3) (l+1)! (2a)^(l+1.5)) / (2l+2)!sqrt(\pi))
    Ref:
      H. B. Schlegel and M. J. Frisch, Int. J. Quant.  Chem., 54(1955), 83-87.
    '''
    if l >= 0:
        f = 2**(2*l+3) * math.factorial(l+1) * (2*expnt)**(l+1.5) \
                / (math.factorial(2*l+2) * math.sqrt(math.pi))
        return math.sqrt(f)
    else:
        raise ValueError('l should be > 0')


def format_atom(atoms, origin=0, axes=1):
    '''
    change nuclear charge to atom symbol, rotate and shift molecule
    if the molecule "atoms" is a string, atoms are separated by ";" or "\\n"
    the coordinates and atom symbols are separated by "," or blank space
    '''
    elementdic = dict((k.upper(),v) for k,v in param.ELEMENTS_PROTON.items())
    fmt_atoms = []
    if isinstance(atoms, str):
        atoms = atoms.replace(';','\n').replace(',',' ')
        for line in atoms.split('\n'):
            if line.strip():
                dat = line.split()
                if dat[0].isdigit():
                    symb = param.ELEMENTS[int(dat[0])][0]
                else:
                    rawsymb = _rm_digit(dat[0])
                    stdsymb = param.ELEMENTS[elementdic[rawsymb.upper()]][0]
                    symb = dat[0].replace(rawsymb, stdsymb)
                c = numpy.array([float(x) for x in dat[1:4]]) - origin
                fmt_atoms.append([symb, numpy.dot(axes, c).tolist()])
    else:
        for atom in atoms:
            if isinstance(atom[0], int):
                symb = param.ELEMENTS[atom[0]][0]
            else:
                rawsymb = _rm_digit(atom[0])
                stdsymb = param.ELEMENTS[elementdic[rawsymb.upper()]][0]
                symb = atom[0].replace(rawsymb, stdsymb)
            c = numpy.array(atom[1]) - origin
            fmt_atoms.append([symb, numpy.dot(axes, c).tolist()])
    return fmt_atoms

#TODO: sort exponents
def format_basis(basis_tab):
    '''
    transform the basis to standard format
    { atom: (l, kappa, ((-exp, c_1, c_2, ..), ..)), ... }
    '''
    elementdic = dict((k.upper(),v) for k,v in param.ELEMENTS_PROTON.items())
    fmt_basis = {}
    for atom in basis_tab.keys():
        symb = _symbol(atom)

        if isinstance(basis_tab[atom], str):
            rawsymb = _rm_digit(symb)
            stdsymb = param.ELEMENTS[elementdic[rawsymb.upper()]][0]
            symb = symb.replace(rawsymb, stdsymb)
            fmt_basis[symb] = basis.load(basis_tab[atom], stdsymb)
        else:
            fmt_basis[symb] = basis_tab[atom]
    return fmt_basis

# transform etb to basis format
def expand_etb(l, n, alpha, beta):
    '''
    expand even-tempered basis, alpha*beta**i, for i = 0..n
    '''
    return [[l, [alpha*beta**i, 1]] for i in reversed(range(n))]
def expand_etbs(etbs):
    basis = [expand_etb(*etb) for etb in etbs]
    return list(itertools.chain.from_iterable(basis))

def shift_ptr(atm, bas, off):
    atm0 = numpy.array(atm)
    bas0 = numpy.array(bas)
    atm0[:,PTR_COORD] += off
    atm0[:,PTR_MASS ] += off
    bas0[:,PTR_EXP  ] += off
    bas0[:,PTR_COEFF] += off
    return atm0.tolist(), bas0.tolist()

# concatenate two mol
def conc_env(atm1, bas1, env1, atm2, bas2, env2):
    atm2, bas2 = shift_ptr(atm2, bas2, len(env1))
    return atm1+atm2, bas1+bas2, env1+env2

# <bas-of-mol1|intor|bas-of-mol2>
def intor_cross(intor, mol1, mol2, comp=1):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = conc_env(mol1._atm, mol1._bas, mol1._env, \
                                mol2._atm, mol2._bas, mol2._env)
    bras = range(nbas1)
    kets = range(nbas1, nbas1+nbas2)
    return moleintor.getints(intor, atmc, basc, envc, bras, kets, comp, 0)

# append (charge, pointer to coordinates, nuc_mod) to _atm
def make_atm_env(atom, ptr=0):
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
    ''' generate arguments for integrals '''
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
    nelectron = -mol.charge
    for ia in range(mol.natm):
        nelectron += mol.atom_charge(ia)
    return nelectron

def copy(mol):
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
    mol = Mole()
    mol.__dict__.update(moldic)
    return mol

def len_spinor(l, kappa):
    if kappa == 0:
        n = (l * 4 + 2)
    elif kappa < 0:
        n = (l * 2 + 2)
    else:
        n = (l * 2)
    return n

def len_cart(l):
    return (l + 1) * (l + 2) // 2

def npgto_nr(mol):
    ''' total number of primitive GTOs'''
    return reduce(lambda n, b: n + (mol.bas_angular(b) * 2 + 1) \
                                * mol.bas_nprim(b),
                  range(len(mol._bas)), 0)
def nao_nr(mol):
    ''' total number of contracted GTOs'''
    return sum([(mol.bas_angular(b) * 2 + 1) * mol.bas_nctr(b) \
                for b in range(len(mol._bas))])

# nao_id0:nao_id1 corresponding to bas_id0:bas_id1
def nao_nr_range(mol, bas_id0, bas_id1):
    nao_id0 = sum([(mol.bas_angular(b) * 2 + 1) * mol.bas_nctr(b) \
                   for b in range(bas_id0)])
    n = sum([(mol.bas_angular(b) * 2 + 1) * mol.bas_nctr(b) \
             for b in range(bas_id0, bas_id1)])
    return nao_id0, nao_id0+n

def nao_2c(mol):
    ''' total number of spinors'''
    return sum([mol.bas_len_spinor(b) * mol.bas_nctr(b) \
                for b in range(len(mol._bas))])

# nao_id0:nao_id1 corresponding to bas_id0:bas_id1
def nao_2c_range(mol, bas_id0, bas_id1):
    nao_id0 = sum([mol.bas_len_spinor(b) * mol.bas_nctr(b) \
                   for b in range(bas_id0)])
    n = sum([mol.bas_len_spinor(b) * mol.bas_nctr(b) \
             for b in range(bas_id0, bas_id1)])
    return nao_id0, nao_id0+n

def ao_loc_nr(mol):
    off = 0
    ao_loc = []
    for i in range(len(mol._bas)):
        ao_loc.append(off)
        off += (mol.bas_angular(i) * 2 + 1) * mol.bas_nctr(i)
    ao_loc.append(off)
    return ao_loc

def ao_loc_2c(mol):
    off = 0
    ao_loc = []
    for i in range(len(mol._bas)):
        ao_loc.append(off)
        off += mol.bas_len_spinor(i) * mol.bas_nctr(i)
    ao_loc.append(off)
    return ao_loc

def time_reversal_map(mol):
    '''tao = time_reversal_map(bas)
    tao(i) = -j  means  T(f_i) = -f_j
    tao(i) =  j  means  T(f_i) =  f_j'''
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
    for ib in range(len(mol._bas)):
        ia = mol.bas_atom(ib)
        l1 = mol.bas_angular(ib)
        if ia == atm_id and l1 == l:
            return ib

def search_ao_nr(mol, atm_id, l, m, atmshell):
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
    '''Define molecular system
mol = Mole()
mol.build(
    verbose,
    output,
    max_memory,
    charge,
    spin,  # 2j
)
    '''
    def __init__(self):
        self.verbose = log.ERROR
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
        self.basis = {}
# self.nucmod = {atom_symbol: nuclear_model, atom_id: nuc_mod}, atom_id is 1-based
        self.nucmod = {}
# self.mass = {atom_symbol: mass, atom_id: mass}, atom_id is 1-based
        self.mass = {}
# self.grids = {atom_type/nuc_charge: [num_grid_radial, num_grid_angular]}
        self.grids = {}
##################################################
# don't modify the following private variables, they are not input options
# _atm, _bas, _env save the formated inputs
# arguments of integrals
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
        self._keys = set(self.__dict__.keys()).union(['_keys'])

    def check_sanity(self, obj):
        if self.verbose > log.QUIET:
            keysub = set(obj.__dict__.keys()) - set(obj._keys)
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
        assert((self.nelectron+self.spin) % 2 == 0)

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

    def shift_ptr(self, atm, bas, off):
        return shift_ptr(atm, bas, off)

    def make_env(self, atoms, basis, pre_env=[], nucmod={}, mass={}):
        return make_env(atoms, basis, pre_env, nucmod, mass)

    def make_atm_env(self, atom, ptr=0):
        return make_atm_env(atom, ptr)

    def make_bas_env(self, basis_add, atom_id=0, ptr=0):
        return make_bas_env(basis_add, atom_id, ptr)

    def tot_electrons(self):
        return tot_electrons(self)


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
        self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3] = coord

    def set_rinv_orig_(self, coord):
        # unit of input coord BOHR
        self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = coord[:3]

#NOTE: atm_id or bas_id start from 0
    def atom_symbol(self, atm_id):
        # a molecule can contain different symbols (C1,C2,..) for same type of
        # atoms
        return _symbol(self.atom[atm_id][0])

    def atom_pure_symbol(self, atm_id):
        # symbol without index, so (C1,C2,...) just return the same symbol 'C'
        return _symbol(self.atom_charge(atm_id))

    def atom_charge(self, atm_id):
        return self._atm[atm_id][CHARGE_OF]

    def atom_coord(self, atm_id):
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

    def atom_nshells(self, atm_id):
        symb = self.atom_symbol(atm_id)
        return len(self._basis[symb])

    def atom_shell_ids(self, atm_id):
        return [ib for ib in range(len(self._bas)) \
                if self.bas_atom(ib) == atm_id]

    def bas_coord(self, bas_id):
        atm_id = self.bas_atom(bas_id) - 1
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

    def bas_atom(self, bas_id):
        return self._bas[bas_id][ATOM_OF]

    def bas_angular(self, bas_id):
        return self._bas[bas_id][ANG_OF]

    def bas_nctr(self, bas_id):
        return self._bas[bas_id][NCTR_OF]

    def bas_nprim(self, bas_id):
        return self._bas[bas_id][NPRIM_OF]

    def bas_kappa(self, bas_id):
        return self._bas[bas_id][KAPPA_OF]

    def bas_exp(self, bas_id):
        nprim = self.bas_nprim(bas_id)
        ptr = self._bas[bas_id][PTR_EXP]
        return numpy.array(self._env[ptr:ptr+nprim])

    def bas_ctr_coeff(self, bas_id):
        nprim = self.bas_nprim(bas_id)
        nctr = self.bas_nctr(bas_id)
        ptr = self._bas[bas_id][PTR_COEFF]
        return numpy.array(self._env[ptr:ptr+nprim*nctr]).reshape(nprim,nctr)

    def bas_len_spinor(self, bas_id):
        l = self.bas_angular(bas_id)
        k = self.bas_kappa(bas_id)
        return len_spinor(l, k)

    def bas_len_cart(self, bas_id):
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
        '''non-relativitic and relativitic integral generator.
        hermi=1 : hermitian, hermi=2 : anti-hermitian'''
        return moleintor.getints(intor, self._atm, self._bas, self._env,
                                 comp=comp, hermi=hermi)

    def intor_symmetric(self, intor, comp=1):
        '''hermi integral generator.'''
        return self.intor(intor, comp, 1)

    def intor_asymmetric(self, intor, comp=1):
        '''anti-hermi integral generator.'''
        return self.intor(intor, comp, 2)

    def intor_cross(self, intor, bras, kets, comp=1):
        '''bras: shell lists of bras, kets: shell lists of kets'''
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


