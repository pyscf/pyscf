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
import numpy
import pyscf.lib.parameters as param
import pyscf.lib.logger as log
import cmd_args
import basis
import moleintor

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


############

_rm_digit = lambda s: ''.join(i for i in s if not i.isdigit())
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

def format_atom(atoms, origin=0, axes=1):
    '''
    change nuclear charge to atom symbol, rotate and shift molecule
    '''
    elementdic = dict((k.upper(),v) for k,v in param.ELEMENTS_PROTON.items())
    fmt_atoms = []
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
    fmt_basis = {}
    for atom in basis_tab.keys():
        symb = _symbol(atom)

        if isinstance(basis_tab[atom], str):
            fmt_basis[symb] = basis.load(basis_tab[atom], _rm_digit(symb))
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
    ''' moleinfo for contracted GTO '''
    def __init__(self):
        self.verbose = log.ERROR
        self.output = None
        self.max_memory = param.MEMORY_MAX

        self.light_speed = param.LIGHTSPEED
        self.charge = 0
        self.spin = 0
        self.symmetry = False

# atom, etb, basis, nucmod, mass, grids to save inputs
# self.atom = [(symb/nuc_charge, (coord(Angstrom):0.,0.,0.),
#               nucmod, mass, rad, ang), ...]
        self.atom = []
# self.basis = {atom_type/nuc_charge: [l, kappa, (expnt, c_1, c_2,..),..]}
        self.basis = {}
# self.nucmod = {atom#: nuclear_model, }, atom# is atom index, 1-based
        self.nucmod = {}
# self.mass = {atom#: mass, }, atom# is atom index, 1-based
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
        self.pgname = 'C1'
        self.nelectron = 0
        self.symm_orb = None
        self.irrep_name = None
        self._built = False
        self._keys = set(self.__dict__.keys() + ['_keys'])

    def check_sanity(self, obj):
        if self.verbose > log.QUITE:
            keysub = set(obj.__dict__.keys()) - obj._keys
            if keysub:
                print('%s has no attributes %s' %
                      (str(obj.__class__), ' '.join(keysub)))

# need "deepcopy" here because in shallow copy, _env may get new elements but
# with ptr_env unchanged
# def __copy__(self):
#        cls = self.__class__
#        newmol = cls.__new__(cls)
#        newmol = ...
# do not use __copy__ to aovid iteratively call copy.copy
    def copy(self):
        import copy
        newmol = copy.copy(self)
        newmol._atm = copy.deepcopy(self._atm)
        newmol._bas = copy.deepcopy(self._bas)
        newmol._env = copy.deepcopy(self._env)
        newmol.atom    = copy.deepcopy(self.atom)
        newmol.basis   = copy.deepcopy(self.basis)
        return newmol

    # cannot use __getstate__ for pickle here, because it affects copy.copy()
    #def __getstate__(self):
    #    return {'atom'    : self.atom, \
    #            'basis'   : self.basis, \
    #            'charge'  : self.charge, \
    #            'spin'    : self.spin, \
    #            'symmetry': self.symmetry, \
    #            'nucmod'  : self.nucmod, \
    #            'mass'    : self.mass, \
    #            'grids'   : self.grids }
    #def __setstate__(self, moldic):
    #    self.__dict__.update(moldic)
    def pack(self):
        return {'atom'    : self.atom,
                'basis'   : self.basis,
                'charge'  : self.charge,
                'spin'    : self.spin,
                'symmetry': self.symmetry,
                'nucmod'  : self.nucmod,
                'mass'    : self.mass,
                'grids'   : self.grids,
                'light_speed': self.light_speed}
    def unpack(self, moldic):
        self.__dict__.update(moldic)


    def update_from_cmdargs(self):
        # Ipython shell conflicts with optparse
        # pass sys.args when using ipython
        try:
            __IPYTHON__
            print('Warn: Ipython shell catchs sys.args')
            return None
        except:
            pass

        if not self._built: # parse cmdline args only once
            opts = cmd_args.cmd_args()

            if opts.verbose:
                self.verbose = opts.verbose
            if opts.max_memory:
                self.max_memory = opts.max_memory

            if opts.output:
                self.output = opts.output

        if self.output is not None:
            if os.path.isfile(self.output):
                #os.remove(self.output)
                if self.verbose > log.QUITE:
                    print('overwrite output file: %s' % self.output)
            else:
                if self.verbose > log.QUITE:
                    print('output file: %s' % self.output)


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
            self.update_from_cmdargs()

        # avoid to open output file twice
        if parse_arg and self.output is not None \
           and self.stdout.name != self.output:
            self.stdout = open(self.output, 'w')

        self.check_sanity(self)

        self._built = True

        if not self.symmetry:
            self.atom = self.format_atom(self.atom)
        else:
            from pyscf import symm
            #if self.symmetry in symm.param.POINTGROUP
            #    self.pgname = self.symmetry
            #    #todo: symm.check_given_symm(self.symmetric, self.atom)
            #    pass
            #else:
            #    self.pgname, inp_atoms = symm.detect_symm(self.atom)
            self.pgname, origin, axes = symm.detect_symm(self.atom)
            self.atom = self.format_atom(self.atom, origin, axes)
        self.basis = self.format_basis(self.basis)

        self._env[PTR_LIGHT_SPEED] = self.light_speed
        self._atm, self._bas, self._env = \
                self.make_env(self.atom, self.basis, self._env, \
                              self.nucmod, self.mass)
        self.natm = self._atm.__len__()
        self.nbas = self._bas.__len__()
        self.nelectron = self.tot_electrons()

        if self.symmetry:
            from pyscf import symm
            eql_atoms = symm.symm_identical_atoms(self.pgname, self.atom)
            symm_orb = symm.symm_adapted_basis(self.pgname, eql_atoms,\
                                               self.atom, self.basis)
            self.irrep_id = [ir for ir in range(len(symm_orb)) \
                             if symm_orb[ir].size > 0]
            self.irrep_name = [symm.irrep_name(self.pgname,ir) \
                               for ir in self.irrep_id]
            self.symm_orb = [c for c in symm_orb if c.size > 0]

        if dump_input and self.verbose >= log.NOTICE:
            self.dump_input()

        log.debug1(self, 'arg.atm = %s', self._atm)
        log.debug1(self, 'arg.bas = %s', self._bas)
        log.debug1(self, 'arg.env = %s', self._env)
        return self._atm, self._bas, self._env

    @classmethod
    def format_atom(self, atom, origin=0, axes=1):
        return format_atom(atom, origin, axes)

    @classmethod
    def format_basis(self, basis_tab):
        return format_basis(basis_tab)

    @classmethod
    def expand_etb(self, l, n, alpha, beta):
        return expand_etb(l, n, alpha, beta)

    @classmethod
    def expand_etbs(self, etbs):
        return expand_etbs(etbs)

    @classmethod
    def shift_ptr(self, atm, bas, off):
        return shift_ptr(atm, bas, off)

    @classmethod
    def make_env(self, atoms, basis, pre_env=[], nucmod={}, mass={}):
        ''' generate arguments for integrals '''

        _atm = []
        _bas = []
        _env = []
        ptr_env = len(pre_env)

        for ia, atom in enumerate(atoms):
            symb = atom[0]
            atm0, env0 = self.make_atm_env(atom, ptr_env)
            ptr_env = ptr_env + len(env0)
            if ia in nucmod:
                atm0[NUC_MOD_OF] = nucmod[ia]
            elif symb in nucmod:
                atm0[NUC_MOD_OF] = nucmod[symb]
            if ia in mass:
                atm0[PTR_MASS] = ptr_env
                env0.append(mass[ia])
                ptr_env = ptr_env + 1
            elif symb in mass:
                atm0[PTR_MASS] = ptr_env
                env0.append(mass[symb])
                ptr_env = ptr_env + 1
            _atm.append(atm0)
            _env.append(env0)

        _basdic = {}
        for symb, basis_add in basis.items():
            bas0, env0 = self.make_bas_env(basis_add, 0, ptr_env)
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

# append (charge, pointer to coordinates, nuc_mod) to _atm
    @classmethod
    def make_atm_env(self, atom, ptr=0):
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
    @classmethod
    def make_bas_env(self, basis_add, atom_id=0, ptr=0):
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

    def tot_electrons(self):
        return tot_electrons(self)


    def dump_input(self):
        if not self._built:
            self.build()

        try:
            filename = os.path.join(os.getcwd(), sys.argv[0])
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

        for nuc,(rad,ang) in self.grids.items():
            self.stdout.write('[INPUT] %s (%d, %d)\n' % (nuc, rad, ang))

        for ia,atom in enumerate(self.atom):
            if ia in self.nucmod \
                and self.nucmod[ia] == param.MI_NUC_GAUSS:
                symb = atom[0]
                if ia in self.mass:
                    mass = self.mass[ia]
                elif symb in self.mass:
                    mass = self.mass[symb]
                else:
                    mass = param.ELEMENTS[_charge(symb)][1]
                nucmod = ', Gaussian nuc-mod, mass %s' % mass
            else:
                nucmod = ''
            self.stdout.write('[INPUT] %d %s %s AA, '\
                              '%s Bohr%s\n' \
                              % (ia+1, _symbol(atom[0]), atom[1], \
                                 map(lambda x: x/param.BOHR, atom[1]), \
                                 nucmod))

        self.stdout.write('[INPUT] basis\n')
        self.stdout.write('[INPUT] l, kappa, [nprim/nctr], ' \
                          'expnt,             c_1 c_2 ...\n')
        for atom, basis in self.basis.items():
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

        log.info(self, 'nuclear repulsion = %.15g', self.nuclear_repulsion())
        if self.symmetry:
            log.info(self, 'point group symmetry = %s', self.pgname)
            for ir in range(self.symm_orb.__len__()):
                log.info(self, 'num. orbitals of %s = %d', \
                         self.irrep_name[ir], self.symm_orb[ir].shape[1])
        log.info(self, 'number of shells = %d', self.nbas)
        log.info(self, 'number of NR pGTOs = %d', self.npgto_nr())
        log.info(self, 'number of NR cGTOs = %d', self.nao_nr())
        if self.verbose >= log.DEBUG1:
            for i in range(len(self._bas)):
                exps = self.exps_of_bas(i)
                log.debug1(self, 'bas %d, expnt(s) = %s', i, str(exps))

        log.info(self, 'CPU time: %12.2f', time.clock())

    def set_common_origin(self, coord):
        if max(coord) < 1e3 and min(coord) > -1e3:
            for i in range(3):
                self._env[PTR_COMMON_ORIG+i] = coord[i]
        else:
            log.warn(self, 'incorrect gauge origin, set common gauge (0,0,0)')
            self._env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3] = (0,0,0)

    def set_rinv_orig(self, coord):
        # unit of input coord BOHR
        self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = coord[:3]

    def set_rinv_by_atm_id(self, atm_id):
        if atm_id >= 0 and atm_id <= self.natm:
            self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = \
                    self.coord_of_atm(atm_id-1)[:]
        else:
            log.warn(self, 'incorrect center, set to first atom')
            self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = \
                    self.coord_of_atm(0)[:]

#NOTE: atm_id or bas_id start from 0
    def symbol_of_atm(self, atm_id):
        # a molecule can contain different symbols (C1,C2,..) for same type of
        # atoms
        return _symbol(self.atom[atm_id][0])

    def pure_symbol_of_atm(self, atm_id):
        # symbol without index, so (C1,C2,...) just return the same symbol 'C'
        return _symbol(self.charge_of_atm(atm_id))

    def charge_of_atm(self, atm_id):
        return self._atm[atm_id][CHARGE_OF]

    def coord_of_atm(self, atm_id):
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

    def coord_of_bas(self, bas_id):
        atm_id = self.atom_of_bas(bas_id) - 1
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

    def nbas_of_atm(self, atm_id):
        symb = self.symbol_of_atm(atm_id)
        return self.basis[symb].__len__()

    def basids_of_atm(self, atm_id):
        return [ib for ib in range(len(self._bas)) \
                if self.atom_of_bas(ib) == atm_id]

    def atom_of_bas(self, bas_id):
        return self._bas[bas_id][ATOM_OF]

    def angular_of_bas(self, bas_id):
        return self._bas[bas_id][ANG_OF]

    def nctr_of_bas(self, bas_id):
        return self._bas[bas_id][NCTR_OF]

    def nprim_of_bas(self, bas_id):
        return self._bas[bas_id][NPRIM_OF]

    def kappa_of_bas(self, bas_id):
        return self._bas[bas_id][KAPPA_OF]

    def exps_of_bas(self, bas_id):
        nprim = self.nprim_of_bas(bas_id)
        ptr = self._bas[bas_id][PTR_EXP]
        return numpy.array(self._env[ptr:ptr+nprim])

    def cgto_coeffs_of_bas(self, bas_id):
        nprim = self.nprim_of_bas(bas_id)
        nctr = self.nctr_of_bas(bas_id)
        ptr = self._bas[bas_id][PTR_COEFF]
        return numpy.array(self._env[ptr:ptr+nprim*nctr]).reshape(nprim,nctr)

    def len_spinor_of_bas(self, bas_id):
        l = self.angular_of_bas(bas_id)
        k = self.kappa_of_bas(bas_id)
        if k == 0:
            n = (l * 4 + 2)
        elif k < 0:
            n = (l * 2 + 2)
        else:
            n = (l * 2)
        return n

    def len_cart_of_bas(self, bas_id):
        l = self._bas[bas_id][ANG_OF]
        return (l + 1) * (l + 2) / 2


    def num_NR_pgto(self):
        return self.npgto_nr()
    def npgto_nr(self):
        ''' total number of primitive GTOs'''
        return reduce(lambda n, b: n + (self.angular_of_bas(b) * 2 + 1) \
                                    * self.nprim_of_bas(b),
                      range(len(self._bas)), 0)

    def num_NR_function(self):
        return self.num_NR_cgto()
    def nao_nr(self):
        ''' total number of contracted GTOs'''
        return sum([(self.angular_of_bas(b) * 2 + 1) * self.nctr_of_bas(b) \
                    for b in range(len(self._bas))])
    def num_NR_cgto(self):
        return self.nao_nr()
# nao_id0:nao_id1 corresponding to bas_id0:bas_id1
    def nao_nr_range(self, bas_id0, bas_id1):
        nao_id0 = sum([(self.angular_of_bas(b) * 2 + 1) * self.nctr_of_bas(b) \
                       for b in range(bas_id0)])
        n = sum([(self.angular_of_bas(b) * 2 + 1) * self.nctr_of_bas(b) \
                 for b in range(bas_id0, bas_id1)])
        return nao_id0, nao_id0+n

    def num_4C_function(self):
        return self.num_4C_cgto()
    def num_4C_cgto(self):
        return self.num_2C_function() * 2
    def nao_4c(self):
        return self.num_4C_cgto()

    def num_2C_function(self):
        return self.nao_2c()
    def nao_2c(self):
        ''' total number of spinors'''
        return sum([self.len_spinor_of_bas(b) * self.nctr_of_bas(b) \
                    for b in range(len(self._bas))])
    def num_2C_cgto(self):
        return self.nao_2c()
# nao_id0:nao_id1 corresponding to bas_id0:bas_id1
    def nao_2c_range(self, bas_id0, bas_id1):
        nao_id0 = sum([self.len_spinor_of_bas(b) * self.nctr_of_bas(b) \
                       for b in range(bas_id0)])
        n = sum([self.len_spinor_of_bas(b) * self.nctr_of_bas(b) \
                 for b in range(bas_id0, bas_id1)])
        return nao_id0, nao_id0+n

    def ao_loc_nr(self):
        off = 0
        ao_loc = []
        for i in range(len(self._bas)):
            ao_loc.append(off)
            off += (self.angular_of_bas(i) * 2 + 1) * self.nctr_of_bas(i)
        ao_loc.append(off)
        return ao_loc

    def ao_loc_2c(self):
        off = 0
        ao_loc = []
        for i in range(len(self._bas)):
            ao_loc.append(off)
            off += self.len_spinor_of_bas(i) * self.nctr_of_bas(i)
        ao_loc.append(off)
        return ao_loc

    def time_reversal_map(self):
        '''tao = time_reversal_map(bas)
        tao(i) = -j  means  T(f_i) = -f_j
        tao(i) =  j  means  T(f_i) =  f_j'''
        tao = []
        i = 0
        for b in self._bas:
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

    def intor(self, intor, dim3=1, hermi=0):
        '''non-relativitic and relativitic integral generator.
        hermi=1 : hermitian, hermi=2 : anti-hermitian'''
        return moleintor.getints(intor, self._atm, self._bas, self._env,
                                 dim3=dim3, hermi=hermi)

    def intor_symmetric(self, intor, dim3=1):
        '''hermi integral generator.'''
        return self.intor(intor, dim3, 1)

    def intor_asymmetric(self, intor, dim3=1):
        '''anti-hermi integral generator.'''
        return self.intor(intor, dim3, 2)

    def intor_cross(self, intor, bras, kets, dim3=1):
        '''bras: shell lists of bras, kets: shell lists of kets'''
        return moleintor.getints(intor, self._atm, self._bas, self._env,
                                 bras, kets, dim3, 0)

    def get_enuc(self):
        return nuclear_repulsion()
    def nuclear_repulsion(self):
        if self.natm == 0:
            return 0
        chargs = numpy.array([self.charge_of_atm(i) for i in range(len(self._atm))])
        coords = numpy.array([self.coord_of_atm(i) for i in range(len(self._atm))])
        xx = coords[:,0].reshape(-1,1) - coords[:,0]
        yy = coords[:,1].reshape(-1,1) - coords[:,1]
        zz = coords[:,2].reshape(-1,1) - coords[:,2]
        r = numpy.sqrt(xx**2 + yy**2 + zz**2 + 1e-60)
        qq = chargs[:,None] * chargs[None,:]
        qq[numpy.diag_indices(len(self._atm))] = 0
        e = (qq/r).sum() * .5
        return e

    def inter_distance(self):
        rr = numpy.zeros((self.natm, self.natm))
        for j in range(len(self._atm)):
            r2 = self.coord_of_atm(j)
            for i in range(j):
                r1 = self.coord_of_atm(i)
                rr[i,j] = rr[j,i] = numpy.linalg.norm(r1-r2)
        return rr

    def spheric_labels(self):
        count = numpy.zeros((self.natm, 9), dtype=int)
        label = []
        i = 0
        for ib in range(len(self._bas)):
            ia = self.atom_of_bas(ib)
            l = self.angular_of_bas(ib)
            strl = param.ANGULAR[l]
            nc = self.nctr_of_bas(ib)
            symb = self.symbol_of_atm(ia)
            for n in range(count[ia,l]+l+1, count[ia,l]+l+1+nc):
                for m in range(-l, l+1):
                    label.append((ia, symb, '%d%s' % (n, strl), \
                                  '%s' % param.REAL_SPHERIC[l][l+m]))
                    i += 1
            count[ia,l] += nc
        return label

    def search_bas_id(self, atm_id, l):
        for ib in range(len(self._bas)):
            ia = self.atom_of_bas(ib)
            l1 = self.angular_of_bas(ib)
            if ia == atm_id and l1 == l:
                return ib

    def search_spheric_id(self, atm_id, l, m, atmshell):
        ibf = 0
        for ib in range(len(self._bas)):
            ia = self.atom_of_bas(ib)
            l1 = self.angular_of_bas(ib)
            nc = self.nctr_of_bas(ib)
            if ia == atm_id and l1 == l:
                return ibf + (atmshell-l)*(l*2+1) + (l+m)
            ibf += (l*2+1) * nc

#TODO:    def search_spinor_id(self, atm_id, l, m, kappa, atmshell):
#TODO:        ibf = 0
#TODO:        for ib in range(len(self._bas)):
#TODO:            ia = self.atom_of_bas(ib)
#TODO:            l1 = self.angular_of_bas(ib)
#TODO:            nc = self.nctr_of_bas(ib)
#TODO:            k = self.kappa_of_bas(bas_id)
#TODO:            if ia == atm_id and l1 == l and k == kappa:
#TODO:                return ibf + (atmshell-l)*(l*4+2) + (l+m)
#TODO:            ibf += (l*4+2) * nc

#TODO:    def labels_of_spinor_GTO(self):
#TODO:        return self.spinor_labels(self):
#TODO:    def spinor_labels(self):
#TODO:        count = numpy.zeros((self.natm, 9), dtype=int)
#TODO:        label = []
#TODO:        i = 0
#TODO:        for ib in range(len(self._bas)):
#TODO:            ia = self.atom_of_bas(ib)
#TODO:            l = self.angular_of_bas(ib)
#TODO:            degen = mol.len_spinor_of_bas(ib)
#TODO:            strl = param.ANGULAR[l]
#TODO:            nc = self.nctr_of_bas(ib)
#TODO:            symb = self.symbol_of_atm(ia)
#TODO:            for n in range(count[ia,l]+l+1, count[ia,l]+l+1+nc):
#TODO:                if l == 0
#TODO:                if degen == l * 2 or degen == l * 4 + 2:
#TODO:                    for m in range(-l*2-1, l*2+1, 2):
#TODO:                        label.append((ia, symb, '%d%s' % (n, strl), m))
#TODO:                        i += 1
#TODO:                if degen == l * 2 + 2 or degen == l * 4 + 2:
#TODO:                    for m in range(-l*2-1, l*2+2, 2):
#TODO:                        label.append((ia, symb, '%d%s' % (n, strl), m))
#TODO:                        i += 1
#TODO:            count[ia,l] += nc
#TODO:        return label


def is_same_mol(mol1, mol2):
    if mol1.atom.__len__() != mol2.atom.__len__():
        return False
    for a1, a2 in zip(mol1.atom, mol2.atom):
        if a1[0] != a2[0] \
           or numpy.linalg.norm(numpy.array(a1[1])-numpy.array(a2[1])) > 2:
            return False
    return True

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
def intor_cross(intor, mol1, mol2, dim3=1):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = conc_env(mol1._atm, mol1._bas, mol1._env, \
                                mol2._atm, mol2._bas, mol2._env)
    bras = range(nbas1)
    kets = range(nbas1, nbas1+nbas2)
    return moleintor.getints(intor, atmc, basc, envc, bras, kets, dim3, 0)

def tot_electrons(mol):
    nelectron = -mol.charge
    for ia in range(mol.natm):
        nelectron += mol.charge_of_atm(ia)
    return nelectron

