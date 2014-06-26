#!/usr/bin/env python
# -*- coding: utf-8
#
# File: mole.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#



__author__ = 'Qiming Sun <osirpt.sun@gmail.com>'
__version__ = '$ 0.2 $'

import os, sys
import tempfile
import time
import math
import numpy
import lib.parameters as param
import lib.logger as log
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
    if isinstance(symb_or_chg, str):
        return symb_or_chg
    else:
        return param.ELEMENTS[symb_or_chg][0]


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
        self.nelectron = 0
        self.charge = 0
        self.spin = 0
        self.symmetry = False

# atom, etb, basis, nucmod, mass, grids to save inputs
# self.atom = [(atom_type/nuc_charge, (coordinate(Angstrom):0.,0.,0.)), ]
        self.atom = []
#    self.etb = [{
#          'atom'      : 1           # atom_type/nuc_charge
#        , 'max_l'     : 0           # for even-tempered basis
#        , 's'         : (2, 2, 1.8) # for etb:(num_basis, alpha, beta)
#        , 'p'         : (0, 1, 1.8) # for etb: eta = alpha*beta**i
#        , 'd'         : (0, 1, 1.8) #           for i in range num_basis
#        , 'f'         : (0,0,0)
#        , 'g'         : (0,0,0)}, ]
        self.etb = {}
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
        self.ptr_env = PTR_ENV_START
        self._env = [0] * self.ptr_env
        self._env[PTR_LIGHT_SPEED] = param.LIGHTSPEED
        self._gauge_method = param.MI_GAUGE_GIAO

        self._built = False
        self.fout = sys.stdout
        self.pgname = 'C1'
        self.symm_orb = None
        self.irrep_name = None

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
        newmol.etb     = copy.deepcopy(self.etb)
        newmol.basis   = copy.deepcopy(self.basis)
        return newmol

    # cannot use __getstate__ for pickle here, because it affects copy.copy()
    #def __getstate__(self):
    #    return {'atom'    : self.atom, \
    #            'basis'   : self.basis, \
    #            'etb'     : self.etb, \
    #            'charge'  : self.charge, \
    #            'spin'    : self.spin, \
    #            'symmetry': self.symmetry, \
    #            'nucmod'  : self.nucmod, \
    #            'mass'    : self.mass, \
    #            'grids'   : self.grids }
    #def __setstate__(self, moldic):
    #    self.__dict__.update(moldic)
    def pack(self):
        return {'atom'    : self.atom, \
                'basis'   : self.basis, \
                'etb'     : self.etb, \
                'charge'  : self.charge, \
                'spin'    : self.spin, \
                'symmetry': self.symmetry, \
                'nucmod'  : self.nucmod, \
                'mass'    : self.mass, \
                'grids'   : self.grids }
    def unpack(self, moldic):
        self.__dict__.update(moldic)


    def update_from_cmdargs(self):
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
                    print 'overwrite output file: %s' % self.output
            else:
                if self.verbose > log.QUITE:
                    print 'output file: %s' % self.output


    def format_atom(self):
        '''
        change nuclear charge to atom symbol
        '''
        if not self.symmetry:
            inp_atoms = self.atom
        else:
            import symm
            #if self.symmetry in symm.param.POINTGROUP
            #    self.pgname = self.symmetry
            #    #todo: symm.check_given_symm(self.symmetric, self.atom)
            #    pass
            #else:
            #    self.pgname, inp_atoms = symm.detect_symm(self.atom)
            self.pgname, inp_atoms = symm.detect_symm(self.atom)
        atoms = []
        for i,atom in enumerate(inp_atoms):
            symb = _symbol(atom[0])
            # atom index = i+1 since atom# is 1-based and _atm is 0-based
            if self.nucmod.has_key(i+1):
                nucmod = self.nucmod[i+1]
            elif self.nucmod.has_key(symb):
                nucmod = self.nucmod[symb]
            else:
                nucmod = param.MI_NUC_POINT
            if self.mass.has_key(i+1):
                mass = self.mass[i+1]
            elif self.mass.has_key(symb):
                mass = self.mass[symb]
            else:
                mass = param.ELEMENTS[_charge(symb)][1]
            if self.grids.has_key(symb):
                rad,ang = self.grids[symb]
            else:
                rad,ang = 80,110
            atoms.append((symb, numpy.array(atom[1]), nucmod, mass, rad, ang))
        return atoms


    def format_basis(self, basis_tab):
        '''
        transform the basis to the contracted format:
        { atom: (l, kappa, ((-exp, c_1, c_2, ..), ..)), ... }
        '''
        #ABORT def append_cgto(blist, b):
        #ABORT     if b[2].__class__ is types.IntType \
        #ABORT        or b[2].__class__ is types.FloatType:
        #ABORT         # append uncontracted GTO: (l, kappa, expnt)
        #ABORT         blist.append((b[0], b[1], (abs(b[2]), 1)))
        #ABORT     else:
        #ABORT         # append cGTO: (l, kappa, (expnt, c_1, c_2,..), ..)
        #ABORT         blist.append(b)

        fmtbas = {}
        for atom in basis_tab.keys():
            if not fmtbas.has_key(_symbol(atom)):
                fmtbas[_symbol(atom)] = []

            if isinstance(basis_tab[atom], str):
                name = basis_tab[atom].lower().replace(' ', '').replace('-', '').replace('_', '')
                bset = basis.alias[name][_rm_digit(atom)]
            else:
                bset = basis_tab[atom]

            for b in bset:
        #ABORT        append_cgto(fmtbas[_symbol(atom)], b)
                fmtbas[_symbol(atom)].append(b)
        return fmtbas

    def format_etb(self, etb):
        '''
        expand the even-tempered basis
        basis = { atom: (l, kappa, ((-exp, c_1, c_2, ..), ..)), ... }
        '''
        bases = {}
        for atom,b in etb.items():
            bases[_symbol(atom)] = []
            max_l = b['max_l']
            for l in range(max_l+1):
                strl = param.ANGULAR[l]
                try:
                    num_basis, alpha, beta = b[strl]
                except:
                    raise KeyError('unkown angular %s' % strl)
                for i in range(num_basis):
                    bases[_symbol(atom)].append((l, 0, (abs(alpha*beta**i), 1)))
        return bases

    def merge_etb(self, etb, basis):
        '''
        merge even-tempered basis
        '''
        #basis = self.format_basis(basis)
        etb = self.format_etb(etb)
        for atom in set(basis.keys() + etb.keys()):
            if basis.has_key(atom):
                if etb.has_key(atom):
                    basis[atom].extend(etb[atom])
            else:
                basis[atom] = etb[atom]

        for atom in self.atom:
            symb = _symbol(atom[0])
            if not basis.has_key(symb) \
               and not basis.has_key(_rm_digit(symb)):
                raise KeyError('no basis for %s' % atom[0])
        return basis

    def make_env(self):
        ''' env: arguments of integrals '''
        # clear env
        self._atm = []
        self._bas = []
        del(self._env[self.ptr_env:])

        self._env[PTR_LIGHT_SPEED] = self.light_speed

        ptr_env = self.ptr_env
        self.natm = self.atom.__len__()
        for i in range(self.natm):
            self._atm.append(self.make_atm_env_by_atm_id(i))
            symb = self.atom[i][0]
            if self.basis.has_key(symb):
                basis_add = self.basis[symb]
            else:
                basis_add = self.basis[_rm_digit(symb)]
            self._bas.extend(self.make_bas_env_by_atm_id(i, basis_add))
        self.nbas = self._bas.__len__()

    def make_atm_env_by_atm_id(self, atom_id):
        a = self.atom[atom_id]
        # append (charge, pointer to coordinates, nuc_mod) to _atm
        self._env.extend([x/param.BOHR for x in a[1]])
        _atm = [_charge(a[0]), self.ptr_env]
        _atm.extend(a[2:])
        self.ptr_env += 3
        return _atm

    def make_bas_env_by_atm_id(self, atom_id, basis_add):
        _bas = []
        # append (atom, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0) to bas
        for b in basis_add:
            angl = b[0]
            if isinstance(b[1], int):
                kappa = b[1]
                b_coeff = numpy.array(b[2:])
            else:
                kappa = 0
                b_coeff = numpy.array(b[1:])
            es = b_coeff[:,0]
            cs = b_coeff[:,1:]
            nprim = cs.shape[0]
            nctr = cs.shape[1]
            self._env.extend(es)
            norm = [gto_norm(angl, e) for e in es]
            # absorb normalization into contraction
            self._env.extend((cs.T * norm).flatten())
            ptr_exp = self.ptr_env
            self.ptr_env += nprim
            ptr_coeff = self.ptr_env
            self.ptr_env += nprim * nctr
            _bas.append([atom_id, angl, nprim, nctr, kappa, \
                         ptr_exp, ptr_coeff, 0])
        return _bas


    def tot_electrons(self):
        nelectron = -self.charge
        for atom in self.atom:
            nelectron += _charge(atom[0])
        return nelectron


    def build_moleinfo(self, dump_input=True):
        self.build(dump_input)
    def build(self, dump_input=True, parse_arg=True):
        # Ipython shell conflicts with optparse
        try:
            __IPYTHON__ is not None
        except:
            if parse_arg:
                self.update_from_cmdargs()
        else:
            print 'Warn: Ipython shell catchs sys.args'

        # avoid to open output file twice
        if parse_arg and self.output is not None \
           and self.fout.name != self.output:
            self.fout = open(self.output, 'w')

        self._built = True

        self.nelectron = self.tot_electrons()
        self.atom = self.format_atom()
        self.basis = self.format_basis(self.basis)
        self.basis = self.merge_etb(self.etb, self.basis)
        if self.symmetry:
            import symm
            eql_atoms = symm.symm_identical_atoms(self.pgname, self.atom)
            symm_orb = symm.symm_adapted_basis(self.pgname, eql_atoms,\
                                               self.atom, self.basis)
            self.irrep_id = [ir for ir in range(len(symm_orb)) \
                             if symm_orb[ir].size > 0]
            self.irrep_name = [symm.irrep_name(self.pgname,ir) \
                               for ir in self.irrep_id]
            self.symm_orb = [c for c in symm_orb if c.size > 0]
        self.make_env()

        if dump_input and self.verbose > log.QUITE:
            self.dump_input()

        log.debug(self, 'arg.atm = %s', self._atm)
        log.debug(self, 'arg.bas = %s', self._bas)
        log.debug(self, 'arg.env = %s', self._env)

#ABORT        # transform cartesian GTO to real spheric or spinor GTO
#ABORT        self.make_gto_cart2sph()
#ABORT        self.make_gto_cart2j_l()


    def dump_input(self):
        # initialize moleinfo first
        if self._bas is []:
            self.nelectron = self.tot_electrons()
            self.atom = self.format_atom()
            self.basis = self.format_basis()
            self.make_env()

        try:
            filename = '%s/%s' % (os.getcwd(), sys.argv[0])
            finput = open(filename, 'r')
            self.fout.write('\n')
            self.fout.write('INFO: **** input file is %s ****\n' % filename)
            self.fout.write(finput.read())
            self.fout.write('INFO: ******************** input file end ********************\n')
            self.fout.write('\n')
        except:
            print 'Warn: input file is not existed'

        self.fout.write('System: %s\n' % str(os.uname()))
        self.fout.write('Date: %s\n' % time.ctime())
        try:
            dn = os.path.dirname(os.path.realpath(__file__))
            self.fout.write('GIT version: ')
            # or command(git log -1 --pretty=%H)
            for branch in 'dev', 'master':
                fname = '/'.join((dn, "../.git/refs/heads", branch))
                fin = open(fname, 'r')
                d = fin.readline()
                fin.close()
                self.fout.write(' '.join((branch, d[:-1], '; ')))
            self.fout.write('\n\n')
        except:
            pass

        self.fout.write('[INPUT] VERBOSE %d\n' % self.verbose)
        self.fout.write('[INPUT] light speed = %s\n' % self.light_speed)
        self.fout.write('[INPUT] number of atoms = %d\n' % self.atom.__len__())
        self.fout.write('[INPUT] num electrons = %d\n' % self.nelectron)

        nucmod = { param.MI_NUC_POINT:'point', \
                param.MI_NUC_GAUSS:'Gaussian', }
        for a,atom in enumerate(self.atom):
            self.fout.write('[INPUT] atom %d, %s,   %s nuclear model, ' \
                            'mass %s, radial %s, angular %s\n' % \
                            (a+1, _symbol(atom[0]), nucmod[atom[2]],
                             atom[3], atom[4], atom[5]))
            self.fout.write('[INPUT]      (%.15g, %.15g, %.15g) AA, ' \
                            % tuple(atom[1]))
            self.fout.write('(%.15g, %.15g, %.15g) Bohr\n' % \
                            tuple(map(lambda x: x/param.BOHR, atom[1])))
        log.info(self, 'nuclear repulsion = %.15g', self.nuclear_repulsion())
        self.fout.write('[INPUT] basis = atom: (l, kappa, nprim/nctr)\n')
        self.fout.write('[INPUT]               (expnt, c_1, c_2, ...)\n')
        for atom, basis in self.basis.items():
            for b in basis:
                if isinstance(b[1], int):
                    kappa = b[1]
                    b_coeff = b[2:]
                else:
                    kappa = 0
                    b_coeff = b[1:]
                self.fout.write('[INPUT] %s : l = %d, kappa = %d, [%d/%d]\n' \
                                % (atom, b[0], kappa, b_coeff.__len__(), \
                                   b_coeff[0].__len__()-1))
                for x in b_coeff:
                    self.fout.write('[INPUT]    exp = %g, c = ' % x[0])
                    for c in x[1:]:
                        self.fout.write('%g, ' % c)
                    self.fout.write('\n')

        self.fout.write('[INPUT] %d set(s) of even-tempered basis\n' \
                        % self.etb.keys().__len__())
        for atom,basis in self.etb.items():
            max_l = basis['max_l']
            self.fout.write('[INPUT] etb for atom %s max_l = %d\n' \
                            % (_symbol(atom), max_l))
            for l in range(max_l+1):
                strl = param.ANGULAR[l]
                self.fout.write('[INPUT]      l = %s; (n, alpha, beta) = %s\n'\
                                % (strl, str(basis[strl])))
        if self.symmetry:
            log.info(self, 'point group symmetry = %s', self.pgname)
            for ir in range(self.symm_orb.__len__()):
                log.info(self, 'num. orbitals of %s = %d', \
                         self.irrep_name[ir], self.symm_orb[ir].shape[1])
        log.info(self, 'number of shells = %d', self.nbas)
        log.info(self, 'number of NR pGTOs = %d', self.num_NR_pgto())
        log.info(self, 'number of NR cGTOs = %d', self.num_NR_function())
        for i in range(self.nbas):
            exps = self.exps_of_bas(i)
            log.info(self, 'bas %d, expnt(s) = %s', i, str(exps))

        log.info(self, 'CPU time: %12.2f', time.clock())

    def set_common_gauge_origin(self, coord):
        if max(coord) < 1e3 and min(coord) > -1e3:
            for i in range(3):
                self._env[PTR_COMMON_ORIG+i] = coord[i]
        else:
            print 'incorrect gauge origin, set common gauge (0,0,0)'
            for i in range(3):
                self._env[PTR_COMMON_ORIG+i] = 0

    def set_rinv_orig(self, coord):
        # unit of input coord BOHR
        self._env[PTR_RINV_ORIG  ] = coord[0]
        self._env[PTR_RINV_ORIG+1] = coord[1]
        self._env[PTR_RINV_ORIG+2] = coord[2]

    def set_shielding_nuc(self, nuc):
        if nuc >= 0 and nuc <= self.atom.__len__():
            self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = \
                    self.coord_of_atm(nuc-1)[:]
        else:
            print 'incorrect center, set to first atom'
            self._env[PTR_RINV_ORIG:PTR_RINV_ORIG+3] = \
                    self.coord_of_atm(0)[:]

    def get_light_speed(self):
        return self._env[PTR_LIGHT_SPEED]
    def set_light_speed(self, c):
        if c > 0:
            self._env[PTR_LIGHT_SPEED] = c
            self.light_speed = c
        else:
            print 'Light speed should be > 0. Set light speed to', \
                    param.LIGHTSPEED

    @property
    def lightspeed(self):
        return self.get_light_speed()
    @lightspeed.setter
    def lightspeed(self, c):
        self.set_light_speed(c)


###############################################
#ABORT    def num_of_atoms(self):
#ABORT        return self.natm
#ABORT
#ABORT    def num_of_shells(self):
#ABORT        return self.nbas
#ABORT    def num_of_bas(self):
#ABORT        return self.nbas

# atm_id or bas_id start from 0
    def symbol_of_atm(self, atm_id):
        # a molecule can contain different symbols (C1,C2,..) for same type of
        # atoms
        return _symbol(self.atom[atm_id][0])

    def pure_symbol_of_atm(self, atm_id):
        # symbol without index, so (C1,C2,...) just return the same symbol 'C'
        return _symbol(_charge(self.atom[atm_id][0]))

    def charge_of_atm(self, atm_id):
        return self._atm[atm_id][CHARGE_OF]

    def coord_of_atm(self, atm_id):
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

    def coord_of_bas(self, bas_id):
        atm_id = self.atom_of_bas[bas_id] - 1
        ptr = self._atm[atm_id][PTR_COORD]
        return numpy.array(self._env[ptr:ptr+3])

    def nbas_of_atm(self, atm_id):
        symb = self.symbol_of_atm[atm_id]
        return self.basis[symb].__len__()

    def basids_of_atm(self, atm_id):
        return [ib for ib in range(self.nbas) \
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
        ''' total number of primitive GTOs'''
        return reduce(lambda n, b: n + (self.angular_of_bas(b) * 2 + 1) \
                                    * self.nprim_of_bas(b),
                      range(self.nbas), 0)

    def num_NR_function(self):
        return self.num_NR_cgto()
    def num_NR_cgto(self):
        ''' total number of contracted GTOs'''
        return reduce(lambda n, b: n + (self.angular_of_bas(b) * 2 + 1) \
                                    * self.nctr_of_bas(b),
                      range(self.nbas), 0)

    def num_4C_function(self):
        return self.num_4C_cgto()
    def num_4C_cgto(self):
        return self.num_2C_function() * 2

    def num_2C_function(self):
        return self.num_2C_cgto()
    def num_2C_cgto(self):
        ''' total number of spinors'''
        return reduce(lambda n, b: n + self.len_spinor_of_bas(b) \
                                    * self.nctr_of_bas(b),
                      range(self.nbas), 0)

    def time_reversal_spinor(self):
        '''tao = time_reversal_spinor(bas)
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

    def intor(self, intor, dim3=1, symmetric=0):
        '''non-relativitic and relativitic integral generator.
        symmetric=1 : hermitian, symmetric=2 : anti-hermitian'''
        return moleintor.mole_intor(self, intor, dim3, symmetric)

    def intor_symmetric(self, intor, dim3=1):
        '''symmetric integral generator.'''
        return self.intor(intor, dim3, 1)

    def intor_asymmetric(self, intor, dim3=1):
        '''anti-symmetric integral generator.'''
        return self.intor(intor, dim3, 2)

    def intor_cross(self, intor, bras, kets, dim3=1):
        '''bras: shell lists of bras, kets: shell lists of kets'''
        return moleintor.intor_cross(self, intor, bras, kets, dim3)

    def nuclear_repulsion(self):
        e = 0
        for j in range(self.natm):
            q2 = self.charge_of_atm(j)
            r2 = self.coord_of_atm(j)
            for i in range(j):
                q1 = self.charge_of_atm(i)
                r1 = self.coord_of_atm(i)
                r = numpy.linalg.norm(r1-r2)
                e += q1 * q2 / r
        return e

    def labels_of_spheric_GTO(self):
        count = numpy.zeros((self.natm, 9), dtype=int)
        label = []
        i = 0
        for ib in range(self.nbas):
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

#TODO:    def labels_of_spinor_GTO(self):
#TODO:        count = numpy.zeros((self.natm, 9), dtype=int)
#TODO:        label = []
#TODO:        i = 0
#TODO:        for ib in range(self.nbas):
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

    def is_same_mol(self, mol):
        if self.atom.__len__() != mol.atom.__len__():
            return False
        for a1, a2 in zip(self.atom, mol.atom):
            if a1[0] != a2[0] \
               or numpy.linalg.norm(numpy.array(a1[1])-numpy.array(a2[1])) > 2:
                return False
        return True

#TODO:
    def _append_mol(self, mol):
        pass

#TODO    def map_basis_nr2r(self):
#TODO        pass

# concatenate two mol
def env_concatenate(atm1, bas1, env1, atm2, bas2, env2):
    import copy
    catm1 = copy.deepcopy(atm1)
    cbas1 = copy.deepcopy(bas1)
    cenv1 = copy.deepcopy(env1)
    natm1 = atm1.__len__()
    ptr_env1 = env1.__len__()

    catm2 = copy.deepcopy(atm2)
    cbas2 = copy.deepcopy(bas2)

    for a in catm2:
        a[PTR_COORD] += ptr_env1
        a[PTR_MASS]  += ptr_env1
    for b in cbas2:
        b[ATOM_OF] += natm1
        b[PTR_EXP] += ptr_env1
        b[PTR_COEFF] += ptr_env1
    catm1.extend(catm2)
    cbas1.extend(cbas2)
    cenv1.extend(env2)
    return catm1, cbas1, cenv1


MINAO_OCC = (
    (1.,),                  # H
    (2.,),                  # He
    (2.,1),                 # Li
    (2.,2),                 # Be
    (2.,2,1./3,1./3,1./3),  # B
    (2.,2,2./3,2./3,2./3),  # C
    (2.,2,1.  ,1.  ,1.  ),  # N
    (2.,2,4./3,4./3,4./3),  # O
    (2.,2,5./3,5./3,5./3),  # F
    (2.,2,2.  ,2.  ,2.  ),  # Ne
#    s  s s p p p p    p    p
    (2.,2,1,2,2,2),                   # Na
    (2.,2,2,2,2,2),                   # Mg
    (2.,2,2,2,2,2,1./3,1./3,1./3),    # Al
    (2.,2,2,2,2,2,2./3,2./3,2./3),    # Si
    (2.,2,2,2,2,2,1.  ,1.  ,1.  ),    # P
    (2.,2,2,2,2,2,4./3,4./3,4./3),    # S
    (2.,2,2,2,2,2,5./3,5./3,5./3),    # Cl
    (2.,2,2,2,2,2,2.  ,2.  ,2.  ),    # Ar
#    s  s s s p p p p p p
    (None                ),           # K
    (2.,2,2,2,2,2,2,2,2,2),           # Ca
#    s  s s s s p p p p p p p p p d  d  d  d  d
    (2.,2,2,2,0,2,2,2,2,2,2,0,0,0,.2,.2,.2,.2,.2),         # Sc
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,.4,.4,.4,.4,.4),         # Ti
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,.6,.6,.6,.6,.6),         # V
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,.8,.8,.8,.8,.8),         # Cr
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,1.,1.,1.,1.,1.),         # Mn
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,1.2,1.2,1.2,1.2,1.2),    # Fe
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,1.4,1.4,1.4,1.4,1.4),    # Co
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,1.6,1.6,1.6,1.6,1.6),    # Ni
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,1.8,1.8,1.8,1.8,1.8),    # Cu
    (2.,2,2,2,2,2,2,2,2,2,2,0,0,0,2. ,2. ,2. ,2. ,2. ),    # Zn
#    s  s s s p p p p p p p    p    p    d d d d d
    (2.,2,2,2,2,2,2,2,2,2,1./3,1./3,1./3,2,2,2,2,2),       # Ga
    (2.,2,2,2,2,2,2,2,2,2,2./3,2./3,2./3,2,2,2,2,2),       # Ge
    (2.,2,2,2,2,2,2,2,2,2,1.  ,1.  ,1.  ,2,2,2,2,2),       # As
    (2.,2,2,2,2,2,2,2,2,2,4./3,4./3,4./3,2,2,2,2,2),       # Se
    (2.,2,2,2,2,2,2,2,2,2,5./3,5./3,5./3,2,2,2,2,2),       # Br
    (2.,2,2,2,2,2,2,2,2,2,2.  ,2.  ,2.  ,2,2,2,2,2),       # Kr
)
def get_minao_occ(symb):
    nuc = _charge(symb)
    assert(nuc <= 36)
    return MINAO_OCC[nuc-1]
