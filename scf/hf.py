#!/usr/bin/env python
#
# File: hf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Hartree-Fock
'''

import os, sys
import tempfile
import ctypes
import copy
import time

import numpy
import scipy.linalg
import pyscf.gto as gto
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import pycint
from pyscf.lib import _vhf
import diis
import chkfile

alib = os.path.join(os.path.dirname(lib.__file__), 'pycint.so')
_cint = ctypes.cdll.LoadLibrary(alib)


__doc__ = '''Options:
self.chkfile = '/dev/shm/...'
self.stdout = '...'
self.diis_space = 6
self.diis_start_cycle = 3
self.damp_factor = 1
self.level_shift_factor = 0
self.conv_threshold = 1e-10
self.max_cycle = 50
self.init_guess(method)         # method = one of 'atom', '1e', 'chkfile', 'minao'
'''


def dump_orbital_coeff(mol, mo):
    label = mol.spheric_labels()
    fmt = ' %10.5f'*mo.shape[1] + '\n'
    for i, c in enumerate(numpy.array(mo)):
        mol.stdout.write(('%d%3s %s%4s ' % label[i]) + (fmt % tuple(c)))



def get_vj_vk(vhfor, mol, dm):
    return vhfor(dm, mol._atm, mol._bas, mol._env)



def scf_cycle(mol, scf, conv_threshold=1e-10, dump_chk=True, init_dm=None):
    cput0 = (time.clock(), time.time())
    if init_dm is None:
        hf_energy, dm = scf.init_guess_method(mol)
    else:
        hf_energy = 0
        dm = init_dm

    scf_conv = False
    cycle = 0
    diis = scf.init_diis()
    h1e = scf.get_hcore(mol)
    s1e = scf.get_ovlp(mol)

    cput1 = cput0
    vhf = 0
    dm_last = 0
    log.debug(scf, 'start scf_cycle')
    while not scf_conv and cycle < max(1, scf.max_cycle):
        vhf = scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = scf.make_fock(h1e, vhf)
        fock = diis(cycle, s1e, dm, fock)

        dm_last = dm
        last_hf_e = hf_energy
        mo_energy, mo_coeff = scf.eig(fock, s1e)
        mo_occ = scf.set_mo_occ(mo_energy, mo_coeff)
        dm = scf.calc_den_mat(mo_coeff, mo_occ)
        hf_energy = scf.calc_tot_elec_energy(vhf, dm, mo_energy, mo_occ)[0]

        log.info(scf, 'cycle= %d E=%.15g (+nuc=%.5f), delta_E= %g', \
                 cycle+1, hf_energy, hf_energy+mol.nuclear_repulsion(), \
                 hf_energy-last_hf_e)

        if abs((hf_energy-last_hf_e)/hf_energy)*1e2 < conv_threshold \
           and scf.check_dm_converge(dm, dm_last, conv_threshold):
            scf_conv = True

        if dump_chk:
            scf.dump_scf_to_chkfile(hf_energy, mo_energy, mo_occ, mo_coeff)
        cput1 = log.timer(scf, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    # one extra cycle of SCF
    dm = scf.calc_den_mat(mo_coeff, mo_occ)
    vhf = scf.get_veff(mol, dm)
    fock = scf.make_fock(h1e, vhf)
    mo_energy, mo_coeff = scf.eig(fock, s1e)
    mo_occ = scf.set_mo_occ(mo_energy, mo_coeff)
    dm = scf.calc_den_mat(mo_coeff, mo_occ)
    hf_energy = scf.calc_tot_elec_energy(vhf, dm, mo_energy, mo_occ)[0]
    if dump_chk:
        scf.dump_scf_to_chkfile(hf_energy, mo_energy, mo_occ, mo_coeff)
    log.timer(scf, 'scf_cycle', *cput0)

    return scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff

class SCF(object):
    ''' SCF: == RHF '''
    def __init__(self, mol):
        if not mol._built:
            sys.stdout.write('Warning: mol.build() is not called in input\n' )
            mol.build()
        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

        _fd,self.chkfile = tempfile.mkstemp()
        os.close(_fd)
        self.conv_threshold = 1e-10
        self.max_cycle = 50
        self.init_guess_method = self._init_guess_by_minao
        self.diis_space = 8
        self.diis_start_cycle = 3
        self.damp_factor = 0
        self.level_shift_factor = 0
        self.direct_scf = True
        self.direct_scf_threshold = 1e-13
##################################################
# don't modify the following private variables, they are not input options
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.hf_energy = 0
        self.scf_conv = False


    def dump_flags(self):
        log.info(self, '\n')
        log.info(self, '******** SCF flags ********')
        log.info(self, 'method = %s', self.__doc__)#self.__class__)
        log.info(self, 'potential = %s', self.get_veff.__doc__)
        log.info(self, 'initial guess = %s', self.init_guess_method.__doc__)
        log.info(self, 'damping factor = %g', self.damp_factor)
        log.info(self, 'level shift factor = %g', self.level_shift_factor)
        log.info(self, 'DIIS start cycle = %d', self.diis_start_cycle)
        log.info(self, 'DIIS space = %d', self.diis_space)
        log.info(self, 'SCF threshold = %g', self.conv_threshold)
        log.info(self, 'max. SCF cycles = %d', self.max_cycle)
        if self.direct_scf:
            log.info(self, 'direct_scf_threshold = %g', \
                     self.direct_scf_threshold)
        log.info(self, 'chkfile to save SCF result = %s', self.chkfile)


    @classmethod
    def eig(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        return e, c

    def init_diis(self):
        diis_a = diis.SCF_DIIS(self)
        diis_a.diis_space = self.diis_space
        #diis_a.diis_start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                f = diis_a.update(s, d, f)
            if cycle < self.diis_start_cycle-1:
                f = damping(s, d*.5, f, self.damp_factor)
                f = level_shift(s, d*.5, f, self.level_shift_factor)
            else:
                fac = self.level_shift_factor \
                        * numpy.exp(self.diis_start_cycle-cycle-1)
                f = level_shift(s, d*.5, f, fac)
            return f
        return scf_diis

    @lib.omnimethod
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h = mol.intor_symmetric('cint1e_kin_sph') \
                + mol.intor_symmetric('cint1e_nuc_sph')
        return h

    @lib.omnimethod
    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        return mol.intor_symmetric('cint1e_ovlp_sph')

    def make_fock(self, h1e, vhf):
        return h1e + vhf

    def dump_scf_to_chkfile(self, *args):
        chkfile.dump_scf(self.mol, self.chkfile, *args)

    def _init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None:
            mol = self.mol
        try:
            return init_guess_by_minao(self, mol)
        except KeyError:
            log.warn(self, 'Fail in generating initial guess from MINAO. ' \
                     'Use 1e initial guess')
            return self._init_guess_by_1e(mol)

    def _init_guess_by_1e(self, mol=None):
        '''Initial guess from one electron system.'''
        if mol is None:
            mol = self.mol
        log.info(self, '\n')
        log.info(self, 'Initial guess from one electron system.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.set_mo_occ(mo_energy, mo_coeff)
        dm = self.calc_den_mat(mo_coeff, mo_occ)
        return 0, dm

    def _init_guess_by_chkfile(self, mol=None):
        '''Read initial guess from chkfile.'''
        if mol is None:
            mol = self.mol
        try:
            chk_mol, scf_rec = chkfile.read_scf(self.chkfile)
        except IOError:
            log.warn(self, 'Fail in reading from %s. Use MINAO initial guess', \
                     self.chkfile)
            return self._init_guess_by_minao(mol)

        log.info(self, 'Read initial guess from file %s.', self.chkfile)

        chk_mol._atm, chk_mol._bas, chk_mol._env = \
                gto.mole.env_concatenate(chk_mol._atm, chk_mol._bas, \
                                         chk_mol._env, \
                                         mol._atm, mol._bas, mol._env)
        chk_mol.nbas = chk_mol._bas.__len__()
        bras = range(chk_mol.nbas-mol.nbas, chk_mol.nbas)
        kets = range(chk_mol.nbas-mol.nbas)
        ovlp = chk_mol.intor_cross('cint1e_ovlp_sph', bras, kets)
        proj = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp_sph'), ovlp)

        if chk_scf_type(scf_rec['mo_coeff']) == 'NR-UHF':
            # read from UHF results
            mo_coeff = numpy.dot(proj, scf_rec['mo_coeff'][0])
            mo_occ = scf_rec['mo_occ'][0]
        else:
            mo_coeff = numpy.dot(proj, scf_rec['mo_coeff'])
            mo_occ = scf_rec['mo_occ']
        return scf_rec['hf_energy'], numpy.dot(mo_coeff*mo_occ, mo_coeff.T)

    def _init_guess_by_atom(self, mol=None):
        '''Initial guess from occupancy-averaged atomic RHF'''
        if mol is None:
            mol = self.mol
        return init_guess_by_atom(self, mol)

    def init_guess(self, method='minao'):
        return self.set_init_guess(method)
    def set_init_guess(self, method='minao', f_init=None):
        if method.lower() == '1e':
            self.init_guess_method = self._init_guess_by_1e
        elif method.lower() == 'chkfile':
            self.init_guess_method = self._init_guess_by_chkfile
        elif method.lower() == 'minao':
            self.init_guess_method = self._init_guess_by_minao
        elif method.lower() == 'atom':
            self.init_guess_method = self._init_guess_by_atom
        elif f_init is not None:
            self.init_guess_method = f_init
        else:
            raise KeyError('Unknown init guess.')

    # non-relativistic SCF uses new direct_scf module, see _vhf.VHFOpt
    def init_direct_scf(self, mol):
        if self.direct_scf:
            natm = lib.c_int_p(ctypes.c_int(mol._atm.__len__()))
            nbas = lib.c_int_p(ctypes.c_int(mol._bas.__len__()))
            atm = lib.c_int_arr(mol._atm)
            bas = lib.c_int_arr(mol._bas)
            env = lib.c_double_arr(mol._env)
            _cint.init_nr_direct_scf_(atm, natm, bas, nbas, env)
            self.set_direct_scf_threshold(self.direct_scf_threshold)
        else:
            _cint.turnoff_direct_scf_()

    # don't use me
    def del_direct_scf(self):
        _cint.del_nr_direct_scf_()

    # don't use me
    def set_direct_scf_threshold(self, threshold):
        _cint.set_direct_scf_cutoff_(lib.c_double_p(ctypes.c_double(threshold)))

    def set_mo_occ(self, mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = self.mol.nelectron / 2
        mo_occ[:nocc] = 2
        if nocc < mo_occ.size:
            log.info(self, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(self, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    # full density matrix for RHF
    @lib.omnimethod
    def calc_den_mat(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        mo = mo_coeff[:,mo_occ>0]
        return numpy.dot(mo*mo_occ[mo_occ>0], mo.T.conj())

    @lib.omnimethod
    def calc_tot_elec_energy(self, vhf, dm, mo_energy, mo_occ):
        sum_mo_energy = numpy.dot(mo_energy, mo_occ)
        # trace (D*V_HF)
        coul_dup = lib.trace_ab(dm, vhf)
        #log.debug(self, 'E_coul = %.15g', (coul_dup.real * .5))
        e = sum_mo_energy - coul_dup * .5
        return e.real, coul_dup * .5

    def scf_cycle(self, mol, *args, **keys):
        return scf_cycle(mol, self, *args, **keys)

    def check_dm_converge(self, dm, dm_last, conv_threshold):
        delta_dm = abs(dm-dm_last).sum()
        dm_change = delta_dm/abs(dm_last).sum()
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < conv_threshold*1e2

    def scf(self):
        cput0 = (time.clock(), time.time())
        self.dump_flags()

        if self.mol.nelectron == 1:
            self.scf_conv = True
            self.mo_energy, self.mo_coeff = self.solve_1e(self.mol)
            self.mo_occ = numpy.zeros_like(self.mo_energy)
            self.mo_occ[0] = 1
            self.hf_energy = self.mo_energy[0]
        else:
            self.init_direct_scf(self.mol)
            # call self.scf_cycle because dhf redefine scf_cycle
            self.scf_conv, self.hf_energy, \
                    self.mo_energy, self.mo_occ, self.mo_coeff \
                    = self.scf_cycle(self.mol, self.conv_threshold)
            self.del_direct_scf()

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.scf_conv)
        if self.verbose >= param.VERBOSE_INFO:
            self.analyze_scf_result(self.mol, self.mo_energy, self.mo_occ, \
                                    self.mo_coeff)
        return etot

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        '''NR Hartree-Fock Coulomb repulsion'''
        if self.direct_scf:
            vj, vk = get_vj_vk(pycint.nr_vhf_direct_o3, mol, dm-dm_last)
            return vhf_last + vj - vk * .5
        else:
            vj, vk = get_vj_vk(pycint.nr_vhf_o3, mol, dm)
            return vj - vk * .5

    @classmethod
    def solve_1e(self, mol):
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        #log.info(self, '1 electron energy = %.15g.', mo_energy[0])
        return mo_energy, mo_coeff

    def dump_final_energy(self, hf_energy, scf_conv):
        e_nuc = self.mol.nuclear_repulsion()
        if scf_conv:
            log.log(self, 'converged electronic energy = %.15g, nuclear repulsion = %.15g', \
                    hf_energy, e_nuc)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles, nuclear repulsion = %.15g', \
                    hf_energy, self.max_cycle, e_nuc)
        log.log(self, 'total molecular energy = %.15g', \
                hf_energy + e_nuc)
        return hf_energy + e_nuc

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        log.info(self, '**** MO energy ****')
        for i in range(mo_energy.__len__()):
            if self.mo_occ[i] > 0:
                log.info(self, 'occupied MO #%d energy= %.15g occ= %g', \
                         i+1, mo_energy[i], mo_occ[i])
            else:
                log.info(self, 'virtual MO #%d energy= %.15g occ= %g', \
                         i+1, mo_energy[i], mo_occ[i])

    def _is_mem_enough(self):
        nbf = self.mol.num_NR_function()
        return nbf**4/1024**2 < self.max_memory

############

def init_guess_by_atom(dev, mol):
    '''Initial guess from atom calculation.'''
    import atom_hf
    atm_scf = atom_hf.get_atm_nrhf_result(mol)
    nbf = mol.num_NR_function()
    dm = numpy.zeros((nbf, nbf))
    hf_energy = 0
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        if atm_scf.has_key(symb):
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        else:
            symb = mol.pure_symbol_of_atm(ia)
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        p1 = p0 + mo_e.__len__()
        dm[p0:p1,p0:p1] = numpy.dot(mo_c*mo_occ, mo_c.T.conj())
        hf_energy += e_hf
        p0 = p1

    log.info(dev, '\n')
    log.info(dev, 'Initial guess from superpostion of atomic densties.')
    for k,v in atm_scf.items():
        log.debug(dev, 'Atom %s, E = %.12g', k, v[0])
    log.debug(dev, 'total atomic HF energy = %.12g', hf_energy)

    hf_energy -= mol.nuclear_repulsion()
    return hf_energy, dm

def init_guess_by_minao(dev, mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    import atom_hf
    log.info(dev, 'initial guess from MINAO')
    #TODO: log.info(dev, 'initial guess from ANO')
    pmol = mol.copy()
    occ = []
    # minao may not contain the atom. do 1e init guess in such case
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        if symb != 'GHOST':
            basis_add = gto.basis.importbas('minao', gto.mole._rm_digit(symb))
            #basis_add = gto.basis.importbas('ano', gto.mole._rm_digit(symb))
            occ.append(atom_hf.get_minao_occ(symb))
            #TODO: occ.append(gto.mole.get_ano_occ(symb))
            pmol._bas.extend(pmol.make_bas_env_by_atm_id(ia, basis_add))
    pmol.nbas = pmol._bas.__len__()
    occ = numpy.hstack(occ)

    kets = range(mol.nbas, pmol.nbas)
    bras = range(mol.nbas)
    ovlp = pmol.intor_cross('cint1e_ovlp_sph', bras, kets)
    c = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp_sph'), ovlp)
    dm = numpy.dot(c*occ,c.T)
    return 0, dm

# eigenvalue of d is 1
def level_shift(s, d, f, factor):
    if factor < 1e-3:
        return f
    else:
        dm_vir = s - reduce(numpy.dot, (s,d,s))
        return f + dm_vir * factor

# maybe need to save old fock matrix
# check diis.DIISDamping
#def damping(self, s, d, f, mo_coeff, mo_occ):
#    if self.damp_factor < 1e-3:
#        return f
#    else:
#        mo = mo_coeff[:,mo_occ<1e-12]
#        dm_vir = numpy.dot(mo, mo.T.conj())
#        f0 = reduce(numpy.dot, (s, dm_vir, f, d, s))
#        f0 = (f0+f0.T.conj()) * (self.damp_factor/(self.damp_factor+1))
#        return f - f0
def damping(s, d, f, factor):
    if factor < 1e-3:
        return f
    else:
        #dm_vir = s - reduce(numpy.dot, (s,d,s))
        #sinv = numpy.linalg.inv(s)
        #f0 = reduce(numpy.dot, (dm_vir, sinv, f, d, s))
        dm_vir = numpy.eye(s.shape[0])-numpy.dot(s,d)
        f0 = reduce(numpy.dot, (dm_vir, f, d, s))
        f0 = (f0+f0.T.conj()) * (factor/(factor+1.))
        return f - f0



################################################
# for general DM
# hermi = 0 : arbitary
# hermi = 1 : hermitian
# hermi = 2 : anti-hermitian
################################################
def dot_eri_dm(eri, dm, hermi=0):
    if dm.ndim == 2:
        vj, vk = _vhf.vhf_jk_incore(eri, dm, hermi=hermi)
    else:
        vjk = []
        for dmi in dm:
            vjk.append(_vhf.vhf_jk_incore(eri, dmi, hermi=hermi))
        vj = numpy.array([v[0] for v in vjk])
        vk = numpy.array([v[1] for v in vjk])
    return vj, vk

class RHF(SCF):
    '''RHF'''
    def __init__(self, mol):
        if mol.nelectron != 1 and mol.nelectron.__mod__(2) is not 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
        SCF.__init__(self, mol)
        self._eri = None

    def init_direct_scf(self, mol):
        if not self._is_mem_enough() and self.direct_scf:
            self.opt = _vhf.VHFOpt()
            self.opt.init_nr_vhf_direct(mol._atm, mol._bas, mol._env)
            self.opt.direct_scf_threshold = self.direct_scf_threshold

    def release_eri(self):
        self._eri = None

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''NR RHF Coulomb repulsion'''
        t0 = (time.clock(), time.time())
        if self._is_mem_enough():
            if self._eri is None:
                self._eri = _vhf.int2e_sph_8fold(mol._atm, mol._bas, mol._env)
            vj, vk = dot_eri_dm(self._eri, dm, hermi=hermi)
            vhf = vj - vk * .5
        else:
            if self.direct_scf:
                #vj, vk = get_vj_vk(pycint.nr_vhf_direct_o3, mol, dm-dm_last)
                vj, vk = _vhf.vhf_jk_direct(dm-dm_last, mol._atm, \
                                            mol._bas, mol._env, self.opt, \
                                            hermi=hermi)
                vhf = vhf_last + vj - vk * .5
            else:
                vj, vk = _vhf.vhf_jk_direct(dm, mol._atm, mol._bas, mol._env, \
                                            hermi=hermi)
                vhf = vj - vk * .5
        log.timer(self, 'vj and vk', *t0)
        return vhf

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        SCF.analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff)
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, ' ** MO coefficients **')
            nmo = mo_coeff.shape[1]
            for i in range(0, nmo, 5):
                log.debug(self, 'MO_id+1       ' + ('      #%d'*5), \
                          *range(i+1,i+6))
                dump_orbital_coeff(mol, mo_coeff[:,i:i+5])
        dm = self.calc_den_mat(mo_coeff, mo_occ)
        self.mulliken_pop(mol, dm, self.get_ovlp(mol))

    def mulliken_pop(self, mol, dm, s):
        mol_verbose_bak = mol.verbose
        mol.verbose = self.verbose
        res = rhf_mulliken_pop(mol, dm, s)
        mol.verbose = mol_verbose_bak
        return res

    def mulliken_pop_with_meta_lowdin_ao(self, mol, dm_ao):
        mol_verbose_bak = mol.verbose
        mol.verbose = self.verbose
        res = rhf_mulliken_pop_with_meta_lowdin_ao(mol, dm_ao)
        mol.verbose = mol_verbose_bak
        return res


def rhf_mulliken_pop(mol, dm, s):
    '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
    m = dm * s
    pop = numpy.array(map(sum, m))
    label = mol.spheric_labels()

    log.info(mol, ' ** Mulliken pop (on non-orthogonal input basis)  **')
    for i, s in enumerate(label):
        log.info(mol, 'pop of  %s %10.5f', '%d%s %s%4s'%s, pop[i])

    log.info(mol, ' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop[i]
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        nuc = mol.charge_of_atm(ia)
        chg[ia] = nuc - chg[ia]
        log.info(mol, 'charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return pop, chg

def rhf_mulliken_pop_with_meta_lowdin_ao(mol, dm_ao):
    '''divede ao into core, valence and Rydberg sets,
    orthonalizing within each set'''
    import dmet
    c = dmet.hf.pre_orth_ao_atm_scf(mol)
    orth_coeff = dmet.hf.orthogonalize_ao(mol, None, c, 'meta_lowdin')
    c_inv = numpy.linalg.inv(orth_coeff)
    dm = reduce(numpy.dot, (c_inv, dm_ao, c_inv.T.conj()))

    pop = dm.diagonal()
    label = mol.spheric_labels()

    log.info(mol, ' ** Mulliken pop (on meta-lowdin orthogonal AOs)  **')
    for i, s in enumerate(label):
        log.info(mol, 'pop of  %s %10.5f', '%d%s %s%4s'%s, pop[i])

    log.info(mol, ' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop[i]
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        nuc = mol.charge_of_atm(ia)
        chg[ia] = nuc - chg[ia]
        log.info(mol, 'charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return pop, chg



class UHF(SCF):
    __doc__ = 'UHF'
    def __init__(self, mol):
        SCF.__init__(self, mol)
        # self.mo_coeff => [mo_a, mo_b]
        # self.mo_occ => [mo_occ_a, mo_occ_b]
        # self.mo_energy => [mo_energy_a, mo_energy_b]

        #self.ms = 0
        self.nelectron_alpha = (mol.nelectron + 1) / 2
        # fix_nelectron_alpha=0 may lead high spin states
        self.fix_nelectron_alpha = 0
        self.break_symmetry = False
        self._eri = None

    def dump_flags(self):
        SCF.dump_flags(self)
        if self.fix_nelectron_alpha:
            log.info(self, 'number electrons alpha = %d, beta = %d', \
                     self.fix_nelectron_alpha,
                     self.mol.nelectron-self.fix_nelectron_alpha)
        else:
            log.info(self, 'number electrons alpha = %d, beta = %d', \
                     self.nelectron_alpha,
                     self.mol.nelectron-self.nelectron_alpha)

    def eig(self, fock, s):
        e_a, c_a = scipy.linalg.eigh(fock[0], s)
        e_b, c_b = scipy.linalg.eigh(fock[1], s)
        return numpy.array((e_a,e_b)), (c_a,c_b)

    def init_diis(self):
        udiis = diis.SCF_DIIS(self)
        udiis.diis_space = self.diis_space
        #udiis.diis_start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                sdf_a = reduce(numpy.dot, (s, d[0], f[0]))
                sdf_b = reduce(numpy.dot, (s, d[1], f[1]))
                errvec = numpy.hstack((sdf_a.T.conj() - sdf_a, \
                                       sdf_b.T.conj() - sdf_b))
                udiis.err_vec_stack.append(errvec)
                log.debug(self, 'diis-norm(errvec) = %g', \
                          numpy.linalg.norm(errvec))
                if udiis.err_vec_stack.__len__() > udiis.diis_space:
                    udiis.err_vec_stack.pop(0)
                f = diis.DIIS.update(udiis, f)
            if cycle < self.diis_start_cycle-1:
                f = (damping(s, d[0], f[0], self.damp_factor), \
                     damping(s, d[1], f[1], self.damp_factor))
                f = (level_shift(s, d[0], f[0],self.level_shift_factor), \
                     level_shift(s, d[1], f[1],self.level_shift_factor))
            else:
                fac = self.level_shift_factor \
                        * numpy.exp(self.diis_start_cycle-cycle-1)
                f = (level_shift(s, d[0], f[0], fac), \
                     level_shift(s, d[1], f[1], fac))
            return numpy.array(f)
        return scf_diis

    @lib.omnimethod
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        hcore = SCF.get_hcore(mol)
        return numpy.array((hcore,hcore))

    def set_mo_occ(self, mo_energy, mo_coeff=None):
        if self.fix_nelectron_alpha > 0:
            n_a = self.nelectron_alpha = self.fix_nelectron_alpha
            n_b = self.mol.nelectron - n_a
        else:
            ee = sorted([(e,0) for e in mo_energy[0]] \
                        + [(e,1) for e in mo_energy[1]])
            n_a = filter(lambda x: x[1]==0, ee[:self.mol.nelectron]).__len__()
            n_b = self.mol.nelectron - n_a
            if n_a != self.nelectron_alpha:
                log.info(self, 'change num. alpha/beta electrons ' \
                         ' %d / %d -> %d / %d', \
                         self.nelectron_alpha,
                         self.mol.nelectron-self.nelectron_alpha, n_a, n_b)
                self.nelectron_alpha = n_a
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[0][:n_a] = 1
        mo_occ[1][:n_b] = 1
        if n_a < mo_energy[0].size:
            log.info(self, 'alpha nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                     n_a, mo_energy[0][n_a-1], mo_energy[0][n_a])
        else:
            log.info(self, 'alpha nocc = %d, HOMO = %.12g, no LUMO,', \
                     n_a, mo_energy[0][n_a-1])
        log.debug(self, '  mo_energy = %s', mo_energy[0])
        log.info(self, 'beta  nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                 n_b, mo_energy[0][n_b-1], mo_energy[0][n_b])
        log.debug(self, '  mo_energy = %s', mo_energy[1])
        return mo_occ

    @lib.omnimethod
    def calc_den_mat(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        mo_a = mo_coeff[0][:,mo_occ[0]>0]
        mo_b = mo_coeff[1][:,mo_occ[1]>0]
        occ_a = mo_occ[0][mo_occ[0]>0]
        occ_b = mo_occ[1][mo_occ[1]>0]
        dm_a = numpy.dot(mo_a*occ_a, mo_a.T.conj())
        dm_b = numpy.dot(mo_b*occ_b, mo_b.T.conj())
        return numpy.array((dm_a,dm_b))

    @lib.omnimethod
    def calc_tot_elec_energy(self, vhf, dm, mo_energy, mo_occ):
        sum_mo_energy = numpy.dot(mo_energy[0], mo_occ[0]) \
                + numpy.dot(mo_energy[1], mo_occ[1])
        # trace (D*V_HF)
        coul_dup = lib.trace_ab(dm[0], vhf[0]) \
                + lib.trace_ab(dm[1], vhf[1])
        e = sum_mo_energy - coul_dup * .5
        return e.real, coul_dup * .5

    def break_spin_sym(self, mol, mo_coeff):
        # break symmetry between alpha and beta
        nocc = mol.nelectron / 2
        if self.break_symmetry == 1: # break spatial symmetry
            nmo = mo_coeff[0].shape[1]
            nvir = nmo - nocc
            if nvir < 5:
                for i in range(nocc-1,nmo):
                    mo_coeff[1][:,nocc-1] += mo_coeff[0][:,i]
                mo_coeff[1][:,nocc-1] *= 1./(nvir+1)
            else:
                for i in range(nocc-1,nocc+5):
                    mo_coeff[1][:,nocc-1] += mo_coeff[0][:,i]
                mo_coeff[1][:,nocc-1] *= 1./2
        elif self.break_symmetry == 2:
            mo_coeff[1][:,nocc-1] = mo_coeff[0][:,nocc]
        else:
            if nocc == 1:
                mo_coeff[1][:,:nocc] = 0
            else:
                mo_coeff[1][:,nocc-1] = 0
        return mo_coeff

    def _init_guess_by_1e(self, mol=None):
        '''Initial guess from one electron system.'''
        if mol is None:
            mol = self.mol
        log.info(self, '\n')
        log.info(self, 'Initial guess from one electron system.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_coeff = self.break_spin_sym(mol, mo_coeff)

        mo_occ = self.set_mo_occ(mo_energy, mo_coeff)
        dm = self.calc_den_mat(mo_coeff, mo_occ)
        return 0, dm

    def _init_guess_by_minao(self, mol=None):
        if mol is None:
            mol = self.mol
        log.info(self, 'initial guess from MINAO')
        try:
            return 0, init_minao_uhf_dm(mol)
        except:
            log.warn(self, 'Fail in generating initial guess from MINAO.' \
                     'Use 1e initial guess')
            return self._init_guess_by_1e(mol)

    def _init_guess_by_atom(self, mol=None):
        if mol is None:
            mol = self.mol
        e, dm = init_guess_by_atom(self, mol)
        return e, numpy.array((dm*.5, dm*.5))

    def _init_guess_by_chkfile(self, mol=None):
        '''Read initial guess from chkfile.'''
        if mol is None:
            mol = self.mol
        try:
            chk_mol, scf_rec = chkfile.read_scf(self.chkfile)
        except IOError:
            log.warn(self, 'Fail in reading from %s. Use MINAO initial guess', \
                     self.chkfile)
            return self._init_guess_by_minao(mol)

        log.info(self, 'Read initial guess from file %s.', self.chkfile)

        chk_mol._atm, chk_mol._bas, chk_mol._env = \
                gto.mole.env_concatenate(chk_mol._atm, chk_mol._bas, \
                                         chk_mol._env, \
                                         mol._atm, mol._bas, mol._env)
        chk_mol.nbas = chk_mol._bas.__len__()
        bras = range(chk_mol.nbas-mol.nbas, chk_mol.nbas)
        kets = range(chk_mol.nbas-mol.nbas)
        ovlp = chk_mol.intor_cross('cint1e_ovlp_sph', bras, kets)
        proj = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp_sph'), ovlp)

        mo_coeff = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        if chk_scf_type(scf_rec['mo_coeff']) == 'NR-RHF':
            # read from RHF results
            mo_coeff = self.break_spin_sym(mol,[mo_coeff,numpy.copy(mo_coeff)])
            mo_occ = (mo_occ,mo_occ)
        mo_coeff = (numpy.dot(proj, mo_coeff[0]), \
                    numpy.dot(proj, mo_coeff[1]))
        dm = (numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].T),
              numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].T))
        return scf_rec['hf_energy'], numpy.array(dm)

    def init_direct_scf(self, mol):
        if not self._is_mem_enough() and self.direct_scf:
            self.opt = _vhf.VHFOpt()
            self.opt.init_nr_vhf_direct(mol._atm, mol._bas, mol._env)
            self.opt.direct_scf_threshold = self.direct_scf_threshold

    def release_eri(self):
        self._eri = None

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''NR UHF Coulomb repulsion'''
        t0 = (time.clock(), time.time())
        if self._is_mem_enough():
            if self._eri is None:
                self._eri = _vhf.int2e_sph_8fold(mol._atm, mol._bas, mol._env)
            vj0, vk0 = dot_eri_dm(self._eri, dm[0], hermi=hermi)
            vj1, vk1 = dot_eri_dm(self._eri, dm[1], hermi=hermi)
            v_a = vj0 + vj1 - vk0
            v_b = vj0 + vj1 - vk1
            vhf = numpy.array((v_a,v_b))
        elif self.direct_scf:
            #vj, vk = get_vj_vk(pycint.nr_vhf_direct_o3, mol, dm-dm_last)
            vj, vk = _vhf.vhf_jk_direct(dm-dm_last, mol._atm, mol._bas, \
                                        mol._env, self.opt, hermi=hermi)
            v_a = vj[0] + vj[1] - vk[0]
            v_b = vj[0] + vj[1] - vk[1]
            vhf = vhf_last + numpy.array((v_a,v_b))
        else:
            #vj, vk = get_vj_vk(pycint.nr_vhf_o3, mol, dm)
            vj, vk = _vhf.vhf_jk_direct(dm, mol._atm, mol._bas, mol._env, \
                                        hermi=hermi)
            v_a = vj[0] + vj[1] - vk[0]
            v_b = vj[0] + vj[1] - vk[1]
            vhf = numpy.array((v_a,v_b))
        log.timer(self, 'vj and vk', *t0)
        return vhf

    def scf(self):
        cput0 = (time.clock(), time.time())
        self.dump_flags()

        if self.mol.nelectron == 1:
            self.scf_conv = True
            h1e = self.get_hcore(self.mol)
            s1e = self.get_ovlp(self.mol)
            self.mo_energy, self.mo_coeff = SCF.eig(self, h1e[0], s1e)
            self.mo_occ = numpy.zeros_like(self.mo_energy)
            self.mo_occ[0] = 1
            self.hf_energy = self.mo_energy[0]
            log.info(self, '1 electron energy = %.15g.', mo_energy[0])
        else:
            self.init_direct_scf(self.mol)
            self.scf_conv, self.hf_energy, \
                    self.mo_energy, self.mo_occ, self.mo_coeff \
                    = self.scf_cycle(self.mol, self.conv_threshold)
            self.del_direct_scf()
            if self.nelectron_alpha * 2 < self.mol.nelectron:
                self.mo_coeff = (self.mo_coeff[1], self.mo_coeff[0])
                self.mo_occ = (self.mo_occ[1], self.mo_occ[0])
                self.mo_energy = (self.mo_energy[1], self.mo_energy[0])

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.scf_conv)
        if self.verbose >= param.VERBOSE_INFO:
            self.analyze_scf_result(self.mol, self.mo_energy, self.mo_occ, \
                                    self.mo_coeff)
        return etot

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        ss, s = spin_square(mol, mo_coeff[0][:,mo_occ[0]>0], \
                                 mo_coeff[1][:,mo_occ[1]>0])
        log.info(self, 'multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)

        log.info(self, '**** MO energy ****')
        for i in range(mo_energy[0].__len__()):
            if mo_occ[0][i] > 0:
                log.info(self, "alpha occupied MO #%d energy = %.15g occ= %g", \
                         i+1, mo_energy[0][i], mo_occ[0][i])
            else:
                log.info(self, "alpha virtual MO #%d energy = %.15g occ= %g", \
                         i+1, mo_energy[0][i], mo_occ[0][i])
        for i in range(mo_energy[1].__len__()):
            if mo_occ[1][i] > 0:
                log.info(self, "beta occupied MO #%d energy = %.15g occ= %g", \
                         i+1, mo_energy[1][i], mo_occ[1][i])
            else:
                log.info(self, "beta virtual MO #%d energy = %.15g occ= %g", \
                         i+1, mo_energy[1][i], mo_occ[1][i])
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, ' ** MO coefficients for alpha spin **')
            nmo = mo_energy.shape[1]
            for i in range(0, nmo, 5):
                log.debug(self, 'MO_id+1       ' + ('      #%d'*5), \
                          *range(i+1,i+6))
                dump_orbital_coeff(mol, mo_coeff[0][:,i:i+5])
            log.debug(self, ' ** MO coefficients for beta spin **')
            for i in range(0, nmo, 5):
                log.debug(self, 'MO_id+1       ' + ('      #%d'*5), \
                          *range(i+1,i+6))
                dump_orbital_coeff(mol, mo_coeff[1][:,i:i+5])

        dm = self.calc_den_mat(mo_coeff, mo_occ)
        self.mulliken_pop(mol, dm, self.get_ovlp(mol))

    def mulliken_pop(self, mol, dm, s):
        mol_verbose_bak = mol.verbose
        mol.verbose = self.verbose
        res = uhf_mulliken_pop(mol, dm, s)
        mol.verbose = mol_verbose_bak
        return res

    def mulliken_pop_with_meta_lowdin_ao(self, mol, dm_ao):
        mol_verbose_bak = mol.verbose
        mol.verbose = self.verbose
        res = uhf_mulliken_pop_with_meta_lowdin_ao(mol, dm_ao)
        mol.verbose = mol_verbose_bak
        return res



def chk_scf_type(mo_coeff):
    if mo_coeff[0].ndim == 2:
        return 'NR-UHF'
    elif isinstance(mo_coeff[0,0], complex):
        return 'R-DHF'
    else:
        return 'NR-RHF'

def init_minao_uhf_dm(mol):
    import atom_hf
    def filter_alpha(x):
        if x > 1:
            return 1
        else:
            return x
    def filter_beta(x):
        if x > 1:
            return x-1
        else:
            return 0
    pmol = mol.copy()
    occa = []
    occb = []
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        occ0 = atom_hf.get_minao_occ(symb)
        minao = gto.basis.minao[gto.mole._rm_digit(symb)]
        occa.append(map(filter_alpha,occ0))
        occb.append(map(filter_beta ,occ0))
        pmol._bas.extend(pmol.make_bas_env_by_atm_id(ia, minao))
    pmol.nbas = pmol._bas.__len__()

    kets = range(mol.nbas, pmol.nbas)
    bras = range(mol.nbas)
    ovlp = pmol.intor_cross('cint1e_ovlp_sph', bras, kets)
    c = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp_sph'), \
                           ovlp)
    dm = (numpy.dot(c*numpy.hstack(occa),c.T), \
          numpy.dot(c*numpy.hstack(occb),c.T))
    return numpy.array(dm)


def map_rhf_to_uhf(mol, rhf):
    assert(isinstance(rhf, RHF))
    uhf = UHF(mol)
    uhf.verbose               = rhf.verbose
    uhf.mo_energy             = numpy.array((rhf.mo_energy,rhf.mo_energy))
    uhf.mo_coeff              = numpy.array((rhf.mo_coeff,rhf.mo_coeff))
    uhf.mo_occ                = numpy.array((rhf.mo_occ,rhf.mo_occ))
    uhf.hf_energy             = rhf.hf_energy
    uhf.diis_space            = rhf.diis_space
    uhf.diis_start_cycle      = rhf.diis_start_cycle
    uhf.damp_factor           = rhf.damp_factor
    uhf.level_shift_factor    = rhf.level_shift_factor
    uhf.scf_conv              = rhf.scf_conv
    uhf.direct_scf            = rhf.direct_scf
    uhf.direct_scf_threshold  = rhf.direct_scf_threshold

    uhf.chkfile               = rhf.chkfile
    uhf.stdout                = rhf.stdout
    uhf.conv_threshold         = rhf.conv_threshold
    uhf.max_cycle         = rhf.max_cycle
    return uhf

def spin_square(mol, occ_mo_a, occ_mo_b):
    # S^2 = S+ * S- + S- * S+ + Sz * Sz
    # S+ = \sum_i S_i+ ~ effective for all beta occupied orbitals
    # S- = \sum_i S_i- ~ effective for all alpha occupied orbitals
    # S+ * S- ~ sum of nocc_a * nocc_b couplings
    # Sz = Msz^2
    nocc_a = occ_mo_a.shape[1]
    nocc_b = occ_mo_b.shape[1]
    ovlp = mol.intor_symmetric('cint1e_ovlp_sph')
    s = reduce(numpy.dot, (occ_mo_a.T, ovlp, occ_mo_b))
    ssx = ssy = (nocc_a+nocc_b)*.25 - 2*(s**2).sum()*.25
    ssz = (nocc_b-nocc_a)**2 * .25
    ss = ssx + ssy + ssz
    #log.debug(mol, "s_x^2 = %.9g, s_y^2 = %.9g, s_z^2 = %.9g" % (ssx,ssy,ssz))
    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

def uhf_mulliken_pop(mol, dm, s):
    '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
    m_a = dm[0] * s
    m_b = dm[1] * s
    pop_a = numpy.array(map(sum, m_a))
    pop_b = numpy.array(map(sum, m_b))
    label = mol.spheric_labels()

    log.info(mol, ' ** Mulliken pop alpha/beta (on non-orthogonal basis) **')
    for i, s in enumerate(label):
        log.info(mol, 'pop of  %s %10.5f  / %10.5f', \
                 '%d%s %s%4s'%s, pop_a[i], pop_b[i])

    log.info(mol, ' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop_a[i] + pop_b[i]
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        nuc = mol.charge_of_atm(ia)
        chg[ia] = nuc - chg[ia]
        log.info(mol, 'charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return (pop_a,pop_b), chg

def uhf_mulliken_pop_with_meta_lowdin_ao(mol, dm_ao):
    import dmet
    c = dmet.hf.pre_orth_ao_atm_scf(mol)
    orth_coeff = dmet.hf.orthogonalize_ao(mol, None, c, 'meta_lowdin')
    c_inv = numpy.linalg.inv(orth_coeff)
    dm_a = reduce(numpy.dot, (c_inv, dm_ao[0], c_inv.T.conj()))
    dm_b = reduce(numpy.dot, (c_inv, dm_ao[1], c_inv.T.conj()))

    pop_a = dm_a.diagonal()
    pop_b = dm_b.diagonal()
    label = mol.spheric_labels()

    log.info(mol, ' ** Mulliken pop alpha/beta (on meta-lowdin orthogonal AOs) **')
    for i, s in enumerate(label):
        log.info(mol, 'pop of  %s %10.5f  / %10.5f', \
                 '%d%s %s%4s'%s, pop_a[i], pop_b[i])

    log.info(mol, ' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop_a[i] + pop_b[i]
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        nuc = mol.charge_of_atm(ia)
        chg[ia] = nuc - chg[ia]
        log.info(mol, 'charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return (pop_a,pop_b), chg


#TODO:class ROHF(SCF):
#TODO:    __doc__ = 'ROHF'
#TODO:    def __init__(self, mol):
#TODO:        SCF.__init__(self, mol)
#TODO:        self.nelectron_alpha = (mol.nelectron + 1) / 2
#TODO:
#TODO:    def get_veff(self, mol, dm):
#TODO:        '''NR HF Coulomb repulsion'''
#TODO:        vj, vk = get_vj_vk(pycint.nr_vhf_o3, mol, dm)
#TODO:        v_1 = vj[0] + vj[1] - vk[0]
#TODO:        v_2 = vj[0] + vj[1] - vk[1]
#TODO:        return (v_1, v_2)


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = 'out_hf'

    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {
        'He': 'ccpvdz'}
    mol.build()

##############
# SCF result
    method = RHF(mol)
    method.init_guess('1e')
    energy = method.scf()
    print energy
