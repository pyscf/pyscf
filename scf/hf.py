#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Hartree-Fock
'''

import os, sys
import tempfile
import ctypes
import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.gto
import pyscf.lib
import pyscf.lib.logger as log
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf import diis
from pyscf.scf import _vhf


'''Options:
self.chkfile = '/dev/shm/...'
self.stdout = '...'
self.diis_start_cycle = 3       # to switch off DIIS, set it to large number
                                # set self.DIIS to None
self.diis.space = 8
self.damp_factor = 1
self.level_shift_factor = 0
self.conv_tol = 1e-10
self.max_cycle = 50
self.init_guess = method        # method = one of 'atom', '1e', 'chkfile'
'''


def kernel(mf, conv_tol=1e-10, dump_chk=True, init_dm=None):
    cput0 = (time.clock(), time.time())
    mol = mf.mol
    if init_dm is None:
        dm = mf.get_init_guess(key=mf.init_guess)
    else:
        dm = init_dm

    if dump_chk:
        # dump mol after reading initialized DM
        chkfile.dump(mf.chkfile, 'mol', format(mol.pack()))

    scf_conv = False
    cycle = 0
    h1e = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)
    try:
        adiis = mf.DIIS(mf)
        adiis.space = mf.diis_space
    except:
        adiis = None

    cput1 = cput0
    hf_energy = 0
    vhf = 0
    dm_last = 0
    log.debug(mf, 'start scf_cycle')
    while not scf_conv and cycle < max(1, mf.max_cycle):
        vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, adiis)

        dm_last = dm
        last_hf_e = hf_energy
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        hf_energy = mf.energy_tot(dm, h1e, vhf)

        log.info(mf, 'cycle= %d E=%.15g, delta_E= %g', \
                 cycle+1, hf_energy, hf_energy-last_hf_e)

        if abs((hf_energy-last_hf_e)/hf_energy)*1e2 < conv_tol \
           and mf.check_dm_conv(dm, dm_last, conv_tol):
            scf_conv = True

        if dump_chk:
            mf.dump_chk(hf_energy, mo_energy, mo_occ, mo_coeff)
        cput1 = log.timer(mf, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    # one extra cycle of SCF
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, None)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    hf_energy = mf.energy_tot(dm, h1e, vhf)
    if dump_chk:
        mf.dump_chk(hf_energy, mo_energy, mo_occ, mo_coeff)
    log.timer(mf, 'scf_cycle', *cput0)

    return scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff



def get_hcore(mol):
    h = mol.intor_symmetric('cint1e_kin_sph') \
      + mol.intor_symmetric('cint1e_nuc_sph')
    return h

def get_ovlp(mol):
    return mol.intor_symmetric('cint1e_ovlp_sph')

def init_guess_by_minao(mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    from pyscf.scf import atom_hf
    from pyscf.scf import addons
    log.info(mol, 'initial guess from MINAO')

    def minao_basis(symb):
        basis_add = pyscf.gto.basis.load('ano', symb)
        occ = []
        basis_new = []
        for l in range(4):
            ndocc, nfrac = atom_hf.frac_occ(symb, l)
            if ndocc > 0:
                occ.extend([2]*ndocc*(2*l+1))
            if nfrac > 1e-15:
                occ.extend([nfrac]*(2*l+1))
                ndocc += 1
            if ndocc > 0:
                basis_new.append([l] + [b[:ndocc+1] for b in basis_add[l][1:]])
        return occ, basis_new

    atmlst = set([pyscf.gto.mole._rm_digit(pyscf.gto.mole._symbol(k)) \
                  for k in mol.basis.keys()])
    basis = {}
    occdic = {}
    for symb in atmlst:
        occ_add, basis_add = minao_basis(symb)
        occdic[symb] = occ_add
        basis[symb] = basis_add
    occ = []
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        occ.append(occdic[symb])
    occ = numpy.hstack(occ)

    pmol = pyscf.gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(mol.atom, basis, [])
    c = addons.project_mo_nr2nr(pmol, 1, mol)

    dm = numpy.dot(c*occ,c.T)
    return dm

def init_guess_by_1e(mol):
    '''Initial guess from one electron system.'''
    mf = RHF(mol)
    return mf.init_guess_by_1e(mol)

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    from pyscf.scf import atom_hf
    atm_scf = atom_hf.get_atm_nrhf(mol)
    nbf = mol.nao_nr()
    dm = numpy.zeros((nbf, nbf))
    hf_energy = 0
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in atm_scf:
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        else:
            symb = mol.atom_pure_symbol(ia)
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        p1 = p0 + mo_e.__len__()
        dm[p0:p1,p0:p1] = numpy.dot(mo_c*mo_occ, mo_c.T.conj())
        hf_energy += e_hf
        p0 = p1

    log.info(mol, '\n')
    log.info(mol, 'Initial guess from superpostion of atomic densties.')
    for k,v in atm_scf.items():
        log.debug(mol, 'Atom %s, E = %.12g', k, v[0])
    log.debug(mol, 'total atomic SCF energy = %.12g', hf_energy)
    return dm

def init_guess_by_chkfile(mol, chkfile_name, project=True):
    from pyscf.scf import addons
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)

    def fproj(mo):
        if project:
            return addons.project_mo_nr2nr(chk_mol, mo, mol)
        else:
            return mo
    if scf_rec['mo_coeff'].ndim == 2:
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        if numpy.iscomplexobj(mo):
            raise RuntimeError('TODO: project DHF orbital to RHF orbital')
        dm = make_rdm1(fproj(mo), mo_occ)
    else: #UHF
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = make_rdm1(fproj(mo[0]), mo_occ[0]) \
           + make_rdm1(fproj(mo[1]), mo_occ[1])
    return dm

def get_init_guess(mol, key='minao'):
    if callable(key):
        return key(mol)
    elif key.lower() == '1e':
        return init_guess_by_1e(mol)
    elif key.lower() == 'atom':
        return init_guess_by_atom(mol)
    elif key.lower() == 'chkfile':
        raise RuntimeError('Call pyscf.scf.hf.init_guess_by_chkfile instead')
    else:
        return init_guess_by_minao(mol)

# eigenvalue of d is 1
def level_shift(s, d, f, factor):
    if factor < 1e-3:
        return f
    else:
        dm_vir = s - reduce(numpy.dot, (s,d,s))
        return f + dm_vir * factor

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

# full density matrix for RHF
def make_rdm1(mo_coeff, mo_occ):
    mocc = numpy.einsum('ij,j->ij', mo_coeff, mo_occ)
    return numpy.dot(mocc, mo_coeff.T.conj())

def energy_elec(mf, dm, h1e=None, vhf=None):
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
    e_coul = numpy.einsum('ji,ji', vhf.conj(), dm).real * .5
    log.debug1(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul

def energy_tot(mf, dm, h1e=None, vhf=None):
    return energy_elec(mf, h1e, vhf, dm)[0] + mf.mol.energy_nuc()

################################################
# for general DM
# hermi = 0 : arbitary
# hermi = 1 : hermitian
# hermi = 2 : anti-hermitian
################################################
def dot_eri_dm(eri, dm, hermi=0):
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        vj, vk = _vhf.incore(eri, dm, hermi=hermi)
    else:
        vjk = [_vhf.incore(eri, dmi, hermi=hermi) for dmi in dm]
        vj = numpy.array([v[0] for v in vjk])
        vk = numpy.array([v[1] for v in vjk])
    return vj, vk

def get_jk(mol, dm, hermi=1, vhfopt=None):
    vj, vk = _vhf.direct(numpy.array(dm, copy=False),
                         mol._atm, mol._bas, mol._env,
                         vhfopt=vhfopt, hermi=hermi)
    return vj, vk

def get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None):
    '''NR Hartree-Fock Coulomb repulsion'''
    ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
    vj, vk = get_jk(mol, ddm, hermi=hermi, vhfopt=vhfopt)
    return numpy.array(vhf_last, copy=False) + vj - vk * .5

def analyze(mf, verbose=logger.DEBUG):
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    log = logger.Logger(mf.stdout, verbose)
    log.info('**** MO energy ****')
    for i in range(len(mo_energy)):
        if mo_occ[i] > 0:
            log.info('occupied MO #%d energy= %.15g occ= %g', \
                     i+1, mo_energy[i], mo_occ[i])
        else:
            log.info('virtual MO #%d energy= %.15g occ= %g', \
                     i+1, mo_energy[i], mo_occ[i])
    if verbose >= logger.DEBUG:
        log.debug(' ** MO coefficients **')
        label = ['%d%3s %s%-4s' % x for x in mf.mol.spheric_labels()]
        dump_mat.dump_rec(mf.stdout, mo_coeff, label, start=1)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_pop(mf.mol, dm, mf.get_ovlp(), verbose)

def mulliken_pop(mol, dm, ovlp=None, verbose=logger.DEBUG):
    '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
    if ovlp is None:
        ovlp = get_ovlp(mol)
    log = logger.Logger(mol.stdout, verbose)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        pop = numpy.einsum('ij->i', dm*ovlp).real
    else: # ROHF
        pop = numpy.einsum('ij->i', (dm[0]+dm[1])*ovlp).real
    label = mol.spheric_labels()

    log.info(' ** Mulliken pop  **')
    for i, s in enumerate(label):
        log.info('pop of  %s %10.5f', '%d%s %s%4s'%s, pop[i])

    log.info(' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop[i]
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        nuc = mol.atom_charge(ia)
        chg[ia] = nuc - chg[ia]
        log.info('charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return pop, chg

def mulliken_pop_meta_lowdin_ao(mol, dm, verbose=logger.DEBUG):
    '''divede ao into core, valence and Rydberg sets,
    orthonalizing within each set'''
    from pyscf.lo import orth
    log = logger.Logger(mol.stdout, verbose)
    c = orth.pre_orth_ao_atm_scf(mol)
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c)
    c_inv = numpy.linalg.inv(orth_coeff)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = reduce(numpy.dot, (c_inv, dm, c_inv.T.conj()))
    else: # ROHF
        dm = reduce(numpy.dot, (c_inv, dm[0]+dm[1], c_inv.T.conj()))

    log.info(' ** Mulliken pop on meta-lowdin orthogonal AOs  **')
    return mulliken_pop(mol, dm, numpy.eye(orth_coeff.shape[0]), verbose)


class SCF(object):
    ''' SCF: == RHF '''
    def __init__(self, mol):
        if not mol._built:
            sys.stderr.write('Warning: mol.build() is not called in input\n' )
            mol.build()
        self.mol = mol
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.stdout = mol.stdout

# the chkfile will be removed automatically, to save the chkfile, assign a
# filename to self.chkfile
        self._chkfile = tempfile.NamedTemporaryFile()
        self.chkfile = self._chkfile.name
        self.conv_tol = 1e-10
        self.max_cycle = 50
        self.init_guess = 'minao'
        self.DIIS = diis.SCF_DIIS
        self.diis_space = 8
        self.diis_start_cycle = 3
        self.damp_factor = 0
        self.level_shift_factor = 0
        self.direct_scf = True
        self.direct_scf_tol = 1e-13
##################################################
# don't modify the following private variables, they are not input options
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.hf_energy = 0
        self.converged = False

        self.opt = None

        self._keys = set(self.__dict__.keys()).union(['_keys'])

    def build(self, mol=None):
        return self.build_(mol)
    def build_(self, mol=None):
        if mol is None:
            mol = self.mol
        mol.check_sanity(self)

        if not self._is_mem_enough() and self.direct_scf:
            self.opt = _vhf.VHFOpt(mol, 'cint2e_sph', 'CVHFnrs8_prescreen',
                                   'CVHFsetnr_direct_scf',
                                   'CVHFsetnr_direct_scf_dm')
            self.opt.direct_scf_tol = self.direct_scf_tol

    def dump_flags(self):
        log.info(self, '\n')
        log.info(self, '******** SCF flags ********')
        log.info(self, 'method = %s', self.__doc__)
        log.info(self, 'potential = %s', self.get_veff.__doc__)
        log.info(self, 'initial guess = %s', self.init_guess)
        log.info(self, 'damping factor = %g', self.damp_factor)
        log.info(self, 'level shift factor = %g', self.level_shift_factor)
        log.info(self, 'DIIS start cycle = %d', self.diis_start_cycle)
        log.info(self, 'DIIS space = %d', self.diis_space)
        log.info(self, 'SCF tol = %g', self.conv_tol)
        log.info(self, 'max. SCF cycles = %d', self.max_cycle)
        log.info(self, 'direct_scf = %s', self.direct_scf)
        if self.direct_scf:
            log.info(self, 'direct_scf_tol = %g', \
                     self.direct_scf_tol)
        if self.chkfile:
            log.info(self, 'chkfile to save SCF result = %s', self.chkfile)


    def eig(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        return e, c

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol)

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
        f = h1e + vhf
        if 0 <= cycle < self.diis_start_cycle-1:
            f = damping(s1e, dm*.5, f, self.damp_factor)
            f = level_shift(s1e, dm*.5, f, self.level_shift_factor)
        elif 0 <= cycle:
            # decay the level_shift_factor
            fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
            f = level_shift(s1e, dm*.5, f, fac)
        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dm, f)
        return f

    def dump_chk(self, *args):
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, *args)

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None:
            mol = self.mol
        return _hack_mol_log(mol, self, init_guess_by_minao)

    def init_guess_by_atom(self, mol=None):
        if mol is None:
            mol = self.mol
        return _hack_mol_log(mol, self, init_guess_by_atom)

    def init_guess_by_1e(self, mol=None):
        '''Initial guess from one electron system.'''
        if mol is None:
            mol = self.mol
        log.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    def init_guess_by_chkfile(self, mol=None, chkfile=None, project=True):
        if mol is None:
            mol = self.mol
        if chkfile is None:
            chkfile = self.chkfile
        return _hack_mol_log(mol, self, init_guess_by_chkfile, chkfile,
                             project=project)

    def get_init_guess(self, mol=None, key='minao'):
        if callable(key):
            return key(mol)
        elif key.lower() == '1e':
            return self.init_guess_by_1e(mol)
        elif key.lower() == 'atom':
            return self.init_guess_by_atom(mol)
        elif key.lower() == 'chkfile':
            try:
                return self.init_guess_by_chkfile(mol)
            except:
                log.warn(self, 'Fail in reading %s. Use MINAO initial guess', \
                         self.chkfile)
                return self.init_guess_by_minao(mol)
        else:
            return self.init_guess_by_minao(mol)

    def get_occ(self, mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = self.mol.nelectron // 2
        mo_occ[:nocc] = 2
        if nocc < mo_occ.size:
            log.info(self, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(self, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    # full density matrix for RHF
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ)

    def energy_elec(self, dm, h1e=None, vhf=None):
        return energy_elec(self, dm, h1e, vhf)

    def energy_tot(self, dm, h1e=None, vhf=None):
        return self.energy_elec(dm, h1e, vhf)[0] + self.mol.energy_nuc()

    def check_dm_conv(self, dm, dm_last, conv_tol):
        dm = numpy.array(dm)
        dm_last = numpy.array(dm_last)
        delta_dm = abs(dm-dm_last).sum()
        dm_change = delta_dm/abs(dm_last).sum()
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < conv_tol*1e2

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())

        self.build()
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = kernel(self, self.conv_tol, init_dm=dm0)

        log.timer(self, 'SCF', *cput0)
        self.dump_energy(self.hf_energy, self.converged)
        if self.verbose >= logger.INFO:
            self.analyze(self.verbose)
        return self.hf_energy

    def get_jk(self, mol, dm, hermi=1):
        t0 = (time.clock(), time.time())
        vj, vk = get_jk(mol, dm, hermi, self.opt)
        log.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''NR Hartree-Fock Coulomb repulsion'''
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last, copy=False) + vj - vk * .5
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5

    def dump_energy(self, hf_energy, converged):
        if converged:
            log.log(self, 'converged SCF energy = %.15g',
                    hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'SCF energy = %.15g after %d cycles',
                    hf_energy, self.max_cycle)

    def analyze(self, verbose=logger.DEBUG):
        return analyze(self, verbose)

    def mulliken_pop(self, mol, dm, ovlp=None, verbose=logger.DEBUG):
        if ovlp is None:
            ovlp = self.get_ovlp(mol)
        return _hack_mol_log(mol, self, mulliken_pop, dm, ovlp, verbose)

    def mulliken_pop_meta_lowdin_ao(self, mol, dm, verbose=logger.DEBUG):
        return _hack_mol_log(mol, self, mulliken_pop_meta_lowdin_ao, dm, verbose)

    def _is_mem_enough(self):
        nbf = self.mol.nao_nr()
        return nbf**4/1e6 < self.max_memory


############

class HF1e(SCF):
    def scf(self, *args):
        log.info(self, '\n')
        log.info(self, '******** 1 electron system ********')
        self.converged = True
        h1e = self.get_hcore(self.mol)
        s1e = self.get_ovlp(self.mol)
        self.mo_energy, self.mo_coeff = self.eig(h1e, s1e)
        self.mo_occ = numpy.zeros_like(self.mo_energy)
        self.mo_occ[0] = 1
        self.hf_energy = self.mo_energy[0] + self.mol.energy_nuc()
        return self.hf_energy


class RHF(SCF):
    '''RHF'''
    def __init__(self, mol):
        if mol.nelectron != 1 and mol.nelectron.__mod__(2) is not 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
# Note: self._eri requires large mount of memory
        SCF.__init__(self, mol)
        self._eri = None
        self._keys = self._keys.union(['_eri'])

    def get_jk(self, mol, dm, hermi=1):
        t0 = (time.clock(), time.time())
        if self._is_mem_enough() or self._eri is not None:
            if self._eri is None:
                self._eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
            vj, vk = dot_eri_dm(self._eri, dm, hermi)
        else:
            vj, vk = get_jk(mol, dm, hermi, self.opt)
        log.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''NR RHF Coulomb repulsion'''
        if self._is_mem_enough() or self._eri is not None:
            vj, vk = self.get_jk(mol, dm, hermi)
            return vj - vk * .5
        else:
            return SCF.get_veff(self, mol, dm, dm_last, vhf_last, hermi)


# use UHF init_guess, get_veff, diis, and intermediates such as fock, vhf, dm
# keep mo_occ, mo_energy, mo_coeff as RHF structure
class ROHF(RHF):
    '''ROHF'''
    def __init__(self, mol):
        SCF.__init__(self, mol)
        self._eri = None
        self._keys = self._keys.union(['_eri'])

    def dump_flags(self):
        SCF.dump_flags(self)
        log.info(self, 'num. doubly occ = %d, num. singly occ = %d', \
                 (self.mol.nelectron-self.mol.spin)//2, self.mol.spin)

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None:
            mol = self.mol
        dm = _hack_mol_log(mol, self, init_guess_by_minao)
        return numpy.array((dm*.5,dm*.5))

    def init_guess_by_atom(self, mol=None):
        if mol is None:
            mol = self.mol
        dm = _hack_mol_log(mol, self, init_guess_by_atom)
        return numpy.array((dm*.5,dm*.5))

    def init_guess_by_1e(self, mol=None):
        if mol is None:
            mol = self.mol
        dm = _hack_mol_log(mol, self, init_guess_by_1e)
        return numpy.array((dm*.5,dm*.5))

    def init_guess_by_1e(self, mol=None):
        '''Initial guess from one electron system.'''
        if mol is None:
            mol = self.mol
        log.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = RHF.eig(self, h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    def init_guess_by_chkfile(self, mol=None, chkfile_name=None, project=True):
        from pyscf.scf import addons
        if mol is None:
            mol = self.mol
        if chkfile_name is None:
            chkfile_name = self.chkfile
        chk_mol, scf_rec = chkfile.load_scf(chkfile_name)

        def fproj(mo):
            if project:
                return addons.project_mo_nr2nr(chk_mol, mo, mol)
            else:
                return mo
        if scf_rec['mo_coeff'].ndim == 2:
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            if numpy.iscomplexobj(mo):
                raise RuntimeError('TODO: project DHF orbital to ROHF orbital')
            dm = self.make_rdm1(fproj(mo), mo_occ)
        else: #UHF
            mo = scf_rec['mo_coeff']
            mo_occ = scf_rec['mo_occ']
            dm =(make_rdm1(fproj(mo[0]), mo_occ[0]),
                 make_rdm1(fproj(mo[1]), mo_occ[1]))
        return dm

    def eig(self, h, s):
# Note Roothaan effective Fock do not provide correct orbital energy.
# We use alpha # fock and beta fock to define orbital energy.
# TODO, check other treatment  J. Chem. Phys. 133, 141102
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        feff, fa, fb = h
        mo_energy, mo_coeff = scipy.linalg.eigh(feff, s)
        mopen = mo_coeff[:,ncore:]
        ea = numpy.einsum('ik,ik->k', mopen, numpy.dot(fa, mopen))
        idx = ea.argsort()
        mo_energy[ncore:] = ea[idx]
        mo_coeff[:,ncore:] = mopen[:,idx]
        return mo_energy, mo_coeff

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
# Roothaan's effective fock
# http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html
#          |  closed     open    virtual
#  ----------------------------------------
#  closed  |    Fc        Fb       Fc
#  open    |    Fb        Fc       Fa
#  virtual |    Fc        Fa       Fc
# Fc = (Fa+Fb)/2
        fa0 = h1e + vhf[0]
        fb0 = h1e + vhf[1]
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        dmsf = dm[0]+dm[1]
        sds = -reduce(numpy.dot, (s1e, dmsf, s1e))
        _, mo_space = scipy.linalg.eigh(sds, s1e)
        fa = reduce(numpy.dot, (mo_space.T, fa0, mo_space))
        fb = reduce(numpy.dot, (mo_space.T, fb0, mo_space))
        feff = (fa + fb) * .5
        feff[:ncore,ncore:nocc] = fb[:ncore,ncore:nocc]
        feff[ncore:nocc,:ncore] = fb[ncore:nocc,:ncore]
        feff[nocc:,ncore:nocc] = fa[nocc:,ncore:nocc]
        feff[ncore:nocc,nocc:] = fa[ncore:nocc,nocc:]
        cinv = numpy.dot(mo_space.T, s1e)
        f = reduce(numpy.dot, (cinv.T, feff, cinv))

        if 0 <= cycle < self.diis_start_cycle-1:
            f = damping(s1e, dmsf*.5, f, self.damp_factor)
            f = level_shift(s1e, dmsf*.5, f, self.level_shift_factor)
        elif 0 <= cycle:
            # decay the level_shift_factor
            fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
            f = level_shift(s1e, dmsf*.5, f, fac)
        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dmsf, f)
# attach alpha and beta fock, because Roothaan effective Fock cannot provide
# correct orbital energy.  To define orbital energy in self.eig, we use alpha
# fock and beta fock.
# TODO, check other treatment  J. Chem. Phys. 133, 141102
        return f, fa0, fb0

    def get_occ(self, mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        mo_occ[:ncore] = 2
        mo_occ[ncore:nocc] = 1
        if nocc < mo_energy.size:
            log.info(self, 'HOMO = %.12g, LUMO = %.12g,', \
                     mo_energy[nocc-1], mo_energy[nocc])
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                log.warn(self.mol, '!! HOMO %.12g == LUMO %.12g', \
                         mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.info(self, 'HOMO = %.12g, no LUMO,', mo_energy[nocc-1])
        if nopen > 0:
            for i in range(ncore, nocc):
                log.debug(self, 'singly occupied orbital energy = %.12g', \
                          mo_energy[i])
        log.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        mo_a = mo_coeff[:,mo_occ>0]
        mo_b = mo_coeff[:,mo_occ==2]
        dm_a = numpy.dot(mo_a, mo_a.T)
        dm_b = numpy.dot(mo_b, mo_b.T)
        return numpy.array((dm_a, dm_b))

    def energy_elec(self, dm, h1e=None, vhf=None):
        import pyscf.scf.uhf
        ee, ecoul = pyscf.scf.uhf.energy_elec(self, dm, h1e, vhf)
        log.debug1(self, 'E_coul = %.15g', ecoul)
        return ee, ecoul

    def init_guess_by_minao(self, mol=None):
        if mol is None:
            mol = self.mol
        dm = _hack_mol_log(mol, self, init_guess_by_minao)
        return numpy.array((dm*.5,dm*.5))

    # pass in a set of density matrix in dm as (alpha,alpha,...,beta,beta,...)
    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        import pyscf.scf.uhf
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5,dm*.5))
        nset = len(dm) // 2
        if self._is_mem_enough() or self._eri is not None:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = pyscf.scf.uhf._makevhf(vj, vk, nset)
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last,copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = pyscf.scf.uhf._makevhf(vj, vk, nset) \
                + numpy.array(vhf_last, copy=False)
        else:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = pyscf.scf.uhf._makevhf(vj, vk, nset)
        return vhf

def _hack_mol_log(mol, dev, fn, *args, **kwargs):
    verbose_bak, mol.verbose = mol.verbose, dev.verbose
    stdout_bak,  mol.stdout  = mol.stdout , dev.stdout
    res = fn(mol, *args, **kwargs)
    mol.verbose = verbose_bak
    mol.stdout  = stdout_bak
    return res


if __name__ == '__main__':
    mol = pyscf.gto.Mole()
    mol.verbose = 5
    mol.output = None#'out_hf'

    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = 'ccpvdz'
    mol.build()

##############
# SCF result
    method = RHF(mol)
    method.init_guess = '1e'
    energy = method.scf()
    print(energy)
