#!/usr/bin/env python


import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.gto
import pyscf.lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.scf import hf
from pyscf.scf import chkfile
from pyscf.scf import diis
from pyscf.scf import _vhf


def init_guess_by_minao(mol):
    dm = hf.init_guess_by_minao(mol)
    return numpy.array((dm*.5,dm*.5))

def init_guess_by_1e(mol):
    '''Initial guess from one electron system.'''
    dm = hf.init_guess_by_1e(mol)
    return numpy.array((dm*.5,dm*.5))

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    dm = hf.init_guess_by_atom(mol)
    return numpy.array((dm*.5,dm*.5))

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
            raise RuntimeError('TODO: project DHF orbital to UHF orbital')
        dm = make_rdm1([fproj(mo),]*2, [mo_occ*.5,]*2)
    else: #UHF
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = make_rdm1([fproj(mo[0]),fproj(mo[1])], mo_occ)
    return dm

def get_init_guess(mol, key='minao'):
    if callable(key):
        return key(mol)
    elif key.lower() == '1e':
        return init_guess_by_1e(mol)
    elif key.lower() == 'atom':
        return init_guess_by_atom(mol)
    elif key.lower() == 'chkfile':
        raise RuntimeError('Call pyscf.scf.uhf.init_guess_by_chkfile instead')
    else:
        return init_guess_by_minao(mol)

def make_rdm1(mo_coeff, mo_occ):
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = numpy.dot(mo_a*mo_occ[0], mo_a.T.conj())
    dm_b = numpy.dot(mo_b*mo_occ[1], mo_b.T.conj())
    return numpy.array((dm_a,dm_b))

def get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1):
    '''NR Hartree-Fock Coulomb repulsion'''
    ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
    vj, vk = _vhf.direct(ddm, mol._atm, mol._bas, mol._env, hermi=hermi)
    nset = len(dm) // 2
    vhf = _makevhf(vj, vk, nset) + numpy.array(vhf_last, copy=False)
    return vhf

def energy_elec(mf, dm, h1e=None, vhf=None):
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = numpy.einsum('ij,ij', h1e.conj(), dm[0]+dm[1])
    e_coul = numpy.einsum('ij,ji', vhf[0].conj(), dm[0]) \
           + numpy.einsum('ij,ji', vhf[1].conj(), dm[1])
    e_coul *= .5
    return e1+e_coul, e_coul

# mo_a and mo_b are occupied orbitals
def spin_square(mo, ovlp=1):
# S^2 = (S+ * S- + S- * S+)/2 + Sz * Sz
# S+ = \sum_i S_i+ ~ effective for all beta occupied orbitals.
# S- = \sum_i S_i- ~ effective for all alpha occupied orbitals.
# There are two cases for S+*S-
# 1) same electron \sum_i s_i+*s_i-, <UHF|s_i+*s_i-|UHF> gives
#       <p|s+s-|q> \gamma_qp = nocc_a
# 2) different electrons for \sum s_i+*s_j- (i\neq j, n*(n-1) terms)
# As a two-particle operator S+*S-
#       <ij|s+s-|ij> - <ij|s+s-|ji> = -<ij|s+s-|ji>
#       = -<ia|jb><jb|ia>
# <UHF|S+*S-|UHF> = nocc_a - <ia|jb><jb|ia>
#
# There are two cases for S-*S+
# 1) same electron \sum_i s_i-*s_i+
#       <p|s-s+|q> \gamma_qp = nocc_b
# 2) different electrons
#       <ij|s-s+|ij> - <ij|s-s+|ji> = -<ij|s-s+|ji>
#       = -<ib|ja><ja|ib>
# <UHF|S+*S-|UHF> = nocc_b - <ib|ja><ja|ib>
#
# Sz*Sz = Msz^2 = (nocc_a-nocc_b)^2
# 1) same electron
#       <p|ss|q> = (nocc_a+nocc_b)/4
# 2) different electrons
#       (<ij|2s1s2|ij>-<ij|2s1s2|ji>)/2 = <ij|s1s2|ij> - <ii|s1s2|ii>
#       = (<ia|ia><ja|ja> - <ia|ia><jb|jb> - <ib|ib><ja|ja> + <ib|ib><jb|jb>)/4
#       - (<ia|ia><ia|ia>+<ib|ib><ib|ib>)/4
#       = (nocc_a^2 - nocc_a*nocc_b - nocc_b*nocc_a + nocc_b^2)/4
#       - (nocc_a+nocc_b)/4
#       = (nocc_a-nocc_b)^2 / 4 - (nocc_a+nocc_b)/4
#
    mo_a, mo_b = mo
    nocc_a = mo_a.shape[1]
    nocc_b = mo_b.shape[1]
    s = reduce(numpy.dot, (mo_a.T, ovlp, mo_b))
    ssxy = (nocc_a+nocc_b) * .5 - (s**2).sum()
    ssz = (nocc_b-nocc_a)**2 * .25
    ss = ssxy + ssz
    s = numpy.sqrt(ss+.25) - .5
    return ss, s*2+1

def analyze(mf, mo_energy=None, mo_occ=None, mo_coeff=None):
    from pyscf.tools import dump_mat
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                            mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
    log.info(mf, 'multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)

    log.info(mf, '**** MO energy ****')
    for i in range(mo_energy[0].__len__()):
        if mo_occ[0][i] > 0:
            log.info(mf, "alpha occupied MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[0][i], mo_occ[0][i])
        else:
            log.info(mf, "alpha virtual MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[0][i], mo_occ[0][i])
    for i in range(mo_energy[1].__len__()):
        if mo_occ[1][i] > 0:
            log.info(mf, "beta occupied MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[1][i], mo_occ[1][i])
        else:
            log.info(mf, "beta virtual MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[1][i], mo_occ[1][i])
    if mf.verbose >= param.VERBOSE_DEBUG:
        log.debug(mf, ' ** MO coefficients for alpha spin **')
        label = ['%d%3s %s%-4s' % x for x in mf.mol.spheric_labels()]
        dump_mat.dump_rec(mf.stdout, mo_coeff[0], label, start=1)
        log.debug(mf, ' ** MO coefficients for beta spin **')
        dump_mat.dump_rec(mf.stdout, mo_coeff[1], label, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_pop(mf.mol, dm, mf.get_ovlp())

def mulliken_pop(mol, dm, ovlp=None):
    '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
    if ovlp is None:
        ovlp = hf.get_ovlp(mol)
    pop_a = numpy.einsum('ij->i', dm[0]*ovlp)
    pop_b = numpy.einsum('ij->i', dm[1]*ovlp)
    label = mol.spheric_labels()

    log.info(mol, ' ** Mulliken pop alpha/beta **')
    for i, s in enumerate(label):
        log.info(mol, 'pop of  %s %10.5f  / %10.5f', \
                 '%d%s %s%4s'%s, pop_a[i], pop_b[i])

    log.info(mol, ' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop_a[i] + pop_b[i]
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        nuc = mol.atom_charge(ia)
        chg[ia] = nuc - chg[ia]
        log.info(mol, 'charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return (pop_a,pop_b), chg

def mulliken_pop_meta_lowdin_ao(mol, dm_ao):
    from pyscf.lo import orth
    c = orth.pre_orth_ao_atm_scf(mol)
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c)
    c_inv = numpy.linalg.inv(orth_coeff)
    dm_a = reduce(numpy.dot, (c_inv, dm_ao[0], c_inv.T.conj()))
    dm_b = reduce(numpy.dot, (c_inv, dm_ao[1], c_inv.T.conj()))

    log.info(mol, ' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mulliken_pop(mol, (dm_a,dm_b), numpy.eye(orth_coeff.shape[0]))

def map_rhf_to_uhf(rhf):
    assert(isinstance(rhf, hf.RHF))
    uhf = UHF(rhf.mol)
    uhf.__dict__.update(rhf.__dict__)
    uhf.mo_energy = numpy.array((rhf.mo_energy,rhf.mo_energy))
    uhf.mo_coeff  = numpy.array((rhf.mo_coeff,rhf.mo_coeff))
    uhf.mo_occ    = numpy.array((rhf.mo_occ,rhf.mo_occ))
    return uhf

class UHF(hf.SCF):
    '''UHF'''
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        # self.mo_coeff => [mo_a, mo_b]
        # self.mo_occ => [mo_occ_a, mo_occ_b]
        # self.mo_energy => [mo_energy_a, mo_energy_b]

        self.DIIS = UHF_DIIS
        self.nelectron_alpha = (mol.nelectron + mol.spin) // 2
        self._eri = None
        self._keys = self._keys.union(['nelectron_alpha', '_eri'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        log.info(self, 'number electrons alpha = %d, beta = %d', \
                 self.nelectron_alpha,
                 self.mol.nelectron-self.nelectron_alpha)

    def eig(self, fock, s):
        e_a, c_a = scipy.linalg.eigh(fock[0], s)
        e_b, c_b = scipy.linalg.eigh(fock[1], s)
        return numpy.array((e_a,e_b)), (c_a,c_b)

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
        f = (h1e+vhf[0], h1e+vhf[1])
        if 0 <= cycle < self.diis_start_cycle-1:
            f = (hf.damping(s1e, dm[0], f[0], self.damp_factor), \
                 hf.damping(s1e, dm[1], f[1], self.damp_factor))
            f = (hf.level_shift(s1e, dm[0], f[0], self.level_shift_factor), \
                 hf.level_shift(s1e, dm[1], f[1], self.level_shift_factor))
        elif 0 <= cycle:
            fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
            f = (hf.level_shift(s1e, dm[0], f[0], fac), \
                 hf.level_shift(s1e, dm[1], f[1], fac))
        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dm, numpy.array(f))
        return f

    def get_occ(self, mo_energy, mo_coeff=None):
        n_a = self.nelectron_alpha
        n_b = self.mol.nelectron - n_a
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
                 n_b, mo_energy[1][n_b-1], mo_energy[1][n_b])
        log.debug(self, '  mo_energy = %s', mo_energy[1])
        if mo_coeff is not None:
            ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                      mo_coeff[1][:,mo_occ[1]>0]),
                                      self.get_ovlp())
            log.debug(self, 'multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ)

    def energy_elec(self, dm, h1e=None, vhf=None):
        return energy_elec(self, dm, h1e, vhf)

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
        if mol is None:
            mol = self.mol
        return _hack_mol_log(mol, self, init_guess_by_1e)

    def init_guess_by_chkfile(self, mol=None, chkfile=None, project=True):
        if mol is None:
            mol = self.mol
        if chkfile is None:
            chkfile = self.chkfile
        return _hack_mol_log(mol, self, init_guess_by_chkfile, chkfile,
                             project=project)

    def get_jk(self, mol, dm, hermi=1):
        t0 = (time.clock(), time.time())
        if self._is_mem_enough() or self._eri is not None:
            if self._eri is None:
                self._eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi)
        else:
            vj, vk = hf.get_jk(mol, dm, hermi, self.opt)
        log.timer(self, 'vj and vk', *t0)
        return vj, vk

    # pass in a set of density matrix in dm as (alpha,alpha,...,beta,beta,...)
    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''NR UHF Coulomb repulsion'''
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5,dm*.5))
        nset = len(dm) // 2
        if self._is_mem_enough() or self._eri is not None:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = _makevhf(vj, vk, nset)
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last,copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = _makevhf(vj, vk, nset) + numpy.array(vhf_last, copy=False)
        else:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = _makevhf(vj, vk, nset)
        return vhf

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())

        self.build()
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.kernel(self, self.conv_tol, init_dm=dm0)
#        if self.nelectron_alpha * 2 < self.mol.nelectron:
#            self.mo_coeff = (self.mo_coeff[1], self.mo_coeff[0])
#            self.mo_occ = (self.mo_occ[1], self.mo_occ[0])
#            self.mo_energy = (self.mo_energy[1], self.mo_energy[0])

        log.timer(self, 'SCF', *cput0)
        self.dump_energy(self.hf_energy, self.converged)
        if self.verbose >= param.VERBOSE_INFO:
            self.analyze(self.mo_energy, self.mo_occ, self.mo_coeff)
        return self.hf_energy

    def analyze(self, mo_energy=None, mo_occ=None, mo_coeff=None):
        return analyze(self, mo_energy, mo_occ, mo_coeff)

    def mulliken_pop(self, mol, dm, ovlp=None):
        return _hack_mol_log(mol, self, mulliken_pop, dm, ovlp)

    def mulliken_pop_meta_lowdin_ao(self, mol, dm):
        return _hack_mol_log(mol, self, mulliken_pop_meta_lowdin_ao, dm)

    def spin_square(self, mo_coeff=None, ovlp=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ovlp is None:
            ovlp = self.get_ovlp()
        return spin_square(mo_coeff, ovlp)


class UHF_DIIS(diis.SCF_DIIS):
    def push_err_vec(self, s, d, f):
        sdf_a = reduce(numpy.dot, (s, d[0], f[0]))
        sdf_b = reduce(numpy.dot, (s, d[1], f[1]))
        errvec = numpy.hstack((sdf_a.T.conj() - sdf_a, \
                               sdf_b.T.conj() - sdf_b))
        log.debug1(self, 'diis-norm(errvec) = %g', numpy.linalg.norm(errvec))
        self.err_vec_stack.append(errvec)
        if self.err_vec_stack.__len__() > self.space:
            self.err_vec_stack.pop(0)

def _hack_mol_log(mol, dev, fn, *args, **kwargs):
    verbose_bak, mol.verbose = mol.verbose, dev.verbose
    stdout_bak,  mol.stdout  = mol.stdout , dev.stdout
    res = fn(mol, *args, **kwargs)
    mol.verbose = verbose_bak
    mol.stdout  = stdout_bak
    return res

def _makevhf(vj, vk, nset):
    if nset == 1:
        vj = vj[0] + vj[1]
        v_a = vj - vk[0]
        v_b = vj - vk[1]
    else:
        vj = vj[:nset] + vj[nset:]
        v_a = vj - vk[:nset]
        v_b = vj - vk[nset:]
    return numpy.array((v_a,v_b))
