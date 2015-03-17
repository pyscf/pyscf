#!/usr/bin/env python


import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.gto
import pyscf.lib
import pyscf.lib.logger as log
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import chkfile
from pyscf.scf import diis
from pyscf.scf import _vhf


def init_guess_by_minao(mol):
    '''Generate initial guess density matrix based on ANO basis, then project
    the density matrix to the basis set defined by ``mol``

    Returns:
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    dm = hf.init_guess_by_minao(mol)
    return numpy.array((dm*.5,dm*.5))

def init_guess_by_1e(mol):
    dm = hf.init_guess_by_1e(mol)
    return numpy.array((dm*.5,dm*.5))

def init_guess_by_atom(mol):
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
    '''One-particle densit matrix

    Returns:
        A list of 2D ndarrays for alpha and beta spins
    '''
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = numpy.dot(mo_a*mo_occ[0], mo_a.T.conj())
    dm_b = numpy.dot(mo_b*mo_occ[1], mo_b.T.conj())
    return numpy.array((dm_a,dm_b))

def get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None):
    r'''Unrestricted Hartree-Fock potential matrix of alpha and beta spins,
    for the given density matrix

    .. math::

        V_{ij}^\alpha &= \sum_{kl} (ij|kl)(\gamma_{lk}^\alpha+\gamma_{lk}^\beta)
                       - \sum_{kl} (il|kj)\gamma_{lk}^\alpha \\
        V_{ij}^\beta  &= \sum_{kl} (ij|kl)(\gamma_{lk}^\alpha+\gamma_{lk}^\beta)
                       - \sum_{kl} (il|kj)\gamma_{lk}^\beta

    Args:
        mol : an instance of :class:`Mole`

        dm : a list of ndarrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  When it is not 0, this function computes
            the increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference HF potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices

    Returns:
        :math:`V_{hf} = (V^\alpha, V^\beta)`.  :math:`V^\alpha` (and :math:`V^\beta`)
        can be a list matrices, corresponding to the input density matrices.

    Examples:

    >>> import numpy
    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dmsa = numpy.random.random((3,mol.nao_nr(),mol.nao_nr()))
    >>> dmsb = numpy.random.random((3,mol.nao_nr(),mol.nao_nr()))
    >>> dms = numpy.vstack((dmsa,dmsb))
    >>> dms.shape
    (6, 2, 2)
    >>> vhfa, vhfb = scf.uhf.get_veff(mol, dms, hermi=0)
    >>> vhfa.shape
    (3, 2, 2)
    >>> vhfb.shape
    (3, 2, 2)
    '''
    if ((isinstance(dm, numpy.ndarray) and dm.ndim == 4) or
        (isinstance(dm[0], numpy.ndarray) and dm[0].ndim == 3) or
        (isinstance(dm[0][0], numpy.ndarray) and dm[0][0].ndim == 2)):
        # remove first dim, compress (dma,dmb)
        dm = numpy.vstack(dm)
    ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
    vj, vk = hf.get_jk(mol, ddm, hermi=hermi, vhfopt=vhfopt)
    nset = len(dm) // 2
    vhf = _makevhf(vj, vk, nset) + numpy.array(vhf_last, copy=False)
    return vhf

def energy_elec(mf, dm, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock

    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
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
    r'''Spin of the given UHF orbitals

    .. math::

        S^2 = \frac{1}{2}(S_+ S_-  +  S_- S_+) + S_z^2

    where :math:`S_+ = \sum_i S_{i+}` is effective for all beta occupied
    orbitals; :math:`S_- = \sum_i S_{i-}` is effective for all alpha occupied
    orbitals.

    1. There are two possibilities for :math:`S_+ S_-`
        1) same electron :math:`S_+ S_- = \sum_i s_{i+} s_{i-}`,

        .. math::

            \sum_i \langle UHF|s_{i+} s_{i-}|UHF\rangle
             = \sum_{pq}\langle p|s_+s_-|q\rangle \gamma_{qp} = n_\alpha

        2) different electrons :math:`S_+ S_- = \sum s_{i+} s_{j-},  (i\neq j)`.
        There are in total :math:`n(n-1)` terms.  As a two-particle operator,

        .. math::

            \langle S_+ S_- \rangle = \langle ij|s_+ s_-|ij\rangle
                                    - \langle ij|s_+ s_-|ji\rangle
                                    = -\langle i^\alpha|j^\beta\rangle
                                       \langle j^\beta|i^\alpha\rangle

    2. Similarly, for :math:`S_- S_+`
        1) same electron

        .. math::

           \sum_i \langle s_{i-} s_{i+}\rangle = n_\beta

        2) different electrons

        .. math::

            \langle S_- S_+ \rangle = -\langle i^\beta|j^\alpha\rangle
                                       \langle j^\alpha|i^\beta\rangle

    3. For :math:`S_z^2`
        1) same electron

        .. math::

            \langle s_z^2\rangle = \frac{1}{4}(n_\alpha + n_\beta)

        2) different electrons

        .. math::

            &\frac{1}{2}\sum_{ij}(\langle ij|2s_{z1}s_{z2}|ij\rangle
                                 -\langle ij|2s_{z1}s_{z2}|ji\rangle) \\
            &=\frac{1}{4}(\langle i^\alpha|i^\alpha\rangle \langle j^\alpha|j^\alpha\rangle
             - \langle i^\alpha|i^\alpha\rangle \langle j^\beta|j^\beta\rangle
             - \langle i^\beta|i^\beta\rangle \langle j^\alpha|j^\alpha\rangle
             + \langle i^\beta|i^\beta\rangle \langle j^\beta|j^\beta\rangle) \\
            &-\frac{1}{4}(\langle i^\alpha|i^\alpha\rangle \langle i^\alpha|i^\alpha\rangle
             + \langle i^\beta|i^\beta\rangle\langle i^\beta|i^\beta\rangle) \\
            &=\frac{1}{4}(n_\alpha^2 - n_\alpha n_\beta - n_\beta n_\alpha + n_\beta^2)
             -\frac{1}{4}(n_\alpha + n_\beta) \\
            &=\frac{1}{4}((n_\alpha-n_\beta)^2 - (n_\alpha+n_\beta))

    In total

    .. math::

        \langle S^2\rangle &= \frac{1}{2}
        (n_\alpha-\sum_{ij}\langle i^\alpha|j^\beta\rangle \langle j^\beta|i^\alpha\rangle
        +n_\beta -\sum_{ij}\langle i^\beta|j^\alpha\rangle\langle j^\alpha|i^\beta\rangle)
        + \frac{1}{4}(n_\alpha-n_\beta)^2 \\

    Args:
        mo : a list of 2 ndarrays
            Occupied alpha and occupied beta orbitals

    Kwargs:
        ovlp : ndarray
            AO overlap

    Returns:
        A list of two floats.  The first is the expectation value of S^2.
        The second is the corresponding 2S+1

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', charge=1, spin=1, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> mf.kernel()
    -75.623975516256706
    >>> mo = (mf.mo_coeff[0][:,mf.mo_occ[0]>0], mf.mo_coeff[1][:,mf.mo_occ[1]>0])
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % spin_square(mo, mol.intor('cint1e_ovlp_sph')))
    S^2 = 0.7570150, 2S+1 = 2.0070027
    '''
    mo_a, mo_b = mo
    nocc_a = mo_a.shape[1]
    nocc_b = mo_b.shape[1]
    s = reduce(numpy.dot, (mo_a.T, ovlp, mo_b))
    ssxy = (nocc_a+nocc_b) * .5 - (s**2).sum()
    ssz = (nocc_b-nocc_a)**2 * .25
    ss = ssxy + ssz
    s = numpy.sqrt(ss+.25) - .5
    return ss, s*2+1

def analyze(mf, verbose=logger.DEBUG):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis
    '''
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    log = logger.Logger(mf.stdout, verbose)
    ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                            mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
    log.info('multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)

    log.info('**** MO energy ****')
    for i in range(mo_energy[0].__len__()):
        if mo_occ[0][i] > 0:
            log.info("alpha occupied MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[0][i], mo_occ[0][i])
        else:
            log.info("alpha virtual MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[0][i], mo_occ[0][i])
    for i in range(mo_energy[1].__len__()):
        if mo_occ[1][i] > 0:
            log.info("beta occupied MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[1][i], mo_occ[1][i])
        else:
            log.info("beta virtual MO #%d energy = %.15g occ= %g",
                     i+1, mo_energy[1][i], mo_occ[1][i])
    if mf.verbose >= logger.DEBUG:
        log.debug(' ** MO coefficients for alpha spin **')
        label = ['%d%3s %s%-4s' % x for x in mf.mol.spheric_labels()]
        dump_mat.dump_rec(mf.stdout, mo_coeff[0], label, start=1)
        log.debug(' ** MO coefficients for beta spin **')
        dump_mat.dump_rec(mf.stdout, mo_coeff[1], label, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_pop(mf.mol, dm, mf.get_ovlp(), log)

def mulliken_pop(mol, dm, ovlp=None, verbose=logger.DEBUG):
    '''Mulliken population analysis
    '''
    if ovlp is None:
        ovlp = hf.get_ovlp(mol)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    pop_a = numpy.einsum('ij->i', dm[0]*ovlp)
    pop_b = numpy.einsum('ij->i', dm[1]*ovlp)
    label = mol.spheric_labels()

    log.info(' ** Mulliken pop alpha/beta **')
    for i, s in enumerate(label):
        log.info('pop of  %s %10.5f  / %10.5f', \
                 '%d%s %s%4s'%s, pop_a[i], pop_b[i])

    log.info(' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop_a[i] + pop_b[i]
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        nuc = mol.atom_charge(ia)
        chg[ia] = nuc - chg[ia]
        log.info('charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return (pop_a,pop_b), chg

def mulliken_pop_meta_lowdin_ao(mol, dm_ao, verbose=logger.DEBUG,
                                pre_orth_method='ANO'):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    from pyscf.lo import orth
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    c = orth.pre_orth_ao(mol, pre_orth_method)
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c)
    c_inv = numpy.linalg.inv(orth_coeff)
    dm_a = reduce(numpy.dot, (c_inv, dm_ao[0], c_inv.T.conj()))
    dm_b = reduce(numpy.dot, (c_inv, dm_ao[1], c_inv.T.conj()))

    log.info(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mulliken_pop(mol, (dm_a,dm_b), numpy.eye(orth_coeff.shape[0]), log)

def map_rhf_to_uhf(rhf):
    '''Take the settings from RHF object'''
    assert(isinstance(rhf, hf.RHF))
    uhf = UHF(rhf.mol)
    uhf.__dict__.update(rhf.__dict__)
    uhf.mo_energy = numpy.array((rhf.mo_energy,rhf.mo_energy))
    uhf.mo_coeff  = numpy.array((rhf.mo_coeff,rhf.mo_coeff))
    uhf.mo_occ    = numpy.array((rhf.mo_occ,rhf.mo_occ))
    return uhf

class UHF(hf.SCF):
    __doc__ = hf.SCF.__doc__ + '''
    Attributes for UHF:
        nelectron_alpha : int
            If given, fix the number of alpha electrons to the given value

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', charge=1, spin=1, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> mf.kernel()
    -75.623975516256706
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % mf.spin_square())
    S^2 = 0.7570150, 2S+1 = 2.0070027
    '''
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        # self.mo_coeff => [mo_a, mo_b]
        # self.mo_occ => [mo_occ_a, mo_occ_b]
        # self.mo_energy => [mo_energy_a, mo_energy_b]

        self.DIIS = UHF_DIIS
        self.nelectron_alpha = (mol.nelectron + mol.spin) // 2
        self._eri = None
        self._keys = self._keys.union(['nelectron_alpha'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        log.info(self, 'number electrons alpha = %d, beta = %d', \
                 self.nelectron_alpha,
                 self.mol.nelectron-self.nelectron_alpha)

    def eig(self, fock, s):
        e_a, c_a = hf.SCF.eig(self, fock[0], s)
        e_b, c_b = hf.SCF.eig(self, fock[1], s)
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

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
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

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None: dm = self.make_rdm1()
        return energy_elec(self, dm, h1e, vhf)

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    def init_guess_by_1e(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_1e(mol)

    def init_guess_by_chkfile(self, mol=None, chkfile=None, project=True):
        if mol is None: mol = self.mol
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(mol, chkfile, project=project)

    def get_jk(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (time.clock(), time.time())
        if self._is_mem_enough() or self._eri is not None:
            if self._eri is None:
                self._eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi)
        else:
            vj, vk = hf.get_jk(mol, dm, hermi, self.opt)
        log.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Hartree-Fock potential matrix for the given density matrices.
        See :func:`scf.uhf.get_veff`
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
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
                self.mo_energy, self.mo_coeff, self.mo_occ \
                = hf.kernel(self, self.conv_tol, dm0=dm0)
#        if self.nelectron_alpha * 2 < self.mol.nelectron:
#            self.mo_coeff = (self.mo_coeff[1], self.mo_coeff[0])
#            self.mo_occ = (self.mo_occ[1], self.mo_occ[0])
#            self.mo_energy = (self.mo_energy[1], self.mo_energy[0])

        log.timer(self, 'SCF', *cput0)
        self.dump_energy(self.hf_energy, self.converged)
        #if self.verbose >= logger.INFO:
        #    self.analyze(self.verbose)
        return self.hf_energy

    def analyze(self, verbose=logger.DEBUG):
        return analyze(self, verbose)

    def mulliken_pop(self, mol=None, dm=None, ovlp=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if ovlp is None: ovlp = self.get_ovlp(mol)
        return mulliken_pop(mol, dm, ovlp, verbose)

    def mulliken_pop_meta_lowdin_ao(self, mol=None, dm=None,
                                    verbose=logger.DEBUG,
                                    pre_orth_method='ANO'):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return mulliken_pop_meta_lowdin_ao(mol, dm, verbose, pre_orth_method)

    def spin_square(self, mo_coeff=None, ovlp=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        if ovlp is None:
            ovlp = self.get_ovlp()
        return spin_square(mo_coeff, ovlp)


class UHF_DIIS(pyscf.lib.diis.DIIS):
    def update(self, s, d, f):
        sdf_a = reduce(numpy.dot, (s, d[0], f[0]))
        sdf_b = reduce(numpy.dot, (s, d[1], f[1]))
        errvec = numpy.hstack((sdf_a.T.conj() - sdf_a, \
                               sdf_b.T.conj() - sdf_b))
        log.debug1(self, 'diis-norm(errvec) = %g', numpy.linalg.norm(errvec))
        pyscf.lib.diis.DIIS.push_err_vec(self, errvec)
        return pyscf.lib.diis.DIIS.update(self, f)

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
