#!/usr/bin/env python


import time
from functools import reduce
import numpy
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import chkfile
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
        mo_coeff = fproj(mo)
        mo_a = mo_coeff[:,mo_occ>0]
        mo_b = mo_coeff[:,mo_occ>1]
        dm_a = numpy.dot(mo_a, mo_a.T)
        dm_b = numpy.dot(mo_b, mo_b.T)
        dm = numpy.array((dm_a, dm_b))
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

def get_fock_(mf, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    f = h1e + vhf
    if f.ndim == 2:
        f = (f, f)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = [dm*.5] * 2
    if 0 <= cycle < diis_start_cycle-1:
        f = (hf.damping(s1e, dm[0], f[0], damp_factor),
             hf.damping(s1e, dm[1], f[1], damp_factor))
    if adiis and cycle >= diis_start_cycle:
        f = adiis.update(s1e, dm, numpy.array(f))
    f = (hf.level_shift(s1e, dm[0], f[0], level_shift_factor),
         hf.level_shift(s1e, dm[1], f[1], level_shift_factor))
    return numpy.array(f)

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''UHF Gradients'''
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]

    focka = reduce(numpy.dot, (mo_coeff[0].T, fock_ao[0], mo_coeff[0]))
    fockb = reduce(numpy.dot, (mo_coeff[1].T, fock_ao[1], mo_coeff[1]))
    g = numpy.hstack((focka[viridxa[:,None],occidxa].reshape(-1),
                      fockb[viridxb[:,None],occidxb].reshape(-1)))
    return g.reshape(-1)

def energy_elec(mf, dm, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock

    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = numpy.einsum('ij,ij', h1e.conj(), dm[0]+dm[1])
    e_coul = numpy.einsum('ij,ji', vhf[0].conj(), dm[0]) \
           + numpy.einsum('ij,ji', vhf[1].conj(), dm[1])
    e_coul *= .5
    return e1+e_coul, e_coul

# mo_a and mo_b are occupied orbitals
def spin_square(mo, s=1):
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
        s : ndarray
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
    s = reduce(numpy.dot, (mo_a.T, s, mo_b))
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
    log.info('multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)

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
        label = mf.mol.spheric_labels(True)
        dump_mat.dump_rec(mf.stdout, mo_coeff[0], label, start=1)
        log.debug(' ** MO coefficients for beta spin **')
        dump_mat.dump_rec(mf.stdout, mo_coeff[1], label, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_meta(mf.mol, dm, s=mf.get_ovlp(), verbose=log)

def mulliken_pop(mol, dm, s=None, verbose=logger.DEBUG):
    '''Mulliken population analysis
    '''
    if s is None:
        s = hf.get_ovlp(mol)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    pop_a = numpy.einsum('ij->i', dm[0]*s)
    pop_b = numpy.einsum('ij->i', dm[1]*s)
    label = mol.spheric_labels(False)

    log.info(' ** Mulliken pop alpha/beta **')
    for i, s in enumerate(label):
        log.info('pop of  %s %10.5f  / %10.5f',
                 '%d%s %s%4s'%s, pop_a[i], pop_b[i])

    log.info(' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(label):
        chg[s[0]] += pop_a[i] + pop_b[i]
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        chg[ia] = mol.atom_charge(ia) - chg[ia]
        log.info('charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return (pop_a,pop_b), chg

def mulliken_pop_meta_lowdin_ao(mol, dm_ao, verbose=logger.DEBUG,
                                pre_orth_method='ANO', s=None):
    return mulliken_meta(mol, dm_ao, verbose, pre_orth_method, s)
def mulliken_meta(mol, dm_ao, verbose=logger.DEBUG, pre_orth_method='ANO',
                  s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    from pyscf.lo import orth
    if s is None:
        s = hf.get_ovlp(mol)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    if isinstance(dm_ao, numpy.ndarray) and dm_ao.ndim == 2:
        dm_ao = numpy.array((dm_ao*.5, dm_ao*.5))
    c = orth.pre_orth_ao(mol, pre_orth_method)
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c, s=s)
    c_inv = numpy.dot(orth_coeff.T, s)
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
        nelec : (int, int)
            If given, freeze the number of (alpha,beta) electrons to the given value

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

        n_a = (mol.nelectron + mol.spin) // 2
        self.nelec = (n_a, mol.nelectron - n_a)
        self._keys = self._keys.union(['nelec'])

    def dump_flags(self):
        if hasattr(self, 'nelectron_alpha'):
            logger.warn(self, 'Note the API updates: attribute nelectron_alpha was replaced by attribute nelec')
            #raise RuntimeError('API updates')
            self.nelec = (self.nelectron_alpha,
                          self.mol.nelectron-self.nelectron_alpha)
            delattr(self, 'nelectron_alpha')
        hf.SCF.dump_flags(self)
        logger.info(self, 'number electrons alpha = %d  beta = %d', *self.nelec)

    def eig(self, fock, s):
        e_a, c_a = hf.SCF.eig(self, fock[0], s)
        e_b, c_b = hf.SCF.eig(self, fock[1], s)
        return numpy.array((e_a,e_b)), (c_a,c_b)

    def get_fock_(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                  diis_start_cycle=None, level_shift_factor=None,
                  damp_factor=None):
        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp
        return get_fock_(self, h1e, s1e, vhf, dm, cycle, adiis,
                         diis_start_cycle, level_shift_factor, damp_factor)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        n_a, n_b = self.nelec
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[0][:n_a] = 1
        mo_occ[1][:n_b] = 1
        if n_a < mo_energy[0].size:
            logger.info(self, 'alpha nocc = %d  HOMO = %.12g  LUMO = %.12g',
                        n_a, mo_energy[0][n_a-1], mo_energy[0][n_a])
            if mo_energy[0][n_a-1]+1e-3 > mo_energy[0][n_a]:
                logger.warn(self, '!! alpha HOMO %.12g >= LUMO %.12g',
                            mo_energy[0][n_a-1], mo_energy[0][n_a])
        else:
            logger.info(self, 'alpha nocc = %d  HOMO = %.12g  no LUMO',
                        n_a, mo_energy[0][n_a-1])
        if self.verbose >= logger.DEBUG:
            numpy.set_printoptions(threshold=len(mo_energy[0]))
            logger.debug(self, '  mo_energy = %s', mo_energy[0])

        if n_b > 0 and n_b < mo_energy[1].size:
            logger.info(self, 'beta  nocc = %d  HOMO = %.12g  LUMO = %.12g',
                        n_b, mo_energy[1][n_b-1], mo_energy[1][n_b])
            if mo_energy[1][n_b-1]+1e-3 > mo_energy[1][n_b]:
                logger.warn(self, '!! beta HOMO %.12g >= LUMO %.12g',
                            mo_energy[1][n_b-1], mo_energy[1][n_b])
            if mo_energy[0][n_a-1]+1e-3 > mo_energy[1][n_b]:
                logger.warn(self, '!! system HOMO %.12g >= system LUMO %.12g',
                            mo_energy[0][n_a-1], mo_energy[1][n_b])
        elif n_b > 0:
            logger.info(self, 'beta nocc = %d  HOMO = %.12g  no LUMO',
                        n_b, mo_energy[1][n_b-1])
        else:
            logger.info(self, 'beta  nocc = %d  no HOMO  LUMO = %.12g',
                        n_b, mo_energy[1][n_b])
        if self.verbose >= logger.DEBUG:
            logger.debug(self, '  mo_energy = %s', mo_energy[1])
            numpy.set_printoptions()

        if mo_coeff is not None:
            ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                      mo_coeff[1][:,mo_occ[1]>0]),
                                      self.get_ovlp())
            logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
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

    def init_guess_by_chkfile(self, chk=None, project=True):
        if chk is None: chk = self.chkfile
        return init_guess_by_chkfile(self.mol, chk, project=project)

    def get_jk_(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        dm = numpy.asarray(dm)
        nao = dm.shape[-1]
        cpu0 = (time.clock(), time.time())
        if self._eri is not None or mol.incore_anyway or self._is_mem_enough():
            if self._eri is None:
                self._eri = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
            vj, vk = hf.dot_eri_dm(self._eri, dm.reshape(-1,nao,nao), hermi)
        else:
            if self.direct_scf:
                self.opt = self.init_direct_scf(mol)
            vj, vk = hf.get_jk(mol, dm.reshape(-1,nao,nao), hermi, self.opt)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj.reshape(dm.shape), vk.reshape(dm.shape)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Hartree-Fock potential matrix for the given density matrices.
        See :func:`scf.uhf.get_veff`

        Args:
            mol : an instance of :class:`Mole`

            dm : a list of ndarrays
                A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5,dm*.5))
        nset = len(dm) // 2
        if (self._eri is not None or not self.direct_scf or
            mol.incore_anyway or self._is_mem_enough()):
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = _makevhf(vj, vk, nset)
        else:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last,copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = _makevhf(vj, vk, nset) + numpy.array(vhf_last, copy=False)
        return vhf

    def analyze(self, verbose=logger.DEBUG):
        return analyze(self, verbose)

    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_pop(mol, dm, s=s, verbose=verbose)

    def mulliken_pop_meta_lowdin_ao(self, mol=None, dm=None,
                                    verbose=logger.DEBUG,
                                    pre_orth_method='ANO', s=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return mulliken_pop_meta_lowdin_ao(mol, dm, s=s, verbose=verbose,
                                           pre_orth_method=pre_orth_method)
    def mulliken_meta(self, *args, **kwargs):
        return self.mulliken_pop_meta_lowdin_ao(*args, **kwargs)

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        if s is None:
            s = self.get_ovlp()
        return spin_square(mo_coeff, s)


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
