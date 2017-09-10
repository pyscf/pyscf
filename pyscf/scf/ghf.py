#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic generalized Hartree-Fock
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import uhf
from pyscf.scf import chkfile

def get_jk(mol, dm, hermi=0,
           with_j=True, with_k=True, jkbuild=hf.get_jk):

    nao = mol.nao_nr()
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        dmaa = dm[:nao,:nao]
        dmab = dm[nao:,:nao]
        dmbb = dm[nao:,nao:]
        dms = (dmaa, dmbb, dmab)
    else:
        n_dm = len(dm)
        dms =([dmi[:nao,:nao] for dmi in dm]
            + [dmi[nao:,nao:] for dmi in dm]
            + [dmi[nao:,:nao] for dmi in dm])
    dms = numpy.asarray(dms)
    if dm[0].dtype == numpy.complex128:
        dms = numpy.vstack((dms.real, dms.imag))
        hermi = 0

    j1, k1 = jkbuild(mol, dms, hermi)

    if dm[0].dtype == numpy.complex128:
        if with_j: j1 = j1[:n_dm*3] + j1[n_dm*3:] * 1j
        if with_k: k1 = k1[:n_dm*3] + k1[n_dm*3:] * 1j

    vj = vk = None
    if with_j:
        vj = numpy.zeros_like(dm)
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            j1 = j1.reshape(3,nao,nao)
            vj[:nao,:nao] = vj[nao:,nao:] = j1[0] + j1[1]
        else:
            j1 = j1.reshape(3,n_dm,nao,nao)
            vj[:,:nao,:nao] = vj[:,nao:,nao:] = j1[0] + j1[1]

    if with_k:
        vk = numpy.zeros_like(dm)
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            k1 = k1.reshape(3,nao,nao)
            vk[:nao,:nao] = k1[0]
            vk[nao:,nao:] = k1[1]
            vk[:nao,nao:] = k1[2]
            vk[nao:,:nao] = k1[2].T.conj()
        else:
            k1 = k1.reshape(3,n_dm,nao,nao)
            vk[:,:nao,:nao] = k1[0]
            vk[:,nao:,nao:] = k1[1]
            vk[:,:nao,nao:] = k1[2]
            vk[:,nao:,:nao] = k1[2].transpose(0,2,1).conj()

    return vj, vk

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = numpy.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = mo_energy.size
    mo_occ = numpy.zeros(nmo)
    nocc = mf.mol.nelectron
    mo_occ[e_idx[:nocc]] = 1
    if mf.verbose >= logger.INFO and nocc < nmo:
        if e_sort[nocc-1]+1e-3 > e_sort[nocc]:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g',
                        e_sort[nocc-1], e_sort[nocc])
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g',
                        e_sort[nocc-1], e_sort[nocc])

    if mf.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)

    if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        ss, s = mf.spin_square(mo_coeff[:,mo_occ>0], mf.get_ovlp())
        logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
    return mo_occ

# mo_a and mo_b are occupied orbitals
def spin_square(mo, s=1):
    r'''Spin of the GHF wavefunction

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

            \langle S_+ S_- \rangle
            =\sum_{ij}(\langle i^\alpha|i^\beta\rangle \langle j^\beta|j^\alpha\rangle
            - \langle i^\alpha|j^\beta\rangle \langle j^\beta|i^\alpha\rangle)

    2. Similarly, for :math:`S_- S_+`
        1) same electron

        .. math::

           \sum_i \langle s_{i-} s_{i+}\rangle = n_\beta

        2) different electrons

        .. math::

            \langle S_- S_+ \rangle
            =\sum_{ij}(\langle i^\beta|i^\alpha\rangle \langle j^\alpha|j^\beta\rangle
            - \langle i^\beta|j^\alpha\rangle \langle j^\alpha|i^\beta\rangle)

    3. For :math:`S_z^2`
        1) same electron

        .. math::

            \langle s_z^2\rangle = \frac{1}{4}(n_\alpha + n_\beta)

        2) different electrons

        .. math::

            &\sum_{ij}(\langle ij|s_{z1}s_{z2}|ij\rangle
                      -\langle ij|s_{z1}s_{z2}|ji\rangle) \\
            &=\frac{1}{4}\sum_{ij}(\langle i^\alpha|i^\alpha\rangle \langle j^\alpha|j^\alpha\rangle
             - \langle i^\alpha|i^\alpha\rangle \langle j^\beta|j^\beta\rangle
             - \langle i^\beta|i^\beta\rangle \langle j^\alpha|j^\alpha\rangle
             + \langle i^\beta|i^\beta\rangle \langle j^\beta|j^\beta\rangle) \\
            &-\frac{1}{4}\sum_{ij}(\langle i^\alpha|j^\alpha\rangle \langle j^\alpha|i^\alpha\rangle
             - \langle i^\alpha|j^\alpha\rangle \langle j^\beta|i^\beta\rangle
             - \langle i^\beta|j^\beta\rangle \langle j^\alpha|i^\alpha\rangle
             + \langle i^\beta|j^\beta\rangle\langle j^\beta|i^\beta\rangle) \\
            &=\frac{1}{4}\sum_{ij}|\langle i^\alpha|i^\alpha\rangle - \langle i^\beta|i^\beta\rangle|^2
             -\frac{1}{4}\sum_{ij}|\langle i^\alpha|j^\alpha\rangle - \langle i^\beta|j^\beta\rangle|^2 \\
            &=\frac{1}{4}(n_\alpha - n_\beta)^2
             -\frac{1}{4}\sum_{ij}|\langle i^\alpha|j^\alpha\rangle - \langle i^\beta|j^\beta\rangle|^2

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
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % spin_square(mo, mol.intor('int1e_ovlp_sph')))
    S^2 = 0.7570150, 2S+1 = 2.0070027
    '''
    nao = mo.shape[0] // 2
    if isinstance(s, numpy.ndarray):
        assert(s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
        s = s[:nao,:nao]
    mo_a = mo[:nao]
    mo_b = mo[nao:]
    saa = reduce(numpy.dot, (mo_a.T.conj(), s, mo_a))
    sbb = reduce(numpy.dot, (mo_b.T.conj(), s, mo_b))
    sab = reduce(numpy.dot, (mo_a.T.conj(), s, mo_b))
    sba = sab.T.conj()
    nocc_a = saa.trace()
    nocc_b = sbb.trace()
    ssxy = (nocc_a+nocc_b) * .5
    ssxy+= sba.trace() * sab.trace() - numpy.einsum('ij,ji->', sba, sab)
    ssz  = (nocc_a+nocc_b) * .25
    ssz += (nocc_a-nocc_b)**2 * .25
    tmp  = saa - sbb
    ssz -= numpy.einsum('ij,ji', tmp, tmp) * .25
    ss = (ssxy + ssz).real
    s = numpy.sqrt(ss+.25) - .5
    return ss, s*2+1

def analyze(mf, verbose=logger.DEBUG, **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment
    '''
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, verbose)

    log.note('**** MO energy ****')
    for i,c in enumerate(mo_occ):
        log.note('MO #%-3d energy= %-18.15g occ= %g', i+1, mo_energy[i], c)
    ovlp_ao = mf.get_ovlp()
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return (mf.mulliken_meta(mf.mol, dm, s=ovlp_ao, verbose=log),
            mf.dip_moment(mf.mol, dm, verbose=log))

def mulliken_pop(mol, dm, s=None, verbose=logger.DEBUG):
    '''Mulliken population analysis
    '''
    nao = mol.nao_nr()
    dma = dm[:nao,:nao]
    dmb = dm[nao:,nao:]
    if s is not None:
        assert(s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
        s = s[:nao,:nao]
    return uhf.mulliken_pop(mol, (dma,dmb), s, verbose)

def mulliken_meta(mol, dm_ao, verbose=logger.DEBUG, pre_orth_method='ANO',
                  s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    nao = mol.nao_nr()
    dma = dm_ao[:nao,:nao]
    dmb = dm_ao[nao:,nao:]
    if s is not None:
        assert(s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
        s = s[:nao,:nao]
    return uhf.mulliken_meta(mol, (dma,dmb), verbose, pre_orth_method, s)

def det_ovlp(mo1, mo2, occ1, occ2, ovlp):
    r''' Calculate the overlap between two different determinants. It is the product
    of single values of molecular orbital overlap matrix.

    Return:
        A list:
            the product of single values: float
            x_a: :math:`\mathbf{U} \mathbf{\Lambda}^{-1} \mathbf{V}^\dagger`
            They are used to calculate asymmetric density matrix
    '''

    if numpy.sum(occ1) != numpy.sum(occ2):
        raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')
    s = reduce(numpy.dot, (mo1[:,occ1>0].T.conj(), ovlp, mo2[:,occ2>0]))
    u, s, vt = numpy.linalg.svd(s)
    x = numpy.dot(u/s, vt)
    return numpy.prod(s), x

def dip_moment(mol, dm, unit_symbol='Debye', verbose=logger.NOTE):
    nao = mol.nao_nr()
    dma = dm[:nao,:nao]
    dmb = dm[nao:,nao:]
    return hf.dip_moment(mol, dma+dmb, unit_symbol, verbose)

canonicalize = hf.canonicalize

class GHF(hf.SCF):
    __doc__ = hf.SCF.__doc__ + '''

    Attributes for GHF method
        GHF orbital coefficients are 2D array.  Let nao be the number of spatial
        AOs, mo_coeff[:nao] are the coefficients of AO with alpha spin;
        mo_coeff[nao:nao*2] are the coefficients of AO with beta spin.
    '''

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        hcore = hf.get_hcore(mol)
        return scipy.linalg.block_diag(hcore, hcore)

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        hcore = hf.get_ovlp(mol)
        return scipy.linalg.block_diag(hcore, hcore)

    get_occ = get_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        occidx = mo_occ > 0
        viridx = ~occidx
        g = reduce(numpy.dot, (mo_coeff[:,occidx].T.conj(), fock,
                               mo_coeff[:,viridx]))
        return g.T.ravel()

    def init_guess_by_minao(self, mol=None):
        return _from_rhf_init_dm(hf.SCF.init_guess_by_minao(self, mol))

    def init_guess_by_atom(self, mol=None):
        return _from_rhf_init_dm(hf.SCF.init_guess_by_atom(self, mol))

    def init_guess_by_chkfile(self, chkfile=None, project=True):
        dma, dmb = uhf.init_guess_by_chkfile(mol, chkfile, project)
        return scipy.linalg.block_diag(dma, dmb)

    def get_jk(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if mol.nao_nr() * 2 == dm[0].shape[0]:  # GHF density matrix, shape (2N,2N)
            return get_jk(mol, dm, hermi, True, True, self.get_jk)
        else:
            if self._eri is not None or mol.incore_anyway or self._is_mem_enough():
                if self._eri is None:
                    self._eri = mol.intor('int2e', aosym='s8')
                vj, vk = hf.dot_eri_dm(self._eri, dm, hermi)
            else:
                vj, vk = hf.SCF.get_jk(self, mol, dm, hermi)
            return vj, vk

    def get_j(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if mol.nao_nr() * 2 == dm[0].shape[0]:  # GHF density matrix, shape (2N,2N)
            return get_jk(mol, dm, hermi, True, False, self.get_jk)[0]
        else:
            return hf.SCF.get_j(self, mol, dm, hermi)

    def get_k(self, mol=None, dm=None, hermi=0):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if mol.nao_nr() * 2 == dm[0].shape[0]:  # GHF density matrix, shape (2N,2N)
            return get_jk(mol, dm, hermi, False, True, self.get_jk)[1]
        else:
            return hf.SCF.get_k(self, mol, dm, hermi)

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if (self._eri is not None or not self.direct_scf or
            mol.incore_anyway or self._is_mem_enough()):
            vj, vk = get_jk(mol, dm, hermi, True, True, self.get_jk)
            vhf = vj - vk
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = get_jk(mol, ddm, hermi, True, True, self.get_jk)
            vhf = vj - vk + numpy.asarray(vhf_last)
        return vhf

    def analyze(self, verbose=None, **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, **kwargs)

    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_pop(mol, dm, s=s, verbose=verbose)

    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method='ANO', s=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_meta(mol, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    @lib.with_doc(spin_square.__doc__)
    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff[:,self.mo_occ>0]
        if s is None: s = self.get_ovlp()
        return spin_square(mo_coeff, s)

    @lib.with_doc(det_ovlp.__doc__)
    def det_ovlp(self, mo1, mo2, occ1, occ2, ovlp=None):
        if ovlp is None: ovlp = self.get_ovlp()
        return det_ovlp(mo1, mo2, occ1, occ2, ovlp)

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, mol=None, dm=None, unit_symbol=None, verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if unit_symbol is None: unit_symbol='Debye'
        return dip_moment(mol, dm, unit_symbol, verbose=verbose)

    def _finalize(self):
        ss, s = self.spin_square()

        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g  '
                        '<S^2> = %.8g  2S+1 = %.8g', self.e_tot, ss, s)
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g after %d cycles  '
                        '<S^2> = %.8g  2S+1 = %.8g',
                        self.e_tot, self.max_cycle, ss, s)
        return self

    def stability(self, verbose=None):
        from pyscf.scf.stability import ghf_stability
        return ghf_stability(self, verbose)

def _from_rhf_init_dm(dm, breaksym=True):
    dma = dm * .5
    dm = scipy.linalg.block_diag(dma, dma)
    if breaksym:
        nao = dma.shape[0]
        idx, idy = numpy.diag_indices(nao)
        dm[idx+nao,idy] = dm[idx,idy+nao] = dma.diagonal() * .05
    return dm


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = 'H 0 0 0; H 0 0 1; O .5 .6 .2'
    mol.basis = 'ccpvdz'
    mol.build()

    mf = GHF(mol)
    mf.kernel()

    dm = mf.init_guess_by_1e(mol)
    dm = dm + 0j
    nao = mol.nao_nr()
    numpy.random.seed(12)
    dm[:nao,nao:] = numpy.random.random((nao,nao)) * .1j
    dm[nao:,:nao] = dm[:nao,nao:].T.conj()
    mf.kernel(dm)
    mf.canonicalize(mf.mo_coeff, mf.mo_occ)
    mf.analyze()
    print(mf.spin_square())
    print(mf.e_tot - -75.9125824421352)
