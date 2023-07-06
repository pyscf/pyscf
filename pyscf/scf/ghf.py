#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import uhf
from pyscf.scf import chkfile
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def init_guess_by_chkfile(mol, chkfile_name, project=None):
    '''Read SCF chkfile and make the density matrix for GHF initial guess.

    Kwargs:
        project : None or bool
            Whether to project chkfile's orbitals to the new basis.  Note when
            the geometry of the chkfile and the given molecule are very
            different, this projection can produce very poor initial guess.
            In PES scanning, it is recommended to swith off project.

            If project is set to None, the projection is only applied when the
            basis sets of the chkfile's molecule are different to the basis
            sets of the given molecule (regardless whether the geometry of
            the two molecules are different).  Note the basis sets are
            considered to be different if the two molecules are derived from
            the same molecule with different ordering of atoms.
    '''
    from pyscf.scf import addons
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)
    if project is None:
        project = not gto.same_basis_set(chk_mol, mol)

    # Check whether the two molecules are similar
    if abs(mol.inertia_moment() - chk_mol.inertia_moment()).sum() > 0.5:
        logger.warn(mol, "Large deviations found between the input "
                    "molecule and the molecule from chkfile\n"
                    "Initial guess density matrix may have large error.")

    if project:
        s = hf.get_ovlp(mol)

    def fproj(mo):
        if project:
            mo = addons.project_mo_nr2nr(chk_mol, mo, mol)
            norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
            mo /= numpy.sqrt(norm)
        return mo

    nao = chk_mol.nao_nr()
    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if getattr(mo[0], 'ndim', None) == 1:  # RHF/GHF/DHF
        if nao*2 == mo.shape[0]:  # GHF or DHF
            if project:
                raise NotImplementedError('Project initial guess from '
                                          'different geometry')
            else:
                dm = hf.make_rdm1(mo, mo_occ)
        else:  # RHF
            mo_coeff = fproj(mo)
            mo_occa = (mo_occ>1e-8).astype(numpy.double)
            mo_occb = mo_occ - mo_occa
            dma, dmb = uhf.make_rdm1([mo_coeff]*2, (mo_occa, mo_occb))
            dm = scipy.linalg.block_diag(dma, dmb)
    else: #UHF
        if getattr(mo[0][0], 'ndim', None) == 2:  # KUHF
            logger.warn(mol, 'k-point UHF results are found.  Density matrix '
                        'at Gamma point is used for the molecular SCF initial guess')
            mo = mo[0]
        dma, dmb = uhf.make_rdm1([fproj(mo[0]),fproj(mo[1])], mo_occ)
        dm = scipy.linalg.block_diag(dma, dmb)
    return dm


@lib.with_doc(hf.get_jk.__doc__)
def get_jk(mol, dm, hermi=0,
           with_j=True, with_k=True, jkbuild=hf.get_jk, omega=None):

    dm = numpy.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dms = dm.reshape(-1,nso,nso)
    n_dm = dms.shape[0]

    dmaa = dms[:,:nao,:nao]
    dmab = dms[:,:nao,nao:]
    dmbb = dms[:,nao:,nao:]
    if with_k:
        if hermi:
            dms = numpy.stack((dmaa, dmbb, dmab))
        else:
            dmba = dms[:,nao:,:nao]
            dms = numpy.stack((dmaa, dmbb, dmab, dmba))
        # Note the off-diagonal block breaks the hermitian
        _hermi = 0
    else:
        dms = numpy.stack((dmaa, dmbb))
        _hermi = 1

    j1, k1 = jkbuild(mol, dms, _hermi, with_j, with_k, omega)

    vj = vk = None
    if with_j:
        vj = numpy.zeros((n_dm,nso,nso), dm.dtype)
        vj[:,:nao,:nao] = vj[:,nao:,nao:] = j1[0] + j1[1]
        vj = vj.reshape(dm.shape)

    if with_k:
        vk = numpy.zeros((n_dm,nso,nso), dm.dtype)
        vk[:,:nao,:nao] = k1[0]
        vk[:,nao:,nao:] = k1[1]
        vk[:,:nao,nao:] = k1[2]
        if hermi:
            vk[:,nao:,:nao] = k1[2].conj().transpose(0,2,1)
        else:
            vk[:,nao:,:nao] = k1[3]
        vk = vk.reshape(dm.shape)

    return vj, vk

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx = numpy.argsort(mo_energy.round(9), kind='stable')
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
        assert (s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
        s = s[:nao,:nao]
    mo_a = mo[:nao]
    mo_b = mo[nao:]
    saa = reduce(numpy.dot, (mo_a.conj().T, s, mo_a))
    sbb = reduce(numpy.dot, (mo_b.conj().T, s, mo_b))
    sab = reduce(numpy.dot, (mo_a.conj().T, s, mo_b))
    sba = sab.conj().T
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

def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment
    '''
    log = logger.new_logger(mf, verbose)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    log.note('**** MO energy ****')
    for i,c in enumerate(mo_occ):
        log.note('MO #%-3d energy= %-18.15g occ= %g', i+MO_BASE, mo_energy[i], c)
    ovlp_ao = mf.get_ovlp()
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    dip = mf.dip_moment(mf.mol, dm, verbose=log)
    if with_meta_lowdin:
        pop_and_chg = mf.mulliken_meta(mf.mol, dm, s=ovlp_ao, verbose=log)
    else:
        pop_and_chg = mf.mulliken_pop(mf.mol, dm, s=ovlp_ao, verbose=log)
    return pop_and_chg, dip

def mulliken_pop(mol, dm, s=None, verbose=logger.DEBUG):
    '''Mulliken population analysis
    '''
    nao = mol.nao_nr()
    dma = dm[:nao,:nao]
    dmb = dm[nao:,nao:]
    if s is not None:
        assert (s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
        s = s[:nao,:nao]
    return uhf.mulliken_pop(mol, (dma,dmb), s, verbose)

def mulliken_meta(mol, dm_ao, verbose=logger.DEBUG,
                  pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    nao = mol.nao_nr()
    dma = dm_ao[:nao,:nao]
    dmb = dm_ao[nao:,nao:]
    if s is not None:
        assert (s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
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

def guess_orbspin(mo_coeff):
    '''Guess the orbital spin (alpha 0, beta 1, unknown -1) based on the
    orbital coefficients
    '''
    nao, nmo = mo_coeff.shape
    mo_a = mo_coeff[:nao//2]
    mo_b = mo_coeff[nao//2:]
    # When all coefficients on alpha AOs are close to 0, it's a beta orbital
    bidx = numpy.all(abs(mo_a) < 1e-14, axis=0)
    aidx = numpy.all(abs(mo_b) < 1e-14, axis=0)
    orbspin = numpy.empty(nmo, dtype=int)
    orbspin[:] = -1
    orbspin[aidx] = 0
    orbspin[bidx] = 1
    return orbspin

class GHF(hf.SCF):
    __doc__ = hf.SCF.__doc__ + '''

    Attributes for GHF method
        GHF orbital coefficients are 2D array.  Let nao be the number of spatial
        AOs, mo_coeff[:nao] are the coefficients of AO with alpha spin;
        mo_coeff[nao:nao*2] are the coefficients of AO with beta spin.
    '''

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.with_soc = None
        self._keys = self._keys.union(['with_soc'])

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        hcore = hf.get_hcore(mol)
        hcore = scipy.linalg.block_diag(hcore, hcore)

        if self.with_soc and mol.has_ecp_soc():
            # The ECP SOC contribution = <|1j * s * U_SOC|>
            s = .5 * lib.PauliMatrices
            ecpso = numpy.einsum('sxy,spq->xpyq', -1j * s, mol.intor('ECPso'))
            hcore = hcore + ecpso.reshape(hcore.shape)
        return hcore

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        s = hf.get_ovlp(mol)
        return scipy.linalg.block_diag(s, s)

    get_occ = get_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        occidx = mo_occ > 0
        viridx = ~occidx
        g = reduce(numpy.dot, (mo_coeff[:,occidx].T.conj(), fock,
                               mo_coeff[:,viridx]))
        return g.conj().T.ravel()

    get_init_guess = hf.RHF.get_init_guess

    @lib.with_doc(hf.SCF.init_guess_by_minao.__doc__)
    def init_guess_by_minao(self, mol=None):
        return _from_rhf_init_dm(hf.SCF.init_guess_by_minao(self, mol))

    @lib.with_doc(hf.SCF.init_guess_by_atom.__doc__)
    def init_guess_by_atom(self, mol=None):
        return _from_rhf_init_dm(hf.SCF.init_guess_by_atom(self, mol))

    @lib.with_doc(hf.SCF.init_guess_by_huckel.__doc__)
    def init_guess_by_huckel(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089.')
        return _from_rhf_init_dm(hf.init_guess_by_huckel(mol))

    @lib.with_doc(hf.SCF.init_guess_by_chkfile.__doc__)
    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project)

    @lib.with_doc(hf.get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        nao = mol.nao
        dm = numpy.asarray(dm)
        # nao = 0 for HF with custom Hamiltonian
        if dm.shape[-1] != nao * 2 and nao != 0:
            raise ValueError('Dimension inconsistent '
                             f'dm.shape = {dm.shape}, mol.nao = {nao}')

        def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
            return hf.RHF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)

        vj, vk = get_jk(mol, dm, hermi, with_j, with_k, jkbuild, omega)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
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
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
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
    def dip_moment(self, mol=None, dm=None, unit_symbol='Debye',
                   verbose=logger.NOTE):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
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

    def convert_from_(self, mf):
        '''Create GHF object based on the RHF/UHF object'''
        from pyscf.scf import addons
        return addons.convert_to_ghf(mf, out=self)

    def stability(self, internal=None, external=None, verbose=None, return_status=False):
        from pyscf.scf.stability import ghf_stability
        return ghf_stability(self, verbose, return_status)

    def nuc_grad_method(self):
        raise NotImplementedError

    def x2c1e(self):
        '''X2C with spin-orbit coupling effects.

        Note the difference to PySCF-1.7. In PySCF it calls spin-free X2C1E.
        This result (mol.GHF().x2c() ) should equal to mol.X2C() although they
        are solved in different AO basis (spherical GTO vs spinor GTO)
        '''
        from pyscf.x2c.x2c import x2c1e_ghf
        return x2c1e_ghf(self)
    x2c = x2c1e

def _from_rhf_init_dm(dm, breaksym=True):
    dma = dm * .5
    dm = scipy.linalg.block_diag(dma, dma)
    if breaksym:
        nao = dma.shape[0]
        idx, idy = numpy.diag_indices(nao)
        dm[idx+nao,idy] = dm[idx,idy+nao] = dma.diagonal() * .05
    return dm


class HF1e(GHF):
    scf = hf._hf1e_scf
