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


from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import chkfile
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def init_guess_by_minao(mol, breaksym=None):
    '''Generate initial guess density matrix based on ANO basis, then project
    the density matrix to the basis set defined by ``mol``

    Returns:
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    return UHF(mol).init_guess_by_minao(mol, breaksym)

def init_guess_by_1e(mol, breaksym=None):
    return UHF(mol).init_guess_by_1e(mol, breaksym)

def init_guess_by_atom(mol, breaksym=None):
    return UHF(mol).init_guess_by_atom(mol, breaksym)

def init_guess_by_huckel(mol, breaksym=None):
    return UHF(mol).init_guess_by_huckel(mol, breaksym)

def init_guess_by_mod_huckel(mol, breaksym=None):
    return UHF(mol).init_guess_by_mod_huckel(mol, breaksym)

def init_guess_by_sap(mol, sap_basis, breaksym=None, **kwargs):
    mf = UHF(mol)
    mf.sap_basis = sap_basis
    return mf.init_guess_by_sap(mol, breaksym)

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    '''Read SCF chkfile and make the density matrix for UHF initial guess.

    Kwargs:
        project : None or bool
            Whether to project chkfile's orbitals to the new basis.  Note when
            the geometry of the chkfile and the given molecule are very
            different, this projection can produce very poor initial guess.
            In PES scanning, it is recommended to switch off project.

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
    im1 = scipy.linalg.eigvalsh(mol.inertia_moment())
    im2 = scipy.linalg.eigvalsh(chk_mol.inertia_moment())
    # im1+1e-7 to avoid 'divide by zero' error
    if abs((im1-im2)/(im1+1e-7)).max() > 0.01:
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

    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if getattr(mo[0], 'ndim', None) == 1:  # RHF
        if numpy.iscomplexobj(mo):
            raise NotImplementedError('TODO: project DHF orbital to UHF orbital')
        mo_coeff = fproj(mo)
        mo_occa = (mo_occ>1e-8).astype(numpy.double)
        mo_occb = mo_occ - mo_occa
        dm = make_rdm1([mo_coeff,mo_coeff], [mo_occa,mo_occb])
    else:  #UHF
        if getattr(mo[0][0], 'ndim', None) == 2:  # KUHF
            logger.warn(mol, 'k-point UHF results are found.  Density matrix '
                        'at Gamma point is used for the molecular SCF initial guess')
            mo = mo[0]
        dm = make_rdm1([fproj(mo[0]),fproj(mo[1])], mo_occ)
    return dm

def _break_dm_spin_symm(mol, dm, breaksym=1):
    dma, dmb = dm
    # For spin polarized system, no need to manually break spin symmetry
    if breaksym and mol.spin == 0 and abs(dma - dmb).max() < 1e-2:
        if breaksym == 1:
            #remove off-diagonal part of beta DM
            dmb = numpy.zeros_like(dma)
            for b0, b1, p0, p1 in mol.aoslice_by_atom():
                dmb[...,p0:p1,p0:p1] = dma[...,p0:p1,p0:p1]
        else:
            # Adjust num. electrons for density matrices (issue #1839)
            # Get overlap matrix
            s1e = mol.intor_symmetric('int1e_ovlp')
            # Compute norm of density matrices
            nelec_half = numpy.einsum('ij,ji->', dma, s1e)
            # Scale density matrices to form doublet state
            dma = dma * (nelec_half+1) / nelec_half
            dmb = dmb * (nelec_half-1) / nelec_half
    return dma, dmb

def get_init_guess(mol, key='minao', **kwargs):
    return UHF(mol).get_init_guess(mol, key, **kwargs)

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix in AO representation

    Args:
        mo_coeff : tuple of 2D ndarrays
            Orbital coefficients for alpha and beta spins. Each column is one orbital.
        mo_occ : tuple of 1D ndarrays
            Occupancies for alpha and beta spins.
    Returns:
        A list of 2D ndarrays for alpha and beta spins
    '''
    mo_a = mo_coeff[0]
    mo_b = mo_coeff[1]
    dm_a = numpy.dot(mo_a*mo_occ[0], mo_a.conj().T)
    dm_b = numpy.dot(mo_b*mo_occ[1], mo_b.conj().T)
    return lib.tag_array((dm_a, dm_b), mo_coeff=mo_coeff, mo_occ=mo_occ)

def make_rdm2(mo_coeff, mo_occ):
    '''Two-particle density matrix in AO representation

    Args:
        mo_coeff : tuple of 2D ndarrays
            Orbital coefficients for alpha and beta spins. Each column is one orbital.
        mo_occ : tuple of 1D ndarrays
            Occupancies for alpha and beta spins.
    Returns:
        A tuple of three 4D ndarrays for alpha,alpha and alpha,beta and beta,beta spins
    '''
    dm1a, dm1b = make_rdm1(mo_coeff, mo_occ)
    dm2aa = (numpy.einsum('ij,kl->ijkl', dm1a, dm1a)
           - numpy.einsum('ij,kl->iklj', dm1a, dm1a))
    dm2bb = (numpy.einsum('ij,kl->ijkl', dm1b, dm1b)
           - numpy.einsum('ij,kl->iklj', dm1b, dm1b))
    dm2ab = numpy.einsum('ij,kl->ijkl', dm1a, dm1b)
    return dm2aa, dm2ab, dm2bb

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
    dm = numpy.asarray(dm)
    dm_last = numpy.asarray(dm_last)
    assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
    nao = dm.shape[-1]
    ddm = dm - dm_last
    # dm.reshape(-1,nao,nao) to remove first dim, compress (dma,dmb)
    vj, vk = hf.get_jk(mol, ddm.reshape(-1,nao,nao), hermi=hermi, vhfopt=vhfopt)
    vj = vj.reshape(dm.shape)
    vk = vk.reshape(dm.shape)
    assert (vj.ndim >= 3 and vj.shape[0] == 2)
    vhf = vj[0] + vj[1] - vk
    vhf += numpy.asarray(vhf_last)
    return vhf

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    f = numpy.asarray(h1e) + vhf
    if f.ndim == 2:
        f = (f, f)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = mf.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, numpy.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, numpy.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = [dm*.5] * 2
    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4 and fock_last is not None:
        f = (hf.damping(f[0], fock_last[0], dampa),
             hf.damping(f[1], fock_last[1], dampa))
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf, f_prev=fock_last)
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (hf.level_shift(s1e, dm[0], f[0], shifta),
             hf.level_shift(s1e, dm[1], f[1], shiftb))
    return numpy.array(f)

def get_occ(mf, mo_energy=None, mo_coeff=None):
    if mo_energy is None: mo_energy = mf.mo_energy
    e_idx_a = numpy.argsort(mo_energy[0])
    e_idx_b = numpy.argsort(mo_energy[1])
    e_sort_a = mo_energy[0][e_idx_a]
    e_sort_b = mo_energy[1][e_idx_b]
    nmo = mo_energy[0].size
    n_a, n_b = mf.nelec
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[0,e_idx_a[:n_a]] = 1
    mo_occ[1,e_idx_b[:n_b]] = 1
    if mf.verbose >= logger.INFO and n_a < nmo and n_b > 0 and n_b < nmo:
        if e_sort_a[n_a-1]+1e-3 > e_sort_a[n_a]:
            logger.warn(mf, 'alpha nocc = %d  HOMO %.15g >= LUMO %.15g',
                        n_a, e_sort_a[n_a-1], e_sort_a[n_a])
        else:
            logger.info(mf, '  alpha nocc = %d  HOMO = %.15g  LUMO = %.15g',
                        n_a, e_sort_a[n_a-1], e_sort_a[n_a])

        if e_sort_b[n_b-1]+1e-3 > e_sort_b[n_b]:
            logger.warn(mf, 'beta  nocc = %d  HOMO %.15g >= LUMO %.15g',
                        n_b, e_sort_b[n_b-1], e_sort_b[n_b])
        else:
            logger.info(mf, '  beta  nocc = %d  HOMO = %.15g  LUMO = %.15g',
                        n_b, e_sort_b[n_b-1], e_sort_b[n_b])

        if e_sort_a[n_a-1]+1e-3 > e_sort_b[n_b]:
            logger.warn(mf, 'system HOMO %.15g >= system LUMO %.15g',
                        e_sort_b[n_a-1], e_sort_b[n_b])

        numpy.set_printoptions(threshold=nmo)
        logger.debug(mf, '  alpha mo_energy =\n%s', mo_energy[0])
        logger.debug(mf, '  beta  mo_energy =\n%s', mo_energy[1])
        numpy.set_printoptions(threshold=1000)

    if mo_coeff is not None and mf.verbose >= logger.DEBUG:
        ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
        logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
    return mo_occ

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''UHF Gradients'''
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb

    ga = mo_coeff[0][:,viridxa].conj().T.dot(fock_ao[0].dot(mo_coeff[0][:,occidxa]))
    gb = mo_coeff[1][:,viridxb].conj().T.dot(fock_ao[1].dot(mo_coeff[1][:,occidxb]))
    return numpy.hstack((ga.ravel(), gb.ravel()))

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock

    Note this function has side effects which cause mf.scf_summary updated.

    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
    if dm is None: dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    if h1e[0].ndim < dm[0].ndim:  # get [0] because h1e and dm may not be ndarrays
        h1e = (h1e, h1e)
    e1 = numpy.einsum('ij,ji->', h1e[0], dm[0])
    e1+= numpy.einsum('ij,ji->', h1e[1], dm[1])
    e_coul =(numpy.einsum('ij,ji->', vhf[0], dm[0]) +
             numpy.einsum('ij,ji->', vhf[1], dm[1])) * .5
    e_elec = (e1 + e_coul).real
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
    return e_elec, e_coul

# mo_a and mo_b are occupied orbitals
def spin_square(mo, s=1):
    r'''Spin square and multiplicity of UHF determinant

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
            &-\frac{1}{4}(\langle i^\alpha|j^\alpha\rangle \langle j^\alpha|i^\alpha\rangle
             + \langle i^\beta|j^\beta\rangle\langle j^\beta|i^\beta\rangle) \\
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
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % spin_square(mo, mol.intor('int1e_ovlp_sph')))
    S^2 = 0.7570150, 2S+1 = 2.0070027
    '''
    mo_a, mo_b = mo
    nocc_a = mo_a.shape[1]
    nocc_b = mo_b.shape[1]
    s = reduce(numpy.dot, (mo_a.conj().T, s, mo_b))
    ssxy = (nocc_a+nocc_b) * .5 - numpy.einsum('ij,ij->', s.conj(), s)
    ssz = (nocc_b-nocc_a)**2 * .25
    ss = (ssxy + ssz).real
    s = numpy.sqrt(ss+.25) - .5
    return ss, s*2+1

def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment;
    Spin density for AOs and atoms;
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    nmo = len(mo_occ[0])
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        mf.dump_scf_summary(log)

        log.note('**** MO energy ****')
        log.note('                             alpha | beta                alpha | beta')
        for i in range(nmo):
            log.note('MO #%-3d energy= %-18.15g | %-18.15g occ= %g | %g',
                     i+MO_BASE, mo_energy[0][i], mo_energy[1][i],
                     mo_occ[0][i], mo_occ[1][i])

    ovlp_ao = mf.get_ovlp()
    if log.verbose >= logger.DEBUG:
        label = mf.mol.ao_labels()
        if with_meta_lowdin:
            log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) for alpha spin **')
            orth_coeff = orth.orth_ao(mf.mol, 'meta_lowdin', s=ovlp_ao)
            c_inv = numpy.dot(orth_coeff.conj().T, ovlp_ao)
            dump_mat.dump_rec(mf.stdout, c_inv.dot(mo_coeff[0]), label,
                              start=MO_BASE, **kwargs)
            log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) for beta spin **')
            dump_mat.dump_rec(mf.stdout, c_inv.dot(mo_coeff[1]), label,
                              start=MO_BASE, **kwargs)
        else:
            log.debug(' ** MO coefficients (expansion on AOs) for alpha spin **')
            dump_mat.dump_rec(mf.stdout, mo_coeff[0], label,
                              start=MO_BASE, **kwargs)
            log.debug(' ** MO coefficients (expansion on AOs) for beta spin **')
            dump_mat.dump_rec(mf.stdout, mo_coeff[1], label,
                              start=MO_BASE, **kwargs)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        log.note("\nTo work with the spin densities directly, `use mulliken_meta_spin()` only printing them here.\n")
        mulliken_meta_spin(mf.mol, dm, s=ovlp_ao, verbose=log)
        return (mf.mulliken_meta(mf.mol, dm, s=ovlp_ao, verbose=log),
                mf.dip_moment(mf.mol, dm, verbose=log))
    else:
        log.note("\nTo work with the spin densities directly, `use mulliken_spin_pop()` only printing them here.\n")
        mulliken_spin_pop(mf.mol, dm, s=ovlp_ao, verbose=log)
        return (mf.mulliken_pop(mf.mol, dm, s=ovlp_ao, verbose=log),
                mf.dip_moment(mf.mol, dm, verbose=log))

def mulliken_pop(mol, dm, s=None, verbose=logger.DEBUG):
    '''Mulliken population analysis
    '''
    if s is None: s = hf.get_ovlp(mol)
    log = logger.new_logger(mol, verbose)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    pop_a = numpy.einsum('ij,ji->i', dm[0], s).real
    pop_b = numpy.einsum('ij,ji->i', dm[1], s).real

    log.info(' ** Mulliken pop       alpha | beta **')
    for i, s in enumerate(mol.ao_labels()):
        log.info('pop of  %s %10.5f | %-10.5f',
                 s, pop_a[i], pop_b[i])
    log.info('In total          %10.5f | %-10.5f', sum(pop_a), sum(pop_b))

    log.note(' ** Mulliken atomic charges   ( Nelec_alpha | Nelec_beta ) **')
    nelec_a = numpy.zeros(mol.natm)
    nelec_b = numpy.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        nelec_a[s[0]] += pop_a[i]
        nelec_b[s[0]] += pop_b[i]
    chg = mol.atom_charges() - (nelec_a + nelec_b)
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note('charge of  %d%s =   %10.5f  (  %10.5f   %10.5f )',
                 ia, symb, chg[ia], nelec_a[ia], nelec_b[ia])
    return (pop_a,pop_b), chg

def mulliken_spin_pop(mol, dm, s=None, verbose=logger.DEBUG):
    r'''Mulliken spin density analysis

    See Eq. 80 in https://arxiv.org/pdf/1206.2234.pdf and the surrounding
    text for more details.

    .. math:: M_{ij} = (D^a_{ij} - D^b_{ij}) S_{ji}

    Mulliken charges

    .. math:: \delta_i = \sum_j M_{ij}

    Returns:
        A list : spin_pop, Ms

        spin_pop : nparray
            Mulliken spin density on each atomic orbitals
        Ms : nparray
            Mulliken spin density on each atom
    '''
    if s is None: s = hf.get_ovlp(mol)

    dma = dm[0]
    dmb = dm[1]

    M = dma - dmb # Spin density

    log = logger.new_logger(mol, verbose)

    spin_pop = numpy.einsum('ij,ji->i', M, s).real

    log.info(' ** Mulliken Spin Density (per AO)  **')
    for i, s in enumerate(mol.ao_labels()):
        log.info('spin_pop of  %s %10.5f', s, spin_pop[i])

    log.note(' ** Mulliken Spin Density (per atom)  **')
    Ms = numpy.zeros(mol.natm) # Spin density per atom
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        Ms[s[0]] += spin_pop[i]

    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note('spin density of  %d %s =   %10.5f',
                 ia, symb, Ms[ia])

    return spin_pop, Ms

def mulliken_meta(mol, dm_ao, verbose=logger.DEBUG,
                  pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    from pyscf.lo import orth
    if s is None: s = hf.get_ovlp(mol)
    log = logger.new_logger(mol, verbose)
    if isinstance(dm_ao, numpy.ndarray) and dm_ao.ndim == 2:
        dm_ao = numpy.array((dm_ao*.5, dm_ao*.5))
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_method, s=s)
    c_inv = numpy.dot(orth_coeff.conj().T, s)
    dm_a = reduce(numpy.dot, (c_inv, dm_ao[0], c_inv.conj().T))
    dm_b = reduce(numpy.dot, (c_inv, dm_ao[1], c_inv.conj().T))

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mulliken_pop(mol, (dm_a,dm_b), numpy.eye(orth_coeff.shape[0]), log)
mulliken_pop_meta_lowdin_ao = mulliken_meta

def mulliken_meta_spin(mol, dm_ao, verbose=logger.DEBUG,
                       pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''Mulliken spin population analysis, based on meta-Lowdin AOs.
    '''
    from pyscf.lo import orth
    if s is None: s = hf.get_ovlp(mol)
    log = logger.new_logger(mol, verbose)
    if isinstance(dm_ao, numpy.ndarray) and dm_ao.ndim == 2:
        dm_ao = numpy.array((dm_ao*.5, dm_ao*.5))
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_method, s=s)
    c_inv = numpy.dot(orth_coeff.conj().T, s)
    dm_a = reduce(numpy.dot, (c_inv, dm_ao[0], c_inv.conj().T))
    dm_b = reduce(numpy.dot, (c_inv, dm_ao[1], c_inv.conj().T))

    log.note(' ** Mulliken spin pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mulliken_spin_pop(mol, (dm_a,dm_b), numpy.eye(orth_coeff.shape[0]), log)
mulliken_spin_pop_meta_lowdin_ao = mulliken_meta_spin


def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    mo_occ = numpy.asarray(mo_occ)
    assert (mo_occ.ndim == 2)
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_fock(dm=dm)
    occidxa = mo_occ[0] == 1
    occidxb = mo_occ[1] == 1
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0

    def eig_(fock, mo_coeff, idx, es, cs):
        if numpy.count_nonzero(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
            e, c = scipy.linalg.eigh(f1)
            es[idx] = e
            cs[:,idx] = numpy.dot(orb, c)

    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.shape)
    eig_(fock[0], mo_coeff[0], occidxa, mo_e[0], mo[0])
    eig_(fock[0], mo_coeff[0], viridxa, mo_e[0], mo[0])
    eig_(fock[1], mo_coeff[1], occidxb, mo_e[1], mo[1])
    eig_(fock[1], mo_coeff[1], viridxb, mo_e[1], mo[1])
    return mo_e, mo

def det_ovlp(mo1, mo2, occ1, occ2, ovlp):
    r''' Calculate the overlap between two different determinants. It is the product
    of single values of molecular orbital overlap matrix.

    .. math::

        S_{12} = \langle \Psi_A | \Psi_B \rangle
        = (\mathrm{det}\mathbf{U}) (\mathrm{det}\mathbf{V^\dagger})
          \prod\limits_{i=1}\limits^{2N} \lambda_{ii}

    where :math:`\mathbf{U}, \mathbf{V}, \lambda` are unitary matrices and single
    values generated by single value decomposition(SVD) of the overlap matrix
    :math:`\mathbf{O}` which is the overlap matrix of two sets of molecular orbitals:

    .. math::

        \mathbf{U}^\dagger \mathbf{O} \mathbf{V} = \mathbf{\Lambda}

    Args:
        mo1, mo2 : 2D ndarrays
            Molecualr orbital coefficients
        occ1, occ2: 2D ndarrays
            occupation numbers

    Return:
        A list:
            the product of single values: float
            (x_a, x_b): 1D ndarrays
            :math:`\mathbf{U} \mathbf{\Lambda}^{-1} \mathbf{V}^\dagger`
            They are used to calculate asymmetric density matrix
    '''
    c1_a = mo1[0][:, occ1[0]>0]
    c1_b = mo1[1][:, occ1[1]>0]
    c2_a = mo2[0][:, occ2[0]>0]
    c2_b = mo2[1][:, occ2[1]>0]
    if c1_a.shape[1] != c2_a.shape[1] or c1_b.shape[1] != c2_b.shape[1]:
        raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')

    o_a = reduce(numpy.dot, (c1_a.conj().T, ovlp, c2_a))
    o_b = reduce(numpy.dot, (c1_b.conj().T, ovlp, c2_b))
    u_a, s_a, vt_a = numpy.linalg.svd(o_a)
    u_b, s_b, vt_b = numpy.linalg.svd(o_b)
    x_a = reduce(numpy.dot, (u_a*numpy.reciprocal(s_a), vt_a))
    x_b = reduce(numpy.dot, (u_b*numpy.reciprocal(s_b), vt_b))
    return numpy.prod(s_a)*numpy.prod(s_b), (x_a, x_b)

def make_asym_dm(mo1, mo2, occ1, occ2, x):
    r'''One-particle asymmetric density matrix

    Args:
        mo1, mo2 : 2D ndarrays
            Molecualr orbital coefficients
        occ1, occ2: 2D ndarrays
            Occupation numbers
        x: 2D ndarrays
            :math:`\mathbf{U} \mathbf{\Lambda}^{-1} \mathbf{V}^\dagger`.
            See also :func:`det_ovlp`

    Return:
        A list of 2D ndarrays for alpha and beta spin

    Examples:

    >>> mf1 = scf.UHF(gto.M(atom='H 0 0 0; F 0 0 1.3', basis='ccpvdz')).run()
    >>> mf2 = scf.UHF(gto.M(atom='H 0 0 0; F 0 0 1.4', basis='ccpvdz')).run()
    >>> s = gto.intor_cross('int1e_ovlp_sph', mf1.mol, mf2.mol)
    >>> det, x = det_ovlp(mf1.mo_coeff, mf1.mo_occ, mf2.mo_coeff, mf2.mo_occ, s)
    >>> adm = make_asym_dm(mf1.mo_coeff, mf1.mo_occ, mf2.mo_coeff, mf2.mo_occ, x)
    >>> adm.shape
    (2, 19, 19)
    '''

    mo1_a = mo1[0][:, occ1[0]>0]
    mo1_b = mo1[1][:, occ1[1]>0]
    mo2_a = mo2[0][:, occ2[0]>0]
    mo2_b = mo2[1][:, occ2[1]>0]
    dm_a = reduce(numpy.dot, (mo1_a, x[0], mo2_a.conj().T))
    dm_b = reduce(numpy.dot, (mo1_b, x[1], mo2_b.conj().T))
    return numpy.array((dm_a, dm_b))

dip_moment = hf.dip_moment

class UHF(hf.SCF):
    __doc__ = hf.SCF.__doc__ + '''
    Attributes for UHF:
        nelec : (int, int)
            If given, freeze the number of (alpha,beta) electrons to the given value.
        level_shift : number or two-element list
            level shift (in Eh) for alpha and beta Fock if two-element list is given.
        init_guess_breaksym : int
             This configuration controls the algorithm used to break the spin
             symmetry of the initial guess:
             - 0 to disable symmetry breaking in the initial guess.
             - 1 to use the default algorithm introduced in pyscf-1.7.
             - 2 to adjust the num. electrons for spin-up and spin-down density matrices (issue #1839).

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', charge=1, spin=1, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> mf.kernel()
    -75.623975516256706
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % mf.spin_square())
    S^2 = 0.7570150, 2S+1 = 2.0070027
    '''

    init_guess_breaksym = getattr(__config__, 'scf_uhf_init_guess_breaksym', 1)

    _keys = {"init_guess_breaksym"}

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        # self.mo_coeff => [mo_a, mo_b]
        # self.mo_occ => [mo_occ_a, mo_occ_b]
        # self.mo_energy => [mo_energy_a, mo_energy_b]
        self.nelec = None

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            return self.mol.nelec
    @nelec.setter
    def nelec(self, x):
        self._nelec = x

    @property
    def nelectron_alpha(self):
        return self.nelec[0]
    @nelectron_alpha.setter
    def nelectron_alpha(self, x):
        logger.warn(self, 'WARN: Attribute .nelectron_alpha is deprecated. '
                    'Set .nelec instead')
        #raise RuntimeError('API updates')
        self.nelec = (x, self.mol.nelectron-x)

    def dump_flags(self, verbose=None):
        hf.SCF.dump_flags(self, verbose)
        logger.info(self, 'number electrons alpha = %d  beta = %d', *self.nelec)

    def eig(self, fock, s):
        e_a, c_a = self._eigh(fock[0], s)
        e_b, c_b = self._eigh(fock[1], s)
        return numpy.array((e_a,e_b)), numpy.array((c_a,c_b))

    get_fock = get_fock

    get_occ = get_occ

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    @lib.with_doc(make_rdm1.__doc__)
    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    @lib.with_doc(make_rdm2.__doc__)
    def make_rdm2(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return make_rdm2(mo_coeff, mo_occ, **kwargs)

    energy_elec = energy_elec

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        dm = hf.SCF.get_init_guess(self, mol, key, **kwargs)
        if self.verbose >= logger.DEBUG1:
            s = self.get_ovlp()
            nelec =(numpy.einsum('ij,ji', dm[0], s).real,
                    numpy.einsum('ij,ji', dm[1], s).real)
            logger.debug1(self, 'Nelec from initial guess = %s', nelec)
        return dm

    def init_guess_by_minao(self, mol=None, breaksym=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        if breaksym is None: breaksym = self.init_guess_breaksym
        # For spin polarized system, no need to manually break spin symmetry
        dm = hf.init_guess_by_minao(mol)
        dma = dmb = dm*.5
        dma, dmb = _break_dm_spin_symm(mol, (dma, dmb), breaksym)
        return numpy.array((dma, dmb))

    def init_guess_by_atom(self, mol=None, breaksym=None):
        if mol is None: mol = self.mol
        if breaksym is None: breaksym = self.init_guess_breaksym
        dm = hf.init_guess_by_atom(mol)
        dma = dmb = dm*.5
        if mol.spin == 0 and breaksym:
            if breaksym == 1:
                #Add off-diagonal part for alpha DM
                dma = mol.intor_symmetric('int1e_ovlp') * 1e-2
                for b0, b1, p0, p1 in mol.aoslice_by_atom():
                    dma[p0:p1,p0:p1] = dmb[p0:p1,p0:p1]
            else:
                dma, dmb = _break_dm_spin_symm(mol, (dma, dmb), breaksym)
        return numpy.array((dma,dmb))

    def init_guess_by_huckel(self, mol=None, breaksym=None):
        if mol is None: mol = self.mol
        if breaksym is None: breaksym = self.init_guess_breaksym
        logger.info(self, 'Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089.')
        mo_energy, mo_coeff = hf._init_guess_huckel_orbitals(mol, updated_rule = False)
        mo_energy = (mo_energy, mo_energy)
        mo_coeff = (mo_coeff, mo_coeff)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        dma, dmb = self.make_rdm1(mo_coeff, mo_occ)
        if breaksym:
            dma, dmb = _break_dm_spin_symm(mol, (dma, dmb))
        return numpy.array((dma,dmb))

    def init_guess_by_mod_huckel(self, mol=None, breaksym=None):
        if mol is None: mol = self.mol
        if breaksym is None: breaksym = self.init_guess_breaksym
        logger.info(self, '''Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089,
employing the updated GWH rule from doi:10.1021/ja00480a005.''')
        mo_energy, mo_coeff = hf._init_guess_huckel_orbitals(mol, updated_rule = True)
        mo_energy = (mo_energy, mo_energy)
        mo_coeff = (mo_coeff, mo_coeff)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        dma, dmb = self.make_rdm1(mo_coeff, mo_occ)
        if breaksym:
            dma, dmb = _break_dm_spin_symm(mol, (dma, dmb), breaksym)
        return numpy.array((dma,dmb))

    def init_guess_by_1e(self, mol=None, breaksym=None):
        if mol is None: mol = self.mol
        if breaksym is None: breaksym = self.init_guess_breaksym
        logger.info(self, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        if isinstance(h1e, numpy.ndarray) and h1e.ndim == s1e.ndim:
            h1e = (h1e, h1e)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        dma, dmb = self.make_rdm1(mo_coeff, mo_occ)
        natm = getattr(mol, 'natm', 0)  # handle custom Hamiltonian
        if natm > 0 and breaksym:
            dma, dmb = _break_dm_spin_symm(mol, (dma, dmb), breaksym)
        return numpy.array((dma,dmb))

    def init_guess_by_sap(self, mol=None, breaksym=None, **kwargs):
        from pyscf.gto.basis import load
        if mol is None: mol = self.mol
        if breaksym is None: breaksym = self.init_guess_breaksym
        sap_basis = self.sap_basis
        logger.info(self, '''Initial guess from superposition of atomic potentials (doi:10.1021/acs.jctc.8b01089)
This is the Gaussian fit version as described in doi:10.1063/5.0004046.''')
        if isinstance(sap_basis, str):
            atoms = [coord[0] for coord in mol._atom]
            sapbas = {}
            for atom in set(atoms):
                single_element_bs = load(sap_basis, atom)
                if isinstance(single_element_bs, dict):
                    sapbas[atom] = numpy.asarray(single_element_bs[atom][0][1:], dtype=float)
                else:
                    sapbas[atom] = numpy.asarray(single_element_bs[0][1:], dtype=float)
            logger.note(self, f'Found SAP basis {sap_basis.split("/")[-1]}')
        elif isinstance(sap_basis, dict):
            sapbas = {}
            for key in sap_basis:
                sapbas[key] = numpy.asarray(sap_basis[key][0][1:], dtype=float)
        else:
            raise RuntimeError('sap_basis is of an unexpected datatype.')
        dm = hf.init_guess_by_sap(mol, sapbas)
        dma = dmb = dm*.5
        dma, dmb = _break_dm_spin_symm(mol, (dma, dmb), breaksym)
        return numpy.array((dma,dmb))

    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        '''Coulomb (J) and exchange (K)

        Args:
            dm : a list of 2D arrays or a list of 3D arrays
                (alpha_dm, beta_dm) or (alpha_dms, beta_dms)
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if (not omega and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            vj, vk = hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = hf.SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
        return vj, vk

    @lib.with_doc(get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            logger.warn(self, 'Incompatible dm dimension. Treat dm as RHF density matrix.')
            dm = numpy.repeat(dm[None]*.5, 2, axis=0)
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj[0] + vj[1] - vk
        else:
            dm_last = numpy.asarray(dm_last)
            dm = numpy.asarray(dm)
            assert dm_last.ndim == 0 or dm_last.ndim == dm.ndim
            ddm = dm - dm_last
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj[0] + vj[1] - vk
            vhf += numpy.asarray(vhf_last)
        return vhf

    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, with_meta_lowdin, **kwargs)

    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_pop(mol, dm, s=s, verbose=verbose)

    def mulliken_spin_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_spin_pop(mol, dm, s=s, verbose=verbose)

    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_meta(mol, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    def mulliken_meta_spin(self, mol=None, dm=None, verbose=logger.DEBUG,
                           pre_orth_method=PRE_ORTH_METHOD, s=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_meta_spin(mol, dm, s=s, verbose=verbose,
                                  pre_orth_method=pre_orth_method)

    @lib.with_doc(spin_square.__doc__)
    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        if s is None:
            s = self.get_ovlp()
        return spin_square(mo_coeff, s)

    canonicalize = canonicalize

    @lib.with_doc(det_ovlp.__doc__)
    def det_ovlp(self, mo1, mo2, occ1, occ2, ovlp=None):
        if ovlp is None: ovlp = self.get_ovlp()
        return det_ovlp(mo1, mo2, occ1, occ2, ovlp)

    @lib.with_doc(make_asym_dm.__doc__)
    def make_asym_dm(self, mo1, mo2, occ1, occ2, x):
        return make_asym_dm(mo1, mo2, occ1, occ2, x)

    def _finalize(self):
        if self.mo_coeff is None or self.mo_occ is None:
            # Skip spin_square (issue #1574)
            return hf.SCF._finalize(self)

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
        '''Create UHF object based on the RHF/ROHF object'''
        tgt = mf.to_uhf()
        self.__dict__.update(tgt.__dict__)
        return self

    def stability(self,
                  internal=getattr(__config__, 'scf_stability_internal', True),
                  external=getattr(__config__, 'scf_stability_external', False),
                  verbose=None,
                  return_status=False,
                  **kwargs):
        '''
        Stability analysis for UHF/UKS method.

        See also pyscf.scf.stability.uhf_stability function.

        Args:
            mf : UHF or UKS object

        Kwargs:
            internal : bool
                Internal stability, within the UHF space.
            external : bool
                External stability. Including the UHF -> GHF and real -> complex
                stability analysis.
            return_status: bool
                Whether to return `stable_i` and `stable_e`

        Returns:
            If return_status is False (default), the return value includes
            two set of orbitals, which are more close to the stable condition.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.

            Else, another two boolean variables (indicating current status:
            stable or unstable) are returned.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.
        '''
        from pyscf.scf.stability import uhf_stability
        return uhf_stability(self, internal, external, verbose, return_status, **kwargs)

    def nuc_grad_method(self):
        from pyscf.grad import uhf
        return uhf.Gradients(self)

    def to_ks(self, xc='HF'):
        '''Convert to UKS object.
        '''
        from pyscf import dft
        return self._transfer_attrs_(dft.UKS(self.mol, xc=xc))

    to_gpu = lib.to_gpu

def _hf1e_scf(mf, *args):
    logger.info(mf, '\n')
    logger.info(mf, '******** 1 electron system ********')
    mf.converged = True
    h1e = mf.get_hcore(mf.mol)
    s1e = mf.get_ovlp(mf.mol)
    if isinstance(h1e, numpy.ndarray) and h1e.ndim == s1e.ndim:
        h1e = (h1e, h1e)
    mf.mo_energy, mf.mo_coeff = mf.eig(h1e, s1e)
    mf.mo_occ = mf.get_occ(mf.mo_energy, mf.mo_coeff)
    mf.e_tot = mf.mo_energy[mf.mo_occ>0][0].real + mf.mol.energy_nuc()
    mf._finalize()
    return mf.e_tot

class HF1e(UHF):
    scf = _hf1e_scf

    def spin_square(self, mo_coeff=None, s=None):
        return .75, 2

del (WITH_META_LOWDIN, PRE_ORTH_METHOD)
