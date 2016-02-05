#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import pyscf.lib
from pyscf.lib import logger
import pyscf.fci
from pyscf import scf
from pyscf import symm


def sort_mo(casscf, mo_coeff, caslst, base=1):
    '''Pick orbitals for CAS space

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        mo_coeff : ndarray or a list of ndarray
            Orbitals for CASSCF initial guess.  In the UHF-CASSCF, it's a list
            of two orbitals, for alpha and beta spin.
        caslst : list of int or nested list of int
            A list of orbital indices to represent the CAS space.  In the UHF-CASSCF,
            it's consist of two lists, for alpha and beta spin.

    Kwargs:
        base : int
            0-based (C-like) or 1-based (Fortran-like) caslst

    Returns:
        An reoreded mo_coeff, which put the orbitals given by caslst in the CAS space

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 4, 4)
    >>> cas_list = [5,6,8,9] # pi orbitals
    >>> mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list)
    >>> mc.kernel(mo)[0]
    -109.007378939813691
    '''
    ncore = casscf.ncore
    if isinstance(ncore, (int, numpy.integer)):
        assert(casscf.ncas == len(caslst))
        nmo = mo_coeff.shape[1]
        if base != 0:
            caslst = [i-base for i in caslst]
        idx = numpy.asarray([i for i in range(nmo) if i not in caslst])
        return numpy.hstack((mo_coeff[:,idx[:ncore]],
                             mo_coeff[:,caslst],
                             mo_coeff[:,idx[ncore:]]))
    else: # UHF-based CASSCF
        if isinstance(caslst[0], (int, numpy.integer)):
            assert(casscf.ncas == len(caslst))
            if base != 0:
                caslsta = [i-1 for i in caslst]
                caslst = (caslsta, caslsta)
        else: # two casspace lists, for alpha and beta
            assert(casscf.ncas == len(caslst[0]))
            assert(casscf.ncas == len(caslst[1]))
            if base != 0:
                caslst = ([i-base for i in caslst[0]],
                          [i-base for i in caslst[1]])
        nmo = mo_coeff[0].shape[1]
        idx = numpy.asarray([i for i in range(nmo) if i not in caslst[0]])
        mo_a = numpy.hstack((mo_coeff[0][:,idx[:ncore[0]]],
                             mo_coeff[0][:,caslst[0]],
                             mo_coeff[0][:,idx[ncore[0]:]]))
        idx = numpy.asarray([i for i in range(nmo) if i not in caslst[1]])
        mo_b = numpy.hstack((mo_coeff[1][:,idx[:ncore[1]]],
                             mo_coeff[1][:,caslst[1]],
                             mo_coeff[1][:,idx[ncore[1]:]]))
        return (mo_a, mo_b)

def select_mo_by_irrep(casscf,  cas_occ_num, mo = None, base=1):
    if mo is None:
        mo = casscf.mo_coeff
    orbsym = pyscf.symm.label_orb_symm(casscf.mol, casscf.mol.irrep_id,
                                                casscf.mol.symm_orb,
                                                mo, s=casscf._scf.get_ovlp())
    orbsym = orbsym[casscf.ncore:]
    caslst = []
    for k, v in cas_occ_num.iteritems():
        orb_irrep = [ casscf.ncore + base + i for i in range(len(orbsym)) if orbsym[i]== symm.irrep_name2id(casscf.mol.groupname,k) ]
        caslst.extend(orb_irrep[:v])
    return caslst

def sort_mo_by_irrep(casscf, mo_coeff, cas_irrep_nocc,
                     cas_irrep_ncore=None, s=None):
    '''Given number of active orbitals for each irrep, form the active space
    wrt the indices of MOs

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        cas_irrep_nocc : list or dict
            Number of active orbitals for each irrep.  It can be a dict, eg
            {'A1': 2, 'B2': 4} to indicate the active space size based on
            irrep names, or {0: 2, 3: 4} for irrep Id,  or a list [2, 0, 0, 4]
            (identical to {0: 2, 3: 4}) in which the list index is served as
            the irrep Id.

    Kwargs:
        cas_irrep_ncore : list or dict
            Number of closed shells for each irrep.  It can be a dict, eg
            {'A1': 6, 'B2': 4} to indicate the closed shells based on
            irrep names, or {0: 6, 3: 4} for irrep Id,  or a list [6, 0, 0, 4]
            (identical to {0: 6, 3: 4}) in which the list index is served as
            the irrep Id.  If cas_irrep_ncore is not given, the program
            will generate a guess based on the lowest :attr:`CASCI.ncore`
            orbitals.
        s : ndarray
            overlap matrix

    Returns:
        sorted orbitals, ordered as [c,..,c,a,..,a,v,..,v]

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvtz', symmetry=True, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.kernel()
    >>> mc = mcscf.CASSCF(mf, 12, 4)
    >>> mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, {'E1gx':4, 'E1gy':4, 'E1ux':2, 'E1uy':2})
    >>> # identical to mo = sort_mo_by_irrep(mc, mf.mo_coeff, {2: 4, 3: 4, 6: 2, 7: 2})
    >>> # identical to mo = sort_mo_by_irrep(mc, mf.mo_coeff, [0, 0, 4, 4, 0, 0, 2, 2])
    >>> mc.kernel(mo)[0]
    -108.162863845084
    '''
    if s is None:
        s = casscf._scf.get_ovlp()
    orbsym = pyscf.symm.label_orb_symm(casscf.mol, casscf.mol.irrep_id,
                                       casscf.mol.symm_orb, mo_coeff, s)
    orbsym = numpy.asarray(orbsym)
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    irreps = set(orbsym)
    orbidx_by_irrep = {}
    for ir in irreps:
        orbidx_by_irrep[ir] = numpy.where(orbsym == ir)[0]

    irrep_ncore = dict([(ir, sum(orbsym[:ncore]==ir)) for ir in irreps])
    if cas_irrep_ncore is not None:
        for k, n in cas_irrep_ncore.iteritems():
            if isinstance(k, str):
                irid = symm.irrep_name2id(casscf.mol.groupname, k)
            else:
                irid = k
            irrep_ncore[irid] = n

    if not isinstance(cas_irrep_nocc, dict):
        # list => dict
        cas_irrep_nocc = dict([(ir, n) for ir,n in enumerate(cas_irrep_nocc)
                               if n > 0])
    irrep_ncas = {}
    count = 0
    for k, n in cas_irrep_nocc.iteritems():
        if isinstance(k, str):
            irid = symm.irrep_name2id(casscf.mol.groupname, k)
        else:
            irid = k
        irrep_ncas[irid] = n
        count += n
    if count != casscf.ncas:
        raise ValueError('Active space size %d != specified active space %s'
                         % (casscf.ncas, cas_irrep_nocc))

    caslst = []
    for ir in irreps:
        nc = irrep_ncore[ir]
        if ir in irrep_ncas:
            no = nc + irrep_ncas[ir]
        else:
            no = nc
        caslst.extend(orbidx_by_irrep[ir][nc:no])

    logger.debug(casscf, 'ncore for each irreps %s', irrep_ncore)
    logger.debug(casscf, 'ncas for each irreps %s', irrep_ncas)
    logger.debug(casscf, 'caslst = %s', caslst)
    return sort_mo(casscf, mo_coeff, sorted(caslst), 0)


def project_init_guess(casscf, init_mo, prev_mol=None):
    '''Project the given initial guess to the current CASSCF problem.  The
    projected initial guess has two parts.  The core orbitals are directly
    taken from the Hartree-Fock orbitals, and the active orbitals are
    projected from the given initial guess.

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        init_mo : ndarray or list of ndarray
            Initial guess orbitals which are not orth-normal for the current
            molecule.  When the casscf is UHF-CASSCF, the init_mo needs to be
            a list of two ndarrays, for alpha and beta orbitals

    Kwargs:
        prev_mol : an instance of :class:`Mole`
            If given, the inital guess orbitals are expanded on the geometry
            specified by prev_mol.  Otherwise, the orbitals are expanded on
            current geometry specified by casscf.mol

    Returns:
        New orthogonal initial guess orbitals with the core taken from
        Hartree-Fock orbitals and projected active space from original initial
        guess orbitals

    Examples:

    .. code:: python

        import numpy
        from pyscf import gto, scf, mcscf
        mol = gto.Mole()
        mol.build(atom='H 0 0 0; F 0 0 0.8', basis='ccpvdz', verbose=0)
        mf = scf.RHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mf, 6, 6)
        mo = mcscf.sort_mo(mc, mf.mo_coeff, [3,4,5,6,8,9])
        print('E(0.8) = %.12f' % mc.kernel(mo)[0])
        init_mo = mc.mo_coeff
        for b in numpy.arange(1.0, 3., .2):
            mol.atom = [['H', (0, 0, 0)], ['F', (0, 0, b)]]
            mol.build(0, 0)
            mf = scf.RHF(mol)
            mf.scf()
            mc = mcscf.CASSCF(mf, 6, 6)
            mo = mcscf.project_init_guess(mc, init_mo)
            print('E(%2.1f) = %.12f' % (b, mc.kernel(mo)[0]))
            init_mo = mc.mo_coeff
    '''
    from pyscf import lo

    def project(mfmo, init_mo, ncore, s):
        nocc = ncore + casscf.ncas
        mo0core = init_mo[:,:ncore]
        s1 = reduce(numpy.dot, (mfmo.T, s, mo0core))
        idx = numpy.argsort(numpy.einsum('ij,ij->i', s1, s1))
        logger.debug(casscf, 'Core indices %s', str(numpy.sort(idx[-ncore:])))
        # take HF core
        mocore = mfmo[:,numpy.sort(idx[-ncore:])]

        # take projected CAS space
        mocas = init_mo[:,ncore:nocc] \
              - reduce(numpy.dot, (mocore, mocore.T, s, init_mo[:,ncore:nocc]))
        mocc = lo.orth.vec_lowdin(numpy.hstack((mocore, mocas)), s)

        # remove core and active space from rest
        mou = init_mo[:,nocc:] \
            - reduce(numpy.dot, (mocc, mocc.T, s, init_mo[:,nocc:]))
        mo = lo.orth.vec_lowdin(numpy.hstack((mocc, mou)), s)

        if casscf.verbose >= logger.DEBUG:
            s1 = reduce(numpy.dot, (mo[:,ncore:nocc].T, s, mfmo))
            idx = numpy.argwhere(abs(s1) > 0.4)
            for i,j in idx:
                logger.debug(casscf, 'Init guess <mo-CAS|mo-hf>  %d  %d  %12.8f',
                             ncore+i+1, j+1, s1[i,j])
        return mo

    ncore = casscf.ncore
    mfmo = casscf._scf.mo_coeff
    s = casscf._scf.get_ovlp()
    if isinstance(ncore, (int, numpy.integer)):
        if prev_mol is not None:
            init_mo = scf.addons.project_mo_nr2nr(prev_mol, init_mo, casscf.mol)
        else:
            assert(mfmo.shape[0] == init_mo.shape[0])
        mo = project(mfmo, init_mo, ncore, s)
    else: # UHF-based CASSCF
        if prev_mol is not None:
            init_mo = (scf.addons.project_mo_nr2nr(prev_mol, init_mo[0], casscf.mol),
                       scf.addons.project_mo_nr2nr(prev_mol, init_mo[1], casscf.mol))
        else:
            assert(mfmo[0].shape[0] == init_mo[0].shape[0])
        mo = (project(mfmo[0], init_mo[0], ncore[0], s),
              project(mfmo[1], init_mo[1], ncore[1], s))
    return mo

# on AO representation
def make_rdm1(casscf, mo_coeff=None, ci=None):
    '''One-particle densit matrix in AO representation

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

    Kwargs:
        ci : ndarray
            CAS space FCI coefficients. If not given, take casscf.ci.
        mo_coeff : ndarray
            Orbital coefficients. If not given, take casscf.mo_coeff.

    Examples:

    >>> import scipy.linalg
    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='sto-3g', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> res = mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> res = mc.kernel()
    >>> natocc = numpy.linalg.eigh(mcscf.make_rdm1(mc), mf.get_ovlp(), type=2)[0]
    >>> print(natocc)
    [ 0.0121563   0.0494735   0.0494735   1.95040395  1.95040395  1.98808879
      2.          2.          2.          2.        ]
    '''
    return casscf.make_rdm1(mo_coeff, ci)

# make both alpha and beta density matrices
def make_rdm1s(casscf, mo_coeff=None, ci=None):
    '''Alpha and beta one-particle densit matrices in AO representation
    '''
    return casscf.make_rdm1s(mo_coeff, ci)

def _is_uhf_mo(mo_coeff):
    return not (isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2)

def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
        dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
    return dm1, dm2

# on AO representation
def make_rdm12(casscf, mo_coeff=None, ci=None):
    if ci is None: ci = casscf.ci
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    assert(not _is_uhf_mo(mo_coeff))
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo_coeff.shape[1]
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci, ncas, nelecas)
    rdm1, rdm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1, mo_coeff.T))
    rdm2 = numpy.dot(mo_coeff, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo_coeff.T)
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo).transpose(2,3,0,1)
    rdm2 = numpy.dot(mo_coeff, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo_coeff.T)
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

def get_fock(casscf, mo_coeff=None, ci=None):
    '''Generalized Fock matrix in AO representation
    '''
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        raise RuntimeError('TODO: UCAS general fock')
    else:
        return casscf.get_fock(mo_coeff, ci)

def cas_natorb(casscf, mo_coeff=None, ci=None, sort=False):
    '''Natrual orbitals in CAS space
    '''
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        raise RuntimeError('TODO: UCAS natrual orbitals')
    else:
        return casscf.cas_natorb(mo_coeff, ci, sort=sort)

def map2hf(casscf, mf_mo=None, base=1, tol=.5):
    '''The overlap between the CASSCF optimized orbitals and the canonical HF orbitals.
    '''
    if mf_mo is None:
        mf_mo = casscf._scf.mo_coeff
    s = casscf.mol.intor_symmetric('cint1e_ovlp_sph')
    s = reduce(numpy.dot, (casscf.mo_coeff.T, s, mf_mo))
    idx = numpy.argwhere(abs(s) > tol)
    for i,j in idx:
        logger.info(casscf, '<mo_coeff-mcscf|mo_coeff-hf>  %d  %d  %12.8f',
                    i+base, j+base, s[i,j])
    return idx

def spin_square(casscf, mo_coeff=None, ci=None, ovlp=None):
    '''Spin square of the UHF-CASSCF wavefunction

    Returns:
        A list of two floats.  The first is the expectation value of S^2.
        The second is the corresponding 2S+1

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='sto-3g', spin=2, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> res = mf.scf()
    >>> mc = mcscf.CASSCF(mf, 4, 6)
    >>> res = mc.kernel()
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
    S^2 = 3.9831589, 2S+1 = 4.1149284
    '''
    from pyscf import scf
    if ci is None: ci = casscf.ci
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if ovlp is None: ovlp = casscf._scf.get_ovlp()
    ncore    = casscf.ncore
    ncas     = casscf.ncas
    nelecas  = casscf.nelecas
    if isinstance(ncore, (int, numpy.integer)):
        return pyscf.fci.spin_op.spin_square0(ci, ncas, nelecas)
    else:
        nocc = (ncore[0] + ncas, ncore[1] + ncas)
        mocas = (mo_coeff[0][:,ncore[0]:nocc[0]], mo_coeff[1][:,ncore[1]:nocc[1]])
        sscas = pyscf.fci.spin_op.spin_square(ci, ncas, nelecas, mocas, ovlp)
        mocore = (mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
        sscore = scf.uhf.spin_square(mocore, ovlp)
        logger.debug(casscf, 'S^2 of core %f  S^2 of cas %f', sscore[0], sscas[0])
        ss = sscas[0]+sscore[0]
        s = numpy.sqrt(ss+.25) - .5
        return ss, s*2+1


def state_average(casscf, weights=(0.5,0.5)):
    ''' State average over the energy.  The energy funcitonal is
    E = w1<psi1|H|psi1> + w2<psi2|H|psi2> + ...

    Note we may need change the FCI solver to

    mc.fcisolver = pyscf.fci.solver(mol, False)

    before calling state_average_(mc), to mix the singlet and triplet states
    '''
    assert(abs(sum(weights)-1) < 1e-10)
    fcibase = casscf.fcisolver
    fcibase_class = casscf.fcisolver.__class__
    class FakeCISolver(fcibase_class):
        def __init__(self):
            self.__dict__.update(fcibase.__dict__)
            self.nroots = len(weights)
        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
# pass self to fcibase_class.kernel function because orbsym argument is stored in self 
# but undefined in fcibase object
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, **kwargs)
            if casscf.verbose >= logger.DEBUG:
                ss = fcibase_class.spin_square(self, c, norb, nelec)
                for i, ei in enumerate(e):
                    logger.debug(casscf, 'state %d  E = %.15g S^2 = %.7f',
                                 i, ei, ss[0][i])
            return numpy.einsum('i,i->', e, weights), c
        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        max_cycle=casscf.ci_response_space,
                                        nroots=self.nroots, **kwargs)
            return numpy.einsum('i,i->', e, weights), c
        def make_rdm1(self, ci0, norb, nelec):
            dm1 = 0
            for i, wi in enumerate(weights):
                dm1 += wi*fcibase_class.make_rdm1(self, ci0[i], norb, nelec)
            return dm1
        def make_rdm12(self, ci0, norb, nelec):
            rdm1 = 0
            rdm2 = 0
            for i, wi in enumerate(weights):
                dm1, dm2 = fcibase_class.make_rdm12(self, ci0[i], norb, nelec)
                rdm1 += wi * dm1
                rdm2 += wi * dm2
            return rdm1, rdm2
        def spin_square(self, ci0, norb, nelec):
            ss = fcibase_class.spin_square(self, ci0, norb, nelec)[0]
            ss = numpy.einsum('i,i->', weights, ss)
            multip = numpy.sqrt(ss+.25)*2
            return ss, multip
    return FakeCISolver()

def state_average_(casscf, weights=(0.5,0.5)):
    casscf.fcisolver = state_average(casscf, weights)
    return casscf


def state_specific(casscf, state=1):
    '''For excited state

    Kwargs:
        state : int
        0 for ground state; 1 for first excited state.
    '''
    fcibase = casscf.fcisolver
    fcibase_class = casscf.fcisolver.__class__
    class FakeCISolver(fcibase_class):
        def __init__(self):
            self.__dict__.update(fcibase.__dict__)
            self.nroots = state+1
            self._civec = None
        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, **kwargs)
            self._civec = c
            if casscf.verbose >= logger.DEBUG:
                ss = fcibase_class.spin_square(self, c[state], norb, nelec)
                logger.debug(casscf, 'state %d  E = %.15g S^2 = %.7f',
                             state+1, e[state], ss[0])
            return e[state], c[state]
        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        max_cycle=casscf.ci_response_space,
                                        nroots=self.nroots, **kwargs)
            self._civec = c
            return e[state], c[state]
        def spin_square(self, fcivec, norb, nelec):
            return pyscf.fci.spin_op.spin_square0(fcivec, norb, nelec)
    return FakeCISolver()

def state_specific_(casscf, state=1):
    casscf.fcisolver = state_specific(casscf, state)
    return casscf



if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import mcscf
    from pyscf import tools
    import pyscf.tools.ring

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['H', c] for c in tools.ring.make(6, 1.2)]
    mol.basis = '6-31g'
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mcscf.CASSCF(m, 6, 6)
    mc.verbose = 4
    emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()
    print(ehf, emc, emc-ehf)
    print(emc - -3.272089958)

    rdm1 = make_rdm1(mc, mo, fcivec)
    rdm1, rdm2 = make_rdm12(mc, mo, fcivec)
    print(rdm1)

    mo1 = cas_natorb(mc)[0]
    numpy.set_printoptions(2)
    print(reduce(numpy.dot, (mo1[:,:6].T, mol.intor('cint1e_ovlp_sph'),
                             mo[:,:6])))

# state average
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = '6-31g'
    mol.symmetry = 1
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()


    mc = mcscf.CASSCF(m, 4, 4)
    mc.verbose = 4
    mc.fcisolver = pyscf.fci.solver(mol, False) # to mix the singlet and triplet
    mc = state_average_(mc, (.64,.36))
    emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()
    mc = mcscf.CASCI(m, 4, 4)
    emc = mc.casci(mo)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -76.003352190262532)


    mc = mcscf.CASSCF(m, 4, 4)
    mc.verbose = 4
    mc = state_average_(mc, (.64,.36))
    emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()
    mc = mcscf.CASCI(m, 4, 4)
    emc = mc.casci(mo)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -75.982520334896776)
