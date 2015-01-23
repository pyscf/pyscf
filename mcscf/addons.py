#!/usr/bin/env python

from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.lib.logger as log

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
    >>> mol = gto.Mole()
    >>> mol.build(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mol, mf, 4, 4)
    >>> cas_list = [5,6,8,9] # pi orbitals
    >>> mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list)
    >>> mc.kernel(mo)[0]
    -109.007378939813691
    '''
    ncore = casscf.ncore
    if isinstance(ncore, int):
        assert(casscf.ncas == len(caslst))
        nmo = mo_coeff.shape[1]
        if base != 0:
            caslst = [i-1 for i in caslst]
        idx = [i for i in range(nmo) if i not in caslst]
        return numpy.hstack((mo_coeff[:,idx[:ncore]], mo_coeff[:,caslst], mo_coeff[:,idx[ncore:]]))
    else: # UHF-based CASSCF
        if isinstance(caslst[0], int):
            assert(casscf.ncas == len(caslst))
            if base != 0:
                caslsta = [i-1 for i in caslst]
                caslst = (caslsta, caslsta)
        else: # two casspace lists, for alpha and beta
            assert(casscf.ncas == len(caslst[0]))
            assert(casscf.ncas == len(caslst[1]))
            if base != 0:
                caslst = ([i-1 for i in caslst[0]], [i-1 for i in caslst[1]])
        nmo = mo_coeff[0].shape[1]
        idx = [i for i in range(nmo) if i not in caslst[0]]
        mo_a = numpy.hstack((mo_coeff[0][:,idx[:ncore[0]]], mo_coeff[0][:,caslst[0]],
                             mo_coeff[0][:,idx[ncore[0]:]]))
        idx = [i for i in range(nmo) if i not in caslst[1]]
        mo_b = numpy.hstack((mo_coeff[1][:,idx[:ncore[1]]], mo_coeff[1][:,caslst[1]],
                             mo_coeff[1][:,idx[ncore[1]:]]))
        return (mo_a, mo_b)

def project_init_guess(casscf, init_mo):
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
        mc = mcscf.CASSCF(mol, mf, 6, 6)
        mo = mcscf.sort_mo(mc, mf.mo_coeff, [3,4,5,6,8,9])
        print('E(0.8) = %.12f' % mc.kernel(mo)[0])
        init_mo = mc.mo_coeff
        for b in numpy.arange(1.0, 3., .2):
            mol.atom = [['H', (0, 0, 0)], ['F', (0, 0, b)]]
            mol.build(0, 0)
            mf = scf.RHF(mol)
            mf.scf()
            mc = mcscf.CASSCF(mol, mf, 6, 6)
            mo = mcscf.project_init_guess(mc, init_mo)
            print('E(%2.1f) = %.12f' % (b, mc.kernel(mo)[0]))
            init_mo = mc.mo_coeff
    '''
    from pyscf import scf
    from pyscf import lo

    def project(mfmo, init_mo, ncore, s):
        mo0core = init_mo[:,:ncore]
        s1 = reduce(numpy.dot, (mfmo.T, s, mo0core))
        idx = numpy.argsort(numpy.einsum('ij,ij->i', s1, s1))
        log.debug(casscf, 'Core indices %s', str(idx[-ncore:][::-1]))
        mocore = mfmo[:,idx[-ncore:][::-1]]
        mou = init_mo[:,ncore:] \
            - reduce(numpy.dot, (mocore, mocore.T, s, init_mo[:,ncore:]))
        mo = numpy.hstack((mocore, mou))
        mo = lo.orth.vec_lowdin(mo, s)

        if casscf.verbose >= log.DEBUG:
            nocc = ncore + casscf.ncas
            s = reduce(numpy.dot, (mo[:,ncore:nocc].T, s, mfmo))
            idx = numpy.argwhere(abs(s) > 0.5)
            for i,j in idx:
                log.debug(casscf, 'Init guess <mo-CAS|mo-hf>  %d  %d  %12.8f',
                          ncore+i+1, j+1, s[i,j])
        return mo

    ncore = casscf.ncore
    mfmo = casscf._scf.mo_coeff
    s = casscf._scf.get_ovlp()
    if isinstance(ncore, int):
        assert(mfmo.shape[0] == init_mo.shape[0])
        mo = project(mfmo, init_mo, ncore, s)
    else: # UHF-based CASSCF
        assert(mfmo[0].shape[0] == init_mo[0].shape[0])
        mo = (project(mfmo[0], init_mo[0], ncore[0], s),
              project(mfmo[1], init_mo[1], ncore[1], s))
    return mo

def _make_rdm1_on_mo(casdm1, ncore, ncas, nmo, docc=True):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    if docc:
        dm1[idx,idx] = 2
    else:
        dm1[idx,idx] = 1
    dm1[ncore:nocc,ncore:nocc] = casdm1
    return dm1

# on AO representation
def make_rdm1(casscf, fcivec=None, mo_coeff=None):
    '''One-particle densit matrix in AO representation

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

    Kwargs:
        fcivec : ndarray
            CAS space FCI coefficients. If not given, take casscf.ci.
        mo_coeff : ndarray
            Orbital coefficients. If not given, take casscf.mo_coeff.

    Examples:

    >>> import scipy.linalg
    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.Mole()
    >>> mol.build(atom='N 0 0 0; N 0 0 1', basis='sto-3g', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> res = mf.scf()
    >>> mc = mcscf.CASSCF(mol, mf, 6, 6)
    >>> res = mc.kernel()
    >>> natocc = numpy.linalg.eigh(mcscf.make_rdm1(mc), mf.get_ovlp(), type=2)[0]
    >>> print(natocc)
    [ 0.0121563   0.0494735   0.0494735   1.95040395  1.95040395  1.98808879
      2.          2.          2.          2.        ]
    '''
    if fcivec is None:
        fcivec = casscf.ci
    if mo_coeff is None:
        mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        return sum(make_rdm1s(casscf, fcivec, mo_coeff))
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo_coeff.shape[1]
    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
    rdm1 = _make_rdm1_on_mo(casdm1, ncore, ncas, nmo, True)
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1, mo_coeff.T))
    return rdm1

# make both alpha and beta density matrices
def make_rdm1s(casscf, fcivec=None, mo_coeff=None):
    '''Alpha and beta one-particle densit matrices in AO representation
    '''
    if fcivec is None:
        fcivec = casscf.ci
    if mo_coeff is None:
        mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        nmo = mo_coeff[0].shape[1]
    else:
        nmo = mo_coeff.shape[1]
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    rdm1a, rdm1b = casscf.fcisolver.make_rdm1s(fcivec, ncas, nelecas)
    if _is_uhf_mo(mo_coeff):
        rdm1a = _make_rdm1_on_mo(rdm1a, ncore[0], ncas, nmo, False)
        rdm1b = _make_rdm1_on_mo(rdm1b, ncore[1], ncas, nmo, False)
        rdm1a = reduce(numpy.dot, (mo_coeff[0], rdm1a, mo_coeff[0].T))
        rdm1b = reduce(numpy.dot, (mo_coeff[1], rdm1b, mo_coeff[1].T))
    else:
        rdm1a = _make_rdm1_on_mo(rdm1a, ncore, ncas, nmo, False)
        rdm1b = _make_rdm1_on_mo(rdm1b, ncore, ncas, nmo, False)
        rdm1a = reduce(numpy.dot, (mo_coeff, rdm1a, mo_coeff.T))
        rdm1b = reduce(numpy.dot, (mo_coeff, rdm1b, mo_coeff.T))
    return rdm1a, rdm1b

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
def make_rdm12(casscf, fcivec=None, mo_coeff=None):
    if fcivec is None:
        fcivec = casscf.ci
    if mo_coeff is None:
        mo_coeff = casscf.mo_coeff
    assert(not _is_uhf_mo(mo_coeff))
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo_coeff.shape[1]
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, nelecas)
    rdm1, rdm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1, mo_coeff.T))
    rdm2 = numpy.dot(mo_coeff, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo_coeff.T)
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo).transpose(2,3,0,1)
    rdm2 = numpy.dot(mo_coeff, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo_coeff.T)
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

def get_fock(casscf, fcivec=None, mo_coeff=None):
    '''Generalized Fock matrix
    '''
    if fcivec is None:
        fcivec = casscf.ci
    if mo_coeff is None:
        mo_coeff = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas

    if _is_uhf_mo(mo_coeff):
        casdm1 = casscf.fcisolver.make_rdm1s(fcivec, ncas, nelecas)
        eris = casscf.update_ao2mo(mo_coeff)
        vhf = (numpy.einsum('ipq->pq', eris.jkcpp) + eris.jC_pp \
               + numpy.einsum('uvpq,uv->pq', eris.aapp, casdm1[0]) \
               - numpy.einsum('upqv,uv->pq', eris.appa, casdm1[0]) \
               + numpy.einsum('uvpq,uv->pq', eris.AApp, casdm1[1]),
               numpy.einsum('ipq->pq', eris.jkcPP) + eris.jc_PP \
                + numpy.einsum('uvpq,uv->pq', eris.aaPP, casdm1[0]) \
                + numpy.einsum('uvpq,uv->pq', eris.AAPP, casdm1[1]) \
                - numpy.einsum('upqv,uv->pq', eris.APPA, casdm1[1]),)
        h1 =(reduce(numpy.dot, (mo_coeff[0].T, casscf.get_hcore()[0], mo_coeff[0])),
             reduce(numpy.dot, (mo_coeff[1].T, casscf.get_hcore()[1], mo_coeff[1])))
        fock = (h1[0] + vhf[0], h1[1] + vhf[1])
    else:
        casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
        eris = casscf.update_ao2mo(mo_coeff)
        vj = numpy.einsum('ipq->pq', eris.jc_pp) * 2 \
           + numpy.einsum('ij,ijpq->pq', casdm1, eris.aapp)
        vk = numpy.einsum('ipq->pq', eris.kc_pp) * 2 \
           + numpy.einsum('ij,ipqj->pq', casdm1, eris.appa)
        h1 = reduce(numpy.dot, (mo_coeff.T, casscf.get_hcore(), mo_coeff))
        fock = h1 + vj - vk * .5
    return fock

def cas_natorb(casscf, fcivec=None, mo_coeff=None):
    '''Restore natrual orbitals
    '''
    log = pyscf.lib.logger.Logger(casscf.stdout, casscf.verbose)
    if fcivec is None:
        fcivec = casscf.ci
    if mo_coeff is None:
        mo_coeff = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas

    def rotate(casdm1, mo_coeff, ncore, nocc, title):
        occ, ucas = scipy.linalg.eigh(-casdm1)
        occ = -occ
        log.info(title)
        log.info(str(occ))
        # restore phase
        for i, k in enumerate(numpy.argmax(abs(ucas), axis=0)):
            if ucas[k,i] < 0:
                ucas[:,i] *= -1
        mo_coeff = mo_coeff.copy()
        mo_coeff[:,ncore:nocc] = numpy.dot(mo_coeff[:,ncore:nocc], ucas)
        return mo_coeff, occ

    if _is_uhf_mo(mo_coeff):
        casdm1 = casscf.fcisolver.make_rdm1s(fcivec, ncas, nelecas)
        moa, occa = rotate(casdm1[0], mo_coeff[0], ncore[0], ncore[0]+ncas,
                           'Natural occs alpha')
        mob, occb = rotate(casdm1[1], mo_coeff[1], ncore[1], ncore[1]+ncas,
                           'Natural occs beta')
        mo_coeff = (moa, mob)
        mo_occ = (occa, occb)
        fcivec = casscf.casci(mo_coeff, fcivec)[2]
    else:
        casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
        mo_coeff, mo_occ = rotate(casdm1, mo_coeff, ncore, ncore+ncas,
                                  'Natural occs')
        fcivec = casscf.casci(mo_coeff, fcivec)[2]
    return fcivec, mo_coeff, mo_occ

def map2hf(casscf, mf_mo=None, base=1, tol=.5):
    '''The overlap between the CASSCF optimized orbitals and the canonical HF orbitals.
    '''
    if mf_mo is None:
        mf_mo = casscf._scf.mo_coeff
    s = casscf.mol.intor_symmetric('cint1e_ovlp_sph')
    s = reduce(numpy.dot, (casscf.mo_coeff.T, s, mf_mo))
    idx = numpy.argwhere(abs(s) > tol)
    for i,j in idx:
        log.info(casscf, '<mo_coeff-mcscf|mo_coeff-hf>  %d  %d  %12.8f',
                 i+base, j+base, s[i,j])
    return idx

def spin_square(casscf, fcivec=None, mo_coeff=None, ovlp=None):
    '''Spin square of the UHF-CASSCF wavefunction

    Returns:
        A list of two floats.  The first is the expectation value of S^2.
        The second is the corresponding 2S+1

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.Mole()
    >>> mol.build(atom='O 0 0 0; O 0 0 1', basis='sto-3g', spin=2, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> res = mf.scf()
    >>> mc = mcscf.CASSCF(mol, mf, 4, 6)
    >>> res = mc.kernel()
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
    S^2 = 3.9831589, 2S+1 = 4.1149284
    '''
    from pyscf import scf
    from pyscf import fci
    if fcivec is None: fcivec = casscf.ci
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if ovlp is None: ovlp = casscf._scf.get_ovlp()
    ncore    = casscf.ncore
    ncas     = casscf.ncas
    nelecas  = casscf.nelecas
    if isinstance(ncore, int):
        nocc = ncore + ncas
        mocas = mo_coeff[:,ncore:nocc]
        return fci.spin_op.spin_square(fcivec, ncas, nelecas, mocas, ovlp)
    else:
        nocc = (ncore[0] + ncas, ncore[1] + ncas)
        mocas = (mo_coeff[0][:,ncore[0]:nocc[0]], mo_coeff[1][:,ncore[1]:nocc[1]])
        sscas = fci.spin_op.spin_square(fcivec, ncas, nelecas, mocas, ovlp)
        mocore = (mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
        sscore = scf.uhf.spin_square(mocore, ovlp)
        log.debug(casscf, 'S^2 of core %f, S^2 of cas %f', sscore[0], sscas[0])
        ss = sscas[0]+sscore[0]
        s = numpy.sqrt(ss+.25) - .5
        return ss, s*2+1


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    import mc1step
    from pyscf import tools
    import pyscf.tools.ring

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['H', c] for c in tools.ring.make(6, 1.2)]
    mol.basis = {'H': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mc1step.CASSCF(mol, m, 6, 6)
    mc.verbose = 4
    emc, e_ci, fcivec, mo = mc.mc1step()
    print(ehf, emc, emc-ehf)
    print(emc - -3.272089958)

    rdm1 = make_rdm1(mc, fcivec, mo)
    rdm1, rdm2 = make_rdm12(mc, fcivec, mo)
    print(rdm1)

    mo1 = cas_natorb(mc)[1]
    numpy.set_printoptions(2)
    print(reduce(numpy.dot, (mo1[:,:6].T, mol.intor('cint1e_ovlp_sph'),
                             mo[:,:6])))
