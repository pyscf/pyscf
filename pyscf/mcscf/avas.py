#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Elvira R. Sayfutyarova <elviras@princeton.edu>
#

'''
Automated construction of molecular active spaces from atomic valence orbitals.
Ref. arXiv:1701.07862 [physics.chem-ph]
'''

import re
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.lib import logger

def kernel(mf, aolabels, threshold=.2, minao='minao', with_iao=False,
           openshelloption=2, canonicalize=True, verbose=None):
    '''AVAS method to construct mcscf active space.
    Ref. arXiv:1701.07862 [physics.chem-ph]

    Args:
        mf : an :class:`SCF` object

        aolabels : string or a list of strings
            AO labels for AO active space

    Kwargs:
        threshold : float
            Tructing threshold of the AO-projector above which AOs are kept in
            the active space.
        minao : str
            A reference AOs for AVAS.
        with_iao : bool
            Whether to use IAO localization to construct the reference active AOs.
        openshelloption : int
            How to handle singly-occupied orbitals in the active space. The
            singly-occupied orbitals are projected as part of alpha orbitals
            if openshelloption=2, or completely kept in active space if
            openshelloption=3.  See Section III.E option 2 or 3 of the
            reference paper for more details.
        canonicalize : bool
            Orbitals defined in AVAS method are local orbitals.  Symmetrizing
            the core, active and virtual space.

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess-for-CASCI/CASSCF

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.mcscf import avas
    >>> mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.6', basis='ccpvtz')
    >>> mf = scf.RHF(mol).run()
    >>> ncas, nelecas, mo = avas.avas(mf, ['Cr 3d', 'Cr 4s'])
    >>> mc = mcscf.CASSCF(mf, ncas, nelecas).run(mo)
    '''
    from pyscf.tools import mo_mapping

    if isinstance(verbose, logger.Logger):
        log = verbose
    elif verbose is not None:
        log = logger.Logger(mf.stdout, verbose)
    else:
        log = logger.Logger(mf.stdout, mf.verbose)
    mol = mf.mol

    log.info('\n** AVAS **')
    if isinstance(mf, scf.uhf.UHF):
        log.note('UHF/UKS object is found.  AVAS takes alpha orbitals only')
        mo_coeff = mf.mo_coeff[0]
        mo_occ = mf.mo_occ[0]
        mo_energy = mf.mo_energy[0]
        assert(openshelloption != 1)
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
    nocc = numpy.count_nonzero(mo_occ != 0)
    ovlp = mol.intor_symmetric('int1e_ovlp')
    log.info('  Total number of HF MOs  is equal to    %d' ,mo_coeff.shape[1])
    log.info('  Number of occupied HF MOs is equal to  %d', nocc)

    mol = mf.mol
    pmol = mol.copy()
    pmol.atom = mol._atom
    pmol.unit = 'B'
    pmol.symmetry = False
    pmol.basis = minao
    pmol.build(False, False)

    baslst = pmol.search_ao_label(aolabels)
    log.info('reference AO indices for %s %s: %s', minao, aolabels, baslst)

    if with_iao:
        from pyscf.lo import iao
        c = iao.iao(mol, mo_coeff[:,:nocc], minao)[:,baslst]
        s2 = reduce(numpy.dot, (c.T, ovlp, c))
        s21 = reduce(numpy.dot, (c.T, ovlp, mo_coeff))
    else:
        s2 = pmol.intor_symmetric('int1e_ovlp')[baslst][:,baslst]
        s21 = gto.intor_cross('int1e_ovlp', pmol, mol)[baslst]
        s21 = numpy.dot(s21, mo_coeff)
    sa = s21.T.dot(scipy.linalg.solve(s2, s21, sym_pos=True))

    if openshelloption == 2:
        wocc, u = numpy.linalg.eigh(sa[:nocc,:nocc])
        log.info('Option 2: threshold %s', threshold)
        ncas_occ = (wocc > threshold).sum()
        nelecas = mol.nelectron - (wocc < threshold).sum() * 2
        mocore = mo_coeff[:,:nocc].dot(u[:,wocc<threshold])
        mocas = mo_coeff[:,:nocc].dot(u[:,wocc>threshold])

        wvir, u = numpy.linalg.eigh(sa[nocc:,nocc:])
        ncas_vir = (wvir > threshold).sum()
        mocas = numpy.hstack((mocas, mo_coeff[:,nocc:].dot(u[:,wvir>threshold])))
        movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
        ncas = mocas.shape[1]

    elif openshelloption == 3:
        docc = nocc - mol.spin
        wocc, u = numpy.linalg.eigh(sa[:docc,:docc])
        log.info('Option 3: threshold %s, num open shell %d', threshold, mol.spin)
        ncas_occ = (wocc > threshold).sum()
        nelecas = mol.nelectron - (wocc < threshold).sum() * 2
        mocore = mo_coeff[:,:docc].dot(u[:,wocc<threshold])
        mocas = mo_coeff[:,:docc].dot(u[:,wocc>threshold])

        wvir, u = numpy.linalg.eigh(sa[nocc:,nocc:])
        ncas_vir = (wvir > threshold).sum()
        mocas = numpy.hstack((mocas, mo_coeff[:,docc:nocc],
                              mo_coeff[:,nocc:].dot(u[:,wvir>threshold])))
        movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
        ncas = mocas.shape[1]

    log.debug('projected occ eig %s', wocc[::-1])
    log.debug('projected vir eig %s', wvir[::-1])
    log.info('Active from occupied = %d , eig %s', ncas_occ, wocc[wocc>threshold][::-1])
    log.info('Inactive from occupied = %d', mocore.shape[1])
    log.info('Active from unoccupied = %d , eig %s', ncas_vir, wvir[wvir>threshold][::-1])
    log.info('Inactive from unoccupied = %d', movir.shape[1])
    log.info('Dimensions of active %d', ncas)
    nalpha = (nelecas + mol.spin) // 2
    nbeta = nelecas - nalpha
    log.info('# of alpha electrons %d', nalpha)
    log.info('# of beta electrons %d', nbeta)

    if canonicalize:
        from pyscf.mcscf import dmet_cas
        def trans(c):
            if c.shape[1] == 0:
                return c
            else:
                csc = reduce(numpy.dot, (c.T, ovlp, mo_coeff))
                fock = numpy.dot(csc*mo_energy, csc.T)
                e, u = scipy.linalg.eigh(fock)
                return dmet_cas.symmetrize(mol, e, numpy.dot(c, u), ovlp, log)
        mo = numpy.hstack([trans(mocore), trans(mocas), trans(movir)])
    else:
        mo = numpy.hstack((mocore, mocas, movir))
    return ncas, nelecas, mo
avas = kernel

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    mol = gto.M(
    verbose = 0,
    atom = '''
           H    0.000000,  0.500000,  1.5   
           O    0.000000,  0.000000,  1.
           O    0.000000,  0.000000, -1.
           H    0.000000, -0.500000, -1.5''',
        basis = 'ccpvdz',
    )

    mf = scf.UHF(mol)
    mf.scf()

    ncas, nelecas, mo = avas(mf, 'O 2p', verbose=4)
    mc = mcscf.CASSCF(mf, ncas, nelecas).set(verbose=4)
    emc = mc.kernel(mo)[0]
    print(emc, -150.51496582534054)

