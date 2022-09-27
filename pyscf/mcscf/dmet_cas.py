#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf.tools import dump_mat
from pyscf import scf, gto
from pyscf import __config__

THRESHOLD = getattr(__config__, 'mcscf_dmet_cas_threshold', 0.05)
OCC_CUTOFF = getattr(__config__, 'mcscf_dmet_cas_occ_cutoff', 1e-6)
BASE = getattr(__config__, 'BASE', 0)
ORTH_METHOD = getattr(__config__, 'mcscf_dmet_cas_orth_method', 'meta_lowdin')
CANONICALIZE = getattr(__config__, 'mcscf_dmet_cas_canonicalize', True)
FREEZE_IMP = getattr(__config__, 'mcscf_dmet_cas_freeze_imp', False)

def kernel(mf, dm, aolabels_or_baslst, threshold=THRESHOLD,
           occ_cutoff=OCC_CUTOFF, base=BASE,
           orth_method=ORTH_METHOD, s=None, canonicalize=CANONICALIZE,
           freeze_imp=FREEZE_IMP, verbose=None):
    '''DMET method to generate CASSCF initial guess.
    Ref. arXiv:1701.07862 [physics.chem-ph]

    Args:
        mf : an :class:`SCF` object

        dm : 2D np.array or a list of 2D array
            Density matrix
        aolabels_or_baslst : string or a list of strings or a list of index
            AO labels or indices

    Kwargs:
        threshold : float
            Entanglement threshold of DMET bath.  If the occupancy of an
            orbital is less than threshold, the orbital is considered as bath
            orbtial.  If occ is greater than (1-threshold), the orbitals are
            taken for core determinant.
        base : int
            0-based (C-style) or 1-based (Fortran-style) for baslst if baslst
            is index list
        orth_method : str
            It can be one of 'lowdin' and 'meta_lowdin'
        s : 2D array
            AO overlap matrix.  This option is mainly used for custom Hamilatonian.
        canonicalize : bool
            Orbitals defined in AVAS method are local orbitals.  Symmetrizing
            the core, active and virtual space.

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess-for-CASCI/CASSCF

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.mcscf import dmet_cas
    >>> mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.6', basis='ccpvtz')
    >>> mf = scf.RHF(mol).run()
    >>> ncas, nelecas, mo = dmet_cas.dmet_cas(mf, ['Cr 3d', 'Cr 4s'])
    >>> mc = mcscf.CASSCF(mf, ncas, nelecas).run(mo)
    '''
    from pyscf import lo

    if isinstance(verbose, logger.Logger):
        log = verbose
    elif verbose is not None:
        log = logger.Logger(mf.stdout, verbose)
    else:
        log = logger.Logger(mf.stdout, mf.verbose)
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2): # ROHF/UHF DM
        dm = sum(dm)
    mol = mf.mol
    if s is None:
        s = mf.get_ovlp()

    if (not isinstance(mf, scf.hf.SCF)) and getattr(mf, '_scf', None):
        mf = mf._scf

    baslst = gto.mole._aolabels2baslst(mol, aolabels_or_baslst, base)

    nao = dm.shape[0]
    nimp = len(baslst)
    log.debug('*** decompose density matrix')
    log.debug('orth AO method = %s', orth_method)
    log.debug('embedding AO list = %s', str(baslst))
    if orth_method is not None:
        corth = lo.orth.orth_ao(mol, method=orth_method, s=s)
        cinv = numpy.dot(corth.T, s)
        dm = reduce(numpy.dot, (cinv, dm, cinv.T))
    else:
        corth = numpy.eye(nao)

    notimp = numpy.asarray([i for i in range(nao) if i not in baslst])
    occi, ui = scipy.linalg.eigh(-dm[baslst[:,None],baslst])
    occi *= -1
    idxi = numpy.argsort(abs(occi-1))
    log.debug('entanglement weight occ = %s', str(occi[idxi]))
    occb, ub = scipy.linalg.eigh(dm[notimp[:,None],notimp])
    idxb = numpy.argsort(abs(occb-1))  # sort by entanglement
    occb = occb[idxb]
    ub = ub[:,idxb]

    # guess ncas and nelecas
    nb = ((occb > occ_cutoff) & (occb < 2-occ_cutoff)).sum()
    log.debug('bath weight occ = %s', occb[:nb])
    cum_nelec = numpy.cumsum(occb[:nb]) + occi.sum()
    cum_nelec = numpy.append(occi.sum(), cum_nelec)
    log.debug('Active space cum nelec imp|[baths] = %f |%s',
              cum_nelec[0], cum_nelec[1:])
    ne_error = abs(cum_nelec.round() - cum_nelec)
    nb4cas = nb
    for i in range(nb):
        if (ne_error[i] < threshold and
            # whether all baths next to ith bath are less important
            (occb[i] < threshold or occb[i] > 2-threshold)):
            nb4cas = i
            break
    ncas = nb4cas + nimp
    nelecas = int(cum_nelec[nb4cas].round())
    ncore = (mol.nelectron - nelecas) // 2
    log.info('From DMET guess, ncas = %d  nelecas = %d  ncore = %d',
             ncas, nelecas, ncore)

    log.debug('DMET impurity and bath orbitals on orthogonal AOs')
    log.debug('DMET %d impurity sites/occ', nimp)
    if log.verbose >= logger.DEBUG1:
        label = mol.ao_labels()
        occ_label = ['#%d/%.5f'%(i+1,x) for i,x in enumerate(occi)]
        #dump_mat.dump_rec(mol.stdout, numpy.dot(corth[:,baslst], ui),
        #                  label=label, label2=occ_label, start=1)
        dump_mat.dump_rec(mol.stdout, ui, label=[label[i] for i in baslst],
                          label2=occ_label, start=1)

    log.debug('DMET %d entangled baths/occ', nb)
    if log.verbose >= logger.DEBUG1:
        occ_label = ['#%d/%.5f'%(i+1,occb[i]) for i in range(nb)]
        #dump_mat.dump_rec(mol.stdout, numpy.dot(corth[:,notimp], ub[:,:nb]),
        #                  label=label, label2=occ_label, start=1)
        dump_mat.dump_rec(mol.stdout, ub[:,:nb], label=[label[i] for i in notimp],
                          label2=occ_label, start=1)

    mob = numpy.dot(corth[:,notimp], ub[:,:nb4cas])
    idxenv = numpy.argsort(-occb[nb4cas:]) + nb4cas
    mo_env = numpy.dot(corth[:,notimp], ub[:,idxenv])

    mocore = mo_env[:,:ncore]
    mocas = numpy.hstack((numpy.dot(corth[:,baslst],ui), mob))
    movir = mo_env[:,ncore:]

    if canonicalize or freeze_imp:
        if mf.mo_energy is None or mf.mo_coeff is None:
            fock = mf.get_hcore()
        else:
            if isinstance(mf.mo_coeff, numpy.ndarray) and mf.mo_coeff.ndim == 2:
                sc = numpy.dot(s, mf.mo_coeff)
                fock = numpy.dot(sc*mf.mo_energy, sc.T)
            else:
                sc = numpy.dot(s, mf.mo_coeff[0])
                fock = numpy.dot(sc*mf.mo_energy[0], sc.T)

        def trans(c):
            if c.shape[1] == 0:
                return c
            else:
                f1 = reduce(numpy.dot, (c.T, fock, c))
                e, u = scipy.linalg.eigh(f1)
                log.debug1('Fock eig %s', e)
                return symmetrize(mol, e, numpy.dot(c, u), s, log)

        if freeze_imp:
            # freeze_imp to avoid canonicalization for impurity orbitals. It
            # reserves the locality of the impurity orbitals even the rest
            # orbitals are diffused due to the canonicalization.
            log.debug('Semi-canonicalization for freeze_imp=True')
            mo = numpy.hstack([trans(mocore), trans(mocas[:,:nimp]),
                               trans(mocas[:,nimp:]), trans(movir)])
        else:
            mo = numpy.hstack([trans(mocore), trans(mocas), trans(movir)])
    else:
        mo = numpy.hstack((mocore, mocas, movir))

    return ncas, nelecas, mo
dmet_cas = guess_cas = kernel


def search_for_degeneracy(e):
    idx = numpy.where(abs(e[1:] - e[:-1]) < 1e-3)[0]
    return numpy.unique(numpy.hstack((idx, idx+1)))

def symmetrize(mol, e, c, s, log):
    if mol.symmetry:
        degidx = search_for_degeneracy(e)
        log.debug1('degidx %s', degidx)
        if degidx.size > 0:
            esub = e[degidx]
            csub = c[:,degidx]
            scsub = numpy.dot(s, csub)
            emin = abs(esub).min() * .5
            es = []
            cs = []
            for i,ir in enumerate(mol.irrep_id):
                so = mol.symm_orb[i]
                sosc = numpy.dot(so.T, scsub)
                s_ir = reduce(numpy.dot, (so.T, s, so))
                fock_ir = numpy.dot(sosc*esub, sosc.T)
                e, u = scipy.linalg.eigh(fock_ir, s_ir)
                idx = abs(e) > emin
                es.append(e[idx])
                cs.append(numpy.dot(mol.symm_orb[i], u[:,idx]))
            es = numpy.hstack(es)
            idx = numpy.argsort(es, kind='mergesort')
            assert (numpy.allclose(es[idx], esub, rtol=1e-3, atol=1e-4))
            c[:,degidx] = numpy.hstack(cs)[:,idx]
    return c

del (THRESHOLD, OCC_CUTOFF, BASE, ORTH_METHOD, CANONICALIZE, FREEZE_IMP)

if __name__ == '__main__':
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

    mf = scf.RHF(mol)
    mf.scf()

    aolst = [i for i,s in enumerate(mol.ao_labels()) if 'H 1s' in s]
    dm = mf.make_rdm1()
    ncas, nelecas, mo = guess_cas(mf, dm, aolst, verbose=4)
    mc = mcscf.CASSCF(mf, ncas, nelecas).set(verbose=4)
    emc = mc.kernel(mo)[0]
    print(emc,)

