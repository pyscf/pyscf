#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf.tools import dump_mat
from pyscf.mcscf.casci_symm import label_symmetry_

def guess_cas(mf, dm, baslst, nelec_tol=.05, occ_cutoff=1e-6, base=0,
              orth_method='meta_lowdin', s=None, canonicalize=True,
              freeze_imp=False, verbose=None):
    '''Using DMET to produce CASSCF initial guess.  Return the active space
    size, num active electrons and the orbital initial guess.
    '''
    from pyscf import lo
    mol = mf.mol
    if isinstance(verbose, logger.Logger):
        log = verbose
    elif verbose is not None:
        log = logger.Logger(mol.stdout, verbose)
    else:
        log = logger.Logger(mol.stdout, mol.verbose)
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2): # ROHF/UHF DM
        dm = sum(dm)
    if base != 0:
        baslst = [i-base for i in baslst]
    if s is None:
        s = mf.get_ovlp()

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

    baslst = numpy.asarray(baslst)
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
        if (ne_error[i] < nelec_tol and
# whether all baths next to ith bath are less important
            (occb[i] < nelec_tol or occb[i] > 2-nelec_tol)):
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
        label = mol.spheric_labels(True)
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

    def search_for_degeneracy(e):
        idx = numpy.where(abs(e[1:] - e[:-1]) < 1e-6)[0]
        return numpy.unique(numpy.hstack((idx, idx+1)))
    def symmetrize(e, c):
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
                idx = numpy.argsort(es)
                assert(numpy.allclose(es[idx], esub))
                c[:,degidx] = numpy.hstack(cs)[:,idx]
        return c

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
                return symmetrize(e, numpy.dot(c, u))

        if freeze_imp:
            log.debug('Semi-canonicalization for freeze_imp=True')
            mo = numpy.hstack([trans(mocore), trans(mocas[:,:nimp]),
                               trans(mocas[:,nimp:]), trans(movir)])
        else:
            mo = numpy.hstack([trans(mocore), trans(mocas), trans(movir)])
    else:
        mo = numpy.hstack((mocore, mocas, movir))

    return ncas, nelecas, mo

def dynamic_cas_space_(mc, baslst, nelec_tol=.05, occ_cutoff=1e-6, base=0,
                       orth_method='meta_lowdin', s=None, canonicalize=True,
                       start_cycle=1, freeze_imp=False, verbose=None):
    '''Dynamically tune CASSCF active space based on DMET-CAS decomposition
    by following steps
    1. DMET-CAS decomposition to get new guess of core/active/external
    2. If the active space changes, update mo, eris, ncore, ncas  INPLACE
    3. Solving FCI problem in the new active space

    Note this method is in test.  It might give incorrect CASSCF answer.
    '''
    old_casci = mc.casci
    if isinstance(verbose, logger.Logger):
        log = verbose
    elif verbose is not None:
        log = logger.Logger(mc.stdout, verbose)
    else:
        log = logger.Logger(mc.stdout, mc.verbose)

    def casci(mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        if envs.get('imacro', 0) < start_cycle or 'casdm1' not in envs:
            return old_casci(mo_coeff, ci0, eris, verbose, envs)

        mol = mc.mol
        ncore = mc.ncore
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+mc.ncas]
        dm1 = numpy.dot(mocore, mocore.T) * 2
        dm1+= reduce(numpy.dot, (mocas, envs['casdm1'], mocas.T))
        ncas, nelecas, mo = guess_cas(mc._scf, dm1, baslst, nelec_tol,
                                      occ_cutoff, base, orth_method, s,
                                      freeze_imp, log)
        nelecb = (nelecas-mol.spin)//2
        neleca = nelecas - nelecb
        nelecas = (neleca, nelecb)

        if (mc.ncas == ncas and mc.nelecas == nelecas):
            return old_casci(mo_coeff, ci0, eris, verbose, envs)

        else:
            logger.note(mc, 'Change active space to CAS (%de+%de, %do)',
                        nelecas[0], nelecas[1], ncas)
            mc.ncas = ncas
            mc.nelecas = nelecas
            mc.ncore = (mol.nelectron - sum(nelecas)) // 2
            mo_coeff[:] = mo
            if mc.mol.symmetry:
                label_symmetry_(mc, mo_coeff)
                mc.fcisolver.orbsym = mc.orbsym[mc.ncore:mc.ncore+ncas]
            eris.__dict__.update(mc.ao2mo(mo).__dict__)

            mc.fcisolver._restart = False
            if freeze_imp:
                mc.frozen = range(mc.ncore, mc.ncore+len(baslst))
                logger.info(mc, 'Freeze impurity %s', mc.frozen)

            # hack casdm1_pref so that it has the same dimension as casdm1
            casdm1_last = envs['casdm1_last']
            casdm1_last[:] = 0
            casdm1_last.resize(ncas**2, refcheck=False)
            casdm1_last.shape = (ncas,)*2
            return old_casci(mo_coeff, None, eris, verbose, envs)

    if freeze_imp:
        mc.frozen = range(mc.ncore, mc.ncore+len(baslst))
        logger.info(mc, 'Freeze impurity %s', mc.frozen)

    mc.casci = casci
    return mc

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
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

    aolst = [i for i,s in enumerate(mol.spheric_labels(1)) if 'H 1s' in s]
    dm = mf.make_rdm1()
    ncas, nelecas, mo = guess_cas(mf, dm, aolst, verbose=4)
    mc = mcscf.CASSCF(mf, ncas, nelecas).set(verbose=4)
    mc = dynamic_cas_space_(mc, aolst)
    emc = mc.kernel(mo)[0]
    print(emc,0)

