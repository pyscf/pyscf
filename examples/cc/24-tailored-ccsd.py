#!/usr/bin/env python

'''
An example how to use the ccsd.callback function to implement Tailored CCSD (TCCSD).
Writes energies of HF, CCSD, CASCI(6,6), and TCCSD(6,6) to 'energies.txt'

See: J. Chem. Phys. 123, 074106 (2005); https://doi.org/10.1063/1.2000251
'''

from functools import partial
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.ci
import pyscf.mcscf
from pyscf.mp.mp2 import _mo_without_core


einsum = partial(np.einsum, optimize=True)

def make_tailored_ccsd(cc, cas):
    """Create tailored CCSD calculation."""

    nelec_cas = sum(cas.nelecas)
    nocc_cas = nelec_cas//2
    # Determine (MO|CAS) overlap:
    mo_cc = _mo_without_core(cc, cc.mo_coeff)
    nocc_cc = cc.get_nocc()
    mo_cc_occ = mo_cc[:,:nocc_cc]
    mo_cc_vir = mo_cc[:,nocc_cc:]
    mo_cas = cas.mo_coeff[:,cas.ncore:cas.ncore+cas.ncas]
    mo_cas_occ = mo_cas[:,:nocc_cas]
    mo_cas_vir = mo_cas[:,nocc_cas:]
    ovlp = cc._scf.get_ovlp()
    pocc = np.linalg.multi_dot((mo_cc_occ.T, ovlp, mo_cas_occ))
    pvir = np.linalg.multi_dot((mo_cc_vir.T, ovlp, mo_cas_vir))

    def get_cas_t1t2(cas):
        """Get T1 and T2 amplitudes from FCI wave function."""
        cisdvec = pyscf.ci.cisd.from_fcivec(cas.ci, cas.ncas, nelec_cas)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, cas.ncas, nocc_cas)
        assert (abs(c0) > 1e-8)
        t1 = c1/c0
        t2 = c2/c0 - einsum('ia,jb->ijab', t1, t1)
        return t1, t2

    t1cas_fci, t2cas_fci = get_cas_t1t2(cas)

    def callback(kwargs):
        """Tailor CCSD amplitudes within CAS."""
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        # Project CCSD amplitudes onto CAS:
        t1cas_cc = einsum('IA,Ii,Aa->ia', t1, pocc, pvir)
        t2cas_cc = einsum('IJAB,Ii,Jj,Aa,Bb->ijab', t2, pocc, pocc, pvir, pvir)
        # Take difference FCI-CCSD within CAS:
        dt1 = (t1cas_fci - t1cas_cc)
        dt2 = (t2cas_fci - t2cas_cc)
        # Rotate difference to CCSD space:
        dt1 = einsum('ia,Ii,Aa->IA', dt1, pocc, pvir)
        dt2 = einsum('ijab,Ii,Jj,Aa,Bb->IJAB', dt2, pocc, pocc, pvir, pvir)
        # Add difference:
        t1 += dt1
        t2 += dt2

    cc.callback = callback
    return cc

# --- Example: Dissociation of N2

t1 = t2 = None
for d in np.arange(0.8, 2.9, 0.2):

    mol = pyscf.gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 %f' % d
    mol.basis = 'cc-pVDZ'
    mol.verbose = 4
    mol.build()

    # Hartree Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    assert mf.converged

    # CCSD
    cc = pyscf.cc.CCSD(mf)
    cc.kernel(t1=t1, t2=t2)
    t1, t2 = cc.t1, cc.t2
    e_ccsd = (cc.e_tot if cc.converged else np.nan)

    # CASCI(6,6)
    cas = pyscf.mcscf.CASCI(mf, 6, 6)
    cas.kernel()
    assert cas.converged

    # TCCSD tailored with CASCI(6,6)
    tcc = pyscf.cc.CCSD(mf)
    tcc = make_tailored_ccsd(tcc, cas)
    tcc.kernel()
    assert tcc.converged

    with open('energies.txt', 'a') as f:
        f.write('%.2f  %.8f  %.8f  %.8f  %.8f\n' % (d, mf.e_tot, e_ccsd, cas.e_tot, tcc.e_tot))
