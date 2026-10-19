#!/usr/bin/env python

'''
Term values of the valence states of O2, compared with experiment.

For a linear molecule the sigma_v parity and the orbital angular momentum of
the CI wavefunction cannot be labelled in the D2h subgroup. The
cylindrical-symmetry FCI solver (direct_spin1_cyl_sym) resolves them, which
allows each valence state to be targeted by its term symbol. The states are
calculated at their experimental equilibrium distances and the term values
Te (relative to X 3Sigma_g-) are compared with
Muller, Dallos, Lischka, Dubrovay, Szalay, Theor. Chem. Acc. 105, 227 (2001).
'''

from pyscf import gto, scf, mcscf
from pyscf.fci import direct_spin1_cyl_sym

HA2CM = 219474.631  # cm^-1 per hartree

def term_value(R, nelecas, wfnsym, singlet=False):
    # The closed-shell (spin=0) RHF solution of O2 is unstable, so the
    # converged open-shell reference is used for all states; the spin state is
    # selected by the active-space Sz (nelecas).
    mol = gto.M(
        atom = 'O 0 0 0; O 0 0 %s' % R,
        basis = 'cc-pvdz',
        symmetry = True,
        spin = 2,
    )
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 8, nelecas)
    mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
    if singlet:
        mc.fix_spin_(ss=0)
    if wfnsym is not None:
        mc.fcisolver.wfnsym = wfnsym
    return mc.kernel()[0]

# state, Re (exp., or MR-AQCC where no exp. value), active (na, nb), wfnsym,
# singlet, exp. Te (cm^-1)
states = [
    ('X 3Sigma_g-',  1.2075, (7, 5), None,   False,     0),
    ("A' 3Delta_u",  1.5129, (7, 5), 'E2uy', False, 34770),
    ('A 3Sigma_u+',  1.5197, (7, 5), 'A1u',  False, 35399),
    ('B 3Sigma_u-',  1.6042, (7, 5), 'A2u',  False, 49793),
    ('  3Pi_g',      1.4845, (7, 5), 'E1gx', False, None),
    ('a 1Delta_g',   1.2156, (6, 6), 'E2gx', True,   7918),
    ('b 1Sigma_g+',  1.2269, (6, 6), 'A1g',  True,  13195),
    ('c 1Sigma_u-',  1.5143, (6, 6), 'A2u',  True,  33057),
    ('  1Pi_g',      1.4610, (6, 6), 'E1gx', True,  None),
]

e0 = term_value(1.2075, (7, 5), None)
print('Term values Te (cm^-1), ground X 3Sigma_g- at R = 1.2075 Angstrom')
print('%-14s %10s %10s %10s' % ('state', 'this calc', 'exp.', 'diff'))
for label, R, nelecas, wfnsym, singlet, exp in states:
    if wfnsym is None and not singlet:
        e = e0
    else:
        e = term_value(R, nelecas, wfnsym, singlet)
    te = (e - e0) * HA2CM
    if exp is None:
        print('%-14s %10.0f %10s %10s' % (label, te, '-', '-'))
    else:
        print('%-14s %10.0f %10d %+10.0f' % (label, te, exp, te - exp))
