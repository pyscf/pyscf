#!/usr/bin/env python

'''
IP/EA-RADC 1RDM dipole moment calculations for water
'''

from pyscf import gto, scf, adc
import numpy as np
from pyscf import lib
import math

#WATER
mol = gto.Mole()
r = 0.957492
x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
y = r * math.cos(104.468205* math.pi/(2 * 180.0))
mol.atom = [
    ['O', (0., 0.    , 0)],
    ['H', (0., -x, y)],
    ['H', (0., x , y)],]
mol.basis = {'H': 'cc-pVDZ',
             'O': 'cc-pVDZ',}
mol.verbose = 0
mol.build()

mf = scf.RHF(mol)
mf.verbose = 0
mf.conv_tol = 1e-12
mf.kernel()

#DIPOLE MOMENT INTEGRALS
def get_dip_moments(adc_var):
    m_name = adc_var.method
    m_type = adc_var.method_type

    print()
    header = "*   1RDM Dipole Moment Contracted Integrals for " + m_type + '/' + m_name + "   *"
    print("*" * len(header))
    print(header)
    print("*" * len(header))

    hline = "-" * len(header)
    print(hline)
    print('Calculating dipole moment components...')
    dip_ints = -adc_var.mol.intor('int1e_r', comp = 3)

    for i in range(dip_ints.shape[0]):
        dip_ints[i] = np.dot(mf.mo_coeff.T, np.dot(dip_ints[i], mf.mo_coeff))

    #NUCLEAR
    print('Attaining nuclear charges and coordinates...')
    charges = myadc.mol.atom_charges()
    coords  = myadc.mol.atom_coords()
    nucl_dip = lib.einsum('i,ix->x', charges, coords)

    #Calculate GS/REF 1RDM
    print('Calculating Reference 1RDM...')
    rdm1_ref = myadc.make_ref_rdm1()

    #Calculate EXCITED STATE 1RDM
    print('Calculating Excited State 1RDM...')
    rdm1_exc = myadc.make_rdm1()

    #REF 1RDM CONT
    ref_dip = lib.einsum("xqr,qr->x", dip_ints, rdm1_ref) + nucl_dip

    #EXS 1RDM CONT
    exc_dip = lib.einsum("xqr,eqr->ex", dip_ints, rdm1_exc) + nucl_dip

    print(hline)
    print("Reference dipole moment (a.u.):")
    print(f"    X: {ref_dip[0]:10.3e}")
    print(f"    Y: {ref_dip[1]:10.3e}")
    print(f"    Z: {ref_dip[2]:10.3e}")
    print(hline)
    print(hline)

    for r in range(exc_dip.shape[0]):
        print(f"Excited state root {r} dipole moment (a.u.):")
        print(f"    X: {exc_dip[r][0]:10.3e}")
        print(f"    Y: {exc_dip[r][1]:10.3e}")
        print(f"    Z: {exc_dip[r][2]:10.3e}")
    print(hline)
    print('*'*len(header))
    print()
    print()
    print()

#ADC CALCS
myadc = adc.ADC(mf)

###IP CALCS
#IP-RADC(2) for 3 roots
myadc.verbose = 0
myadc.kernel(nroots = 3)
get_dip_moments(myadc)

#IP-RADC(2)-x for 1 root
myadc.method = "adc(2)-x"
myadc.kernel(nroots = 1)
get_dip_moments(myadc)

#IP-RADC(3) for 1 root
myadc.method = "adc(3)"
myadc.kernel(nroots = 1)
get_dip_moments(myadc)

###EA CALCS
#EA-RADC(2) for 3 roots
myadc.method = "adc(2)"
myadc.method_type = "ea"
myadc.kernel(nroots = 3)
get_dip_moments(myadc)

#EA-RADC(2)-x for 1 root
myadc.method = "adc(2)-x"
myadc.method_type = "ea"
myadc.kernel(nroots = 1)
get_dip_moments(myadc)

#EA-RADC(3) for 1 root
myadc.method = "adc(3)"
myadc.method_type = "ea"
myadc.kernel(nroots = 1)
get_dip_moments(myadc)

###EE CALCS
#EE-RADC(2) for 3 roots
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.kernel(nroots = 3)
get_dip_moments(myadc)

#EE-RADC(2)-x for 1 root
myadc.method = "adc(2)-x"
myadc.method_type = "ee"
myadc.kernel(nroots = 1)
get_dip_moments(myadc)

#EE-RADC(3) for 1 root
myadc.method = "adc(3)"
myadc.method_type = "ee"
myadc.kernel(nroots = 1)
get_dip_moments(myadc)
