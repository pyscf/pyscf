#!/usr/bin/env python

'''
IP/EA-RADC 1RDM dipole moment calculations for water
'''

from pyscf import gto, scf, adc
import numpy as np
from pyscf import lib
import math

#WATER
r = 0.969286393
mol = gto.Mole()
mol.atom = [
    ['O', (0., 0.    , -r/2   )],
    ['H', (0., 0.    ,  r/2)],]
mol.basis = {'O':'aug-cc-pvdz',
                'H':'aug-cc-pvdz'}
mol.verbose = 0
mol.symmetry = False
mol.spin  = 1
mol.build()

mf = scf.UHF(mol)
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
    #Calculate GS/REF 1RDM
    print('Calculating Reference 1RDM...')
    rdm1_ref = myadc.make_ref_rdm1()

    #Calculate EXCITED STATE 1RDM
    print('Calculating Excited State 1RDM...')
    rdm1_exc_i = myadc.make_rdm1()
    rdm1_exc = (np.array(rdm1_exc_i[0]),np.array(rdm1_exc_i[1]))

    #REF 1RDM CONT
    ref_dip = lib.einsum("xqr,qr->x", adc_var.dip_mom[0], rdm1_ref[0]) + \
                         lib.einsum("xqr,qr->x", adc_var.dip_mom[1], rdm1_ref[1]) + adc_var.dip_mom_nuc

    #EXS 1RDM CONT
    exc_dip = lib.einsum("xqr,eqr->ex", adc_var.dip_mom[0], rdm1_exc[0]) + \
                         lib.einsum("xqr,eqr->ex", adc_var.dip_mom[1], rdm1_exc[1]) + adc_var.dip_mom_nuc

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
