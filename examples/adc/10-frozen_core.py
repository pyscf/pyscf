#!/usr/bin/env python

import math
from pyscf import gto, scf, adc

mol = gto.Mole()
r = 0.957492
x = r * math.sin(104.468205 * math.pi/(2 * 180.0))
y = r * math.cos(104.468205* math.pi/(2 * 180.0))
mol.atom = [
    ['O', (0., 0.    , 0)],
    ['H', (0., -x, y)],
    ['H', (0., x , y)],]
mol.basis = {'H': 'aug-cc-pVDZ',
                'O': 'aug-cc-pVDZ',}
mol.build()
mf = scf.RHF(mol).run()

#
# In RADC, frozen orbitals can be specified by setting the frozen parameter when instantiating an RADC class.
# It can be an integer, a list of orbital indices, or 'chemcore' to automatically freeze the chemical core orbitals.
#

#
# Freeze the inner most two orbitals.
#
myadc = adc.ADC(mf,frozen=2)
#EA-RADC(3) for 3 roots and properties
myadc.verbose = 4
myadc.compute_properties = True
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel(nroots = 3)
myadc.analyze()

#
# Auto-generate the number of core orbitals to be frozen.
#
from pyscf.data import elements
myadc = adc.ADC(mf,frozen = elements.chemcore(mol))
print('Number of core orbital frozen: %d' % myadc.frozen)
myadc.verbose = 4
myadc.method = "adc(3)"
myadc.method_type = "ee"
eea,vea,pea,xea = myadc.kernel(nroots = 3)

#
# Freeze orbitals based on the list of indices.
#
myadc = adc.ADC(mf,frozen = [0,1,38,39,40])
myadc.verbose = 4
myadc.method = "adc(3)"
myadc.method_type = "ip"
eea,vea,pea,xea = myadc.kernel(nroots = 3)


#
# In UADC, this parameter should be announced as None or a array-like object with two elements
#
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
mf = scf.UHF(mol).run()

#
# Freeze the inner most one alpha orbital.
#
myadc = adc.ADC(mf,frozen=(1,0))
#EA-UADC(3) for 4 roots
myadc.verbose = 4
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel(nroots=4)

#
# Auto-generate the number of core orbitals to be frozen.
#
myadc = adc.ADC(mf,frozen = (elements.chemcore(mol), elements.chemcore(mol)))
print('Number of core alpha orbital frozen: %d' % myadc.frozen[0])
print('Number of core beta orbital frozen: %d' % myadc.frozen[1])
myadc.verbose = 4
myadc.method = "adc(3)"
myadc.method_type = "ee"
eea,vea,pea,xea = myadc.kernel(nroots = 3)


#
# Freeze orbitals based on the list of indices.
#
myadc = adc.ADC(mf,frozen = ([0,1,29,30,31],[0,1,29,30,31]))
myadc.verbose = 4
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel(nroots = 3)


#
# In ROADC, this parameter should be announced as None or an integer or a list of orbital indices
#
mf = scf.ROHF(mol).run()
#
# Freeze the inner most one orbital.
#
myadc = adc.ADC(mf,frozen=[1,1])
myadc.verbose = 4
myadc.method = "adc(3)"
myadc.method_type = "ee"
myadc.compute_properties = True
myadc.compute_spin_square = True
eee,vee,pee,xee = myadc.kernel(nroots=4)
myadc.analyze()
