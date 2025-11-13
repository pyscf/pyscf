#!/usr/bin/env python

from pyscf import gto, scf,adc

mol = gto.M(atom='C 0 0 0; O 0 0 1.2', basis='augccpvdz')
mf = scf.RHF(mol).run()
umf = scf.UHF(mol).run()

#
# CVS approximation is available for both RADC and UADC, which can help evaluate the core-ionization energy.
# A CVS calculation can be implemented by setting the ncvs variable in radc/uadc to a non-zero value.
# The ncvs value n sets the n energetically lowest occupied orbitals as core orbital.
# Here we demonstrate the CVS-ADC calculation for the core-ionization energy of CO molecule.
#

#1.Calculate the first three singlet core-excited states of CO(C 1s as the core orbital) using CVS-RADC
myadc = adc.ADC(mf).set(verbose = 4,method = "adc(2)",ncvs = 1)
eip,vip,pip,xip = myadc.kernel(nroots=3)
myadc.analyze()

#2.Calculate the first three singlet core-excited states of CO(C 1s and O 1s as the core orbital) using CVS-UADC
myuadc = adc.ADC(umf).set(verbose = 4,method = "adc(2)-x",ncvs = 2)
eip,vip,pip,xip = myuadc.kernel(nroots=3)
myuadc.compute_dyson_mo()

#3. Using density fitting
myadc_df = adc.ADC(mf).set(verbose = 4,method = "adc(3)",ncvs = 1).density_fit('augccpvdz-ri')
eip,vip,pip,xip = myadc_df.kernel(nroots=3)
