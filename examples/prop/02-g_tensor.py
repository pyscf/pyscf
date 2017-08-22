#!/usr/bin/env python

from pyscf import gto, scf, dft
from pyscf.prop import gtensor
mol = gto.M(atom='''
            C 0 0 0
            N 0 0 1.1747
            ''',
            basis='ccpvdz', spin=1, charge=0, verbose=3)
mf = scf.UHF(mol).run()
gobj = gtensor.uhf.GTensor(mf).set(verbose=4)
gobj.kernel()

#
# 2-electron SOC (SSO+SOO) is enabled for para-magnetic term by default.  Its
# contributions can be controlled by attributes dia_soc2e, para_soc2e
#
gobj.dia_soc2e = 'SSO+SOO'
gobj.para_soc2e = 'SSO+SOO'
gobj.so_eff_charge = True
gobj.kernel()


gobj.dia_soc2e = None
gobj.para_soc2e = 'SSO'
gobj.so_eff_charge = True
gobj.kernel()


gobj.dia_soc2e = None
gobj.para_soc2e = 'SOMF'
gobj.so_eff_charge = True
gobj.kernel()


gobj.dia_soc2e = None
gobj.para_soc2e = 'AMFI+SOMF'
gobj.so_eff_charge = True
gobj.kernel()


#
# Attribute so_eff_charge controls whether to use Koseki effective charge in
# 1-electron SOC integrals.  By default effective charge is used in
# diamagnetic term but not in paramagnetic term.
#
gobj.dia_soc2e = None
gobj.para_soc2e = None
gobj.so_eff_charge = True
gobj.kernel()


#
# Attribute gauge_orig controls whether to use GIAO.  GIAO is used by default.
#
gobj.gauge_orig = mol.atom_coord(1)  # on N atom
gobj.dia_soc2e = False
gobj.para_soc2e = True
gobj.so_eff_charge = False
gobj.kernel()


#
# In pure DFT (LDA, GGA), CPSCF has no effects.  Setting cphf=False can switch
# off CPSCF.
#
mf = dft.UKS(mol).set(xc='bp86').run()
gobj = gtensor.uks.GTensor(mf).set(verbose=4)
gobj.cphf = False
gobj.gauge_orig = (0,0,0)
gobj.kernel()


#
# Only UHF and UKS are supported in g-tensor module.  ROHF and ROKS need to be
# transfered to UHF or UKS before calling g-tensor methods.
#
mf = scf.RKS(mol).run()
mf = scf.convert_to_uhf(mf)
gobj = gtensor.uhf.GTensor(mf).set(verbose=4)
print(gobj.kernel())
