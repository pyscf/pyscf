#!/usr/bin/env python

'''
Gamma-point ppRPA excitation energies in the hole-hole channel
for the nitrogen-vacancy (NV-) center in diamond
'''

import os

from pyscf.pbc import df, dft, gto, lib

# For ppRPA excitation energy of N-electron system in hole-hole channel
# mean field is (N+2)-electron

# nitrogen vacancy in diamond (NV-)
# see in Table.1 https://doi.org/10.1021/acs.jpclett.4c00184
cell = gto.Cell()
cell.build(unit='angstrom',
           a=[[7.136000, 0.000000, 0.000000],
              [0.000000, 7.136000, 0.000000],
              [0.000000, 0.000000, 7.136000]],
           atom=[
               ["C", (0.906073512160, 2.664323563008, 2.694339584224)],
               ["C", (0.894090626784, 2.684866651264, 6.241909316128)],
               ["C", (0.899084378176, 6.235829736704, 2.679298965664)],
               ["C", (0.892434989152, 6.244294788160, 6.243564953760)],
               ["C", (4.459048614208, 2.719457426496, 2.676951150304)],
               ["C", (4.441660180288, 2.664323563008, 6.229926430752)],
               ["C", (4.460040375488, 6.254734827520, 2.675959389024)],
               ["C", (4.456700798848, 6.235829736704, 6.236915564736)],
               ["C", (0.000121369088, 0.000121369088, -0.000121369088)],
               ["C", (-0.010102670688, -0.010102670688, 3.575517319296)],
               ["C", (-0.010102670688, 3.560482680704, 0.010102670688)],
               ["C", (0.008425625056, 3.562468772224, 3.573531227776)],
               ["C", (3.560482680704, -0.010102670688, 0.010102670688)],
               ["C", (3.562468772224, 0.008425625056, 3.573531227776)],
               ["C", (3.562468772224, 3.562468772224, -0.008425625056)],
               ["C", (0.892434989152, 0.892434989152, 0.891705154752)],
               ["C", (0.894090626784, 0.894090626784, 4.451133113248)],
               ["C", (0.899084378176, 4.456700798848, 0.900170206208)],
               ["C", (0.906073512160, 4.441660180288, 4.471676201504)],
               ["C", (4.456700798848, 0.899084378176, 0.900170206208)],
               ["C", (4.441660180288, 0.906073512160, 4.471676201504)],
               ["C", (4.460040375488, 4.460040375488, 0.881265115392)],
               ["C", (4.459048614208, 4.459048614208, 4.416542338016)],
               ["C", (-0.006901660896, 1.786905622208, 1.780569760480)],
               ["C", (0.002512956672, 1.784927308928, 5.351072569760)],
               ["C", (0.006719393184, 5.349206041920, 1.786793836768)],
               ["C", (-0.006901660896, 5.355430118208, 5.349094256480)],
               ["C", (3.557699819104, 1.801090513056, 1.799927287968)],
               ["C", (3.623443694336, 1.707240281568, 5.428759597120)],
               ["C", (3.570268719936, 5.363769868640, 1.772230010048)],
               ["C", (3.557699819104, 5.336072590720, 5.334909365632)],
               ["C", (2.675027363200, 2.675027363200, 0.892428602432)],
               ["C", (2.675779925760, 6.243436648480, 0.892563294432)],
               ["C", (2.675027363200, 6.243571340480, 4.460972401312)],
               ["C", (6.243436648480, 2.675779925760, 0.892563294432)],
               ["C", (6.243571340480, 2.675027363200, 4.460972401312)],
               ["C", (6.244306769504, 6.244306769504, 0.891693173408)],
               ["C", (6.243436648480, 6.243436648480, 4.460219838752)],
               ["C", (1.786905622208, -0.006901660896, 1.780569760480)],
               ["C", (1.784927308928, 0.002512956672, 5.351072569760)],
               ["C", (1.801090513056, 3.557699819104, 1.799927287968)],
               ["C", (1.707240281568, 3.623443694336, 5.428759597120)],
               ["C", (5.349206041920, 0.006719393184, 1.786793836768)],
               ["C", (5.355430118208, -0.006901660896, 5.349094256480)],
               ["C", (5.363769868640, 3.570268719936, 1.772230010048)],
               ["C", (5.336072590720, 3.557699819104, 5.334909365632)],
               ["C", (2.664323563008, 0.906073512160, 2.694339584224)],
               ["C", (2.684866651264, 0.894090626784, 6.241909316128)],
               ["C", (2.719457426496, 4.459048614208, 2.676951150304)],
               ["C", (2.664323563008, 4.441660180288, 6.229926430752)],
               ["C", (6.235829736704, 0.899084378176, 2.679298965664)],
               ["C", (6.244294788160, 0.892434989152, 6.243564953760)],
               ["C", (6.254734827520, 4.460040375488, 2.675959389024)],
               ["C", (6.235829736704, 4.456700798848, 6.236915564736)],
               ["C", (1.784927308928, 1.784927308928, -0.002512956672)],
               ["C", (1.707240281568, 1.707240281568, 3.512556305664)],
               ["C", (1.786905622208, 5.355430118208, 0.006901660896)],
               ["C", (1.801090513056, 5.336072590720, 3.578300180896)],
               ["C", (5.355430118208, 1.786905622208, 0.006901660896)],
               ["C", (5.336072590720, 1.801090513056, 3.578300180896)],
               ["C", (5.349206041920, 5.349206041920, -0.006719393184)],
               ["C", (5.363769868640, 5.363769868640, 3.565731280064)],
               ["N", (3.661895773440, 3.661895773440, 3.474104226560)],
           ],
           dimension=3,
           max_memory=90000,
           verbose=5,
           basis='cc-pvdz',
           # create a (N+2)-electron system
           charge=-3,
           precision=1e-12)

gdf = df.RSDF(cell)
gdf.auxbasis = "cc-pvdz-ri"
gdf_fname = 'gdf_ints.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

# =====> Part I. Restricted ppRPA <=====
# After SCF, PySCF might fail in Makov-Payne correction
# save chkfile to restart
chkfname = 'scf.chk'
if os.path.isfile(chkfname):
    kmf = dft.RKS(cell).rs_density_fit()
    kmf.xc = "b3lyp"
    kmf.exxdiv = None
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = lib.chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.RKS(cell).rs_density_fit()
    kmf.xc = "b3lyp"
    kmf.exxdiv = None
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.chkfile = chkfname
    kmf.kernel()

# direct diagonalization, N6 scaling
# ppRPA can be solved in an active space
from pyscf.pprpa.rpprpa_direct import RppRPADirect
pp = RppRPADirect(kmf, nocc_act=50, nvir_act=50, nelec="n+2")
# number of two-electron removal states to print
pp.hh_state = 50
# solve for singlet states
pp.kernel("s")
# solve for triplet states
pp.kernel("t")
pp.analyze()

# =====> Part II. Unrestricted ppRPA <=====
chkfname = 'uscf.chk'
if os.path.isfile(chkfname):
    kmf = dft.UKS(cell).rs_density_fit()
    kmf.xc = "b3lyp"
    kmf.exxdiv = None
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = lib.chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.UKS(cell).rs_density_fit()
    kmf.xc = "b3lyp"
    kmf.exxdiv = None
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.chkfile = chkfname
    kmf.kernel()

# direct diagonalization, N6 scaling
# ppRPA can be solved in an active space
from pyscf.pprpa.upprpa_direct import UppRPADirect
pp = UppRPADirect(kmf, nocc_act=50, nvir_act=50, nelec="n+2")
# number of two-electron addition states to print
pp.pp_state = 50
# solve ppRPA
pp.kernel()
pp.analyze()

