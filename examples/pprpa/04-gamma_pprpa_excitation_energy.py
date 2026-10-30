#!/usr/bin/env python

'''
Gamma-point ppRPA excitation energies in the particle-particle channel
for a carbon vacancy in diamond
'''

import os

from pyscf.pbc import df, dft, gto, lib

# For ppRPA excitation energy of N-electron system in particle-particle channel
# mean field is (N-2)-electron

# carbon vacancy in diamond
# see Table.1 in https://doi.org/10.1021/acs.jctc.4c00829
cell = gto.Cell()
cell.build(unit='angstrom',
           a=[[7.136000, 0.000000, 0.000000],
              [0.000000, 7.136000, 0.000000],
              [0.000000, 0.000000, 7.136000]],
           atom=[
               ["C", (0.001233209267, 0.001233209267, 1.777383281725)],
               ["C", (0.012225089066, 1.794731051862, 3.561871956432)],
               ["C", (1.782392880032, 1.782392880032, 5.362763822515)],
               ["C", (1.780552680464, -0.013167298675, -0.008776569968)],
               ["C", (3.557268948138, 0.012225089066, 1.790128043568)],
               ["C", (5.339774910934, 1.794731051862, 1.790128043568)],
               ["C", (-0.013167298675, 1.780552680464, -0.008776569968)],
               ["C", (1.782392880032, 3.569607119968, -0.010763822515)],
               ["C", (1.794731051862, 5.339774910934, 1.790128043568)],
               ["C", (2.671194343578, 0.900046784093, 0.881569117555)],
               ["C", (0.887735886768, 0.887735886768, 6.244403380954)],
               ["C", (0.900046784093, 2.680805656422, 4.470430882445)],
               ["C", (2.680805656422, 4.451953215907, 0.881569117555)],
               ["C", (0.895786995277, 6.238213500378, 0.896853305635)],
               ["C", (0.930821705350, 0.930821705350, 2.673641624787)],
               ["C", (4.421178294650, 0.930821705350, 2.678358375213)],
               ["C", (0.900046784093, 2.671194343578, 0.881569117555)],
               ["C", (6.238213500378, 0.895786995277, 0.896853305635)],
               ["C", (2.680805656422, 0.900046784093, 4.470430882445)],
               ["C", (4.451953215907, 2.680805656422, 0.881569117555)],
               ["C", (0.930821705350, 4.421178294650, 2.678358375213)],
               ["C", (1.794731051862, 0.012225089066, 3.561871956432)],
               ["C", (0.012225089066, 3.557268948138, 1.790128043568)],
               ["C", (3.569607119968, 1.782392880032, -0.010763822515)],
               ["C", (1.736746319267, 1.736746319267, 1.671367479693)],
               ["C", (5.351404126874, 0.000595873126, 0.004129648157)],
               ["C", (0.000595873126, 5.351404126874, 0.004129648157)],
               ["C", (2.676000000000, 2.676000000000, 6.244000000000)],
               ["C", (6.244000000000, 2.676000000000, 2.676000000000)],
               ["C", (2.676000000000, 6.244000000000, 2.676000000000)],
               ["C", (0.000595873126, 0.000595873126, 5.347870351843)],
               ["C", (0.001233209267, 5.350766790733, 3.574616718275)],
               ["C", (1.780552680464, 5.365167298675, 5.360776569968)],
               ["C", (3.571447319536, -0.013167298675, 5.360776569968)],
               ["C", (5.365167298675, 1.780552680464, 5.360776569968)],
               ["C", (5.365167298675, 3.571447319536, -0.008776569968)],
               ["C", (5.350766790733, 5.350766790733, 1.777383281725)],
               ["C", (4.464264113232, 0.887735886768, 6.243596619046)],
               ["C", (4.451953215907, 2.671194343578, 4.470430882445)],
               ["C", (2.671194343578, 4.451953215907, 4.470430882445)],
               ["C", (0.895786995277, 6.249786499622, 4.455146694365)],
               ["C", (4.421178294650, 4.421178294650, 2.673641624787)],
               ["C", (6.249786499622, 4.456213004723, 0.896853305635)],
               ["C", (0.887735886768, 4.464264113232, 6.243596619046)],
               ["C", (6.249786499622, 0.895786995277, 4.455146694365)],
               ["C", (4.456213004723, 6.249786499622, 0.896853305635)],
               ["C", (5.350766790733, 0.001233209267, 3.574616718275)],
               ["C", (-0.013167298675, 3.571447319536, 5.360776569968)],
               ["C", (3.571447319536, 5.365167298675, -0.008776569968)],
               ["C", (3.615253680733, 1.736746319267, 3.680632520307)],
               ["C", (3.615253680733, 3.615253680733, 1.671367479693)],
               ["C", (6.244000000000, 2.676000000000, 6.244000000000)],
               ["C", (6.244000000000, 6.244000000000, 2.676000000000)],
               ["C", (1.736746319267, 3.615253680733, 3.680632520307)],
               ["C", (2.676000000000, 6.244000000000, 6.244000000000)],
               ["C", (3.557268948138, 5.339774910934, 3.561871956432)],
               ["C", (5.351404126874, 5.351404126874, 5.347870351843)],
               ["C", (3.569607119968, 3.569607119968, 5.362763822515)],
               ["C", (5.339774910934, 3.557268948138, 3.561871956432)],
               ["C", (4.464264113232, 4.464264113232, 6.244403380954)],
               ["C", (4.456213004723, 6.238213500378, 4.455146694365)],
               ["C", (6.238213500378, 4.456213004723, 4.455146694365)],
               ["C", (6.244000000000, 6.244000000000, 6.244000000000)],
           ],
           dimension=3,
           max_memory=90000,
           verbose=5,
           basis='cc-pvdz',
           # create a (N-2)-electron system
           charge=2,
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
pp = RppRPADirect(kmf, nocc_act=50, nvir_act=50)
# number of two-electron addition states to print
pp.pp_state = 50
# solve for singlet states
pp.kernel("s")
# solve for triplet states
pp.kernel("t")
pp.analyze()

# Davidson algorithm, N4 scaling
# ppRPA can be solved in an active space
from pyscf.pprpa.rpprpa_davidson import RppRPADavidson
pp = RppRPADavidson(kmf, nocc_act=50, nvir_act=50, nroot=50)
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
pp = UppRPADirect(kmf, nocc_act=50, nvir_act=50)
# number of two-electron addition states to print
pp.pp_state = 50
# solve ppRPA
pp.kernel()
pp.analyze()

