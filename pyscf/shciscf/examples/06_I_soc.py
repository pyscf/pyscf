from pyscf import gto, scf, mcscf
from pyscf.shciscf import shci, socutils
import numpy

# molecule object
# to carry out ecp calculation, spin-orbit pseudo potential has to be provided
# crenbl and crenbs ecp are currently the only available in pyscf, for more ecps
# visit http://www.tc.uni-koeln.de/PP/index.en.html it is only available in molpro format.
# or http://www.cosmologic-services.de/basis-sets/basissets.php the dhf-()-2c basis sets also have spin-orbit ecps.
mol = gto.M(
    verbose = 4,
    atom = "I 0 0 0",
    basis = 'cc-pvdz-pp',
    ecp = '''
I nelec 28
I ul
2      1.00000000             0.00000000
I S
2     40.03337600            49.98964900
2     17.30057600           281.00655600
2      8.85172000            61.41673900
I P
2     15.72014100            67.41623900   -134.832478
2     15.20822200           134.80769600    134.807696
2      8.29418600            14.56654800    -29.133096
2      7.75394900            28.96842200     28.968422
I D
2     13.81775100            35.53875600    -35.538756
2     13.58780500            53.33975900     35.559839
2      6.94763000             9.71646600     -9.716466
2      6.96009900            14.97750000      9.985000
I F
2     18.52295000           -20.17661800     13.451079
2     18.25103500           -26.08807700    -13.044039
2      7.55790100            -0.22043400      0.146956
2      7.59740400            -0.22164600     -0.110823
    ''',
    spin = 1)

# mean-field object
mf = scf.RHF(mol)
mf.kernel()

# state-average mcscf calculation with valence active space
mc = mcscf.CASSCF(mf, 4, 7).state_average_(numpy.ones(3)/3.0)
mc.internal_rotation = True
mc.mc1step()

# writes the SOC integrals, by default picture change effects are
# considered using breit-pauli Hamiltonian
socutils.writeSOCIntegrals(mc)

# perform a one-step SHCI calculation
mch = shci.SHCISCF(mf, 4, 7).state_average_(numpy.ones(6)/6.0)
mch.fcisolver.DoSOC = True
mch.fcisolver.DoRDM = False
# the first three doublet states are degenerate at non-relativistic calculations
# due to spin-orbit coupling, it splits into a four-fold degenerate state and
# a two-fold degenerate state
mch.fcisolver.nroots = 6
mch.fcisolver.sweep_iter = [0]
mch.fcisolver.sweep_epsilon = [1e-5]
mch.fcisolver.dets = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,7]]
shci.dryrun(mch, mc.mo_coeff)

e_ecp = shci.readEnergy(mch.fcisolver)

# then do a very small all electron calculation for reference
mol_all_elec = gto.M(
    verbose = 4,
    atom = 'I 0 0 0',
    basis = 'ano@6s5p3d1f',
    max_memory = 2000,
    spin = 1)
mf_nr = scf.RHF(mol_all_elec)
mf_nr.kernel()

mc_nr = mcscf.CASSCF(mf_nr, 4, 7).state_average_(numpy.ones(3)/3.0)
mc_nr.internal_rotation = True
mc_nr.mc2step()

socutils.writeSOCIntegrals(mc_nr, pictureChange1e = 'bp', pictureChange2e = 'bp')

mch_nr = shci.SHCISCF(mf_nr, 4, 7).state_average_(numpy.ones(6)/6.0)
mch_nr.fcisolver.DoSOC = True
mch_nr.fcisolver.DoRDM = False
mch_nr.fcisolver.nroots = 6
mch_nr.fcisolver.sweep_iter = [0]
mch_nr.fcisolver.sweep_epsilon = [1e-5]
mch_nr.fcisolver.dets = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,7]]
shci.dryrun(mch_nr, mc_nr.mo_coeff)

e_bp = shci.readEnergy(mch_nr.fcisolver)

mf_x2c = scf.sfx2c1e(scf.RHF(mol_all_elec))
mf_x2c.kernel()
mc_x2c = mcscf.CASSCF(mf_x2c, 4, 7).state_average_(numpy.ones(3)/3.0)
mc_x2c.mc2step()

socutils.writeSOCIntegrals(mc_x2c, pictureChange1e = 'x2c1', pictureChange2e = 'x2c')
mch_x2c = shci.SHCISCF(mf_x2c, 4, 7).state_average_(numpy.ones(6)/6.0)
mch_x2c.fcisolver.DoSOC = True
mch_x2c.fcisolver.DoRDM = False
mch_x2c.fcisolver.nroots = 6
mch_x2c.fcisolver.sweep_iter = [0]
mch_x2c.fcisolver.sweep_epsilon = [1e-5]
mch_x2c.fcisolver.dets = [[0,1,2,3,4,5,6],[0,1,2,3,4,5,7]]
shci.dryrun(mch_x2c, mc.mo_coeff)

e_x2c = shci.readEnergy(mch_x2c.fcisolver)

print(    "%11s%20s %12s%11s%12s %11s \n" %("", "SOECP(Har)   (cm^-1)", "  BP(Har) ","  (cm^-1)", " X2C(Har)", "cm^-1"))
for i in range(6):
    print("State %i : %10.4f %10.1f %12.4f %10.1f %12.4f %10.1f\n" %(i,
     e_ecp[i], (e_ecp[i]-e_ecp[0]) * 219470.,
     e_bp[i],  (e_bp[i] -e_bp[0] ) * 219470.,
     e_x2c[i], (e_x2c[i]-e_x2c[0]) * 219470.))
