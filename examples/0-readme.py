# import gto to setup molecule
from pyscf import gto
# import scf to run RHF,UHF etc
from pyscf import scf

'''
PySCF doesn't have its own input parser.  The input file is a Python script.
Before going throught the rest part, be sure the PySCF path is added in PYTHONPATH.
'''

mol = gto.M(
# gto.Mole is a class to hold molecule informations.
# verbose       print level, (1..5), big num prints more details.
#               It can be overwritten by command line options "-v" or "-q".
# output        name of output file.
#               the default value is None, which prints all things to sys.stdout
#               It can be overwritten by command line options "-o <filename>"
# charge        = 0 by default.
# spin          2 * S_z, n_alpha - n_beta. Default is 0.
# symmetry      True/False or specified symmetry symbol (Dooh, Coov, D2h, ...)
# MUST 'build' the molecule before doing anything else.
    verbose = 5,
    output = 'out_h2o',
# Coordinates are in Angstrom
    atom = '''
      O     0    0       0
      h     0    -.757   .587
     1      0    .757    .587''',
# basis = {atom_type/nuc_charge: 'name of basis sets'}
# or one name for all atoms such as   basis = 'cc-pvdz'
# case insensitive
    basis = '6-31g',
)
# For more details, see pyscf/gto/mole.py

#
# The package follow the convention that each method has its class to hold
# control parameters.  The calculation can be executed by the kernel function.
# Eg, to do Hartree-Fock, (1) create HF object, (2) call kernel function
#
mf = scf.RHF(mol)
print('E=%.15g' % mf.kernel())
