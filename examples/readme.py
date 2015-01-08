# Pyscf didn't implement its own input parser.  The input file is a Python
# script.  To ensure Python find the modules of Pyscf, e.g. scf, gto, 
# /path/to/pyscf  must be added to environment variable PYTHONPATH.

# import gto to setup molecule
from pyscf import gto
# import scf to run RHF,UHF etc
from pyscf import scf

mol = gto.Mole()
# gto.Mole is a class to hold molecule informations.
# verbose       print level, (1..5), big num prints more details.
#               It can be overwritten by command line options "-v" or "-q".
# output        name of output file.
#               the default value is None, which prints all things to sys.stdout
#               It can be overwritten by command line options "-o <filename>"
# max_memory    max. memories (in MB) to use. Default is lib.parameters.MEMORY_MAX
#               It can be overwritten by command line options "-m size"
#               Note this parameter is not accurately followed in the program.
#               The actual memory usage might be ~200 MB more than this value
# charge        = 0 by default.
# spin          2 * S_z, n_alpha - n_beta. Default is 0.
# symmetry      detect and use symmetry (up to D2h) or not. Default is
#               False which implies C1 symmetry.
# MUST 'build' the molecule before doing any other things
mol.build(
    verbose = 5,
    output = 'out_h2o',
# atom = [(atom_type/nuc_charge, (coordinates:0.,0.,0.)),
#              ... ]   Coordinates are in Angstrom
# case insensitive
    atom = [
    ['O' , (0. , 0.    , 0.  )],
    ['H' , (0. , -.757 , .587)],
    [1   , (0. , .757  , .587)]],
# basis = {atom_type/nuc_charge: 'name of basis sets'}
# or one name for all atoms such as   basis = 'cc-pvdz'
# case insensitive
    basis = {'O': '6-31g',
# or directly input the basis
#                 [[angular-1,
#                   (expnt1,    contraction-coeffs-for-expnt1),
#                   (expnt2,    contraction-coeffs-for-expnt2),
#                    ...]
#                  [angular-2,
#                   (...) ...] ]
             'H': [[0,
                    (5.4471780, 0.1562850),
                    (0.8245470, 0.9046910),],
                   [0,
                    (0.1831920, 1.0000000),]],
            },
)
# For more details, see pyscf/gto/mole.py

rhf = scf.RHF(mol)
# scf.RHF or scf.UHF is a class to do SCF. Taking RHF as an example
# RHF.verbose           copy from Mole.verbose. It canNOT be controled by
#                       command line options directly
# RHF.max_memory        copy from Mole.verbose. It canNOT be controled by
#                       command line options directly
# RHF.chkfile           file to store MO coefficients, orbital energies
# RHF.conv_threshold     default is 1e-10
# RHF.max_cycle     default is 50
# RHF.init_guess = 'name'  'name' is case-insensitivecan. It be one of
#                       '1e': first density matrix from Hcore
#                       'chkfile': read from RHF.chkfile
#                       'atom': superposition of occ-averaged atomic HF density
#                       'minao': project from minimal AO basis (by default)
# RHF.DIIS              is used by default. set to None to turn off DIIS.
# RHF.diis_space        default is 8
# RHF.diis_start_cycle  default is 3
# RHF.damp_factor       between 0 and 1. Default is 0 which does not play damp
# RHF.level_shift_factor  default is 0
# RHF.direct_scf        default is True. Do direct-SCF when RHF.max_memory
#                       is not large enough to hold the whole 2-e integrals
# RHF.direct_scf_threshold  Matrix elements are ignored if less than me. Default 1e-13
# An initial guess of density matrix can be passed to the scf method
# e.g., dm = numpy.eye(mol.nao_nr()); rhf.scf(dm)
print('E=%.15g' % rhf.scf())
# after doing SCF, RHF.mo_coeff, RHF.mo_energy, RHF.mo_occ, RHF.hf_energy,
# RHF.converged (True/Flase, to see if SCF converged) will be held in RHF class.
print(rhf.mo_coeff.shape)
print(rhf.mo_energy)
print(rhf.mo_occ)
print(rhf.hf_energy)
print(rhf.converged)
# For more detail of class RHF or UHF, see pyscf/scf/hf.py

