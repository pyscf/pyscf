#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to control DIIS parameters and how to use different
DIIS schemes (CDIIS, ADIIS, EDIIS) in SCF calculations.

Note the calculations in this example is served as a demonstration for the use
of DIIS.  Without other convergence technique, none of them can converge.
'''

from pyscf import gto, scf, dft
import scipy
import scipy.linalg
import time
import numpy.linalg
import numpy
import sys
import copy

atom_str = 'H 0.0 0.0 0.0; O 1.0 0.0 0.0'
mol = gto.M(atom=atom_str, basis='6-31+g*', spin=1, verbose=4)

mf = scf.RHF(mol)
mf.max_cycle = 150
#mf.diis = scf.EDIIS()

#unconverged_dm = mf.get_init_guess(mol)
#unconverged_fock = mf.get_fock(dm=unconverged_dm)
#mo_energy, mo_coeff = mf.eig(unconverged_fock, mf.get_ovlp())
#mo_occ = mf.get_occ(mo_energy, mo_coeff)
#unconverged_dm = mf.make_rdm1(mo_coeff, mo_occ)

mf.kernel()

scf.RHF(mol).newton().kernel()

diis_c = copy.copy(mf.mo_coeff)

#converged_dm = mf.make_rdm1()
diis_energy = mf.energy_tot()
diis_energy_e = mf.energy_elec()[0]


lieconverger = scf.M3SOSCF(scf.RHF(mol), 16)
#lieconverger.setCurrentDm(converged_dm)
#lieconverger.initLieAlgebraMatrix()
#lieconverger.setLastEigvalsFromCurrentDm(0.001)
#lieconverger.genDensityMatrixFromLieAlgebraCoeffs()

conv, m3_energy, mo_e, m3_c, occs = lieconverger.converge()
print(m3_energy - diis_energy)
print("DIIS: " + str(diis_energy_e))


#print(lieconverger.getEnergy(includeNuc=True))
#algMatrix, eigvals = lieconverger.getLieAlgebraCoeffs()

#sys.exit(0)

#gnconv, gnconvl = lieconverger.getNumericalDerivativeInLieRepresentation(algMatrix, eigvals)
#gaconv = lieconverger.getEnergyGradientInLieRepresentation(algMatrix, eigvals)

#lieconverger.setCurrentDm(unconverged_dm)
#algMatrix, eigvals = lieconverger.getLieAlgebraCoeffs()
#gnguess1, gnguess1l = lieconverger.getNumericalDerivativeInLieRepresentation(algMatrix, eigvals)
#gaguess1 = lieconverger.getEnergyGradientInLieRepresentation(algMatrix, eigvals)


#edm = mf.init_guess_by_minao(mol)
#unconverged_fock = mf.get_fock(dm=edm)
#mo_energy, mo_coeff = mf.eig(unconverged_fock, mf.get_ovlp())
#mo_occ = mf.get_occ(mo_energy, mo_coeff)
#
#converged_fock = mf.get_fock(dm=converged_dm)
#mo_energy2, mo_coeff2 = mf.eig(converged_fock, mf.get_ovlp())
#mo_occ2 = mf.get_occ(mo_energy2, mo_coeff2)
#
#uniMatrix = numpy.linalg.inv(mo_coeff) @ mo_coeff2
#print("UNI CHECK: " + str(numpy.linalg.norm(numpy.linalg.inv(uniMatrix) - uniMatrix.conj().T)))
#print("Conv Check: " + str(numpy.linalg.norm(mo_coeff @ uniMatrix - mo_coeff2)))


#amatrix = scipy.linalg.logm(uniMatrix)

#for i in range(len(amatrix)):
#    s = ""
#    for j in range(len(amatrix)):
#        if abs(amatrix[i][j]) < 10**-7:
#            s += "      "
#        else:
#            if abs(round(amatrix[i][j].real, 2)) == 0:
#                s += "0.00  "
#            else:
#                s += str(abs(round(amatrix[i][j].real, 2))) + "  "
    #print(s)

#print("INIT OCCS: " + str(mo_occ))
#print("FINAL OCCS: " + str(mo_occ2))





#lieconverger.setCurrentDm(edm)
#algMatrix, eigvals = lieconverger.getLieAlgebraCoeffs()
#gnguess2 = lieconverger.getNumericalDerivativeInLieRepresentation(algMatrix, eigvals)
#gaguess2 =  lieconverger.getEnergyGradientInLieRepresentation(algMatrix, eigvals)

#print("Conv Numerical:        " + str(gnconvl))
#print("Conv Analytic:         " + str(numpy.linalg.norm(gaconv)))
#print("Guess Minao Numerical: " + str(gnguess1l))
#print("Guess Minao Analytic:  " + str(numpy.linalg.norm(gaguess1)))
#print("Guess 1e Numerical:    " + str(numpy.linalg.norm(gnguess2)))
#print("Guess 1e Analytic:     " + str(numpy.linalg.norm(gaguess2)))
#print()
#print("Energy Conv:           " + str(mf.energy_tot(converged_dm)))
#print("Energy Minao:          " + str(mf.energy_tot(unconverged_dm)))
#print("Energy 1e:             " + str(mf.energy_tot(edm)))




