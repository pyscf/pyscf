#!/usr/bin/env python

from functools import reduce
import numpy
import scipy.linalg
from pyscf import scf
from pyscf import gto
from pyscf import mcscf, fci

'''
Triplet and quintet energy gap of Iron-Porphyrin molecule

In this example, we use density matrix embedding theory
(ref. Q Sun, JCTC, 10(2014), 3784) to generate initial guess.
'''

#
# For 3d transition metal, people usually consider the so-called double
# d-shell effects for CASSCF calculation.  Double d-shell here refers to 3d
# and 4d atomic orbitals.  Density matrix embedding theory (DMET) provides a
# method to generate CASSCF initial guess in terms of localized orbitals.
# Given DMET impurity and truncated bath, we can select Fe 3d and 4d orbitals
# and a few entangled bath as the active space.
#


##################################################
#
# Define DMET active space
#
# This function is defined here as a simplified implementation of dmet_cas
# active space function.  It's recommended to use the mcscf.dmet_cas module to
# generate the DMET active space.  See also 43-dmet_cas.py
#
##################################################
def dmet_cas(mc, dm, implst):
    from pyscf import lo
    nao = mc.mol.nao_nr()
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nimp = len(implst)
    nbath = ncas - nimp
    corth = lo.orth.orth_ao(mol, method='meta_lowdin')
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    cinv = numpy.dot(corth.T, s)
    #
    # Sum over spin-orbital DMs, then transform spin-free DM to orthogonal basis
    #
    dm = reduce(numpy.dot, (cinv, dm[0]+dm[1], cinv.T))

    #
    # Decomposing DM to get impurity orbitals, doubly occupied core orbitals
    # and entangled bath orbitals.  Active space is consist of impurity plus
    # truncated bath.
    #
    implst = numpy.asarray(implst)
    notimp = numpy.asarray([i for i in range(nao) if i not in implst])
    occi, ui = scipy.linalg.eigh(-dm[implst][:,implst])
    occb, ub = scipy.linalg.eigh(-dm[notimp][:,notimp])
    bathorb = numpy.dot(corth[:,notimp], ub)
    imporb = numpy.dot(corth[:,implst], ui)
    mocore = bathorb[:,:ncore]
    mocas  = numpy.hstack((imporb, bathorb[:,ncore:ncore+nbath]))
    moext  = bathorb[:,ncore+nbath:]

    #
    # Restore core, active and external space to "canonical" form.  Spatial
    # symmetry is reserved in this canonicalization.
    #
    hf_orb = mc._scf.mo_coeff
    fock = reduce(numpy.dot, (s, hf_orb*mc._scf.mo_energy, hf_orb.T, s))

    fockc = reduce(numpy.dot, (mocore.T, fock, mocore))
    e, u = scipy.linalg.eigh(fockc)
    mocore = numpy.dot(mocore, u)
    focka = reduce(numpy.dot, (mocas.T, fock, mocas))
    e, u = scipy.linalg.eigh(focka)
    mocas = numpy.dot(mocas, u)
    focke = reduce(numpy.dot, (moext.T, fock, moext))
    e, u = scipy.linalg.eigh(focke)
    moext = numpy.dot(moext, u)

    #
    # Initial guess
    #
    mo_init = numpy.hstack((mocore, mocas, moext))
    return mo_init




##################################################
#
# Quintet
#
##################################################

mol = gto.Mole()
mol.atom = [
    ['Fe', (0.      , 0.0000  , 0.0000)],
    ['N' , (1.9764  , 0.0000  , 0.0000)],
    ['N' , (0.0000  , 1.9884  , 0.0000)],
    ['N' , (-1.9764 , 0.0000  , 0.0000)],
    ['N' , (0.0000  , -1.9884 , 0.0000)],
    ['C' , (2.8182  , -1.0903 , 0.0000)],
    ['C' , (2.8182  , 1.0903  , 0.0000)],
    ['C' , (1.0918  , 2.8249  , 0.0000)],
    ['C' , (-1.0918 , 2.8249  , 0.0000)],
    ['C' , (-2.8182 , 1.0903  , 0.0000)],
    ['C' , (-2.8182 , -1.0903 , 0.0000)],
    ['C' , (-1.0918 , -2.8249 , 0.0000)],
    ['C' , (1.0918  , -2.8249 , 0.0000)],
    ['C' , (4.1961  , -0.6773 , 0.0000)],
    ['C' , (4.1961  , 0.6773  , 0.0000)],
    ['C' , (0.6825  , 4.1912  , 0.0000)],
    ['C' , (-0.6825 , 4.1912  , 0.0000)],
    ['C' , (-4.1961 , 0.6773  , 0.0000)],
    ['C' , (-4.1961 , -0.6773 , 0.0000)],
    ['C' , (-0.6825 , -4.1912 , 0.0000)],
    ['C' , (0.6825  , -4.1912 , 0.0000)],
    ['H' , (5.0441  , -1.3538 , 0.0000)],
    ['H' , (5.0441  , 1.3538  , 0.0000)],
    ['H' , (1.3558  , 5.0416  , 0.0000)],
    ['H' , (-1.3558 , 5.0416  , 0.0000)],
    ['H' , (-5.0441 , 1.3538  , 0.0000)],
    ['H' , (-5.0441 , -1.3538 , 0.0000)],
    ['H' , (-1.3558 , -5.0416 , 0.0000)],
    ['H' , (1.3558  , -5.0416 , 0.0000)],
    ['C' , (2.4150  , 2.4083  , 0.0000)],
    ['C' , (-2.4150 , 2.4083  , 0.0000)],
    ['C' , (-2.4150 , -2.4083 , 0.0000)],
    ['C' , (2.4150  , -2.4083 , 0.0000)],
    ['H' , (3.1855  , 3.1752  , 0.0000)],
    ['H' , (-3.1855 , 3.1752  , 0.0000)],
    ['H' , (-3.1855 , -3.1752 , 0.0000)],
    ['H' , (3.1855  , -3.1752 , 0.0000)],
]
mol.basis = 'ccpvdz'
mol.verbose = 4
mol.output = 'fepor.out'
mol.spin = 4
mol.symmetry = True
mol.build()

mf = scf.ROHF(mol)
mf = scf.fast_newton(mf)

#
# CAS(8e, 11o)
#
# mcscf.approx_hessian approximates the orbital hessian.  It does not affect
# CASSCF results.
#
mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 11, 8))
# Function mol.search_ao_label returns the indices of the required AOs
# It is equivalent to the following expression
#idx = [i for i,s in enumerate(mol.ao_labels()) if 'Fe 3d' in s or 'Fe 4d' in s]
idx = mol.search_ao_label(['Fe 3d', 'Fe 4d'])
mo = dmet_cas(mc, mf.make_rdm1(), idx)

mc.fcisolver.wfnsym = 'Ag'
mc.kernel(mo)
#mc.analyze()
e_q = mc.e_tot  # -2244.82910509839
cas_q = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]







##################################################
#
# Triplet
#
##################################################
#
# Slow convergence is observed in the triplet state.  In this system, the CI
# coefficients and orbital rotation are strongly coupled.  Small orbital
# rotation leads to significant change of CI eigenfunction.  The micro iteration
# is not able to predict the right orbital rotations since the first order
# approximation for orbital gradients and CI hamiltonian are just too far to the
# exact value.
# 

mol.spin = 2
mol.build(0, 0)  # (0, 0) to avoid dumping input file again

mf = scf.ROHF(mol)
mf = scf.fast_newton(mf)

#
# CAS(8e, 11o)
#
mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 11, 8))
idx = mol.search_ao_label(['Fe 3d', 'Fe 4d'])
mo = dmet_cas(mc, mf.make_rdm1(), idx)
#
# 1. Small spin contaimination is observed for the default FCI solver.
#    Call fci.addons.fix_spin_ to force FCI wfn following the triplet state.
#
# 2. Without specifying wfnsym for fcisolver, it may converge to B2g or B3g
#    states.  The two states are very close to B1g solution (~ 1 mEh higher).
#
# 3. mc.frozen = ... to freeze the 4D orbitals in active space.  Without
#    doing so, it's possible for the optimizer to cross the barrier, and
#    mixing the 4d and 4s orbital, then converge to a nearby solution which
#    involves 4s orbitals.  The 4s character solution is energetically lower
#    than the target solution (~0.5 mEh).  But it has quite different active
#    space feature to the initial guess.
#
fci.addons.fix_spin_(mc.fcisolver, ss=2)  # Triplet, ss = S*(S+1)
mc.fcisolver.wfnsym = 'B1g'
mc.frozen = numpy.arange(mc.ncore+5, mc.ncore+10)  # 6th-10th active orbitals are Fe 4d
mc.kernel(mo)
mo = mc.mo_coeff

#
# Using the frozen-4d wfn as the initial guess,  we can converge the triplet
# to the correct active space
#
mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 11, 8))
fci.addons.fix_spin_(mc.fcisolver, ss=2)
mc.fcisolver.wfnsym = 'B1g'
mc.kernel(mo)
#mc.analzye()
e_t = mc.e_tot  # -2244.81493852189
cas_t = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]



print('E(T) = %.15g  E(Q) = %.15g  gap = %.15g' % (e_t, e_q, e_t-e_q))
# E(T) = -2244.81493852189  E(Q) = -2244.82910509839  gap = 0.0141665764999743

# The triplet and quintet active space are not perfectly overlaped
s = reduce(numpy.dot, (cas_t.T, mol.intor('cint1e_ovlp_sph'), cas_q))
print('Active space overlpa <T|Q> ~ %f' % numpy.linalg.det(s)) # 0.307691






##################################################
#
# Output the active space orbitals to molden format
#
##################################################
from pyscf import tools
tools.molden.from_mo(mol, 'triplet-cas.molden', cas_t)
tools.molden.from_mo(mol, 'quintet-cas.molden', cas_q)
