#!/usr/bin/env python
#
# Contributors:
#       Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
from pyscf import scf
from pyscf import gto
from pyscf import mcscf
from pyscf import dmrgscf
from pyscf import mrpt
#
# Adjust mpi runtime schedular to execute the calculation with multi-processor
#
# NOTE DMRG-NEVPT2 requires about 10 GB memory per processor in this example
#
dmrgscf.settings.MPIPREFIX = 'mpirun -np 8'


'''
Triplet and quintet energy gap of Iron-Porphyrin molecule using DMRG-CASSCF
and DMRG-NEVPT2 methods.  DMRG is an approximate FCI solver.  It can be used
to handle large active space.  This example is the next step to example
018-dmet_cas_for_feporph.py
'''

#
# Following 018-dmet_cas_for_feporph.py, we still use density matrix embedding
# theory (DMET) to generate CASSCF initial guess.  The active space includes
# the Fe double d-shell, 4s shell, and the ligand N 2pz orbitals to describe
# metal-ligand pi bond and pi backbond.
#


##################################################
#
# Define DMET active space
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
mol.output = 'fepor-dmrgscf.out'
mol.spin = 4
mol.symmetry = True
mol.build()

mf = scf.ROHF(mol)
mf = scf.fast_newton(mf)

#
# CAS(16e, 20o)
#
# mcscf.approx_hessian approximates the orbital hessian.  It does not affect
# results.  The N-2pz orbitals introduces more entanglement to environment.
# 5 bath orbitals which have the strongest entanglement to impurity are
# considered in active space.
#
mc = mcscf.approx_hessian(dmrgscf.dmrgci.DMRGSCF(mf, 20, 16))
# Function mol.search_ao_label returns the indices of the required AOs
# It is equivalent to the following expression
#idx = [i for i,s in enumerate(mol.ao_labels())
#       if 'Fe 3d' in s or 'Fe 4d' in s or 'Fe 4s' in s or 'N 2pz' in s]
idx = mol.search_ao_label(['Fe 3d', 'Fe 4d', 'Fe 4s', 'N 2pz'])
mo = dmet_cas(mc, mf.make_rdm1(), idx)

mc.fcisolver.wfnsym = 'Ag'
mc.kernel(mo)
#mc.analyze()
e_q = mc.e_tot  # -2244.90267106288
cas_q = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]

#
# call DMRG-NEVPT2 (about 2 days, 100 GB memory)
#
ept2_q = mrpt.NEVPT(mc).kernel()





##################################################
#
# Triplet
#
##################################################

mol.spin = 2
mol.build(0, 0)

mf = scf.ROHF(mol)
mf = scf.fast_newton(mf)

#
# CAS(16e, 20o)
#
# Unlike CAS(8e, 11o) which is easily to draw 4s-character orbitals into the
# active space, the larger active space, which includes 4s orbitals, does not
# have such issue on MCSCF wfn.
#
mc = mcscf.approx_hessian(dmrgscf.dmrgci.DMRGSCF(mf, 20, 16))
idx = mol.search_ao_label(['Fe 3d', 'Fe 4d', 'Fe 4s', 'N 2pz'])
mo = dmet_cas(mc, mf.make_rdm1(), idx3d)
mc.fcisolver.wfnsym = 'B1g'
mc.kernel(mo)
mo = mc.mo_coeff
#mc.analzye()
e_t = mc.e_tot  # -2244.88920313881
cas_t = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]

#
# call DMRG-NEVPT2 (about 2 days, 100 GB memory)
#
ept2_t = mrpt.NEVPT(mc).kernel()

print('E(T) = %.15g  E(Q) = %.15g  gap = %.15g' % (e_t, e_q, e_t-e_q))
# E(T) = -2244.88920313881  E(Q) = -2244.90267106288  gap = 0.0134679240700279

# The triplet and quintet active space are not perfectly overlaped
s = reduce(numpy.dot, (cas_t.T, mol.intor('cint1e_ovlp_sph'), cas_q))
print('Active space overlpa <T|Q> ~ %f' % numpy.linalg.det(s))

print('NEVPT2: E(T) = %.15g  E(Q) = %.15g' % (ept2_t, ept2_q))
# E(T) = -3.52155285166390  E(Q) = -3.46277436661638






##################################################
#
# Output the active space orbitals to molden format
#
##################################################
from pyscf import tools
tools.molden.from_mo(mol, 'triplet-cas.molden', cas_t)
tools.molden.from_mo(mol, 'quintet-cas.molden', cas_q)
