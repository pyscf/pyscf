#!/usr/bin/env python
#
# Contributors:
#       Zhendong Li <zhendongli2008@gmail.com>
#       Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto, scf, mcscf
from pyscf import tools

mol = gto.Mole()
mol.verbose = 4
mol.output = 'hs.out'
mol.atom = '''
  Mo  7.34411020207581      1.17495097908005      6.72284263905920
  Fe  7.84036274632279      2.40948832380662      3.90857987198295
  S   8.11413397508734      3.34683967317511      5.92473122721237
  Cl  9.42237962692288      2.83901882053830      2.40523971787167
  S   7.63129189448338      0.24683725427736      4.48256715460659
  Cl  5.78420653505383      3.15381896731458      3.13969003482939
  N   7.05276738605521      2.42445066370842      8.66076404425459
  N   6.64167403727998     -1.00707407196440      7.11600799320614
  N   5.24002742536262      1.70306900993116      5.97156233521481
  H   6.27407522563538      2.37009344884271      9.32021452836747
  H   7.93656914286549      2.22405698642280      9.14675757456406
  H   7.12313637861828      3.37423478174186      8.26848891472229
  H   4.53157107313027      2.02429015953190      6.63557341863725
  H   5.36325579034589      2.43796505637839      5.24458486826946
  H   4.86812692298093      0.90155604764634      5.45231738540969
  H   5.82209673287966     -1.30027608533573      7.65147593202357
  H   6.56861368828978     -1.33871396574670      6.14286445056596
  H   7.48831993436433     -1.42577562013418      7.51985443522153
  S   9.40594188780477      0.42545761808747      7.87277304102829
  H   8.82966334944139     -0.10099345030206      8.99111747895267
'''
mol.basis = 'tzp-dkh'
mol.charge = -1
mol.spin = 8
mol.build()

#
# X2C correction for relativistic effects
#
# first pass, to generate initial guess
#
mf = scf.UHF(mol).x2c()
mf.chkfile = 'hs.chk'
mf.level_shift = 0.1
mf.conv_tol = 1e-2
mf.kernel()
#
# second pass to converge SCF calculation
#
mf = mf.newton()
mf.conv_tol = 1e-12
mf.kernel()




##################################################
#
# Analyze SCF results and make MCSCF initial guess
#
##################################################

# This parameter to control the call to matplotlib
ifplot = False

def sqrtm(s):
    e, v = numpy.linalg.eigh(s)
    return numpy.dot(v*numpy.sqrt(e), v.T.conj())

def lowdin(s):
    e, v = numpy.linalg.eigh(s)
    return numpy.dot(v/numpy.sqrt(e), v.T.conj())

##################################################
#
# 1. Read UHF-alpha/beta orbitals from chkfile
#
##################################################
fname = 'hs.chk'
chkfile = fname
mol, mf = scf.chkfile.load_scf(chkfile)
mo_coeff = mf["mo_coeff"]
ma = mo_coeff[0]
mb = mo_coeff[1]
nb = ma.shape[1]
nalpha = (mol.nelectron+mol.spin)/2
nbeta  = (mol.nelectron-mol.spin)/2
print('Nalpha = %d, Nbeta %d, Sz = %d, Norb = %d' % (nalpha, nbeta, mol.spin, nb))

#=============================
# DUMP from chkfile to molden
#=============================
#
# One can view the orbitals in many visualization tool like Jmol, IBOviewer
#
moldenfile = fname+'0.molden'
tools.molden.from_chkfile(moldenfile, chkfile)
if 0:
    # Jmol script to generate orbital images.  Run this jmol script in command line
    #       jmol hs.spt
    # It writes out images for 10 HOMO and 10 LUMO orbitals.
    tools.molden.from_mo(mol, fname+'_alpha.molden', ma)
    tools.molden.from_mo(mol, fname+'_beta.molden', mb)
    jmol_script = 'hs.spt'
    fspt = open(jmol_script,'w')
    fspt.write('''
initialize;
set background [xffffff];
set frank off
set autoBond true;
set bondRadiusMilliAngstroms 66;
set bondTolerance 0.5;
set forceAutoBond false;
#cd /home/abc/pyscf/examples/dmrg
''')
    fspt.write('load %s_beta.molden;\n' % fname)
    fspt.write('rotate -30 y;\n')
    fspt.write('rotate 20 x;\n')
    for i in range(nalpha-10,nalpha+10):
        fspt.write('isoSurface MO %d fill noMesh noDots;\n' % (i+1))
        fspt.write('#color isoSurface translucent 0.6 [x00ff00];\n')
        fspt.write('write JPG 90 "%s-alpha-%d.jpg";\n' % (jmol_script, (i+1)))

    fspt.write('load %s_alpha.molden;\n' % fname)
    fspt.write('rotate -30 y;\n')
    fspt.write('rotate 20 x;\n')
    for i in range(nbeta-10,nbeta+10):
        fspt.write('isoSurface MO %d fill noMesh noDots;\n' % (i+1))
        fspt.write('#color isoSurface translucent 0.6 [x0000ff];\n')
        fspt.write('write JPG 90 "%s-beta-%d.jpg";\n' % (jmol_script, (i+1)))
    fspt.close()


##################################################
#
# 2. Sanity check, using eg orthogonality
#
##################################################
ova = mol.intor_symmetric("cint1e_ovlp_sph")
diff = reduce(numpy.dot,(mo_coeff[0].T,ova,mo_coeff[0])) - numpy.identity(nb)
print(numpy.linalg.norm(diff))
diff = reduce(numpy.dot,(mo_coeff[1].T,ova,mo_coeff[1])) - numpy.identity(nb)
print(numpy.linalg.norm(diff))

#=============================
# Natural orbitals
# Lowdin basis X=S{-1/2}
# psi = chi * C
#     = chi' * C'
#     = chi*X*(X{-1}C')
#=============================
pTa = numpy.dot(ma[:,:nalpha],ma[:,:nalpha].T)
pTb = numpy.dot(mb[:,:nbeta],mb[:,:nbeta].T)
pT = pTa+pTb
pT = 0.5*pT
# Lowdin basis
s12 = sqrtm(ova)
s12inv = lowdin(ova)
pT = reduce(numpy.dot,(s12,pT,s12))
print('idemponency of DM: %s' % numpy.linalg.norm(pT.dot(pT)-pT))
enorb = mf["mo_energy"]
print('\nCMO_enorb:')
print(enorb)
if ifplot:
    import matplotlib.pyplot as plt
    plt.plot(range(nb),enorb[0],'ro')
    plt.plot(range(nb),enorb[1],'bo')
    plt.show()

#
# Non-orthogonal cases: FC=SCE
# Fao = SC*e*C{-1} = S*C*e*Ct*S
# OAO basis:
# F = Xt*Fao*X = S1/2*C*e*Ct*S1/2
#
fa = reduce(numpy.dot,(ma,numpy.diag(enorb[0]),ma.T))
fb = reduce(numpy.dot,(mb,numpy.diag(enorb[1]),mb.T))
fav = (fa+fb)/2
fock_sf = fOAO = reduce(numpy.dot,(s12,fav,s12))
#
# Small level shift on density matrix to break occupation degeneracy in natual
# orbitals
#
shift = 1e-7
pTshift = pT + shift*fOAO
#
# 'natural' occupations and orbitals
#
eig,coeff = scipy.linalg.eigh(pTshift)
eig = 2*eig
print('Natual occupancy %s ' % eig)
eig[abs(eig)<1.e-14]=0.0
if ifplot:
    import matplotlib.pyplot as plt
    plt.plot(range(nb),eig,'ro')
    plt.show()
#
# Rotate back to AO representation and check orthogonality
#
coeff = numpy.dot(s12inv,coeff)
ova = mol.intor_symmetric("cint1e_ovlp_sph")
diff = reduce(numpy.dot,(coeff.T,ova,coeff)) - numpy.identity(nb)
print('CtSC-I',numpy.linalg.norm(diff))


##################################################
#
# 3. Search for active space
#
##################################################

#
# 3.1 Transform the entire MO space into core, active, and external space
# based on natural occupancy
#
# Expectation value of natural orbitals <i|F|i>
fexpt = reduce(numpy.dot,(coeff.T,ova,fav,ova,coeff))
enorb = numpy.diag(fexpt)
# Sort by occupancy
index = numpy.argsort(-eig)
enorb = enorb[index]
nocc  = eig[index]
coeff = coeff[:,index]
#
# Reordering and define active space according to thresh
#
thresh = 0.01
active = (thresh <= nocc) & (nocc <= 2-thresh)
print('\nNatural orbitals:')
print('Offdiag(F) = %s' % numpy.linalg.norm(fexpt - numpy.diag(enorb)))
for i in range(nb):
    print('orb:',i,active[i],nocc[i],enorb[i])
actIndices = numpy.where(active)[0]
print('active orbital indices %s' % actIndices)
print('Num active orbitals %d' % len(actIndices))
cOrbs = coeff[:,:actIndices[0]]
aOrbs = coeff[:,actIndices]
vOrbs = coeff[:,actIndices[-1]+1:]
nb = cOrbs.shape[0]
nc = cOrbs.shape[1]
na = aOrbs.shape[1]
nv = vOrbs.shape[1]
print('core orbs:',cOrbs.shape)
print('act  orbs:',aOrbs.shape)
print('vir  orbs:',vOrbs.shape)
assert nc+na+nv == nb

#
# 3.2 Localizing core, active, external space separately, based on certain
# local orbitals.
#
# We now dump out UHF natual orbitals and localized orbitals to help identify
# active space.
#
# dump UHF natrual orbital
#
tools.molden.from_mo(mol, fname+'_uno.molden', coeff)

#=============================
# localized orbitals
#=============================
iflocal  = False
if iflocal:
    # We implemented different localization later
    from pyscf.tools import localizer
    loc = localizer.localizer(mol,ma[:,:mol.nelectron/2],'boys')
    loc.verbose = 10
    new_coeff = loc.optimize()
    loc = localizer.localizer(mol,ma[:,mol.nelectron/2:],'boys')
    new_coeff2 = loc.optimize()
    lmo = numpy.hstack([new_coeff,new_coeff2])
    tools.molden.from_mo(mol, fname+'lmo.molden', lmo)

#
# Test orthogonality because occasionally localization procedure may break the
# orbital orthogonality (when AO functions are close to linear dependent).
#
cOrbsOAO = numpy.dot(s12,cOrbs)
aOrbsOAO = numpy.dot(s12,aOrbs)
vOrbsOAO = numpy.dot(s12,vOrbs)
print('Ortho-cOAO',numpy.linalg.norm(numpy.dot(cOrbsOAO.T,cOrbsOAO)-numpy.identity(nc)))
print('Ortho-aOAO',numpy.linalg.norm(numpy.dot(aOrbsOAO.T,aOrbsOAO)-numpy.identity(na)))
print('Ortho-vOAO',numpy.linalg.norm(numpy.dot(vOrbsOAO.T,vOrbsOAO)-numpy.identity(nv)))

#==========================================
# Now try to get localized molecular orbitals (SCDM)
#==========================================
def scdm(coeff, overlap, aux):
#
# Argument coeff is a set of orthogonal orbitals |O> (eg occupied HF
# orbitals); aux is a set of localized orbitals.  One can define a subset |B>
# of aux, which has the closest overlap to the coeff space.
# The (orthogonalized) resultant local orbitals |B> can be considered as the
# localized coeff |O>
#
#       |B> = |O><O|aux>, in which det(<O|aux>) is maximized;
#       return lowdin(|B>)
#
    no = coeff.shape[1]
    ova = reduce(numpy.dot,(coeff.T, overlap, aux))
    # ova = no*nb
    q,r,piv = scipy.linalg.qr(ova, pivoting=True)
    # piv[:no] defines the subset of aux which has the largest overlap to coeff space
    bc = ova[:,piv[:no]]

    ova = numpy.dot(bc.T,bc)
    s12inv = lowdin(ova)
    cnew = reduce(numpy.dot,(coeff,bc,s12inv))
    return cnew
#
# Various choices for the localized orbitals
# * the non-orthogonal AOs
#       aux=numpy.identity(nb)
# * Lowdin orthogonalized AOs
aux = s12inv
# * Meta-lowdin orthogonalized AOs
#       from pyscf import lo
#       aux = lo.orth.orth_ao(mol,method='meta_lowdin',pre_orth_ao=lo.orth.pre_orth_ao(mol))
# * ...
#
ova = mol.intor_symmetric("cint1e_ovlp_sph")
clmo = scdm(cOrbs, ova, aux)  # local "AOs" in core space
almo = scdm(aOrbs, ova, aux)  # local "AOs" in active space
vlmo = scdm(vOrbs, ova, aux)  # local "AOs" in external space


#
# 3.3 Sorting each space (core, active, external) based on "orbital energy" to
# prevent high-lying orbitals standing in valence space.
#
# Get <i|F|i>
def psort(ova, fav, coeff):
    # pT is density matrix, fav is Fock matrix
    # OCC-SORT
    pTnew = 2.0*reduce(numpy.dot,(coeff.T,s12,pT,s12,coeff))
    nocc  = numpy.diag(pTnew)
    index = numpy.argsort(-nocc)
    ncoeff = coeff[:,index]
    nocc   = nocc[index]
    enorb = numpy.diag(reduce(numpy.dot,(ncoeff.T,ova,fav,ova,ncoeff)))
    return ncoeff, nocc, enorb

# E-SORT
mo_c, n_c, e_c = psort(ova, fav, clmo)
mo_o, n_o, e_o = psort(ova, fav, almo)
mo_v, n_v, e_v = psort(ova, fav, vlmo)
#
# coeff is the local molecular orbitals
#
coeff = numpy.hstack((mo_c, mo_o, mo_v))

#
# Test orthogonality for the localize MOs as before
#
diff = reduce(numpy.dot,(coeff.T,ova,coeff)) - numpy.identity(nb)
print('diff=',numpy.linalg.norm(diff))
tools.molden.from_mo(mol, fname+'_scdm.molden', coeff)

#
# Population analysis to confirm that our LMO (coeff) make sense
#
#==========================================
# lowdin-pop of the obtained LMOs in OAOs
#==========================================
lcoeff = s12.dot(coeff)
# Orthogonality test
diff = reduce(numpy.dot,(lcoeff.T,lcoeff)) - numpy.identity(nb)
print('diff=',numpy.linalg.norm(diff))

print('\nLowdin population for LMOs:')

pthresh = 0.02
labels = mol.ao_labels(None)
ifACTONLY = False #True
nelec = 0.0
nact = 0.0
for iorb in range(nb):
    vec = lcoeff[:,iorb]**2
    idx = numpy.argwhere(vec>pthresh)[0]
    if ifACTONLY == False:
        if iorb < nc:
            print(' iorb_C=',iorb,' occ=',n_c[iorb],' fii=',e_c[iorb])
            nelec += n_c[iorb]
        elif iorb >= nc and iorb < nc+na:
            print(' iorb_A=',iorb,' occ=',n_o[iorb-nc],' faa=',e_o[iorb-nc])
            nelec += n_o[iorb-nc]
        else:
            print(' iorb_V=',iorb,' occ=',n_v[iorb-nc-na],' fvv=',e_v[iorb-nc-na])
            nelec += n_v[iorb-nc-na]
        for iao in idx:
            print('    iao=',labels[iao],' pop=',vec[iao])
    else:
        if iorb >= nc and iorb < nc+na:
            print(' iorb_A=',iorb,' faa=',e_o[iorb-nc])
            for iao in idx:
                print('    iao=',labels[iao],' pop=',vec[iao])
print('nelec=',nelec)


#
# 3.4 select 'active' orbitals
#
# By reading the orbital images with Jmol, we characterized some of the local
# orbitals
#
a1 = [80,82,83,84,85,86] # S-3p = 7
o1 = [ 2]*6  # approximate occupancies, to help obtain the electrons in active space
a2 = [87,88,89,90,91,92,93,94,95,96] # Fe-Mo (3d,4d) = 10
o2 = [ 1]*8+[0]*2  # approximate occupancies
a3 = [97,98,99,101,103,105] # Mo-s + Fe4d = 6
o3 = [0]*6  # approximate occupancies

#
# There are many different choices for active space, here we just demonstrate
# one which is consists of Fe 3d, Mo 4d and S 3p orbitals
#
#==========================
# select 'active' orbitals
#==========================
caslst = a1+a2
norb = len(caslst)
ne_act = sum(o1) + sum(o2)
s = 1 # 0,1,2,3,4, High-spin case ms = s
ne_alpha = ne_act/2 + s
ne_beta  = ne_act/2 - s
nalpha = ne_alpha
nbeta = ne_beta
norb = len(caslst)
print('norb/nacte=',norb,[nalpha,nbeta])


##################################################
#
# 4. DMRG-CASSCF and DMRG-NEVPT2
#
##################################################
#
# Adjust the MPI schedular and scratch directory if needed.
# NOTE the DMRG-NEVPT2 is expensive, it requires about 8 GB memory per processor
#
#from pyscf.dmrgscf import settings
#settings.MPIPREFIX = 'srun'
#settings.BLOCKSCRATCHDIR = '/scratch'

from pyscf.dmrgscf import DMRGCI, DMRGSCF
from pyscf import mrpt

#
# Redirect output to another file
#
mol.build(verbose=7, output = 'hs_dmrg.out')

mf = scf.RHF(mol).x2c()
mc = DMRGSCF(mf, norb, [nalpha,nbeta])
mc.chkfile = 'hs_mc.chk'
mc.max_memory = 30000
mc.fcisolver.maxM = 1000
mc.fcisolver.tol = 1e-6
orbs = mc.sort_mo(caslst, coeff, base=0)

# Setting natorb to generate natural orbitals in active space.  It's only
# needed for the DMRG-CASSCF calculations with subsequent DMRG-NEVPT2
# calculations.
mc.natorb = True

mc.mc1step(orbs)


#
# CASCI-NEVPT2
#
# If DMRG-CASSCF was finished without any problems (eg convergence, wall time
# limits on cluster etc),  one can simply continue with DMRG-NEVPT2
#       mrpt.NEVPT(mc).kernel()
#
# But it's highly possible that one needs to restore the calculation from
# previous work.  The following is an example to restore the calculation.
# Assuming DMRG-CASSCF has converged and the DMRG temporary files were
# deleted, we just need the DMRG-CASCI calculation with the converged MCSCF
# orbitals to get the DMRG wavefunction.
#
mc = mcscf.CASCI(mf, norb, [nalpha,nbeta])
mc.chkfile = 'hs_mc.chk'
mo = scf.chkfile.load('hs_mc.chk', "mcscf/mo_coeff")
mc.fcisolver = DMRGCI(mol,maxM=500, tol =1e-8)
#
# Tune DMRG parameters.  It's not necessary in most scenario.
#
#mc.fcisolver.outputlevel = 3
#mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16, 20, 24, 28, 30, 34]
#mc.fcisolver.scheduleMaxMs  = [200, 400, 800, 1200, 2000, 4000, 3000, 2000, 1000, 500]
#mc.fcisolver.scheduleTols   = [0.0001, 0.0001, 0.0001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7 ]
#mc.fcisolver.scheduleNoises = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0]
#mc.fcisolver.twodot_to_onedot = 38
#mc.fcisolver.maxIter = 50
mc.casci(mo)

#
# DMRG-NEVPT2
#
mrpt.NEVPT(mc).kernel()

#
# There is also a fast DMRG-NEVPT2 implementation.  See also the example
# pyscf/examples/dmrg/02-dmrg_nevpt2.py
#
mrpt.NEVPT(mc).compress_approx().kernel()

##################################################
#
# Don't forget to clean up the scratch.  DMRG calculation can produce large
# amount of temporary files.
#
##################################################
