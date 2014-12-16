import numpy
import scipy.linalg
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.ao2mo
import pyscf.lib.logger as logger

mol = pyscf.gto.Mole()
mol.verbose = 0
#mol.atom = [
#    ['O', ( 0., 0.    , 0.   )],
#    ['H', ( 0., -0.757, 0.587)],
#    ['H', ( 0., 0.757 , 0.587)],]
#mol.basis = {'H': 'cc-pvdz',
#             'O': 'cc-pvdz',}
mol.atom = [
    ['N', ( 0., 0.    , 0.   )],
    ['N', ( 0., 0.    , 1.5)],]
mol.basis = {'N': '6-311G'}
mol.build()

mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print ehf
mc = pyscf.mcscf.CASSCF(mol, mf, 4, 4)
mc.verbose = 0
#mo = pyscf.mcscf.addons.sort_mo(mc, mf.mo_coeff, (3,4,6,7,8,9), 1)
#emc = mc.mc1step(mo)[0] + mol.nuclear_repulsion()
emc = mc.mc1step()[0] + mol.nuclear_repulsion()
#print emc, -76.0926176464
print emc


############################################
#
#  natural orbitals
#
def make_natorb(casscf):
    fcivec = casscf.ci
    mo = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nocc = ncore + ncas

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
# alternatively, return alpha, beta 1-pdm seperately
#casdm1a,casdm1b = casscf.fcisolver.make_rdm1s(fcivec, ncas, nelecas)

    occ, ucas = scipy.linalg.eigh(casdm1)
    logger.debug(casscf, 'Natural occs')
    logger.debug(casscf, str(occ))
    natocc = numpy.zeros(mo.shape[1])
    natocc[:ncore] = 1
    natocc[ncore:nocc] = occ[::-1] * .5

# transform the CAS natural orbitals from MO repr. to AO repr.
# mo[AO_idx,MO_idx]
    natorb_in_cas = numpy.dot(mo[:,ncore:nocc], ucas[:,::-1])
    natorb_on_ao = numpy.hstack((mo[:,:ncore], natorb_in_cas, mo[:,nocc:]))
    return natorb_on_ao, natocc

natorb, natocc = make_natorb(mc)
#
############################################


############################################
#
#  integrals
#
ncore = mc.ncore
ncas = mc.ncas
nocc = ncore + ncas
mo = mc.mo_coeff
nvir = mo.shape[1] - nocc

h1e_ao = mc.get_hcore()
h1e_mo = reduce(numpy.dot, (natorb.T, h1e_ao, natorb))

v2e = pyscf.ao2mo.incore.full(mf._eri, natorb) # v2e has 4-fold symmetry now
# To do integral transformation outcore, and save the transformed integrals in
# file "erifile", using h5py to read the integrals
#import tmepfile
#import h5py
#erifile = tempfile.NamedTemporaryFile()
#ao2mo.outcore.full(mol, natorb, erifile.name)
#file_v2e = h5py.File(erifile.name)
#v2e = file_v2e['eri_mo']

nmo = natorb.shape[1]
v2e = pyscf.ao2mo.restore(1, v2e, nmo) # to remove 4-fold symmetry, turn v2e to n**4 array
v2e = v2e.transpose(0,2,1,3) # to physics notation

beta = numpy.sqrt(natocc)
alpha = numpy.sqrt(1-natocc)

tmpa = h1e_mo + numpy.einsum('prqr,r->pq', v2e, 2 * beta**2) - numpy.einsum('prrq,r->pq', v2e, beta**2)
h1e_bogo = numpy.einsum('p,pq,q->pq', alpha, tmpa, alpha)

tmpb = h1e_mo + numpy.einsum('prqr,r->pq', v2e, 2 * beta**2) - numpy.einsum('prrq,r->pq', v2e, beta**2)
h1e_bogo -= numpy.einsum('p,pq,q->pq', beta, tmpb, beta)

tmp3 = numpy.einsum('pqrr,p,q,r,r->pq', v2e, alpha, beta, alpha, beta) \
     + numpy.einsum('qprr,q,p,r,r->pq', v2e, alpha, beta, alpha, beta)
h1e_bogo -= tmp3

wpqrs = numpy.einsum('pqsr,p,q,r,s->pqrs', v2e, alpha, alpha, beta, beta)

# transform to semi-canonical basis
e1, c1 = scipy.linalg.eigh(h1e_bogo[:ncore,:ncore])
e2, c2 = scipy.linalg.eigh(h1e_bogo[ncore:nocc,ncore:nocc])
e3, c3 = scipy.linalg.eigh(h1e_bogo[nocc:,nocc:])

e_bogo = numpy.hstack((e1,e2,e3))
print e_bogo

c = numpy.zeros((nmo,nmo))
c[:ncore,:ncore] = c1
c[ncore:nocc,ncore:nocc] = c2
c[nocc:,nocc:] = c3

wpqrs = numpy.einsum('pqrs,px->xqrs', wpqrs, c)
wpqrs = numpy.einsum('pqrs,qx->pxrs', wpqrs, c)
wpqrs = numpy.einsum('pqrs,rx->pqxs', wpqrs, c)
wpqrs = numpy.einsum('pqrs,sx->pqrx', wpqrs, c)

# Core-to-external
dpq = e_bogo[nocc:][:,None] + e_bogo[nocc:]
drs = e_bogo[:ncore][:,None] + e_bogo[:ncore]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1/dpqrs.reshape(nvir,nvir,ncore,ncore)
#ectmp2 = -.25 * numpy.einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_ccee =  -2 * numpy.einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_ccee +=  numpy.einsum('pqrs,pqsr,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore], wpqrs[nocc:,nocc:,:ncore,:ncore], dpqrs)
print "CT-MP2 Doubles Energy (CCEE):", ectmp2_ccee

# Core-to-active
dpq = e_bogo[ncore:nocc][:,None] + e_bogo[ncore:nocc]
drs = e_bogo[:ncore][:,None] + e_bogo[:ncore]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1/dpqrs.reshape(ncas,ncas,ncore,ncore)
#ectmp2 = -.25 * numpy.einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_ccaa =  -2 * numpy.einsum('pqrs,pqrs', wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore]**2, dpqrs)
ectmp2_ccaa +=  numpy.einsum('pqrs,pqsr,pqrs', wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore], wpqrs[ncore:nocc,ncore:nocc,:ncore,:ncore], dpqrs)
print "CT-MP2 Doubles Energy (CCAA):", ectmp2_ccaa

# Active-to-external
dpq = e_bogo[nocc:][:,None] + e_bogo[nocc:]
drs = e_bogo[ncore:nocc][:,None] + e_bogo[ncore:nocc]
dpqrs = dpq.reshape(-1,1) + drs.reshape(-1)
dpqrs = 1/dpqrs.reshape(nvir,nvir,ncas,ncas)
#ectmp2 = -.25 * numpy.einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,:ncore,:ncore]**2, dpqrs)
ectmp2_aaee =  -2 * numpy.einsum('pqrs,pqrs', wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc]**2, dpqrs)
ectmp2_aaee +=  numpy.einsum('pqrs,pqsr,pqrs', wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc], wpqrs[nocc:,nocc:,ncore:nocc,ncore:nocc], dpqrs)
print "CT-MP2 Doubles Energy (AAEE):", ectmp2_aaee

print "\nCT-MP2 Correlation Energy:   ", ectmp2_ccee + ectmp2_ccaa + ectmp2_aaee
print "CASSCF + CT-MP2 Total Energy:", emc + ectmp2_ccee + ectmp2_ccaa + ectmp2_aaee
