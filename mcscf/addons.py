#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf import ao2mo

def sort_mo(casscf, mo, caslst, base=1):
    assert(casscf.ncas == len(caslst))
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas
    nmo = mo.shape[0]
    if base != 0:
        caslst = [i-1 for i in caslst]
    idx = [i for i in range(nmo) if i not in caslst]
    return numpy.hstack((mo[:,idx[:ncore]], mo[:,caslst], mo[:,idx[ncore:]]))

def _make_rdm1_on_mo(casdm1, ncore, ncas, nmo, docc=True):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    if docc:
        dm1[idx,idx] = 2
    else:
        dm1[idx,idx] = 1
    dm1[ncore:nocc,ncore:nocc] = casdm1
    return dm1

# on AO representation
def make_rdm1(casscf, fcivec=None, mo=None):
    if fcivec is None:
        fcivec = casscf.ci
    if mo is None:
        mo = casscf.mo_coeff
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
    rdm1 = _make_rdm1_on_mo(casdm1, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo, rdm1, mo.T))
    return rdm1

# make both alpha and beta density matrices
def make_rdm1s(casscf, fcivec=None, mo=None):
    if fcivec is None:
        fcivec = casscf.ci
    if mo is None:
        mo = casscf.mo_coeff
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    rdm1a, rdm1b = casscf.fcisolver.make_rdm1s(fcivec, ncas, nelecas)
    rdm1a = _make_rdm1_on_mo(rdm1a, ncore, ncas, nmo, False)
    rdm1b = _make_rdm1_on_mo(rdm1b, ncore, ncas, nmo, False)
    rdm1a = reduce(numpy.dot, (mo, rdm1a, mo.T))
    rdm1b = reduce(numpy.dot, (mo, rdm1b, mo.T))
    return rdm1a, rdm1b

def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
        dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
    return dm1, dm2

# on AO representation
def make_rdm12(casscf, fcivec=None, mo=None):
    if fcivec is None:
        fcivec = casscf.ci
    if mo is None:
        mo = casscf.mo_coeff
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo.shape[1]
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, nelecas)
    rdm1, rdm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo, rdm1, mo.T))
    rdm2 = numpy.dot(mo, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo.T)
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo).transpose(2,3,0,1)
    rdm2 = numpy.dot(mo, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo.T)
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

# generalized fock matrix
def make_fock(casscf, fcivec=None, mo=None):
    if fcivec is None:
        fcivec = casscf.ci
    if mo is None:
        mo = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    mocc = mo[:,:nocc]

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
    eris = casscf.update_ao2mo(mo)
    vj = numpy.einsum('ipq->pq', eris['jc_pp']) * 2 \
       + numpy.einsum('ij,ijpq->pq', casdm1, eris['aapp'])
    vk = numpy.einsum('ipqi->pq', eris['kc_pp']) * 2 \
       + numpy.einsum('ij,ipqj->pq', casdm1, eris['appa'])

    h1 = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))
    fock = h1 + vj - vk * .5
    return fock

def restore_cas_natorb(casscf, fcivec=None, mo=None):
    log = lib.logger.Logger(casscf.stdout, casscf.verbose)
    if fcivec is None:
        fcivec = casscf.ci
    if mo is None:
        mo = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)
    occ, ucas = numpy.linalg.eigh(-casdm1)
    occ = -occ
    log.info('Natural occs')
    log.info(str(occ))
    # restore phase
    for i, k in enumerate(numpy.argmax(abs(ucas), axis=0)):
        if ucas[k,i] < 0:
            ucas[:,i] *= -1
    mo = mo.copy()
    mo[:,ncore:nocc] = numpy.dot(mo[:,ncore:nocc], ucas)
    fcivec = casscf.casci(mo, fcivec)[2]
    return fcivec, mo

def map2hf(casscf, base=1, threshold=.5):
    s = casscf.mol.intor_symmetric('cint1e_ovlp_sph')
    s = reduce(numpy.dot, (casscf.mo_coeff.T, s, casscf._scf.mo_coeff))
    idx = numpy.argwhere(abs(s) > threshold)
    for i,j in idx:
        casscf.stdout.write('<mo-mcscf|mo-hf>  %d  %d  %12.8f\n'
                            % (i+base,j+base,s[i,j]))
    casscf.stdout.flush()


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    import mc1step
    from pyscf import tools
    import pyscf.tools.ring

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['H', c] for c in tools.ring.make(6, 1.2)]
    mol.basis = {'H': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mc1step.CASSCF(mol, m, 6, 6)
    mc.verbose = 4
    emc, e_ci, fcivec, mo = mc.mc1step()
    emc += mol.nuclear_repulsion()
    print(ehf, emc, emc-ehf)
    print(emc - -3.272089958)

    rdm1 = make_rdm1(mc, fcivec, mo)
    rdm1, rdm2 = make_rdm12(mc, fcivec, mo)
    print(rdm1)

    mo1 = restore_cas_natorb(mc)[1]
    numpy.set_printoptions(2)
    print(reduce(numpy.dot, (mo1[:,:6].T, mol.intor('cint1e_ovlp_sph'),
                             mo[:,:6])))
