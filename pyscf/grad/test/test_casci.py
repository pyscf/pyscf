#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


from functools import reduce
import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import ao2mo
from pyscf.scf import cphf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import casci as casci_grad
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad.mp2 import _shell_prange, has_frozen_orbitals


def kernel(mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None,
           verbose=None):
    if mo_coeff is None: mo_coeff = mc._scf.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    mo_energy = mc._scf.mo_energy

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)

# gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(numpy.dot, (mo_cas, casdm1, mo_cas.T))
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_occ, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nocc,ncas)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    gfock = reduce(numpy.dot, (mo_occ.T, h1 + vhf_c + vhf_a, mo_occ)) * 2
    gfock[:,ncore:nocc] = reduce(numpy.dot, (mo_occ.T, h1 + vhf_c, mo_cas, casdm1))
    gfock[:,ncore:nocc] += numpy.einsum('uviw,vuwt->it', aapa, casdm2)
    dme0 = reduce(numpy.dot, (mo_occ, (gfock+gfock.T)*.5, mo_occ.T))
    aapa = vj = vk = vhf_c = vhf_a = h1 = gfock = None

    dm1 = dm_core + dm_cas
    vhf1c, vhf1a = mf_grad.get_veff(mol, (dm_core, dm_cas))

    diag_idx = numpy.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    #casdm2 = casdm2_cc = None

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm1)
        #de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = None
        de[k] += numpy.einsum('xij,ij->x', vhf1c[:,p0:p1], dm1[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2
        dm2[i,i,ncore:nocc,ncore:nocc] = casdm1 * 2
        dm2[ncore:nocc,ncore:nocc,i,i] = casdm1 * 2
        dm2[i,ncore:nocc,ncore:nocc,i] =-casdm1
        dm2[ncore:nocc,i,i,ncore:nocc] =-casdm1
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    eri0 = ao2mo.restore(1, ao2mo.full(mc._scf._eri, mo_coeff), nmo)
    Imat = numpy.einsum('pjkl,qjkl->pq', eri0, dm2)

    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    neleca, nelecb = mol.nelec

    h1 =-(mol.intor('int1e_ipkin', comp=3)
         +mol.intor('int1e_ipnuc', comp=3))
    s1 =-mol.intor('int1e_ipovlp', comp=3)
    eri1 = mol.intor('int2e_ip1', comp=3).reshape(3,nao,nao,nao,nao)
    eri1 = numpy.einsum('xipkl,pj->xijkl', eri1, mo_coeff)
    eri1 = numpy.einsum('xijpl,pk->xijkl', eri1, mo_coeff)
    eri1 = numpy.einsum('xijkp,pl->xijkl', eri1, mo_coeff)
    h0 = reduce(numpy.dot, (mo_coeff.T, mc._scf.get_hcore(), mo_coeff))
    g0 = ao2mo.restore(1, ao2mo.full(mol, mo_coeff), nmo)

    def hess():
        nocc = mol.nelectron//2
        nvir = nmo - nocc
        eri_mo = g0
        eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
        h = eri_mo[nocc:,:nocc,nocc:,:nocc] * 4
        h-= numpy.einsum('cdlk->ckdl', eri_mo[nocc:,nocc:,:nocc,:nocc])
        h-= numpy.einsum('cldk->ckdl', eri_mo[nocc:,:nocc,nocc:,:nocc])
        for a in range(nvir):
            for i in range(nocc):
                h[a,i,a,i] += eai[a,i]
        return -h.reshape(nocc*nvir,-1)
    hh = hess()
    ee = mo_energy[:,None] - mo_energy

    for k,(sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        mol.set_rinv_origin(mol.atom_coord(k))
        vrinv = -mol.atom_charge(k) * mol.intor('int1e_iprinv', comp=3)

# 2e AO integrals dot 2pdm
        for i in range(3):
            g1 = numpy.einsum('pjkl,pi->ijkl', eri1[i,p0:p1], mo_coeff[p0:p1])
            g1 = g1 + g1.transpose(1,0,2,3)
            g1 = g1 + g1.transpose(2,3,0,1)
            g1 *= -1
            hx =(numpy.einsum('pq,pi,qj->ij', h1[i,p0:p1], mo_coeff[p0:p1], mo_coeff)
               + reduce(numpy.dot, (mo_coeff.T, vrinv[i], mo_coeff)))
            hx = hx + hx.T
            sx = numpy.einsum('pq,pi,qj->ij', s1[i,p0:p1], mo_coeff[p0:p1], mo_coeff)
            sx = sx + sx.T

            fij =(hx[:neleca,:neleca]
                  - numpy.einsum('ij,j->ij', sx[:neleca,:neleca], mo_energy[:neleca])
                  - numpy.einsum('kl,ijlk->ij', sx[:neleca,:neleca],
                                 g0[:neleca,:neleca,:neleca,:neleca]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:neleca,:neleca],
                                 g0[:neleca,:neleca,:neleca,:neleca])
                  + numpy.einsum('ijkk->ij', g1[:neleca,:neleca,:neleca,:neleca]) * 2
                  - numpy.einsum('ikkj->ij', g1[:neleca,:neleca,:neleca,:neleca]))

            fab =(hx[neleca:,neleca:]
                  - numpy.einsum('ij,j->ij', sx[neleca:,neleca:], mo_energy[neleca:])
                  - numpy.einsum('kl,ijlk->ij', sx[:neleca,:neleca],
                                 g0[neleca:,neleca:,:neleca,:neleca]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:neleca,:neleca],
                                 g0[neleca:,:neleca,:neleca,neleca:])
                  + numpy.einsum('ijkk->ij', g1[neleca:,neleca:,:neleca,:neleca]) * 2
                  - numpy.einsum('ikkj->ij', g1[neleca:,:neleca,:neleca,neleca:]))

            fai =(hx[neleca:,:neleca]
                  - numpy.einsum('ai,i->ai', sx[neleca:,:neleca], mo_energy[:neleca])
                  - numpy.einsum('kl,ijlk->ij', sx[:neleca,:neleca],
                                 g0[neleca:,:neleca,:neleca,:neleca]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:neleca,:neleca],
                                 g0[neleca:,:neleca,:neleca,:neleca])
                  + numpy.einsum('ijkk->ij', g1[neleca:,:neleca,:neleca,:neleca]) * 2
                  - numpy.einsum('ikkj->ij', g1[neleca:,:neleca,:neleca,:neleca]))
            c1 = numpy.zeros((nmo,nmo))
            c1[:neleca,:neleca] = -.5 * sx[:neleca,:neleca]
            c1[neleca:,neleca:] = -.5 * sx[neleca:,neleca:]
            cvo1 = numpy.linalg.solve(hh, fai.ravel()).reshape(-1,neleca)
            cov1 = -(sx[neleca:,:neleca] + cvo1).T
            c1[neleca:,:neleca] = cvo1
            c1[:neleca,neleca:] = cov1
            v1 = numpy.einsum('pqai,ai->pq', g0[:,:,neleca:,:neleca], cvo1) * 4
            v1-= numpy.einsum('paiq,ai->pq', g0[:,neleca:,:neleca,:], cvo1)
            v1-= numpy.einsum('piaq,ai->pq', g0[:,:neleca,neleca:,:], cvo1)
            fij += v1[:neleca,:neleca]
            fab += v1[neleca:,neleca:]
            c1[:ncore,ncore:neleca] = -fij[:ncore,ncore:] / ee[:ncore,ncore:neleca]
            c1[ncore:neleca,:ncore] = -fij[ncore:,:ncore] / ee[ncore:neleca,:ncore]
            m = nocc - neleca
            c1[nocc:,neleca:nocc] = -fab[m:,:m] / ee[nocc:,neleca:nocc]
            c1[neleca:nocc,nocc:] = -fab[:m,m:] / ee[neleca:nocc,nocc:]
            h0c1 = h0.dot(c1)
            h0c1 = h0c1 + h0c1.T
            g0c1 = numpy.einsum('pjkl,pi->ijkl', g0, c1)
            g0c1 = g0c1 + g0c1.transpose(1,0,2,3)
            g0c1 = g0c1 + g0c1.transpose(2,3,0,1)

            de[k,i] += numpy.einsum('ij,ji', h0c1, dm1)
            de[k,i] += numpy.einsum('ijkl,jilk', g0c1, dm2)*.5

    de += rhf_grad.grad_nuc(mol)
    return de

def _response_dm1(mycc, Xvo, eris=None):
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    with_frozen = has_frozen_orbitals(mycc)
    if eris is None or with_frozen:
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc.mo_occ
        mo_coeff = mycc.mo_coeff
        def fvind(x):
            x = x.reshape(Xvo.shape)
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, v, mo_coeff[:,:nocc]))
            return v * 2
    else:
        mo_energy = eris.fock.diagonal()
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        ovvo = numpy.empty((nocc,nvir,nvir,nocc))
        for i in range(nocc):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 4 - ovvo[i].transpose(1,0,2)
            ovvo[i]-= eris.oovv[i].transpose(2,1,0)
        def fvind(x):
            return numpy.einsum('iabj,bj->ai', ovvo, x.reshape(Xvo.shape))
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1

def casci_grad_with_ccsd_solver(mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None,
                                verbose=None):
    if mo_coeff is None: mo_coeff = mc._scf.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelecas)
    no = mc.nelecas[0]
    for i in range(no):
        casdm1[i,i] -= 2
    for i in range(no):
        for j in range(no):
            casdm2[i,i,j,j] -= 4
            casdm2[i,j,j,i] += 2
    for i in range(no):
        casdm2[i,i,:,:] -= casdm1 * 2
        casdm2[:,:,i,i] -= casdm1 * 2
        casdm2[:,i,i,:] += casdm1
        casdm2[i,:,:,i] += casdm1

    mc.mo_occ = mc._scf.mo_occ
    mask = numpy.zeros(nmo, dtype=bool)
    mask[ncore:nocc] = True
    mc.frozen = numpy.where(~mask)[0]
    mc.get_frozen_mask = lambda *args: mask
    d1 = (casdm1[:no,:no] * .5, casdm1[:no,no:] * .5,
          casdm1[no:,:no] * .5, casdm1[no:,no:] * .5)
    casdm2 = (casdm2 + casdm2.transpose(1,0,2,3)) * .5
    vvvv = casdm2[no:,no:,no:,no:]
    d2 = (casdm2[:no,no:,:no,no:] * .5,
          ao2mo.restore(4, vvvv, ncas-no) * .25,
          casdm2[:no,:no,:no,:no] * .25,
          casdm2[:no,:no,no:,no:] * .5,
          casdm2[:no,no:,no:,:no] * .5, casdm2,
          casdm2[:no,no:,no:,no:],
          casdm2[:no,:no,:no,no:])
    mc.mo_coeff = mo_coeff
    t1 = t2 = l1 = l2 = ci
    de = ccsd_grad.grad_elec(mc.Gradients(), t1, t2, l1, l2, None, atmlst,
                             d1, d2, verbose)
    de += rhf_grad.grad_nuc(mol)
    return de

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = 'N 0 0 0; N 0 0 1.2; H 1 1 0; H 1 1 1.2'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_casci_grad(self):
        mc = mcscf.CASCI(mf, 4, 4).run()
        g1 = casci_grad.Gradients(mc).kernel()
        self.assertAlmostEqual(lib.fp(g1), -0.066025991364829367, 7)

        g1ref = kernel(mc)
        self.assertAlmostEqual(abs(g1-g1ref).max(), 0, 6)

        mcs = mc.as_scanner()
        pmol = mol.copy()
        e1 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e2 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        self.assertAlmostEqual(g1[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_casci_grad_excited_state(self):
        mc = mcscf.CASCI(mf, 4, 4)
        mc.fcisolver.nroots = 3
        g_scan = mc.nuc_grad_method().as_scanner(state=2)
        g1 = g_scan(mol, atmlst=range(4))[1]
        self.assertAlmostEqual(lib.fp(g1), -0.058112001722577293, 6)

        g2 = g_scan.kernel(state=0)
        self.assertAlmostEqual(lib.fp(g2), -0.066025991364829367, 6)

        mcs = mc.as_scanner()
        pmol = mol.copy()
        e1 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e2 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        self.assertAlmostEqual(g1[1,2], (e1[2]-e2[2])/0.002*lib.param.BOHR, 5)

    def test_casci_grad_with_ccsd_solver(self):
        mol = gto.Mole()
        mol.atom = 'N 0 0 0; N 0 0 1.2; H 1 1 0; H 1 1 1.2'
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        mc = mcscf.CASCI(mf, 4, 4).run()
        g1 = mc.nuc_grad_method().kernel(atmlst=range(mol.natm))
        self.assertAlmostEqual(lib.fp(g1), -0.066025991364829367, 8)

        g1ref = casci_grad_with_ccsd_solver(mc)
        self.assertAlmostEqual(abs(g1-g1ref).max(), 0, 7)

    def test_scanner(self):
        mc = mcscf.CASCI(mf, 4, 4)
        gs = mc.nuc_grad_method().as_scanner().as_scanner()
        e, g1 = gs(mol.atom)
        self.assertAlmostEqual(e, -108.38187009571901, 8)
        self.assertAlmostEqual(lib.fp(g1), -0.066025991364829367, 6)

    def test_state_specific_scanner(self):
        mc = mcscf.CASCI(mf, 4, 4)
        gs = mc.state_specific_(2).nuc_grad_method().as_scanner()
        e, de = gs(mol)
        self.assertAlmostEqual(e, -108.27330628098245, 8)
        self.assertAlmostEqual(lib.fp(de), -0.058111987691940134, 6)

        mcs = gs.base
        pmol = mol.copy()
        e1 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e2 = mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        self.assertAlmostEqual(de[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_state_average_scanner(self):
        mc = mcscf.CASCI(mf, 4, 4)
        gs = mc.state_average_([0.5, 0.5]).nuc_grad_method().as_scanner()
        e, de = gs(mol)
        self.assertAlmostEqual(e, -108.37395097152324, 8)
        self.assertAlmostEqual(lib.fp(de), -0.1170409338178659, 6)
        self.assertRaises(RuntimeError, mc.nuc_grad_method().as_scanner, state=2)

        mcs = gs.base
        pmol = mol.copy()
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e1 = mcs.e_average
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        e2 = mcs.e_average
        self.assertAlmostEqual(de[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_state_average_mix_scanner(self):
        mc = mcscf.CASCI(mf, 4, 4)
        mc = mcscf.addons.state_average_mix_(mc, [mc.fcisolver, mc.fcisolver], (.5, .5))
        gs = mc.nuc_grad_method().as_scanner()
        e, de = gs(mol)
        self.assertAlmostEqual(e, -108.38187009582806, 8)
        self.assertAlmostEqual(lib.fp(de), -0.0660259910725428, 6)

        mcs = gs.base
        pmol = mol.copy()
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2'))
        e1 = mcs.e_average
        mcs(pmol.set_geom_('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2'))
        e2 = mcs.e_average
        self.assertAlmostEqual(de[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_with_x2c_scanner(self):
        with lib.light_speed(20.):
            mcs = mcscf.CASCI(mf, 4, 4).as_scanner().x2c()
            gscan = mcs.nuc_grad_method().as_scanner()
            g1 = gscan(mol)[1]
            self.assertAlmostEqual(lib.fp(g1), -0.0707065428512548, 6)

            e1 = mcs('N 0 0 0; N 0 0 1.201; H 1 1 0; H 1 1 1.2')
            e2 = mcs('N 0 0 0; N 0 0 1.199; H 1 1 0; H 1 1 1.2')
            self.assertAlmostEqual(g1[1,2], (e1-e2)/0.002*lib.param.BOHR, 5)

    def test_with_qmmm_scanner(self):
        from pyscf import qmmm
        mol = gto.Mole()
        mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                       H                 -0.00000000   -0.84695236    0.59109389
                       H                 -0.00000000    0.89830571    0.52404783 '''
        mol.verbose = 0
        mol.basis = '6-31g'
        mol.build()

        coords = [(0.5,0.6,0.1)]
        #coords = [(0.0,0.0,0.0)]
        charges = [-0.1]
        mf = qmmm.add_mm_charges(scf.RHF(mol), coords, charges)
        mc = mcscf.CASCI(mf, 4, 4).as_scanner()
        e_tot, g = mc.nuc_grad_method().as_scanner()(mol)
        self.assertAlmostEqual(e_tot, -75.98156095286714, 8)
        self.assertAlmostEqual(lib.fp(g), 0.08335504754051845, 6)
        e1 = mc(''' O                  0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        e2 = mc(''' O                 -0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        ref = (e1 - e2)/0.002 * lib.param.BOHR
        self.assertAlmostEqual(g[0,0], ref, 4)

        mf = scf.RHF(mol)
        mc = qmmm.add_mm_charges(mcscf.CASCI(mf, 4, 4).as_scanner(), coords, charges)
        e_tot, g = mc.nuc_grad_method().as_scanner()(mol)
        self.assertAlmostEqual(e_tot, -75.98156095286714, 7)
        self.assertAlmostEqual(lib.fp(g), 0.08335504754051845, 5)

    def test_symmetrize(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        g = mol.RHF.run().CASCI(4, 4).run().Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), 0.11555543375018221, 6)

    # issue 1909
    def test_small_mem(self):
        mol = gto.M(atom="""
            H                 -0.00021900   -0.20486000   -2.17721200
            H                 -0.00035900   -1.27718700   -2.17669400
            """, basis='6-31G')
        casci = mol.CASCI(2, 2).run()
        grad = casci.nuc_grad_method()
        grad.max_memory = 0
        nuc_grad = grad.kernel()
        self.assertAlmostEqual(lib.fp(nuc_grad), 0.09424065197659935, 7)


if __name__ == "__main__":
    print("Tests for CASCI gradients")
    unittest.main()
