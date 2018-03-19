#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

from functools import reduce
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.dft import numint
from pyscf.scf import hf_symm
from pyscf.data import nist
from pyscf.soscf.newton_ah import _gen_rhf_response


def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute (A+B)x
    
    Kwargs:
        wfnsym : int
            Point group symmetry for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert(mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym

    if fock_ao is None:
        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])
    else:
        fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

    hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = _gen_rhf_response(mf, singlet=singlet, hermi=0)

    def vind(zs):
        nz = len(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs).reshape(-1,nvir,nocc)
            zs[:,sym_forbid] = 0
        dmvo = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            # *2 for double occupancy
            dmvo[i] = reduce(numpy.dot, (orbv, z.reshape(nvir,nocc)*2, orbo.T))
        v1ao = vresp(dmvo)
        #v1vo = numpy.asarray([reduce(numpy.dot, (orbv.T, v, orbo)) for v in v1ao])
        v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir,nocc)
        for i, z in enumerate(zs):
            v1vo[i]+= numpy.einsum('ps,sq->pq', fvv, z.reshape(nvir,nocc))
            v1vo[i]-= numpy.einsum('ps,rp->rs', foo, z.reshape(nvir,nocc))
        if wfnsym is not None and mol.symmetry:
            v1vo[:,sym_forbid] = 0
        return v1vo.reshape(nz,-1)

    return vind, hdiag
gen_tda_hop = gen_tda_operation

def get_ab(mf, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''A and B matrices for TDDFT response function.

    A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
    B[i,a,j,b] = (ia||jb)
    '''
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    assert(mo_coeff.dtype == numpy.double)

    mol = mf.mol
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]
    mo = numpy.hstack((orbo,orbv))
    nmo = nocc + nvir

    eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
    a = numpy.diag(eai.T.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb

        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb

    if hasattr(mf, 'xc') and hasattr(mf, '_numint'):
        from pyscf.dft import rks
        from pyscf.dft import numint
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        add_hf_(a, b, hyb)

        xctype = ni._xc_type(mf.xc)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        make_rho = ni._gen_rho_evaluator(mol, dm0, hermi=1)[0]
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, 'LDA')
                fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[2]
                frr = fxc[0]

                rho_o = lib.einsum('rp,pi->ri', ao, orbo)
                rho_v = lib.einsum('rp,pi->ri', ao, orbv)
                rho_ov = numpy.einsum('ri,ra->ria', rho_o, rho_v)
                w_ov = numpy.einsum('ria,r->ria', rho_ov, weight*frr)
                iajb = lib.einsum('ria,rjb->iajb', rho_ov, w_ov) * 2
                a += iajb
                b += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
                rho = make_rho(0, ao, mask, 'GGA')
                vxc, fxc = ni.eval_xc(mf.xc, rho, 0, deriv=2)[1:3]
                vgamma = vxc[1]
                frho, frhogamma, fgg = fxc[:3]

                rho_o = lib.einsum('xrp,pi->xri', ao, orbo)
                rho_v = lib.einsum('xrp,pi->xri', ao, orbv)
                rho_ov = numpy.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += numpy.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                # sigma1 ~ \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
                sigma1 = numpy.einsum('xr,xria->ria', rho[1:4], rho_ov[1:4])

                w_ov = numpy.empty_like(rho_ov)
                w_ov[0]  = numpy.einsum('r,ria->ria', frho, rho_ov[0])
                w_ov[0] += numpy.einsum('r,ria->ria', 2*frhogamma, sigma1)
                f_ov = numpy.einsum('r,ria->ria', 4*fgg, sigma1)
                f_ov+= numpy.einsum('r,ria->ria', 2*frhogamma, rho_ov[0])
                w_ov[1:] = numpy.einsum('ria,xr->xria', f_ov, rho[1:4])
                w_ov[1:]+= numpy.einsum('r,xria->xria', 2*vgamma, rho_ov[1:4])
                w_ov *= weight[:,None,None]
                iajb = lib.einsum('xria,xrjb->iajb', rho_ov, w_ov) * 2
                a += iajb
                b += iajb

        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        elif xctype == 'MGGA':
            raise NotImplementedError('meta-GGA')

    else:
        add_hf_(a, b)

    return a, b

def get_nto(tdobj, state, threshold=0.3, verbose=None):
    r'''
    Natural transition orbital analysis.

    The natural transition density matrix between ground state and excited
    state :math:`Tia = \langle \Psi_{ex} | i a^\dagger | \Psi_0 \rangle` can
    be transformed to diagonal form through SVD
    :math:`T = O \sqrt{\lambda} V^\dagger`. O and V are occupied and virtual
    natural transition orbitals. The diagonal elements :math:`\lambda` are the
    weights of the occupied-virtual orbital pair in the excitation.

    Ref: Martin, R. L., JCP, 118, 4775-4777

    Note in the TDHF/TDDFT calculations, the excitation part (X) is
    interpreted as the CIS coefficients and normalized to 1. The de-excitation
    part (Y) is ignored.

    Args:
        state : int
            Excited state ID.  state = 1 means the first excited state.

    Kwargs:
        threshold : float
            Above which the NTO coefficients will be printed in the output.

    Returns:
        A list (weights, NTOs).  NTOs are natural orbitals represented in AO
        basis. The first N_occ NTOs are occupied NTOs and the rest are virtual
        NTOs.
    '''
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    cis_t1 = tdobj.xy[state-1][0]
    # TDDFT (X,Y) has X^2-Y^2=1.
    # Renormalizing X (X^2=1) to map it to CIS coefficients
    cis_t1 *= 1. / numpy.linalg.norm(cis_t1)

# TODO: Comparing to the NTOs defined in JCP, 142, 244103.  JCP, 142, 244103
# provides a method to incorporate the Y matrix in the transition density
# matrix.  However, it may break the point group symmetry of the NTO orbitals
# when the system has degenerated irreducible representations.

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        o_sym = orbsym[mo_occ==2]
        v_sym = orbsym[mo_occ==0]
        nto_o = numpy.eye(nocc)
        nto_v = numpy.eye(nvir)
        weights_o = numpy.zeros(nocc)
        weights_v = numpy.zeros(nvir)
        for ir in set(orbsym):
            o_idx = numpy.where(o_sym == ir)[0]
            if o_idx.size > 0:
                dm_oo = numpy.dot(cis_t1[:,o_idx].T, cis_t1[:,o_idx])
                weights_o[o_idx], nto_o[o_idx[:,None],o_idx] = numpy.linalg.eigh(dm_oo)

            v_idx = numpy.where(v_sym == ir)[0]
            if v_idx.size > 0:
                dm_vv = numpy.dot(cis_t1[v_idx], cis_t1[v_idx].T)
                weights_v[v_idx], nto_v[v_idx[:,None],v_idx] = numpy.linalg.eigh(dm_vv)

        # weights in descending order
        idx = numpy.argsort(-weights_o)
        weights_o = weights_o[idx]
        nto_o = nto_o[:,idx]
        o_sym = o_sym[idx]

        idx = numpy.argsort(-weights_v)
        weights_v = weights_v[idx]
        nto_v = nto_v[:,idx]
        v_sym = v_sym[idx]

        nto_orbsym = numpy.hstack((o_sym, v_sym))

        if nocc < nvir:
            weights = weights_o
        else:
            weights = weights_v

    else:
        nto_v, w, nto_oT = numpy.linalg.svd(cis_t1)
        nto_o = nto_oT.conj().T
        weights = w**2
        nto_orbsym = None

    idx = numpy.argmax(abs(nto_o.real), axis=0)
    nto_o[:,nto_o[idx,numpy.arange(nocc)].real<0] *= -1
    idx = numpy.argmax(abs(nto_v.real), axis=0)
    nto_v[:,nto_v[idx,numpy.arange(nvir)].real<0] *= -1

    occupied_nto = numpy.dot(orbo, nto_o)
    virtual_nto = numpy.dot(orbv, nto_v)
    nto_coeff = numpy.hstack((occupied_nto, virtual_nto))

    if mol.symmetry:
        nto_coeff = lib.tag_array(nto_coeff, orbsym=nto_orbsym)

    log = logger.new_logger(tdobj, verbose)
    if log.verbose >= logger.INFO:
        log.info('State %d: %g eV  NTO largest component %s',
                 state, tdobj.e[state-1]*nist.HARTREE2EV, weights[0])
        o_idx = numpy.where(abs(nto_o[:,0]) > threshold)[0]
        v_idx = numpy.where(abs(nto_v[:,0]) > threshold)[0]
        fmt = '%' + str(lib.param.OUTPUT_DIGITS) + 'f (MO #%d)'
        log.info('    Occ-NTO: %s',
                 ' '.join([(fmt % (nto_o[i,0], i+1)) for i in o_idx]))
        log.info('    Vir-NTO: %s',
                 ' '.join([(fmt % (nto_v[i,0], i+1+nocc)) for i in v_idx]))
    return weights, nto_coeff


def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e11/e_wn

    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    mol = tdobj.mol
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords)
    with mol.with_common_orig(charge_center):
        dip_ints = mol.intor_symmetric('int1e_r', comp=3)

    dip_ints = numpy.einsum('xpq,pi,qj->xij', dip_ints, orbo.conj(), orbv)

    if tdobj.singlet:
        log.note('\n**** Singlet excitation energies and oscillator strengths ****')
    else:
        log.note('\n**** Triplet excitation energies and oscillator strengths ****')

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        x_sym = (orbsym[mo_occ==0,None] & orbsym[mo_occ==2]).ravel()
    else:
        x_sym = None

    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        trans_dip = numpy.einsum('xij,ji->x', dip_ints, x)
        # TODO JCP, 143, 234103
        f_oscillator = 2./3. * ei * numpy.dot(trans_dip, trans_dip)
        if x_sym is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator)
        else:
            wfnsym_id = x_sym[abs(x).argmax()]
            wfnsym = symm.irrep_id2name(mol.groupname, wfnsym_id)
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator)

        if log.verbose >= logger.INFO:
            x = x.T
            o_idx, v_idx = numpy.where(abs(x) > 0.1)
            for i, j in zip(o_idx, v_idx):
                log.info('    %4d -> %-4d %.5f', i, j+nocc, x[i,j])

    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        for i, ei in enumerate(tdobj.e):
            x, y = tdobj.xy[i]
            trans_dip = numpy.einsum('xij,ji->x', dip_ints, x)
            f_oscillator = 2./3. * ei * numpy.dot(trans_dip, trans_dip)
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, trans_dip[0], trans_dip[1], trans_dip[2],
                     numpy.dot(trans_dip, trans_dip), f_oscillator)

# UV spectrum
# * transition dipole
# * velocity moment
# * magnetic dipole


class TDA(lib.StreamObject):
    '''Tamm-Dancoff approximation

    Attributes:
        conv_tol : float
            Diagonalization convergence tolerance.  Default is 1e-9.
        nstates : int
            Number of TD states to be computed. Default is 3.

    Saved results:

        converged : bool
            Diagonalization converged or not
        e : 1D array
            excitation energy for each excited state.
        xy : A list of two 2D arrays
            Excitation coefficients (X, with shape (nvir,nocc)) and de-excitation
            coefficients (Y, with shape (nvir,nocc)) for each excited state.
            (X,Y) are normalized to 1/2 in RHF/RKS methods and normalized to 1
            for UHF/UKS methods. In the TDA calculation, Y = 0.
    '''
    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.mol = mf.mol
        self.chkfile = mf.chkfile
        self._scf = mf

        self.conv_tol = 1e-9
        self.nstates = 3
        self.singlet = True
        self.wfnsym = None
        self.lindep = 1e-12
        self.level_shift = 0
        self.max_space = 50
        self.max_cycle = 100
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        # xy = (X,Y), normalized to 1/2: 2(XX-YY) = 1
        # In TDA, Y = 0
        self.converged = None
        self.e = None
        self.xy = None
        self._keys = set(self.__dict__.keys())

    @property
    def nroots(self):
        return self.nstates
    @nroots.setter
    def nroots(self, x):
        self.nstates = x

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self._scf.__class__)
        if self.singlet:
            log.info('nstates = %d singlet', self.nstates)
        else:
            log.info('nstates = %d triplet', self.nstates)
        log.info('wfnsym = %s', self.wfnsym)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('eigh lindep = %g', self.lindep)
        log.info('eigh level_shift = %g', self.level_shift)
        log.info('eigh max_space = %d', self.max_space)
        log.info('eigh max_cycle = %d', self.max_cycle)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        log.info('\n')

    def gen_vind(self, mf):
        '''Compute Ax'''
        return gen_tda_hop(mf, singlet=self.singlet, wfnsym=self.wfnsym)

    @lib.with_doc(get_ab.__doc__)
    def get_ab(self, mf=None):
        if mf is None: mf = self._scf
        return get_ab(mf)

    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond

    def init_guess(self, mf, nstates=None, wfnsym=None):
        if nstates is None: nstates = self.nstates
        if wfnsym is None: wfnsym = self.wfnsym

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = numpy.where(mo_occ==2)[0]
        viridx = numpy.where(mo_occ==0)[0]
        eai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])

        if wfnsym is not None and mf.mol.symmetry:
            orbsym = hf_symm.get_orbsym(mf.mol, mf.mo_coeff)
            sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym
            eai[sym_forbid] = 1e99

        nov = eai.size
        nroot = min(nstates, nov)
        x0 = numpy.zeros((nroot, nov))
        idx = numpy.argsort(eai.ravel())
        for i in range(nroot):
            x0[i,idx[i]] = 1  # lowest excitations
        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)
        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_space=self.max_space,
                              verbose=self.verbose)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nvir,nocc)*numpy.sqrt(.5),0) for xi in x1]

        lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)
        return self.e, self.xy

    analyze = analyze
    get_nto = get_nto

    def nuc_grad_method(self):
        from pyscf.tddft import rhf_grad
        return rhf_grad.Gradients(self)

CIS = TDA


def gen_tdhf_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert(mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ==2)[0]
    viridx = numpy.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]

    if wfnsym is not None and mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        sym_forbid = (orbsym[viridx].reshape(-1,1) ^ orbsym[occidx]) != wfnsym

    #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
    #fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
    #foo = fock[occidx[:,None],occidx]
    #fvv = fock[viridx[:,None],viridx]
    foo = numpy.diag(mo_energy[occidx])
    fvv = numpy.diag(mo_energy[viridx])

    e_ai = hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = _gen_rhf_response(mf, singlet=singlet, hermi=0)

    def vind(xys):
        nz = len(xys)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nvir,nocc): 2 ~ X,Y
            xys = numpy.copy(zs).reshape(nz,2,nvir,nocc)
            xys[:,:,sym_forbid] = 0
        dms = numpy.empty((nz,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            # *2 for double occupancy
            dmx = reduce(numpy.dot, (orbv, x*2, orbo.T))
            dmy = reduce(numpy.dot, (orbo, y.T*2, orbv.T))
            dms[i] = dmx + dmy  # AX + BY

        v1ao = vresp(dms)
        v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir,nocc)
        v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0,nocc,nocc,nmo))
        v1ov = v1ov.reshape(-1,nocc,nvir).transpose(0,2,1)
        hx = numpy.empty((nz,2,nvir,nocc), dtype=v1vo.dtype)
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            hx[i,0] = v1vo[i]
            hx[i,0]+= numpy.einsum('ps,sq->pq', fvv, x)  # AX
            hx[i,0]-= numpy.einsum('ps,rp->rs', foo, x)  # AX
            hx[i,1] =-v1ov[i]
            hx[i,1]-= numpy.einsum('ps,sq->pq', fvv, y)  #-AY
            hx[i,1]+= numpy.einsum('ps,rp->rs', foo, y)  #-AY

        if wfnsym is not None and mol.symmetry:
            hx[:,:,sym_forbid] = 0
        return hx.reshape(nz,-1)

    return vind, hdiag


class TDHF(TDA):
    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf):
        return gen_tdhf_operation(mf, singlet=self.singlet, wfnsym=self.wfnsym)

    def init_guess(self, mf, nstates=None, wfnsym=None):
        x0 = TDA.init_guess(self, mf, nstates, wfnsym)
        y0 = numpy.zeros_like(x0)
        return numpy.hstack((x0,y0))

    def kernel(self, x0=None, nstates=None):
        '''TDHF diagonalization with non-Hermitian eigenvalue solver
        '''
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < 1e-4) & (w.real > 0))[0]
            idx = realidx[w[realidx].real.argsort()]
            return w[idx].real, v[:,idx].real, idx

        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=self.verbose)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nvir,nocc)
            norm = 2*(lib.norm(x)**2 - lib.norm(y)**2)
            norm = 1/numpy.sqrt(norm)
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
        lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.tddft import rhf_grad
        return rhf_grad.Gradients(self)

RPA = TDHF


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol).run()
    td = TDA(mf)
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.90276464  11.90276464  16.86036434]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 11.01747918  11.01747918  13.16955056]

    td = TDHF(mf)
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309285]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440705]

