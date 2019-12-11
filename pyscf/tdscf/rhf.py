#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Recent Advances in Density Functional Methods, Chapter 5, M. E. Casida
#

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.dft import numint
from pyscf.scf import hf_symm
from pyscf.data import nist
from pyscf.soscf.newton_ah import _gen_rhf_response
from pyscf import __config__

OUTPUT_THRESHOLD = getattr(__config__, 'tdscf_rhf_get_nto_threshold', 0.3)
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)


def gen_tda_operation(mf, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute (A+B)x
    
    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
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
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff) % 10
        sym_forbid = (orbsym[occidx,None] ^ orbsym[viridx]) != wfnsym

    if fock_ao is None:
        #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

    hdiag = (fvv.diagonal().reshape(-1,1) - foo.diagonal()).T
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = _gen_rhf_response(mf, singlet=singlet, hermi=0)

    def vind(zs):
        nz = len(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs).reshape(-1,nocc,nvir)
            zs[:,sym_forbid] = 0
        dmov = numpy.empty((nz,nao,nao))
        for i, z in enumerate(zs):
            # *2 for double occupancy
            dmov[i] = reduce(numpy.dot, (orbo, z.reshape(nocc,nvir)*2, orbv.conj().T))
        v1ao = vresp(dmov)
        #v1ov = numpy.asarray([reduce(numpy.dot, (orbo.T, v, orbv)) for v in v1ao])
        v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0,nocc,nocc,nmo)).reshape(-1,nocc,nvir)
        for i, z in enumerate(zs):
            v1ov[i]+= numpy.einsum('sp,qs->qp', fvv, z.reshape(nocc,nvir))
            v1ov[i]-= numpy.einsum('sp,pr->sr', foo, z.reshape(nocc,nvir))
        if wfnsym is not None and mol.symmetry:
            v1ov[:,sym_forbid] = 0
        return v1ov.reshape(nz,-1)

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

    e_ia = lib.direct_sum('a-i->ia', mo_energy[viridx], mo_energy[occidx])
    a = numpy.diag(e_ia.ravel()).reshape(nocc,nvir,nocc,nvir)
    b = numpy.zeros_like(a)

    def add_hf_(a, b, hyb=1):
        eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
        a += numpy.einsum('iabj->iajb', eri_mo[:nocc,nocc:,nocc:,:nocc]) * 2
        a -= numpy.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:]) * hyb

        b += numpy.einsum('iajb->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * 2
        b -= numpy.einsum('jaib->iajb', eri_mo[:nocc,nocc:,:nocc,nocc:]) * hyb

    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
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

def get_nto(tdobj, state=1, threshold=OUTPUT_THRESHOLD, verbose=None):
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
            If state < 0, state ID is counted from the last excited state.

    Kwargs:
        threshold : float
            Above which the NTO coefficients will be printed in the output.

    Returns:
        A list (weights, NTOs).  NTOs are natural orbitals represented in AO
        basis. The first N_occ NTOs are occupied NTOs and the rest are virtual
        NTOs.
    '''
    if state == 0:
        logger.warn(tdobj, 'Excited state starts from 1. '
                    'Set state=1 for first excited state.')
        state_id = state
    elif state < 0:
        state_id = state
    else:
        state_id = state - 1

    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]

    cis_t1 = tdobj.xy[state_id][0]
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
            idx = numpy.where(o_sym == ir)[0]
            if idx.size > 0:
                dm_oo = numpy.dot(cis_t1[idx], cis_t1[idx].T)
                weights_o[idx], nto_o[idx[:,None],idx] = numpy.linalg.eigh(dm_oo)

            idx = numpy.where(v_sym == ir)[0]
            if idx.size > 0:
                dm_vv = numpy.dot(cis_t1[:,idx].T, cis_t1[:,idx])
                weights_v[idx], nto_v[idx[:,None],idx] = numpy.linalg.eigh(dm_vv)

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
        nto_o, w, nto_vT = numpy.linalg.svd(cis_t1)
        nto_v = nto_vT.conj().T
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
                 state_id+1, tdobj.e[state_id]*nist.HARTREE2EV, weights[0])
        o_idx = numpy.where(abs(nto_o[:,0]) > threshold)[0]
        v_idx = numpy.where(abs(nto_v[:,0]) > threshold)[0]
        fmt = '%' + str(lib.param.OUTPUT_DIGITS) + 'f (MO #%d)'
        log.info('    occ-NTO: ' +
                 ' + '.join([(fmt % (nto_o[i,0], i+MO_BASE))
                             for i in o_idx]))
        log.info('    vir-NTO: ' +
                 ' + '.join([(fmt % (nto_v[i,0], i+MO_BASE+nocc))
                             for i in v_idx]))
    return weights, nto_coeff


def analyze(tdobj, verbose=None):
    log = logger.new_logger(tdobj, verbose)
    mol = tdobj.mol
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    nocc = numpy.count_nonzero(mo_occ == 2)

    e_ev = numpy.asarray(tdobj.e) * nist.HARTREE2EV
    e_wn = numpy.asarray(tdobj.e) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    if tdobj.singlet:
        log.note('\n** Singlet excitation energies and oscillator strengths **')
    else:
        log.note('\n** Triplet excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff) % 10
        x_sym = (orbsym[mo_occ==2,None] ^ orbsym[mo_occ==0]).ravel()
    else:
        x_sym = None

    f_oscillator = tdobj.oscillator_strength()
    for i, ei in enumerate(tdobj.e):
        x, y = tdobj.xy[i]
        if x_sym is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm  f=%.4f',
                     i+1, e_ev[i], wave_length[i], f_oscillator[i])
        else:
            wfnsym_id = x_sym[abs(x).argmax()]
            wfnsym = symm.irrep_id2name(mol.groupname, wfnsym_id)
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm  f=%.4f',
                     i+1, wfnsym, e_ev[i], wave_length[i], f_oscillator[i])

        if log.verbose >= logger.INFO:
            o_idx, v_idx = numpy.where(abs(x) > 0.1)
            for o, v in zip(o_idx, v_idx):
                log.info('    %4d -> %-4d %12.5f',
                         o+MO_BASE, v+MO_BASE+nocc, x[o,v])

    if log.verbose >= logger.INFO:
        log.info('\n** Transition electric dipole moments (AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_dip = tdobj.transition_dipole()
        for i, ei in enumerate(tdobj.e):
            dip = trans_dip[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, dip[0], dip[1], dip[2], numpy.dot(dip, dip),
                     f_oscillator[i])

        log.info('\n** Transition velocity dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z        Dip. S.      Osc.')
        trans_v = tdobj.transition_velocity_dipole()
        f_v = tdobj.oscillator_strength(gauge='velocity', order=0)
        for i, ei in enumerate(tdobj.e):
            v = trans_v[i]
            log.info('%3d    %11.4f %11.4f %11.4f %11.4f %11.4f',
                     i+1, v[0], v[1], v[2], numpy.dot(v, v), f_v[i])

        log.info('\n** Transition magnetic dipole moments (imaginary part, AU) **')
        log.info('state          X           Y           Z')
        trans_m = tdobj.transition_magnetic_dipole()
        for i, ei in enumerate(tdobj.e):
            m = trans_m[i]
            log.info('%3d    %11.4f %11.4f %11.4f',
                     i+1, m[0], m[1], m[2])
    return tdobj


def transition_dipole(tdobj, xy=None):
    '''Transition dipole moments in the length gauge'''
    mol = tdobj.mol
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor_symmetric('int1e_r', comp=3)
    return tdobj._contract_multipole(ints, hermi=True, xy=xy)

def transition_velocity_dipole(tdobj, xy=None):
    '''Transition dipole moments in the velocity gauge (imaginary part only)
    '''
    ints = tdobj.mol.intor('int1e_ipovlp', comp=3, hermi=2)
    v = tdobj._contract_multipole(ints, hermi=False, xy=xy)
    return -v

def transition_magnetic_dipole(tdobj, xy=None):
    '''Transition magnetic dipole moments (imaginary part only)'''
    mol = tdobj.mol
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_cg_irxp', comp=3, hermi=2)
    m_pol = tdobj._contract_multipole(ints, hermi=False, xy=xy)
    return -m_pol

def transition_quadrupole(tdobj, xy=None):
    '''Transition quadrupole moments in the length gauge'''
    mol = tdobj.mol
    nao = mol.nao_nr()
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_rr', comp=9, hermi=0).reshape(3,3,nao,nao)
    quad = tdobj._contract_multipole(ints, hermi=True, xy=xy)
    return quad

def transition_velocity_quadrupole(tdobj, xy=None):
    '''Transition quadrupole moments in the velocity gauge (imaginary part only)
    '''
    mol = tdobj.mol
    nao = mol.nao_nr()
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_irp', comp=9, hermi=0).reshape(3,3,nao,nao)
    ints = ints + ints.transpose(1,0,3,2)
    quad = tdobj._contract_multipole(ints, hermi=True, xy=xy)
    return -quad

def transition_magnetic_quadrupole(tdobj, xy=None):
    '''Transition magnetic quadrupole moments (imaginary part only)'''
    XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = range(9)
    mol = tdobj.mol
    nao = mol.nao_nr()
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(3,9,nao,nao)
    m_ints = (ints[:,[YZ,ZX,XY]] - ints[:,[ZY,XZ,YX]]).transpose(1,0,2,3)
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_irpr', comp=27, hermi=0).reshape(9,3,nao,nao)
    m_ints += ints[[YZ,ZX,XY]] - ints[[ZY,XZ,YX]]
    m_quad = tdobj._contract_multipole(m_ints, hermi=True, xy=xy)
    return -m_quad

def transition_octupole(tdobj, xy=None):
    '''Transition octupole moments in the length gauge'''
    mol = tdobj.mol
    nao = mol.nao_nr()
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_rrr', comp=27, hermi=0).reshape(3,3,3,nao,nao)
    o_pol = tdobj._contract_multipole(ints, hermi=True, xy=xy)
    return o_pol

def transition_velocity_octupole(tdobj, xy=None):
    '''Transition octupole moments in the velocity gauge (imaginary part only)
    '''
    mol = tdobj.mol
    nao = mol.nao_nr()
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor('int1e_irrp', comp=27, hermi=0).reshape(3,3,3,nao,nao)
    ints = ints + ints.transpose(2,1,0,4,3)
    with mol.with_common_orig(_charge_center(mol)):
        ints += mol.intor('int1e_irpr', comp=27, hermi=0).reshape(3,3,3,nao,nao)
    o_pol = tdobj._contract_multipole(ints, hermi=True, xy=xy)
    return -o_pol

def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return numpy.einsum('z,zr->r', charges, coords)/charges.sum()

def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    if xy is None: xy = tdobj.xy
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]

    nstates = len(xy)
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo.conj(), orbv)
    ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo.conj(), orbv)
    pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
    if isinstance(xy[0][1], numpy.ndarray):
        if hermi:
            pol += [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        else:  # anti-Hermitian
            pol -= [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol

def oscillator_strength(tdobj, e=None, xy=None, gauge='length', order=0):
    if e is None: e = tdobj.e

    if gauge == 'length':
        trans_dip = transition_dipole(tdobj, xy)
        f = 2./3. * numpy.einsum('s,sx,sx->s', e, trans_dip, trans_dip)
        return f

    else:  # velocity gauge
        # Ref. JCP, 143, 234103
        trans_dip = transition_velocity_dipole(tdobj, xy)
        f = 2./3. * numpy.einsum('s,sx,sx->s', 1./e, trans_dip, trans_dip)

        if order > 0:
            m_dip = .5 * transition_magnetic_dipole(tdobj, xy)
            f_m = numpy.einsum('s,sx,sx->s', e, m_dip, m_dip)
            f_m = nist.ALPHA**2/6 * f_m.real
            f += f_m

            quad = .5 * transition_velocity_quadrupole(tdobj, xy)
            f_quad = numpy.einsum('s,sxy,sxy->s', e, quad, quad)
            f_quad-= 1./3 * numpy.einsum('s,sxx,sxx->s', e, quad, quad)
            f_quad = nist.ALPHA**2/20 * f_quad.real
            f += f_quad
            logger.debug(tdobj, '    First order correction to oscillator '
                         'strength (velocity gague)')
            logger.debug(tdobj, '    %s', f_m+f_quad)

        if order > 1:
            m_quad = -1./6 * 1j*transition_magnetic_quadrupole(tdobj, xy)
            f_m = numpy.einsum('s,sy,szx,xyz->s', e, trans_dip*1j, m_quad,
                               lib.LeviCivita)
            f_m = nist.ALPHA**3/9 * f_m.real
            f += f_m

            o_pol = -1./6 * 1j*transition_velocity_octupole(tdobj, xy)
            f_o = numpy.einsum('s,sy,sxxy->s', e, trans_dip*1j, o_pol)
            f_o = -2*nist.ALPHA**2/45 * f_o.real
            f += f_o
            logger.debug(tdobj, '    Second order correction to oscillator '
                         'strength (velocity gague)')
            logger.debug(tdobj, '    %s', f_m+f_o)

    return f


def as_scanner(td):
    '''Generating a scanner/solver for TDA/TDHF/TDDFT PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total TDA/TDHF/TDDFT energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    TDA/TDDFT and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, tdscf
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> td_scanner = tdscf.TDHF(scf.RHF(mol)).as_scanner()
    >>> de = td_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    [ 0.34460866  0.34460866  0.7131453 ]
    >>> de = td_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    [ 0.14844013  0.14844013  0.47641829]
    '''
    from pyscf import gto
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)

    class TD_Scanner(td.__class__, lib.SinglePointScanner):
        def __init__(self, td):
            self.__dict__.update(td.__dict__)
            self._scf = td._scf.as_scanner()
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mf_scanner = self._scf
            mf_e = mf_scanner(mol)
            self.mol = mol
            self.kernel(**kwargs)
            return mf_e + self.e
    return TD_Scanner(td)


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
            The two 2D arrays are Excitation coefficients X (shape [nocc,nvir])
            and de-excitation coefficients Y (shape [nocc,nvir]) for each
            excited state.  (X,Y) are normalized to 1/2 in RHF/RKS methods and
            normalized to 1 for UHF/UKS methods. In the TDA calculation, Y = 0.
    '''
    conv_tol = getattr(__config__, 'tdscf_rhf_TDA_conv_tol', 1e-9)
    nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)
    singlet = getattr(__config__, 'tdscf_rhf_TDA_singlet', True)
    lindep = getattr(__config__, 'tdscf_rhf_TDA_lindep', 1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift', 0)
    max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 50)
    max_cycle = getattr(__config__, 'tdscf_rhf_TDA_max_cycle', 100)

    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.mol = mf.mol
        self._scf = mf
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        self.wfnsym = None

        # xy = (X,Y), normalized to 1/2: 2(XX-YY) = 1
        # In TDA, Y = 0
        self.converged = None
        self.e = None
        self.xy = None

        keys = set(('conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift',
                    'max_space', 'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

    @property
    def nroots(self):
        return self.nstates
    @nroots.setter
    def nroots(self, x):
        self.nstates = x

    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.e

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
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
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]

        if wfnsym is not None and mf.mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mf.mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsym = hf_symm.get_orbsym(mf.mol, mf.mo_coeff) % 10
            e_ia[(orbsym[occidx,None] ^ orbsym[viridx]) != wfnsym] = 1e99

        nov = e_ia.size
        nroot = min(nstates, nov)
        x0 = numpy.zeros((nroot, nov))
        idx = numpy.argsort(e_ia.ravel())
        for i in range(nroot):
            x0[i,idx[i]] = 1  # Koopmans' excitations
        return x0

    def kernel(self, x0=None, nstates=None):
        '''TDA diagonalization solver
        '''
        cpu0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > POSTIVE_EIG_THRESHOLD**2)[0]
            return w[idx], v[:,idx], idx

        self.converged, self.e, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        self.xy = [(xi.reshape(nocc,nvir)*numpy.sqrt(.5),0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDA', *cpu0)
        log.note('Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.xy

    analyze = analyze
    get_nto = get_nto
    oscillator_strength = oscillator_strength

    _contract_multipole = _contract_multipole  # needed by following methods
    transition_dipole              = transition_dipole
    transition_quadrupole          = transition_quadrupole
    transition_octupole            = transition_octupole
    transition_velocity_dipole     = transition_velocity_dipole
    transition_velocity_quadrupole = transition_velocity_quadrupole
    transition_velocity_octupole   = transition_velocity_octupole
    transition_magnetic_dipole     = transition_magnetic_dipole
    transition_magnetic_quadrupole = transition_magnetic_quadrupole

    as_scanner = as_scanner

    def nuc_grad_method(self):
        from pyscf.grad import tdrhf
        return tdrhf.Gradients(self)

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
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff) % 10
        sym_forbid = (orbsym[occidx,None] ^ orbsym[viridx]) != wfnsym

    #dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    #fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
    #fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
    #foo = fock[occidx[:,None],occidx]
    #fvv = fock[viridx[:,None],viridx]
    foo = numpy.diag(mo_energy[occidx])
    fvv = numpy.diag(mo_energy[viridx])

    hdiag = (fvv.diagonal().reshape(-1,1) - foo.diagonal()).T
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo,orbv)), order='F')
    vresp = _gen_rhf_response(mf, singlet=singlet, hermi=0)

    def vind(xys):
        nz = len(xys)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys).reshape(nz,2,nocc,nvir)
            xys[:,:,sym_forbid] = 0
        dms = numpy.empty((nz,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nocc,nvir)
            # *2 for double occupancy
            dmx = reduce(numpy.dot, (orbo, x*2, orbv.T))
            dmy = reduce(numpy.dot, (orbv, y.T*2, orbo.T))
            dms[i] = dmx + dmy  # AX + BY

        v1ao = vresp(dms)
        v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0,nocc,nocc,nmo)).reshape(-1,nocc,nvir)
        v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc,nmo,0,nocc)).reshape(-1,nvir,nocc)
        hx = numpy.empty((nz,2,nocc,nvir), dtype=v1ov.dtype)
        for i in range(nz):
            x, y = xys[i].reshape(2,nocc,nvir)
            hx[i,0] = v1ov[i]
            hx[i,0]+= numpy.einsum('sp,qs->qp', fvv, x)  # AX
            hx[i,0]-= numpy.einsum('sp,pr->sr', foo, x)  # AX
            hx[i,1] =-v1vo[i].T
            hx[i,1]-= numpy.einsum('sp,qs->qp', fvv, y)  #-AY
            hx[i,1]+= numpy.einsum('sp,pr->sr', foo, y)  #-AY

        if wfnsym is not None and mol.symmetry:
            hx[:,:,sym_forbid] = 0
        return hx.reshape(nz,-1)

    return vind, hdiag


class TDHF(TDA):
    '''Time-dependent Hartree-Fock

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
            The two 2D arrays are Excitation coefficients X (shape [nocc,nvir])
            and de-excitation coefficients Y (shape [nocc,nvir]) for each
            excited state.  (X,Y) are normalized to 1/2 in RHF/RKS methods and
            normalized to 1 for UHF/UKS methods. In the TDA calculation, Y = 0.
    '''
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
        cpu0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > POSTIVE_EIG_THRESHOLD))[0]
            # If the complex eigenvalue has small imaginary part, both the
            # real part and the imaginary part of the eigenvector can
            # approximately be used as the "real" eigen solutions.
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                      real_eigenvectors=True)

        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=log)

        nocc = (self._scf.mo_occ>0).sum()
        nmo = self._scf.mo_occ.size
        nvir = nmo - nocc
        self.e = w
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return x*norm, y*norm
        self.xy = [norm_xy(z) for z in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.timer('TDDFT', *cpu0)
        log.note('Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tdrhf
        return tdrhf.Gradients(self)

RPA = TDRHF = TDHF

from pyscf import scf
scf.hf.RHF.TDA = lib.class_as_method(TDA)
scf.hf.RHF.TDHF = lib.class_as_method(TDHF)
scf.rohf.ROHF.TDA = None
scf.rohf.ROHF.TDHF = None
scf.hf_symm.ROHF.TDA = None
scf.hf_symm.ROHF.TDHF = None

del(OUTPUT_THRESHOLD)


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
    td = mf.TDA()
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.90276464  11.90276464  16.86036434]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 11.01747918  11.01747918  13.16955056]

    td = mf.TDHF()
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.83487199  11.83487199  16.66309285]

    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [ 10.8919234   10.8919234   12.63440705]

