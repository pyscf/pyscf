#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf.mcscf import newton_casscf, casci, mc1step
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger, pack_tril, current_memory, einsum, tag_array
from pyscf.grad import sacasscf
from pyscf.mcscf.casci import cas_natorb

from pyscf.mcpdft.pdft_eff import _contract_eff_rho
from pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from pyscf.mcpdft import _dms

from itertools import product
from scipy import linalg
import numpy as np
import gc

BLKSIZE = gen_grid.BLKSIZE

def gfock_sym(mc, mo_coeff, casdm1, casdm2, h1e, eris):
    """Assume that h2e v_j = v_k"""
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas

    nao, nmo = mo_coeff.shape

    # gfock = Generalized Fock, Adv. Chem. Phys., 69, 63

    # MRH: I need to replace aapa with the equivalent array from veff2
    # I'm not sure how the outcore file-paging system works
    # I also need to generate vhf_c and vhf_a from veff2 rather than the
    # molecule's actual integrals. The true Coulomb repulsion should already be
    # in veff1, but I need to generate the "fake" vj - vk/2 from veff2
    h1e_mo = mo_coeff.T @ h1e @ mo_coeff + eris.vhf_c
    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=h1e_mo.dtype)
    vhf_a = np.zeros((nmo, nmo), dtype=h1e_mo.dtype)

    for i in range(nmo):
        jbuf = eris.ppaa[i]
        aapa[:, :, i, :] = jbuf[ncore:nocc, :, :]
        vhf_a[i] = np.tensordot(jbuf, casdm1, axes=2)

    vhf_a *= 0.5
    # we have assumed that vj = vk: vj - vk/2 = vj - vj/2 = vj/2
    gfock = np.zeros((nmo, nmo))
    gfock[:, :ncore] = (h1e_mo[:, :ncore] + vhf_a[:, :ncore]) * 2
    gfock[:, ncore:nocc] = h1e_mo[:, ncore:nocc] @ casdm1
    gfock[:, ncore:nocc] += einsum('uviw,vuwt->it', aapa, casdm2)

    return gfock

def xc_response(ot, vot, rho, Pi, weights, moval_occ, aoval, mo_occ, mo_occup, ncore, nocc, casdm2_pack, ndpi, mo_cas):
    vrho, vPi = vot


    # Vpq + Vpqrs * Drs ; I'm not sure why the list comprehension down
    # there doesn't break ao's stride order but I'm not complaining
    vrho = _contract_eff_rho(vPi, rho.sum(0), add_eff_rho=vrho)

    tmp_dv = np.stack(
        [
            ot.get_veff_1body(rho, Pi, [ao_i, moval_occ], weights, kern=vrho)
            for ao_i in aoval
        ],
        axis=0,
    )
    tmp_dv = (tmp_dv * mo_occ[None, :, :] * mo_occup[None, None, :nocc]).sum(2)

    # Vpuvx * Lpuvx ; remember the stupid slowest->fastest->medium
    # stride order of the ao grid arrays
    moval_cas = np.ascontiguousarray(moval_occ[..., ncore:].transpose(0,2,1)).transpose(0,2,1)

    tmp_dv1 = ot.get_veff_2body_kl(rho, Pi, moval_cas, moval_cas, weights, symm=True, kern=vPi)
    # tmp_dv.shape = ndpi,ngrids,ncas*(ncas+1)//2
    tmp_dv1 = np.tensordot(tmp_dv1, casdm2_pack, axes=(-1,-1))
    # tmp_dv.shape = ndpi, ngrids, ncas, ncas
    tmp_dv1[0] = (tmp_dv1[:ndpi] * moval_cas[:ndpi, :, None, :]).sum(0)
    # Chain and product rule
    tmp_dv1[1:ndpi] *= moval_cas[0, :, None, :]
    # Chain and product rule
    tmp_dv1 = tmp_dv1.sum(-1)
    # tmp_dv.shape = ndpi, ngrids, ncas
    tmp_dv1 = np.tensordot(aoval[:, :ndpi], tmp_dv1, axes=((1, 2), (0, 1)))
    # tmp_dv.shape = comp, nao (orb), ncas (dm2)
    tmp_dv1 = np.einsum('cpu,pu->cp', tmp_dv1, mo_cas)
    # tmp_dv.shape = comp, ncas

    return tmp_dv + tmp_dv1


def pack_casdm2(cascm2, ncas):
    diag_idx = np.arange(ncas)  # for puvx
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx

    casdm2_pack = (cascm2 + cascm2.transpose(0, 1, 3, 2)).reshape(ncas**2, ncas, ncas)
    casdm2_pack = pack_tril(casdm2_pack).reshape(ncas, ncas, -1)
    casdm2_pack[:, :, diag_idx] *= 0.5
    return casdm2_pack

def sum_terms(mf_grad, mol, atmlst,dm1, gfock, coul_term, dvxc):
    de_hcore = np.zeros((len(atmlst), 3))
    de_renorm = np.zeros((len(atmlst), 3))
    de_coul = np.zeros((len(atmlst), 3))
    de_xc = np.zeros((len(atmlst), 3))

    aoslices = mol.aoslice_by_atom()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        h1ao = hcore_deriv(ia)
        de_hcore[k] += np.tensordot(h1ao, dm1)
        de_renorm[k] -= np.tensordot(s1[:, p0:p1], gfock[p0:p1])*2
        de_coul[k] += coul_term(p0, p1)
        de_xc[k] += dvxc[:, p0:p1].sum(1)*2

    de_nuc = mf_grad.grad_nuc(mol, atmlst)

    return de_hcore, de_coul, de_xc, de_nuc, de_renorm,

def mcpdft_HellmanFeynman_grad (mc, ot, veff1, veff2, mo_coeff=None, ci=None,
        atmlst=None, mf_grad=None, verbose=None, max_memory=None,
        auxbasis_response=False):
    '''Modification of pyscf.grad.casscf.kernel to compute instead the
    Hellman-Feynman gradient terms of MC-PDFT. From the differentiated
    Hamiltonian matrix elements, only the core and Coulomb energy parts
    remain. For the renormalization terms, the effective Fock matrix is
    as in CASSCF, but with the same Hamiltonian substutition that is
    used for the energy response terms. '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc.get_rhf_base ().nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    if max_memory is None: max_memory = mc.max_memory
    t0 = (logger.process_clock (), logger.perf_counter ())

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape

    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

    spin = abs(nelecas[0] - nelecas[1])
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError(
            "hybrid on-top functionals with different exchange,correlation components"
        )

    cas_hyb = hyb[0]
    ot_hyb = 1.0 - cas_hyb

    if cas_hyb > 1e-11:
        # TODO: actually implement this in a more efficient manner
        # That is, lets not call grad_elec, but lets actually do this our self maybe?
        # Can then use get_pdft_veff with drop_mcwfn = False to automatically get
        # Generalized fock matrix terms, but then have to deal explicitly with the
        # derivative of eri terms and auxbasis stuff. Also, get_pdft_veff with
        # drop_mcwfn=False means the get_wfn_response doesn't have to add the
        # eris to the veff2 since it should just include it already
        if auxbasis_response:
            from pyscf.df.grad import casscf as casscf_grad
        else:
            from pyscf.grad import casscf as casscf_grad

        cas_grad = casscf_grad.Gradients(mc)

        de_cas = cas_hyb * cas_grad.grad_elec(
            mo_coeff=mo_coeff, ci=ci, atmlst=atmlst, verbose=verbose
        )

    # gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = 2 * mo_core @ mo_core.T
    dm_cas = mo_cas @ casdm1 @ mo_cas.T

    gfock = gfock_sym(mc, mo_coeff, casdm1, casdm2, ot_hyb*mc.get_hcore() + veff1, veff2)

    dme0 = mo_coeff @ (0.5*(gfock+gfock.T)) @ mo_coeff.T
    del gfock

    if atmlst is None:
        atmlst = range(mol.natm)

    de_grid = np.zeros ((len(atmlst),3))
    de_wgt = np.zeros ((len(atmlst),3))
    de_aux = np.zeros ((len(atmlst),3))

    t0 = logger.timer (mc, 'PDFT HlFn gfock', *t0)
    mo_coeff, ci, mo_occup = cas_natorb (mc, mo_coeff=mo_coeff, ci=ci)
    mo_occ = mo_coeff[:,:nocc]
    mo_cas = mo_coeff[:,ncore:nocc]

    dm1 = dm_core + dm_cas
    dm1 = tag_array (dm1, mo_coeff=mo_coeff, mo_occ=mo_occup)

    # MRH: vhf1c and vhf1a should be the TRUE vj_c and vj_a (no vk!)
    vj = mf_grad.get_jk (dm=dm1)[0]
    if auxbasis_response:
        de_aux += ot_hyb*np.squeeze (vj.aux[:,:,atmlst,:])

    # MRH: Now I have to compute the gradient of the on-top energy
    # This involves derivatives of the orbitals that construct rho and Pi and
    # therefore another set of potentials. It also involves the derivatives of
    # quadrature grid points which propagate through the densities and
    # therefore yet another set of potentials. The orbital-derivative part
    # includes all the grid points and some of the orbitals (- sign); the
    # grid-derivative part includes all of the orbitals and some of the grid
    # points (+ sign). I'll do a loop over grid sections and make arrays of
    # type (3,nao, nao) and (3,nao, ncas, ncas, ncas). I'll contract them
    # within the grid loop for the grid derivatives and in the following
    # orbital loop for the xc derivatives
    # MRH, 05/09/2020: The actual spin density doesn't matter at all in PDFT!
    # I could probably save a fair amount of time by not screwing around with
    # the actual spin density! Also, the cumulant decomposition can always be
    # defined without the spin-density matrices and it's still valid!
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    twoCDM = _dms.dm2_cumulant (casdm2, casdm1)
    dm1 = tag_array (dm1, mo_coeff=mo_occ, mo_occ=mo_occup[:nocc])
    make_rho = ot._numint._gen_rho_evaluator (mol, dm1, hermi=1, with_lapl=False)[0]
    dvxc = np.zeros ((3,nao))
    idx = np.array ([[1,4,5,6],[2,5,7,8],[3,6,8,9]], dtype=np.int_)
    # For addressing particular ao derivatives
    if ot.xctype == 'LDA': idx = idx[:,0:1] # For LDAs, no second derivatives

    casdm2_pack = pack_casdm2(twoCDM, ncas)
    full_atmlst = -np.ones (mol.natm, dtype=np.int_)

    t1 = logger.timer (mc, 'PDFT HlFn quadrature setup', *t0)
    for k, ia in enumerate (atmlst):
        full_atmlst[ia] = k

    # for LDA we need 1 deriv, GGA: 2 deriv
    # for mGGA with tau, we only need 2 deriv
    ao_deriv = (1, 2, 2)[ot.dens_deriv]
    ndao = (1, 4, 10)[ao_deriv]
    ndrho = (1, 4, 5)[ot.dens_deriv]
    ndpi = (1, 4)[ot.Pi_deriv]
    ncols = 1.05 * 3 * (ndao * (nao + nocc) + max(ndrho * nao, ndpi * ncas * ncas))

    for ia, (coords, w0, w1) in enumerate (rks_grad.grids_response_cc (
            ot.grids)):
        # For the xc potential derivative, I need every grid point in the
        # entire molecule regardless of atmlist. (Because that's about orbs.)
        # For the grid and weight derivatives, I only need the gridpoints that
        # are in atmlst. It is conceivable that I can make this more efficient
        # by only doing cross-combinations of grids and AOs, but I don't know
        # how "mask" works yet or how else I could do this.
        gc.collect ()
        ngrids = coords.shape[0]

        remaining_floats = (max_memory - current_memory()[0]) * 1e6 / 8
        blksize = int (remaining_floats / (ncols*BLKSIZE)) * BLKSIZE
        blksize = max (BLKSIZE, min (blksize, ngrids, BLKSIZE*1200))
        t1 = logger.timer (mc, 'PDFT HlFn quadrature atom {} mask and memory '
            'setup'.format (ia), *t1)

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0 + blksize)
            mask = gen_grid.make_mask(mol, coords[ip0:ip1])
            logger.info(
                mc,
                ("PDFT gradient atom {} slice {}-{} of {} total").format(
                    ia, ip0, ip1, ngrids
                ),
            )

            ao = ot._numint.eval_ao(mol, coords[ip0:ip1], deriv=ao_deriv, non0tab=mask)

            # Need 1st derivs for LDA, 2nd for GGA, etc.
            t1 = logger.timer(
                mc, ("PDFT HlFn quadrature atom {} ao " "grids").format(ia), *t1
            )
            # Slice down ao so as not to confuse the rho and Pi generators
            if ot.xctype == "LDA":
                aoval = ao[0]
            elif ot.xctype == "GGA":
                aoval = ao[:4]
            elif ot.xctype == "MGGA":
                aoval = ao[:4]
            else:
                raise ValueError("Unknown xctype: {}".format(ot.xctype))

            rho = make_rho (0, aoval, mask, ot.xctype) / 2.0
            rho = np.stack ((rho,)*2, axis=0)
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {} rho '
                'calc').format (ia), *t1)
            Pi = get_ontop_pair_density (ot, rho, aoval, twoCDM, mo_cas,
                ot.Pi_deriv, mask)
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {} Pi '
                'calc').format (ia), *t1)

            # TODO: consistent format requirements for shape of ao grid
            if ot.xctype == 'LDA':
                aoval = ao[:1]
            moval_occ = _grid_ao2mo (mol, aoval, mo_occ, mask)
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {} ao2mo '
                'grid').format (ia), *t1)
            aoval = np.ascontiguousarray ([ao[ix].transpose (0,2,1)
                for ix in idx[:,:ndao]]).transpose (0,1,3,2)
            ao = None
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {} ao grid '
                'reshape').format (ia), *t1)
            eot, vot = ot.eval_ot (rho, Pi, weights=w0[ip0:ip1])[:2]
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {} '
                'eval_ot').format (ia), *t1)
            puvx_mem = 2 * ndpi * (ip1-ip0) * ncas * ncas * 8 / 1e6
            remaining_mem = max_memory - current_memory ()[0]
            logger.info (mc, ('PDFT gradient memory note: working on {} grid '
                'points; estimated puvx usage = {:.1f} of {:.1f} remaining '
                'MB').format ((ip1-ip0), puvx_mem, remaining_mem))

            # Weight response
            de_wgt += np.tensordot (eot, w1[atmlst,...,ip0:ip1], axes=(0,2))
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {} weight '
                'response').format (ia), *t1)

            # Find the atoms that are a part of the atomlist
            # grid correction shouldn't be added if they aren't there
            # The last stuff to vectorize is in get_veff_2body!
            k = full_atmlst[ia]

            tmp_dv = xc_response(ot, vot, rho, Pi, w0[ip0:ip1], moval_occ, aoval, mo_occ, mo_occup, ncore, nocc,
                                 casdm2_pack, ndpi, mo_cas)

            if k >=0: de_grid[k] += 2*tmp_dv.sum(1) # Grid response
            dvxc -= tmp_dv #XC response

            tmp_dv = None
            t1 = logger.timer (mc, ('PDFT HlFn quadrature atom {}').format (ia), *t1)

            rho = Pi = eot = vot = aoval = moval_occ = None
            gc.collect ()

    def coul_term(p0, p1):
        return np.tensordot(vj[:,p0:p1], dm1[p0:p1])*2

    de_hcore, de_coul, de_xc, de_nuc, de_renorm = sum_terms(mf_grad, mol, atmlst, dm1, dme0, coul_term,
                                                                        dvxc)
    # Deal with hybridization
    de_hcore *= ot_hyb
    de_coul *= ot_hyb

    logger.debug (mc, "MC-PDFT Hellmann-Feynman nuclear:\n{}".format (de_nuc))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman hcore component:\n{}".format (
        de_hcore))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman coulomb component:\n{}".format
        (de_coul))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman xc component:\n{}".format (
        de_xc))
    logger.debug (mc, ("MC-PDFT Hellmann-Feynman quadrature point component:"
        "\n{}").format (de_grid))
    logger.debug (mc, ("MC-PDFT Hellmann-Feynman quadrature weight component:"
        "\n{}").format (de_wgt))
    logger.debug (mc, "MC-PDFT Hellmann-Feynman renorm component:\n{}".format (
        de_renorm))

    de = de_nuc + de_hcore + de_coul + de_renorm + de_xc + de_grid + de_wgt


    if auxbasis_response:
        de += de_aux
        logger.debug (mc, "MC-PDFT Hellmann-Feynman aux component:\n{}".format
            (de_aux))

    if cas_hyb > 1e-11:
        de += de_cas
        logger.debug(mc, "MC-PDFT Hellmann-Feynman CAS component:\n{}".format(de_cas))

    t1 = logger.timer (mc, 'PDFT HlFn total', *t0)

    return de

# TODO: docstrings (parent classes???)
# TODO: add a consistent threshold for elimination of degenerate-state rotations
class Gradients (sacasscf.Gradients):

    def __init__(self, pdft, state=None):
        super().__init__(pdft, state=state)
        # TODO: gradient of PDFT state-average energy
        # i.e., state = 0 & nroots > 1 case
        if self.state is None and self.nroots == 1:
            self.state = 0
        self.e_mcscf = self.base.e_mcscf
        self._not_implemented_check ()

    def _not_implemented_check (self):
        name = self.__class__.__name__
        if (isinstance (self.base, casci.CASCI) and not
            isinstance (self.base, mc1step.CASSCF)):
            raise NotImplementedError (
                "{} for CASCI-based MC-PDFT".format (name)
            )
        ot, otxc, nelecas = self.base.otfnal, self.base.otxc, self.base.nelecas
        spin = abs (nelecas[0]-nelecas[1])
        omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff (
            otxc, spin=spin)
        hyb_x, hyb_c = hyb
        if abs(hyb_x - hyb_c) >1e-11:
            raise NotImplementedError (
                "{} for hybrid MC-PDFT functionals with different exchange, correlation".format (name)
            )
        if omega:
            raise NotImplementedError (
                "{} for range-separated MC-PDFT functionals".format (name)
            )

    def get_wfn_response (self, state=None, verbose=None, mo=None,
            ci=None, veff1=None, veff2=None, nlag=None, **kwargs):
        if state is None: state = self.state
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if nlag is None: nlag = self.nlag
        if (veff1 is None) or (veff2 is None):
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[state],
                incl_coul=True, paaa_only=True, drop_mcwfn=True)

        log = logger.new_logger(self, verbose)

        sing_tol = getattr(self, 'sing_tol_sasa', 1e-8)
        fcasscf = self.make_fcasscf(state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]
        def my_hcore ():
            return self.base.get_hcore () + veff1
        fcasscf.get_hcore = my_hcore

        nelecas = self.base.nelecas
        spin = abs (nelecas[0]-nelecas[1])
        omega, alpha, hyb = self.base.otfnal._numint.rsh_and_hybrid_coeff(self.base.otxc, spin=spin)
        if omega:
            raise NotImplementedError("range-separated MC-PDFT functionals")
        if abs(hyb[0] - hyb[1]) > 1e-11:
            raise NotImplementedError("hybrid on-top functional with different exchange,correlation components")
        cas_hyb = hyb[0]

        if cas_hyb > 1e-11:# and len(ci) > 1:
            # For SS-CASSCF, there are no Lagrange multipliers
            # This is only needed in SA case
            eris = self.base.ao2mo(mo_coeff=mo)
            terms = ["vhf_c", "papa", "ppaa", "j_pc", "k_pc"]
            for term in terms:
                setattr(eris, term, getattr(veff2, term) + cas_hyb*getattr(eris, term)[:])
            veff2.vhf_c
            veff2 = eris


        g_all_state = newton_casscf.gen_g_hop (fcasscf, mo, ci[state], veff2, verbose)[0]

        g_all = np.zeros (nlag)
        g_all[:self.ngorb] = g_all_state[:self.ngorb]
        # Eliminate gradient of self-rotation and rotation into
        # degenerate states
        spin_states = np.asarray (self.spin_states)
        idx_spin = spin_states==spin_states[state]
        e_gap = self.e_mcscf-self.e_mcscf[state] if self.nroots>1 else [0.0]
        idx_degen = np.abs (e_gap)<sing_tol
        idx = np.where (idx_spin & idx_degen)[0]
        assert (state in idx)
        gci_state = g_all_state[self.ngorb:]
        ci_proj = np.asarray ([ci[i].ravel () for i in idx])
        gci_sa = np.dot (ci_proj, gci_state)
        gci_state -= np.dot (gci_sa, ci_proj)
        gci = g_all[self.ngorb:]
        offs = 0
        if state>0:
            offs = sum ([na * nb for na, nb in zip(
                        self.na_states[:state], self.nb_states[:state])])
        ndet = self.na_states[state]*self.nb_states[state]
        gci[offs:][:ndet] += gci_state

        # Debug
        log.debug("g_all mo:\n{}".format(g_all[:self.ngorb]))
        log.debug("g_all CI:\n{}".format(g_all[self.ngorb:]))

        return g_all

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None,
            ci=None, eris=None, mf_grad=None, veff1=None, veff2=None,
            **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]

        return mcpdft_HellmanFeynman_grad(
            fcasscf,
            self.base.otfnal,
            veff1,
            veff2,
            mo_coeff=mo,
            ci=ci[state],
            atmlst=atmlst,
            mf_grad=mf_grad,
            verbose=verbose,
        )

    def get_init_guess (self, bvec, Adiag, Aop, precond):
        '''Initial guess should solve the problem for SA-SA rotations'''
        sing_tol = getattr (self, 'sing_tol_sasa', 1e-8)
        ci = self.base.ci
        state = self.state
        if self.nroots == 1: ci = [ci,]
        idx_spin = [i for i in range (self.nroots)
                    if self.spin_states[i]==self.spin_states[state]]
        ci_blk = np.asarray ([ci[i].ravel () for i in idx_spin])
        b_orb, b_ci = self.unpack_uniq_var (bvec)
        b_ci_blk = np.asarray ([b_ci[i].ravel () for i in idx_spin])
        x0 = np.zeros_like (bvec)
        if self.nroots > 1:
            b_sa = np.dot (ci_blk.conjugate (), b_ci[state].ravel ())
            A_sa = 2 * self.weights[state] * (self.e_mcscf
                - self.e_mcscf[state])
            idx_null = np.abs (A_sa)<sing_tol
            assert (idx_null[state])
            A_sa = A_sa[idx_spin]
            idx_null = np.abs (A_sa)<sing_tol
            if np.any (np.abs (b_sa[idx_null])>=sing_tol):
                logger.warn (self, 'Singular Hessian in CP-MCPDFT!')
            idx_null &= np.abs (b_sa)<sing_tol
            A_sa[idx_null] = sing_tol
            x0_sa = -b_sa / A_sa # Hessian is diagonal so: easy
            ovlp = ci_blk.conjugate () @ b_ci_blk.T
            logger.debug (self, 'Linear response SA-SA part:\n{}'.format (
                ovlp))
            logger.debug (self, 'Linear response SA-CI norms:\n{}'.format (
                linalg.norm (b_ci_blk.T - ci_blk.T @ ovlp, axis=1)))
            if self.ngorb: logger.debug (self, 'Linear response orbital '
                'norms:\n{}'.format (linalg.norm (bvec[:self.ngorb])))
            logger.debug (self, 'SA-SA Lagrange multiplier for root '
                '{}:\n{}'.format (state, x0_sa))
            x0_orb, x0_ci = self.unpack_uniq_var (x0)
            x0_ci[state] = np.dot (x0_sa, ci_blk).reshape (
                self.na_states[state], self.nb_states[state])
            x0 = self.pack_uniq_var (x0_orb, x0_ci)
        r0 = bvec + Aop (x0)
        r0_orb, r0_ci = self.unpack_uniq_var (r0)
        r0_ci_blk = np.asarray ([r0_ci[i].ravel () for i in idx_spin])
        ovlp = ci_blk.conjugate () @ r0_ci_blk.T
        logger.debug (self, 'Lagrange residual SA-SA part after solving SA-SA'
            ' part:\n{}'.format (ovlp))
        logger.debug (self, 'Lagrange residual SA-CI norms after solving SA-SA'
            ' part:\n{}'.format (linalg.norm (r0_ci_blk.T - ci_blk.T @ ovlp,
            axis=1)))
        if self.ngorb: logger.debug (self, 'Lagrange residual orbital norms '
            'after solving SA-SA part:\n{}'.format (linalg.norm (
                r0_orb)))
        x0 += precond (-r0)
        r1_orb, r1_ci = self.unpack_uniq_var (r0)
        r1_ci_blk = np.asarray ([r1_ci[i].ravel () for i in idx_spin])
        ovlp = ci_blk.conjugate () @ r1_ci_blk.T
        logger.debug (self, 'Lagrange residual SA-SA part after first '
            'precondition:\n{}'.format (ovlp))
        logger.debug (self, 'Lagrange residual SA-CI norms after first '
            'precondition:\n{}'.format (linalg.norm (r1_ci_blk.T - ci_blk.T @ ovlp,
            axis=1)))
        if self.ngorb: logger.debug (self, 'Lagrange residual orbital norms '
            'after first precondition:\n{}'.format (linalg.norm (
                r1_orb)))
        return x0

    def kernel (self, **kwargs):
        '''Cache the effective Hamiltonian terms so you don't have to
        calculate them twice
        '''

        state = kwargs["state"] if "state" in kwargs else self.state
        if state is None:
            raise NotImplementedError("Gradient of PDFT state-average energy")

        self.state = state  # Not the best code hygiene maybe
        mo = kwargs["mo"] if "mo" in kwargs else self.base.mo_coeff
        ci = kwargs["ci"] if "ci" in kwargs else self.base.ci
        if isinstance(ci, np.ndarray):
            ci = [ci]  # hack hack hack...

        kwargs["ci"] = ci
        if ("veff1" not in kwargs) or ("veff2" not in kwargs):
            kwargs["veff1"], kwargs["veff2"] = self.base.get_pdft_veff(
                mo, ci, incl_coul=True, paaa_only=True, state=state, drop_mcwfn=True
            )

        if "mf_grad" not in kwargs:
            kwargs["mf_grad"] = self.base.get_rhf_base().nuc_grad_method()

        return super().kernel (**kwargs)

    def project_Aop (self, Aop, ci, state):
        '''Wrap the Aop function to project out redundant degrees of
        freedom for the CI part.  What's redundant changes between
        SA-CASSCF and MC-PDFT so modify this part in child classes.
        '''
        weights, e_mcscf = np.asarray (self.weights), np.asarray (self.e_mcscf)
        try:
            A_sa = 2 * weights[state] * (e_mcscf - e_mcscf[state])
        except IndexError as e:
            assert (self.nroots == 1), e
            A_sa = 0
        except Exception as e:
            print (self.weights, self.e_mcscf)
            raise (e)
        idx_spin = [i for i in range (self.nroots)
                    if self.spin_states[i]==self.spin_states[state]]
        if self.nroots==1:
            ci_blk = ci.reshape (1, -1)
            ci = [ci]
        else:
            ci_blk = np.asarray ([ci[i].ravel () for i in idx_spin])
            A_sa = A_sa[idx_spin]
        def my_Aop (x):
            Ax = Aop (x)
            x_orb, x_ci = self.unpack_uniq_var (x)
            Ax_orb, Ax_ci = self.unpack_uniq_var (Ax)
            for i, j in product (range (self.nroots), repeat=2):
                if self.spin_states[i] != self.spin_states[j]: continue
                Ax_ci[i] -= np.dot (Ax_ci[i].ravel (), ci[j].ravel ()) * ci[j]
            # Add back in the SA rotation part but from the true energy conds
            x_sa = np.dot (ci_blk.conjugate (), x_ci[state].ravel ())
            Ax_ci[state] += np.dot (x_sa * A_sa, ci_blk).reshape (
                Ax_ci[state].shape)
            Ax = self.pack_uniq_var (Ax_orb, Ax_ci)
            return Ax
        return my_Aop
