import logging
import numpy as np
from timeit import default_timer as timer

import pyscf
import pyscf.lo
import pyscf.scf
import pyscf.mp
import pyscf.ci
import pyscf.cc
import pyscf.pbc
import pyscf.pbc.mp
import pyscf.pbc.ci
import pyscf.pbc.cc

from .util import einsum, get_time_string
from . import ao2mo_j3c

log = logging.getLogger(__name__)

class ClusterSolver:

    def __init__(self, cluster, solver, mo_coeff, mo_occ, active, frozen):
        self.cluster = cluster
        self.mf = self.cluster.mf

        self.solver = solver
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.active = active
        self.frozen = frozen

        self.pbc = hasattr(self.mf.mol, "a")
        self.fock = self.cluster.base.get_fock()
        # Intermediates
        self._eris = None
        self._solver = None
        # Standard output values
        self.c1 = None
        self.c2 = None
        self.t1 = None
        self.t2 = None
        self.converged = False
        self.e_corr = 0.0           # Note that this is the full correlation energy
        # Optional output values
        self.dm1 = None
        self.ip_energy = None
        self.ip_coeff = None
        self.ea_energy = None
        self.ea_coeff = None

    def run(self):

        if len(self.active) == 1:
            log.info("Only one orbital in cluster; no correlation energy.")
            self.solver = None

        if self.solver is None:
            pass
        elif self.solver == "MP2":
            self.run_mp2()
        elif self.solver in ("CCSD", "CCSD(T)"):
            self.run_ccsd()
        elif self.solver == "CISD":
            # Currently not maintained
            self.run_cisd()
        elif self.solver in ("FCI-spin0", "FCI-spin1"):
            raise NotImplementedError()
            self.run_fci()
        else:
            raise ValueError("Unknown solver: %s" % self.solver)

        if self.solver in ("CCSD", "CCSD(T)"):
            self.print_t_diagnostic()

        log.debug("Full cluster correlation energy = %.10g htr", self.e_corr)


    def run_mp2(self):
        cls = pyscf.pbc.mp.MP2 if self.pbc else pyscf.mp.MP2
        mp2 = cls(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.frozen)
        self._solver = mp2

        t0 = timer()
        if self.pbc:
            mo_act = self.mo_coeff[:,self.active]
            f_act = np.linalg.multi_dot((mo_act.T, self.fock, mo_act))
            eris = mp2.ao2mo(direct_init=True, mo_energy=np.diag(f_act), fock=fock)
        elif hasattr(mp2, "with_df"):
            eris = mp2.ao2mo(store_eris=True)
        else:
            eris = mp2.ao2mo()
        self._eris = eris
        t = (timer()-t0)
        log.debug("Time for integral transformation [s]: %.3f (%s)", t, get_time_string(t))

        self.e_corr, self.c2 = mp2.kernel(eris=eris, hf_reference=True)
        self.converged = True

    def run_cisd(self):
        # NOT MAINTAINED
        cls = pyscf.pbc.ci.CISD if self.pbc else pyscf.ci.CISD
        ci = cls(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.frozen)
        self._solver = ci

        # Integral transformation
        t0 = timer()
        eris = ci.ao2mo()
        self._eris = eris
        t = (timer()-t0)
        log.debug("Time for integral transformation [s]: %.3f (%s)", t, get_time_string(t))

        t0 = timer()
        log.info("Running CISD...")
        ci.kernel(eris=eris)
        log.info("CISD done. converged: %r", ci.converged)
        t = (timer()-t0)
        log.debug("Time for CISD [s]: %.3f (%s)", t, get_time_string(t))

        self.converged = ci.converged
        self.e_corr = ci.e_corr

        # Renormalize
        c0, c1, c2 = pyscf.ci.cisdvec_to_amplitudes(ci.ci)
        self.c1 = c1/c0
        self.c2 = c2/c0

    def run_ccsd(self):
        cls = pyscf.pbc.cc.CCSD if self.pbc else pyscf.cc.CCSD
        cc = cls(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.frozen)
        self._solver = cc

        # Integral transformation
        t0 = timer()
        if self.pbc:
            mo_act = self.mo_coeff[:,self.active]
            f_act = np.linalg.multi_dot((mo_act.T, self.fock, mo_act))
            if hasattr(self.mf.with_df, "_cderi") and isinstance(self.mf.with_df._cderi, np.ndarray):
                eris = ao2mo_j3c.ao2mo_ccsd(cc, fock=f_act)
            else:
                eris = cc.ao2mo_direct(fock=f_act)
        else:
            eris = cc.ao2mo()
        self._eris = eris
        t = (timer()-t0)
        log.debug("Time for AO->MO: %.3f (%s)", t, get_time_string(t))

        t0 = timer()
        log.info("Running CCSD...")
        cc.kernel(eris=eris)
        log.info("CCSD done. converged: %r", cc.converged)
        t = (timer()-t0)
        log.info("Time for CCSD: %.3f (%s)", t, get_time_string(t))

        self.converged = cc.converged
        self.e_corr = cc.e_corr
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.c1 = cc.t1
        self.c2 = cc.t2 + einsum("ia,jb->ijab", cc.t1, cc.t1)

        if self.cluster.opts.make_rdm1:
            t0 = timer()
            log.info("Making RDM1...")
            self.dm1 = cc.make_rdm1(eris=eris, ao_repr=True)
            log.info("RDM1 done. Lambda converged: %r", cc.converged_lambda)
            if not cc.converged_lambda:
                log.warning("WARNING: Solution of lambda equation not converged!")
            t = (timer()-t0)
            log.info("Time for RDM1: %.3f (%s)", t, get_time_string(t))

        #def eom_ccsd(kind, nroots=3, sort_weight=True, r1_min=0.01):
        def eom_ccsd(kind, nroots=3):
            kind = kind.upper()
            assert kind in ("IP", "EA")
            log.info("Running %s-EOM-CCSD (nroots=%d)...", kind, nroots)
            eom_funcs = {"IP" : cc.ipccsd , "EA" : cc.eaccsd}
            t0 = timer()
            e, c = eom_funcs[kind](nroots=nroots, eris=eris)
            t = (timer()-t0)
            log.info("Time for %s-EOM-CCSD: %.3f (%s)", kind, t, get_time_string(t))
            if nroots == 1:
                e, c = [e], [c]
            return e, c

            #s = self.cluster.base.get_ovlp()
            #lo = self.cluster.base.lo
            #for root in range(nroots):
            #    r1 = c[root][:cc.nocc]
            #    qp = np.linalg.norm(r1)**2
            #    log.info("  %s-EOM-CCSD root= %2d , energy= %+.8g , QP-weight= %.5g", kind, root, e[root], qp)
            #    if qp < 0.0 or qp > 1.0:
            #        log.error("Error: QP-weight not between 0 and 1!")
            #    r1lo = einsum("i,ai,ab,bl->l", r1, eris.mo_coeff[:,:cc.nocc], s, lo)

            #    if sort_weight:
            #        order = np.argsort(-r1lo**2)
            #        for ao, lab in enumerate(np.asarray(self.mf.mol.ao_labels())[order]):
            #            wgt = r1lo[order][ao]**2
            #            if wgt < r1_min*qp:
            #                break
            #            log.info("  * Weight of %s root %2d on OrthAO %-16s = %10.5f", kind, root, lab, wgt)
            #    else:
            #        for ao, lab in enumerate(ao_labels):
            #            wgt = r1lo[ao]**2
            #            if wgt < r1_min*qp:
            #                continue
            #            log.info("  * Weight of %s root %2d on OrthAO %-16s = %10.5f", kind, root, lab, wgt)

            #return e, c

        if self.cluster.opts.ip_eom:
            self.ip_energy, self.ip_coeff = eom_ccsd("IP")

        if self.cluster.opts.ea_eom:
            self.ea_energy, self.ea_coeff = eom_ccsd("EA")

    #def run_fci(self):
    #    nocc_active = len(self.active_occ)
    #    casci = pyscf.mcscf.CASCI(self.mf, self.nactive, 2*nocc_active)
    #    solverobj = casci
    #    # Solver options
    #    casci.verbose = 10
    #    casci.canonicalization = False
    #    #casci.fix_spin_(ss=0)
    #    # TEST SPIN
    #    if solver == "FCI-spin0":
    #        casci.fcisolver = pyscf.fci.direct_spin0.FCISolver(self.mol)
    #    casci.fcisolver.conv_tol = 1e-9
    #    casci.fcisolver.threads = 1
    #    casci.fcisolver.max_cycle = 400
    #    #casci.fcisolver.level_shift = 5e-3

    #    if solver_options:
    #        spin = solver_options.pop("fix_spin", None)
    #        if spin is not None:
    #            log.debug("Setting fix_spin to %r", spin)
    #            casci.fix_spin_(ss=spin)

    #        for key, value in solver_options.items():
    #            log.debug("Setting solver attribute %s to value %r", key, value)
    #            setattr(casci.fcisolver, key, value)

    #    # The sorting of the orbitals above should already have placed the CAS in the correct position

    #    log.debug("Running FCI...")
    #    if self.nelectron_target is None:
    #        e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff)
    #    # Chemical potential loop
    #    else:

    #        S = self.mf.get_ovlp()
    #        px = self.get_local_projector(mo_coeff)
    #        b = np.linalg.multi_dot((S, self.C_local, self.C_local.T, S))

    #        t = np.linalg.multi_dot((S, mo_coeff, px))
    #        h1e = casci.get_hcore()
    #        h1e_func = casci.get_hcore

    #        cptmin = -4
    #        cptmax = 0
    #        #cptmin = -0.5
    #        #cptmax = +0.5

    #        ntol = 1e-6
    #        e_tot = None
    #        wf = None

    #        def electron_error(chempot):
    #            nonlocal e_tot, wf

    #            #casci.get_hcore = lambda *args : h1e - chempot*b
    #            casci.get_hcore = lambda *args : h1e - chempot*(S-b)

    #            e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff, ci0=wf)
    #            #e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff)
    #            dm1xx = np.linalg.multi_dot((t.T, casci.make_rdm1(), t))
    #            nx = np.trace(dm1xx)
    #            nerr = (nx - self.nelectron_target)
    #            log.debug("chempot=%16.8g, electrons=%16.8g, error=%16.8g", chempot, nx, nerr)
    #            assert casci.converged

    #            if abs(nerr) < ntol:
    #                log.debug("Electron error |%e| below tolerance of %e", nerr, ntol)
    #                raise StopIteration

    #            return nerr

    #        try:
    #            scipy.optimize.brentq(electron_error, cptmin, cptmax)
    #        except StopIteration:
    #            pass

    #        # Reset hcore Hamiltonian
    #        casci.get_hcore = h1e_func

    #    #assert np.allclose(mo_coeff_casci, mo_coeff)
    #    #dma, dmb = casci.make_rdm1s()
    #    #log.debug("Alpha: %r", np.diag(dma))
    #    #log.debug("Beta: %r", np.diag(dmb))
    #    log.debug("FCI done. converged: %r", casci.converged)
    #    #log.debug("Shape of WF: %r", list(wf.shape))
    #    cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, 2*nocc_active)
    #    C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc_active)
    #    # Intermediate normalization
    #    log.debug("Weight of reference determinant = %.8e", C0)
    #    renorm = 1/C0
    #    C1 *= renorm
    #    C2 *= renorm

    #    converged = casci.converged
    #    e_corr_full = self.energy_factor*(e_tot - self.mf.e_tot)

    #    # Create fake CISD object
    #    cisd = pyscf.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)

    #    if eris is None:
    #        t0 = MPI.Wtime()
    #        eris = cisd.ao2mo()
    #        log.debug("Time for integral transformation: %s", get_time_string(MPI.Wtime()-t0))

    #    pC1, pC2 = self.get_local_amplitudes(cisd, C1, C2)
    #    e_corr = self.get_local_energy(cisd, pC1, pC2, eris=eris)



    def print_t_diagnostic(self):
        log.info("Diagnostic")
        log.info("**********")
        try:
            dg_t1 = self._solver.get_t1_diagnostic()
            dg_d1 = self._solver.get_d1_diagnostic()
            dg_d2 = self._solver.get_d2_diagnostic()
            log.info("  (T1<0.02: good / D1<0.02: good, D1<0.05: fair / D2<0.15: good, D2<0.18: fair)")
            log.info("  (good: MP2~CCSD~CCSD(T) / fair: use MP2/CCSD with caution)")
            dg_t1_msg = "good" if dg_t1 <= 0.02 else "inadequate!"
            dg_d1_msg = "good" if dg_d1 <= 0.02 else ("fair" if dg_d1 <= 0.05 else "inadequate!")
            dg_d2_msg = "good" if dg_d2 <= 0.15 else ("fair" if dg_d2 <= 0.18 else "inadequate!")
            fmtstr = "  * %2s=%6g (%s)"
            log.info(fmtstr, "T1", dg_t1, dg_t1_msg)
            log.info(fmtstr, "D1", dg_d1, dg_d1_msg)
            log.info(fmtstr, "D2", dg_d2, dg_d2_msg)
            if dg_t1 > 0.02 or dg_d1 > 0.05 or dg_d2 > 0.18:
                log.warning("  WARNING: some diagnostic(s) indicate CCSD may not be adequate.")
        except Exception as e:
            log.error("ERROR in T-diagnostic: %s", e)
