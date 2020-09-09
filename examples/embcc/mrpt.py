import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.ci
import pyscf.cc
from pyscf import molstructures
from pyscf import embcc

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="cc-pvdz")
parser.add_argument("--benchmarks", choices=["CISD", "CCSD", "FCI", "CASCI", "CASSCF"], nargs="*")
parser.add_argument("--benchmark-cas", type=int, nargs=2, default=(6,6))
parser.add_argument("--cas-project", action="store_true")
parser.add_argument("--bath-size", type=float, nargs=2, default=[0.0, 0.0])
parser.add_argument("--ircs", type=float, nargs=3, default=[0.8, 3.0, 0.1])
parser.add_argument("--invert-scan", action="store_true")
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="FCI")
parser.add_argument("--output", default="energies.txt")
parser.add_argument("--symmetry", action="store_true")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

ircs = np.arange(args.ircs[0], args.ircs[1]+1e-14, args.ircs[2])
if args.invert_scan:
    ircs = ircs[::-1]

if args.benchmarks:
    structure_builder = functools.partial(molstructures.build_dimer, atoms=["N", "N"], symmetry=True)
else:
    structure_builder = functools.partial(molstructures.build_dimer, atoms=["N", "N"], add_labels=True)

dm0 = None
casci_mo = None
casscf_mo = None
for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    mol = structure_builder(irc, basis=args.basis, verbose=4)
    #mol = structure_builder(irc, basis=args.basis, verbose=4, symmetry=args.symmetry)
    #mol.verbose=6
    mol.max_memory = 100000

    mf = pyscf.scf.RHF(mol)
    mf.kernel(dm0)

    log.info("HF eigenvalues:\n%r", mf.mo_energy)
    mo_stab, _ = mf.stability()
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("HF stable? %r", stable)
    #while (not stable):
    #    dm0 = mf.make_rdm1(mo_stab, mf.mo_occ)
    #    mf.kernel(dm0)
    #    mo_stab, _ = mf.stability()
    #    stable = np.allclose(mo_stab, mf.mo_coeff)
    #    log.info("HF stable? %r", stable)
    #assert stable

    dm0 = mf.make_rdm1()

    mf.analyze()
    #1/0

    if args.benchmarks:

        if "CASCI" in args.benchmarks:
            import pyscf.mcscf

            cas_space = {'A1g' : 1 , 'A1u' : 1, 'E1gx' : 1, "E1gy" : 1, 'E1ux' : 1, "E1uy" : 1}
            core_space = {'A1g' : 2 , 'A1u' : 2}

            casci = pyscf.mcscf.CASCI(mf, *args.benchmark_cas)
            casci.canonicalize=False
            #mo = pyscf.mcscf.sort_mo_by_irrep(casci, mf.mo_coeff, cas_space)
            mo = pyscf.mcscf.sort_mo_by_irrep(casci, mf.mo_coeff, cas_space, cas_irrep_ncore=core_space)
            wf_cas = casci.kernel(mo)[2]

            #if casci_mo is None or not args.cas_project:
            #    casci.kernel()
            #    casci_mo = casci.mo_coeff
            #else:
            #    casci_mo = pyscf.mcscf.project_init_guess(casci, casci_mo)
            #    casci.kernel(casci_mo)

            #casscf = pyscf.mcscf.CASSCF(mf, *args.benchmark_cas)
            #mo = pyscf.mcscf.sort_mo_by_irrep(casscf, mf.mo_coeff, cas_space, cas_irrep_ncore=core_space)
            #casscf.kernel(mo)
            #if casscf_mo is None or not args.cas_project:
            #    casscf.kernel()
            #    casscf_mo = casscf.mo_coeff
            #else:
            #    casscf_mo = pyscf.mcscf.project_init_guess(casscf, casscf_mo)
            #    casscf.kernel(casscf_mo)

            import pyscf.mrpt

            #nevpt2 = pyscf.mrpt.NEVPT(casci)
            #nevpt2.kernel()

            #nevpt2_casscf = pyscf.mrpt.NEVPT(casscf)
            #nevpt2_casscf.kernel()

            # Normal CC
            cc = pyscf.cc.CCSD(mf)
            cc.kernel()

            # Tailored CC
            cisdvec = pyscf.ci.cisd.from_fcivec(wf_cas, 6, 6)
            C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, 6, 3)
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm
            T1 = C1
            T2 = C2 - np.einsum("ia,jb->ijab", C1, C1)

            assert T1.shape == (3,3)
            assert T2.shape == (3,3,3,3)

            occ = np.s_[-3:]
            vir = np.s_[:3]

            def tailorfunc(T1in, T2in):
                T1out = T1in.copy()
                T2out = T2in.copy()
                T1out[occ,vir] = T1
                T2out[occ,occ,vir,vir] = T2
                return T1out, T2out

            #tcc = pyscf.cc.CCSD(mf, mo_coeff=casci.mo_coeff)
            tcc = pyscf.cc.CCSD(mf, mo_coeff=casci.mo_coeff)
            tcc.tailorfunc = tailorfunc
            tcc.kernel()

            with open("casci.txt", "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, tcc.e_tot, casci.e_tot))
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, tcc.e_tot, casci.e_tot, casscf.e_tot, casci.e_tot+nevpt2.e_corr, casscf.e_tot+nevpt2_casscf.e_corr))
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, casci.e_tot, casscf.e_tot, cc.e_tot, cc2.e_tot))

        else:
            run_benchmarks(mf, args.benchmarks, irc, "benchmark-"+args.output, ircidx==0)

    else:

        cc = embcc.EmbCC(mf, bath_size=args.bath_size)
        #cc.make_custom_cluster(ao_labels, symmetry_factor=2, solver=args.solver)
        cc.make_custom_cluster(["N1 2p"], symmetry_factor=1, solver=args.solver)
        cc.make_custom_cluster(["N2 2p"], symmetry_factor=1, solver=args.solver)
        #cc.make_custom_cluster(["1s", "2s", "3s"], solver="CCSD", bath_size=1.0)

        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run()

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2))
