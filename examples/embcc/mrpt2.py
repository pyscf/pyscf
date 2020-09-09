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
# General
parser.add_argument("--molecule", choices=["HF", "N2", "Cr2"], required=True)
parser.add_argument("--basis", default="cc-pVDZ")
#parser.add_argument("--ircs", type=float, nargs=3, default=[0.8, 3.0, 0.1])
#parser.add_argument("--ircs", type=float, nargs=3)
parser.add_argument("--output", default="energies.txt")
parser.add_argument("--invert-scan", action="store_true")

# Benchmarks
#parser.add_argument("--benchmarks", choices=["CISD", "CCSD", "FCI", "CASCI", "CASSCF", "AVAS-CASCI", "AVAS-CASSCF", "NEVPT2@CASCI", "NEVPT2@CASSCF"], nargs="*")
parser.add_argument("--benchmarks", nargs="*")
#parser.add_argument("--benchmark-cas", type=int, nargs=2, default=(6,6))
#parser.add_argument("--cas-project", action="store_true")
parser.add_argument("--cas-size", type=int, nargs=2)
parser.add_argument("--symmetry", action="store_true")

# EmbCC
parser.add_argument("--bath-size", type=float, nargs=2, default=[0.0, 0.0])
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="FCI")

args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

#ircs = np.arange(args.ircs[0], args.ircs[1]+1e-14, args.ircs[2])
if args.invert_scan:
    ircs = ircs[::-1]

if args.benchmarks:
    symmetry = True
    add_labels = False
else:
    symmetry = False
    add_labels = True

EPS = 1e-14

if args.molecule == "HF":
    atoms = ["F", "H"]
    ircs = np.arange(0.6, 4.0+EPS, 0.1)
    if args.cas_size in (None, [2, 2]):
        cas_size = (2, 2)
        cas_space = {"A1" : 2}
        core_space = {"A1" : 2, "E1x" : 1, "E1y" : 1}
    elif args.cas_size == [8, 5]:
        cas_size = args.cas_size
        cas_space = {"A1" : 3, "E1x" : 1, "E1y" : 1}
        core_space = {"A1" : 1}
    else:
        raise ValueError("CAS size: %r", args.cas_size)

elif args.molecule == "N2":
    atoms = ["N", "N"]
    ircs = np.arange(0.8, 3.0+EPS, 0.1)
    cas_size = (6, 6)
    cas_space = {'A1g' : 1 , 'A1u' : 1, 'E1gx' : 1, "E1gy" : 1, 'E1ux' : 1, "E1uy" : 1}
    core_space = {'A1g' : 2 , 'A1u' : 2}
    avas_ao_labels = ["2p"]
elif args.molecule == "Cr2":
    atoms = ["Cr", "Cr"]
    ircs = np.arange(1.5, 4.0+EPS, 0.1)
    if args.cas_size in (None, [12, 12]):
        cas_size = (12, 12)
        core_space = {'A1g':5, 'A1u':5}
        cas_space = {'A1g':2, 'A1u':2,
                'E1ux':1, 'E1uy':1, 'E1gx':1, 'E1gy':1,
                'E2ux':1, 'E2uy':1, 'E2gx':1, 'E2gy':1}

#structure_builder = functools.partial(molstructures.build_dimer, atoms=atoms, symmetry=symmetry, add_labels=add_labels)
structure_builder = functools.partial(molstructures.build_dimer, atoms=atoms, symmetry=args.symmetry, add_labels=add_labels)

dm0 = None
casci_mo = None
casscf_mo = None
for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    mol = structure_builder(irc, basis=args.basis, verbose=4)
    mol.max_memory = 50000

    mf = pyscf.scf.RHF(mol)
    mf.max_cycle = 100
    #mf.level_shift = 0.4
    mf.kernel(dm0)

    log.info("HF eigenvalues:\n%r", mf.mo_energy)
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("HF stable? %r", stable)

    #if dm0 is None and not stable:
    if False:
        while not stable:
            dm0 = mf.make_rdm1(mo_stab, mf.mo_occ)
            mf.kernel(dm0)
            mo_stab = mf.stability()[0]
            stable = np.allclose(mo_stab, mf.mo_coeff)
            log.info("HF stable? %r", stable)

    dm0 = mf.make_rdm1()
    mf.analyze()

    # TEST AVAS
    if False:
        from pyscf.mcscf import avas
        ao_labels = ['F1 2pz', 'H2 1s']
        norb, ne_act, orbs = avas.avas(mf, ao_labels, canonicalize=False)
        print(norb, ne_act)

        casci = pyscf.mcscf.CASCI(mf, norb, ne_act)
        casci.kernel(orbs)

        casscf = pyscf.mcscf.CASSCF(mf, norb, ne_act)
        casscf.kernel(orbs)

        with open(args.output, "a") as f:
            f.write("%.3f  %.8g  %.8g  %.8g\n" % (irc, mf.e_tot, casci.e_tot, casscf.e_tot))
        continue

    #with open("mf.txt", "a") as f:
    #    f.write("%.3f  %.8f\n" % (irc, mf.e_tot))
    #continue

    if args.benchmarks:

        run_benchmarks(mf, args.benchmarks, irc, args.output, print_header=(ircidx==0),
                cas_size=cas_size, cas_space=cas_space, core_space=core_space,
                avas_ao_labels=avas_ao_labels)

        #    # Tailored CC
        #    cisdvec = pyscf.ci.cisd.from_fcivec(wf_cas, 6, 6)
        #    C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, 6, 3)
        #    renorm = 1/C0
        #    C1 *= renorm
        #    C2 *= renorm
        #    T1 = C1
        #    T2 = C2 - np.einsum("ia,jb->ijab", C1, C1)

        #    assert T1.shape == (3,3)
        #    assert T2.shape == (3,3,3,3)

        #    occ = np.s_[-3:]
        #    vir = np.s_[:3]

        #    def tailorfunc(T1in, T2in):
        #        T1out = T1in.copy()
        #        T2out = T2in.copy()
        #        T1out[occ,vir] = T1
        #        T2out[occ,occ,vir,vir] = T2
        #        return T1out, T2out

        #    #tcc = pyscf.cc.CCSD(mf, mo_coeff=casci.mo_coeff)
        #    tcc = pyscf.cc.CCSD(mf, mo_coeff=casci.mo_coeff)
        #    tcc.tailorfunc = tailorfunc
        #    tcc.kernel()

    else:

        cc = embcc.EmbCC(mf, bath_size=args.bath_size)

        if args.molecule == "HF":
            #cc.make_custom_cluster(["F1 2pz"], solver=args.solver)
            cc.make_custom_cluster(["F1 2pz"], solver=args.solver, solver_options={"fix_spin" : 0})

            #cc.make_custom_cluster(["F1 2s", "F1 2p"], solver=args.solver)
            cc.make_custom_cluster(["H2 1s"], solver=args.solver, solver_options={"fix_spin" : 0})
            cc.make_custom_cluster(["F1 \ds", "F1 2px", "F1 2py"], solver="CCSD", bath_size=1.0)
            #cc.make_custom_cluster(["F1 1s"], solver=args.solver)
            #cc.make_custom_cluster(["F1 2s"], solver=args.solver)
            #cc.make_custom_cluster(["F1 2px"], solver=args.solver)
            #cc.make_custom_cluster(["F1 2py"], solver=args.solver)

        elif args.molecule == "N2":
            solver_opts = {"fix_spin" : 0}
            #solver_opts = None
            #cc.make_custom_cluster(["N1 2p"], symmetry_factor=2, solver=args.solver, solver_options=solver_opts)
            cc.make_custom_cluster(["N1 2p"], solver=args.solver, solver_options=solver_opts)
            cc.make_custom_cluster(["N2 2p"], solver=args.solver, solver_options=solver_opts)
            #cc.make_custom_cluster(["N2 2p"], symmetry_factor=1, solver=args.solver)
            cc.make_custom_cluster(["1s", "2s"], solver="CCSD", bath_size=1.0, coupled_bath=True)

        elif args.molecule == "Cr2":
            cc.make_custom_cluster(["Cr1 4s", "Cr1 3d"], symmetry_factor=2, solver=args.solver)

        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run()

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2))
