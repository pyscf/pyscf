# SN2 reaction of Wouter et al
import sys
import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf

from pyscf import molstructures
from pyscf import embcc

import sn2_struct
from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="cc-pVDZ")

parser.add_argument("--local-type", choices=["IAO", "AO", "LAO"], default="IAO")
parser.add_argument("--minao", default="minao")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-size", type=float, nargs=2)

#parser.add_argument("--impurity", nargs="*", default=["F1", "H1", "C1"])
parser.add_argument("--impurity", nargs="*", default=["F1", "H1", "C1", "H2", "C2"])
#parser.add_argument("--impurity", nargs="*", default=["F1", "H1", "C1", "H2", "C2", "H3", "C3"])

#parser.add_argument("--mp2-correction", action="store_true")

parser.add_argument("--ircs", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5, 6, 7, 8])
parser.add_argument("--benchmarks", nargs="*", choices=["MP2", "CISD", "CCSD", "FCI"])
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

dm0 = None
#ref_orbitals = None
refdata = None

for ircidx, irc in enumerate(args.ircs):
    if MPI_rank == 0:
        log.info("IRC=%.2f", irc)

    mol = sn2_struct.structure(irc, args.basis, args.basis)

    mf = pyscf.scf.RHF(mol)
    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)
    assert mf.converged
    dm0 = mf.make_rdm1()

    if args.benchmarks:
        run_benchmarks(mf, args.benchmarks, irc, "benchmarks-"+args.output, ircidx==0)

    else:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                minao=args.minao,
                bath_type=args.bath_type, bath_size=args.bath_size,
                #mp2_correction=args.mp2_correction,
                dmet_bath_tol=1e-9,
                )

        cc.make_atom_cluster(args.impurity)
        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        #if ref_orbitals is not None:
        #    cc.set_reference_orbitals(ref_orbitals)

        cc.set_refdata(refdata)
        cc.run()
        refdata = cc.get_refdata()

        #ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCC  DeltaMP2 EmbCC(dMP2)  FullCC\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_delta_mp2,
                    cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
