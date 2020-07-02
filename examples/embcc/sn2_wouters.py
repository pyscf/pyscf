# SN2 reaction of Wouter et al
import sys
import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

from pyscf import molstructures
from pyscf import embcc

import sn2_struct

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-b", "--basis", default="cc-pVDZ")

parser.add_argument("--local-type", choices=["IAO", "AO", "LAO"], default="IAO")
parser.add_argument("--minao", default="minao")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-size", type=float, nargs=2)

parser.add_argument("--impurity", nargs="*", default=["F1", "H1", "C1"])
#parser.add_argument("--impurity", nargs="*", default=["F1", "H1", "C1", "H2", "C2"])
#parser.add_argument("--impurity", nargs="*", default=["F1", "H1", "C1", "H2", "C2", "H3", "C3"])

parser.add_argument("--mp2-correction", action="store_true")

parser.add_argument("--benchmark", choices=["CISD", "CCSD", "FCI"], nargs="*")
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs


if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

ircs = np.arange(0, 9)

ref_orbitals = None
refdata = None

for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.2f", irc)

    mol = sn2_struct.structure(irc, args.basis, args.basis)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.benchmark:

        if "CCSD" in args.benchmark:
            cc = pyscf.cc.CCSD(mf)
            cc.kernel()
            assert cc.converged

            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

    else:

        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                minao=args.minao,
                bath_type=args.bath_type, bath_size=args.bath_size,
                mp2_correction=args.mp2_correction,
                )

        cc.make_atom_cluster(args.impurity)
        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        cc.set_refdata(refdata)
        cc.run()
        refdata = cc.get_refdata()

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCC  DeltaMP2 EmbCC(dMP2)  FullCC\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_delta_mp2,
                    cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
