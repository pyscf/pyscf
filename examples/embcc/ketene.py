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

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-b", "--basis", default="cc-pVDZ")
parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--full-ccsd", action="store_true")
parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--bath-type")
parser.add_argument("--name", default="ketene")
parser.add_argument("--ircs", type=float, nargs=3, default=[0.6, 2.3, 0.1])
parser.add_argument("--output")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.output is None:
    args.output = args.name + ".out"

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

#ircs = np.arange(0.6, 3.3+1e-14, 0.1)
ircs = np.arange(args.ircs[0], args.ircs[1]+1e-14, args.ircs[2])
structure_builder = molstructures.build_ketene

for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    mol = structure_builder(irc, basis=args.basis, verbose=5)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.full_ccsd:
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        assert cc.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

    else:
        cc = embcc.EmbCC(mf, bath_type=args.bath_type, tol_bath=args.tol_bath)
        #ecc.create_custom_clusters([("O1", "H3")])
        cc.make_atom_clusters()
        #oh_cluster = cc.merge_clusters(("O1", "H3"))
        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run(max_power=args.max_power)
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(w)  EmbCCSD(z)\n")
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(alt)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_tot_alt))
