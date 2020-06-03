import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
from pyscf import molstructures
from pyscf import embcc

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-b", "--basis", default="cc-pVDZ")
parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--atoms", nargs=2, default=["H", "H"])
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmark", choices=["CISD", "CCSD", "FCI"])
parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--name", default="output")
parser.add_argument("--ircs", type=float, nargs=3, default=[0.5, 2.5, 0.1])
parser.add_argument("--clusters", default="atoms")
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

ircs = np.arange(args.ircs[0], args.ircs[1]+1e-14, args.ircs[2])
structure_builder = functools.partial(molstructures.build_dimer, atoms=args.atoms)

for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    mol = structure_builder(irc, basis=args.basis, verbose=4)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.benchmark:
        if args.benchmark == "CISD":
            import pyscf.ci
            cc = pyscf.ci.CISD(mf)
        elif args.benchmark == "CCSD":
            import pyscf.cc
            cc = pyscf.cc.CCSD(mf)
        elif args.benchmark == "FCI":
            import pyscf.fci
            cc = pyscf.fci.FCI(mol, mf.mo_coeff)
        cc.kernel()
        assert cc.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

    else:
        cc = embcc.EmbCC(mf, solver=args.solver, tol_bath=args.tol_bath)
        if args.clusters == "atoms":
            cc.make_atom_clusters()
        elif args.clusters == "aos":
            cc.make_ao_clusters()

        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run(max_power=args.max_power)
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(B)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_tot_alt))
