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
#parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmark", choices=["CISD", "CCSD", "FCI"])
#parser.add_argument("--tol-bath", type=float, default=1e-3)
#parser.add_argument("--name", default="output")
parser.add_argument("--distances", type=float, nargs="*")
parser.add_argument("--distances-range", type=float, nargs=3, default=[2.6, 5.0, 0.2])
#parser.add_argument("--clusters", default="atoms")
parser.add_argument("--bath-type")
parser.add_argument("--bath-target-size", type=int, nargs=2, default=[0,0])
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])

structure_builder = molstructures.build_H2O_CH6

for idist, dist in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("Distance=%.2f", dist)

    mol = structure_builder(dist, basis=args.basis, verbose=4)

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
            f.write("%3f  %.8e  %.8e\n" % (dist, mf.e_tot, cc.e_tot))

    else:
        if idist == 0:
            cc = embcc.EmbCC(mf, solver=args.solver, bath_type=args.bath_type, bath_target_size=args.bath_target_size)#  tol_bath=args.tol_bath)
            cc.make_atom_clusters()

            cluster = cc.merge_clusters(["O1", "H1", "H2", "C4", "C7"])

            if MPI_rank == 0:
                cc.print_clusters()
        else:
            cc.reset(mf=mf)

        conv = cc.run()
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if idist == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(dMP2)  EmbCCSD(v,dMP2)  EmbCCSD(sC)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (dist, mf.e_tot, cc.e_tot, cc.e_tot_v, cc.e_tot_dmp2, cc.e_tot_v_dmp2, mf.e_tot+cluster.e_corr))
