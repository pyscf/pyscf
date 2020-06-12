import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.mp

from pyscf import molstructures
from pyscf import embcc

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-b", "--basis", default="cc-pVTZ")
#parser.add_argument("-p", "--max-power", type=int, default=0)
#parser.add_argument("--full-ccsd", action="store_true")
#parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--atomtype", default="H")
parser.add_argument("--size", type=int, default=10)
parser.add_argument("--distances", type=float, nargs=3, default=[0.5, 3.0, 0.1])
parser.add_argument("--bath-type")
parser.add_argument("--bath-target-size", type=int, nargs=2, default=[None, None])
parser.add_argument("-o", "--output", default="energies.txt")
parser.add_argument("--benchmark", choices=["MP2", "CCSD"])
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

distances = np.arange(args.distances[0], args.distances[1]+1e-14, args.distances[2])

atoms = args.size*[args.atomtype]
structure_builder = functools.partial(molstructures.build_ring, atoms=atoms)

dm0 = None

for icalc, distance in enumerate(distances):
    if MPI_rank == 0:
        log.info("Distance=%.3f", distance)

    mol = structure_builder(distance, basis=args.basis, verbose=4)

    mf = pyscf.scf.RHF(mol)
    mf.kernel(dm0=dm0)
    mf.stability()
    dm0 = mf.make_rdm1()

    if args.benchmark == "CCSD":
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        assert cc.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot))

    elif args.benchmark == "MP2":
        mp2 = pyscf.mp.MP2(mf)
        mp2.kernel()

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (distance, mf.e_tot, mp2.e_tot))
    else:

        if icalc == 0:
            cc = embcc.EmbCC(mf, bath_type=args.bath_type, bath_target_size=args.bath_target_size)# , tol_bath=args.tol_bath)
            #cc.make_atom_clusters()
            cc.make_custom_atom_cluster(["H1"], symmetry_factor=args.size)
            if MPI_rank == 0:
                cc.print_clusters()
        else:
            cc.reset(mf=mf)

        conv = cc.run()
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if icalc == 0:
                with open(args.output, "a") as f:
                    f.write("#distance  HF  EmbCCSD  EmbCCSD(vir)  EmbCCSD(dMP2)  EmbCCSD(v,dMP2)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_tot_v, cc.e_tot_dmp2, cc.e_tot_v_dmp2))
