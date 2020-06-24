import sys
import logging
import argparse

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
parser.add_argument("-b", "--basis", default="cc-pVDZ")
#parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--benchmark", choices=["MP2", "CCSD"])
#parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--bath-type")
parser.add_argument("--bath-tol", type=float, nargs=2, default=[1e-3, 1e-3])
parser.add_argument("--bath-size", type=float, nargs=2, default=[None, None])
parser.add_argument("--distances", type=float, nargs="*")
parser.add_argument("--distances-arange", type=float, nargs=3, default=[0.6, 2.3, 0.1])
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.distances is None:
    #args.distances = np.arange(*(np.asarray(args.distances_arange)+[0,1e-12,0]))
    args.distances = np.arange(args.distances_arange[0], args.distances_arange[1]+1e-12, args.distances_arange[2])

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

structure_builder = molstructures.build_ketene

ref_orbitals = None
for i, dist in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.3f", dist)

    mol = structure_builder(dist, basis=args.basis, verbose=5)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.benchmark == "CCSD":
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        assert cc.converged
        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (dist, mf.e_tot, cc.e_tot))
    elif args.benchmark == "MP2":
        mp2 = pyscf.mp.MP2(mf)
        mp2.kernel()
        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (dist, mf.e_tot, mp2.e_tot))

    else:
        #if ircidx == 0:
        #    cc = embcc.EmbCC(mf, bath_type=args.bath_type, bath_target_size=args.bath_target_size) #, tol_bath=args.tol_bath)
        #    #ecc.create_custom_clusters([("O1", "H3")])
        #    cc.make_atom_clusters()
        #    #oh_cluster = cc.merge_clusters(("O1", "H3"))
        #    if MPI_rank == 0:
        #        cc.print_clusters()
        #else:
        #    cc.reset(mf=mf)
        cc = embcc.EmbCC(mf, bath_type=args.bath_type, bath_size=args.bath_size, bath_tol=args.bath_tol)
        cc.make_all_atom_clusters()

        if i == 0:
            cc.print_clusters()

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        cc.run()

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if i == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(dMP2)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e\n" % (dist, mf.e_tot, cc.e_tot, cc.e_tot_dmp2))
