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
parser.add_argument("-b", "--basis", default="cc-pVTZ")
parser.add_argument("--benchmark", choices=["CCSD", "FNO-CCSD"])
parser.add_argument("--fno-ccsd-nvir", type=int)
parser.add_argument("--local-type", choices=["AO", "LAO", "IAO"], default="AO")
parser.add_argument("--bath-type")
parser.add_argument("--bath-size", type=float, nargs=2)
parser.add_argument("--distances", type=float, nargs=3, default=[0.7, 3.3, 0.1])
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

args.distances = np.arange(args.distances[0], args.distances[1]+1e-14, args.distances[2])
#structure_builder = molstructures.build_ethanol
structure_builder = molstructures.build_chloroethanol

ref_orbitals = None
for i, dist in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.3f", dist)

    mol = structure_builder(dist, basis=args.basis, verbose=5)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.benchmark:
        if args.benchmark == "CCSD":
            ccsd = pyscf.cc.CCSD(mf)
            ccsd.kernel()
            assert ccsd.converged
        elif args.benchmark == "FNO-CCSD":
            ccsd = pyscf.cc.FNOCCSD(mf, nvir_act=args.fno_ccsd_nvir)
            ccsd.kernel()

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (dist, mf.e_tot, ccsd.e_tot))

    #if not args.no_embccsd:
    else:
        cc = embcc.EmbCC(mf, local_type=args.local_type, bath_type=args.bath_type, bath_size=args.bath_size)
        #cc.make_all_atom_clusters()
        #cc.make_atom_cluster(["O1", "H3"])
        cc.make_atom_cluster(["O4", "H9"])

        if i == 0:
            cc.print_clusters()

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        cc.run()

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if i == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(vir)  EmbCCSD(dem)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (dist, mf.e_tot, cc.e_tot, mf.e_tot+cc.e_corr_v, mf.e_tot+cc.e_corr_d))
