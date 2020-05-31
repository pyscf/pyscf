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
parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--full-ccsd", action="store_true")
parser.add_argument("--tol-bath", type=float, default=1e-8)
parser.add_argument("--name", default="sn2")
parser.add_argument("--C-per-cluster", type=int, default=1)
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
ircs = np.arange(0, 9)

#cluster = ["F1", "H1", "C1"]
#main_cluster = ["F1", "H1", "C1"]

clusters = []
for i in range(1, 13, args.C_per_cluster):
    c = []
    for j in range(0, args.C_per_cluster):
        if (i+j) == 1:
            c.append("F%d" % (i+j))
        c.append("C%d" % (i+j))
        c.append("H%d" % (i+j))
    clusters.append(c)

# Main cluster with F2
#main_cluster_name="F2" + args.C-per-cluster*"-CH2"
#clusters[0].append("F1")

for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    #mol = structure_builder(irc, basis=args.basis, verbose=0)
    mol = sn2_struct.structure(irc, args.basis, args.basis)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.full_ccsd:
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        assert cc.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

    else:
        cc = embcc.EmbCC(mf, tol_bath=args.tol_bath)
        #cc.add_custom_cluster(main_cluster, name=main_cluster_name)
        #cc.add_custom_cluster(clusters[0], name=main_cluster_name)
        cc.add_custom_cluster(clusters[0])
        for cluster in clusters[1:]:
            cc.add_custom_cluster(cluster)

        #cc.create_atom_clusters()
        #cc.merge_clusters(cluster)

        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run(max_power=args.max_power)
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(1C)  EmbCCSD(1C,v)  EmbCCSD(1C,f)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, mf.e_tot+cc.e_ccsd_v,
                    mf.e_tot+cc.clusters[0].e_ccsd, mf.e_tot+cc.clusters[0].e_ccsd_v, mf.e_tot+cc.clusters[0].e_cl_ccsd))
