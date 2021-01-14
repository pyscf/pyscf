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
parser.add_argument("--basis", default="cc-pVDZ")
parser.add_argument("--full-ccsd", action="store_true")
#parser.add_argument("--bath-tol", type=float, default=1e-8)
parser.add_argument("--bath-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
parser.add_argument("--dmet-bath-tol", type=float, default=0.05)
parser.add_argument("--ncarbon", type=int, default=1)
parser.add_argument("--ncluster", type=int, default=1)
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

ircs = np.arange(0, 9)

#cluster = ["F1", "H1", "C1"]
#main_cluster = ["F1", "H1", "C1"]

clusters = []
for i in range(1, 13, args.ncarbon):
    c = []
    for j in range(0, args.ncarbon):
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

        with open("energies.txt", "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

    else:

        energies_ccsd = []
        energies_ccsd_dmp2 = []
        energies_full = []

        for tol in args.bath_tols:
            cc = embcc.EmbCC(mf,
                    dmet_bath_tol=args.dmet_bath_tol,
                    bath_tol=tol)
            #cc.add_custom_cluster(main_cluster, name=main_cluster_name)
            #cc.add_custom_cluster(clusters[0], name=main_cluster_name)
            #cc.make_custom_atom_cluster(clusters[0])
            #for cluster in clusters[1:]:
            #    cc.make_custom_atom_cluster(cluster)

            kwargs = {"bath_tol_per_electron" : False}
            for cluster in clusters[:args.ncluster]:
                cc.make_atom_cluster(cluster, **kwargs)

            #cc.create_atom_clusters()
            #cc.merge_clusters(cluster)
            cc.run()

            energies_ccsd.append(cc.e_tot)
            energies_ccsd_dmp2.append(cc.e_tot + cc.e_delta_mp2)
            energies_full.append(mf.e_tot + cc.e_corr_full)

        files = ["ccsd.txt", "ccsd-dmp2.txt", "ccsd-full.txt"]
        if MPI_rank == 0:
            if ircidx == 0:
                title = "#IRC  HF  " + "  ".join([("bath-%e" % t) for t in args.bath_tols]) + "\n"
                for fname in files:
                    with open(fname, "a") as f:
                        #f.write("#IRC  HF  EmbCCSD  EmbCCSD(alt)  EmbCCSD(1C)  EmbCCSD(1C,alt)  EmbCCSD(1C,f)\n")
                        f.write(title)
            fmtstr = ((len(energies_ccsd)+2) * "  %+16.12e") + "\n"
            with open(files[0], "a") as f:
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_tot_alt, mf.e_tot+cc.clusters[0].e_corr, mf.e_tot+cc.clusters[0].e_corr_alt, mf.e_tot+cc.clusters[0].e_corr_full))
                f.write(fmtstr % (irc, mf.e_tot, *energies_ccsd))
            with open(files[1], "a") as f:
                f.write(fmtstr % (irc, mf.e_tot, *energies_ccsd_dmp2))
            with open(files[2], "a") as f:
                f.write(fmtstr % (irc, mf.e_tot, *energies_full))

