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
parser.add_argument("--benchmark", choices=["CISD", "CCSD", "FCI", "CASCI", "CASSCF"])
parser.add_argument("--benchmark-cas", type=int, nargs=2, default=(6,6))
parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--name", default="n2")
parser.add_argument("--ircs", type=float, nargs=3, default=[0.8, 3.0, 0.1])
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
structure_builder = functools.partial(molstructures.build_dimer, atoms=["N", "N"])

for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    #irc = 3.0
    mol = structure_builder(irc, basis=args.basis, verbose=4)
    #mol.verbose=6

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    #mf.analyze()
    #1/0

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
        elif args.benchmark == "CASCI":
            import pyscf.mcscf
            cc = pyscf.mcscf.CASCI(mf, *args.benchmark_cas)
        elif args.benchmark == "CASSCF":
            import pyscf.mcscf
            cc = pyscf.mcscf.CASSCF(mf, *args.benchmark_cas)

        cc.kernel()
        assert cc.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

    else:
        cc = embcc.EmbCC(mf, tol_bath=args.tol_bath)
        #cc.make_atom_clusters()
        #cc.make_custom_cluster("2p", solver="FCI")
        #cc.make_custom_cluster(["2p", "3pz"], solver="FCI")
        #cc.make_custom_cluster(["N1 2p", "N1 3p"], solver="FCI")
        #cc.make_custom_cluster(["N2 2p", "N2 3p"], solver="FCI")
        #cc.make_custom_cluster(["N1 2s", "N1 2p", "N1 3p"], solver="FCI")
        #cc.make_custom_cluster(["N2 2s", "N2 2p", "N2 3p"], solver="FCI")
        #cc.make_custom_cluster(["N1 2s", "N1 2p", "N1 3s", "N1 3p"], solver="FCI")
        #cc.make_custom_cluster(["N2 2s", "N2 2p", "N2 3s", "N2 3p"], solver="FCI")

        #cc.make_custom_cluster(["N1 2s", "N1 2pz", "N1 3s", "N1 3pz"], solver="FCI")
        #cc.make_custom_cluster(["N2 2s", "N2 2pz", "N2 3s", "N2 3pz"], solver="FCI")
        #cc.make_custom_cluster(["N1 2px", "N1 2py", "N1 3px", "N1 3py"], solver="FCI")
        #cc.make_custom_cluster(["N2 2px", "N2 2py", "N2 3px", "N2 3py"], solver="FCI")

        cc.make_ao_clusters(solver="CISD")
        #cc.merge_clusters(["0 N1 1s", "0 N1 2s", "0 N1 3s"])
        #cc.merge_clusters(["1 N2 1s", "1 N2 2s", "1 N2 3s"])

        #cc.make_custom_cluster(["2s", "2pz", "3s", "3pz"], solver="FCI")
        #cc.make_custom_cluster(["2px", "2py", "3px", "3py"], solver="FCI")

        #cc.make_custom_cluster(["N1 2s", "N1 3s", "N1 2pz", "N1 3pz", "N1 3dz^2"], solver="FCI")
        #cc.make_custom_cluster(["N2 2s", "N2 3s", "N2 2pz", "N2 3pz", "N2 3dz^2"], solver="FCI")
        #cc.make_custom_cluster(["N1 2px", "N1 2py", "N1 3px", "N1 3py", "N1 3dyz", "N1 3dxz"], solver="FCI")
        #cc.make_custom_cluster(["N2 2px", "N2 2py", "N2 3px", "N2 3py", "N2 3dyz", "N2 3dxz"], solver="FCI")

        #cc.make_rest_cluster(solver="CCSD")

        if ircidx == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run(max_power=args.max_power)
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(w)  EmbCCSD(z)\n")
                    f.write("#IRC  HF  EmbCCSD\n")
            with open(args.output, "a") as f:
                #f.write("%3f  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, mf.e_tot+cc.e_ccsd_v))
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, mf.e_tot+cc.e_ccsd_v, mf.e_tot+cc.e_ccsd_w, mf.e_tot+cc.e_ccsd_z))
                f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))
