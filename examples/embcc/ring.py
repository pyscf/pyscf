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

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="sto-6g")
parser.add_argument("--atomtype", default="H")
parser.add_argument("--size", type=int, default=6)
parser.add_argument("--distances", nargs="*", type=float)
parser.add_argument("--distances-range", type=float, nargs=3, default=[0.4, 4.0, 0.2])
parser.add_argument("--dmet-bath-tol", type=float, default=1e-8)
parser.add_argument("--bath-type", default=None)
parser.add_argument("--bath-size", type=float, nargs=2)
parser.add_argument("--maxiter", type=int, default=1)
parser.add_argument("--local-type", default="IAO")
parser.add_argument("-o", "--output", default="energies.txt")
#parser.add_argument("--benchmarks", nargs="*", default=["MP2", "FCI"])
parser.add_argument("--benchmarks", nargs="*")
parser.add_argument("--check-stable", action="store_true")
parser.add_argument("--localize-fragment")
#parser.add_argument("--check-stable", action="store_true")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])
    args.distances = args.distances[::-1]

atoms = args.size*[args.atomtype]
structure_builder = functools.partial(molstructures.build_ring, atoms=atoms)

dm0 = None
refdata = None

for icalc, distance in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("Distance=%.3f", distance)

    mol = structure_builder(distance, basis=args.basis, verbose=4)

    mf = pyscf.scf.RHF(mol)
    mf.kernel(dm0=dm0)
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("HF stable? %r", stable)

    if args.check_stable:
        while not stable:
            dm0 = mf.make_rdm1(mo_stab, mf.mo_occ)
            mf.kernel(dm0)
            mo_stab = mf.stability()[0]
            stable = np.allclose(mo_stab, mf.mo_coeff)
            log.info("HF stable? %r", stable)

    dm0 = mf.make_rdm1()

    if args.benchmarks:
        run_benchmarks(mf, args.benchmarks, distance, "benchmarks.txt", icalc==0,
                cas_size=(args.size, args.size))

    if True:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                dmet_bath_tol=args.dmet_bath_tol,
                bath_type=args.bath_type, bath_size=args.bath_size, maxiter=args.maxiter,
                #solver="FCI"
                solver="CISD",
                #energy_part="first-vir",
                #energy_part="democratic",
                #solver="CCSD"
                localize_fragment=args.localize_fragment,
                )
        #cc.make_all_atom_clusters()
        #cc.make_all_atom_clusters(nelectron_target=1.0)
        #cc.make_all_atom_clusters(solver_options={"fix_spin" : 0.0})

        cc.make_atom_cluster("H1", symmetry_factor=args.size)

        raise SystemExit()
        # Add HF fragment
        #rest = ["H%d" % i for i in range(2, args.size+1)]
        #cc.make_atom_cluster(rest, solver=None)

        #if icalc == 0:
        #    cc.print_clusters()

        #if refdata is not None:
        #    cc.set_refdata(refdata)

        #cc.run()

        #cc.clusters[0].create_orbital_file("dist-%.1f" % distance)

        #refdata = cc.get_refdata()

        if MPI_rank == 0:
            if icalc == 0:
                with open(args.output, "a") as f:
                    f.write("#distance  HF  DMET  EmbCCSD  EmbCCSD(dMP2)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_dmet, cc.e_tot, cc.e_tot + cc.e_delta_mp2))
