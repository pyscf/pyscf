import sys
import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
from pyscf import molstructures
from pyscf import embcc

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="cc-pVDZ")
parser.add_argument("--solver", default="CCSD")
parser.add_argument("--benchmarks", nargs="*")
#parser.add_argument("--c-list", nargs="*", default=list(range(1, 21)))
parser.add_argument("--ncarbon", type=int, default=3)
parser.add_argument("--maxbath", type=int, default=100)
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-energy-tol", type=float, default=1e-7)
#parser.add_argument("--bath-energy-tol", type=float, default=-1)

parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

for i, n in enumerate(range(20, args.maxbath+1, 2)):

    if MPI_rank == 0:
        log.info("Number of Carbon atoms: %2d", n)
        log.info("==========================")

    mol = molstructures.build_alkane(args.ncarbon, basis=args.basis, verbose=4)

    mf = pyscf.scf.RHF(mol)
    t0 = MPI.Wtime()
    mf.kernel()
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("Time for HF (s): %.3f", (MPI.Wtime()-t0))
    assert stable
    assert mf.converged

    if args.benchmarks:
        #run_benchmarks(mf, args.benchmarks, n, "benchmarks.txt", print_header=(i==0), factor=factor)
        run_benchmarks(mf, args.benchmarks, nelec, "benchmarks.txt", print_header=(i==0))
        continue

    cc = embcc.EmbCC(mf,
            local_type=args.local_type,
            bath_type=args.bath_type,
            #bath_energy_tol=args.bath_energy_tol,
            bath_size=n,
            solver=args.solver,
            )

    solver_opts = {}
    cc.make_all_atom_clusters(solver_options=solver_opts)

    cc.run()

    if MPI_rank == 0:
        if (i == 0):
            with open(args.output, "a") as f:
                f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2\n")
        with open(args.output, "a") as f:
            #f.write(("%2d" + 5*"  %12.8e" + "\n") % (n, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
            #f.write(("%2d  " + 5*"  %16.12e" + "\n") % (nelec, factor*mf.e_tot, factor*cc.e_tot, factor*cc.e_delta_mp2, factor*(cc.e_tot+cc.e_delta_mp2), factor*(mf.e_tot+cc.e_corr_full)))
            f.write(("%2d  " + 4*"  %16.12e" + "\n") % (n, mf.e_tot, cc.e_corr, cc.e_delta_mp2, cc.e_corr+cc.e_delta_mp2))
