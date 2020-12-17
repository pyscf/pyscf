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
parser.add_argument("--solver", default="CCSD(T)")
parser.add_argument("--ncarbon", type=int, default=3)
#parser.add_argument("--c-list", type=int, nargs="*", default=list(range(1, 2)))
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-energy-tols", nargs="*", type=float, default=[1, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, -1])
parser.add_argument("--bath-tols", nargs="*", type=float, default=[1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, -1])
parser.add_argument("--density-fit", action="store_true")
parser.add_argument("--max-memory", type=int)
parser.add_argument("--fragment-size", type=int, default=1, choices=[0, 1, 2, 3])
#parser.add_argument("--bath-energy-tol", type=float, default=-1)

parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)


if args.fragment_size == 0:
    mol = molstructures.build_alkane(args.ncarbon, basis=args.basis, verbose=4, max_memory=args.max_memory)
else:
    mol = molstructures.build_alkane(args.ncarbon, basis=args.basis, numbering="c-units", verbose=4, max_memory=args.max_memory)

mf = pyscf.scf.RHF(mol)
if args.density_fit:
    #mf = mf.density_fit()
    mf = mf.density_fit(auxbasis="cc-pVDZ-ri")
    log.info("Auxiliary basis size: %4d" % mf.with_df.get_naoaux())

t0 = MPI.Wtime()
mf.kernel()
mo_stab = mf.stability()[0]
stable = np.allclose(mo_stab, mf.mo_coeff)
log.info("Time for HF (s): %.3f", (MPI.Wtime()-t0))
assert stable
assert mf.converged


tols = args.bath_tols if args.bath_tols else args.bath_energy_tols

for i, tol in enumerate(tols):

    if args.bath_tols:
        kwargs = {"bath_tol" : tol}
    else:
        kwargs = {"bath_energy_tol" : tol}

    cc = embcc.EmbCC(mf,
            local_type=args.local_type,
            bath_type=args.bath_type,
            bath_energy_tol=tol,
            solver=args.solver,
            **kwargs,
            )

    solver_opts = {}
    #n = (args.ncarbon//2)
    n = 3*(args.ncarbon//2) + 1
    #cc.make_atom_cluster(["C%d" % n, "H%d" % n], solver_options=solver_opts)
    cc.make_atom_cluster(["C%d" % n], solver_options=solver_opts)
    #if args.fragment_size == 0:
    #    cc.make_all_atom_clusters(solver_options=solver_opts)
    #elif args.fragment_size == 1:
    #    for j in range(n):
    #        cc.make_atom_cluster(["C%d" % j, "H%d" % j], solver_options=solver_opts)
    #elif args.fragment_size == 2:
    #    for j in range(0, n, 2):
    #        cc.make_atom_cluster(["C%d" % j, "H%d" % j, "C%d" % (j+1), "H%d" % (j+1)], solver_options=solver_opts)
    #elif args.fragment_size == 3:
    #    for j in range(0, n, 3):
    #        cc.make_atom_cluster(["C%d" % j, "H%d" % j, "C%d" % (j+1), "H%d" % (j+1), "C%d" % (j+2), "H%d" % (j+2)], solver_options=solver_opts)

    cc.kernel()

    nactive = [c.nactive for c in cc.clusters]
    nactive = np.amax(nactive)
    norb = mol.nao_nr()

    if MPI_rank == 0:
        if (i == 0):
            with open(args.output, "a") as f:
                f.write("#TOL  N  HF  CCSD  (T)  CCSD(T)\n")
        with open(args.output, "a") as f:
            fmtstr = "%.3e  %3d  %16.8g  %16.8g  %16.8g  %16.8g\n"
            f.write(fmtstr % (tol, nactive, mf.e_tot, cc.e_corr, cc.e_pert_t, cc.e_corr+cc.e_pert_t))
