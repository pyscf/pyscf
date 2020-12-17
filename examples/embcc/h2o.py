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

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="aug-cc-pVTZ")
parser.add_argument("--solver", default="CCSD(T)")
parser.add_argument("--benchmarks", nargs="*")
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
#parser.add_argument("--bath-energy-tol", type=float, default=1e-10)
parser.add_argument("--density-fit", action="store_true")
parser.add_argument("--bath-tols", type=float, nargs="*", default=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, -1])
#parser.add_argument("--bath-tols", type=float, nargs="*", default=[-1])
#parser.add_argument("--bath-tols", type=float, nargs="*", default=[1e-6])

parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

mol = pyscf.M(
    #atom = """
    #    O1 0 0      0
    #    H2 0 -2.757 2.587
    #    H3 0  2.757 2.587""",
    atom = "H 0 0 0 ; F 0 0 1.5",
    #atom = "Ne 0 0 0",
    basis = args.basis,
)

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
#assert stable
assert mf.converged
if not stable:
    dm1 = mf.make_rdm1(mo_stab, mf.mo_occ)
    mf.kernel(dm1)
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
assert stable

#if args.benchmarks:
#    #run_benchmarks(mf, args.benchmarks, n, "benchmarks.txt", print_header=(i==0), factor=factor)
#    run_benchmarks(mf, args.benchmarks, nelec, "benchmarks.txt", print_header=(i==0), factor=factor)
cc = pyscf.cc.CCSD(mf)
cc.kernel()
et = cc.ccsd_t()
log.info("Benchmark energies: CCSD = %16.8g, (T) = %16.8g" % (cc.e_corr, et))

for i, tol in enumerate(args.bath_tols):

    cc = embcc.EmbCC(mf,
            local_type=args.local_type,
            bath_type=args.bath_type,
            bath_tol=tol,
            solver=args.solver,
            )

    solver_opts = {}
    cc.make_all_atom_clusters(solver_options=solver_opts)
    #cc.make_atom_cluster("F")
    cc.run()

    nactive = [c.nactive for c in cc.clusters]
    nactive = np.amax(nactive)
    norb = mol.nao_nr()

    if MPI_rank == 0:
        if (i == 0):
            with open(args.output, "a") as f:
                f.write("#tol  N  HF  CCSD  (T)  CCSD(T)\n")
        with open(args.output, "a") as f:
            fmtstr = "%.3e  %3d  %16.8g  %16.8g  %16.8g  %16.8g\n"
            f.write(fmtstr % (tol, nactive, mf.e_tot, cc.e_corr, cc.e_pert_t, cc.e_corr+cc.e_pert_t))
