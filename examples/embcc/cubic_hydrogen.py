import sys
import logging
import argparse
import itertools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.pbc
from pyscf import embcc

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="gth-dzv")
parser.add_argument("--solver", default="CCSD")
parser.add_argument("--benchmarks", nargs="*")
#parser.add_argument("--c-list", nargs="*", default=list(range(1, 21)))
#parser.add_argument("--maxbath", type=int, default=100)
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")
parser.add_argument("--dmet-bath-tol", type=float, default=0.05)
parser.add_argument("--bath-energy-tol", type=float, default=1e-7)
# System
parser.add_argument("--bond-length", type=float, default=0.8)
parser.add_argument("--supercell", type=int, nargs=3, default=None)
parser.add_argument("--k-points", type=int, nargs=3)
#parser.add_argument("--bath-energy-tol", type=float, default=-1)

parser.add_argument("--precision", type=float,
        #default=1e-6,
        default=1e-5,
        help="Precision for density fitting, determines cell.mesh")

parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

#a_eq = args.bond_length
#a_step = 2.0*a_eq/100.0

n = 2
#a_list = a_eq + a_step*np.arange(-n, n+1)
#a_list = [a_eq]
a_list = [2.0]

for i, a in enumerate(a_list):

    if MPI_rank == 0:
        log.info("Lattice constant= %.3f", a)
        log.info("=======================")

    atom = "H 0 0 0 ; H 0 0 1"
    amat = a*np.eye(3)
    cell = pyscf.pbc.gto.M(atom=atom, a=amat, basis=args.basis,
            precision=args.precision, verbose=10)
    if args.supercell:
        from pyscf.pbc import tools
        cell = tools.pbc.super_cell(cell, args.supercell)

    if args.k_points is None:
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        kpts = cell.make_kpts(args.k_points)
        mf = pyscf.pbc.scf.KRHF(cell, kpts)
    #mf.exxdiv = None

    mf = mf.density_fit()
    t0 = MPI.Wtime()
    mf.kernel()
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("Time for HF [s]: %.3f", (MPI.Wtime()-t0))
    assert stable
    assert mf.converged

    ncells = np.product(args.supercell)
    #if args.supercell:
    #    enuc = cell.energy_nuc()
    #    nsc = np.product(args.supercell)
    #    emf = (mf.e_tot-enuc)/nsc + enuc
    #else:
    #    emf = mf.e_tot
    #emf = mf.e_tot

    #if args.benchmarks:
    #    run_benchmarks(mf, args.benchmarks, a, "benchmarks.txt", print_header=(i==0), factor=efac, pbc=True, k_points=(args.k_points is not None))
    #    continue
    #from pyscf import mp2
    from pyscf.pbc import mp
    if args.k_points:
        mp2 = mp.KMP2(mf)
    else:
        mp2 = mp.MP2(mf)
    mp2.kernel()

    if args.k_points is not None:
        from pyscf.pbc.tools import k2gamma
        mf_gamma = k2gamma.k2gamma(mf, args.k_points)
        #mf_gamma.e_tot = mf.e_tot
        #mf_gamma.e_tot = mf.e_tot
        mf_gamma.converged = mf.converged
        mf = mf_gamma
        #mf.mol.verbose=10
        mf = mf.density_fit()

    mp2_sc = mp.MP2(mf)
    mp2_sc.kernel()
    if args.k_points:
        nks = np.product(args.k_points)
    else:
        nks = 1

    with open("energies.txt", "a") as f:
        f.write("%.8g  %.8g  %.8g\n" % (mf.e_tot/ncells, mp2.e_corr/ncells, mp2_sc.e_corr/nks))

    1/0


    implabel = "H000"
    print(mf.mol.atom)
    mf.mol.atom[0] = (implabel, mf.mol.atom[0][1])
    mf.mol.build(False, False)
    print(mf.mol.atom)

    cc = embcc.EmbCC(mf,
            #local_type=args.local_type,
            minao=args.minao,
            dmet_bath_tol=args.dmet_bath_tol,
            #bath_type=args.bath_type,
            bath_energy_tol=args.bath_energy_tol,
            #bath_size=n,
            solver=args.solver,
            )

    solver_opts = {}
    if args.supercell:
        symfac = np.product(args.supercell)
    else:
        symfac = 1.0
    if args.use_pbc is not None:
        cc.make_atom_cluster(implabel, symmetry_factor=symfac, solver_options=solver_opts)
    else:
        cc.make_atom_cluster(implabel, symmetry_factor=symfac, solver_options=solver_opts)
    cc.run()

    if MPI_rank == 0:
        if (i == 0):
            with open(args.output, "a") as f:
                f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2\n")
        with open(args.output, "a") as f:
            f.write(("%.5f  " + 4*"  %16.12e" + "\n") % (a, mf.e_tot, cc.e_corr, cc.e_delta_mp2, cc.e_corr+cc.e_delta_mp2))
