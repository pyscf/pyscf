import sys
import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
#import pyscf.gto
import pyscf.pbc

from pyscf import embcc

#import pyscf.pbc.df as df
#from mpi4pyscf.pbc import df

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--precision", type=float,
        default=1e-4,
        help="Precision for density fitting, determines cell.mesh")

#parser.add_argument("--basis", nargs="*", help="Basis sets: 1) for H2O-N, 2) rest of impurity, 3) Rest of surface",
#        #default=["gth-dzvp", "gth-dzvp", "gth-dzvp"])
#        default=["gth-dzv", "gth-dzvp", "gth-dzvp"])
parser.add_argument("--basis", default="gth-tzvp")
parser.add_argument("--pseudopot", default="gth-pade", help="Pseudo potential.")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")
parser.add_argument("--df", choices=["gaussian", "mixed"], default="gaussian")
parser.add_argument("--xc")
parser.add_argument("--exxdiv", default="ewald")

parser.add_argument("--solver", default="CCSD")
parser.add_argument("--benchmarks", nargs="*")
parser.add_argument("--max-memory", type=int, default=1e5)
parser.add_argument("-o", "--output", default="energies.txt")

parser.add_argument("--bath-type")
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--dmet-bath-tol", type=float, default=1e-8)
parser.add_argument("--bath-tol", type=float, nargs=2)
parser.add_argument("--bath-size", type=int, nargs=2)
parser.add_argument("--bath-relative-size", type=float, nargs=2)
parser.add_argument("--mp2-correction", type=int, nargs=2)
# Load and restore DF integrals


args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.bath_size is None:
    args.bath_size = args.bath_relative_size
    del args.bath_relative_size


if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%20s: %r", name, value)

atom = "Li 0 0 0 ; H 0 0 2 ; Li 0 0 4 ; H 0 0 6  "
a_matrix = 2.0*np.eye(3)
a_matrix[2,2] = 25.0

cell = pyscf.pbc.gto.Cell(
        atom=atom, basis=args.basis,
        a=a_matrix,
        dimension=2,
        pseudo=args.pseudopot,
        precision=args.precision,
        verbose=10)
cell.build()

if args.xc in (None, "HF"):
    mf = pyscf.pbc.scf.RHF(cell)
else:
    import pyscf.pbc.dft
    mf = pyscf.pbc.dft.RKS(cell)
    mf.xc = args.xc

# Exxdiv
if args.exxdiv != "ewald":
    if args.exxdiv == "none":
        args.exxdiv = None
    mf.exxdiv = args.exxdiv

# Density fitting
if args.df == "gaussian":
    mf = mf.density_fit()
    #mf.with_df = df.GDF(cell)
elif args.df == "mixed":
    mf = mf.mix_density_fit()
# Even tempered Gaussian as auxiliary basis [should be default anyway?]
if False:
    mf.with_df.auxbasis = pyscf.pbc.df.aug_etb(cell)

t0 = MPI.Wtime()
mf.kernel()
log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)

t0 = MPI.Wtime()
#f = mf.get_fock()
j, k = mf.get_jk()
log.info("Time for Fock: %.2g", MPI.Wtime()-t0)

dm = mf.make_rdm1()
t0 = MPI.Wtime()
#f = mf.get_fock()
j, k = mf.with_df.get_jk(dm)
log.info("Time for Fock: %.2g", MPI.Wtime()-t0)

assert mf.converged

raise SystemExit()

if args.benchmarks:
    energies = []
    import pyscf.pbc
    for bm in args.benchmarks:
        t0 = MPI.Wtime()
        if bm == "MP2":
            import pyscf.pbc.mp
            mp2 = pyscf.pbc.mp.MP2(mf)
            mp2.kernel()
            energies.append(mf.e_tot + mp2.e_corr)
        elif bm == "CISD":
            import pyscf.pbc.ci
            ci = pyscf.pbc.ci.CISD(mf)
            ci.kernel()
            assert ci.converged
            energies.append(mf.e_tot + ci.e_corr)
        elif bm == "CCSD":
            import pyscf.pbc.cc
            cc = pyscf.pbc.cc.CCSD(mf)
            cc.kernel()
            assert cc.converged
            energies.append(mf.e_tot + cc.e_corr)
        elif bm == "FCI":
            import pyscf.pbc.fci
            fci = pyscf.pbc.fci.FCI(mol, mf.mo_coeff)
            fci.kernel()
            assert fci.converged
            energies.append(mf.e_tot + fci.e_corr)
        log.info("Time for %s: %.2g", bm, MPI.Wtime()-t0)

    if icalc == 0:
        with open(args.output, "w") as f:
            f.write("#distance  HF  " + "  ".join(args.benchmarks) + "\n")
    with open(args.output, "a") as f:
        f.write(("%.3f  %.8e" + (len(args.benchmarks)*"  %.8e") + "\n") % (distance, mf.e_tot, *energies))

#elif not args.no_embcc:
else:
    cc = embcc.EmbCC(mf,
            local_type=args.local_type,
            minao=args.minao,
            solver=args.solver,
            dmet_bath_tol=args.dmet_bath_tol,
            bath_type=args.bath_type,
            bath_size=args.bath_size,
            bath_tol=args.bath_tol,
            mp2_correction=args.mp2_correction)

    #cc.make_atom_cluster(args.impurity_atoms)
    cc.make_atom_cluster(args.impurity_atoms, basis_proj_occ="gth-szv")
    #cc.make_atom_cluster(args.impurity_atoms, basis_proj_occ="gth-dzv", basis_proj_vir="gth-dzv")

    #cc.set_refdata(refdata)
    cc.print_clusters()

    #if icalc == 0 and MPI_rank == 0:
    #    cc.print_clusters_orbitals()

    cc.run()

    #refdata = cc.get_refdata()

    if MPI_rank == 0:
        if icalc == 0:
            with open(args.output, "a") as f:
                f.write("#distance  HF  EmbCC  dMP2  EmbCCSD+dMP2  FullCC\n")
        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
