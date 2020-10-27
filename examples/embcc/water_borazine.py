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

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
#parser.add_argument("--basis", default="borazine", choices=["borazine", "boronene"])
#parser.add_argument("--basis", default="cc-pVDZ")
parser.add_argument("--basis", nargs="*", default=["cc-pVDZ", "cc-pVDZ"])
#parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmarks", nargs="*")
#parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--distances", type=float, nargs="*",
        default=[2.8, 2.9, 3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])
parser.add_argument("--distances-range", type=float, nargs=3, default=[2.8, 8.0, 0.2])
parser.add_argument("--local-type", choices=["IAO", "AO", "LAO"], default="IAO")

parser.add_argument("--solver", default="CCSD")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-size", type=int, nargs=2)
parser.add_argument("--bath-tol", type=float, nargs=2)
parser.add_argument("--dmet-bath-tol", type=float, default=1e-8)

parser.add_argument("--impurity", nargs="*")
parser.add_argument("--impurity-number", type=int, default=1)
#parser.add_argument("--impurity", nargs="*", default=["O1", "H1", "H2", "N1", "B1", "B3"])
#parser.add_argument("--mp2-correction", action="store_true")
#parser.add_argument("--use-ref-orbitals-bath", type=int, default=0)
parser.add_argument("--minao", default="minao")
parser.add_argument("--mp2-correction", nargs=2, type=int, default=[1, 1])

#parser.add_argument("--counterpoise", choices=["none", "water", "water-full", "borazine", "borazine-full"])
#parser.add_argument("--fragment", choices=["all", "water", "borazine"], default="all")
parser.add_argument("--fragment", choices=["all", "water", "surface"], default="all")
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

#args.use_ref_orbitals_bath = bool(args.use_ref_orbitals_bath)

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])
del args.distances_range

if len(args.basis) == 1:
    args.basis = 2*args.basis

if args.impurity is None:
    args.impurity = ["O1", "H1", "H2"]
    if args.impurity_number >= 1:
        args.impurity += ["N1"]
        # TEST
        #args.impurity += ["H4"]
    if args.impurity_number >= 2:
        args.impurity += ["B1", "B3"]
    if args.impurity_number >= 3:
        args.impurity += ["N2", "N3"]
        #raise NotImplementedError()
#args.impurity = ["O1", "H1", "H2", "N1", "N2", "N3", "B1", "B2", "B3", "H3", "H4", "H5", "H6", "H7", "H8"]

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

structure_builder = molstructures.build_water_borazine

dm0 = None
refdata = None

basis_dict = {
     #"O1" : args.basis[0],
     "H1" : args.basis[0],
     #"H2" : args.basis[0],
     "N1" : args.basis[0],
     "default" : args.basis[1]
     }

for idist, dist in enumerate(args.distances):

    if MPI_rank == 0:
        log.info("Distance=%.2f", dist)
        log.info("=============")

    #mol = structure_builder(dist, counterpoise=args.counterpoise, basis=args.basis, verbose=4)
    #mol = structure_builder(dist, basis=args.basis, verbose=4)
    mol = structure_builder(dist, basis=basis_dict, verbose=4)

    if args.fragment != "all":
        water, borazine = mol.make_counterpoise_fragments([["O1", "H1", "H2"]])
        if args.fragment == "water":
            mol = water
        else:
            mol = borazine

    mf = pyscf.scf.RHF(mol)
    #mf = mf.density_fit()
    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.2g", (MPI.Wtime()-t0))
    assert mf.converged
    dm0 = mf.make_rdm1()

    if args.benchmarks:
        run_benchmarks(mf, args.benchmarks, dist, "benchmarks.txt", print_header=(idist==0))

    else:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                solver=args.solver,
                minao=args.minao,
                dmet_bath_tol=args.dmet_bath_tol,
                bath_type=args.bath_type, bath_size=args.bath_size, bath_tol=args.bath_tol,
                mp2_correction=args.mp2_correction,
                #use_ref_orbitals_bath=args.use_ref_orbitals_bath,
                )
        cc.make_atom_cluster(args.impurity)

        if idist == 0 and MPI_rank == 0:
            cc.print_clusters()

        #cc.set_refdata(refdata)
        cc.run()
        #refdata = cc.get_refdata()

        if MPI_rank == 0:
            if idist == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCC  EmbCC(vir)  EmbCC(dem)  EmbCC(dMP2)  EmbCC(vir,dMP2)  Embcc(dem,dMP2)\n")
                    f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2  EmbCC(full)\n")
            with open(args.output, "a") as f:
                f.write(("%3f" + 5*"  %12.8e" + "\n") % (dist, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
