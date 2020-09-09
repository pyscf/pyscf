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
parser.add_argument("-b", "--basis", default="aug-cc-pVDZ")
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmarks", nargs="*", choices=["MP2", "CISD", "CCSD", "FCI"])
#parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--distances", type=float, nargs="*",
        default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.6, 3.0, 4.0, 5.0])
parser.add_argument("--distances-range", type=float, nargs=3, default=[0.6, 5.0, 0.1])
parser.add_argument("--local-type", choices=["IAO", "AO", "LAO"], default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-size", type=float, nargs=2)
parser.add_argument("--impurity", nargs="*",
        #default=["O*", "H*"]
        default=["O*", "H*", "C2"]
        )
#parser.add_argument("--mp2-correction", action="store_true")
#parser.add_argument("--use-ref-orbitals-bath", type=int, default=0)
parser.add_argument("--minao", default="minao")

#parser.add_argument("--counterpoise", choices=["none", "water", "water-full", "borazine", "borazine-full"])
parser.add_argument("--molecule", choices=["ethanol", "ethenol"])
parser.add_argument("--fragment", choices=["all", 1, 2], default="all")
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

#args.use_ref_orbitals_bath = bool(args.use_ref_orbitals_bath)

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])
del args.distances_range

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

if args.molecule == "ethanol":
    structure_builder = molstructures.build_ethanol
elif args.molecule == "ethenol":
    structure_builder = molstructures.build_ethenol

dm0 = None
ref_orbitals = None
refdata = None

for idist, dist in enumerate(args.distances):

    if MPI_rank == 0:
        log.info("Distance=%.2f", dist)
        log.info("=============")

    #mol = structure_builder(dist, counterpoise=args.counterpoise, basis=args.basis, verbose=4)
    mol = structure_builder(dist, basis=args.basis, verbose=4)

    #if args.fragment != "all":
    #    frags = mol.make_counterpoise_fragments([["O*", "H*"]])
    #    if args.fragment == "water":
    #        mol = water
    #    else:
    #        mol = boronene

    mf = pyscf.scf.RHF(mol)
    #mf = mf.density_fit()
    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)
    assert mf.converged
    dm0 = mf.make_rdm1()

    if args.benchmarks:
        energies = []
        for bm in args.benchmarks:
            t0 = MPI.Wtime()
            if bm == "MP2":
                import pyscf.mp
                mp2 = pyscf.mp.MP2(mf)
                mp2.kernel()
                energies.append(mf.e_tot + mp2.e_corr)
            elif bm == "CISD":
                import pyscf.ci
                ci = pyscf.ci.CISD(mf)
                ci.kernel()
                assert ci.converged
                energies.append(mf.e_tot + ci.e_corr)
            elif bm == "CCSD":
                import pyscf.cc
                cc = pyscf.cc.CCSD(mf)
                cc.kernel()
                assert cc.converged
                energies.append(mf.e_tot + cc.e_corr)
            elif bm == "FCI":
                import pyscf.fci
                fci = pyscf.fci.FCI(mol, mf.mo_coeff)
                fci.kernel()
                assert fci.converged
                energies.append(mf.e_tot + fci.e_corr)
            log.info("Time for %s: %.2g", bm, MPI.Wtime()-t0)

        if idist == 0:
            with open(args.output, "w") as f:
                f.write("#distance  HF  " + "  ".join(args.benchmarks) + "\n")
        with open(args.output, "a") as f:
            f.write(("%.3f  %.8e" + (len(args.benchmarks)*"  %.8e") + "\n") % (dist, mf.e_tot, *energies))

    else:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                minao=args.minao,
                bath_type=args.bath_type, bath_size=args.bath_size,
                #mp2_correction=args.mp2_correction,
                #use_ref_orbitals_bath=args.use_ref_orbitals_bath,
                )
        cc.make_atom_cluster(args.impurity)
        if idist == 0 and MPI_rank == 0:
            cc.print_clusters()

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        cc.set_refdata(refdata)
        cc.run()
        refdata = cc.get_refdata()

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if idist == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCC  EmbCC(vir)  EmbCC(dem)  EmbCC(dMP2)  EmbCC(vir,dMP2)  Embcc(dem,dMP2)\n")
                    f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2  EmbCC(full)\n")
            with open(args.output, "a") as f:
                f.write(("%3f" + 5*"  %12.8e" + "\n") % (dist, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
