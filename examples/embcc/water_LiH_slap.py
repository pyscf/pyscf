import sys
import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
#import pyscf.gto
import pyscf.pbc


#import pyscf.pbc.df
from pyscf import molstructures
#import pyscf.molstructures.lattices
from pyscf import embcc

#import pyscf.pbc.df as df
#from mpi4pyscf.pbc import df

#from pyscf.molstructures import mod_for_counterpoise

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
# CELL
#parser.add_argument("--lattice-const", type=float, nargs=2, default=[2.51, 30.0],
#        help="Lattice constants a (plane) and c (non-periodic dimension) in Angstrom")
#parser.add_argument("--supercell", type=int, nargs=3,
#        default=[2, 2, 1],
#        #default=[3, 3, 1],
#        #default=[4, 4, 1],
#        help="Supercell size in each direction")
parser.add_argument("--vacuum-size", type=float,
        #default=30.0)
        default=35.0)
parser.add_argument("--distances", type=float, nargs="*",
        #default=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        #default=[2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        #default=[2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        default=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0],
        help="Set of substrate-surface distances to calculate")
parser.add_argument("--ke-cutoff", type=float, help="Planewave cutoff")
parser.add_argument("--precision", type=float,
        #default=1e-6,
        default=1e-4,
        help="Precision for density fitting, determines cell.mesh")

parser.add_argument("--exp-to-discard", type=float,
        help="Threshold for discarding diffuse Gaussians, helps convergence.")

#parser.add_argument("--basis", nargs="*", help="Basis sets: 1) for H2O-N, 2) rest of impurity, 3) Rest of surface",
#        #default=["gth-dzvp", "gth-dzvp", "gth-dzvp"])
#        default=["gth-dzv", "gth-dzvp", "gth-dzvp"])
parser.add_argument("--basis", default="gth-dzvp")

parser.add_argument("--dimension", type=int, default=2)
parser.add_argument("--layers", type=int, default=1)

parser.add_argument("--pseudopot", default="gth-pade", help="Pseudo potential.")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")

parser.add_argument("--localize-fragment")
parser.add_argument("--impurity-index", type=int, default=1)
parser.add_argument("--impurity-atoms", nargs="*")
        #default=["H*2", "O*1", "H*0", "N#0"],
        #default=["H*2", "O*1", "H*0", "N#0", "B#1"],
        #default=["H*2", "O*1", "H*1", "N#0", "B#1", "N#2"],

parser.add_argument("--solver", default="CCSD")
parser.add_argument("--benchmarks", nargs="*")
parser.add_argument("--max-memory", type=int, default=1e5)
parser.add_argument("-o", "--output", default="energies.txt")

parser.add_argument("--bath-type")
#parser.add_argument("--local-type", choices=["IAO", "LAO", "AO"], default="IAO")
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--dmet-bath-tol", type=float, default=1e-8)
parser.add_argument("--bath-tol", type=float, nargs=2)
parser.add_argument("--bath-size", type=int, nargs=2)
parser.add_argument("--bath-relative-size", type=float, nargs=2)
parser.add_argument("--mp2-correction", type=int, nargs=2)
# Load and restore DF integrals
parser.add_argument("--cderi-name", default="cderi-%.2f")
parser.add_argument("--cderi-load", action="store_true")
parser.add_argument("--cderi-save", action="store_true")

parser.add_argument("--df", choices=["gaussian", "mixed"], default="gaussian")
#parser.add_argument("--no-embcc", action="store_true")
parser.add_argument("--preconverge-mf", action="store_true")
parser.add_argument("--xc")

parser.add_argument("--exxdiv", default="ewald")
parser.add_argument("--verbose", type=int, default=4)

# Counterpoise
parser.add_argument("--fragment", choices=["all", "water", "surface"], default="all")

args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

#def make_basis(basis):
#
#    if len(basis) == 1:
#        basis = [basis[0], basis[0], basis[0]]
#    else:
#        raise ValueError()
#
#
#    elif len(basis) == 2:
#        basis = [basis[0], basis[0], basis[1]]
#    elif len(basis) != 3:
#        raise ValueError()
#
#    basis_dict = {"default" : basis[2]}
#    for atom in args.impurity_atoms:
#        #if atom in ("H*0", "N#0"):
#        if atom in ("H*0", "H*2", "O*1", "N#0"):
#            basis_dict[atom] = basis[0]
#        else:
#            basis_dict[atom] = basis[1]
#    return basis_dict

#args.basis = make_basis(args.basis)
#args.basis = args.basis[0]

if args.bath_size is None:
    args.bath_size = args.bath_relative_size
    del args.bath_relative_size

if args.minao == "full":
    args.minao = args.basis

if args.impurity_atoms is None:
    args.impurity_atoms = ["H0", "O0"]
    if args.impurity_index > 0:
        args.impurity_atoms += ["Li1"]
    #if args.impurity_index > 1:
    #    args.impurity_atoms += ["H2"]
    #if args.impurity_index > 2:
    #    args.impurity_atoms += ["Li3"]
    #if args.impurity_index > 3:
    #    args.impurity_atoms += ["H4"]
    #if args.impurity_index > 4:
    #    args.impurity_atoms += ["Li5"]
    #if args.impurity_index > 5:
    #    args.impurity_atoms += ["H6"]
    #if args.impurity_index > 6:
    #    raise NotImplementedError()
    if args.impurity_index > 1:
        args.impurity_atoms += ["H2"]
    if args.impurity_index > 2:
        args.impurity_atoms += ["Li3"]
    if args.impurity_index > 3:
        args.impurity_atoms += ["Li3_2"]
    if args.impurity_index > 4:
        args.impurity_atoms += ["H4_2"]
    if args.impurity_index > 5:
        args.impurity_atoms += ["Li5"]
    if args.impurity_index > 6:
        args.impurity_atoms += ["H6"]
    if args.impurity_index > 7:
        args.impurity_atoms += ["H6_2"]
    if args.impurity_index > 8:
        args.impurity_atoms += ["Li7_2"]
    if args.impurity_index > 9:
        args.impurity_atoms += ["Li8"]
    if args.impurity_index > 10:
        args.impurity_atoms += ["H9_2"]
    if args.impurity_index > 11:
        raise NotImplementedError()

    del args.impurity_index

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%20s: %r", name, value)

def setup_cell(distance, args, **kwargs):
    """Setup PySCF cell object."""

    cell = pyscf.molstructures.build_H2O_16LiH(distance,
            vacuum=args.vacuum_size,
            layers=args.layers,
            basis=args.basis,
            pseudo=args.pseudopot,
            dimension=args.dimension,
            precision=args.precision,
            verbose=args.verbose, **kwargs)

    #if args.ke_cutoff is not None:
    #    cell.ke_cutoff = args.ke_cutoff
    #if args.exp_to_discard is not None:
    #    cell.exp_to_discard = args.exp_to_discard
    #if args.max_memory is not None:
    #    cell.max_memory = args.max_memory

    #res = cell.search_ao_label("Li1")
    #print(res)
    #print(np.asarray(cell.ao_labels())[res])

    #res = cell.search_ao_label(args.impurity_atoms)
    #print(res)
    #print(args.impurity_atoms)
    #print(np.asarray(cell.ao_labels())[res])
    #1/0

    return cell

dm0 = None
refdata = None

for icalc, distance in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.2f", distance)

    cell = setup_cell(distance, args)

    log.info("incore_anyway? %r", cell.incore_anyway)

    # Counterpoise
    #if args.fragment != "all":
    #    water, surface = cell.make_counterpoise_fragments([["H*0", "O*1", "H*2"]])
    #    if args.fragment == "water":
    #        cell = water
    #    elif args.fragment == "surface":
    #        cell = surface

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

    cderi_name = args.cderi_name % distance
    if args.cderi_load:
        log.debug("Loading DF integrals from file %s", cderi_name)
        mf.with_df._cderi = cderi_name
    elif args.cderi_save:
        log.debug("Saving DF integrals in file %s", cderi_name)
        mf.with_df._cderi_to_save = cderi_name

    # Start from HF in reduced basis
    #if dm0 is None and args.preconverge_mf:
    if args.preconverge_mf:
        cell0 = cell.copy()
        cell0.exp_to_discard = 0.2
        cell0._built = False
        cell0.build(True, False)
        mask = np.isin(cell.ao_labels(), cell0.ao_labels())
        if not np.all(mask):
            mf0 = pyscf.pbc.scf.RHF(cell0)
            # Exxdiv
            if args.exxdiv != "ewald":
                mf0.exxdiv = args.exxdiv
            mf0 = mf0.density_fit()
            t0 = MPI.Wtime()
            mf0.kernel()
            log.info("Time for mean-field in reduced basis: %.2g", MPI.Wtime()-t0)
            assert mf0.converged

            dm0 = np.zeros((cell.nao_nr(), cell.nao_nr()))
            dm0[np.ix_(mask, mask)] = mf0.make_rdm1()
            # Fix the number of electrons of density guess
            ne = np.sum(dm0 * mf.get_ovlp())
            dm0 *= cell.nelectron / ne
            ne = np.sum(dm0 * mf.get_ovlp())
            assert np.isclose(ne, cell.nelectron)

    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.4g", (MPI.Wtime()-t0))
    dm0 = mf.make_rdm1()
    assert mf.converged

    log.info("type(mf._eri) = %r", type(mf._eri))

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
            with open(args.output + ".bench", "w") as f:
                f.write("#distance  HF  " + "  ".join(args.benchmarks) + "\n")
        with open(args.output + ".bench", "a") as f:
            f.write(("%.3f  %.8e" + (len(args.benchmarks)*"  %.8e") + "\n") % (distance, mf.e_tot, *energies))

    #t0 = MPI.Wtime()
    #mp2.ao2mo()
    #log.info("Time for ao2mo: %.2g", MPI.Wtime()-t0)
    #1/0

    #elif not args.no_embcc:
    #else:
    if True:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                localize_fragment=args.localize_fragment,
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
