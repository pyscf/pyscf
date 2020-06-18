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

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-b", "--basis", default="cc-pVDZ")
parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--full-ccsd", action="store_true")
#parser.add_argument("--no-embccsd", action="store_true")
#parser.add_argument("--tol-bath", type=float, default=1e-5)
#parser.add_argument("--tol-vno", type=float, default=1e-3)
parser.add_argument("--bath-type")
parser.add_argument("--bath-target-size", type=int, nargs=2, default=[None, None])
#parser.add_argument("--tol-vno", type=float, default=1e-3)
parser.add_argument("--ircs", type=float, nargs=3, default=[0.6, 3.3, 0.2])
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

#ircs = np.arange(0.6, 3.3+1e-14, 0.1)
ircs = np.arange(args.ircs[0], args.ircs[1]+1e-14, args.ircs[2])
structure_builder = molstructures.build_ethanol

ref_orbitals = None
for ircidx, irc in enumerate(ircs):
    if MPI_rank == 0:
        log.info("IRC=%.3f", irc)

    mol = structure_builder(irc, basis=args.basis, verbose=5)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    if args.full_ccsd:
        ccsd = pyscf.cc.CCSD(mf)
        ccsd.kernel()
        assert ccsd.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, ccsd.e_tot))

    #if not args.no_embccsd:
    else:
        if False:
            if ircidx == 0:
                cc = embcc.EmbCC(mf, bath_type=args.bath_type, bath_target_size=args.bath_target_size,
                        tol_bath=args.tol_bath, tol_vno=args.tol_vno)
                #cc = embcc.EmbCC(mf, tol_bath=args.tol_bath, benchmark=ccsd)
                #ecc.create_custom_clusters([("O1", "H3")])
                cc.make_atom_clusters()
                #oh_cluster = cc.merge_clusters(("O1", "H3"))
                #coh_cluster = cc.merge_clusters(("C1", "O1", "H3"))
                if MPI_rank == 0:
                    cc.print_clusters()
            else:
                cc.reset(mf=mf)
        else:
            cc = embcc.EmbCC(mf, bath_type=args.bath_type, bath_target_size=args.bath_target_size)
            cc.make_iao_atom_clusters()
            if ref_orbitals is not None:
                cc.set_reference_orbitals(ref_orbitals)


        #conv = cc.run(max_power=args.max_power)
        conv = cc.run()
        if MPI_rank == 0:
            assert conv

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if ircidx == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(1C)  EmbCCSD(1C,v)  EmbCCSD(1C,f)\n")
                    #f.write("#IRC  HF  EmbCCSD  EmbCCSD(vir)  EmbCCSD(var)  EmbCCSD(var2)  EmbCCSD(var3)\n")
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(vir)  EmbCCSD(dMP2)  EmbCCSD(v,dMP2)\n")
            with open(args.output, "a") as f:
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_tot_v, cc.e_tot_var, cc.e_tot_var2))
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_tot_v, cc.e_tot_var, cc.e_tot_var2, cc.e_tot_var3))
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, cc.e_tot_v, cc.e_tot_dmp2, cc.e_tot_v_dmp2))
