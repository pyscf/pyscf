import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

from pyscf import embcc
from pyscf.embcc.molecules import *

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

class ScanIRC:

    def __init__(self, name, structure_builder, max_power=0, full_ccsd=False, tol_bath=None, mol_options=None):
        self.name = name
        self.structure_builder = structure_builder
        self.max_power = max_power
        self.full_ccsd = full_ccsd
        self.tol_bath = tol_bath
        self.mol_options = mol_options or {}

    def run(self, ircs):

        for ircidx, irc in enumerate(ircs):
            if MPI_rank == 0:
                log.info("IRC=%.3f", irc)

            mol = self.structure_builder(irc, **self.mol_options)

            mf = pyscf.scf.RHF(mol)
            mf.kernel()

            if options["full_ccsd"]:
                cc = pyscf.cc.CCSD(mf)
                cc.kernel()
                assert cc.converged

                with open(self.name + ".out", "a") as f:
                    f.write("%3f  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot))

            else:
                cc = embcc.EmbCCSD(mf)
                #ecc.create_custom_clusters([("O1", "H3")])
                cc.create_atom_clusters()
                cc.merge_clusters(("H3", "O1"))
                if didx == 0 and MPI_rank == 0:
                    cc.print_clusters()

                conv = cc.run(max_power=options["max_power"])
                if MPI_rank == 0:
                    assert conv

                if MPI_rank == 0:
                    if didx == 0:
                        with open(output, "a") as f:
                            f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(1C)  EmbCCSD(1C,v)  EmbCCSD(1C,f)\n")
                    with open(output, "a") as f:
                        f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (irc, mf.e_tot, cc.e_tot, mf.e_tot+cc.e_ccsd_v, mf.e_tot+cc.get_cluster("H3,O1").e_ccsd, mf.e_tot+cc.get_cluster("H3,O1").e_ccsd_v, mf.e_tot+cc.get_cluster("H3,O1").e_cl_ccsd))
