import sys
import numpy
from pyscf import gto, tools

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

nf = mol.nao_nr()
orb = numpy.random.random((nf,4))

tools.dump_mat.dump_mo(mol, orb)

dm = numpy.eye(3)
tools.dump_mat.dump_tri(sys.stdout, dm)

mol = gto.M(atom='C 0 0 0',basis='6-31g')
dm = numpy.eye(mol.nao_nr())
tools.dump_mat.dump_rec(sys.stdout, dm, mol.spheric_labels(True),
                        ncol=9, digits=2)
