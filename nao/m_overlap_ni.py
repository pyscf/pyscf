from __future__ import division, print_function
import numpy as np

#
#
#
def overlap_ni(me, sp1, sp2, R1, R2, level=None):
    """
      Computes overlap for an atom pair. The atom pair is given by a pair of species indices
      and the coordinates of the atoms.
      Args: 
        sp1,sp2 : specie indices, and
        R1,R2 :   respective coordinates in Bohr, atomic units
      Result:
        matrix of orbital overlaps
      The procedure uses the numerical integration in coordinate space.
    """
    from pyscf import gto
    from pyscf import dft
    from pyscf.nao.m_gauleg import leggauss_ab
    #from pyscf.nao.m_ao_eval import ao_eval
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval #_libnao
#    from timeit import default_timer as timer
    
    assert(sp1>-1)
    assert(sp2>-1)

    shape = [me.sp2norbs[sp] for sp in (sp1,sp2)]
    
#    start1 = timer()
    if ((R1-R2)**2).sum()<1e-7 :
      mol = gto.M( atom=[ [int(me.sp2charge[sp1]), R1]],)
    else :
      mol = gto.M( atom=[ [int(me.sp2charge[sp1]), R1], [int(me.sp2charge[sp2]), R2] ],)
#    end1 = timer()
    
#    start2 = timer()
    atom2rcut=np.array([me.sp_mu2rcut[sp].max() for sp in (sp1,sp2)])
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3 if level is None else level # precision as implemented in pyscf
    grids.radi_method=leggauss_ab
    grids.build(atom2rcut=atom2rcut)
#    end2 = timer()
    
#    start3 = timer()
    ao1 = ao_eval(me, R1, sp1, grids.coords)
    ao2 = ao_eval(me, R2, sp2, grids.coords)
#    end3 = timer()
    
#    start4 = timer()
    ao1 = ao1 * grids.weights
    overlaps = np.einsum("ij,kj->ik", ao1, ao2) #      overlaps = np.matmul(ao1, ao2.T)
#    end4 = timer()
    
#    print(end1-start1, end2-start2, end3-start3, end4-start4)
#    0.020702847745269537 0.0079775620251894 0.025534397922456264 0.0045219180174171925
    
    return overlaps
