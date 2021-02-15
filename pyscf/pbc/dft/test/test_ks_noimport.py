#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Artem Pulkin <gpulkin@gmail.com>
#

import unittest
import numpy

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import df

class BN(unittest.TestCase):
    """
    Monolayer hexagonal boron nitride simple LDA model compared against
    `OpenMX <http://www.openmx-square.org/>`_ v 3.8. Following is the
    corresponding input file:
    
    .. code-block:: none
        System.CurrrentDirectory ./
        System.Name _gr
        data.path /export/scratch/openmx_tests/DFT_DATA13
        level.of.stdout 1
        level.of.fileout 1
        
        Species.Number 2
        <Definition.of.Atomic.Species
         B   B7.0-s1p1    B_CA13
         N   N7.0-s1p1    N_CA13
        Definition.of.Atomic.Species>
        
        Atoms.UnitVectors.Unit Ang
        
        <Atoms.UnitVectors
        2.515 0.0 0.0
        1.2575 2.178053891 0.0
        0.0 0.0 10.0
        Atoms.UnitVectors>
        
        Atoms.Number 2
        
        Atoms.SpeciesAndCoordinates.Unit   Frac
        <Atoms.SpeciesAndCoordinates
           1    B    0.33333333333333    0.33333333333333    0.5     1.5     1.5
           2    N    0.66666666666667    0.66666666666667    0.5     2.5     2.5
        Atoms.SpeciesAndCoordinates>
        
        scf.XcType                  LDA
        scf.SpinPolarization        off
        scf.EigenvalueSolver        band
        scf.Kgrid                   3 3 1
        scf.Mixing.Type             rmm-diis
        
        Band.dispersion on
        Band.Nkpath 3
        
        <Band.kpath
          3  0.5 0.0 0.0  0.0 0.0 0.0  M G
          3  0.0 0.0 0.0  0.6666666667 0.3333333333 0.0  G K
          3   0.6666666666 0.3333333333 0.0  0.5 0.0 0.0  K M
        Band.kpath>
        
    The test case is under construction.
    """
    
    k_points_cartesian_bohr = [
        [0.661017637338074, -0.3816387107717475, 0.0],
        [0.330508818669037, -0.19081935538587375, 0.0],
        [0.0, 0.0, 0.0],
        [0.44067842491408316, 0.0, 0.0],
        [0.8813568498281663, 0.0, 0.0],
        [0.7711872435170185, -0.19081935538587372, 0.0],
        [0.661017637338074, -0.3816387107717475, 0.0]
    ]


    bands_hartree = [
        [-0.842066600017214, -0.648283009659859, -0.492574666054724, -0.336259059007073, -0.120464454049752, 0.135165429701139, 0.424409442740589, 0.469400069009213],
        [-0.914469823273887, -0.520404527903842, -0.44959478067300607, -0.434574288860007, 0.021825968812443, 0.12144591716363101, 0.274989742357433, 0.304298742813527],
        [-0.958216228044807, -0.484752567508547, -0.388488278604123, -0.38848781183717407, 0.09819301148466598, 0.103233114821577, 0.136738819356143, 0.136739210436253],
        [-0.901142346781313, -0.536740309674447, -0.480714557955078, -0.41826540431865605, -0.001123876879672, 0.177930037655232, 0.242098199261799, 0.36921183082317505],
        [-0.826070563998053, -0.607353411513363, -0.59354899893476, -0.299343371359265, -0.149278200607161, 0.29295749686182, 0.40014717665440697, 0.479661775180689],
        [-0.835499249093366, -0.636304100142041, -0.534912679419229, -0.322031767475658, -0.129877835821144, 0.212960530752593, 0.404614663360821, 0.481950877444352],
        [-0.842066600017214, -0.648283009659859, -0.492574666054724, -0.336259059007073, -0.120464454049752, 0.135165429701139, 0.424409442740589, 0.469400069009213]
    ]

# gth-szv results with MDF
#    bands_hartree = [
#[[-0.79216492, -0.5957496 , -0.44516754, -0.29029039, -0.07589355, 0.21641387, 0.51330468, 0.51737139],
# [-0.85926929, -0.47127114, -0.40354241, -0.38502347,  0.06123951, 0.18843132, 0.33915469, 0.35464616],
# [-0.90277877, -0.43213462, -0.3430856 , -0.34295502,  0.11920423, 0.15892713, 0.18394559, 0.18422438],
# [-0.84631886, -0.48905854, -0.43215181, -0.36949771,  0.04041299, 0.24454995, 0.31943137, 0.41599979],
# [-0.77755574, -0.56044286, -0.54122393, -0.25458428, -0.10326825, 0.39781545, 0.49173201, 0.5386686 ],
# [-0.78602643, -0.58469819, -0.48708744, -0.27642628, -0.08486146, 0.30302124, 0.49977162, 0.53225523],
# [-0.79216492, -0.5957496 , -0.44516754, -0.29029039, -0.07589355, 0.21641387, 0.51330468, 0.51737139],]

    atomic_coordinates_cartesian_angstrom = [
        [1.2574999999999876, 0.7260179636666594, 5.0],
        [2.5150000000000126, 1.4520359273333405, 5.0]
    ]

    unit_cell_angstrom = [
        [2.515, 0.0, 0.0],
        [1.2575, 2.178053891, 0.0],
        [0.0, 0.0, 10.0]
    ]

    def setUp(self):
        self.cell = pbcgto.M(
            unit = 'Angstrom',
            atom = list(zip(['B','N'],self.atomic_coordinates_cartesian_angstrom)),
            a = self.unit_cell_angstrom,
            basis = 'gth-szv',
            pseudo = 'gth-lda',
            mesh = [32,32,150],
            verbose = 4,
        )

    def test_bands_high_cost(self):
        model = pbcdft.KRKS(self.cell, self.cell.make_kpts([3,3,1]))
        model.xc = 'lda,'
        model.kernel()
        e,w = model.get_bands(self.k_points_cartesian_bohr)
        e = numpy.asarray(e)
        avg = numpy.mean(e-self.bands_hartree)
        delta = e-self.bands_hartree-avg
        dev_max = numpy.abs(delta).max()
        print("Maximum deviation:", dev_max, " Hartree")
        assert dev_max < 1e-4

#    def setUp(self):
#        self.cell = pbcgto.M(
#            unit = 'Angstrom',
#            atom = list(zip(['B','N'],self.atomic_coordinates_cartesian_angstrom)),
#            a = self.unit_cell_angstrom,
#            dimension = 2,
#            basis = 'ccpvdz',
#            mesh = [11,11,51],
#            verbose = 4,
#        )
#
#    def test_bands(self):
#        model = pbcdft.KRKS(self.cell, self.cell.make_kpts([3,3,1]))
#        model.with_df = df.MDF(self.cell)
#        model.with_df.auxbasis = None
#        model.with_df.kpts = self.cell.make_kpts([3,3,1])
#        model.with_df.kpts_band = self.k_points_cartesian_bohr
#        model.xc = 'lda,'
#        model.grids = pbcdft.BeckeGrids(self.cell)
#        model.kernel()
#        e,w = model.get_bands(self.k_points_cartesian_bohr)
#        #avg = numpy.mean(e-self.bands_hartree)
#        #delta = e-self.bands_hartree-avg
#        #dev_max = numpy.abs(delta).max()
#        #print("Maximum deviation:", dev_max, " Hartree")
#        #assert dev_max < 1e-4

if __name__ == '__main__':
    unittest.main()
