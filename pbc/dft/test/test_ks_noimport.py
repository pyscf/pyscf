#!/usr/bin/env python
#
# Author: Artem Pulkin <gpulkin@gmail.com>
#

import unittest, json
import numpy

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

class Monolayer_hBN_OpenMX(unittest.TestCase):
    """
    Monolayer hexagonal boron nitride simple LDA model compared against
    `OpenMX <http://www.openmx-square.org/>`_ v 3.8. Following is the
    corresponding input file:
    
    .. code-block:: none
        System.CurrentDirectory ./
        System.Name _hBN
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
    
    bands_hartree_gamma = [
        [-0.861950848488467, -0.67011357660001, -0.507120859560528, -0.362669234096556, -0.142309695816326, 0.114407484255813, 0.418869731632146, 0.460765115141375],
        [-0.936575184390801, -0.535873193029748, -0.46220882829624, -0.46199775997269105, 0.002110899055795, 0.100733643242764, 0.264696274970974, 0.291349517733883],
        [-0.980941112976582, -0.512924095027141, -0.397055805051337, -0.39705341031315394, 0.079779255961822, 0.085156614935957, 0.11822968560863702, 0.118231705594354],
        [-0.922973838912517, -0.552505747301973, -0.495703333829416, -0.445709403534751, -0.021188058411891, 0.159817953111987, 0.22857626934484102, 0.359966450463679],
        [-0.8448546289178172, -0.631808504806979, -0.610787361051, -0.32564331424668, -0.17088539489055396, 0.286930948731176, 0.380538832039589, 0.47514935141294895],
        [-0.8549430762742248, -0.658275734293208, -0.552152879431288, -0.348370743265163, -0.151615495191965, 0.19745165589653602, 0.394107889840298, 0.475664218990589],
        [-0.861950848488467, -0.67011357660001, -0.507120859560528, -0.362669234096556, -0.142309695816326, 0.114407484255813, 0.418869731632146, 0.460765115141375]
    ]
    
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
        with open("openmx_hBN_s1p1_basis.json",'r') as f:
            basis = json.load(f)
            #print basis
            #exit()
        self.cell = pbcgto.M(
            unit = 'Angstrom',
            atom = list(zip(['B','N'],self.atomic_coordinates_cartesian_angstrom)),
            a = self.unit_cell_angstrom,
            basis = basis,
            pseudo = 'gth-lda',
            gs = [16,16,75],
            verbose = 4,
        )
        print self.cell._pseudo
        exit()
        
    def __plot__(self,one,two):
        """
        Plots band structures against each other. It is here for debug
        purposes: `pyplot.show()` blocks the thread.
        """
        from matplotlib import pyplot
        for x in numpy.array(one).swapaxes(0,1):
            pyplot.plot(numpy.arange(len(x)),x,marker = "x",color = "b")
        for x in numpy.array(two).swapaxes(0,1):
            pyplot.plot(numpy.arange(len(x)),x,marker = "x",color = "r")
        pyplot.show()

    def _test_bands_gamma(self):
        model = pbcdft.KRKS(self.cell, self.cell.make_kpts([1,1,1]))
        model.xc = 'lda'
        model.kernel()
        e,w = model.get_bands(self.k_points_cartesian_bohr)
        avg = numpy.mean(e-self.bands_hartree_gamma)
        delta = e-self.bands_hartree_gamma-avg
        self.__plot__(e-avg, self.bands_hartree_gamma)
        dev_max = numpy.abs(delta).max()
        print "Maximum deviation:", dev_max, " Hartree"
        assert dev_max < 1e-4

    def _test_bands(self):
        model = pbcdft.KRKS(self.cell, self.cell.make_kpts([3,3,1]))
        model.xc = 'lda'
        model.kernel()
        e,w = model.get_bands(self.k_points_cartesian_bohr)
        avg = numpy.mean(e-self.bands_hartree)
        delta = e-self.bands_hartree-avg
        self.__plot__(e-avg, self.bands_hartree)
        dev_max = numpy.abs(delta).max()
        print "Maximum deviation:", dev_max, " Hartree"
        assert dev_max < 1e-4
        
class Monolayer_hBN_cp2k(unittest.TestCase):
    """
    Monolayer hexagonal boron nitride simple LDA model compared against
    `CP2K <https://www.cp2k.org>`_ v 4.1(17462). Following is the
    corresponding input file:
    
    .. code-block:: none
        &GLOBAL
          PROJECT _hBN
          RUN_TYPE ENERGY
          PRINT_LEVEL MEDIUM
        &END GLOBAL
        &FORCE_EVAL
          METHOD Quickstep
          &SUBSYS
            &KIND B
              &BASIS
                1
                  1 0 1 4 1 1
                    2.8854084023  0.1420731829 -0.0759815770
                    0.8566849689 -0.0083257749 -0.2508281584
                    0.2712991753 -0.6707104603 -0.4610296144
                    0.0826101984 -0.4241277148 -0.4419922734
              &END BASIS
              &POTENTIAL
                2    1
                 0.43392956    2    -5.57864173     0.80425145
                2
                 0.37384326    1     6.23392822
                 0.36039317    0
              &END POTENTIAL
            &END KIND
            &KIND N
              &BASIS
                1
                  1 0 1 4 1 1
                    6.1526903413  0.1506300537 -0.0950603476
                    1.8236332280 -0.0360100734 -0.2918864295
                    0.5676628870 -0.6942023212 -0.4739050050
                    0.1628222852 -0.3878929987 -0.3893418670
              &END BASIS
              &POTENTIAL
                2    3
                 0.28917923    2   -12.23481988     1.76640728
                2
                 0.25660487    1    13.55224272
                 0.27013369    0
              &END POTENTIAL
            &END KIND
            &CELL
              A 2.515000000 0.000000000  0.000000000
              B 1.257500000 2.178053891  0.000000000
              C 0.000000000 0.000000000 10.000000000
            &END CELL
            &COORD
              B 1.257500000 0.7260179636666594 5.0
              N 2.515000000 1.4520359273333405 5.0
            &END COORD
          &END SUBSYS
          &DFT
            &KPOINTS
              FULL_GRID
              SCHEME MONKHORST-PACK 1 1 1
              &BAND_STRUCTURE
                ADDED_MOS 4
                &KPOINT_SET
                  UNITS CART_BOHR
                  NPOINTS 2
                  SPECIAL_POINT 0.5 0.0 0.0
                  SPECIAL_POINT 0 0 0
                  SPECIAL_POINT 0.6666666667 0.3333333333 0.0
                  SPECIAL_POINT 0.5 0.0 0.0
                &END KPOINT_SET
                FILE_NAME bands
              &END BAND_STRUCTURE
            &END KPOINTS
            &QS
              EXTRAPOLATION USE_PREV_WF
            &END QS
            &MGRID
              CUTOFF 150
              NGRIDS 1
            &END MGRID
            &XC
              &XC_FUNCTIONAL PADE
              &END XC_FUNCTIONAL
            &END XC
            &SCF
              EPS_SCF 1.0E-10
              MAX_SCF 300
            &END SCF
          &END DFT
        &END FORCE_EVAL
    """
        
    atomic_coordinates_cartesian_angstrom = [
        [1.2574999999999876, 0.7260179636666594, 5.0],
        [2.5150000000000126, 1.4520359273333405, 5.0]
    ]
    
    unit_cell_angstrom = [
        [2.515, 0.0, 0.0],
        [1.2575, 2.178053891, 0.0],
        [0.0, 0.0, 10.0]
    ]
    
    k_points_cartesian_bohr = [
        [0.661017637338074, -0.3816387107717475, 0.0],
        [0.330508818669037, -0.19081935538587375, 0.0],
        [0.0, 0.0, 0.0],
        [0.44067842491408316, 0.0, 0.0],
        [0.8813568498281663, 0.0, 0.0],
        [0.7711872435170185, -0.19081935538587372, 0.0],
        [0.661017637338074, -0.3816387107717475, 0.0]
    ]

    def setUp(self):
        self.cell = pbcgto.M(
            unit = 'Angstrom',
            atom = list(zip(['B','N'],self.atomic_coordinates_cartesian_angstrom)),
            a = self.unit_cell_angstrom,
            basis = "gth-szv",
            pseudo = 'gth-pbe',
            gs = [20,20,75],
            verbose = 4,
            rcut = 40,
        )
        with open("test_ks_noimport_hBN_cp2k.json",'r') as f:
            self.reference_data = json.load(f)
        #self.reference_data = {
            #"units": {"energy": "Hartree"},
            #"e": [
                #[
                    #[-0.7352401668289757, -0.542443617099548, -0.38536985723910133, -0.23600882312393842, -0.01850650744791426, 0.2724005054425331, 0.5782536796301777, 0.5896418889951032],
                    #[-0.8044656368628388, -0.41172553809910795, -0.3415628405056045, -0.33093945145268144, 0.1244790386237603, 0.24496329051318277, 0.40640662949663836, 0.4155415492954867],
                    #[-0.8486944729921913, -0.37786008735184934, -0.27686259069839253, -0.276862103402359, 0.18698491685078059, 0.22118239296114078, 0.23912902666595426, 0.23912946214544128],
                    #[-0.7912458408323351, -0.4300514343803395, -0.3723482073151307, -0.31546456660920746, 0.10249532265286361, 0.30391556462042996, 0.38221693761157444, 0.48146095350977347],
                    #[-0.7196615757612375, -0.5094251500674526, -0.48397912632189377, -0.19991743680227816, -0.04681937350493613, 0.46941601425143864, 0.5468146548032186, 0.6100634272256907],
                    #[-0.728721143644091, -0.5315864479582714, -0.4296515186839471, -0.22201485848844454, -0.027826214239108565, 0.3640163326374332, 0.5648942229394857, 0.6020252406490155],
                    #[-0.7352401668289757, -0.542443617099548, -0.38536985723910133, -0.23600882312393842, -0.01850650744791426, 0.2724005054425331, 0.5782536796301777, 0.5896418889951032]
                #], [
                    #[-0.7151233532563337, -0.5156591083801442, -0.36809396116744914, -0.21153483332506906, 0.008412695260566632, 0.29788652384467434, 0.5909839768726532, 0.5956539842901624],
                    #[-0.7810533673678941, -0.3935951191885153, -0.32704203205174603, -0.3019360766262951, 0.14299956704118313, 0.27061563231240743, 0.4185461188852132, 0.43326100566295334],
                    #[-0.8242079994480276, -0.34775565133542613, -0.2676819903875838, -0.2671529835040205, 0.20148960484626188, 0.24082759731657982, 0.26417584177211056, 0.2645020173778499],
                    #[-0.7683010723476863, -0.4111633135017639, -0.35478788213458695, -0.2869473537322482, 0.12229248288975404, 0.32618017037284314, 0.39850367014172866, 0.4943330869724064],
                    #[-0.7011666961668662, -0.4802007273234356, -0.46197228951392966, -0.17810966373146225, -0.017366027830692087, 0.4755010459294395, 0.5721478728244473, 0.6156026502292877],
                    #[-0.7092164811976251, -0.5046753429632893, -0.4088025865525772, -0.19839766300421954, -0.00015858472447809763, 0.3829537211294652, 0.5788244429329062, 0.6095159714465944],
                    #[-0.7151233532563337, -0.5156591083801442, -0.36809396116744914, -0.21153483332506906, 0.008412695260566632, 0.29788652384467434, 0.5909839768726532, 0.5956539842901624]
                #]
            #],
            #"description": "Electronic band structure obtained from CP2K"
        #}
        
    def __plot__(self,one,two):
        """
        Plots band structures against each other. It is here for debug
        purposes: `pyplot.show()` blocks the thread.
        """
        from matplotlib import pyplot
        for x in numpy.array(one).swapaxes(0,1):
            pyplot.plot(numpy.arange(len(x)),x,marker = "x",color = "b")
        for x in numpy.array(two).swapaxes(0,1):
            pyplot.plot(numpy.arange(len(x)),x,marker = "x",color = "r")
        print "Maximum deviation:", numpy.abs(one-two).max(), " Hartree"
        pyplot.show()

    def test_bands_gamma(self):
        """
        Test Gamma-point calculation.
        """
        model = pbcdft.KRKS(self.cell, self.cell.make_kpts([1,1,1]))
        model.xc = 'pbe'
        model.kernel()
        e, w = model.get_bands(self.k_points_cartesian_bohr)
        ref = self.reference_data["e"][0]
        avg = numpy.mean(e-ref)
        e -= avg
        delta = e - ref
        self.__plot__(e, ref)
        dev_max = numpy.abs(delta).max()
        
        assert dev_max < 1e-5 # Hartree
        # 0.00462320015175

    def _test_bands(self):
        """
        Test 3x3 k-grid calculation.
        """
        model = pbcdft.KRKS(self.cell, self.cell.make_kpts([3,3,1]))
        model.xc = 'lda'
        model.kernel()
        e, w = model.get_bands(self.k_points_cartesian_bohr)
        ref = self.reference_data["e"][1]
        avg = numpy.mean(e-ref)
        e -= avg
        delta = e - ref
        self.__plot__(e, ref)
        dev_max = numpy.abs(delta).max()
        
        assert dev_max < 1e-5 # Hartree
        # 0.00654300478426

if __name__ == '__main__':
    unittest.main()
