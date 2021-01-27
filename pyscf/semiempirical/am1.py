#!/usr/bin/env python

'''
MNDO-AM1
(In testing)

Ref:
[1] J. J. Stewart, J. Comp. Chem. 10, 209 (1989)
[2] J. J. Stewart, J. Mol. Model 10, 155 (2004)
'''

import ctypes
import copy
import numpy
import warnings
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mopac_param, mindo3

warnings.warn('AM1 model is under testing')

libsemiempirical = lib.load_library('libsemiempirical')
ndpointer = numpy.ctypeslib.ndpointer
libsemiempirical.MOPAC_rotate.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ndpointer(dtype=numpy.double),  # xi
    ndpointer(dtype=numpy.double),  # xj
    ndpointer(dtype=numpy.double),  # w
    ndpointer(dtype=numpy.double),  # e1b
    ndpointer(dtype=numpy.double),  # e2a
    ndpointer(dtype=numpy.double),  # enuc
    ndpointer(dtype=numpy.double),  # alp
    ndpointer(dtype=numpy.double),  # dd
    ndpointer(dtype=numpy.double),  # qq
    ndpointer(dtype=numpy.double),  # am
    ndpointer(dtype=numpy.double),  # ad
    ndpointer(dtype=numpy.double),  # aq
    ndpointer(dtype=numpy.double),  # fn1
    ndpointer(dtype=numpy.double),  # fn2
    ndpointer(dtype=numpy.double),  # fn3
    ctypes.c_int
]
repp = libsemiempirical.MOPAC_rotate


MOPAC_DD = numpy.array((0.,
    0.       , 0.       ,
    2.0549783, 1.4373245, 0.9107622, 0.8236736, 0.6433247, 0.4988896, 0.4145203, 0.,
    0.       , 0.       , 1.4040443, 1.1631107, 1.0452022, 0.9004265, 0.5406286, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.3581113, 0., 1.2472095, 0., 0.       , 0.8458104, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.4878778, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.8750829, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4078712, 0.8231596, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.0684105, 0.       , 0., 0., 0., 0.,
))

MOPAC_QQ = numpy.array((0.,
    0.       , 0.       ,
    1.7437069, 1.2196103, 0.7874223, 0.7268015, 0.5675528, 0.4852322, 0.4909446, 0.,
    0.       , 0.       , 1.2809154, 1.3022422, 0.8923660, 1.0036329, 0.8057208, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.5457406, 0., 1.0698642, 0., 0.       , 1.0407133, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.1887388, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.5424241, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.1658281, 0.8225156, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 1.0540926, 0.       , 0., 0., 0., 0.,
))

MOPAC_AM = numpy.array((0.,
    0.4721793, 0.       ,
    0.2682837, 0.3307607, 0.3891951, 0.4494671, 0.4994487, 0.5667034, 0.6218302, 0.,
    0.5      , 0.       , 0.2973172, 0.3608967, 0.4248440, 0.4331617, 0.5523705, 0.,
    0.5      , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.4336641, 0., 0.3737084, 0., 0.       , 0.5526071, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.5527544, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.3969129, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3608967, 0.4733554, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.4721793, 0.5      , 0.5      ,0.5      , 0.5      , 0.       ,
))

MOPAC_AD = numpy.array((0.,
    0.4721793, 0.       ,
    0.2269793, 0.3356142, 0.5045152, 0.6082946, 0.7820840, 0.9961066, 1.2088792, 0.,
    0.       , 0.       , 0.2630229, 0.3829813, 0.3275319, 0.5907115, 0.7693200, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.2317423, 0., 0.3180309, 0., 0.       , 0.6024598, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.4497523, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.2926605, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3441817, 0.5889395, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.9262742, 0.       , 0., 0., 0., 0.,
))

MOPAC_AQ = numpy.array((0.,
    0.4721793, 0.       ,
    0.2614581, 0.3846373, 0.5678856, 0.6423492, 0.7883498, 0.9065223, 0.9449355, 0.,
    0.       , 0.       , 0.3427832, 0.3712106, 0.4386854, 0.6454943, 0.6133369, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.2621165, 0., 0.3485612, 0., 0.       , 0.5307555, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.4631775, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.3360599, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 0.3999442, 0.5632724, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.2909059, 0.       , 0., 0., 0., 0.,
))

MOPAC_ALP = numpy.array((0.,
    2.8823240, 0.       ,
    1.2501400, 1.6694340, 2.4469090, 2.6482740, 2.9472860, 4.4553710, 5.5178000, 0.,
    1.6680000, 0.       , 1.9765860, 2.2578160, 2.4553220, 2.4616480, 2.9193680, 0.,
    1.4050000, 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.4845630, 0., 2.1364050, 0., 0.       , 2.5765460, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.2994240, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.4847340, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 2.1961078, 2.4916445, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 2.5441341, 1.5      , 1.5      ,1.5      , 1.5      , 0.       ,
))

MOPAC_ZS = numpy.array((0.,
    1.1880780, 0.       ,
    0.7023800, 1.0042100, 1.6117090, 1.8086650, 2.3154100, 3.1080320, 3.7700820, 0.,
    0.       , 0.       , 1.5165930, 1.8306970, 1.9812800, 2.3665150, 3.6313760, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.9542990, 0., 1.2196310, 0., 0.       , 3.0641330, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.1028580, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 2.0364130, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4353060, 2.6135910, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 4.0000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_ZP = numpy.array((0.,
    0.       , 0.       ,
    0.7023800, 1.0042100, 1.5553850, 1.6851160, 2.1579400, 2.5240390, 2.4946700, 0.,
    0.       , 0.       , 1.3063470, 1.2849530, 1.8751500, 1.6672630, 2.0767990, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.3723650, 0., 1.9827940, 0., 0.       , 2.0383330, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.1611530, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.9557660, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.4353060, 2.0343930, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.3000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_ZD = numpy.array((0.,
    0.       , 0.       ,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.,
    0.       , 0.       , 1.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 1.0000000, 0., 0.       , 0., 0.       , 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 1.0000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 0.       , 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.0000000, 1.0000000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.3000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_USS = numpy.array((0.,
    -11.396427, 0.       ,
    -5.128000,-16.602378,-34.492870,-52.028658,-71.860000,-97.830000,-136.105579,0.,
    0.       , 0.       ,-24.353585,-33.953622,-42.029863,-56.694056,-111.613948,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,-21.040008, 0.,-34.183889, 0., 0.       ,-104.656063,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,-103.589663,0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,-19.941578, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       ,-40.568292,-75.239152, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,-11.906276, 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_UPP = numpy.array((0.,
    0.       , 0.       ,
    -2.721200,-10.703771,-22.631525,-39.614239,-57.167581,-78.26238,-104.889885, 0.,
    0.       , 0.       ,-18.363645,-28.934749,-34.030709,-48.717049,-76.640107, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,-17.655574, 0.,-28.640811, 0., 0.       ,-74.930052, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,-74.429997, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,-11.110870, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       ,-28.089187,-57.832013, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
)) * 1./mopac_param.HARTREE2EV

MOPAC_GSS = numpy.array((0.,
    12.8480000, 0.       ,
    7.3000000, 9.0000000,10.5900000,12.2300000,13.5900000,15.4200000,16.9200000, 0.,
    0.       , 0.       , 8.0900000, 9.8200000,11.5600050,11.7863290,15.0300000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,11.8000000, 0.,10.1686050, 0., 0.       ,15.0364395, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,15.0404486, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 10.800000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 9.8200000,12.8800000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0.,12.8480000, 0.       , 0., 0., 0., 0.,
))

MOPAC_GSP = numpy.array((0.,
    0.       , 0.       ,
    5.4200000, 7.4300000, 9.5600000,11.4700000,12.6600000,14.4800000,17.2500000, 0.,
    0.       , 0.       , 6.6300000, 8.3600000, 5.2374490, 8.6631270,13.1600000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,11.1820180, 0., 8.1444730, 0., 0.       ,13.0346824, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,13.0565580, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 9.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 8.3600000,11.2600000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_GPP = numpy.array((0.,
    0.       , 0.       ,
    5.0000000, 6.9700000, 8.8600000,11.0800000,12.9800000,14.5200000,16.7100000, 0.,
    0.       , 0.       , 5.9800000, 7.3100000, 7.8775890,10.0393080,11.3000000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,13.3000000, 0., 6.6719020, 0., 0.       ,11.2763254, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       ,11.1477837, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,14.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 7.3100000, 9.9000000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_GP2 = numpy.array((0.,
    0.       , 0.       ,
    4.5200000, 6.2200000, 7.8600000, 9.8400000,11.5900000,12.9800000,14.9100000, 0.,
    0.       , 0.       , 5.4000000, 6.5400000, 7.3076480, 7.7816880, 9.9700000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0.,12.9305200, 0., 6.2697060, 0., 0.       , 9.8544255, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 9.9140907, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0.,13.5000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 6.5400000, 8.8300000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0.,
))

MOPAC_HSP = numpy.array((0.,
    0.       , 0.       ,
    0.8300000, 1.2800000, 1.8100000, 2.4300000, 3.1400000, 3.9400000, 4.8300000, 0.,
    0.       , 0.       , 0.7000000, 1.3200000, 0.7792380, 2.5321370, 2.4200000, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.4846060, 0., 0.9370930, 0., 0.       , 2.4558683, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 2.4563820, 0.,
    0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.       , 0.       , 0., 0., 0., 0., 0., 0., 0., 0., 1.3000000, 0., 0., 0., 0., 0., 0.,
    0.       , 0.       , 0.       , 1.3200000, 2.2600000, 0.       , 0.       , 0., 0., 0., 0., 0.       , 0., 0.       , 0., 0.1000000, 0.       , 0., 0., 0., 0.,
))

MOPAC_IDEA_FN1 = numpy.zeros((108,10))
dat = (
    1 , 1,  0.1227960,
    1 , 2,  0.0050900,
    1 , 3, -0.0183360,
    6 , 1,  0.0113550,
    6 , 2,  0.0459240,
    6 , 3, -0.0200610,
    6 , 4, -0.0012600,
    7 , 1,  0.0252510,
    7 , 2,  0.0289530,
    7 , 3, -0.0058060,
    8 , 1,  0.2809620,
    8 , 2,  0.0814300,
    9 , 1,  0.2420790,
    9 , 2,  0.0036070,
    13, 1,  0.0900000,
    14, 1,  0.25,
    14, 2,  0.061513,
    14, 3,  0.0207890,
    15, 1, -0.0318270,
    15, 2,  0.0184700,
    15, 3,  0.0332900,
    16, 1, -0.5091950,
    16, 2, -0.0118630,
    16, 3,  0.0123340,
    17, 1,  0.0942430,
    17, 2,  0.0271680,
    35, 1,  0.0666850,
    35, 2,  0.0255680,
    53, 1,  0.0043610,
    53, 2,  0.0157060,
)
MOPAC_IDEA_FN1[dat[0::3],dat[1::3]] = numpy.array(dat[2::3]) / mopac_param.HARTREE2EV

MOPAC_IDEA_FN2 = numpy.zeros((108,10))
dat = (
    1 , 1,  5.0000000,
    1 , 2,  5.0000000,
    1 , 3,  2.0000000,
    6 , 1,  5.0000000,
    6 , 2,  5.0000000,
    6 , 3,  5.0000000,
    6 , 4,  5.0000000,
    7 , 1,  5.0000000,
    7 , 2,  5.0000000,
    7 , 3,  2.0000000,
    8 , 1,  5.0000000,
    8 , 2,  7.0000000,
    9 , 1,  4.8000000,
    9 , 2,  4.6000000,
    13, 1, 12.3924430,
    14, 1,  9.000,
    14, 2,  5.00,
    14, 3,  5.00,
    15, 1,  6.0000000,
    15, 2,  7.0000000,
    15, 3,  9.0000000,
    16, 1,  4.5936910,
    16, 2,  5.8657310,
    16, 3, 13.5573360,
    17, 1,  4.0000000,
    17, 2,  4.0000000,
    35, 1,  4.0000000,
    35, 2,  4.0000000,
    53, 1,  2.3000000,
    53, 2,  3.0000000,
)
MOPAC_IDEA_FN2[dat[0::3],dat[1::3]] = dat[2::3]

MOPAC_IDEA_FN3 = numpy.zeros((108,10))
dat = (
    1 , 1,  1.2000000,
    1 , 2,  1.8000000,
    1 , 3,  2.1000000,
    6 , 1,  1.6000000,
    6 , 2,  1.8500000,
    6 , 3,  2.0500000,
    6 , 4,  2.6500000,
    7 , 1,  1.5000000,
    7 , 2,  2.1000000,
    7 , 3,  2.4000000,
    8 , 1,  0.8479180,
    8 , 2,  1.4450710,
    9 , 1,  0.9300000,
    9 , 2,  1.6600000,
    13, 1,  2.0503940,
    14, 1,  0.911453,
    14, 2,  1.995569,
    14, 3,  2.990610,
    15, 1,  1.4743230,
    15, 2,  1.7793540,
    15, 3,  3.0065760,
    16, 1,  0.7706650,
    16, 2,  1.5033130,
    16, 3,  2.0091730,
    17, 1,  1.3000000,
    17, 2,  2.1000000,
    35, 1,  1.5000000,
    35, 2,  2.3000000,
    53, 1,  1.8000000,
    53, 2,  2.2400000,
)
MOPAC_IDEA_FN3[dat[0::3],dat[1::3]] = dat[2::3]
del(dat)


@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore(mol):
    assert(not mol.has_ecp())
    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_u.append(MOPAC_USS[z])
        else:
            basis_u.append(MOPAC_UPP[z])
    # U term
    hcore = numpy.diag(_to_ao_labels(mol, basis_u))

    # if method == 'INDO' or 'CINDO'
    #    # Nuclear attraction
    #    gamma = _get_gamma(mol)
    #    z_eff = mopac_param.CORE[atom_charges]
    #    vnuc = numpy.einsum('ij,j->i', gamma, z_eff)
    #    aoslices = mol.aoslice_by_atom()
    #    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
    #        idx = numpy.arange(p0, p1)
    #        hcore[idx,idx] -= vnuc[ia]

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        for ja in range(ia):
            w, e1b, e2a, enuc = _get_jk_2c_ints(mol, ia, ja)
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b
    return hcore

def _get_jk_2c_ints(mol, ia, ja):
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    ri = mol.atom_coord(ia) #?*lib.param.BOHR
    rj = mol.atom_coord(ja) #?*lib.param.BOHR
    w = numpy.zeros((10,10))
    e1b = numpy.zeros(10)
    e2a = numpy.zeros(10)
    enuc = numpy.zeros(1)
    AM1_MODEL = 2
    repp(zi, zj, ri, rj, w, e1b, e2a, enuc,
         MOPAC_ALP, MOPAC_DD, MOPAC_QQ, MOPAC_AM, MOPAC_AD, MOPAC_AQ,
         MOPAC_IDEA_FN1, MOPAC_IDEA_FN2, MOPAC_IDEA_FN3, AM1_MODEL)

    tril2sq = lib.square_mat_in_trilu_indices(4)
    w = w[:,tril2sq][tril2sq]
    e1b = e1b[tril2sq]
    e2a = e2a[tril2sq]

    if mopac_param.CORE[zj] <= 1:
        e2a = e2a[:1,:1]
        w = w[:,:,:1,:1]
    if mopac_param.CORE[zi] <= 1:
        e1b = e1b[:1,:1]
        w = w[:1,:1]
    # enuc from repp integrals is wrong due to the unit of MOPAC_IDEA_FN2 and
    # MOPAC_ALP
    return w, e1b, e2a, enuc[0]


@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk(mol, dm):
    dm = numpy.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = numpy.zeros_like(dm)
    vk = numpy.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints(z) for z in set(atom_charges)}

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = numpy.arange(p0, p1)
        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = numpy.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = numpy.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    for ia, (i0, i1) in enumerate(aoslices[:,2:]):
        w = _get_jk_2c_ints(mol, ia, ia)[0]
        vj[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            w = _get_jk_2c_ints(mol, ia, ja)[0]
            vj[:,i0:i1,i0:i1] += numpy.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += numpy.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += numpy.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += numpy.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    return vj, vk


def energy_nuc(mol):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    enuc = 0
    alp = MOPAC_ALP
    exp = numpy.exp
    gamma = mindo3._get_gamma(mol, MOPAC_AM)
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            rij = distances_in_AA[ia,ja]
            scale = 1. + exp(-alp[ni] * rij) + exp(-alp[nj] * rij)

            nt = ni + nj
            if (nt == 8 or nt == 9):
                if (ni == 7 or ni == 8):
                    scale += (rij - 1.) * exp(-alp[ni] * rij)
                if (nj == 7 or nj == 8):
                    scale += (rij - 1.) * exp(-alp[nj] * rij)
            enuc = mopac_param.CORE[ni] * mopac_param.CORE[nj] * gamma[ia,ja] * scale

            fac1 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[ni], exp(-MOPAC_IDEA_FN2[ni] * (rij - MOPAC_IDEA_FN3[ni])**2))
            fac2 = numpy.einsum('i,i->', MOPAC_IDEA_FN1[nj], exp(-MOPAC_IDEA_FN2[nj] * (rij - MOPAC_IDEA_FN3[nj])**2))
            enuc += mopac_param.CORE[ni] * mopac_param.CORE[nj] / rij * (fac1 + fac2)
    return enuc


def get_init_guess(mol):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = numpy.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = mopac_param.CORE[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return numpy.diag(dm_diag)


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    mol = mf._mindo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    e_ref = _get_reference_energy(mol)

    mf.e_heat_formation = e_tot * mopac_param.HARTREE2KCAL + e_ref
    logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
                 e_ref, mf.e_heat_formation)
    return e_tot.real


class RAM1(scf.hf.RHF):
    '''RHF-AM1 for closed-shell systems'''
    def __init__(self, mol):
        scf.hf.RHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mindo_mol)

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.hf.RHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        raise NotImplementedError


class UAM1(scf.uhf.UHF):
    '''UHF-AM1 for open-shell systems'''
    def __init__(self, mol):
        scf.uhf.UHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        self.nelec = self._mindo_mol.nelec
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.uhf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        dm = get_init_guess(self._mindo_mol) * .5
        return numpy.stack((dm,dm))

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.uhf.UHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        import umindo3_grad
        return umindo3_grad.Gradients(self)


def _make_mindo_mol(mol):
    assert(not mol.has_ecp())
    def make_sto_6g(n, l, zeta):
        es = mopac_param.gexps[(n, l)]
        cs = mopac_param.gcoefs[(n, l)]
        return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]

    def principle_quantum_number(charge):
        if charge < 3:
            return 1
        elif charge < 10:
            return 2
        elif charge < 18:
            return 3
        else:
            return 4

    mindo_mol = copy.copy(mol)
    atom_charges = mindo_mol.atom_charges()
    atom_types = set(atom_charges)
    basis_set = {}
    for charge in atom_types:
        n = principle_quantum_number(charge)
        l = 0
        sto_6g_function = make_sto_6g(n, l, mopac_param.ZS3[charge])
        basis = [sto_6g_function]

        if charge > 2:  # include p functions
            l = 1
            sto_6g_function = make_sto_6g(n, l, mopac_param.ZP3[charge])
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis
    mindo_mol.basis = basis_set

    z_eff = mopac_param.CORE[atom_charges]
    mindo_mol.nelectron = int(z_eff.sum() - mol.charge)

    mindo_mol.build(0, 0)
    return mindo_mol


def _to_ao_labels(mol, labels):
    ao_loc = mol.ao_loc
    degen = ao_loc[1:] - ao_loc[:-1]
    ao_labels = [[label]*n for label, n in zip(labels, degen)]
    return numpy.hstack(ao_labels)

def _get_beta0(atnoi,atnoj):
    "Resonanace integral for coupling between different atoms"
    return mopac_param.BETA3[atnoi-1,atnoj-1]

def _get_alpha(atnoi,atnoj):
    "Part of the scale factor for the nuclear repulsion"
    return mopac_param.ALP3[atnoi-1,atnoj-1]

def _get_jk_1c_ints(z):
    if z < 3:  # H, He
        j_ints = numpy.zeros((1,1))
        k_ints = numpy.zeros((1,1))
        j_ints[0,0] = mopac_param.GSSM[z]
    else:
        j_ints = numpy.zeros((4,4))
        k_ints = numpy.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3))
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2))

        j_ints[0,0] = mopac_param.GSSM[z]
        j_ints[0,1:] = j_ints[1:,0] = mopac_param.GSPM[z]
        j_ints[p_off_idx] = mopac_param.GP2M[z]
        j_ints[p_diag_idx] = mopac_param.GPPM[z]

        k_ints[0,1:] = k_ints[1:,0] = mopac_param.HSPM[z]
        k_ints[p_off_idx] = mopac_param.HP2M[z]
    return j_ints, k_ints


def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL


if __name__ == '__main__':
    mol = gto.M(atom='''O  0  0  0
                        H  0 -0.757  .587
                        H  0  0.757  .587''')

    mf = RAM1(mol).run(conv_tol=1e-6)
    print(mf.e_heat_formation)

    mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
    mf = UAM1(mol).run(conv_tol=1e-6)
    print(mf.e_heat_formation)



