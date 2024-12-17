# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
SMD solvent model, copied from GPU4PYSCF with modification for CPU
'''

import numpy as np
from pyscf import lib, gto
from pyscf.data import radii
from pyscf.dft import gen_grid
from pyscf.solvent import pcm
from pyscf.solvent._attach_solvent import _for_scf
from pyscf.lib import logger

@lib.with_doc(_for_scf.__doc__)
def smd_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = SMD(mf.mol)
    return _for_scf(mf, solvent_obj, dm)

# Inject PCM to SCF, TODO: add it to other methods later
from pyscf import scf
scf.hf.RHF.SMD = smd_for_scf
scf.uhf.UHF.SMD = smd_for_scf
hartree2kcal = 627.509451

# database from https://comp.chem.umn.edu/solvation/mnsddb.pdf
# solvent name: [n, n25, alpha, beta, gamma, epsilon, phi, psi)
solvent_db = {
    '1,1,1-trichloroethane':[1.4379, 1.4313, 0.0, 0.09, 36.24, 7.0826, 0.0, 0.60],
    '1,1,2-trichloroethane':[1.4717, 1.4689, 0.13, 0.13, 48.97, 7.1937, 0.0, 0.60],
    '1,2,4-trimethylbenzene':[1.5048, 1.5024, 0.0, 0.19, 42.03, 2.3653, 0.667, 0.0],
    '1,2-dibromoethane':[1.5387, 1.5364, 0.10, 0.17, 56.93, 4.9313, 0.0, 0.5],
    '1,2-dichloroethane':[1.4448, 1.4425, 0.10, 0.11, 45.86, 10.125, 0.0, 0.5],
    '1,2-ethanediol':[1.4318, 1.4306, 0.58, 0.78, 69.07, 40.245, 0.0, 0.5],
    '1,4-dioxane':[1.4224, 1.4204, 0.00, 0.64, 47.14, 2.2099, 0.0, 0.0],
    '1-bromo-2-methylpropane':[1.4348, 1.4349, 0.00, 0.12, 34.69, 7.7792, 0.0, 0.2],
    '1-bromooctane':[1.4524, 1.4500, 0.0, 0.12, 41.28, 5.0244, 0.0, 0.111],
    '1-bromopentane':[1.4447, 1.4420, 0.00, 0.12, 38.7, 6.269, 0.0, 0.167],
    '1-bromopropane':[1.4343, 1.4315, 0.0, 0.12, 36.36, 8.0496, 0.0, 0.250],
    '1-butanol':[1.3993, 1.3971, 0.37, 0.48, 35.88, 17.332, 0.0, 0.0],
    '1-chlorohexane':[1.4199, -1, 0.0, 0.10, 37.03, 5.9491, 0.0, 0.143],
    '1-chloropentane':[1.4127, 1.4104, 0.0, 0.1, 35.12, 6.5022, 0.0, 0.167],
    '1-chloropropane':[1.3879, 1.3851, 0.0, 0.1, 30.66, 8.3548, 0.0, 0.25],
    '1-decanol':[1.4372, 1.4353, 0.37, 0.48, 41.04, 7.5305, 0.0, 0.0],
    '1-fluorooctane':[1.3935, 1.3927, 0.0, 0.10, 33.92, 3.89, 0.0, 0.111],
    '1-heptanol':[1.4249, 1.4224, 0.37, 0.48, 38.5, 11.321, 0.0, 0.0],
    '1-hexanol':[1.4178, 1.4162, 0.37, 0.48, 37.15, 12.51, 0.0, 0.0],
    '1-hexene':[1.3837, 1.385, 0.00, 0.07, 25.76, 2.0717, 0.0, 0.0],
    '1-hexyne':[1.3989, 1.3957, 0.12, 0.10, 28.79, 2.615, 0.0, 0.0],
    '1-iodobutane':[1.5001, 1.4958, 0.00, 0.15, 40.65, 6.173, 0.0, 0.0],
    '1-iodohexadecane':[1.4806, -1, 0.00, 0.15, 46.48, 3.5338, 0.0, 0.0],
    '1-iodopentane':[1.4959, -1, 0.00, 0.15, 41.56, 5.6973, 0.0, 0.0],
    '1-iodopropane':[1.5058, 1.5027, 0.00, 0.15, 41.45, 6.9626, 0.0, 0.0],
    '1-nitropropane':[1.4018, 1.3996, 0.00, 0.31, 43.32, 23.73, 0.0, 0.0],
    '1-nonanol':[1.4333, 1.4319, 0.37, 0.48, 40.14, 8.5991, 0.0, 0.0],
    '1-octanol':[1.4295, 1.4279, 0.37, 0.48, 39.01, 9.8629, 0.0, 0.0],
    '1-pentanol':[1.4101, 1.4080, 0.37, 0.48, 36.5, 15.13, 0.0, 0.0],
    '1-pentene':[1.3715, 1.3684, 0.00, 0.07, 22.24, 1.9905, 0.0, 0.0],
    '1-propanol':[1.3850, 1.3837, 0.37, 0.48, 33.57, 20.524, 0.0, 0.0],
    '2,2,2-trifluoroethanol':[1.2907, -1, 0.57, 0.25, 42.02, 26.726, 0.0, 0.5],
    '2,2,4-trimethylpentane':[1.3915, 1.3889, 0.00, 0.00, 26.38, 1.9358, 0.0, 0.0],
    '2,4-dimethylpentane':[1.3815, 1.3788, 0.0, 0.00, 25.42, 1.8939, 0.0, 0.0],
    '2,4-dimethylpyridine':[1.5010, 1.4985, 0.0, 0.63, 46.86, 9.4176, 0.625, 0.0],
    '2,6-dimethylpyridine':[1.4953, 1.4952, 0.0, 0.63, 44.64, 7.1735, 0.625, 0.0],
    '2-bromopropane':[1.4251, 1.4219, 0.00, 0.14, 33.46, 9.3610, 0.0, 0.25],
    '2-butanol':[1.3978, 1.3949, 0.33, 0.56, 32.44, 15.944, 0.0, 0.0],
    '2-chlorobutane':[1.3971, 1.3941, 0.00, 0.12, 31.1, 8.3930, 0.0, 0.2],
    '2-heptanone':[1.4088, 1.4073, 0.0, 0.51, 37.6, 11.658, 0.0, 0.0],
    '2-hexanone':[1.4007, 1.3987, 0.0, 0.51, 36.63, 14.136, 0.0, 0.0],
    '2-methoxyethanol':[1.4024, 1.4003, 0.30, 0.84, 44.39, 17.2, 0.0, 0.0],
    '2-methyl-1-propanol':[1.3955, 1.3938, 0.37, 0.48, 32.38, 16.777, 0.0, 0.0],
    '2-methyl-2-propanol':[1.3878, 1.3852, 0.31, 0.60, 28.73, 12.47, 0.0, 0.0],
    '2-methylpentane':[1.3715, 1.3687, 0.0, 0.00, 24.3, 1.89, 0.0, 0.0],
    '2-methylpyridine':[1.4957, 1.4984, 0.0, 0.58, 47.5, 9.9533, 0.714, 0.0],
    '2-nitropropane':[1.3944, 1.3923, 0.00, 0.33, 42.16, 25.654, 0.00, 0.0],
    '2-octanone':[1.4151, 1.4133, 0.00, 0.51, 37.29, 9.4678, 0.0, 0.0],
    '2-pentanone':[1.3895, 1.3885, 0.00, 0.51, 33.46, 15.200, 0.0, 0.0],
    '2-propanol':[1.3776, 1.3752, 0.33, 0.56, 30.13, 19.264, 0.0, 0.0],
    '2-propen-1-ol':[1.4135, 0.00, 0.38, 0.48, 36.39, 19.011, 0.0, 0.0],
    'E-2-pentene':[1.3793, 1.3761, 0.00, 0.07, 23.62, 2.051, 0.0, 0.0],
    '3-methylpyridine':[1.5040, 1.5043, 0.0, 0.54, 49.61, 11.645, 0.714, 0.0],
    '3-pentanone':[1.3924, 1.3905, 0.00, 0.51, 35.61, 16.78, 0.0, 0.0],
    '4-heptanone':[1.4069, 1.4045, 0.00, 0.51, 35.98, 12.257, 0.0, 0.0],
    '4-methyl-2-pentanone':[1.3962, 1.394, 0.0, 0.51, 33.83, 12.887, 0.0, 0.0],
    '4-methylpyridine':[1.5037, 1.503, 0.0, 0.54, 50.17, 11.957, 0.714, 0.0],
    '5-nonanone':[1.4195, 0.00, 0.0, 0.51, 37.83, 10.6, 0.0, 0.0],
    'acetic acid':[1.3720, 1.3698, 0.61, 0.44, 39.01, 6.2528, 0.0, 0.0],
    'acetone':[1.3588, 1.3559, 0.04, 0.49, 33.77, 20.493, 0.0, 0.0],
    'acetonitrile':[1.3442, 1.3416, 0.07, 0.32, 41.25, 35.688, 0.0, 0.0],
    'acetophenone':[1.5372, 1.5321, 0.00, 0.48, 56.19, 17.44, 0.667, 0.0],
    'aniline':[1.5863, 1.5834, 0.26, 0.41, 60.62, 6.8882, 0.857, 0.0],
    'anisole':[1.5174, 1.5143, 0.0, 0.29, 50.52, 4.2247, 0.75, 0.0],
    'benzaldehyde':[1.5463, 1.5433, 0.0, 0.39, 54.69, 18.220, 0.857, 0.0],
    'benzene':[1.5011, 1.4972, 0.0, 0.14, 40.62, 2.2706, 1.0, 0.0],
    'benzonitrile':[1.5289, 1.5257, 0.0, 0.33, 55.83, 25.592, 0.75, 0.0],
    'benzylalcohol':[1.5396, 1.5384, 0.33, 0.56, 52.96, 12.457, 0.75, 0.0],
    'bromobenzene':[1.5597, 1.5576, 0.0, 0.09, 50.72, 5.3954, 0.857, 0.143],
    'bromoethane':[1.4239, 1.4187, 0.0, 0.12, 34.0, 9.01, 0.0, 0.333],
    'bromoform':[1.6005, 1.5956, 0.15, 0.06, 64.58, 4.2488, 0.0, 0.75],
    'butanal':[1.3843, 1.3766, 0.0, 0.45, 35.06, 13.45, 0.0, 0.0],
    'butanoic acid':[1.3980, 1.3958, 0.60, 0.45, 37.49, 2.9931, 0.0, 0.0],
    'butanone':[1.3788, 1.3764, 0.00, 0.51, 34.5, 18.246, 0.0, 0.0],
    'butanonitrile':[1.3842, 1.382, 0.0, 0.36, 38.75, 24.291, 0.0, 0.0],
    'butylethanoate':[1.3941, 1.3923, 0.0, 0.45, 35.81, 4.9941, 0.0, 0.0],
    'butylamine':[1.4031, 1.3987, 0.16, 0.61, 33.74, 4.6178, 0.0, 0.0],
    'n-butylbenzene':[1.4898, 1.4874, 0.0, 0.15, 41.33, 2.36, 0.6, 0.0],
    'sec-butylbenzene':[1.4895, 1.4878, 0.0, 0.16, 40.35, 2.3446, 0.60, 0.0],
    'tert-butylbenzene':[1.4927, 1.4902, 0.0, 0.16, 39.78, 2.3447, 0.6, 0.0],
    'carbon disulfide':[1.6319, 1.6241, 0.0, 0.07, 45.45, 2.6105, 0.0, 0.0],
    'carbon tetrachloride':[1.4601, 1.4574, 0.00, 0.00, 38.04, 2.2280, 0.0, 0.8],
    'chlorobenzene':[1.5241, 1.5221, 0.0, 0.07, 47.48, 5.6968, 0.857, 0.143],
    'chloroform':[1.4459, 1.4431, 0.15, 0.02, 38.39, 4.7113, 0.0, 0.75],
    'a-chlorotoluene':[1.5391, 0.0, 0.0, 0.33, 53.04, 6.7175, 0.75, 0.125],
    'o-chlorotoluene':[1.5268, 1.5233, 0.00, 0.07, 47.43, 4.6331, 0.75, 0.125],
    'm-cresol':[1.5438, 1.5394, 0.57, 0.34, 51.37, 12.44, 0.75, 0.0],
    'o-cresol':[1.5361, 1.5399, 0.52, 0.30, 53.11, 6.76, 0.75, 0.0],
    'cyclohexane':[1.4266, 1.4235, 0.00, 0.00, 35.48, 2.0165, 0.0, 0.0],
    'cyclohexanone':[1.4507, 1.4507, 0.00, 0.56, 49.76, 15.619, 0.00, 0.0],
    'cyclopentane':[1.4065, 1.4036, 0.00, 0.00, 31.49, 1.9608, 0.0, 0.0],
    'cyclopentanol':[1.4530, -1, 0.32, 0.56, 46.8, 16.989, 0.0, 0.0],
    'cyclopentanone':[1.4366, 1.4347, 0.00, 0.52, 47.21, 13.58, 0.0, 0.0],
    'decalin (cis/trans mixture)':[1.4753, 1.472, 0.00, 0.00, 43.82, 2.196, 0.0, 0.0],
    'cis-decalin':[1.4810, 1.4788, 0.00, 0.00, 45.45, 2.2139, 0.0, 0.0],
    'n-decane':[1.4102, 1.4094, 0.00, 0.00, 33.64, 1.9846, 0.0, 0.0],
    'dibromomethane':[1.5420, 1.5389, 0.10, 0.10, 56.21, 7.2273, 0.0, 0.667],
    'butylether':[1.3992, 1.3968, 0.00, 0.45, 35.98, 3.0473, 0.0, 0.0],
    'o-dichlorobenzene':[1.5515, 1.5491, 0.00, 0.04, 52.72, 9.9949, 0.75, 0.25],
    'E-1,2-dichloroethene':[1.4454, 1.4435, 0.09, 0.05, 37.13, 2.14, 0.0, 0.5],
    'Z-1,2-dichloroethene':[1.4490, 1.4461, 0.11, 0.05, 39.8, 9.2, 0.0, 0.5],
    'dichloromethane':[1.4242, 1.4212, 0.10, 0.05, 39.15, 8.93, 0.0, 0.667],
    'diethylether':[1.3526, 1.3496, 0.00, 0.41, 23.96, 4.2400, 0.0, 0.0],
    'diethylsulfide':[1.4430, 1.4401, 0.00, 0.32, 35.36, 5.723, 0.0, 0.0],
    'diethylamine':[1.3864, 1.3825, 0.08, 0.69, 28.57, 3.5766, 0.0, 0.0],
    'diiodomethane':[1.7425, 1.738, 0.05, 0.23, 95.25, 5.32, 0.0, 0.0],
    'diisopropyl ether':[1.3679, 1.3653, 0.00, 0.41, 24.86, 3.38, 0.0, 0.0],
    'cis-1,2-dimethylcyclohexane':[1.4360, 1.4336, 0.00, 0.00, 36.28, 2.06, 0.0, 0.0],
    'dimethyldisulfide':[1.5289, 1.522, 0.00, 0.28, 48.06, 9.6, 0.0, 0.0],
    'N,N-dimethylacetamide':[1.4380, 1.4358, 0.00, 0.78, 47.62, 37.781, 0.0, 0.0],
    'N,N-dimethylformamide':[1.4305, 1.4280, 0.00, 0.74, 49.56, 37.219, 0.0, 0.0],
    'dimethylsulfoxide':[1.4783, 1.4783, 0.00, 0.88, 61.78, 46.826, 0.0, 0.0],
    'diphenylether':[1.5787, -1, 0.00, 0.20, 38.5, 3.73, 0.923, 0.0],
    'dipropylamine':[1.4050, 1.4018, 0.08, 0.69, 32.11, 2.9112, 0.0, 0.0],
    'n-dodecane':[1.4216, 1.4151, 0.00, 0.00, 35.85, 2.0060, 0.0, 0.0],
    'ethanethiol':[1.4310, 1.4278, 0.00, 0.24, 33.22, 6.667, 0.0, 0.0],
    'ethanol':[1.3611, 1.3593, 0.37, 0.48, 31.62, 24.852, 0.0, 0.0],
    'ethylethanoate':[1.3723, 1.3704, 0.00, 0.45, 33.67, 5.9867, 0.0, 0.0],
    'ethylmethanoate':[1.3599, 1.3575, 0.00, 0.38, 33.36, 8.3310, 0.0, 0.0],
    'ethylphenylether':[1.5076, 1.5254, 0.00, 0.32, 46.65, 4.1797, 0.667, 0.0],
    'ethylbenzene':[1.4959, 1.4932, 0.00, 0.15, 41.38, 2.4339, 0.75, 0.0],
    'fluorobenzene':[1.4684, 1.4629, 0.00, 0.10, 38.37, 5.42, 0.857, 0.143],
    'formamide':[1.4472, 1.4468, 0.62, 0.60, 82.08, 108.94, 0.0, 0.0],
    'formicacid':[1.3714, 1.3693, 0.75, 0.38, 53.44, 51.1, 0.0, 0.0],
    'n-heptane':[1.3878, 1.3855, 0.00, 0.00, 28.28, 1.9113, 0.0, 0.0],
    'n-hexadecane':[1.4345, 1.4325, 0.00, 0.00, 38.93, 2.0402, 0.0, 0.0],
    'n-hexane':[1.3749, 1.3722, 0.00, 0.00, 25.75, 1.8819, 0.0, 0.0],
    'hexanoicacid':[1.4163, 1.4146, 0.60, 0.45, 39.65, 2.6, 0.0, 0.0],
    'iodobenzene':[1.6200, 1.6172, 0.00, 0.12, 55.72, 4.5470, 0.857, 0.0],
    'iodoethane':[1.5133, 1.5100, 0.00, 0.15, 40.96, 7.6177, 0.0, 0.0],
    'iodomethane':[1.5380, 1.5270, 0.00, 0.13, 43.67, 6.8650, 0.0, 0.0],
    'isopropylbenzene':[1.4915, 1.4889, 0.00, 0.16, 39.85, 2.3712, 0.667, 0.0],
    'p-isopropyltoluene':[1.4909, 1.4885, 0.00, 0.19, 38.34, 2.2322, 0.600, 0.0],
    'mesitylene':[1.4994, 1.4968, 0.00, 0.19, 39.65, 2.2650, 0.667, 0.0],
    'methanol':[1.3288, 1.3265, 0.43, 0.47, 31.77, 32.613, 0.0, 0.0],
    'methylbenzoate':[1.5164, 1.5146, 0.00, 0.46, 53.5, 6.7367, 0.600, 0.0],
    'methylbutanoate':[1.3878, 1.3847, 0.00, 0.45, 35.44, 5.5607, 0.0, 0.0],
    'methylethanoate':[1.3614, 1.3589, 0.00, 0.45, 35.59, 6.8615, 0.0, 0.0],
    'methylmethanoate':[1.3433, 1.3415, 0.00, 0.38, 35.06, 8.8377, 0.0, 0.0],
    'methylpropanoate':[1.3775, 1.3742, 0.00, 0.45, 35.18, 6.0777, 0.0, 0.0],
    'N-methylaniline':[1.5684, 1.5681, 0.17, 0.43, 53.11, 5.9600, 0.75, 0.0],
    'methylcyclohexane':[1.4231, 1.4206, 0.00, 0.00, 33.52, 2.024, 0.0, 0.0],
    'N-methylformamide(E/Zmixture)':[1.4319, 1.4310, 0.40, 0.55, 55.44, 181.56, 0.0, 0.0],
    'nitrobenzene':[1.5562, 1.5030, 0.00, 0.28, 57.54, 34.809, 0.667, 0.0],
    'nitroethane':[1.3917, 1.3897, 0.02, 0.33, 46.25, 28.29, 0.0, 0.0],
    'nitromethane':[1.3817, 1.3796, 0.06, 0.31, 52.58, 36.562, 0.0, 0.0],
    'o-nitrotoluene':[1.5450, 1.5474, 0.0, 0.27, 59.12, 25.669, 0.6, 0.0],
    'n-nonane':[1.4054, 1.4031, 0.0, 0.0, 32.21, 1.9605, 0.0, 0.0],
    'n-octane':[1.3974, 1.3953, 0.0, 0.0, 30.43, 1.9406, 0.0, 0.0],
    'n-pentadecane':[1.4315, 1.4298, 0.0, 0.0, 38.34, 2.0333, 0.0, 0.0],
    'pentanal':[1.3944, 1.3917, 0.0, 0.4, 36.62, 10.0, 0.0, 0.0],
    'n-pentane':[1.3575, 1.3547, 0.0, 0.0, 22.3, 1.8371, 0.0, 0.0],
    'pentanoic acid':[1.4085, 1.4060, 0.60, 0.45, 38.4, 2.6924, 0.0, 0.0],
    'pentyl ethanoate':[1.4023, -1.0, 0.0, 0.45, 36.23, 4.7297, 0.0, 0.0],
    'pentylamine':[1.448, 1.4093, 0.16, 0.61, 35.54, 4.2010, 0.0, 0.0],
    'perfluorobenzene':[1.3777, 1.3761, 0.00, 0.00, 31.74, 2.029, 0.5, 0.5],
    'propanal':[1.3636, 1.3593, 0.00, 0.45, 32.48, 18.5, 0.0, 0.0],
    'propanoic acid':[1.3869, 1.3848, 0.60, 0.45, 37.71, 3.44, 0.0, 0.0],
    'propanonitrile':[1.3655, 1.3633, 0.02, 0.36, 38.5, 29.324, 0.0, 0.0],
    'propyl ethanoate':[1.3842, 1.3822, 0.0, 0.45, 34.26, 5.5205, 0.0, 0.0],
    'propylamine':[1.3870, 1.3851, 0.16, 0.61, 31.31, 4.9912, 0.0, 0.0],
    'pyridine':[1.5095, 1.5073, 0.0, 0.52, 52.62, 12.978, 0.833, 0.0],
    'tetrachloroethene':[1.5053, 1.5055, 0.0, 0.0, 45.19, 2.268, 0.0, 0.667],
    'tetrahydrofuran':[1.4050, 1.4044, 0.0, 0.48, 39.44, 7.4257, 0.0, 0.0],
    'tetrahydrothiophene-S,S-dioxide':[1.4833, -1.0, 0.0, 0.88, 87.49, 43.962, 0.0, 0.0],
    'tetralin':[1.5413, 1.5392, 0.0, 0.19, 47.74, 2.771, 0.6, 0.0],
    'thiophene':[1.5289, 1.5268, 0.0, 0.15, 44.16, 2.7270, 0.8, 0.0],
    'thiophenol':[1.5893, 1.580, 0.09, 0.16, 55.24, 4.2728, 0.857, 0.0],
    'toluene':[1.4961, 1.4936, 0.0, 0.14, 40.2, 2.3741, 0.857, 0.0],
    'trans-decalin':[1.4695, 1.4671, 0.0, 0.0, 42.19, 2.1781, 0.0, 0.0],
    'tributylphosphate':[1.4224, 1.4215, 0.0, 1.21, 27.55, 8.1781, 0.0, 0.0],
    'trichloroethene':[1.4773, 1.4556, 0.08, 0.03, 41.45, 3.422, 0.0, 0.6],
    'triethylamine':[1.4010, 1.3980, 0.0, 0.79, 29.1, 2.3832, 0.0, 0.0],
    'n-undecane':[1.4398, 1.4151, 0.0, 0.0, 34.85, 1.991, 0.0, 0.0],
    'water':[1.3328, 1.3323, 0.82, 0.35, -1.0, 78.355, -1.0, -1.0],
    'xylene (mixture)':[1.4995, 1.4969, 0.0, 0.16, 41.38, 2.3879, 0.75, 0.0],
    'm-xylene':[1.4972, 1.4946, 0.0, 0.16, 40.98, 2.3478, 0.75, 0.0],
    'o-xylene':[1.5055, 1.5029, 0.0, 0.16, 42.83, 2.5454, 0.75, 0.0],
    'p-xylene':[1.4958, 1.4932, 0.0, 0.16, 40.32, 2.2705, 0.75, 0.0],
    '':[0]*8
}

def smd_radii(alpha):
    '''
    eq. (16)
    use smd radii if defined
    use Bondi radii if defined
    use 2.0 otherwise
    '''
    radii_table = radii.VDW.copy() * radii.BOHR
    radii_table[1] = 1.20
    radii_table[6] = 1.85
    radii_table[7] = 1.89
    if alpha >= 0.43:
        r = 1.52
    else:
        r = 1.52 + 1.8 * (0.43 - alpha)
    radii_table[8] = r
    radii_table[9] = 1.73
    radii_table[14] = 2.47
    radii_table[15] = 2.12
    radii_table[16] = 2.49
    radii_table[17] = 2.38
    #radii_table[35] = 3.06 # original SMD
    # following value from SMD18
    # https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.201803652
    radii_table[35] = 2.60
    radii_table[53] = 2.74
    return radii_table/radii.BOHR

import ctypes
from pyscf.lib import load_library
try:
    libsolvent = load_library('libsolvent')
except (IOError, NameError):
    libsolvent = None

def get_cds_legacy(smdobj):
    if libsolvent is None:
        raise RuntimeError(
            'SMD module is not available. '
            'You can compile this module with cmake option "-DENABLE_SMD=ON"')

    mol = smdobj.mol
    natm = mol.natm
    soln, _, sola, solb, solg, _, solc, solh = smdobj.solvent_descriptors
    #symbols = [mol.atom_s(ia) for ia in range(mol.natm)]
    charges = np.asarray(mol.atom_charges(), dtype=np.int32, order='F')
    coords = np.asarray(mol.atom_coords(unit='B'), dtype=np.float64, order='C')
    icds = 1 if smdobj.solvent.upper() == 'WATER' else 2
    dcds = np.empty([natm,3])
    mnsol_interface =  libsolvent.mnsol_interface_

    double_ndptr = np.ctypeslib.ndpointer(dtype=np.float64)
    int_ndptr = np.ctypeslib.ndpointer(dtype=np.int32)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    int_ptr = ctypes.POINTER(ctypes.c_int)

    mnsol_interface.argtypes = [
        double_ndptr, int_ndptr,
        int_ptr,
        double_ptr, double_ptr, double_ptr, double_ptr, double_ptr, double_ptr,
        int_ptr,
        double_ptr, double_ptr, double_ndptr]
    natm = ctypes.byref(ctypes.c_int(natm))
    icds = ctypes.byref(ctypes.c_int(icds))
    soln = ctypes.byref(ctypes.c_double(soln))
    sola = ctypes.byref(ctypes.c_double(sola))
    solb = ctypes.byref(ctypes.c_double(solb))
    solg = ctypes.byref(ctypes.c_double(solg))
    solc = ctypes.byref(ctypes.c_double(solc))
    solh = ctypes.byref(ctypes.c_double(solh))
    gcds = ctypes.c_double()
    areacds = ctypes.c_double()

    mnsol_interface(coords, charges,
                    natm,
                    sola, solb, solc, solg, solh, soln,
                    icds,
                    ctypes.byref(gcds), ctypes.byref(areacds), dcds)
    return gcds.value / hartree2kcal, dcds

class SMD(pcm.PCM):
    _keys = {
        'intopt', 'method', 'e_cds', 'solvent_descriptors', 'r_probe', 'sasa_ng'
    }
    def __init__(self, mol, solvent=''):
        super().__init__(mol)
        self.vdw_scale = 1.0
        self.sasa_ng = 590 # quadrature grids for calculating SASA
        self.r_probe = 0.4/radii.BOHR
        self.method = 'SMD'  # use IEFPCM for electrostatic
        if solvent not in solvent_db:
            raise RuntimeError(f'{solvent} is not available in SMD')
        self._solvent = solvent
        self.solvent_descriptors = solvent_db[solvent]
        self.radii_table = smd_radii(self.solvent_descriptors[2])
        self.e_cds = None

    def build(self, ng=None):
        if self.radii_table is None:
            vdw_scale = self.vdw_scale
            self.radii_table = vdw_scale * pcm.modified_Bondi + self.r_probe
        mol = self.mol
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = pcm.gen_surface(mol, rad=self.radii_table, ng=ng)
        self._intermediates = {}
        F, A = pcm.get_F_A(self.surface)
        D, S = pcm.get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.eps
        f_epsilon = (epsilon - 1.0)/(epsilon + 1.0)
        DA = D*A
        DAS = np.dot(DA, S)
        K = S - f_epsilon/(2.0*np.pi) * DAS
        R = -f_epsilon * (np.eye(K.shape[0]) - 1.0/(2.0*np.pi)*DA)
        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        int2c2e = mol._add_suffix('int2c2e')
        fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        self.v_grids_n = np.dot(atom_charges, v_ng)

    @property
    def solvent(self):
        return self._solvent

    @solvent.setter
    def solvent(self, solvent):
        self._solvent = solvent
        self.solvent_descriptors = solvent_db[solvent]
        self.radii_table = smd_radii(self.solvent_descriptors[2])
        self.eps = self.solvent_descriptors[5]

    @property
    def sol_desc(self):
        return self.solvent_descriptors

    @sol_desc.setter
    def sol_desc(self, values):
        '''
        format of sol desc
        [n, n25, alpha, beta, gamma, epsilon, phi, psi]
        '''
        assert len(values) == 8
        self.solvent_descriptors = values
        self.radii_table = smd_radii(self.solvent_descriptors[2])
        self.eps = values[5]

    def dump_flags(self, verbose=None):
        n, _, alpha, beta, gamma, _, phi, psi = self.solvent_descriptors
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'frozen = %s'       , self.frozen)
        logger.info(self, '---------- SMD solvent descriptors -------')
        logger.info(self, f'n     = {n}')
        logger.info(self, f'alpha = {alpha}')
        logger.info(self, f'beta  = {beta}')
        logger.info(self, f'gamma = {gamma}')
        logger.info(self, f'phi   = {phi}')
        logger.info(self, f'psi   = {psi}')
        logger.info(self, '--------------------- end ----------------')
        logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        logger.info(self, 'radii_table %s', self.radii_table*radii.BOHR)
        if self.atom_radii:
            logger.info(self, 'User specified atomic radii %s', str(self.atom_radii))
        return self

    def get_cds(self):
        return get_cds_legacy(self)[0]

    def nuc_grad_method(self, grad_method):
        from pyscf.solvent.grad import smd as smd_grad
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from pyscf import scf
        if isinstance(grad_method.base, (scf.hf.RHF, scf.uhf.UHF)):
            return smd_grad.make_grad_object(grad_method)
        else:
            raise RuntimeError('Only SCF gradient is supported')

    def Hessian(self, hess_method):
        from pyscf.solvent.hessian import smd as smd_hess
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from pyscf import scf
        if isinstance(hess_method.base, (scf.hf.RHF, scf.uhf.UHF)):
            return smd_hess.make_hess_object(hess_method)
        else:
            raise RuntimeError('Only SCF gradient is supported')

    def reset(self, mol=None):
        super().reset(mol)
        self.e_cds = None
        return self
