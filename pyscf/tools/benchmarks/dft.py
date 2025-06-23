# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Benchmark tests for DFT and density fitting DFT methods. This module can be
executed fromm cli:

    python -m pyscf.tools.benchmarks.dft --save-json=pyscf-2.9-dft.json --print-summary=True
'''

import os
import pyscf
from pyscf.tools.benchmarks.utils import molecules_dir, benchmark, cli_parser, run

def run_rks(atom, basis, xc, auxbasis=None, with_grad=False):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    mf = mol.RKS(xc=xc)
    if auxbasis is not None:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.grids.atom_grid = (99,590)
    mf.nlcgrids.atom_grid = (50,194)
    mf.conv_tol = 1e-8
    mf.run()
    if with_grad:
        mf.Gradients().run()
    return mf.e_tot

def benchmark_df_rks():
    for xyz_file in [
        '020_Vitamin_C.xyz',
        '031_Inosine.xyz',
        '033_Bisphenol_A.xyz',
        '037_Mg_Porphin.xyz',
        '042_Penicillin_V.xyz',
        '045_Ochratoxin_A.xyz',
    ]:
        xyz_file = os.path.join(molecules_dir, 'organic', xyz_file)
        for xc in ['pbe', 'b3lyp', 'wb97m-v']:
            for basis in ['def2-tzvp']:
                mol_name = xyz_file.rsplit('.', 1)[0]
                benchmark(
                    lambda: run_rks(xyz_file, basis, xc, 'def2-universal-jkfit'),
                    label=f'DFRKS {mol_name} {xc}/{basis}')

def benchmark_rks_gradients():
    for xyz_file in [
        '020_Vitamin_C.xyz',
        '031_Inosine.xyz',
        '033_Bisphenol_A.xyz',
        '037_Mg_Porphin.xyz',
        '042_Penicillin_V.xyz',
        '045_Ochratoxin_A.xyz',
        '052_Cetirizine.xyz',
        '057_Tamoxifen.xyz',
        '066_Raffinose.xyz',
        '084_Sphingomyelin.xyz',
        '095_Azadirachtin.xyz',
        '113_Taxol.xyz',
    ]:
        xyz_file = os.path.join(molecules_dir, 'organic', xyz_file)
        for xc in ['pbe', 'b3lyp', 'wb97m-v']:
            for basis in ['def2-tzvp']:
                mol_name = xyz_file.rsplit('.', 1)[0]
                benchmark(
                    lambda: run_rks(xyz_file, basis, xc, with_grad=True),
                    label=f'RKS {mol_name} {xc}/{basis} Gradients',
                    tags=['slow'])

def main():
    parser = cli_parser()
    args = parser.parse_args()
    benchmark_df_rks()
    benchmark_rks_gradients()
    run(output_file=args.save_json, print_summary=args.print_summary)

if __name__ == '__main__':
    main()
