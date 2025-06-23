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
Benchmark example. This module can be executed fromm cli:

    python -m pyscf.tools.benchmarks.h2o --save-json=pyscf-2.9-h2o.json --print-summary=True
'''

import os
import pyscf
from pyscf.tools.benchmarks.utils import molecules_dir, benchmark, cli_parser, run

@benchmark
def water_hf():
    mol = pyscf.M(atom=f'{molecules_dir}/water_clusters/002.xyz', basis='cc-pvdz', verbose=0)
    mf = mol.RHF()
    counts = 0
    def scf_iter_counts(envs):
        nonlocal counts
        counts += 1
    mf.callback = scf_iter_counts
    mf.run()
    return {'E': mf.e_tot, 'SCF iters': counts}

@benchmark
def water_tdhf():
    mol = pyscf.M(atom=f'{molecules_dir}/water_clusters/002.xyz', basis='cc-pvdz', verbose=0)
    mf = mol.RHF().run()
    td = mf.TDHF()
    td.run()
    return {'E': mf.e_tot}

def main():
    parser = cli_parser()
    args = parser.parse_args()
    run(output_file=args.save_json, print_summary=args.print_summary)

if __name__ == '__main__':
    main()
