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

import os
import sys
import time
import json
import inspect
import functools
import traceback
from pathlib import Path
import numpy as np
from pyscf import lib

molecules_dir = os.path.abspath(f'{__file__}/../molecules')

def get_sys_info():
    sys_info = lib.format_sys_info()
    np_libs = np.__config__.blas_ilp64_opt_info
    np_blas = np_libs['libraries']
    sys_info.append(f'BLAS for NumPy: {", ".join(np_blas)}')

    try:
        import lief
        libdir = os.path.abspath(f'{lib.__file__}/..')
        # Shared libraries that PySCF links to
        if sys.platform == "darwin":
            binary = lief.parse(f'{libdir}/libnp_helper.dylib')
        else:
            binary = lief.parse(f'{libdir}/libnp_helper.so')
        pyscf_blas = [so for so in binary.libraries if 'mkl' in so or 'blas' in so]
        sys_info.append(f'BLAS for PySCF: {", ".join(pyscf_blas)}')
    except ImportError:
        pass

    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        sys_info.append('Brand: {0}'.format(info.get('brand_raw', '')))
        sys_info.append('Hz: {0}'.format(info.get('hz_actual_friendly', '')))
        sys_info.append('Arch: {0}'.format(info.get('arch', '')))
        sys_info.append('Bits: {0}'.format(info.get('bits', '')))
        sys_info.append('Count: {0}'.format(info.get('count', '')))
        sys_info.append('L1 Data Cache Size: {0}'.format(info.get('l1_data_cache_size', '')))
        sys_info.append('L1 Instruction Cache Size: {0}'.format(info.get('l1_instruction_cache_size', '')))
        sys_info.append('L2 Cache Size: {0}'.format(info.get('l2_cache_size', '')))
        sys_info.append('L2 Cache Line Size: {0}'.format(info.get('l2_cache_line_size', '')))
        sys_info.append('L2 Cache Associativity: {0}'.format(info.get('l2_cache_associativity', '')))
        sys_info.append('L3 Cache Size: {0}'.format(info.get('l3_cache_size', '')))
        sys_info.append('Stepping: {0}'.format(info.get('stepping', '')))
        sys_info.append('Model: {0}'.format(info.get('model', '')))
        sys_info.append('Family: {0}'.format(info.get('family', '')))
    except ImportError:
        pass

    # TODO: Add memory info

    sys_info = '\n'.join(sys_info)
    return sys_info

_tasks = {}

def benchmark(func_or_label, label=None, tags=None):
    '''
    Register benchmark tests. This function can be called as a function decorator:

        @benchmark('test I')
        def test_fn():
            ...

    or called directly to regster a function:

        for basis in config['basis']:
            def f():
                mf = pyscf.M(..., basis=basis).RHF().run()
                return mf.e_tot
            benchmark(f, label=f'RHF-{basis}')
    '''
    if callable(func_or_label):
        if label is None:
            mod = func_or_label.__module__
            fname = func_or_label.__name__
            label = f'{mod}.{fname}'
        if label in _tasks:
            raise RuntimeError(f'task {label} is already registered')
        _tasks[label] = func_or_label
        return func_or_label

    elif isinstance(func_or_label, str):
        return functools.partial(benchmark, label=func_or_label)

def exec_tasks(tasks):
    results = {}
    for name, f in tasks.items():
        print(f'Processing {name}')
        t0 = time.perf_counter()
        try:
            out = f()
        except Exception as e:
            print(e)
            traceback.print_stack()
            out = str(e)
        finally:
            t1 = time.perf_counter()
        print(f'{name}: {t1-t0:.3f}s')

        try:
            json.dumps(out)
        except Exception:
            out = None
        results[name] = {
            'result': out,
            'time': t1 - t0
        }
    return results

def get_caller_filename():
    frame = inspect.stack()[1]
    return frame.filename

def cli_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run benchmark and write timing data in JSON document"
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default=None,
        help=('Filename to save results in JSON format. '
              'By default, the module name is used as the filename for the output.')
    )
    parser.add_argument(
        '--print-summary',
        action='store_true',
    )
    return parser

def run(tasks=None, output_file=None, print_summary=False):
    if print_summary:
        # Ensure pandas library is installed
        import pandas as pd

    if output_file is None:
        fname = os.path.basename(get_caller_filename())
        output_file = Path(fname).with_suffix('.json')
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if tasks is None:
        tasks = _tasks
    if not tasks:
        print('Empty benchmark tasks.')
        sys.exit(1)
    print(f'Collected {len(tasks)} benchmark tests.')

    results = exec_tasks(tasks)

    if print_summary:
        print(get_sys_info())
        df = pd.DataFrame.from_dict(results)
        df['result'] = df['result'].apply(lambda x: 'N/A' if x is None else x)
        df['time'] = df['time'].apply(lambda x: f"{x:.2f} s")
        pd.set_option('display.max_rows', None) # Print all rows
        print(df)
        pd.reset_option('display.max_rows')

    with open(output_file, 'w') as f:
        json.dump({
            'sys_info': get_sys_info(),
            'timing': results
        }, f)
