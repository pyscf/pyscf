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
import functools
import traceback
import io
import contextlib
from pathlib import Path
import numpy as np
from pyscf import lib

molecules_dir = os.path.abspath(f'{__file__}/../molecules')

def _numpy_blas_info():
    if hasattr(np.__config__, 'CONFIG'): # 2.0
        np_blas = np.__config__.CONFIG['Build Dependencies']['blas']['name']
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
            lines = buf.getvalue().splitlines()
        for i, line in enumerate(lines):
            if line.startswith('blas'):
                break
        else:
            return None

        for line in lines[i+1:]:
            if not line.startswith(' '):
                return None
            keys = line.split(' = ')
            if keys[0].strip() == 'libraries':
                np_blas = keys[1]
                break
    return np_blas

def get_sys_info():
    '''Collect OS, libraries, and runtime environment information'''
    sys_info = lib.format_sys_info()
    np_blas = _numpy_blas_info()
    if np_blas:
        sys_info.append(f'BLAS for NumPy: {np_blas}')

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
        sys_info.append('Count: {0}'.format(info.get('count', '')))
        sys_info.append('L1 Data Cache Size: {0}'.format(info.get('l1_data_cache_size', '')))
        sys_info.append('L1 Instruction Cache Size: {0}'.format(info.get('l1_instruction_cache_size', '')))
        sys_info.append('L2 Cache Size: {0}'.format(info.get('l2_cache_size', '')))
        sys_info.append('L2 Cache Line Size: {0}'.format(info.get('l2_cache_line_size', '')))
        sys_info.append('L2 Cache Associativity: {0}'.format(info.get('l2_cache_associativity', '')))
        sys_info.append('L3 Cache Size: {0}'.format(info.get('l3_cache_size', '')))
    except ImportError:
        pass

    # TODO: Add memory info

    sys_info = '\n'.join(sys_info)
    return sys_info

_tasks = {}

def benchmark(func_or_label, label=None, tags=None):
    '''
    Register a benchmark test.

    Args:
        label (str): Unique name for the test. Defaults to module.function_name
        tags (list[str]): To filter benchmark tests from the command line.

    This function can be used as a decorator:

        @benchmark('test I')
        def test_fn():
            return result

    Or called directly to register functions:

        for basis in config['basis']:
            def f():
                mf = pyscf.M(..., basis=basis).RHF().run()
                return mf.e_tot
            benchmark(f, label=f'RHF-{basis}')

    If the function for benchmark returns a value, the result will be JSON
    serialized and stored in the output file, if applicable. Therefore, it is
    recommended to return JSON-serializable objects from the function. If the
    function raises errors, the error status will be saved as the result of the
    function.
    '''
    if callable(func_or_label):
        if label is None:
            mod = func_or_label.__module__
            if mod == '__main__':
                mod = os.path.basename(get_caller_filename())[:-3]
            fname = func_or_label.__name__
            label = f'{mod}.{fname}'
        if label in _tasks:
            raise RuntimeError(f'task {label} is already registered')
        _tasks[label] = (func_or_label, tags)
        return func_or_label

    elif isinstance(func_or_label, str):
        return functools.partial(benchmark, label=func_or_label)

def exec_tasks(tasks):
    results = {}
    for name, (f, tags) in tasks.items():
        print(f'Processing {name}')
        t0 = time.perf_counter()
        try:
            # save the output as a dict
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
            'time': t1 - t0,
            'tags': tags,
        }
    return results

def get_caller_filename():
    import __main__
    return os.path.abspath(__main__.__file__)

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
    # TODO: task filter
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
        print()
        print(get_sys_info())
        print()
        output = {}
        for name, val in results.items():
            if isinstance(val['result'], dict):
                output[name] = {'time': val['time'], **val['result']}
            else:
                output[name] = {'time': val['time'], 'result': val['results']}
        df = pd.DataFrame.from_dict(output, orient='index')
        df['time'] = df['time'].apply(lambda x: f"{x:.2f} s")

        pd.set_option('display.max_rows', None) # Print all rows
        print(df)
        pd.reset_option('display.max_rows')

    with open(output_file, 'w') as f:
        json.dump({
            'sys_info': get_sys_info(),
            'timing': results
        }, f)
