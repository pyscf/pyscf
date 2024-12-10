#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

"""
Verify Examples
===============

Author: Matthew R. Hennefarth

Script used to automatically run and verify PySCF example codes terminate
successfully. For any job that does not terminate normally, the stderr of the
example will be printed to the output. This script will exit with 0 only if all
examples terminate normally. 

Initially introduced in [PR 2379](https://github.com/pyscf/pyscf/pull/2379). 

Usage
-------------

From the main pyscf repository directory, the tests can be run as 
```sh
./tools/verify_examples.py examples
```
This will run all example files (which can be very long). To run only a subset
of examples, provide instead a path to a subdirectory. For example, to run only
the example files in `pyscf/examples/gto` the command
```sh
./tools/verify_examples.py examples/gto
```
It is also possible to run the examples in parallel using the `-j` or `--jobs`
flag (this is similar to make). As an example, to run the jobs in parallel over
4 threads,
```sh
./tools/verify_examples.py -j 8
```
Note that the environmental variable such as `OMP_NUM_THREADS` should be set to
an appropriate value such that number of jobs * OMP_NUM_THREADS does not exceed
the maximum number of cores on the computer.

"""

import os
import sys
import time
import subprocess
import argparse
import logging

import multiprocessing as mp
from glob import glob
from enum import Enum

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

logger = logging.getLogger()


class StdOutFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.ERROR


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(StdOutFilter())

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)

logger.handlers = []
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)


class ANSIColors(Enum):
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"


def colorize(text, color):
    if sys.stdout.isatty():
        return f"\033[{color.value}{text}{ANSIColors.RESET.value}"
    else:
        return text


class Status(Enum):
    OK = colorize("ok", ANSIColors.GREEN)
    FAIL = colorize("FAILED", ANSIColors.RED)


def get_path(p):
    if not os.path.isdir(p):
        raise ValueError("Path does not point to directory")

    if os.path.basename(p) == "examples":
        return p

    if os.path.isdir(os.path.join(p, "examples")):
        return os.path.join(p, "examples")

    return p


class ExampleResults:
    def __init__(self):
        self.common_prefix = ""
        self.failed_examples = []
        self.passed = 0
        self.failed = 0
        self.filtered = 0
        self.time = 0.0
        self.status = Status.OK


def run_example(progress, nexamples, example, failed_examples, common_prefix):
    idx, lock = progress

    status = Status.OK
    directory = os.path.dirname(example)
    try:
        subprocess.run(
            ["python3", os.path.basename(example)],
            cwd=directory,
            capture_output=False,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        status = Status.FAIL
        failed_examples.append((example, e.stderr))

    with lock:
        idx.value += 1
        percent = int(100 * (idx.value) / nexamples)

    message = (
        f"[{percent:3}%]: {os.path.relpath(example, common_prefix)} ... {status.value}"
    )
    logger.info(message)


def run_examples(example_path, num_threads):
    examples = [
        y for x in os.walk(example_path) for y in glob(os.path.join(x[0], "*.py"))
    ]
    # remove symlinks?
    # examples = list(set([os.path.realpath(e) for e in examples]))

    examples = sorted(examples, key=lambda e: e.split("/"))

    results = ExampleResults()
    results.common_prefix = os.path.dirname(os.path.commonpath(examples))
    results.filtered = 0

    with mp.Manager() as manager:
        failed_examples = manager.list()
        progress = (manager.Value("i", 0), manager.Lock())

        logger.info("")
        logger.info(f"running {len(examples)} examples")
        tic = time.perf_counter()
        with mp.Pool(num_threads) as pool:
            pool.starmap(
                run_example,
                [
                    (
                        progress,
                        len(examples),
                        example,
                        failed_examples,
                        results.common_prefix,
                    )
                    for example in examples
                ],
            )
        results.time = time.perf_counter() - tic
        results.failed_examples = list(failed_examples)

    results.failed = len(results.failed_examples)
    results.passed = len(examples) - results.failed
    results.status = Status.FAIL if results.failed else Status.OK

    return results


def log_failures(results):
    logger.info("")
    logger.info("failures: ")
    logger.info("")

    for e, msg in results.failed_examples:
        logger.info(f"---- {os.path.relpath(e, results.common_prefix)} stderr ----")
        logger.info(msg)

    logger.info("")
    logger.info("failures:")
    for e, _ in results.failed_examples:
        logger.info(f"    {os.path.relpath(e, results.common_prefix)}")


def main():
    parser = argparse.ArgumentParser(description="Verify pyscf examples")
    parser.add_argument(
        "path",
        type=str,
        default="examples",
        help="Path to examples directory (default: ./)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel threads (default: 1)",
    )
    args = parser.parse_args()

    example_path = get_path(args.path)

    results = run_examples(example_path, args.jobs)

    if results.status is Status.FAIL:
        log_failures(results)

    logger.info("")
    logger.info(
        f"example results: {results.status.value}. {results.passed} passed; {results.failed} failed; {results.filtered} filtered out; finished in {results.time:.2f}s"
    )
    logger.info("")

    if results.status is Status.OK:
        sys.exit(0)
    else:
        logger.error(
            f"{ANSIColors.RED.value}error{ANSIColors.RESET.value}: examples failed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
