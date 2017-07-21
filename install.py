from __future__ import division
import numpy as np
import argparse
import os
import subprocess


def compile_C_code(verbose=0, build_type="Release", clean=0):
    """
        Compile automaticaly the C and Fortran code
        when installing the program with python setup.py
    """

    if build_type not in ["Release", "Debug", "None"]:
        raise ValueError("build type must be Release, Debug or None")

    if clean == 1:
        ret = subprocess.call("rm -rf lib/build", shell = True)
    try:
        os.mkdir("lib/build")
    except:
        print("lib/build")
    os.chdir("lib/build")

    cmd = "cmake -DCMAKE_BUILD_TYPE=" + build_type + " .."
    ret = subprocess.call(cmd, shell = True)
    if ret != 0:
        raise ValueError("cmake returned error {0}".format(ret))

    cmd = "make VERBOSE={0}".format(verbose)
    ret = subprocess.call(cmd, shell = True)
    if ret != 0:
        raise ValueError("make returned error {0}".format(ret))
    os.chdir("../..")



parser = argparse.ArgumentParser()
parser.add_argument('--command', type=str, default = "build", help="run: python setup.py <command>")
parser.add_argument('--compile', type=int, default = 0, help="compile C and Fortran code")
parser.add_argument('--clean', type=int, default = 0, help="compile C and Fortran code")
parser.add_argument('--build_type', type=str, default = "Release", help="compile code in Release or Debug mode")
parser.add_argument('--verbose', type=int, default = 0, help="verbosity for compilation")

args = parser.parse_args()

if args.compile==1:
    compile_C_code(verbose=args.verbose, build_type=args.build_type, clean= args.clean)

if args.command not in ["install", "build", "None"]:
    raise ValueError("command must be build or install")

if args.command != "None":
    cmd = "python setup.py " + args.command
    ret = subprocess.call(cmd, shell = True)
    if ret != 0:
        raise ValueError(cmd + " returned error {0}".format(ret))
