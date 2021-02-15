from __future__ import division
import numpy as np
import subprocess

# The bash script for Torque
script = """#!/bin/bash  
#PBS -q parallel
#PBS -l nodes=1:ppn=6
#PBS -l mem=5gb
#PBS -l cput=400:00:00
#PBS -N calc_C60


ulimit -s unlimited

export NPROCS=`wc -l < $PBS_NODEFILE`
export OMP_NUM_THREADS=${NPROCS}
export PATH="/Path/To/Python/binary:$PATH" 
export LD_LIBRARY_PATH="/Path/To/Needed/Library:$LD_LIBRARY_PATH" # libxc, libcint, libxcfun
export PYTHONPATH="/Path/To/Pyscf:$PYTHONPATH"

# ASE necessay for using ase.units
#ASE
export ASE_HOME=/Path/To/ASE
export PYTHONPATH="${ASE_HOME}:$PYTHONPATH"
export PATH="${ASE_HOME}/tools:$PATH"

# you may need to change the job directory
export LSCRATCH_DIR=/scratch/$USER/jobs/$PBS_JOBID
mkdir -p $LSCRATCH_DIR
cd $PBS_O_WORKDIR

# load the right module depending your pyscfi/siesta compilation
# you better to use the same compiler for both program to avoid problems
ml purge
ml load intel/2015b FFTW/3.3.4-intel-2015b
"""

# the range of our 200 calculations
xyz_range = np.arange(0, 5000, 25)

start = 0
step = 4
end = 0

while end < xyz_range.size:
    
    end = start + step
    if end > xyz_range.size:
        end = xyz_range.size + 1

    calcs = xyz_range[start:end]
    include = ["calc_polarizability.py"]
    for i, xyz in enumerate(calcs[0:calcs.shape[0]]):
        if xyz < 10:
            num = "00000{0}".format(xyz)
        elif xyz < 100:
            num = "0000{0}".format(xyz)
        elif xyz < 1000:
            num = "000{0}".format(xyz)
        elif xyz < 10000:
            num = "00{0}".format(xyz)
        else:
            raise ValueError("xyz too large?? {0}".format(xyz))
        include.append("calc_"+num)

    fcalc = calcs[0]
    ecalc = calcs[calcs.shape[0]-1]+25

    lines = script
    for files in include:
        lines += "cp -r " + files + " $LSCRATCH_DIR\n"
        
    lines += "cd $LSCRATCH_DIR\n"
    lines += "./calc_polarizability.py --np ${NPROCS} " + "--start {0} --end {1} >& calc_{0}to{1}.out\n".format(fcalc, ecalc)

    lines += "export RESULTS_DIR=$PBS_O_WORKDIR\n"
    lines += "mkdir -p $RESULTS_DIR\n"
    lines += "cp -r * $RESULTS_DIR\n"
    fname = "run.calc_C60_{0}to_{1}.sh".format(fcalc, ecalc)
    f = open(fname, "w")
    f.write(lines) # write bash script
    f.close()
    start = end
    
    # submit job to Torque
    subprocess.call("qsub " + fname, shell=True)
