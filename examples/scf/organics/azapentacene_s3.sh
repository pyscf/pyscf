#!/bin/bash
#PBS -j oe
#PBS -l walltime=72:00:00
#PBS -l mem=30gb
#PBS -l nodes=1:ppn=8
#PBS -m n


SUBMIT_HOST=$PBS_O_HOST
SUBMIT_SERVER=$PBS_SERVER
SUBMIT_QUEUE=$PBS_O_QUEUE
SUBMIT_WORKDIR=$PBS_O_WORKDIR
JOBID=$PBS_JOBID
JOBNAME=$PBS_JOBNAME
QUEUE=$PBS_QUEUE
O_PATH=$PBS_O_PATH
O_HOME=$PBS_O_HOME
NODES=$(< $PBS_NODEFILE)
NODES_UNIQUE=$(echo "$NODES" | sort -u)
RETURN_VALUE=0
NODE_SCRATCHDIR="/lscratch/ccprak16/azapentacene_s3_$PBS_JOBID"
NODE_WORKDIR="/scratch/ccprak16/azapentacene_s3_$PBS_JOBID"

echo ------------------------------------------------------
echo "Job is running on nodes"
echo "$NODES" | sed 's/^/    /g'
echo ------------------------------------------------------
echo qsys: job was submitted from $SUBMIT_HOST
echo qsys: originating queue is $SUBMIT_QUEUE
echo qsys: executing queue is $QUEUE
echo qsys: original working directory is $SUBMIT_WORKDIR
echo qsys: job identifier is $JOBID
echo qsys: job name is $JOBNAME
echo qsys: current home directory is $O_HOME
echo qsys: PATH = $O_PATH
echo ------------------------------------------------------
echo
echo

echo "Setting up job..."
echo "Creating Working Directory..."
mkdir -p $NODE_WORKDIR
echo "Restarting bashrc..."
source /export/home/ccprak16/.bashrc
echo "Restarting conda..."
source /export/home/ccprak16/anaconda3/etc/profile.d/conda.sh
cd /export/home/ccprak16/anaconda3/envs
echo "Activating conda environment"
conda activate fdiis_pyscf
cd $PBS_O_WORKDIR
echo "Setting up PYTHONPATH..."
export PYTHONPATH='/export/home/ccprak16/Documents/Linus/FDIIS/pyscf_workspace/pyscf'
echo "Finished Setting up!"
echo


echo "Starting job execution..."
python azapentacene_s3.py > $NODE_WORKDIR/azapentacene_s3.py.out
echo "Finished job execution!"

echo "Copying files to final directory..."
cp $NODE_WORKDIR/azapentacene_s3.py.out $PBS_O_WORKDIR
echo "Finished copying!"

exit $RETURN_VALUE