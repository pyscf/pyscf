#!/bin/bash
#
#PBS -N fe-ni-s4-cn2-1
#PBS -j oe
#PBS -l walltime=86400
#PBS -l mem=110100480000b
#PBS -l vmem=110100480000b
#PBS -m n
#PBS -l nodes=1:ppn=8
#
###################################
#
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
NODE_WORKDIR="/scratch/ccprak16/fe-ni-s4-cn2-1_$PBS_JOBID"
NODE_SCRATCHDIR="/lscratch/ccprak16/fe-ni-s4-cn2-1_$PBS_JOBID"
#
###################################
#
 print_info() {
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
}

stage_in() {
    rm -f "$SUBMIT_WORKDIR/job_not_successful"

    echo "Calculation working directory: $NODE_WORKDIR"
    echo "            scratch directory: $NODE_SCRATCHDIR"

    # create workdir and cd to it.
    if ! mkdir -m700 -p $NODE_SCRATCHDIR $NODE_WORKDIR; then
        echo "Could not create scratch($NODE_SCRATCHDIR) or workdir($NODE_WORKDIR)" >&2
        exit 1
    fi
    cd $NODE_WORKDIR

    echo
    echo ------------------------------------------------------
    echo
}

stage_out() {
    if [ "$RETURN_VALUE" != "0" ]; then
        touch "$SUBMIT_WORKDIR/job_not_successful"
    fi

    echo
    echo ------------------------------------------------------
    echo

    echo "Final files in $SUBMIT_WORKDIR:"
    (
        cd $SUBMIT_WORKDIR
        ls -l | sed 's/^/    /g'
    )

    echo
    echo "More files can be found in $NODE_WORKDIR and $NODE_SCRATCHDIR on"
    echo "$NODES_UNIQUE" | sed 's/^/    /g'
    echo
    echo "Sizes of these files:"

    if echo "$NODE_SCRATCHDIR"/* | grep -q "$NODE_SCRATCHDIR/\*$"; then
        # no files in scratchdir:
        du -shc * | sed 's/^/    /g'
    else
        du -shc * "$NODE_SCRATCHDIR"/* | sed 's/^/    /g'
    fi

    echo
    echo "If you want to delete these, run:"
    for node in $NODES_UNIQUE; do
        echo "    ssh $node rm -r \"$NODE_WORKDIR\" \"$NODE_SCRATCHDIR\""
    done
}

handle_error() {
    # Make sure this function is only called once
    # and not once for each parallel process
    trap ':' 2 9 15

    echo
    echo "#######################################"
    echo "#-- Early termination signal caught --#"
    echo "#######################################"
    echo
    error_hooks
    stage_out
}

payload_hooks() {
:
for FILEORDIRPATH in $PBS_O_WORKDIR/fe-ni-s4-cn2-1.in; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$PBS_O_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$NODE_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$NODE_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $PBS_O_WORKDIR/potential.pot; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$PBS_O_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$NODE_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$NODE_WORKDIR/$DIR"
    fi
done


export QCSCRATCH="$NODE_SCRATCHDIR"
/export/home/ccprak16/bin/versions/q-chem/qchem-5.2 -nt 8 "fe-ni-s4-cn2-1.in" "fe-ni-s4-cn2-1.out"
RETURN_VALUE=$?

# check if job terminated successfully
if ! tail -n 30 "fe-ni-s4-cn2-1.out" | grep -q "Thank you very much for using Q-Chem.  Have a nice day."; then
    RETURN_VALUE=1
fi

for FILEORDIRPATH in $NODE_WORKDIR/fe-ni-s4-cn2-1.out; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/fe-ni-s4-cn2-1.in.fchk; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/plots; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/fe-ni-s4-cn2-1.out.plots; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/cap_adc_*_*.data; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/Epsilon.data; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done
for FILEORDIRPATH in $NODE_WORKDIR/PEQS.data; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done


}

error_hooks() {
:
for FILEORDIRPATH in $NODE_WORKDIR/fe-ni-s4-cn2-1.out; do
    FILEORDIR=$(echo "$FILEORDIRPATH" | sed -e s@^$NODE_WORKDIR//*@@)
    if [ -r "$FILEORDIRPATH" ]; then 
        CPARGS="--dereference" 
        [ -d "$FILEORDIRPATH" ] && CPARGS="--recursive"
        DIR=$(dirname "$FILEORDIR")
        mkdir -p "$PBS_O_WORKDIR/$DIR"
        cp $CPARGS "$FILEORDIRPATH" "$PBS_O_WORKDIR/$DIR"
    fi
done

}
#
###################################
#
# Run the stuff:

print_info
stage_in

# If catch signals 2 9 15, run this function:
trap 'handle_error' 2 9 15

payload_hooks
stage_out
exit $RETURN_VALUE
