#!/bin/bash

fname=${1:-siesta.ANI}
suffi=${2:-0}
echo ".ANI file with MM geometries            : " $fname
echo "Suffix to choose geometries for QM part : " $suffi

mkdir -p xyz
echo "cp $fname xyz/"
cp $fname xyz/
echo "cd xyz"
cd xyz
LIT=`head -n1 $fname` 
NLINES_PER_FILE=$(expr $LIT + 2)
echo "rm x*"
rm x*
echo "split -l $NLINES_PER_FILE -d -a 6 $fname"
split -l $NLINES_PER_FILE -d -a 6 $fname
echo "xyz2fdf.py x0*$suffi > xyz2fdf.py.out"
xyz2fdf.py x0*$suffi &> xyz2fdf.py.out
cd ..
ls xyz/*.fdf
