#!/bin/bash

#EXCLUDES="./tests/test_ewald.py"
EXCLUDES=""

unset OMP_NUM_THREADS
find . -name 'test*.py' -not -path './tests*' -not -path './examples*' | while read i
do
  if [ ! `echo $EXCLUDES | grep "$(basename i)"` ]; then
    python $i
  fi
done
