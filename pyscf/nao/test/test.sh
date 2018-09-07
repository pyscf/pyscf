#!/bin/bash
#run it by : sh test.sh >& test.out 
LIST="$(ls test_*.py)"
for i in $LIST; do
     if python "$i" ; then
         echo  "$i \t ====> Command succeeded."
     else
         echo  "$i \t ====> Command failed."
     fi  
done

