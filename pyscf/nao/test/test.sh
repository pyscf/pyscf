#!/bin/bash
#run it by : sh test.sh >& test.out 

date="$(date +'%d/%m/%Y')"
now=$(date +"%T")
echo "============================================================================"
echo "Starting TEST at $date --- $now, please wait...IT TAKES FEW MINUTES" 
echo "============================================================================"
LIST="$(ls test_0*.py)"

for i in $LIST; do
     if python "$i" ; then
         echo  "$i  ====> Command succeeded."
	 m=$((m+1))
     else
	 echo "============================================="
	 echo  "$i  ====> Command failed."
	 echo "============================================="
	 n=$((n+1))
	 arr += "$i"
     fi
done
echo "==================================================================="
echo "=============> Report: $m succeeded and $n failed.<================"
echo "==================FINISHED! AT $NOW"======================"         
echo "==================================================================="

