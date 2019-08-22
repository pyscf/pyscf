#!/bin/bash

ml Anaconda3/5.1.0

red=$( tput setaf 1 )
green=$( tput setaf 2 )
sab=$( tput setaf 4 )
white=$( tput setaf 7 )
NC=$( tput setaf 255 )

date="$(date +'%d/%m/%Y')"
now=$(date +"%T")


echo "${sab}============================================================================"
echo "${sab}Starting TEST at $date --- $now, please wait...IT TAKES A FEW MINS" 
echo "${sab}============================================================================${NC}"
printf "<p style="color:green">============================================================================<br />\
        Starting TEST at $date --- $now, please wait...IT TAKES A FEW MINS<br /> 
        ============================================================================<br />" >>failed.html

echo "${green}============================================="
echo "$i ====> Repository will be UPDATED!"
echo "=============================================${NC}"
echo "${white}"
git pull
git pull https://github.com/cfm-mpc/pyscf nao


LIST="$(ls test_*.py)"
err=("${arr[@]}")
m=0
n=0

for i in $LIST; do
    echo "${white}"
    if python "$i" ; then    
      printf "<p style="color:green">=============================================<br />\
      $i ====> TEST succeeded.<br />\
      =============================================<br />">> correct.html
      echo "${green}============================================="
      echo "$i ====> TEST succeeded."
      echo "=============================================${NC}"
      m=$((m+1))
      arr=( "${arr[@]}" "$i" )
    else
      printf "<p style="color:red">=============================================<br />\
	    $i ====> TEST failed.<br />\
	    =============================================<br />">>failed.html
	    echo "${red}============================================="
	    echo "$i ====> TEST failed."
	    echo "=============================================${NC}"
	    n=$((n+1))
      err+=( "$i" )
    fi

done 
now=$(date +"%T")
echo "${sab}=============> Report: $m succeeded and $n failed.<================"
echo "${red}FAILED TEST are: " 
for l in "${err[@]}"; do echo "$l" ; done
echo "${sab}=======================FINISHED! AT $now ==========================${NC}"
printf "<p style="color:green">==================> Report: $m succeeded and $n failed.<===================<br />\
        =======================FINISHED! AT $now ==========================">>failed.html


#mailing
SUBJECT="Testing result. Hostname: `/bin/hostname`;"
SUBJECT="$SUBJECT Date: `/bin/date`; User: `/usr/bin/whoami`."
EMAIL="ma.mansoury@gmail.com"
EMAILMESSAGE=./failed.html 
echo mutt -e 'set content_type="text/html"' "$EMAIL" -s "$SUBJECT" '<' "$EMAILMESSAGE"
mutt -e 'set content_type="text/html"' "$EMAIL" -s "$SUBJECT" <"$EMAILMESSAGE"


#moving data to home
mkdir REPORT
mv *.html ./REPORT/
mv REPORT $HOME/

git clean -f
