# a tool to parse the recipes/ folder from 
# https://github.com/QMCPACK/pseudopotentiallibrary
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

dir1='basis/ccecp-basis'
cd recipes
for ecp in 'ccECP' 'ccECP_He_core' 'ccECP_reg'; do
    dir="../${dir1}/${ecp}"
    mkdir -p ${dir}
    for basis in 'cc-pVDZ' 'cc-pVTZ' 'cc-pVQZ' 'cc-pV5Z' 'cc-pV6Z' 'aug-cc-pVDZ' 'aug-cc-pVTZ' 'aug-cc-pVQZ' 'aug-cc-pV5Z' 'aug-cc-pV6Z'; do
        basfile="${dir}/ccECP_${basis}.dat"
        echo "# ccECP basis sets" >> ${basfile}
        echo "# https://pseudopotentiallibrary.org/" > ${basfile}
        echo 'BASIS "ao basis" PRINT' >> ${basfile}
        for atom in *; do
            file="${atom}/${ecp}/${atom}.${basis}.nwchem"
            if [ -f $file ]; then
                echo '#BASIS SET' >> ${basfile}
                cat $file >> ${basfile}
            fi
        done
        echo 'END' >> ${basfile}
    done
    ecpfile="${dir}/ccECP.dat"
    echo "# ccECP effective core potentials" > ${ecpfile}
    echo "# https://pseudopotentiallibrary.org/" >> ${ecpfile}
    echo 'ECP' >> ${ecpfile}
    for atom in *; do
        file="${atom}/${ecp}/${atom}.ccECP.nwchem"
        if [ -f $file ]; then
            cat $file >> ${ecpfile}
        fi
    done
    echo 'END' >> ${ecpfile}

    # remove zero contraction coefficients from Li and Be in ccECP_reg
    sed -e '/Li s/ {N;/\n.*0\.000000/d}' -e '/Be s/ {N;/\n.*0\.000000/d}' ${ecpfile} > tmp
    mv tmp ${ecpfile}

    # hacky way to remove zero contraction coefficients (H,He)
    sed -e '/H S/d' -e '/He s/d' -e '/ 0\.00000/d' ${ecpfile} > tmp
    mv tmp ${ecpfile}
done


