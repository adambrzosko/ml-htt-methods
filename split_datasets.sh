#!/usr/bin/bash

#This script will split the datasets held in the folder you give it and save the splits inside a new sub-folder called split, that is so that you don't destror your datasets accidently. It first creates the jobs: "Add_BDTOddEven_${i}.sh" and then runs them 

folder=$1
cd $folder
rm Add_BDT*

yourfilenames=`ls *tt*`
i=0
mkdir split
for eachfile in $yourfilenames
do echo $eachfile 
    i=$(($i + 1))
    echo $i
    OUTPUT="Add_BDTOddEven_${i}.sh"
    echo "cd $folder" >> $OUTPUT
    echo "export SCRAM_ARCH=slc6_amd64_gcc481" >> $OUTPUT
    echo "eval \`scramv1 runtime -sh\`" >> $OUTPUT
    echo "source ../../scripts/setup_libs.sh" >> $OUTPUT

    echo "ulimit -c 0" >> $OUTPUT
    echo "inputNumber=\$SGE_TASK_ID" >> $OUTPUT

    echo "export \"OMP_NUM_THREADS=1\"" >> $OUTPUT

    echo "./../../ml-htt-methods/split_root.py ${eachfile} BDTOddEven" >> $OUTPUT
    chmod +x $OUTPUT
    #qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q $OUTPUT
    #rm Add_*
   #echo $i
done
 
#python split_root.py $filename


##qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q ./call_fit.py ../output/outputTTMT/GluGluHToTauTau_M-125_tt_2018_5.root


