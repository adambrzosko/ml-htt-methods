i=0
folder=$1
cd $folder/split
rm Add_*
yourfilenames=`ls *tt*`

for eachfile in $yourfilenames
do echo $eachfile 
    i=$(($i + 1))
    echo $i
    OUTPUT="Add_BDTOddEven_${i}.sh"
    echo "export SCRAM_ARCH=slc6_amd64_gcc481" >> $OUTPUT
    echo "eval \`scramv1 runtime -sh\`" >> $OUTPUT
    echo "cd ${folder}/split" >> $OUTPUT
    echo "source ../../../scripts/setup_libs.sh" >> $OUTPUT

    echo "ulimit -c 0" >> $OUTPUT
    echo "inputNumber=\$SGE_TASK_ID" >> $OUTPUT

    echo "export \"OMP_NUM_THREADS=1\"" >> $OUTPUT

    echo "./../../../ml-htt-methods/call_fit.py ${eachfile}" >> $OUTPUT
    chmod +x $OUTPUT
    #qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q $OUTPUT
    #rm Add_*
   #echo $i
done


