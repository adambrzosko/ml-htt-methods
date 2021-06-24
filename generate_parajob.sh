#!/bin/bash

year=$1
dir=$2
chan=$3

OUTPUT="batch_annotate_parajob_${chan}_${year}.sh"

echo "cd $PWD" > $OUTPUT

echo "export SCRAM_ARCH=slc6_amd64_gcc481" >> $OUTPUT
echo "eval \`scramv1 runtime -sh\`" >> $OUTPUT
echo "source ../scripts/setup_libs.sh" >> $OUTPUT

echo "ulimit -c 0" >> $OUTPUT
echo "inputNumber=\$SGE_TASK_ID" >> $OUTPUT

echo "export \"OMP_NUM_THREADS=1\"" >> $OUTPUT

if [ $chan == "em" ]; then
  config="lowmass_config_em.yaml"
else
  config="lowmass_config.yaml"
fi

echo "python annotate_file_split.py filelist/tmp_${year}_split/${chan}/x\${inputNumber} lowmass_v1 --path ${dir}  --output-folder ${dir} --training lowmass --era ${year} --channel ${chan} --config-training ${config} &> filelist/tmp_${year}_split/${chan}/\${inputNumber}.log" >> $OUTPUT 

chmod +x $OUTPUT

