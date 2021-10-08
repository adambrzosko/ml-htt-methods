#!/bin/bash

year=$1
dir=$2

OUTPUT="batch_annotate_parajob_${year}.sh"

echo "cd $PWD" > $OUTPUT

echo "export SCRAM_ARCH=slc6_amd64_gcc481" >> $OUTPUT
echo "eval \`scramv1 runtime -sh\`" >> $OUTPUT
echo "source ../scripts/setup_libs.sh" >> $OUTPUT

echo "ulimit -c 0" >> $OUTPUT
echo "inputNumber=\$SGE_TASK_ID" >> $OUTPUT

echo "export \"OMP_NUM_THREADS=1\"" >> $OUTPUT

# note the BDT is only trained for the tt channel at the moment so this channel is hard coded for now
# the name of the BDT training is also hard coded for now

echo "python annotate_file_split.py filelist/tmp_${year}_split/tt/x\${inputNumber} IC_01Jun2020 --path ${dir} --model_folder ./data_tauspinner_01Jun2020_${year}/ --output-folder ${dir} --training tauspinner --era ${year} --channel tt --config-training data_tauspinner_01Jun2020_${year}/tt_${year}_config_inc.yaml &> filelist/tmp_${year}_split/tt/\${inputNumber}.log" >> $OUTPUT 

chmod +x $OUTPUT

