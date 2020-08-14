#!/bin/bash

dir=/vols/build/cms/dw515/new_tes/CMSSW_8_0_25/src/UserCode/ICHiggsTauTau/Analysis/HiggsTauTauRun2/

cd /vols/build/cms/dw515/embed_fix/CMSSW_10_2_19/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval `scramv1 runtime -sh`
source $dir/scripts/setup_libs.sh
# for keras others use
# conda activate mlFramework
# source ~/.profile

cd $dir/ml-htt-methods
ulimit -c 0
inputNumber=$SGE_TASK_ID

# tt 
export OMP_NUM_THREADS=1
python annotate_file_split_18.py filelist/tmp_2018_split/tt/x${inputNumber} IC_01Jun2020 --model_folder ./data_tauspinner_01Jun2020_2018/ --output-folder /vols/cms/dw515/Offline/output/SM/cpprod_2018/ --training tauspinner --era 2018 --channel tt --config-training data_tauspinner_01Jun2020_2018/tt_2018_config_inc.yaml &> filelist/tmp_2018_split/tt/${inputNumber}.log 

