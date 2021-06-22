## HiggsTauTau CP ML methods
Repository for creating training datasets from ROOT files 
and then train on with an algorithm of choice (XGBoost, keras, sklearn etc) / or implement others.

### Git instructions

`git@github.com:danielwinterbottom/ml-htt-methods.git`
`git checkout lowmass` 

### Train

`python train_lowmass.py -c ${chan} -y ${year} -s ${sign}`

sign selects whether you train on even or odd events - we train seperatly on both to make sure we never train on the same events that we use in the analysis (in signal / background templates)   

### Annotate

To annotate files ROOT files with trained (XBG) model follow these steps:

- choose the year you are running for and directory you are adding the BDT scores for

  `YEAR=2018`
  `DIR=/vols/cms/dw515/Offline/output/MSSM/mssm_2018_v2_lowmass/`
  `CHAN=tt`

- make directroies and generate your job scripts 

  `./generate_parajob.sh $YEAR $DIR $CHAN`

  `mkdir -p err && mkdir -p out`

  `mkdir -p filelist/tmp_${YEAR}_split/ && mkdir -p filelist/tmp_${YEAR}_split/${CHAN}/`

  `ls ${DIR}/{,*/}*_${CHAN}_*.root | awk -F "${DIR}" '{print $2}' | cut -d"/" -f2- > filelist/full_${CHAN}_${YEAR}.txt`

  `cd filelist/tmp_${YEAR}_split/${CHAN}/ && python ../../../geteratejobs.py --filelist=../../full_${CHAN}_${YEAR}.txt --dir=$DIR && cd ../../../`

- submit the jobs to the batch

  `max=$(ls filelist/tmp_${YEAR}_split/${CHAN}/x* | wc -l)`
  `qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 1-${max}:1 batch_annotate_parajob_${CHAN}_${YEAR}.sh`

