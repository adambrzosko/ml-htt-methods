YEAR=$1
DIR=$2
CHAN=$3


./generate_parajob.sh $YEAR $DIR $CHAN

mkdir -p err && mkdir -p out

mkdir -p filelist/tmp_${YEAR}_split/ && mkdir -p filelist/tmp_${YEAR}_split/${CHAN}/

ls ${DIR}/{,*/}*_${CHAN}_*.root | awk -F "${DIR}" '{print $2}' | cut -d"/" -f2- > filelist/full_${CHAN}_${YEAR}.txt

cd filelist/tmp_${YEAR}_split/${CHAN}/ && python ../../../geteratejobs.py --filelist=../../full_${CHAN}_${YEAR}.txt --dir=$DIR && cd ../../../

max=$(ls filelist/tmp_${YEAR}_split/${CHAN}/x* | wc -l)
qsub -e err/ -o out/ -cwd -V -l h_rt=3:0:0 -q hep.q -t 1-${max}:1 batch_annotate_parajob_${CHAN}_${YEAR}.sh
