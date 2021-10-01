#!/usr/bin/bash
folder=$1

mkdir $folder/safe
cd $folder/split

rm Add_BDTOddEven_*
rm ../Add_BDTOddEven_*
yourfilename=`ls ../*tt*`
#echo $yourfilename
i=0
for eachfile in $yourfilename
do #echo $eachfile
    VAR=$eachfile
    VAR2=${VAR##*/}
    NAME=${VAR2%.*}"_*"
    echo "$NAME"
    OUTPUT="Add_BDTOddEven_${i}.sh"
    echo "cd ${folder}/split" >> $OUTPUT
#note, above / is not necessary if your folder already ends with /
    echo "hadd $VAR2 $NAME" >> $OUTPUT
    echo "mv ${VAR2} ../safe/" >> $OUTPUT
    chmod +x $OUTPUT
    i=$(($i + 1))
done
#HToTauTauUncorrelatedDecay_Filtered_tt_2018.root' | cut -d . -f 1

