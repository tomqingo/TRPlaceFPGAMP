#!/bin/bash

designs=(171 172 175 176 180 181)
cnt=0
sample=25
thread=$1
start_id=$2
end_id=$3
log_dir=$4
spread_iter=$5
for design in ${designs[@]:$start_id:$end_id}
do
    for sampleid in {1..25};
    do
	cnt=$(( cnt + 1))
	echo $cnt/${#designs[@]} "Design_"$design
	python data_generation.py d$design --flow all -s $spread_iter -l $log_dir --sampleid $sampleid &
	if [ $cnt == $thread ]
	then
		wait
		cnt=0
	fi
    done
done
