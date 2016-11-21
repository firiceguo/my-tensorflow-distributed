#!/bin/sh

dir=./log/1

> $dir/CPU0
> $dir/Mem0
> $dir/Net0
> $dir/Dsk0
> $dir/Nfs0

collectl -sC -oT > $dir/CPU0 &
collectl -sM -oT > $dir/Mem0 &
collectl -sN -oT > $dir/Net0 &
collectl -sD -oT > $dir/Dsk0 &
collectl -sF -oT > $dir/Nfs0 &

echo "Running..."

date > $dir/output-0
python $1 --job_name="ps" --task_index=0 >> $dir/output-0 &
python $1 --job_name="worker" --task_index=0 >> $dir/output-0
date >> $dir/output-0

killall collectl
