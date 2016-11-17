#!/bin/sh

dir=./log/1

> $dir/CPU1
> $dir/Mem1
> $dir/Net1
> $dir/Dsk1
> $dir/Nfs1

collectl -sC -oT > $dir/CPU1 &
collectl -sM -oT > $dir/Mem1 &
collectl -sN -oT > $dir/Net1 &
collectl -sD -oT > $dir/Dsk1 &
collectl -sF -oT > $dir/Nfs1 &

echo "Running..."

date > $dir/output-1
python $1 --job_name="worker" --task_index=1 >> $dir/output-1
date >> $dir/output-1

killall collectl

