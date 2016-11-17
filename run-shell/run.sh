#!/bin/sh

dir=./log

> $dir/CPU
> $dir/Mem
> $dir/Net
> $dir/Dsk
> $dir/Nfs

collectl -sC -oT > $dir/CPU &
collectl -sM -oT > $dir/Mem &
collectl -sN -oT > $dir/Net &
collectl -sD -oT > $dir/Dsk &
collectl -sF -oT > $dir/Nfs &

echo "Running..."

date > output-single
python $1 >> output-single
date >> output-single

killall collectl
