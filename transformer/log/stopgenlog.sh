#!/bin/sh

killall collectl &

cat Net0 | head -n 4 > Net00
cat Net1 | head -n 4 > Net01

cat Net0 | grep eth0 >> Net00
cat Net1 | grep eth0 >> Net01
