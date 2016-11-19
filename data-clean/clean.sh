#!/bin/sh

# $1 is Net;
# $2 is output;

cat $1 | grep eth0 > $1-eth0
cat $1 | grep lo > $1-lo

cat $2 | grep Accuracy > $2-acc 
