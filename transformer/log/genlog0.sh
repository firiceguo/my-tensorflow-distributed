#!/bin/sh

> CPU0
> Mem0
> Net0
> Dsk0
> Nfs0

collectl -sC -oT > CPU0 &
collectl -sM -oT > Mem0 &
collectl -sN -oT > Net0 &
collectl -sD -oT > Dsk0 &
collectl -sF -oT > Nfs0 &
