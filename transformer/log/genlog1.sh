#!/bin/sh

> CPU1
> Mem1
> Net1
> Dsk1
> Nfs1

collectl -sC -oT > CPU1 &
collectl -sM -oT > Mem1 &
collectl -sN -oT > Net1 &
collectl -sD -oT > Dsk1 &
collectl -sF -oT > Nfs1 &
