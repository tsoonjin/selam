#!/bin/bash
t=1;
for i in `ls $1/*.jpg | sort -g`; do
    printf -v newName "$1/tmp/%08d.jpg" $t
    cp $i $newName;
    let t=t+1;
done
