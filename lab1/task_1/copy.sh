#!/usr/bin/env bash

FILES=$3
for FILE in "${@:4}";
do
  FILES+=,$FILE
done

scp $1/\{$FILES\} $2
# ./copy.sh testuser2@20.61.118.247:copied testuser3@52.148.252.15:copied nowytest1.txt nowytest2.txt
