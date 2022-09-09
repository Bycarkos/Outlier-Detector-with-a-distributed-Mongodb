#!/bin/bash

for i in {1..8}
do

        ssh-copy-id student@mongo-${i}.grup01.bdnr
done
