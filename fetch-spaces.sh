#!/bin/sh
experiments=./experiments/

if [ $# -eq 2 ]; then
    rm -rf $experiments/$2
fi

cd $experiments
git clone $1
