#!/bin/sh
spaces_path=./lib/model/spaces
rm -rf $spaces_path

if [ $# -eq 0 ]; then
    cp -r lib/model/spaces_example $spaces_path
else
    git clone --depth 1 $1 $spaces_path
fi
