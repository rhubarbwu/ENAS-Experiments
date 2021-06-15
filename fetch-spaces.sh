#!/bin/sh
spaces_path=./lib/model/spaces
rm -rf $spaces_path
git clone $1 $spaces_path
