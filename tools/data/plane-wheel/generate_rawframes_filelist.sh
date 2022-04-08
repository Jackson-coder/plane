#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py plane-wheel data/plane-wheel/rawframes/ --subset train --level 2 --format rawframes --shuffle
echo "Filelist for rawframes generated."

cd tools/data/plane-wheel/
