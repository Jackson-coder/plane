#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py plane-wheel data/plane-wheel/videos/ --level 2 --format videos --shuffle
echo "Filelist for videos generated."

cd tools/data/plane-wheel/
