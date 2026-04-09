#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$REPO_ROOT/src/ames.py" --iseq1 protein:randoms:100:evolv \
                                -pm pmo \
                                -ps 100 -ng 10000 \
                                --engine simulacrum \
                                --nobackup \
                                -b0 0.1 \
                                -ann \
                                -bt 10.0 \
                                -ann_s 250 \
                                -ann_e 7500 \
                                -o "$1"


python "$REPO_ROOT/src/visualames.py" -l "$1"/progress.log 

