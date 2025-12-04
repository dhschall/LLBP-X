#!/bin/bash

# MIT License
#
# Copyright (c) 2025 David Schall
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


set -x -e



OUT=results_g5/


BRMODELS=""
BRMODELS="${BRMODELS} TSL64k"
BRMODELS="${BRMODELS} LLBP"
BRMODELS="${BRMODELS} LLBPX"



for model in $BRMODELS; do

    ## Create output directory
    OUTDIR="${OUT}/${model}/"

    ./build/ARM/gem5.opt --outdir=$OUTDIR se-llbp.py --bp=$model

done


for model in $BRMODELS; do

    ## Create output directory
    OUTDIR="${OUT}/${model}/"

    MISP=$(awk '/.+mispredictDueToPredictor_0.+DirectCond/ {print $2}' "${OUTDIR}/stats.txt")
    echo "Mispredictions $model: $MISP"

done




echo "Simulation complete."
