#!/bin/bash
#SBATCH --job-name isoclassify
#SBATCH --time 1:0:0
#SBATCH --qos bbdefault
#SBATCH --ntasks 6
#SBATCH --nodes 1
#SBATCH --account daviesgr-cartography
#SBATCH --constraint cascadelake
#SBATCH --mail-type ALL

set -e

module purge; module load bluebear
module load bear-apps/2019b
module load IPython/7.9.0-foss-2019b-Python-3.7.4

source ~/.virtualenvs/python3.7-base/bin/activate

# RUN isoclassify direct method https://github.com/alexlyttle/isoclassify (fork)
# REQUIRES isoclassify to be installed and environment variables set
# INPUT the following global variables
SUFFIX=DR14_IRFM

N_JOBS=4  # Number of cores (or threads) to run on (note, up to 4GB RAM usage per job)
INPUT_CSV=$PHD/kepler-dwarfs/data/isoclassify/isoclassify_inputs_${SUFFIX}.csv
OUTPUT_CSV=$PHD/kepler-dwarfs/data/isoclassify/isoclassify_outputs_${SUFFIX}.csv
BASEOUTDIR=$PHD/isoclassify_output/kepler-dwarfs_${SUFFIX}  # Full path to save individual star outputs

isoclassify multiproc direct $N_JOBS $INPUT_CSV $OUTPUT_CSV --baseoutdir $BASEOUTDIR --plot save-png
