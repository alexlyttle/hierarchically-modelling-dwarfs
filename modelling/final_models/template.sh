#!/bin/bash
#SBATCH --job-name star_NAME
#SBATCH --time 30:0
#SBATCH --qos bbdefault
#SBATCH --ntasks 4
#SBATCH --nodes 1
#SBATCH --account daviesgr-cartography
#SBATCH --mail-type NONE

set -e

module purge; module load bluebear
module load bear-apps/2019b
module load Python/3.7.4-GCCcore-8.3.0
module load foss/2019b

source ~/.virtualenvs/stellr/bin/activate

python -u PY_PATH NAME
