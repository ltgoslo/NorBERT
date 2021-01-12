#!/bin/bash

module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load NLPL-PyTorch/1.6.0-gomkl-2019b-Python-3.7.4
module load NLPL-transformers/4.1.1-gomkl-2019b-Python-3.7.4

python sanity_check.py
