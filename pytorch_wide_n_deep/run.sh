#!/usr/bin/env bash

python usingpytorch.py |& tee wide_n_deep.log
python plot_log.py --file wide_n_deep.log
