#!/usr/bin/env bash

python SK_0.2.py --seed 0 |& tee seed0.log
python SK_0.2.py --seed 1 |& tee seed1.log
python SK_0.2.py --seed 12 |& tee seed12.log
python SK_0.2.py --seed 123 |& tee seed123.log
python SK_0.2.py --seed 1234 |& tee seed1234.log
python SK_0.2.py --seed 12345 |& tee seed12345.log
python SK_0.2.py --seed 123456 |& tee seed123456.log
