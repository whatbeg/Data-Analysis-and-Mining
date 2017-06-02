#!/usr/bin/env bash

python SK_0.2.py --datastart 0 --dataend 10000 |& tee set1.log
python SK_0.2.py --datastart 10000 --dataend 20000 |& tee set2.log
python SK_0.2.py --datastart 20000 --dataend 30000 |& tee set3.log
python SK_0.2.py --datastart 30000 --dataend 40000 |& tee set4.log
python SK_0.2.py --datastart 40000 --dataend 50000 |& tee set5.log
python SK_0.2.py --datastart 50000 --dataend 60000 |& tee set6.log
