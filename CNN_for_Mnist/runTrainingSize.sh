#!/usr/bin/env bash

python SK_s.py --SKs 0.2 --usedatasize 500 |& tee 500.log
python SK_s.py --SKs 0.2 --usedatasize 1000 |& tee 1000.log
python SK_s.py --SKs 0.2 --usedatasize 2000 |& tee 2000.log
python SK_s.py --SKs 0.2 --usedatasize 5000 |& tee 5000.log
python SK_s.py --SKs 0.2 --usedatasize 10000 |& tee 10000.log
python SK_s.py --SKs 0.2 --usedatasize 20000 |& tee 20000.log
python SK_s.py --SKs 0.2 --usedatasize 60000 |& tee 60000.log

