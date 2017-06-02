#!/usr/bin/env bash

python SK_s.py --SKs 2 |& tee LOG/sk_2.log
python SK_s.py --SKs 1.5 |& tee LOG/sk_1.5.log
python SK_s.py --SKs 0.5 |& tee LOG/sk_0.5.log
python SK_s.py --SKs 0.2 |& tee LOG/sk_0.2.log

