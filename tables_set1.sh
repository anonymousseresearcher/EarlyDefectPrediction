#!/usr/bin/env bash

echo Ranking sampling policies evaluated on all 6 learners

cat ./results/table_1/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_brier.csv

cat ./results/table_1/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_d2h.csv

cat ./results/table_1/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_recall.csv

cat ./results/table_1/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_pf.csv

cat ./results/table_1/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_auc.csv

cat ./results/table_1/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_ifa.csv

cat ./results/table_1/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > table_1_gm.csv

echo results generated at the current working directory
