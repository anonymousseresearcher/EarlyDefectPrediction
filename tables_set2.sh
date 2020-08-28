#!/usr/bin/env bash

echo Ranking sampling policies evaluated on all 6 learners

cat ./results/table_2/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_brier.csv

cat ./results/table_2/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_d2h.csv

cat ./results/table_2/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_recall.csv

cat ./results/table_2/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_pf.csv

cat ./results/table_2/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_auc.csv

cat ./results/table_2/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_ifa.csv

cat ./results/table_2/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > table_2_gm.csv

echo results generated at the current working directory
