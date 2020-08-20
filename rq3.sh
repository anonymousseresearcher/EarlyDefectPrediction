#!/usr/bin/env bash

echo Ranking sampling policies evaluated on brier across all 5 learners

cat ./results/rq3/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > rq3_scott_knott_results_brier.csv

echo Ranking sampling policies evaluated on d2h across all 5 learners

cat ./results/rq3/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > rq3_scott_knott_results_d2h.csv

echo Ranking sampling policies evaluated on recall across all 5 learners

cat ./results/rq3/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > rq3_scott_knott_results_recall.csv

echo Ranking sampling policies evaluated on pf across all 5 learners

cat ./results/rq3/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > rq3_scott_knott_results_pf.csv

echo Ranking sampling policies evaluated on auc across all 5 learners

cat ./results/rq3/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > rq3_scott_knott_results_roc_auc.csv

echo Ranking sampling policies evaluated on ifa across all 5 learners

cat ./results/rq3/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > rq3_scott_knott_results_ifa.csv

echo rq3 results generated at the current working directory
