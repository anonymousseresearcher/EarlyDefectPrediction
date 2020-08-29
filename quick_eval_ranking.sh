#!/usr/bin/env bash

echo "Ranking sampling policies evaluated on all 6 learners (Table IV and Table V individual rankings)"

cat ./results/table_1/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > table4_brier.csv

echo table4_brier.csv generated

cat ./results/table_1/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > table4_d2h.csv

echo table4_d2h.csv generated

cat ./results/table_1/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > table4_recall.csv

echo table4_recall.csv generated

cat ./results/table_1/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > table4_pf.csv

echo table4_pf.csv generated

cat ./results/table_1/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > table4_auc.csv

echo table4_auc.csv generated

cat ./results/table_1/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > table4_ifa.csv

echo table4_ifa.csv generated

cat ./results/table_1/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > table4_gm.csv

echo table4_gm.csv generated

cat ./results/table_2/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > table5_brier.csv

echo table5_brier.csv generated

cat ./results/table_2/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > table5_d2h.csv

echo table5_d2h.csv generated

cat ./results/table_2/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > table5_recall.csv

echo table5_recall.csv generated

cat ./results/table_2/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > table5_pf.csv

echo table5_pf.csv generated

cat ./results/table_2/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > table5_auc.csv

echo table5_auc.csv generated

cat ./results/table_2/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > table5_ifa.csv

echo table5_ifa.csv generated

cat ./results/table_2/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > table5_gm.csv

echo table5_gm.csv generated

echo "14 csv files generated at current working directory"
