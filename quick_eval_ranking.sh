#!/usr/bin/env bash

echo Ranking sampling policies evaluated on all 6 learners 

cat ./result/table2/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > table4_brier.csv

cat ./result/table2/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > table4_d2h.csv

cat ./result/table2/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > table4_recall.csv

cat ./result/table2/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > table4_pf.csv

cat ./result/table2/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > table4_auc.csv

cat ./result/table2/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > table4_ifa.csv

cat ./result/table2/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > table4_gm.csv

cat ./result/table3/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > table5_brier.csv

cat ./result/table3/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > table5_d2h.csv

cat ./result/table3/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > table5_recall.csv

cat ./result/table3/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > table5_pf.csv

cat ./result/table3/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > table5_auc.csv

cat ./result/table3/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > table5_ifa.csv

cat ./result/table3/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > table5_gm.csv

14 csv files generated in the current working directory 
