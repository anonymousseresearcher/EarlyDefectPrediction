#!/usr/bin/env bash

echo "Ranking sampling policies evaluated on all 6 learners (Table IV and Table V individual rankings) using inputs (txt files) in ./output/table4/ and ./output/table5/"

cat ./output/table4/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_brier.csv

echo "1 of 14 csv files generated"

cat ./output/table4/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_d2h.csv

echo "2 of 14 csv files generated"

cat ./output/table4/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_recall.csv

echo "3 of 14 csv files generated"

cat ./output/table4/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_pf.csv

echo "4 of 14 csv files generated"

cat ./output/table4/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_auc.csv

echo "5 of 14 csv files generated"

cat ./output/table4/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_ifa.csv

echo "6 of 14 csv files generated"

cat ./output/table4/zgm.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_gm.csv

echo "7 of 14 csv files generated"

cat ./output/table5/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_brier.csv

echo "8 of 14 csv files generated"

cat ./output/table5/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_d2h.csv

echo "9 of 14 csv files generated"

cat ./output/table5/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_recall.csv

echo "10 of 14 csv files generated"

cat ./output/table5/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_pf.csv

echo "11 of 14 csv files generated"

cat ./output/table5/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_auc.csv

echo "12 of 14 csv files generated"

cat ./output/table5/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_ifa.csv

echo "13 of 14 csv files generated"

cat ./output/table5/zgm.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table5/scratch_table5_gm.csv

echo "14 csv files generated the current working directory"


