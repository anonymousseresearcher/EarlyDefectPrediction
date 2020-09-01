#!/usr/bin/env bash

echo "Ranking sampling policies evaluated on all 6 learners, Table IV and Table V individual rankings"

echo "Ranking... 1 of 14" 

cat ./results/table_1/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_brier.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_brier.csv generated

echo "Ranking... 2 of 14" 

cat ./results/table_1/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_d2h.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_d2h.csv generated

echo "Ranking... 3 of 14" 

cat ./results/table_1/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_recall.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_recall.csv generated

echo "Ranking... 4 of 14" 

cat ./results/table_1/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_pf.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_pf.csv generated

echo "Ranking... 5 of 14" 
cat ./results/table_1/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_auc.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_auc.csv generated

echo "Ranking... 6 of 14" 
cat ./results/table_1/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_ifa.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_ifa.csv generated

echo "Ranking... 7 of 14" 
cat ./results/table_1/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table4_gm.csv

echo "Note: RESAMPLE_150_25_25 refers to E"
echo quick_table4_gm.csv generated

echo "Ranking... 8 of 14" 
cat ./results/table_2/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_brier.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_brier.csv generated

echo "Ranking... 9 of 14" 
cat ./results/table_2/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_d2h.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_d2h.csv generated

echo "Ranking... 10 of 14" 
cat ./results/table_2/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_recall.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_recall.csv generated

echo "Ranking... 11 of 14" 
cat ./results/table_2/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_pf.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_pf.csv generated

echo "Ranking... 12 of 14" 
cat ./results/table_2/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_auc.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_auc.csv generated

echo "Ranking... 13 of 14" 
cat ./results/table_2/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_ifa.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_ifa.csv generated

echo "Ranking... 14 of 14" 
cat ./results/table_2/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > quick_table5_gm.csv

echo "Note: RESAMPLE_150_25_25 refers to E and  RECENT_RELEASE refers to RR in the paper"
echo quick_table5_gm.csv generated

echo "14 csv files generated at current working directory"
