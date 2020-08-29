# [Paper: Early Life Cycle Software Analytics. Why? How?](https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/paper.pdf) 

<img src="https://upload.wikimedia.org/wikipedia/commons/7/73/Alarm_Clock_Vector.svg" width="350">

Abstract—Many methods in defect prediction are “datahungry”;
i.e. (1) given a choice of using more data, or some
smaller sample, researchers assume that more is better; (2) when
data is missing, researchers take elaborate steps to transfer data
from another project; and (3) given a choice of older data or
some more recent sample, researchers usually ignore older data.
Based on the analysis of hundreds of popular Github projects
(with 1.2 million commits), we suggest that for defect prediction,
there is limited value in such data-hungry approaches. Data
for our sample of projects last for 84 months and contains
3,728 commits (median values). Across these projects, most of
the defects occur very early in their life cycle. Hence, defect
predictors learned from the first 150 commits and four months
perform just as well as anything else.
This means that, contrary to the “data-hungry” approach,
(1) small samples of data from these projects are all that is
needed for defect prediction; (2) transfer learning has limited
value since it is needed only for the first 4 of 84 months (i.e. just
4% of the life cycle); (3) after the first few months, we need not
continually update our defect prediction models.
We hope these results inspire other researchers to adopt a
‘simplicity-first” approach to their work. Certainly, there are
domains that require a complex and data-hungry analysis. But
before assuming complexity, it is prudent to check the raw data
looking for “short cuts” that simplify the whole analysis.

## To reproduce the tables in results section of the paper follow the instructions below:

### Prerequisites

* Linux Terminal
* python 2, python 3
* Git support
* pandas

### On your linux terminal

$ `git clone https://github.com/anonymousseresearcher/EarlyDefectPrediction.git`

$ `cd EarlyDefectPrediction`

## (1) [Time to Run: 1 min] Final Result (RQ1,RQ2 and RQ3): To generate Table IV and V of the paper, execute the script below:

$ `python3 generate_tables.py`

**After successful execution, 2 csv files (Tables) will be generated at the current working directory**

## (2) [Time to Run: 30 min] To understand Table IV and V in detail execute the scripts below. The two scripts ranks each policy:learner (pair) on all 7-evaluation measures using Scott-Knott:

$ `chmod +x tables_set1.sh tables_set2.sh`

$ `./tables_set1.sh`

**After successful execution, 7 (csv) files on for each evaluation measure will be generated at the current working directory**
**Note: These 7 csv's are used as an input to (1)**

$ `./tables_set2.sh`

**After successful execution, 7 (csv) files on for each evaluation measure will be generated at the current working directory**
**Note: These 7 csv's are used as an input to (1)**

## Reports for (2) are computed from [here](https://github.com/anonymousseresearcher/EarlyDefectPrediction/tree/master/results/detailed_report) (Project-Release wise report across all learners and evaluation measures)

## Projects with release information are available [here](https://github.com/anonymousseresearcher/EarlyDefectPrediction/tree/master/data)

