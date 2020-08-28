# Paper: Early Life Cycle Software Analytics. Why? How?

<img src="https://upload.wikimedia.org/wikipedia/commons/7/73/Alarm_Clock_Vector.svg" width="350">

Defect prediction is a widely studied
area of software analytics research.
Standard methods in defect prediction  are *data hungry;* i.e.
(1)given a choice of using more
data, or some small sampler, researchers typically assume that more is better;
(2)also, when data is missing, researchers take elaborate steps to transfer data from another projects;
(3)a given a choice of older data or some more recent sample,
researchers using ignore the older information.

Based on analysis of hundreds  of popular Github projects (with 1.2 million commits),
we suggest that for defect prediction, there is limited value in such data hungry approaches.
Data for our sample of projects lasts
  for 84  months   and contains  3,728  commits (median values).
Across these projects, most of the defects occur very early
in their life cycle.
Hence,
defect predictors learned from the first
150 commits and four months  perform
just as well as predictors learned from any other sample of that data.

This means that, contrary to the *data hungry* approach,  (1) small samples of data from these projects
are all that is needed for defect prediction;
(2) transfer learning has limited value
since it is needed only for the first   4 our of 84 months (i.e. 
just  4\% of the life cycle);
(3) after the first few months, we can ignore  newly arriving data.

We hope this work inspires other researchers to adopt a *simplicity-first*
approach to their work. Certainly, there are domains  that require
a complex and data hungry analysis. But before   assuming
complexity, it can be useful to examine the raw data looking for short cuts
that simplify the whole
analysis. 

## To reproduce the tables in results section of the paper follow the instructions below:

### Prerequisites

* Linux Terminal
* python 2, python 3
* Git support
* pandas

### On your linux terminal

$ `git clone https://github.com/anonymousseresearcher/EarlyDefectPrediction.git`

$ `cd EarlyDefectPrediction`

## (1) Final Result: To generate 2 tables used in the results section of the paper.

$ `python3 generate_tables.py`

**After successful execution, 2 csv files (Tables) will be generated at the current working directory**

## (2) Intermediate Result: To generate ranks for each policy:learner on all 7-evaluation measures using Scott-Knott, follow the instructions below:

$ `chmod +x .sh tables_set1.sh tables_set2.sh`

$ `./tables_set1.sh`

**After successful execution, 7 (csv) files on for each evaluation measure will be generated at the current working directory, move the csvs to a separate folder**
**Note: These 7 csv's are used as an input to (1) 

$ `./tables_set2.sh`

**After successful execution, 7 (csv) files on for each evaluation measure will be generated at the current working directory, move the csvs to a separate folder**
**Note: These 7 csv's are used as an input to (1)

