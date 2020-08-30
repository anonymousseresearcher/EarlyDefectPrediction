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

## We offer two approaches to reproduce RQ1, RQ2, and RQ3 results as shown in the image below:
1. ### Quick Replication (OR) 
2. ### Replication from Scratch

<img src="https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/images/overview.PNG" width="500">

## Prerequisites

* Linux Terminal
* python 2.7.5 and python 3.6.7
* Git support

### On your Linux terminal

1. $ `git clone https://github.com/anonymousseresearcher/EarlyDefectPrediction.git`
1. $ `cd EarlyDefectPrediction`
1. $ `pip3 install -r requirements.txt`

## 1. For `Quick Replication` follow the steps in the image below

<img src="https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/images/quick.PNG" width="900">

## 2. To replicate from `scratch` follow the steps in the image below

<img src="https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/images/scratch.PNG" width="900">



## Resources

1. ## Projects with release information are available [here](https://github.com/anonymousseresearcher/EarlyDefectPrediction/tree/master/data)
2. ## Project predictions (pre-generated) results are available [here](https://github.com/anonymousseresearcher/EarlyDefectPrediction/tree/master/results/detailed_report) (Project-Release wise report across all learners and evaluation measures)


