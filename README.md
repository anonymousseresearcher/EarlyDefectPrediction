# [Paper: Early Life Cycle Software Defect Prediction. Why? How?](https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/paper.pdf) 

<img src="https://upload.wikimedia.org/wikipedia/commons/7/73/Alarm_Clock_Vector.svg" width="350">

Abstract—Many researchers assume that, for software analyt-ics,  “more  data  is  better”.  We  write  to  show  that,  at  least  forlearning  defect  predictors,  this  may  not  be  true.To demonstrate this, we analyzed hundreds of popular GitHubprojects.   These   projects   ran   for   84   months   and   contained3,728  commits  (median  values).  Across  these  projects,  most  ofthe  defects  occur  very  early  in  their  life  cycle.  Hence,  defectpredictors  learned  from  the  first  150  commits  and  four  monthsperform  just  as  well  as  anything  else.  This  means  that,  at  leastfor the projects studied here, after the first few months, we neednot  continually  update  our  defect  prediction  models.We  hope  these  results  inspire  other  researchers  to  adopt  a“simplicity-first.” approach to their work. Indeed, some domainsrequire a complex and data-hungry analysis. But before assumingcomplexity, it is prudent to check the raw data looking for “shortcuts”  that  simplify  the  whole  analysis.

## We offer two approaches to reproduce RQ1, RQ2, and RQ3 results as shown in the image below:
1. ### Quick Replication (1 hr) (OR) 
2. ### Replication from Scratch (24 hr) (slow)

<img src="https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/images/overview.PNG" width="500">

## Prerequisites

* Linux Terminal
* python 2.7.5 and python 3.6.7
* Git support

### On your Linux terminal

1. $ `git clone https://github.com/anonymousseresearcher/EarlyDefectPrediction.git`
1. $ `cd EarlyDefectPrediction`
1. $ `pip3 install -r requirements.txt`

## 1. For `Quick Replication` (1 hr) follow the steps in the image below

<img src="https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/images/quick.PNG" width="900">

## 2. To replicate from `scratch` (24 hr) follow the steps in the image below

<img src="https://github.com/anonymousseresearcher/EarlyDefectPrediction/blob/master/images/scratch.PNG" width="900">

## Dataset

1. ## Projects [here](https://github.com/anonymousseresearcher/EarlyDefectPrediction/tree/master/data)
2. ## Project release information [here](https://github.com/anonymousseresearcher/EarlyDefectPrediction/tree/master/data/releases)

