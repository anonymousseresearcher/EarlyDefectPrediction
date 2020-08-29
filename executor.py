import os

import calendar
import csv
import time
from scipy.stats import *

from os import path
from sklearn.ensemble import RandomForestClassifier,  RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn import preprocessing, metrics

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC


# from Util import *
from Constants import *
from multiprocessing import Process
from random import *



import SMOTE
import feature_selector
import CFS

from data_manager import *

import numpy as np


from multiprocessing import Pool, cpu_count

import metrices
import warnings
warnings.filterwarnings("ignore")



Dummy_Flag = False

def toNominal(changes):

    releaseDF = changes
    releaseDF.loc[releaseDF['Buggy'] >= 1, 'Buggy'] = 1
    releaseDF.loc[releaseDF['Buggy'] <= 0, 'Buggy'] = 0
    d = {1: True, 0: False}
    releaseDF['Buggy'] = releaseDF['Buggy'].map(d)

    return changes

def splitChanges(changesDF):

    defectsPerHalf = getBugCount(changesDF)/2
    nonDefectsPerHalf = getCleanCount(changesDF)/2

    onehalfIndex = []
    otherhalfIndex = []

    changesDF = changesDF.reset_index()

    for index, row in changesDF.iterrows():

        added = False
        if row['Buggy'] > 0 and defectsPerHalf > 0:
            onehalfIndex.append(index)
            defectsPerHalf -= 1
            added = True

        if row['Buggy'] == 0 and nonDefectsPerHalf > 0:
            onehalfIndex.append(index)
            nonDefectsPerHalf -= 1
            added = True

        if added == False:
            otherhalfIndex.append(index)



    onehalfChanges = changesDF.copy(deep=True)
    otherhalfChanges = changesDF.copy(deep=True)


    onehalfChanges = onehalfChanges.drop(onehalfChanges.index[onehalfIndex]).copy(deep=True)
    otherhalfChanges = otherhalfChanges.drop(otherhalfChanges.index[otherhalfIndex]).copy(deep=True)

    onehalfChanges = onehalfChanges.drop(labels=['index'], axis=1)
    otherhalfChanges = otherhalfChanges.drop(labels=['index'], axis=1)

    return otherhalfChanges, onehalfChanges

def getFreshCopy(trainChangesCpy, testChangesCpy, trainApproach):

    trainChangesNom = toNominal(trainChangesCpy)
    testChangesNom = toNominal(testChangesCpy)

    if Dummy_Flag == True:
        trainChangesSplit, tuneChangesSplit = splitChanges(trainChangesNom)
        trainChangesProcessed = customPreProcess(trainChangesSplit)
        testChangesProcessed = customPreProcess(testChangesNom)
        tuneChangesProcessed =  customPreProcess(tuneChangesSplit)
    else:
        trainChangesProcessed = customPreProcess(trainChangesNom)
        testChangesProcessed = customPreProcess(testChangesNom)
        tuneChangesProcessed = None

    if 'RESAMPLE' not in trainApproach:
        trainChangesSMOTE = MLUtil().apply_smote(trainChangesProcessed)
        return trainChangesSMOTE, tuneChangesProcessed, testChangesProcessed
    else:
        return trainChangesProcessed,  tuneChangesProcessed, testChangesProcessed

def customPreProcess(changesDF):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:
        if c != 'Buggy':
            changesDF[c] = changesDF[c] + abs(changesDF[c].min()) + 0.00001

    changesDF.insert(0, 'loc', changesDF['la'] + changesDF['ld'])

    changesDF['la'] = changesDF['la'] / changesDF['lt']
    changesDF['ld'] = changesDF['ld'] / changesDF['lt']

    changesDF['lt'] = changesDF['lt'] / changesDF['nf']
    changesDF['nuc'] = changesDF['nuc'] / changesDF['nf']

    changesDF = changesDF.drop('nd', 1)
    changesDF = changesDF.drop('rexp', 1)

    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc':
            changesDF[c] = changesDF[c] + 0.0000001

    """
    log Normalization
    """
    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc':
            changesDF[c] = np.log10(changesDF[c])

    return changesDF

def getFileName(projectName):
    return './output/project_' + projectName + "_results.csv"

def getHeader():

    header = ['projectName', 'trainApproach', 'train_info', 'testReleaseDate', 'train_changes', 'test_changes',
              'train_Bug_Per', 'test_Bug_Per', 'features_selected', 'classifier']

    header += METRICS_LIST


    return header

def resultExists(p):

    filePath = './output/project_' + p + "_results.csv"

    return path.exists(filePath)

def valid(changes):
    return changes is not None and len(changes) > 1 and (len(changes[changes['Buggy'] > 0]) > 5 and len(changes[changes['Buggy'] == 0]) > 5)

class MLUtil(object):

    def __init__(self):
        self.cores = 1

    def get_features(self,df):
        fs = feature_selector.featureSelector()
        df,_feature_nums,features = fs.cfs_bfs(df)
        return df,features

    def apply_pca(self, df):
        return df

    def apply_cfs(self,df):

        copyDF = df.copy(deep=True)

        y = copyDF.Buggy.values
        X = copyDF.drop(labels=['Buggy'], axis=1)

        X = X.values

        selected_cols = CFS.cfs(X, y)

        finalColumns = []

        for s in selected_cols:

            if s > -1:
                finalColumns.append(copyDF.columns.tolist()[s])

        finalColumns.append('Buggy')

        if 'loc' in finalColumns:
            # internal attribute remove
            finalColumns.remove('loc')


        return None, finalColumns


    def apply_normalize(self, df):
        """
        Not used
        :param df:
        :return:
        """

        return df

    def apply_smote(self,df):

        originalDF = df.copy(deep=True)

        try:
            cols = df.columns
            smt = SMOTE.smote(df)
            df = smt.run()
            df.columns = cols
        except:

            return originalDF


        return df



def getSimpleName(classifier, PREFIX=''):

    if isinstance(classifier, RandomForestClassifier) or isinstance(classifier, RandomForestRegressor):
        return PREFIX + 'RF'
    elif isinstance(classifier, LogisticRegression) or isinstance(classifier, LinearRegression) :
        return PREFIX + 'LR'
    elif isinstance(classifier, GaussianNB):
        return PREFIX + 'NB'
    elif isinstance(classifier, KNeighborsClassifier):
        return PREFIX + 'KNN'
    elif isinstance(classifier, DecisionTreeClassifier):
        return PREFIX + 'DT'
    elif isinstance(classifier, SVC):
        return PREFIX + 'SVM'
    else:
        return "getSimpleName "+str(classifier)

"""
Classifiers from Ghotra et. al 2013
"""
def getClassifiers():
    return [
        KNeighborsClassifier(n_neighbors=5),
             DecisionTreeClassifier(),
             LogisticRegression(),
             RandomForestClassifier(),
             GaussianNB(),SVC()
    ]

def ignoreList():
    return [
        [KNeighborsClassifier][0],
        [DecisionTreeClassifier][0],
        [LogisticRegression][0],
        [RandomForestClassifier][0],
        [GaussianNB][0],
        [SVC][0]
    ]

def getTrue(labels):

    c = 0

    for x in labels:

        if x or x == 'True':
            c += 1

    return c


def getFalse(labels):
    c = 0

    for x in labels:

        if x == False or x == 'False':
            c += 1

    return c

def getFeatureSelectors():
    return ['CFS']


def getSimpleNames():

    return [ getSimpleName(clf) for clf in getClassifiers() ]


def computeMeasures(test_df, clf, timeRow, codeChurned):


    F = {}

    if getSimpleName(clf) == 'KNN' and  test_df.shape[0] < 6: #because knn needs 5 neighbors minimum
        return None

    test_y = test_df.Buggy
    test_X = test_df.drop(labels=['Buggy'], axis=1)

    testStart = time.time()

    try:
        predicted = clf.predict(test_X)
    except:
        predicted = None



    testDiff = time.time() - testStart
    timeRow.append(testDiff)


    try:
        abcd = metrices.measures(test_y, predicted, codeChurned )
    except:
        abcd = None

    errorMessage = 'MEASURE_ERROR'

    if abcd is None:
        errorMessage = 'CLF_ERROR'


    try:
        F['recall'] = [abcd.calculate_recall()]
    except:
        F['recall'] = [errorMessage]

    try:
        F['pf'] = [abcd.get_pf()]
    except:
        F['pf'] = [errorMessage]


    try:
        F['gm'] = [abcd.get_g_score()]
    except:
        F['gm'] = [errorMessage]

    try:
        F['d2h'] = [abcd.calculate_d2h()]
    except:
        F['d2h'] = [errorMessage]


    try:
        F['ifa'] = [abcd.get_ifa()]
    except:
        print('\t ',errorMessage)
        F['ifa'] = [errorMessage]

    try:
        F['roc_auc'] = [abcd.get_roc_auc_score()]
    except:
        F['roc_auc'] = [errorMessage]

    try:
        F['brier'] = [abcd.brier()]
    except:
        F['brier'] = [errorMessage]


    return F

UNIT_TEST = False

def performPredictionRunner(projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach, returnResults=False):

    if valid(originalTrainChanges) == False or valid(originalTestChanges) == False:

        return


    trainChangesCpy2 = originalTrainChanges.copy(deep=True)


    selected_cols = []
    for fselector in  ['CFS']:

        trainChangesSMOTE,tuneChangesProcessed,testChangesProcessed = getFreshCopy(originalTrainChanges.copy(deep=True),originalTestChanges.copy(deep=True), trainApproach)
        trainChanges, tuneChanges, testChanges = None, None, None


        testLocList = testChangesProcessed['loc'].values.tolist()

        if fselector == 'CFS':
            someDF, selected_cols = MLUtil().apply_cfs(trainChangesSMOTE)



            if Dummy_Flag == True:
                trainChanges = trainChangesSMOTE[selected_cols]
                tuneChanges = tuneChangesProcessed[selected_cols]
                testChanges = testChangesProcessed[selected_cols]
            else:
                trainChanges = trainChangesSMOTE[selected_cols]
                testChanges = testChangesProcessed[selected_cols]

        trainChangesSMOTE, tuneChangesProcessed, testChangesProcessed = None,None,None


        """
        same order of learners
        """
        learners = getClassifiers()
        THE_LEARNERS_LIST = ignoreList()
        """
        """

        learnerIndex = -1

        for learner in THE_LEARNERS_LIST:

            learnerIndex += 1

            trainChangesCpy, tuneChangesCpy, testChangesCpy = None, None, None

            if Dummy_Flag == True:
                trainChangesCpy, tuneChangesCpy, testChangesCpy = trainChanges.copy(deep=True), tuneChanges.copy(deep=True), testChanges.copy(deep=True)
            else:
                trainChangesCpy,  testChangesCpy = trainChanges.copy(deep=True),  testChanges.copy(deep=True)

            trainY = trainChangesCpy.Buggy
            trainX = trainChangesCpy.drop(labels=['Buggy'], axis=1)

            if Dummy_Flag == True:
                tuneX = tuneChangesCpy.drop(labels=['Buggy'], axis=1)
                tuneY = tuneChangesCpy.Buggy

            if Dummy_Flag and getTrue(trainY) > 0 and getTrue(tuneY) > 0 and getFalse(trainY) > 0 and getFalse(tuneY) > 0:

               continue

            elif getTrue(trainY) > 0 and getFalse(trainY) > 0 :

                clf = learners[learnerIndex]

                classifierName =  getSimpleName(clf)


                clf.fit(trainX, trainY)

                F = computeMeasures(testChangesCpy, clf, [], testLocList)

            else:
                F = None
                clf = learners[learnerIndex]
                classifierName = getSimpleName(clf)
                fselector = None

            metricsReport = F

            featuresSelectedStr = ''
            for sc in selected_cols:
                featuresSelectedStr += '$' + sc


            result = [projectName, trainApproach, trainReleaseDate, testReleaseDate,
                      len(trainChangesCpy),
                      len(testChangesCpy),

                      percentage(len(trainChangesCpy[trainChangesCpy['Buggy'] > 0]),
                                 len(trainChangesCpy)),
                      percentage(len(testChangesCpy[testChangesCpy['Buggy'] > 0]),
                                 len(testChangesCpy)) ,
                      featuresSelectedStr, classifierName]

            if metricsReport is not None:
                for key in metricsReport.keys():

                    result += metricsReport[key]
            else:
                for m in METRICS_LIST:
                    result.append(str('UNABLE'))

            if UNIT_TEST == False:

                if returnResults:
                    return metricsReport['balance']
                else:
                    writeRow(getFileName(projectName), result)
            else:
                print("***************** ", classifierName, fselector, trainChangesCpy.columns.tolist(), testChangesCpy.columns.tolist(), " Results **************************")
                if metricsReport is not None:

                    print('\tprecision', metricsReport['precision'])
                    print('\trecall', metricsReport['recall'])
                    print('\tpf', metricsReport['pf'])
                    print('\troc_auc', metricsReport['roc_auc'])
                    print('\td2h', metricsReport['d2h'])
                    print('*** \n ')
                    return metricsReport['precision'], metricsReport['recall'], metricsReport['pf'], selected_cols
                else:

                    print(metricsReport, getTrue(trainY) , getFalse(trainY) )
                    return None, None, None, None



def getLastXChangesEndDate(startDate, endDate, project):


    releases = project.getReleases()


    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() < endDate:
            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))


    return changeDF

def getLastXChanges(currentReleaseObj, project, months):

    if currentReleaseObj is None:
        return None

    releases = project.getReleases()

    if months == math.inf:
        startDate = 0
    else:
        startDate = currentReleaseObj.getStartDate() - (months * one_month)

    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() < currentReleaseObj.getStartDate():
            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))


    return changeDF


def getReleasesAfter(commits, releaseList):

    releases = []
    changes  = 0

    for r in releaseList:

        if changes >= commits:
            releases.append(r)

        changes += len(r.getChanges())

    return releases

def getFirstChangesBetween(project, startDate, endDate):

    releases = project.getReleases()


    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() <= endDate:

            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))



    return changeDF



def runAllExperiments(projectName):

    print("Appending prediction evaluations for ",projectName, ' to ',str(os.getcwd())+"/output/"+projectName+'.csv')

    projectObj = getProject(projectName)
    releaseList = projectObj.getReleases()

    commitsList = [150]
    testReleaseList = getReleasesAfter(max(commitsList), releaseList)
    projectStart = min([r.getStartDate() for r in releaseList])

    for testReleaseObj in testReleaseList:

        projectChanges = projectObj.getAllChanges()
        for commits in commitsList:

            if len(projectChanges) >= max(commitsList):  # ensure equal number of experiments

                trainingRegion = projectChanges.head(commits).copy(deep=True)

                if trainingRegion is not None and len(trainingRegion) > 0:
                    buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
                    nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

                    buggySamples = 25
                    nonBuggySamples = 25  # abs(samples - buggySamples)

                    if buggySamples > 5 and nonBuggySamples > 5:

                        if len(buggyChangesDF) >= buggySamples and len(nonBuggyChangesDF) >= nonBuggySamples:
                            RESAMPLE_COMMITS = buggyChangesDF.sample(buggySamples).copy(deep=True).append(
                                nonBuggyChangesDF.sample(nonBuggySamples).copy(deep=True)).copy(deep=True)

                            performPredictionRunner(projectName, RESAMPLE_COMMITS, testReleaseObj.getChanges(),
                                                    str(len(RESAMPLE_COMMITS)), testReleaseObj.getReleaseDate(),
                                                    'RESAMPLE_' + str(commits) + '_' + str(buggySamples) + '_' + str(
                                                        nonBuggySamples))

            trainingRegion = getFirstChangesBetween(projectObj, projectStart, testReleaseObj.getStartDate())
            buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
            nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

            buggySamples = 25
            nonBuggySamples = 25

            if buggySamples > 5 and nonBuggySamples > 5:

                if len(buggyChangesDF) >= buggySamples and len(nonBuggyChangesDF) >= nonBuggySamples:
                    RESAMPLE_COMMITS = buggyChangesDF.sample(buggySamples).copy(deep=True).append(
                        nonBuggyChangesDF.sample(nonBuggySamples).copy(deep=True)).copy(deep=True)

                    performPredictionRunner(projectName, RESAMPLE_COMMITS, testReleaseObj.getChanges(),
                                            str(len(RESAMPLE_COMMITS)), testReleaseObj.getReleaseDate(),
                                            'RESAMPLE_' + str(buggySamples) + '_' + str(nonBuggySamples))

    previousRelease = None

    for testReleaseObj in testReleaseList:

        allChangesDF = getLastXChangesEndDate(0, testReleaseObj.getStartDate(), projectObj)
        recent3MonthChanges = getLastXChanges(testReleaseObj, projectObj, 3)
        recent6MonthChanges = getLastXChanges(testReleaseObj, projectObj, 6)

        if allChangesDF is not None and len(allChangesDF) >= max(commitsList):
            performPredictionRunner(projectName, allChangesDF, testReleaseObj.getChanges(), str(len(allChangesDF)),
                                    testReleaseObj.getReleaseDate(), 'ALL')

        if recent3MonthChanges is not None:
            performPredictionRunner(projectName, recent3MonthChanges, testReleaseObj.getChanges(),
                                    str(len(recent3MonthChanges)),
                                    testReleaseObj.getReleaseDate(), '3MONTHS')

        if recent6MonthChanges is not None:
            performPredictionRunner(projectName, recent6MonthChanges, testReleaseObj.getChanges(),
                                    str(len(recent6MonthChanges)),
                                    testReleaseObj.getReleaseDate(), '6MONTHS')
        #
        if previousRelease is not None:
            performPredictionRunner(projectName, previousRelease.getChanges(), testReleaseObj.getChanges(),
                                    previousRelease.getReleaseDate(), testReleaseObj.getReleaseDate(),
                                    'RECENT_RELEASE')

        previousRelease = testReleaseObj



def run_train_test(projectName):

    if resultExists(projectName) == False:

        writeRow(getFileName(projectName), getHeader())
        try:
            runAllExperiments(projectName)
        except Exception as e:
            print("Error processing : ", projectName,
                  'Please extract zip (some projects are compressed to by-pass GitHub size limit)', e)
    else:
        print('Results for ', projectName, ' already exist (remove csv) if you wish to generate!')


def percentage(numer, denom):

    if denom > 0:
        return float(float(numer)*100/float(denom))
    else:
        return 0


def writeRow(filename, rowEntry):

    with open(filename, newline='', mode='a') as status_file:
        writer = csv.writer(status_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(rowEntry)


def get_common_releases(df, samplingPolicies):

    releaseList = []


    for samplingPolicy in samplingPolicies:

        samplingReleaseList  = df[ df['trainApproach'] == samplingPolicy ]['testReleaseDate'].values.tolist()
        # print(samplingPolicy, len(samplingReleaseList))
        if samplingReleaseList is None or len(samplingReleaseList) == 0:
            return []
        else:
            releaseList.append(samplingReleaseList)

    testReleaseSet = None

    for releases in releaseList:

        if testReleaseSet is None:

            testReleaseSet = list(set(releases))
            continue
        else:

           testReleaseSet =  list( set(testReleaseSet) & set(releases) )



    return testReleaseSet


def removeExistingFiles():

    print("Attemping to remove existing results")

    for table in ['table4', 'table5']:
        for metric in METRICS_LIST:
            filetoremove = './output/' + table + '/z' + metric + '.txt'
            try:
                if path.exists(filetoremove):
                    try:
                        os.remove(filetoremove)
                        print('Existing ', filetoremove,  ' removed!')
                    except Exception as e:
                        print('[ERROR] : Unable to remove ',filetoremove, str(e))
            except:
                continue



def collect_inputs_for_measures(metric, projectStartDateMap):



    for table in ['table4', 'table5']:

        if table == 'table4':
            samplingPolicies = []

            samplingPolicies.append('RESAMPLE_150_25_25')
            samplingPolicies.append('3MONTHS')
            samplingPolicies.append('6MONTHS')
            samplingPolicies.append('ALL')


        elif table == 'table5':

            samplingPolicies = []
            samplingPolicies.append('RECENT_RELEASE')
            samplingPolicies.append('RESAMPLE_150_25_25')

        f = open('./output/'+table+'/z' + metric  + '.txt', "a+")

        print("Generating ",'./output/'+table+'/z' + metric  + '.txt')

        for classifier in getSimpleNames():

            for selRule in  samplingPolicies:

                metricValues = []

                for p in getProjectNames():

                    df = pd.read_csv('./output/project_' + p + '_results.csv')

                    sixmonths = projectStartDateMap[p] + (6 * one_month )
                    df = df[ df['testReleaseDate'] > sixmonths ]
                    df = df[ df['classifier'] == classifier ]

                    commonReleases = get_common_releases(df, samplingPolicies)

                    if len(df) > 0:
                        sDF = df[ (df['testReleaseDate'].isin(commonReleases) ) & ( df['trainApproach'] == selRule ) ]
                    else:
                        continue

                    v = sDF[metric].values.tolist()

                    metricValues += v


                f.write(to_pretty_label(selRule) + "_" + classifier + "\n")
                line = ''
                for c in metricValues:

                    line += str(c) + " "

                f.write(line.strip() + "\n\n")


def to_pretty_label(selRule):
    if "3MONTHS" in selRule:
        return selRule.replace('3MONTHS', 'M3')
    elif "6MONTHS" in selRule:
        return selRule.replace('6MONTHS', 'M6')
    elif "RECENT:RELEASE" in selRule:
        return selRule.replace('RECENT:RELEASE', 'RR')
    elif "RESAMPLE" in selRule:
        return 'E'
    else:
        return selRule

def generate_scottknott_inputs():
    removeExistingFiles()

    projectStartDateMap = {}

    for p in getProjectNames():
        rrs = getProject(p).getReleases()
        projectStartDateMap[p] = min([r.getStartDate() for r in rrs])

    procs = []

    for metric in METRICS_LIST:
        proc = Process(target=collect_inputs_for_measures(metric, projectStartDateMap), args=(metric,))
        procs.append(proc)
        proc.start()

    # Complete the processes
    for proc in procs:
        proc.join()


def generate_project_results():
    procs = []
    projectNames = getProjectNames()
    for name in projectNames:
        print("Processing project ", name, ' in parallel!')
        proc = Process(target=run_train_test, args=(name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()





