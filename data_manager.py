
from builtins import print

import time
import numpy as np

import feature_selector
import pandas as pd

"""
@author : Anonymous
"""

import glob, os
from Constants import *


def getBugCount( xDF):

    return len(xDF[xDF['Buggy'] > 0])

def getCleanCount( xDF):

    return len(xDF[xDF['Buggy'] == 0])


PROJECTS_155 = ['ActionBarSherlock']

# #, 'active_merchant', 'ajenti', 'android', 'androidannotations', 'AnySoftKeyboard', 'apollo', 'assetic', 'backup', 'beets',
#                'bitcoin',  'brackets', 'brakeman', 'brotli', 'Cachet',  'camel', 'canal', 'capybara', 'cassandra', 'Catch', 'CodeIgniter', 'codis', 'compass',
#                'coreclr', 'curl', 'dagger', 'devise', 'diaspora', 'disruptor', 'django-rest-framework', 'django-tastypie', 'django', 'dompdf', 'druid', 'elasticsearch',
#                'eventmachine', 'facebook-android-sdk', 'fat_free_crm', 'fluentd', 'formtastic', 'fresco', 'gevent', 'glide', 'go-ethereum', 'google-api-php-client',
#                 'gradle', 'grape', 'grav', 'greenDAO', 'GSYVideoPlayer', 'guava', 'gunicorn', 'home-assistant', 'homebrew-cask', 'Hystrix', 'Imagine', 'ionic',
#                 'ipython', 'istio', 'jadx', 'jekyll', 'jieba', 'jinja2', 'jq', 'jsoup', 'junit', 'kafka', 'leakcanary', 'Lean', 'libgdx', 'libsass', 'libsodium',
#                 'logstash', 'macdown', 'macvim', 'masscan', 'mechanize', 'memcached', 'metasploit-framework', 'metrics', 'mezzanine', 'middleman', 'mopidy', 'mosh',
#                 'mybatis-3', 'nanomsg', 'neovim', 'netty', 'newspaper', 'Newtonsoft.Json', 'numpy', 'okhttp', 'pandas', 'passenger', 'patternlab-php', 'peewee',
#                 'PHP-CS-Fixer', 'phpunit', 'picasso', 'piwik', 'portia', 'postgres', 'powerline', 'prawn', 'predis', 'proxygen', 'pry', 'puphpet', 'pyspider',
#                 'pyston', 'raiden', 'rails_best_practices', 'rasa', 'realm-java', 'redis', 'redisson', 'requests', 'rhino', 'roboguice', 'rubocop', 'ruby',
#                 'scikit-learn', 'seaborn', 'ServiceStack', 'SFML',   'shadowsocks-csharp', 'Signal-Android', 'SignalR', 'springside4',
#                'state_machine', 'sunspot', 'swoole', 'symfony', 'sympy', 'sysdig', 'taiga-back', 'tesseract', 'thanos', 'ThinkUp', 'tiled', 'titan', 'tweepy', 'Twig',
#                'twitter', 'Validation', 'vcr', 'vert.x', 'wagtail', 'watchman', 'WP-API', 'wp-cli', 'yii', 'zf2', 'zipline', 'zulip']

def getProjectNames():
    return PROJECTS_155


"""
Constants
"""


data_attribute = 'author_date_unix_timestamp'

class project(object):

    def __init__(self, name):
        self.name = name
        self.releases = getReleases(name)

        tempStartDate = math.inf
        tempEndDate = 0

        for r in self.releases:
            tempStartDate = min(tempStartDate, r.getStartDate())
            tempEndDate= max(tempEndDate, r.getReleaseDate())

        self.years = (tempEndDate - tempStartDate)/one_year


    def getYears(self):

        return self.years


    def getReleases(self):
        return self.releases

    def getName(self):
        return self.name

    def getAllChanges(self):

        changesDF = None
        changes = 0
        for r in self.releases:


            changes += len(r.getChanges())

            if changesDF is None:
                changesDF = r.getChanges()
            else:
                changesDF = changesDF.append(r.getChanges())



        return changesDF


class release(object):

    def __init__(self,release_date, changes, startDate):

        self.release_date = release_date
        self.changes = changes
        self.startDate = startDate


    def getReleaseDate(self):
        return self.release_date

    def getStartDate(self):
        return self.startDate

    def getChanges(self):
        return self.changes

    def __str__(self):
        return str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.release_date)) + " : "+str(
                len(self.changes)))



def getReleasesBefore(project, releaseDate):

    pastReleases = []

    for r in project.getReleases():

        if r.getReleaseDate() < releaseDate:
            pastReleases.append(r)


    return pastReleases

def getReleases(p):



    df = pd.read_csv('./data/'+p+'.csv')

    releaseObjects = []


    prevPeriod = None

    releaseDates = pd.read_csv('./data/releases/' + p + ".csv")['releases'].values.tolist()


    added = CONSIDER_FIRST_X_RELEASES

    for currentPeriod in releaseDates:

        if added <= 0:
            break

        if prevPeriod is None:
            prevPeriod = currentPeriod
            continue
        else:
            period = [prevPeriod, currentPeriod]


            tempDF = df[ (df[data_attribute] > prevPeriod) & (df[data_attribute] <= currentPeriod) ]


            rDF = formatDF(tempDF)

            if len(rDF) > 1:
                releaseObjects.append(release(currentPeriod, rDF, tempDF['author_date_unix_timestamp'].min()))
                added -= 1

            prevPeriod = currentPeriod


    return releaseObjects

def get_features(df):
    fs = feature_selector.featureSelector()
    df, _feature_nums, features = fs.cfs_bfs(df)
    return df, features


def formatDF(rdf):

    """
    Works for QT and OPEN-STACK
    releaseDF = rdf.copy(deep=True)
    releaseDF = releaseDF[['la','ld','nf','nd','ns','ent','revd','nrev','rtime','tcmt','hcmt','self',
                    'ndev','age','nuc','app','aexp','rexp','oexp','arexp','rrexp','orexp','asexp','rsexp','osexp','asawr','rsawr','osawr',
                    'commit_id','author_date','fixcount',
                    'bugcount']]
    releaseDF = releaseDF.drop(labels=['commit_id','author_date','fixcount'], axis=1)
    releaseDF = releaseDF.fillna(0)
    releaseDF.rename(columns={'bugcount': 'Buggy'}, inplace=True)
    """

    releaseDF = rdf.copy(deep=True)

    releaseDF = releaseDF[ [ 'author_date_unix_timestamp', 'ns','nd','nf','entropy', 'la','ld','lt','ndev','age','nuc','exp','rexp', 'sexp' , 'fix', 'contains_bug' ] ]


    dropList = ['author_date_unix_timestamp']

    releaseDF = releaseDF.drop(labels=dropList, axis=1)
    releaseDF = releaseDF.fillna(0)

    releaseDF = releaseDF[ (releaseDF['contains_bug'] == True) | (releaseDF['contains_bug'] == False) ]

    releaseDF.rename(columns={'contains_bug': 'Buggy'}, inplace=True)


    return releaseDF





def getProject(p):
    return project(p)






