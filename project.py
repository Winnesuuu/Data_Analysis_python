from glob import glob
import pandas as pd
from numpy import log, mean
from local_config import config
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold

def estimate_probability(the_X, the_model):
    predicted = the_model.predict_proba(the_X)
    return pd.DataFrame(data=predicted, columns=['prob_lowincome','prob_highincome'], index=the_X.index)

def calc_log_loss(x):
    if x['incomelevel'] == 1:
        return -log(x['prob_highincome'])
    return -log(1-x['prob_highincome'])


def create_df():

    # load data
    print ('loading data...')
    global df
    fList = glob('adult1.csv' % (config['data_path'], config['slash']))
    dfList = list()
    for f in fList:
        print('loading %s' % (f))
        df = pd.read_csv(f,names = ["age", "workclass","fnlwgt","education","education_num","marital_status", "occupation", \
                    "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native_country","incomelevel"])
        #print(len(df), len(df.columns))
        dfList.append(df)

    df = pd.concat(dfList)
    #print(len(df), len(df.columns))

    #only use grade E for this in-class example
    #df = df[df.grade=='E']

    return df

#some data visualization to get a sense of the distribution in the data
# =============================================================================
# plt.hist(df['age'])
# df['sex'].value_counts().plot(kind='bar')
# plt.hist(df['education_num'])
# df['race'].value_counts().plot(kind='bar')
# =============================================================================


#high_earner = ['>50K.']

def income_level(i):
    income = i.find('<=50K')
    if income == -1:
        return 1
    return 0 

def country(i):
    country = i.find('United-States')
    if country == -1:
        return 0
    return 1
    

def clean_df(df):

    df = df.replace(r' *\? *',np.NaN, regex=True)
    df = df.dropna()
    df = df.drop(columns=['capital-loss'])
    df = df.drop(columns=['capital-gain'])
    df = df.drop(columns=['education'])
    df["incomelevel"] = df["incomelevel"].apply(income_level)
    df["native_country"] = df["native_country"].apply(country)

    map1 = {' Female':0, ' Male':1}
    df['sex'] = df['sex'].map(map1)
    map2 = {' Amer-Indian-Eskimo':0, ' Asian-Pac-Islander':1, ' Black':2, ' Other':3, \
       ' White':4}
    df['race'] = df['race'].map(map2)


    return df 


def normalize_df(df,features):
    scaler = StandardScaler().fit(df[features])
    df[features] = StandardScaler().fit_transform(df[features])
    return df, scaler

def split_df(df):
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df, df['incomelevel'], test_size=0.2, stratify=df['incomelevel'], random_state=0) #0 to ensure repeatability; None for a different state each time
    return df_X_train, df_X_test, df_y_train, df_y_test

def prediction(df_X_train, df_X_test, df_y_train, df_y_test, features, c=100.0, use_cv=False, coef_df=pd.DataFrame()):

    if use_cv:
        model = LogisticRegressionCV(tol=1.0e-4, penalty='l2', Cs=25, fit_intercept=True, n_jobs=5, cv=StratifiedKFold(n_splits=5),scoring='neg_log_loss', solver='liblinear', refit=True, random_state=0)
    else:
        model = LogisticRegression(tol=1.0e-4, penalty='l2', C=c, fit_intercept=True, warm_start=True, solver='liblinear')

    model.fit(df_X_train[features], df_y_train)

    #use model to make in-sample and out-of-sample predictions
    df_is = estimate_probability(df_X_train[features],model)
    df_X_train = pd.concat([df_X_train,df_is], axis=1, join='outer')
    df_X_train['log_loss'] = df_X_train.apply(calc_log_loss,1)
    log_loss_is = mean(df_X_train.log_loss.values)

    df_oos = estimate_probability(df_X_test[features],model)
    df_X_test = pd.concat([df_X_test,df_oos], axis=1, join='outer')
    df_X_test['log_loss'] = df_X_test.apply(calc_log_loss,1)
    log_loss_oos = mean(df_X_test.log_loss.values)

    # add coefficients + more data to dataframe & label them
    if use_cv: c = model.C_[0]
    coef_df = coef_df.append(pd.Series([c,log_loss_is,log_loss_oos,model.intercept_[0]] + model.coef_[0].tolist()),ignore_index=True)
    coef_df.columns = ['c','log_loss_is','log_loss_oos','intercept'] + features

    return df_X_train, df_X_test, coef_df


def run(use_cv=False):

    features = ['age','education_num','sex','race','native_country']
    df = create_df()
    df = clean_df(df)
    df, scaler = normalize_df(df,features)
    df_X_train, df_X_test, df_y_train, df_y_test = split_df(df)
    df_X_train, df_X_test, coef_df = prediction(df_X_train, df_X_test, df_y_train, df_y_test, features, use_cv=use_cv, c=100.0)

    return coef_df


if __name__ == '__main__':
    coef_df = run()