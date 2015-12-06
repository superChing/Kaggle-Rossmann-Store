# coding: utf-8
import itertools

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder


class TimeSeriesCVSplit:
    """
    data split for cross validation in **forecasting** task.

    What it differs from KFolds is that :
        1. validation folds may overlapping,
        2. train folds never exceed over validation in terms of time.
            i.e. the index lates than validation's are dropped.

    Parameter
    ===
    timeindex (int):  train's timeindex, shoud be integer.
    vali_len (int): len(time span of validation data), usually a test_time_span
    step (int): distance between every validaiton fold.
        step * n_folds= days covered by validation.

    Example
    ===
    In:
        dt=pd.Series(pd.date_range('2015/1/1','2015/1/5',unit='d'))
        dt=(dt-dt.min()).dt.days
        dt=dt.append(dt)  # dt x 2  = [1,2,3,4,5,1,2,3,4,5]
        [(x,y) for x,y in TimeSeriesCVSplit(dt,2,2,2)]

    Out:
        [(array([0, 1, 2, 5, 6, 7]), array([3, 4, 8, 9])),
         (array([0, 5]), array([1, 2, 6, 7]))]

    """

    def __init__(self, timeindex, n_folds, vali_len, step):
        self.n_folds = n_folds
        self.timeindex = pd.Series(timeindex)

        self.time_end = self.timeindex.max()

        self.vali = vali_len
        self.step = step

    def __iter__(self):
        t_end = None
        for i in range(self.n_folds):
            t_end = t_end - self.step if t_end else self.time_end + 1
            t_start = t_end - self.vali
            vali_idx = np.where(self.timeindex.between(t_start, t_end - 1))[0]  # between is inclusive
            train_idx = np.where(self.timeindex < t_start)[0]
            yield train_idx, vali_idx

    def __len__(self):
        return self.n_folds


class ObjIndexer(TransformerMixin, BaseEstimator):
    """turn object type columns of dataframe into random ordinal encoding.
    Args:
        ordinal : ??
    Returns:
        dataframe

    TODO:
        handle unseen values
        ordinal indexing if specified.
        unit test.
    """

    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, Xdf, y=None):
        if self.columns:
            _columns = self.columns
        else:
            _columns = filter(lambda x: Xdf[x].dtype.kind == "O", Xdf.columns)

        # random ordinal encoding for categorical columns
        random_indexers = {}
        for col in _columns:
            random_indexers[col] = LabelEncoder().fit(Xdf[col])
        self.random_indexers_ = random_indexers
        # #ohe
        # ohe_transformers = {}
        # for col in self.onehot_columns :
        #     x=Xdf[[col]]
        #     x=random_indexers[col].transform(x)
        #     ohe_transformers[col]= OneHotEncoder(sparse=False).fit(x)
        # self.ohe_transformers_=ohe_transformers

        # if self.ordinal: raise NotImplementedError()

        # from collections import defaultdict
        # transform=defaultdict(list)
        # for col,transf in random_indexers.items():
        #     transform[col].append(transf.transform)
        # # for col,transf in ohe_transformers.items():
        # #     transform[col].append(transf.transform)
        #
        # self.transform_ =transform
        return self

    def transform(self, Xdf, y=None):
        Xdf = Xdf.copy()
        for col, trans in self.random_indexers_.items():
            Xdf[col] = trans.transform(Xdf[col])

        # xs=[]
        # for col,transforms in self.transform_.items():
        #     x=Xdf[[col]]
        #     for f in transforms: x=f(x)
        #     xs.append(x if x.ndim==2 else x.reshape(-1,1))
        #
        # codes = np.hstack(xs)
        #
        # # remove original columns
        # Xdf = Xdf[Xdf.columns.difference(self.transform_.keys())]
        # X = np.hstack([Xdf.values, codes])
        # return X
        return Xdf


# TODO
# port it to comform sklearn API , so that we can use it in Pipeline. see RossmannStore utils.dynamic_features
# reference: rossmann_store.utils.dynamic_features
def state_duration_features(df):
    """ extract from categorical columns the time interval features.

    given a categorical column that represent a state of the subject 'along time'.
    extract feature of:
     1. time elapse since state start/last state switching
        column: TimeFromStart_{col}, TimeToEnd_{col}
     2. time interval/duration of state the subject are being in (that may take into account of future data).
        column: StateSpan_{col}

    **ASSUMPTION**:
        the dataframe's index is time increasing and continuous i.e. an common difference arithmetic sequence.

    Examples:
    In:
         df=pd.DataFrame([0,1,1,1,2,1],columns=['a'])
         print(state_duration_features(df))
    Out:
           TimeFromStart_a  StateSpan_a  TimeToEnd_a
        0              NaN          NaN            0
        1                0            3            2
        2                1            3            1
        3                2            3            0
        4                0            1            0
        5                0          NaN          NaN

    """
    # diff=df.index[1:].values-df.index[:-1].values
    # assert np.all(diff[1:]==diff[:-1] )# check it is common difference arithmetic sequence
    new_features = pd.DataFrame(index=df.index)
    from itertools import chain
    for col in df.columns:
        gen = itertools.groupby(df[col])
        groups = list(list(v) for k, v in gen)  # this is 'local' groupby ,e.x. [[1,1,1],[0,0]]

        elapse = np.array(list(x for x in chain.from_iterable(range(0, len(g)) for g in groups)),
                          np.float)  # bug np and pd dont eat chain
        elapse[:len(groups[0])] = np.nan
        new_features['TimeFromStart_' + col] = elapse

        span = np.array(list(chain.from_iterable([len(g)] * len(g) for g in groups)), np.float)
        span[:len(groups[0])] = np.nan
        span[-len(groups[-1]):] = np.nan
        new_features['StateSpan_' + col] = span

        to_end = np.array(list(chain.from_iterable(range(len(g) - 1, -1, -1) for g in groups)), np.float)
        to_end[-len(groups[-1]):] = np.nan
        new_features['TimeToEnd_' + col] = to_end

    return new_features


def event_features(df):
    """time elapse since the last k event

    the event time series is spikish state sequence, that doen't have state duration,
    e.g. Holiday by daily series . that 0 means no event, 1 means event.
    It can not be captured by status duration or elapse by using state_duration_features.

    add column: ElapseSinceLastEvent_{col}

    # TODO
    **CAVEAT:** for expedient I just set it = ElapseSinceLastState_{col}.shift(1),
        its not proper, wrong count happens two steps after events.

    **ASSUMPTION**:
        the dataframe's index is time increasing and continuous i.e. an common difference arithmetic sequence.

    Examples:
            df=pd.DataFrame([0,1,0,0,0,0,1,0,0,0],columns=['a'])
            print(event_features(df).T)
        Out:
                            0   1  2  3  4  5  6   7   8   9
            TimeFromLast_a NaN NaN  0  0  1  2  3   0   0   1
            TimeToNext_a     1   1  4  3  2  1  1 NaN NaN NaN
    """
    new_features = pd.DataFrame(index=df.index)
    from itertools import chain
    for col in df.columns:
        gen = itertools.groupby(df[col])
        groups = list(list(v) for k, v in gen)  # this is 'local' groupby ,e.x. [[1,1,1],[0,0]]
        elapse = np.array(list(chain.from_iterable(range(0, len(g)) for g in groups)),
                          np.float)  # bug np and pd dont eat chain
        elapse[:len(groups[0])] = np.nan
        new_features['TimeFromLast_' + col] = np.concatenate([[np.nan], elapse[:-1]])

        timeto = np.array(list(chain.from_iterable(range(len(g), 0, -1) for g in groups)),
                          np.float)  # bug np and pd dont eat chain
        timeto[-len(groups[-1]):] = np.nan
        new_features['TimeToNext_' + col] = timeto
    return new_features


def state_dynamic_features(df, lags=[]):
    """dynamic of states , it's not about time dynamic.

    1. switching dynamic -- what is last or next k states .
       columns with dynamic of only 2 states  is redundant , would be skipped.
       Add column {col}'_lag'{lag}
    2. dynamic of each specific state.
            If you want it then one hot encode it then numerical_dynamic_features.

    **ASSUMPTION**:
        the dataframe's index is time increasing and continuous i.e. an common difference arithmetic sequence.

    Args:
        lags (int): accept negative. lag 0 = original.
    Examples:
        df=pd.DataFrame([3,1,1,1,2],columns=['a'])
        print(state_dynamic_features(df,lags=[1,2]))
        Out:
               a_lag1  a_lag2
            0     NaN     NaN
            1       3     NaN
            2       3     NaN
            3       3     NaN
            4       1       3

    """
    new_features = pd.DataFrame(index=df.index)
    from itertools import chain
    for col in df.columns:
        gen = itertools.groupby(df[col])
        groups = list(list(v) for k, v in gen)  # this is 'local' groupby ,e.x. [[1,1,1],[0,0]]
        # 2 state dynamic is redundant , skip
        if df[col].nunique(dropna=True) == 2:
            continue
        else:
            group_len = [len(g) for g in groups]
            group_values = [g[0] for g in groups]
            for lag in lags:
                if lag > 0:  # shift +
                    lag_group_values = [np.nan] * lag + group_values[:-lag]
                else:  # or -
                    lag_group_values = group_values[-lag:] + [np.nan] * -lag

                lag_values = chain.from_iterable([v] * length for length, v in zip(group_len, lag_group_values))
                lag_values = list(lag_values)
                new_features[col + '_lag' + str(lag)] = lag_values
    return new_features


def continuous_dynamic_features(df, lags=[]):
    """ extract from numerical columns the dynamic features.
    """
    new_features = pd.DataFrame(index=df.index)
    for col in df.columns:
        for lag in lags:
            lag_values = df[col].shift(lag).values
            new_features[col + '_lag' + str(lag)] = lag_values
    return new_features


# def dynamic_prediction_monkey_patching(self,Xdf):
#     """ recursive prediction
#     **ASSUMPTION**:
#         1. X is sorted by time
#         2. Xdf is dataframe with "Sales_lag{x}" where x is lag number.
#
#     """
#     # check X
#     _Xdf=Xdf.copy()
#     _Xdf=_Xdf.sort_values(by=['Year','Month','Day'])
#
#     pattern='Sales_lag'
#     y_cols=filter(lambda x: pattern in x ,Xdf.columns)
#     map(lambda x: int(x[len(pattern):]), y_cols)
#
#     for store in Xdf['Store']
#     import collections
#     pred_history=collections.defaultdict(list)  # store:prediction-history mapping
#     for t,x in Xdf.iterrows():
#         store=x['Store']
#         for _t in range(1,t):
#             Xdf[]=pred_history[store].
#         if x.isnull().any(): TypeError('NaN in data')
#         y=self.predict(x)
#         pred_history[store].append(y)
