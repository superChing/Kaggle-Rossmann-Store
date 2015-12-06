# coding: utf-8
import itertools
import numpy as np
import pandas as pd
import sklearn_ext
# from sklearn import metrics
from sklearn.utils.validation import check_array
# from sklearn.base import TransformerMixin, BaseEstimator

# ======= IO ==========
from io import StringIO

str_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
str_dtype = {k: np.str for k in str_cols}


def _transform(df):
    df.Date = pd.to_datetime(df.Date)

    for col in str_dtype:
        if col in df.columns:
            df[col] = df[col].astype(np.str)

    # column DayOfWeek is redundant, remove it
    df = df.drop('DayOfWeek', axis=1)
    df = df.sort_values(by=['Store', 'Date'])
    df = df.reset_index(drop=True)
    return df


# def imputeSomeX(train_df:pd.DataFrame):
#     df=train_df
#     cols=['Promo','StateHoliday','SchoolHoliday','Open']
#     for col in cols:
#         train_df[col].


def load_train(string=None):
    if string:
        df = pd.read_csv(StringIO(unicode(string)), dtype=str_dtype)
    else:
        df = pd.read_csv('data/' + 'train.csv', dtype=str_dtype)

    df.Date = pd.to_datetime(df.Date)

    # fill missing indices
    dr = pd.date_range(df.Date.min(), df.Date.max(), freq=pd.datetools.day)
    idx_df = pd.DataFrame(list(itertools.product(dr, range(1, 1116))), columns=[
        'Date', 'Store'])  # complete index
    df = idx_df.merge(df, on=['Date', 'Store'], how='left')

    # merge will convert categoryical to object type, so I put transform here.
    df = _transform(df)
    df = imputeSomeX(df)
    return df


def load_test(string=None):
    if string:
        df = pd.read_csv(StringIO(unicode(string)), dtype=str_dtype)
    else:
        df = pd.read_csv('data/' + 'test.csv', dtype=str_dtype)

    # 6 missing Open happens in test data.
    # assume Open since otherewise 0 sales will not count into score by its
    # evaluation defifnition.
    df.loc[df['Open'].isnull(), 'Open'] = 0
    return _transform(df)


def load_store(string=None):
    if string:
        df = pd.read_csv(StringIO(unicode(string)), dtype=str_dtype)
    else:
        df = pd.read_csv('data/' + 'store.csv', dtype=str_dtype)

    for col in str_dtype:
        if col in df.columns:
            df[col] = df[col].astype(np.str)

    return df


# ============= MISC ==============

def drop_incomplete_stores(df):  # drop stores that has NaN in Sales.
    df = df.copy()
    ind = df.Sales.isnull()
    black_list = df[ind].Store.unique()
    df = df[~df.Store.isin(black_list)]
    return df


def sample_df(df, store_frac=1, time_range=['2014-01-01', '2014-12-31'], drop_na_store=True, use_days_from=False,
              reindex=False):
    """
    drop_na_store only drop that has NaN sales in time_range.
    use_days_from use days from first day of time range

    """
    df = df.copy()
    df = df[df.Date.between(*time_range)]

    if drop_na_store:
        df = drop_incomplete_stores(df)

    stores = pd.Series(df.Store.unique()).sample(frac=store_frac).unique()
    df = df[df.Store.isin(stores)]

    if use_days_from:
        df.Date = (df.Date - pd.Timestamp(time_range[0])).dt.days

    if reindex:
        df = df.sort(columns=['Date', 'Store'])
        df = df.reset_index(drop=True)
    # the doc says the index need to be sequential for time series plot (that's )
    # if there's missing(NaN or absent from records) of subject at some
    # timepoints, the tsplot will be problematic of "interploation"

    return df



def loss(y, yhat):
    y = check_array(y, ensure_2d=False)
    yhat = check_array(yhat, ensure_2d=False)
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if yhat.ndim == 1:
        yhat = yhat.reshape((-1, 1))
    if y.shape[1] != yhat.shape[1]:
        raise ValueError("y_true and y_pred have different number of output ({0}!={1})".format(
            y.shape[1], yhat.shape[1]))

    # y=0 is ignored
    yhat = yhat[y != 0]
    y = y[y != 0]
    return np.sqrt(np.mean(((y - yhat) / y) ** 2))


def total_rmspe(rmspes, len_test):
    """
    單獨訓練每個subject, 結合各subject的分數。
    case1: 每個element都是 a rmspe error of a time series.
    case2: 每個row都是cv errors of a time series. i.e. input is 2d array.
            這個case要column為單位計算rmspe然後再平均cv errors.
    """

    def _totalrmspe(rmspes):  # rmspes is a 1d vector , i.e. case 1
        sq_errors = rmspes ** 2 * len_test
        return np.sqrt(sum(sq_errors) / (len(sq_errors) * len_test))

    rmspes = np.array(rmspes)
    if rmspes.ndim == 2:  # cv case
        return np.mean([_totalrmspe(rmspes[:, i]) for i in range(rmspes.shape[1])])
    else:
        return _totalrmspe(rmspes)

# ============ Features =================

def basic_features(df, store_info):
    """merge store, and extracts date features.
    Args:
        df (dataframe): dataframe
    Returns:
        datframe: index-preserved (and order-preserved)
        to_be_dropped: columns that useless
    """
    to_be_dropped = []  # columns to be dropped
    # ======= join sotre and preserve index========
    index_snapshot = df.index
    index_name = df.index.name or 'index'
    df = df.reset_index()  # put index into columns
    # join columns will drop index
    df = df.merge(store_info, on='Store', how='left', )
    df = df.set_index(index_name, drop=True)  # put index back
    df = df.reindex(index_snapshot)

    # ====== time features =========
    df['Year'] = df.Date.dt.year
    df['Quarter'] = df.Date.dt.quarter
    df['Month'] = df.Date.dt.month
    df['YearWeek'] = df.Date.dt.weekofyear
    df['MonthDay'] = df.Date.dt.day
    df['DaysInMonth'] = df.Date.dt.days_in_month
    df['WeekDay'] = df.Date.dt.weekday
    df['YearDay'] = df.Date.dt.dayofyear
    df['MonthEnd'] = df.Date.dt.is_month_end
    df['MonthStart'] = df.Date.dt.is_month_start
    df['QStart'] = df.Date.dt.is_quarter_start
    df['QEnd'] = df.Date.dt.is_quarter_end
    df['YStart'] = df.Date.dt.is_year_start
    df['YEnd'] = df.Date.dt.is_year_end
    to_be_dropped += ['Date']

    # Calculate time since competitor open
    df['TimeSinceCompetitionOpen'] = \
        12 * (df.Year - df.CompetitionOpenSinceYear) + \
        (df.Month - df.CompetitionOpenSinceMonth)
    to_be_dropped += ['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']

    df['TimeSincePromo2Open'] = \
        12 * (df.Year - df.Promo2SinceYear) + \
        (df.YearWeek - df.Promo2SinceWeek) / 4.0
    to_be_dropped += ['Promo2SinceYear', 'Promo2SinceWeek']

    # is Promo2-ing
    str2month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df['IsPromo2Month'] = 0
    for interval in df.PromoInterval.unique():
        if interval not in ('nan', '', 'NaN'):
            for monthStr in interval.split(','):
                msk = (df.Month == str2month[monthStr]) & (
                    df.PromoInterval == interval)
                df.loc[msk, 'IsPromo2Month'] = 1
    # validate
    assert np.all(index_snapshot == df.index)
    return df, to_be_dropped


def dynamic_features(traintest,
                     event_columns=[], event_offsets=[],
                     state_columns=[], state_offsets=[],
                     conti_columns=[], time_offsets=[]):
    """not order-preserved
    numerical_columns (list): features with continuous value.

    CAVEAT: because the short of information at start and end of sequence, the result ends with NaNs(may be large) at start or end of sequence.
    CAVEAT: if data has nan, result will peculiar.  every nan will be treated as different states.  , [nan, nan] are seen as two different state.

    """
    # _df=pd.concat([train,test],axis=0,ignore_index=True,)
    # _traintest=traintest.sort_values(by=['Store','Date']).reset_index(drop=True)
    state_lags = np.array(state_offsets) * -1
    time_lags = np.array(time_offsets) * -1
    event_lags = np.array(event_offsets) * -1

    vertiacl_accum = []
    for store in traintest['Store'].unique():
        _df = traintest[traintest['Store'] == store].sort_values(by=['Date'])

        diff = _df['Date'][1:].values - _df['Date'][:-1].values
        # check it is common difference arithmetic sequence
        assert np.all(diff[1:] == diff[:-1])
        
        df1 = sklearn_ext.state_duration_features(_df[state_columns])
        df2 = sklearn_ext.state_dynamic_features(_df[state_columns], lags=state_lags)
        df3 = sklearn_ext.event_features(_df[event_columns])
        df4 = sklearn_ext.state_dynamic_features(_df[event_columns], lags=event_lags)
        df5 = sklearn_ext.continuous_dynamic_features(_df[conti_columns], lags=time_lags)
        index_df = _df[['Date', 'Store']]
        df = pd.concat([index_df, df1, df2, df3, df4, df5], axis=1)
        vertiacl_accum.append(df)

    df = pd.concat(vertiacl_accum, axis=0)  # .reset_index(drop=True)
    assert len(df) == len(traintest)
    return traintest.merge(df, on=['Store', 'Date'], how='left')
