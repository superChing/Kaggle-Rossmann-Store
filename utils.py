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
dtypes = {k: np.str for k in str_cols}


def _impute_via_last_year(df):
    """ assume date-store dataframe"""
    one_year = pd.Timedelta(364, 'D')
    # the index of rows that has nan
    nan_index = df.index[df.isnull().any(axis=1)]
    df.loc[nan_index, :] = df.loc[nan_index - one_year, :].values
    return df


def _impute_a_feature(df):
    """df is a date-by-store dataframe"""

    def _find_complete_col(df, cols):  # first complete data column(no nan)
        for col in cols:
            if (not df[col].isnull().any()):
                yield col

    # select rows/dates that are complete(no nan).
    nan_msk_by_date = df.isnull().apply(np.sum, axis=1).astype(np.bool)
    complete_sub_df = df[~nan_msk_by_date]
    # store-index groups that has same values
    store_groups_by_pattern = complete_sub_df.T.groupby(complete_sub_df.T.columns.tolist()).groups.values()
    for stores in store_groups_by_pattern:
        try:
            col = next(_find_complete_col(df, stores))
        except StopIteration as e:
            # raise ValueError("there's no complete data in this group : {}".format(stores)) from e
            # print("there's no complete data in this group : {}".format(stores))
            # print('attempt to impute via last year.')
            df.loc[:, stores] = _impute_via_last_year(df[stores].copy())
        else:
            # assign all stores to the same
            for store in stores:
                df[store] = df[col]
    return df


def impute(df, columns):
    """impute nans for some columns that is imputable by simple logic.
        # cols = ['Promo', 'StateHoliday', 'SchoolHoliday', ]  # 'Open']
    df=pd.DataFrame({'a':[1,2,3,np.nan,5],'b':[1,2,3,4,5],'c':[0,1,2,np.nan,4]},
         index = pd.to_datetime(['20140101', '20140102','20140103', '20150101','20150102']))
    """

    shape_shot = df.shape
    for col in columns:
        # df is a date-by-store dataframe, is incomplete -- at some dates all stores got nan.
        _df = df.pivot(index='Date', columns='Store', values=col)
        _df = _impute_a_feature(_df)
        df = df.set_index(['Date', 'Store'])
        df[col] = _df.stack()
        df = df.reset_index()

    assert shape_shot == df.shape

    # impute BV state
    # msk = (df.Date.between('2014-03-03', '2014-03-07') |
    #        df.Date.between('2014-04-14', '2014-04-26') |
    #        df.Date.between('2014-06-10', '2014-06-21') |
    #        df.Date.between('2014-07-30', '2014-09-15') |
    #        df.Date.between('2014-10-27', '2014-10-31') |
    #        df.Date.between('2014-12-24', '2015-01-05'))
    # df.loc[df.Sales.isnull(), 'SchoolHoliday'] = 0
    # df.loc[df.Sales.isnull() & msk, 'SchoolHoliday'] = 1
    return df


def load_train(string=None):
    if string:
        df = pd.read_csv(StringIO(unicode(string)), dtype=dtypes)
    else:
        df = pd.read_csv('data/' + 'train.csv', dtype=dtypes)

    df.Date = pd.to_datetime(df.Date)

    # fill missing indices
    dr = pd.date_range(df.Date.min(), df.Date.max(), freq=pd.datetools.day)
    idx_df = pd.DataFrame(list(itertools.product(dr, range(1, 1116))), columns=[
        'Date', 'Store'])  # complete index
    df = idx_df.merge(df, on=['Date', 'Store'], how='left')

    # merge will convert categoryical to object type, so I put transform here.
    df.Date = pd.to_datetime(df.Date)
    df = df.drop('DayOfWeek', axis=1)

    df = impute(df, ['Promo', 'StateHoliday', 'SchoolHoliday', 'Open'])
    df = df.sort_values(by=['Store', 'Date'])
    df = df.reset_index(drop=True)
    return df


def load_test(string=None):
    if string:
        df = pd.read_csv(StringIO(unicode(string)), dtype=dtypes)
    else:
        df = pd.read_csv('data/' + 'test.csv', dtype=dtypes)

    # 6 missing Open happens in test data.
    # assume Open since otherewise 0 sales will not count into score by its
    # evaluation defifnition.
    df.loc[df['Open'].isnull(), 'Open'] = 0

    df.Date = pd.to_datetime(df.Date)
    df = df.drop('DayOfWeek', axis=1)

    df = df.sort_values(by=['Store', 'Date'])
    df = df.reset_index(drop=True)
    return df


def load_store(string=None, state=None):
    if string:
        store_df = pd.read_csv(StringIO(unicode(string)), dtype=dtypes)
        state_df = pd.read_csv(StringIO(unicode(state)), dtype=dtypes)
    else:
        store_df = pd.read_csv('data/' + 'store.csv', dtype=dtypes)
        state_df = pd.read_csv('data/' + 'state.csv', dtype=dtypes)

    state_df['State'] = state_df.State.str.replace('HB,NI', 'NI')  # NI比較大
    df = store_df.merge(state_df, on='Store', how='left')
    return df


def load_holiday(string=None):
    if string:
        holiday_df = pd.read_csv(StringIO(unicode(string)), dtype=dtypes)
    else:
        holiday_df = pd.read_csv('data/' + 'holiday.csv', dtype=dtypes)

    df = holiday_df
    df[['Month', 'Day']] = df['Date'].str.strip().str.split().apply(lambda x: pd.Series(x))
    str2month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df['Month'] = df['Month'].replace(str2month)
    df['Date'] = pd.to_datetime(df.apply(lambda x: '{}-{}-{}'.format(x.Year, x.Month, x.Day), axis=1))
    df = df[['Date', 'Holiday name', 'Holiday type', 'Where it is observed']].copy()
    df.columns = ['Date', 'HolidayName', 'HolidayType', 'HolidayStates']
    df['HolidayStates'] = df.HolidayStates.fillna(
        "BW, BY, BE, BB, HB, HH, HE, MV, NI, NW, RP, SL, SN, ST, SH, TH")
    df['HolidayName'] = df.HolidayName.str.strip()
    df['HolidayType'] = df.HolidayType.str.strip()
    assert not df['Date'].duplicated().any()
    return df


def holiday_features(main, holiday):
    """
    main dataframe is indexed by data and store and has column about store's state .
    join hoiday data by date, and check if the holiday should happen to that state, set it to nan if not."""
    df = main.merge(holiday, on=['Date'], how='left')

    def _is_in_holiday_states(row):
        states_in_holiday = row['HolidayStates']
        if not pd.isnull(states_in_holiday) :
            res = row['State'] in states_in_holiday
        else:
            res = False
        return res

    in_holiday_msk =  df[['State', 'HolidayStates']].apply(_is_in_holiday_states, axis=1)
    df.loc[in_holiday_msk, 'IsHoliday'] = 1
    assert len(df) == len(main)
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


def loss(Y, Yhat, sales_idx=0):
    # turn into numpy array
    Y = check_array(Y, ensure_2d=False)
    Yhat = check_array(Yhat, ensure_2d=False)
    # turn into 1d array
    if Y.ndim != 1 and Y.shape[1] > 1:  # if 2d output
        y = Y[:, sales_idx]  # extract target , 1d array
    y = Y.reshape((-1,))  # 1d array
    if Yhat.ndim != 1 and Yhat.shape[1] > 1:
        yhat = Yhat[:, sales_idx]
    yhat = Yhat.reshape((-1,))

    assert y.ndim == yhat.ndim == 1, str((y.shape, yhat.shape))
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
    df.loc[df.Promo2SinceYear == 0, 'TimeSincePromo2Open'] = np.nan
    to_be_dropped += ['Promo2SinceYear', 'Promo2SinceWeek']

    # is Promo2-ing
    str2month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df['IsPromo2Month'] = 0
    for interval in df.PromoInterval.unique():
        if str(interval) not in ('nan', '', 'NaN', 'None'):
            for monthStr in interval.split(','):
                msk = (df.Month == str2month[monthStr]) & (df.PromoInterval == interval)
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


def padding(df, days=120):
    """可以做成context manager , 但是有些問題就是，df在context 裡會被copy然後modify，我拿不到reference"""
    one_year = pd.Timedelta(364, 'd')
    one_day = pd.Timedelta(1, 'd')
    padding_days = pd.Timedelta(days, 'd')

    condition = pd.Series([False] * len(df), index=df.index)
    for store in df.Store.unique():
        start_padding_day = df[df.Store == store].Date.max() + one_day
        end_padding_day = start_padding_day + padding_days
        condition = condition | \
                    (df.Date.between(start_padding_day - one_year, end_padding_day - one_year) & (df.Store == store))

    # # padding for test stores, subset of train
    #     start_padding_day=test_end+one_day
    #     end_padding_day=start_padding_day+padding_days
    #     condition1=df.Date.between(start_padding_day-one_year,end_padding_day-one_year) & \
    #                 df.Store.isin(test.Store.unique())
    #     # padding for non-test stores
    #     start_padding_day=train_end+one_day
    #     end_padding_day=start_padding_day+padding_days
    #     condition2=df.Date.between(start_padding_day-one_year,end_padding_day-one_year) & \
    #                (~df.Store.isin(test.Store.unique()) )
    #     condition=condition1|condition2

    _df = df[condition].copy()
    _df.Date = _df.Date + one_year
    padded_df = pd.concat([df, _df], axis=0).reset_index(drop=True)

    return padded_df


def depadding(df, days=120):
    one_day = pd.Timedelta(1, 'd')
    padding_days = pd.Timedelta(days, 'd')
    # 從後面剪掉padding的天數
    df = df.reset_index(drop=True)
    condition = pd.Series([False] * len(df), index=df.index)
    for store in df.Store.unique():
        start_padding_day = df[df.Store == store].Date.max() - padding_days
        condition = condition | ((df.Date < start_padding_day) & (df.Store == store))

    # train_end=train_df.Date.max()
    #     test_end=test_df.Date.max()
    #     start_padding_day=test_end+one_day
    #     condition1=(df.Date<start_padding_day) & df.Store.isin(test_df.Store.unique())
    #     start_padding_day=train_end+one_day
    #     condition2=(df.Date<start_padding_day )& (~df.Store.isin(test_df.Store.unique()))
    #     df=df[condition1 | condition2 ]

    return df[condition].reset_index(drop=True)


def moving_avg(df, col, ma_degree, min_periods_ratio=0.7):
    new = pd.Series(index=df.index, name="{0}_MA{1}".format(col, str(ma_degree)))
    for store in df.Store.unique():
        s = df[df.Store == store][col].copy()
        s = pd.rolling_window(s, ma_degree, 'boxcar', center=True, min_periods=int(ma_degree * min_periods_ratio))
        new.loc[s.index] = s
    return new


def plot_timeseries(train_df, prediction_df, store, time_range=['2014-01-01', '2014-12-31'], title=None):
    from bokeh.io import show
    from bokeh.plotting import figure
    df = train_df[(train_df.Store == store) & (train_df.Date.between(*time_range))]
    yhat = prediction_df[(prediction_df.Store == store) & (prediction_df.Date.between(*time_range))]

    # preprocess
    state_holiday = df.Date[(df.StateHoliday != '0') & (df.StateHoliday != 'nan')]
    school_holiday = df.Date[df.SchoolHoliday == 1]
    weekend = df.Date[(df.Date.dt.weekday == 5) | (df.Date.dt.weekday == 6)]
    promo_ranges = []
    for k, g in itertools.groupby(df[['Date', 'Promo']].values, lambda x: x[1]):  # group by Promo
        if k == 1:
            g = list(map(lambda x: x[0], g))
            promo_ranges.append((g[0], g[-1]))

    # ---------------------- plot -----------------------------
    from bokeh.models import HoverTool, BoxAnnotation, ColumnDataSource
    TOOLS = 'wheel_zoom,pan,resize,reset'
    p = figure(x_axis_type="datetime", plot_width=1000, plot_height=400, tools=TOOLS)

    # sales
    source = ColumnDataSource(data=df)
    p.line('Date', 'Sales', source=source)
    p.circle('Date', 'Sales', size=3, source=source)

    source = ColumnDataSource(data=yhat)
    p.line('Date', 'yhat', source=source)
    p.circle('Date', 'yhat', size=3, source=source)

    # state holiday
    p.ray(x=state_holiday, y=0, length=0, angle=np.pi / 2, color='red', line_dash=[5, 5])
    p.ray(x=state_holiday, y=0, length=0, angle=-np.pi / 2, color='red', line_dash=[5, 5])
    # schoold holiday
    p.ray(x=school_holiday, y=0, length=0, angle=np.pi / 2, color='green', line_dash=[1, 8])
    p.ray(x=school_holiday, y=0, length=0, angle=-np.pi / 2, color='green', line_dash=[1, 8])
    # weekend??
    p.ray(x=weekend, y=0, length=0, angle=np.pi / 2, color='yellow', line_dash=[2, 5])
    p.ray(x=weekend, y=0, length=0, angle=-np.pi / 2, color='yellow', line_dash=[2, 5])
    # promo  , the band is inclusive
    boxs = [BoxAnnotation(plot=p, left=l.value / 1e6, right=r.value / 1e6, fill_alpha=0.1, fill_color='green') for l, r
            in promo_ranges]
    p.renderers.extend(boxs)

    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Sale"
    if title:
        p.title = title
    else:
        p.title = 'store' + str(store)
    hover = HoverTool()
    hover.tooltips = {"value": "$y", "time": '$x'}
    p.add_tools(hover)

    show(p)  # show the results


def predict_via_mean(train, test):
    # TODO 加重89月,加入Holiday
    # LB 0.13952
    # prediction file: https://www.kaggle.com/shearerp/rossmann-store-sales/store-dayofweek-promo-0-13952
    # This model predicts the geometric mean of past sales grouped by Store,DayOfWeek,Promo.
    train = train.copy()
    test = test.copy()
    train['WeekDay'] = train.Date.dt.weekday
    test['WeekDay'] = test.Date.dt.weekday

    features = ['Store', 'DayOfWeek', 'Promo']
    train = train[train.Sales > 10]
    model = train.groupby(features).apply(lambda subdf: np.expm1(np.mean(np.log1p(subdf['Sales']))))
    model.name = 'Prediction'
    predict = test.merge(model, how='left', on=['Store', 'DayOfWeek', 'Promo'])[['Id', 'Prediction']]
    predict = predict.fillna(0)
    return predict
