
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import itertools


# In[3]:

def _transform(df):
    df.Date=pd.to_datetime(df.Date)
        
    cat_cols=['StateHoliday']
    for col in cat_cols:
        df[col] = df[col].astype(np.str)
        df[col] = df[col].astype('category')
  
    #column DayOfWeek is redundant, remove it
    df=df.drop('DayOfWeek',axis=1)
    
    return df
    
def get_train():   
    df=pd.read_csv('train.csv')
    df.Date=pd.to_datetime(df.Date)

    #fill missing indices
    dr=pd.date_range(df.Date.min(), df.Date.max(),freq=pd.datetools.day)
    idx_df=pd.DataFrame(list(itertools.product(dr,range(1,1116))),columns=['Date','Store']) #complete index
    df=idx_df.merge(df,on=['Date','Store'],how='left')

    #merge will convert categoryical to object type, so I put transform here.
    df= _transform(df) 
    return df

    
def get_test():
    test=pd.read_csv('test.csv')
    return _transform(test)
    

    
def get_store():
    df=pd.read_csv('store.csv')
    
    cat_cols=['StoreType','Assortment','PromoInterval']
    for col in cat_cols:
        df[col] = df[col].astype(np.str)
        df[col] = df[col].astype('category') 

    return df


# In[5]:

def drop_incomplete_stores(df): # drop stores that has NaN in Sales.
    df=df.copy()
    ind=df.Sales.isnull()
    black_list=df[ind].Store.unique()
    df=df[~df.Store.isin(black_list)]
    return df

def sample_df(df,store_frac=1,time_range=['2014-01-01','2014-12-31'],drop_na_store=True, use_days_from=False,reindex=False):
    """
    drop_na_store only drop that has NaN sales in time_range.
    use_days_from use days from first day of time range
    
    """
    df=df.copy()
    df=df[df.Date.between(*time_range)]
    
    if drop_na_store:
        df=drop_incomplete_stores(df)

    stores=pd.Series(df.Store.unique()).sample(frac=store_frac).unique()
    df=df[df.Store.isin(stores)]
    
    if use_days_from:
        df.Date=(df.Date-pd.Timestamp(time_range[0])).dt.days

    if reindex:
        df=df.sort(columns=['Date','Store'])
        df=df.reset_index(drop=True)
    # the doc says the index need to be sequential for time series plot (that's )
    #if there's missing(NaN or absent from records) of subject at some timepoints, the tsplot will be problematic of "interploation"

    return df


# In[6]:

# !! plz save the ipynb in advance
if __name__=='__main__': 
    get_ipython().system('ipython nbconvert --to=python utility.ipynb')


# In[ ]:




# In[ ]:



