import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#https://github.com/portfolioplus/pytickersymbols
from pytickersymbols import PyTickerSymbols
import finnhub
from datetime import datetime
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
nltk.download("vader_lexicon")
from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, stem_text
from ta import add_all_ta_features 
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

#cache the stock tickers so they don't have to load everytime
@st.cache
def get_ticker():
    stock_data = PyTickerSymbols()
    indices = stock_data.get_all_indices()
    tickers = []
    for index in indices:
        tickers.append(stock_data.get_yahoo_ticker_symbols_by_index(index))
    tickers_flat = [item for sublist in tickers for item in sublist]
    tickers_flater = [item for sublist in tickers_flat for item in sublist]
    return sorted(tickers_flater)

#fetch all tickers
ticks = get_ticker()
ticks = list(np.unique(ticks))

#streamlit interactibles and headings
st.header('Using ML to predict the Stock Market')
st.subheader('A Big Data Project by Valeriia, Louisa and Alexander')
fin = st.sidebar.checkbox('Display company financials?')
show_news = st.sidebar.checkbox('Display company news?')
timeframe = st.sidebar.selectbox('Select period:',['max','1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd'])
slct = st.selectbox('Select your stock:',ticks, index = ticks.index("TSLA"))

#get the stock data from selected ticker
def get_stock(slct):
    stock = yf.Ticker(slct)
    return stock
  
stock = get_stock(slct)

#take historical data from stock in desired timeframe
hist = stock.history(period = timeframe)
hist['datetime']=hist.index
hist.reset_index()
hist['datetime']=hist['datetime'].astype('datetime64[ns]')

#plot historical data
st.set_option('deprecation.showPyplotGlobalUse', False)
_=plt.plot(hist.datetime, hist["Open"])
_=plt.xlabel("Date")
_=plt.ylabel("Opening")
st.pyplot()

#fetch and add company news
finnhub_client = finnhub.Client(api_key="c2si65iad3ic1qis06lg")
result=(finnhub_client.company_news(slct, _from="2020-11-15", to="2021-06-03"))
news=pd.DataFrame(result)
for i in range(news.shape[0]):
    news.datetime[i]= datetime.utcfromtimestamp(news.datetime[i]).strftime('%Y-%m-%d')
try:
    news['datetime']=news['datetime'].astype('datetime64[ns]')
    df=news.merge(hist,on='datetime',how="left")
    for i in range(df.shape[0]):
        df.summary[i] = strip_numeric(df.summary[i])
        df.summary[i] = strip_punctuation(df.summary[i])
        df.summary[i] = strip_multiple_whitespaces(df.summary[i])
        df.summary[i] = df.summary[i].lower()
        
    for i in range(df.shape[0]):
        df.headline[i] = strip_numeric(df.headline[i])
        df.headline[i] = strip_punctuation(df.headline[i])
        df.headline[i] = strip_multiple_whitespaces(df.headline[i])
        df.headline[i] = df.headline[i].lower()
        
    sid = sia()
    df['sentiment_vd_headline'] = df['headline'].apply(lambda headline: sid.polarity_scores(headline)['compound'])
    df['sentiment_vd_summary'] = df['summary'].apply(lambda summary: sid.polarity_scores(summary)['compound'])
    st.write("Sentiment analysis of headlines:", df.sentiment_vd_headline.mean())
    st.write("Sentiment analysis of summary:", df.sentiment_vd_summary.mean())
    
except:
    st.write("No sentiment analysis possible")

#show additional info if checkbox in sidebar is checked
if show_news == True:
    try:
        st.write(news.headline)
    except:
        st.write("No news available. Look for a more interesting stock.")

if fin == True:
    st.write(stock.financials)



# building the data for prediction
def get_news(company, date_from='2021-06-01', date_to=None):
    '''
    returns dataframe with average sentiment of news headline and sentiment of news summary for every date in a given timeframe
    company: symbol, example ZM
    date_from: string format yyyy-mm-dd
    date_to: string format yyyy-mm-dd
    '''
    sid = sia()
    if date_to is None:
        date_to = datetime.today().strftime("%Y-%m-%d")
    URL = 'https://finnhub.io/api/v1/company-news?symbol={}&from={}&to={}&token=c2si65iad3ic1qis06lg'.format(company, date_from, date_to)
    r = requests.get(URL)
    news_df = pd.DataFrame(r.json())
    news_df['datetime'] = [datetime.utcfromtimestamp(i).strftime('%Y-%m-%d') for i in news_df.datetime]
    news_df.drop(['id','image', 'related','source', 'url'], axis=1, inplace=True)
    news_df['headline_sentiment'] = [sid.polarity_scores(c)['compound'] for c in news_df['headline']]
    news_df['summary_sentiment'] = [sid.polarity_scores(c)['compound'] for c in news_df['summary']]
    
    news_dates = news_df.groupby(['datetime']).mean().sort_index().reset_index()

    news_dates['Date'] = pd.to_datetime(news_dates['datetime'], format='%Y-%m-%d')
    return news_dates


def get_recommendation_trends(company):
    URL = 'https://finnhub.io/api/v1/stock/recommendation?symbol={}&token=c2si65iad3ic1qis06lg'.format(company)
    r = requests.get(URL)
    recommendation_df = pd.DataFrame(r.json())
    dfs = []
    for i, month in recommendation_df.period.iteritems():
        period = month
        df_month = pd.DataFrame({
        'period': pd.date_range(
            start = pd.Timestamp(period),                        
            end = pd.Timestamp(period) + pd.offsets.MonthEnd(0),  # <-- 2018-08-31 with MonthEnd
            freq = 'D'
            )
        })
        df_month['month'] = month
        dfs.append(df_month)
    df = pd.concat(dfs)
    df.period = [datetime.strftime(x, '%Y-%m-%d') for x in df.period]
    df = df.merge(recommendation_df,how='left', left_on='month', right_on='period')
    df.drop(['month','period_y'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['period_x'], format='%Y-%m-%d')
    return df.drop(['period_x'],1)

add_all_ta_features(
    hist, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

news_df = get_news(slct)
trend_df = get_recommendation_trends(slct)
data = hist.merge(news_df, how='inner', on='Date').merge(trend_df, how='inner', on='Date').drop(['symbol','datetime_y','datetime_x'],1)
st.write('Data to train the model on')
st.write(data)

def predict(data, move_days=7):
    # shifted values column, like this the model can learn what the price is going to be x days later
    data["shift_close"]=data[["Close"]].shift(-move_days)
    #x without the NaNs 
    X = data.drop(['shift_close','Date'],1)[:-move_days]
    y = data[['shift_close']][:-move_days]
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

    # training a model on the dataset where we have all data
    param_grid = {'min_samples_leaf': [2,3,5,10,20], 'n_estimators': [3,5,10,50,100], 'max_depth':[20,30,40]}
    rf = RandomForestRegressor(random_state = 42)
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)
    grid_search.fit(X_train, y_train)
    params = grid_search.best_params_

    rf=RandomForestRegressor(min_samples_leaf=params['min_samples_leaf'], n_estimators=params['n_estimators'], max_depth = params['max_depth'],random_state=42)
    rf.fit(X_train, y_train)
    y_pred_existing=rf.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred_existing)
    
    # Now the X variable is going to be the features for the days where we have no "in x Days price"
    last_days=data.tail(move_days)
    X_pred=last_days.drop(['shift_close','Date'],1)
    
    #values for the next 7 days 
    pred=rf.predict(X_pred)
    #predicted_dates = data['Date']
    last_days['shift_close'] = pred
    
    return last_days, MSE
predicted, MSE =predict(data)

st.write('Predicted values')
st.write(predicted[['Date','shift_close']])


st.write('MSE:' + str(MSE))
