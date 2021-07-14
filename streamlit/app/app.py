import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score

#cache the stock tickers so they don't have to load everytime
@st.cache
def get_ticker(market = "US"):
    tickers = []
    t = finnhub_client.stock_symbols(market)
    for ticker in t:
        tickers.append(ticker["displaySymbol"])
    
    return tickers

#set up finnhub client
finnhub_client = finnhub.Client(api_key="c2si65iad3ic1qis06lg")


#streamlit interactibles and headings
st.header('Using ML to predict the Stock Market')
st.subheader('A Big Data Project by Valeriia, Louisa and Alexander')
show_news = st.sidebar.checkbox('Display company news?')
timeframe = st.sidebar.selectbox('Select period to display historical data:',['max','1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd'])
predictframe = st.sidebar.slider('Select days to predict:',min_value=1, max_value = 7, value=7)
go = st.sidebar.button("Predict!")


#With this part, the user can choose the desired stock market he wants to view, however, US market works best because it has the most data from finnhub so we're limiting it to US
#markets = ["AS","AT","AX","BA","BC","BD","BE","BK","BO","BR","CN","CO","CR","DB","DE","DU","F","HE","HK","HM","IC","IR","IS","JK","JO","KL","KQ","KS","L","LN","LS","MC","ME","MI","MU","MX","NE","NL","NS","NZ","OL","PA","PM","PR","QA","RG","SA","SG","SI","SN","SR","SS","ST","SW","SZ","T","TA","TL","TO","TW","US","V","VI","VN","VS","WA","HA","SX","TG","SC"]
#market = st.sidebar.selectbox('Select market:', markets, index = markets.index("US"))

#fetch all tickers and get index of TSLA as starting point for US market
ticks = get_ticker()
ticks = list(np.unique(ticks))

#try except is in case there would be a different market selected
try:
    index = ticks.index("TSLA")
except:
    index = 0

#Select the desired stock from the market and get stock data from yfinance
slct = st.selectbox('Select your stock:',ticks, index = index)

#Maybe possible to cache this, however, as stocks are constantly changing, it's kept as-is so that a refresh always shows the newest data 
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
fig = px.line(hist, x="datetime", y="Close", title=f'Closing Price of {slct}', labels = {"datetime":"Date", "Close":"Closing Price [$]"},template = 'seaborn')
st.plotly_chart(fig)

with st.spinner(text='Fetching news ...'):
    #fetch and add company news
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

with st.spinner(text='Fetching additional data ...'):
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

    # View accuracy score
    score = rf.score(X_test, y_pred_existing)

    # Now the X variable is going to be the features for the days where we have no "in x Days price"
    last_days=data.tail(move_days)
    X_pred=last_days.drop(['shift_close','Date'],1)
    
    #values for the next 7 days 
    pred=rf.predict(X_pred)
    #predicted_dates = data['Date']
    last_days['shift_close'] = pred
    
    return last_days, MSE, score

if go == True:
    with st.spinner(text='Calculating prediction ...'):
        predicted, MSE, score =predict(data, move_days=predictframe)

        #setting the index to date
        predicted.index = predicted.Date

        #shifting it back forward for x days
        predicted.index = predicted.index.shift(predictframe, freq = "D")
        st.write('Predicted values')
        st.write(predicted[['Date','shift_close']])


        st.write('MSE:' + str(MSE))
        st.write('Score:' + str(score))

        #joining with the historicaL data to create a subframe with just the new dates then appending that to the rest of the data but only take last 20 entries, sorted by date
        graph = hist.append(hist.join(predicted[["Date", "shift_close"]], how= 'right')).sort_index().tail(20)
        graph.rename(columns = {"shift_close":"Prediction", "Close":"Historical Data"}, inplace = True)

        #plotting
        fig2 = px.line(graph, x=graph.index, y=["Historical Data","Prediction"], title=f'Predicted closing price for {slct}', labels = {"index":"Date"}, template = 'seaborn', range_x = [min(graph.index),max(graph.index.shift(1, freq = "D"))])
        fig2.update_traces(mode='markers+lines')
        fig2.update_layout(xaxis_title='Date', yaxis_title='Closing Price [$]')
        st.plotly_chart(fig2)
