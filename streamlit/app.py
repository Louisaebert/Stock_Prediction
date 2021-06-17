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